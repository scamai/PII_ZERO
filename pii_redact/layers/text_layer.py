"""Text layer: extract text with coordinates, then run NER to produce RedactionBoxes.

Dispatch logic
--------------
PDF_STRUCTURED  ->  PyMuPDF native text extraction (no OCR)
SCAN_FORM / ID_DOC / MEDICAL / UNKNOWN  ->  PaddleOCR word-level bounding boxes

All extracted text is passed through:
  1. Presidio AnalyzerEngine (built-in + regex recognizers)
  2. InsuranceNER fine-tuned spaCy model (if available)

The resulting RedactionBoxes carry source="presidio" or source="insurance_ner".

SAFETY: solid-fill only — blur is never called anywhere in this module.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Set offline mode for HuggingFace BEFORE any transformers import.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

from pii_redact.config import settings
from pii_redact.models import DocumentType, RedactionBox

if TYPE_CHECKING:
    import numpy as np
    from presidio_analyzer import AnalyzerEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A "word record" normalised across extraction backends:
#   {"text": str, "x": int, "y": int, "w": int, "h": int, "page": int}
_WordRecord = dict[str, Any]


# ---------------------------------------------------------------------------
# Lazy singletons for heavy models
# ---------------------------------------------------------------------------

_ocr_engine: Any | None = None
_analyzer: "AnalyzerEngine | None" = None
_insurance_ner: Any | None = None


def _get_ocr() -> Any:
    """Return PaddleOCR instance, loading on first call."""
    global _ocr_engine
    if _ocr_engine is not None:
        return _ocr_engine

    from paddleocr import PaddleOCR  # noqa: PLC0415

    paddleocr_dir = str(
        Path(settings.models.paddleocr_dir).resolve()
    )

    try:
        _ocr_engine = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            det_model_dir=paddleocr_dir,
            rec_model_dir=paddleocr_dir,
            cls_model_dir=paddleocr_dir,
            show_log=False,
        )
        logger.info("PaddleOCR loaded from %s", paddleocr_dir)
    except Exception as exc:
        logger.warning(
            "PaddleOCR failed to load from %s (%s). "
            "Trying default PaddleOCR (may attempt download if online).",
            paddleocr_dir,
            exc,
        )
        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

    return _ocr_engine


def _get_analyzer() -> "AnalyzerEngine":
    global _analyzer
    if _analyzer is None:
        from pii_redact.ner.presidio_setup import build_analyzer_engine  # noqa: PLC0415

        _analyzer = build_analyzer_engine()
    return _analyzer


def _get_insurance_ner() -> Any:
    global _insurance_ner
    if _insurance_ner is None:
        from pii_redact.ner.insurance_ner import InsuranceNER  # noqa: PLC0415

        _insurance_ner = InsuranceNER()
    return _insurance_ner


# ---------------------------------------------------------------------------
# Text extraction helpers
# ---------------------------------------------------------------------------

def _extract_words_pymupdf(file_path: str | Path) -> list[_WordRecord]:
    """Extract word-level bounding boxes from a native-text PDF via PyMuPDF."""
    import fitz  # PyMuPDF  # noqa: PLC0415

    records: list[_WordRecord] = []
    doc = fitz.open(str(file_path))
    try:
        for page_idx, page in enumerate(doc):
            # get_text("words") returns tuples:
            # (x0, y0, x1, y1, word, block_no, line_no, word_no)
            for word_tuple in page.get_text("words"):
                x0, y0, x1, y1, word, *_ = word_tuple
                if not word.strip():
                    continue
                records.append(
                    {
                        "text": word,
                        "x": int(x0),
                        "y": int(y0),
                        "w": int(x1 - x0),
                        "h": int(y1 - y0),
                        "page": page_idx,
                    }
                )
    finally:
        doc.close()
    return records


def _extract_words_paddleocr(
    file_path: str | Path,
    page_image: "np.ndarray | None" = None,
) -> list[_WordRecord]:
    """Run PaddleOCR on an image (or image path) and return word records.

    If *page_image* is supplied (numpy array, HWC BGR or RGB) it is used
    directly.  Otherwise *file_path* is read as an image file.
    """
    ocr = _get_ocr()

    if page_image is not None:
        raw_result = ocr.ocr(page_image, cls=True)
    else:
        raw_result = ocr.ocr(str(file_path), cls=True)

    records: list[_WordRecord] = []
    if not raw_result:
        return records

    # PaddleOCR returns a list-of-pages; each page is a list of lines.
    # Each line: [polygon_points, (text, confidence)]
    # polygon_points: [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]
    for page_idx, page_lines in enumerate(raw_result):
        if not page_lines:
            continue
        for line in page_lines:
            if line is None:
                continue
            polygon, (text, _conf) = line
            if not text or not text.strip():
                continue
            xs = [pt[0] for pt in polygon]
            ys = [pt[1] for pt in polygon]
            x = int(min(xs))
            y = int(min(ys))
            w = int(max(xs) - x)
            h = int(max(ys) - y)
            records.append(
                {
                    "text": text,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "page": page_idx,
                }
            )
    return records


# ---------------------------------------------------------------------------
# NER helpers
# ---------------------------------------------------------------------------

def _run_presidio(
    words: list[_WordRecord],
    min_confidence: float,
) -> list[RedactionBox]:
    """Run Presidio over the full reconstructed text, then map spans back to boxes."""
    if not words:
        return []

    analyzer = _get_analyzer()

    # Group words by page and rebuild a line of text per page so that
    # multi-word entity spans can be detected.
    pages: dict[int, list[_WordRecord]] = {}
    for w in words:
        pages.setdefault(w["page"], []).append(w)

    boxes: list[RedactionBox] = []

    for page_idx, page_words in pages.items():
        # Build a flat text string with space separators, tracking char offsets.
        text_parts: list[str] = []
        offsets: list[tuple[int, int, _WordRecord]] = []  # (start, end, word_record)
        cursor = 0
        for wr in page_words:
            word_text = wr["text"]
            start = cursor
            end = cursor + len(word_text)
            offsets.append((start, end, wr))
            text_parts.append(word_text)
            cursor = end + 1  # +1 for the space separator

        full_text = " ".join(text_parts)

        try:
            results = analyzer.analyze(text=full_text, language="en")
        except Exception as exc:
            logger.warning("Presidio analyze failed on page %d: %s", page_idx, exc)
            continue

        for result in results:
            if result.score < min_confidence:
                continue

            # Find all word records whose character span overlaps the entity span.
            entity_start = result.start
            entity_end = result.end
            hit_words = [
                wr
                for (ws, we, wr) in offsets
                if ws < entity_end and we > entity_start
            ]
            if not hit_words:
                continue

            # Merge bounding boxes of all overlapping words.
            x = min(wr["x"] for wr in hit_words)
            y = min(wr["y"] for wr in hit_words)
            right = max(wr["x"] + wr["w"] for wr in hit_words)
            bottom = max(wr["y"] + wr["h"] for wr in hit_words)
            entity_text = full_text[entity_start:entity_end]

            boxes.append(
                RedactionBox(
                    x=x,
                    y=y,
                    w=right - x,
                    h=bottom - y,
                    entity_type=result.entity_type,
                    confidence=result.score,
                    source="presidio",
                    page=page_idx,
                    text_found=entity_text,
                )
            )

    return boxes


def _run_insurance_ner(
    words: list[_WordRecord],
    min_confidence: float,
) -> list[RedactionBox]:
    """Run the fine-tuned InsuranceNER model and map entity spans to boxes."""
    if not words:
        return []

    ner = _get_insurance_ner()

    pages: dict[int, list[_WordRecord]] = {}
    for w in words:
        pages.setdefault(w["page"], []).append(w)

    boxes: list[RedactionBox] = []

    for page_idx, page_words in pages.items():
        text_parts: list[str] = []
        offsets: list[tuple[int, int, _WordRecord]] = []
        cursor = 0
        for wr in page_words:
            word_text = wr["text"]
            start = cursor
            end = cursor + len(word_text)
            offsets.append((start, end, wr))
            text_parts.append(word_text)
            cursor = end + 1

        full_text = " ".join(text_parts)

        try:
            entities = ner.analyze(full_text)
        except Exception as exc:
            logger.warning(
                "InsuranceNER failed on page %d: %s", page_idx, exc
            )
            continue

        for ent in entities:
            entity_start = ent["start"]
            entity_end = ent["end"]
            hit_words = [
                wr
                for (ws, we, wr) in offsets
                if ws < entity_end and we > entity_start
            ]
            if not hit_words:
                continue

            x = min(wr["x"] for wr in hit_words)
            y = min(wr["y"] for wr in hit_words)
            right = max(wr["x"] + wr["w"] for wr in hit_words)
            bottom = max(wr["y"] + wr["h"] for wr in hit_words)

            # InsuranceNER doesn't provide a confidence score; use a fixed value
            # just above the configured threshold so it always passes.
            confidence = max(min_confidence + 0.05, 0.80)

            boxes.append(
                RedactionBox(
                    x=x,
                    y=y,
                    w=right - x,
                    h=bottom - y,
                    entity_type=ent["label"],
                    confidence=confidence,
                    source="insurance_ner",
                    page=page_idx,
                    text_found=ent["text"],
                )
            )

    return boxes


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------

class TextLayer:
    """Extract text with bounding boxes and detect PII using Presidio + InsuranceNER.

    Usage::

        layer = TextLayer()
        boxes = layer.process("path/to/doc.pdf", DocumentType.PDF_STRUCTURED)
    """

    def process(
        self,
        file_path: str | Path,
        document_type: DocumentType,
        page_image: "np.ndarray | None" = None,
    ) -> list[RedactionBox]:
        """Extract text and return RedactionBoxes for all detected PII.

        Parameters
        ----------
        file_path:
            Path to the source document (PDF or image file).
        document_type:
            Controls which extraction backend is used.
        page_image:
            Optional pre-rendered numpy image array (HWC, uint8).  When
            supplied for non-structured types, OCR runs on this array instead
            of re-reading *file_path*.  Useful when the caller has already
            rendered the page at a specific DPI.

        Returns
        -------
        list[RedactionBox]
            One box per detected PII span.  Source is "presidio" or
            "insurance_ner".  Boxes may overlap; deduplication is the
            responsibility of the pipeline coordinator.
        """
        file_path = Path(file_path)
        min_confidence: float = settings.redaction.min_confidence

        # --- Step 1: extract words with coordinates -------------------------
        if document_type == DocumentType.PDF_STRUCTURED:
            words = _extract_words_pymupdf(file_path)
            logger.debug(
                "PyMuPDF extracted %d words from %s", len(words), file_path.name
            )
        else:
            # SCAN_FORM, ID_DOC, MEDICAL, PHOTO, UNKNOWN → OCR
            words = _extract_words_paddleocr(file_path, page_image=page_image)
            logger.debug(
                "PaddleOCR extracted %d words from %s", len(words), file_path.name
            )

        if not words:
            logger.info("No text found in %s (doc_type=%s)", file_path.name, document_type)
            return []

        # --- Step 2: NER chain ----------------------------------------------
        presidio_boxes = _run_presidio(words, min_confidence)
        insurance_boxes = _run_insurance_ner(words, min_confidence)

        all_boxes = presidio_boxes + insurance_boxes
        logger.info(
            "%s: %d presidio + %d insurance_ner boxes (total %d)",
            file_path.name,
            len(presidio_boxes),
            len(insurance_boxes),
            len(all_boxes),
        )
        return all_boxes
