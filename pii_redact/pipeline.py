"""PIIRedactionPipeline — orchestrates classify → route → redact → save → audit.

Sub-pipeline routing:
  PDF_STRUCTURED → text NER + native PDF redaction
  SCAN_FORM      → render at 300 DPI + visual (handwriting) + text (OCR)
  PHOTO          → visual only (faces, plates, QR) + EXIF strip
  ID_DOC         → OCR text + visual (face) + EXIF strip
  MEDICAL        → text (OCR or native) + scispaCy + insurance NER
  UNKNOWN        → all layers (conservative)

All heavy imports are lazy to avoid slow startup.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from pii_redact.classifier import classify_document
from pii_redact.models import DocumentResult, DocumentType, PageResult, RedactionBox

if TYPE_CHECKING:
    from pii_redact.config import Settings

logger = logging.getLogger(__name__)

_SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_pdf_pages(pdf_path: Path, dpi: int = 300) -> list:
    """Render each PDF page to a BGR numpy array at *dpi* DPI.

    Returns a list of (page_index, np.ndarray) tuples.
    """
    import fitz  # PyMuPDF
    import numpy as np

    pages = []
    doc = fitz.open(str(pdf_path))
    try:
        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        for i, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            import cv2

            bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            pages.append((i, bgr))
    finally:
        doc.close()
    return pages


def _extract_pdf_text_boxes(pdf_path: Path) -> list[tuple[int, list[RedactionBox]]]:
    """Extract text from the native PDF text layer and run NER.

    Returns a list of (page_index, [RedactionBox, ...]) tuples.
    """
    try:
        import fitz

        from pii_redact.ner.presidio_setup import build_analyzer_engine as get_analyzer  # lazy

        analyzer = get_analyzer()
        results_by_page: list[tuple[int, list[RedactionBox]]] = []
        doc = fitz.open(str(pdf_path))
        try:
            for page_idx, page in enumerate(doc):
                blocks = page.get_text("dict")["blocks"]
                page_boxes: list[RedactionBox] = []
                for block in blocks:
                    if block.get("type") != 0:  # 0 = text block
                        continue
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if not text:
                                continue
                            bbox = span.get("bbox", [0, 0, 0, 0])  # (x0, y0, x1, y1)
                            x0, y0, x1, y1 = (int(v) for v in bbox)

                            try:
                                hits = analyzer.analyze(text=text, language="en")
                            except Exception:
                                hits = []

                            for hit in hits:
                                page_boxes.append(
                                    RedactionBox(
                                        x=x0,
                                        y=y0,
                                        w=x1 - x0,
                                        h=y1 - y0,
                                        entity_type=hit.entity_type,
                                        confidence=hit.score,
                                        source="presidio",
                                        page=page_idx,
                                        text_found=text[hit.start:hit.end],
                                    )
                                )
                results_by_page.append((page_idx, page_boxes))
        finally:
            doc.close()
        return results_by_page
    except Exception as exc:
        logger.warning("PDF text extraction failed: %s", exc)
        return []


def _run_ocr_on_image(image, page_idx: int = 0) -> list[RedactionBox]:
    """Run PaddleOCR + Presidio NER on a numpy image array."""
    boxes: list[RedactionBox] = []
    try:
        from paddleocr import PaddleOCR  # type: ignore[import]

        from pii_redact.ner.presidio_setup import build_analyzer_engine as get_analyzer

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        analyzer = get_analyzer()

        result = ocr.ocr(image, cls=True)
        if not result or not result[0]:
            return []

        for line in result[0]:
            if not line:
                continue
            coords, (text, confidence) = line
            # coords: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            xs = [pt[0] for pt in coords]
            ys = [pt[1] for pt in coords]
            x, y = int(min(xs)), int(min(ys))
            w = int(max(xs) - min(xs))
            h = int(max(ys) - min(ys))

            if not text.strip():
                continue

            try:
                hits = analyzer.analyze(text=text, language="en")
            except Exception:
                hits = []

            for hit in hits:
                boxes.append(
                    RedactionBox(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        entity_type=hit.entity_type,
                        confidence=hit.score,
                        source="presidio",
                        page=page_idx,
                        text_found=text[hit.start:hit.end],
                    )
                )
    except Exception as exc:
        logger.warning("OCR failed on page %d: %s", page_idx, exc)
    return boxes


def _run_insurance_ner_on_image(image, page_idx: int = 0) -> list[RedactionBox]:
    """Run domain-specific insurance NER over OCR text."""
    boxes: list[RedactionBox] = []
    try:
        from pii_redact.ner.insurance_ner import InsuranceNER  # lazy

        ins_ner = InsuranceNER()

        from paddleocr import PaddleOCR  # type: ignore[import]

        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        result = ocr.ocr(image, cls=True)
        if not result or not result[0]:
            return []

        for line in result[0]:
            if not line:
                continue
            coords, (text, _confidence) = line
            xs = [pt[0] for pt in coords]
            ys = [pt[1] for pt in coords]
            x, y = int(min(xs)), int(min(ys))
            w = int(max(xs) - min(xs))
            h = int(max(ys) - min(ys))

            try:
                hits = ins_ner.analyze(text)
            except Exception:
                hits = []

            for hit in hits:
                boxes.append(
                    RedactionBox(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        entity_type=hit.get("entity_type", "INSURANCE"),
                        confidence=hit.get("score", 0.75),
                        source="insurance_ner",
                        page=page_idx,
                        text_found=text,
                    )
                )
    except Exception as exc:
        logger.debug("Insurance NER on image failed: %s", exc)
    return boxes


def _apply_pdf_redactions(
    input_path: Path,
    output_path: Path,
    boxes_by_page: dict[int, list[RedactionBox]],
    fill_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Burn solid-fill redaction boxes into the PDF and save to *output_path*."""
    import fitz

    doc = fitz.open(str(input_path))
    try:
        for page_idx, page in enumerate(doc):
            page_boxes = boxes_by_page.get(page_idx, [])
            for box in page_boxes:
                rect = fitz.Rect(box.x, box.y, box.x + box.w, box.y + box.h)
                page.add_redact_annot(rect, fill=fill_color)
            if page_boxes:
                page.apply_redactions()
        doc.save(str(output_path), garbage=4, deflate=True)
    finally:
        doc.close()


def _apply_image_redactions(
    input_path: Path,
    output_path: Path,
    boxes: list[RedactionBox],
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Solid-fill redaction boxes on a raster image and save to *output_path*."""
    import cv2

    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")
    for box in boxes:
        x1, y1 = box.x, box.y
        x2, y2 = box.x + box.w, box.y + box.h
        cv2.rectangle(image, (x1, y1), (x2, y2), fill_color, thickness=-1)
    cv2.imwrite(str(output_path), image)


def _strip_exif(image_path: Path) -> list[str]:
    """Remove all EXIF metadata from an image file in-place. Returns stripped field names."""
    stripped: list[str] = []
    try:
        import piexif  # type: ignore[import]

        exif_dict = piexif.load(str(image_path))
        gps = exif_dict.get("GPS", {})
        if gps:
            stripped.append("GPS")
        # Wipe all EXIF data
        piexif.remove(str(image_path))
        stripped.extend(["Exif", "0th", "1st"])
    except Exception as exc:
        logger.debug("EXIF stripping failed (may not be JPEG/TIFF): %s", exc)
    return stripped


def _write_audit_log(result: DocumentResult, log_dir: Path, settings) -> None:
    """Write a JSON audit log entry for *result*."""
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{result.doc_id}.json"

        include_text: bool = settings.audit.get("include_text", False)
        entry = {
            "doc_id": result.doc_id,
            "filename": result.filename,
            "doc_type": result.doc_type,
            "pages": result.pages,
            "total_processing_ms": round(result.total_processing_ms, 1),
            "entity_counts": result.entity_counts,
            "source_counts": result.source_counts,
            "detections": [
                box.as_dict(include_text=include_text) for box in result.all_boxes
            ],
        }
        log_file.write_text(json.dumps(entry, indent=2, default=str))
        logger.debug("Audit log written: %s", log_file)
    except Exception as exc:
        logger.warning("Failed to write audit log: %s", exc)


# ---------------------------------------------------------------------------
# PIIRedactionPipeline
# ---------------------------------------------------------------------------


class PIIRedactionPipeline:
    """Orchestrates the full PII detection and redaction workflow."""

    def __init__(self, settings, vault=None, use_vlm: bool = False):
        """
        Parameters
        ----------
        settings:
            A pii_redact.config.Settings instance.
        vault:
            Optional PIIVault for storing/restoring original PII values.
        use_vlm:
            Enable Qwen3-VL visual extractor. Lazy-loads the model on first
            use; GPU-first with CPU fallback. Default off (requires download).
        """
        self._settings = settings
        self._vault = vault
        self._use_vlm = use_vlm
        self._vlm = None  # loaded lazily

    def _get_vlm(self):
        """Return the VLMExtractor singleton, loading it on first call."""
        if self._vlm is None:
            from pii_redact.ner.vlm_extractor import VLMExtractor

            model_id = self._settings.models.get("vlm_model_id")
            prefer_gpu = self._settings.models.get("vlm_prefer_gpu", True)
            self._vlm = VLMExtractor(model_id=model_id, prefer_gpu=prefer_gpu)
            logger.info("VLMExtractor created: %s (gpu=%s)", model_id, prefer_gpu)
        return self._vlm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        input_path: Path,
        output_path: Path,
        dry_run: bool = False,
    ) -> DocumentResult:
        """Classify → route → redact → save → audit.

        Parameters
        ----------
        input_path:
            Source document to redact.
        output_path:
            Destination for the redacted output (ignored when dry_run=True).
        dry_run:
            When True, detect PII but do not write any files.

        Returns
        -------
        DocumentResult
            Full result including all detected RedactionBoxes.
        """
        t0 = time.monotonic()
        input_path = Path(input_path)
        output_path = Path(output_path)

        doc_id = str(uuid.uuid4())
        doc_type = classify_document(input_path)

        logger.info(
            "Processing %s  type=%s  dry_run=%s",
            input_path.name,
            doc_type,
            dry_run,
        )

        # Route to the appropriate sub-pipeline
        try:
            if doc_type == DocumentType.PDF_STRUCTURED:
                result = self._process_pdf_structured(input_path, output_path, doc_id, doc_type, dry_run)
            elif doc_type == DocumentType.SCAN_FORM:
                result = self._process_scan_form(input_path, output_path, doc_id, doc_type, dry_run)
            elif doc_type == DocumentType.PHOTO:
                result = self._process_photo(input_path, output_path, doc_id, doc_type, dry_run)
            elif doc_type == DocumentType.ID_DOC:
                result = self._process_id_doc(input_path, output_path, doc_id, doc_type, dry_run)
            elif doc_type == DocumentType.MEDICAL:
                result = self._process_medical(input_path, output_path, doc_id, doc_type, dry_run)
            else:  # UNKNOWN — conservative: run everything
                result = self._process_unknown(input_path, output_path, doc_id, doc_type, dry_run)
        except Exception as exc:
            logger.error("Pipeline error for %s: %s", input_path, exc, exc_info=True)
            result = DocumentResult(
                doc_id=doc_id,
                filename=input_path.name,
                doc_type=doc_type,
                pages=0,
            )

        result.total_processing_ms = (time.monotonic() - t0) * 1000

        # Audit log
        log_dir = Path(self._settings.audit.get("log_dir", "./audit_logs"))
        _write_audit_log(result, log_dir, self._settings)

        logger.info(
            "Done %s: %d box(es) in %.0f ms",
            input_path.name,
            len(result.all_boxes),
            result.total_processing_ms,
        )
        return result

    def inspect(self, input_path: Path) -> DocumentResult:
        """Dry run — detect PII without modifying any files."""
        return self.process(input_path, output_path=input_path, dry_run=True)

    def process_batch(self, input_dir: Path, output_dir: Path) -> list[DocumentResult]:
        """Process all supported files in *input_dir*, writing to *output_dir*."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        candidates = [
            p for p in sorted(input_dir.iterdir())
            if p.is_file() and p.suffix.lower() in _SUPPORTED_EXTENSIONS
        ]
        logger.info("Batch: found %d supported file(s) in %s", len(candidates), input_dir)

        results: list[DocumentResult] = []
        for file_path in candidates:
            out_path = output_dir / file_path.name
            try:
                r = self.process(file_path, out_path)
            except Exception as exc:
                logger.error("Batch item failed %s: %s", file_path, exc)
                r = DocumentResult(
                    doc_id=str(uuid.uuid4()),
                    filename=file_path.name,
                    doc_type=DocumentType.UNKNOWN,
                    pages=0,
                )
            results.append(r)
        return results

    # ------------------------------------------------------------------
    # Sub-pipelines
    # ------------------------------------------------------------------

    def _process_pdf_structured(
        self,
        input_path: Path,
        output_path: Path,
        doc_id: str,
        doc_type: DocumentType,
        dry_run: bool,
    ) -> DocumentResult:
        """PDF with native text layer: NER on text spans + optional visual layer."""
        from pii_redact.config import settings as cfg

        page_results: list[PageResult] = []
        boxes_by_page: dict[int, list[RedactionBox]] = {}

        # Text layer NER
        text_results = _extract_pdf_text_boxes(input_path)
        for page_idx, page_boxes in text_results:
            boxes_by_page.setdefault(page_idx, []).extend(page_boxes)

        # Optional visual layer
        skip_visual = cfg.pipeline.get("skip_visual_for_structured_pdfs", False)
        if not skip_visual:
            dpi = cfg.pipeline.get("ocr_dpi", 300)
            rendered = _render_pdf_pages(input_path, dpi=dpi)
            for page_idx, img in rendered:
                visual_boxes = self._run_visual_layer(img, doc_type, page_idx)
                boxes_by_page.setdefault(page_idx, []).extend(visual_boxes)

        # Build page results
        all_pages = sorted(set(
            list(boxes_by_page.keys()) + [i for i, _ in (text_results or [])]
        ))
        for page_idx in all_pages:
            page_results.append(
                PageResult(
                    page=page_idx,
                    doc_type=doc_type,
                    redaction_boxes=boxes_by_page.get(page_idx, []),
                )
            )

        if not dry_run and any(pr.redaction_boxes for pr in page_results):
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _apply_pdf_redactions(input_path, output_path, boxes_by_page)

        return DocumentResult(
            doc_id=doc_id,
            filename=input_path.name,
            doc_type=doc_type,
            pages=len(page_results),
            page_results=page_results,
        )

    def _process_scan_form(
        self,
        input_path: Path,
        output_path: Path,
        doc_id: str,
        doc_type: DocumentType,
        dry_run: bool,
    ) -> DocumentResult:
        """Scanned form: render pages to images → OCR text NER + visual (handwriting)."""
        from pii_redact.config import settings as cfg

        dpi = cfg.pipeline.get("ocr_dpi", 300)
        page_results: list[PageResult] = []
        boxes_by_page: dict[int, list[RedactionBox]] = {}
        rendered_pages: list[tuple[int, object]] = []

        suffix = input_path.suffix.lower()
        if suffix == ".pdf":
            rendered_pages = _render_pdf_pages(input_path, dpi=dpi)
        else:
            import cv2

            img = cv2.imread(str(input_path))
            if img is not None:
                rendered_pages = [(0, img)]

        for page_idx, img in rendered_pages:
            t_page = time.monotonic()
            boxes: list[RedactionBox] = []
            boxes.extend(_run_ocr_on_image(img, page_idx))
            boxes.extend(self._run_visual_layer(img, doc_type, page_idx))
            page_ms = (time.monotonic() - t_page) * 1000

            boxes_by_page[page_idx] = boxes
            page_results.append(
                PageResult(
                    page=page_idx,
                    doc_type=doc_type,
                    redaction_boxes=boxes,
                    processing_ms=page_ms,
                )
            )

        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if suffix == ".pdf":
                _apply_pdf_redactions(input_path, output_path, boxes_by_page)
            else:
                _apply_image_redactions(input_path, output_path, boxes_by_page.get(0, []))

        return DocumentResult(
            doc_id=doc_id,
            filename=input_path.name,
            doc_type=doc_type,
            pages=len(page_results),
            page_results=page_results,
        )

    def _process_photo(
        self,
        input_path: Path,
        output_path: Path,
        doc_id: str,
        doc_type: DocumentType,
        dry_run: bool,
    ) -> DocumentResult:
        """Accident / scene photo: visual layer only + EXIF strip."""
        import cv2

        img = cv2.imread(str(input_path))
        boxes: list[RedactionBox] = []
        if img is not None:
            boxes = self._run_visual_layer(img, doc_type, page_idx=0)

        stripped_fields: list[str] = []
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _apply_image_redactions(input_path, output_path, boxes)
            stripped_fields = _strip_exif(output_path)

        from pii_redact.models import MetadataStripResult

        page_result = PageResult(
            page=0,
            doc_type=doc_type,
            redaction_boxes=boxes,
            metadata=MetadataStripResult(stripped_fields=stripped_fields),
        )
        return DocumentResult(
            doc_id=doc_id,
            filename=input_path.name,
            doc_type=doc_type,
            pages=1,
            page_results=[page_result],
        )

    def _process_id_doc(
        self,
        input_path: Path,
        output_path: Path,
        doc_id: str,
        doc_type: DocumentType,
        dry_run: bool,
    ) -> DocumentResult:
        """Driver's license / passport / insurance card: OCR + visual (face) + EXIF strip."""
        import cv2

        img = cv2.imread(str(input_path))
        boxes: list[RedactionBox] = []
        if img is not None:
            boxes.extend(_run_ocr_on_image(img, page_idx=0))
            boxes.extend(self._run_visual_layer(img, doc_type, page_idx=0))

        stripped_fields: list[str] = []
        if not dry_run:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            _apply_image_redactions(input_path, output_path, boxes)
            stripped_fields = _strip_exif(output_path)

        from pii_redact.models import MetadataStripResult

        page_result = PageResult(
            page=0,
            doc_type=doc_type,
            redaction_boxes=boxes,
            metadata=MetadataStripResult(stripped_fields=stripped_fields),
        )
        return DocumentResult(
            doc_id=doc_id,
            filename=input_path.name,
            doc_type=doc_type,
            pages=1,
            page_results=[page_result],
        )

    def _process_medical(
        self,
        input_path: Path,
        output_path: Path,
        doc_id: str,
        doc_type: DocumentType,
        dry_run: bool,
    ) -> DocumentResult:
        """Medical documents: text NER (native or OCR) + scispaCy + insurance NER."""
        from pii_redact.config import settings as cfg

        dpi = cfg.pipeline.get("ocr_dpi", 300)
        suffix = input_path.suffix.lower()
        page_results: list[PageResult] = []
        boxes_by_page: dict[int, list[RedactionBox]] = {}

        if suffix == ".pdf":
            # Try native text layer first
            text_results = _extract_pdf_text_boxes(input_path)
            for page_idx, page_boxes in text_results:
                boxes_by_page.setdefault(page_idx, []).extend(page_boxes)

            # Also render for insurance NER
            rendered = _render_pdf_pages(input_path, dpi=dpi)
            for page_idx, img in rendered:
                boxes_by_page.setdefault(page_idx, []).extend(
                    _run_insurance_ner_on_image(img, page_idx)
                )
                boxes_by_page[page_idx].extend(self._run_scispacy(img, page_idx))

            for page_idx in sorted(boxes_by_page):
                page_results.append(
                    PageResult(
                        page=page_idx,
                        doc_type=doc_type,
                        redaction_boxes=boxes_by_page[page_idx],
                    )
                )

            if not dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                _apply_pdf_redactions(input_path, output_path, boxes_by_page)
        else:
            import cv2

            img = cv2.imread(str(input_path))
            if img is not None:
                boxes: list[RedactionBox] = []
                boxes.extend(_run_ocr_on_image(img, 0))
                boxes.extend(_run_insurance_ner_on_image(img, 0))
                boxes.extend(self._run_scispacy(img, 0))
                boxes_by_page[0] = boxes
                page_results.append(
                    PageResult(page=0, doc_type=doc_type, redaction_boxes=boxes)
                )

            if not dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                _apply_image_redactions(input_path, output_path, boxes_by_page.get(0, []))

        return DocumentResult(
            doc_id=doc_id,
            filename=input_path.name,
            doc_type=doc_type,
            pages=len(page_results),
            page_results=page_results,
        )

    def _process_unknown(
        self,
        input_path: Path,
        output_path: Path,
        doc_id: str,
        doc_type: DocumentType,
        dry_run: bool,
    ) -> DocumentResult:
        """Conservative: run all layers."""
        from pii_redact.config import settings as cfg

        suffix = input_path.suffix.lower()
        dpi = cfg.pipeline.get("ocr_dpi", 300)
        page_results: list[PageResult] = []
        boxes_by_page: dict[int, list[RedactionBox]] = {}

        if suffix == ".pdf":
            for page_idx, page_boxes in _extract_pdf_text_boxes(input_path):
                boxes_by_page.setdefault(page_idx, []).extend(page_boxes)

            for page_idx, img in _render_pdf_pages(input_path, dpi=dpi):
                boxes_by_page.setdefault(page_idx, []).extend(
                    self._run_visual_layer(img, doc_type, page_idx)
                )
                boxes_by_page[page_idx].extend(_run_insurance_ner_on_image(img, page_idx))
                boxes_by_page[page_idx].extend(self._run_scispacy(img, page_idx))

            for page_idx in sorted(boxes_by_page):
                page_results.append(
                    PageResult(page=page_idx, doc_type=doc_type, redaction_boxes=boxes_by_page[page_idx])
                )

            if not dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                _apply_pdf_redactions(input_path, output_path, boxes_by_page)
        else:
            import cv2

            img = cv2.imread(str(input_path))
            if img is not None:
                boxes: list[RedactionBox] = []
                boxes.extend(_run_ocr_on_image(img, 0))
                boxes.extend(self._run_visual_layer(img, doc_type, 0))
                boxes.extend(_run_insurance_ner_on_image(img, 0))
                boxes.extend(self._run_scispacy(img, 0))
                boxes_by_page[0] = boxes
                page_results.append(
                    PageResult(page=0, doc_type=doc_type, redaction_boxes=boxes)
                )

            if not dry_run:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                _apply_image_redactions(input_path, output_path, boxes_by_page.get(0, []))
                _strip_exif(output_path)

        return DocumentResult(
            doc_id=doc_id,
            filename=input_path.name,
            doc_type=doc_type,
            pages=len(page_results),
            page_results=page_results,
        )

    # ------------------------------------------------------------------
    # Layer helpers
    # ------------------------------------------------------------------

    def _run_visual_layer(
        self,
        image,
        doc_type: DocumentType,
        page_idx: int,
    ) -> list[RedactionBox]:
        """Run VisualLayer + optional VLM extractor on an image array."""
        import tempfile

        import cv2

        boxes: list[RedactionBox] = []

        try:
            from pii_redact.layers.visual_layer import VisualLayer

            vl = VisualLayer()
            vl_boxes = vl.process(image, doc_type)
            for box in vl_boxes:
                box.page = page_idx
            boxes.extend(vl_boxes)
        except Exception as exc:
            logger.warning("VisualLayer failed on page %d: %s", page_idx, exc)

        if self._use_vlm:
            # Write array to a temp file so VLM can read it via PIL
            try:
                with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
                    tmp_path = Path(tmp.name)
                cv2.imwrite(str(tmp_path), image)
                vlm_boxes = self._get_vlm().extract(tmp_path, page=page_idx)
                boxes.extend(vlm_boxes)
                logger.debug("VLM found %d entity candidate(s) on page %d", len(vlm_boxes), page_idx)
            except Exception as exc:
                logger.warning("VLM extraction failed on page %d: %s", page_idx, exc)
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        return boxes

    def _run_scispacy(self, image, page_idx: int = 0) -> list[RedactionBox]:
        """Run scispaCy NER over OCR text from *image*."""
        boxes: list[RedactionBox] = []
        try:
            from paddleocr import PaddleOCR  # type: ignore[import]

            from pii_redact.config import settings as cfg

            scispacy_model = cfg.models.get("scispacy_model", "en_core_sci_lg")
            import spacy

            nlp = spacy.load(scispacy_model)

            ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
            result = ocr.ocr(image, cls=True)
            if not result or not result[0]:
                return []

            for line in result[0]:
                if not line:
                    continue
                coords, (text, _confidence) = line
                if not text.strip():
                    continue
                xs = [pt[0] for pt in coords]
                ys = [pt[1] for pt in coords]
                x, y = int(min(xs)), int(min(ys))
                w = int(max(xs) - min(xs))
                h = int(max(ys) - min(ys))

                doc = nlp(text)
                for ent in doc.ents:
                    boxes.append(
                        RedactionBox(
                            x=x,
                            y=y,
                            w=w,
                            h=h,
                            entity_type=ent.label_,
                            confidence=0.75,
                            source="scispacy",
                            page=page_idx,
                            text_found=ent.text,
                        )
                    )
        except Exception as exc:
            logger.debug("scispaCy NER failed: %s", exc)
        return boxes
