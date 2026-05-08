"""Redaction engine: solid-fill image regions and PDF text layers.

SAFETY CONTRACT: This module NEVER uses blur, pixelation, or any reversible
masking technique. Only opaque solid-fill rectangles are applied, making
reconstruction attacks (e.g., Bishop Fox Unredacter) impossible.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Sequence

import cv2
import fitz  # PyMuPDF
import numpy as np
import piexif
from PIL import Image

from pii_redact.models import RedactionBox


# ---------------------------------------------------------------------------
# Box manipulation helpers
# ---------------------------------------------------------------------------


def _boxes_overlap(a: RedactionBox, b: RedactionBox, pad: int = 0) -> bool:
    """Return True when two boxes (already padded) overlap or touch."""
    return (
        a.x <= b.x + b.w
        and a.x + a.w >= b.x
        and a.y <= b.y + b.h
        and a.y + a.h >= b.y
    )


def _union_box(a: RedactionBox, b: RedactionBox) -> RedactionBox:
    """Return the smallest box enclosing both a and b (same page assumed)."""
    x = min(a.x, b.x)
    y = min(a.y, b.y)
    x2 = max(a.x + a.w, b.x + b.w)
    y2 = max(a.y + a.h, b.y + b.h)
    # Keep metadata from the higher-confidence box
    src = a if a.confidence >= b.confidence else b
    return RedactionBox(
        x=x,
        y=y,
        w=x2 - x,
        h=y2 - y,
        entity_type=src.entity_type,
        confidence=max(a.confidence, b.confidence),
        source=src.source,
        page=a.page,
        text_found="",  # merged box has no single text value
    )


def deduplicate_boxes(boxes: list[RedactionBox]) -> list[RedactionBox]:
    """Deduplicate overlapping redaction boxes using two independent passes.

    Pass A — text-span deduplication:
        Groups boxes by (page, text_found) where text_found is not None/empty.
        Within each group, keeps the single box with highest confidence.
        Tiebreaks by entity-type specificity score (higher = more specific type wins).

    Pass B — coordinate IoU deduplication:
        For boxes on the same page where text_found is None or empty, computes IoU.
        Sorts by confidence descending; suppresses any box with IoU > 0.5 against
        a kept box.

    Args:
        boxes: Raw detection boxes from one or more pipeline layers.

    Returns:
        Deduplicated list of RedactionBox objects.
    """
    if not boxes:
        return []

    # Entity-type specificity scores for tiebreaking
    _TYPE_SPECIFICITY: dict[str, int] = {
        "US_SSN": 100,
        "NPI": 99,
        "EIN": 98,
        "DEA_NUM": 97,
        "ROUTING_NUM": 96,
        "POLICY_NUM": 95,
        "CLAIM_REF": 94,
        "ADJUSTER_ID": 93,
        "CREDIT_CARD": 92,
        "IBAN_CODE": 91,
        "EMAIL_ADDRESS": 90,
        "PHONE_NUMBER": 89,
        "IP_ADDRESS": 88,
        "DATE_TIME": 80,
        "PERSON": 70,
        "LOCATION": 60,
        "ORG": 50,
        "ORGANIZATION": 50,
        "NRP": 40,
    }

    def _specificity(box: RedactionBox) -> int:
        return _TYPE_SPECIFICITY.get(box.entity_type, 0)

    def _sort_key(box: RedactionBox):
        # Higher specificity first (most specific entity type wins),
        # then higher confidence breaks remaining ties.
        return (_specificity(box), box.confidence)

    # Split into text-layer boxes (text_found set) and visual boxes (text_found empty/None)
    text_boxes: list[RedactionBox] = []
    visual_boxes: list[RedactionBox] = []
    for b in boxes:
        if b.text_found:  # non-empty string — text-layer detection
            text_boxes.append(b)
        else:
            visual_boxes.append(b)

    # ------------------------------------------------------------------
    # Pass A: text-span deduplication
    # ------------------------------------------------------------------
    from collections import defaultdict

    groups: dict[tuple[int, str], list[RedactionBox]] = defaultdict(list)
    for b in text_boxes:
        groups[(b.page, b.text_found)].append(b)

    pass_a_result: list[RedactionBox] = []
    for group in groups.values():
        # Keep the box with highest (confidence, specificity)
        best = max(group, key=_sort_key)
        pass_a_result.append(best)

    # ------------------------------------------------------------------
    # Pass B: coordinate IoU deduplication (visual / raster boxes)
    # ------------------------------------------------------------------

    def _iou(a: RedactionBox, b: RedactionBox) -> float:
        """Compute IoU of two axis-aligned boxes."""
        # Intersection
        ix1 = max(a.x, b.x)
        iy1 = max(a.y, b.y)
        ix2 = min(a.x + a.w, b.x + b.w)
        iy2 = min(a.y + a.h, b.y + b.h)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        union = a.w * a.h + b.w * b.h - inter
        if union <= 0:
            return 0.0
        return inter / union

    # Group visual boxes by page then apply NMS
    by_page_visual: dict[int, list[RedactionBox]] = defaultdict(list)
    for b in visual_boxes:
        by_page_visual[b.page].append(b)

    pass_b_result: list[RedactionBox] = []
    for page_boxes in by_page_visual.values():
        # Sort by confidence descending
        sorted_boxes = sorted(page_boxes, key=lambda b: b.confidence, reverse=True)
        kept: list[RedactionBox] = []
        suppressed = [False] * len(sorted_boxes)
        for i, candidate in enumerate(sorted_boxes):
            if suppressed[i]:
                continue
            kept.append(candidate)
            for j in range(i + 1, len(sorted_boxes)):
                if suppressed[j]:
                    continue
                if _iou(candidate, sorted_boxes[j]) > 0.5:
                    suppressed[j] = True
        pass_b_result.extend(kept)

    return pass_a_result + pass_b_result


def merge_boxes(boxes: list[RedactionBox], padding: int = 0) -> list[RedactionBox]:
    """Pad, merge overlapping boxes per page, and sort by (page, y, x).

    Args:
        boxes:   Raw detection boxes.
        padding: Pixels to expand each box on all four sides before merging.

    Returns:
        Deduplicated, merged, sorted list.
    """
    if not boxes:
        return []

    # Apply padding (clamp to 0 to avoid negative dimensions)
    padded: list[RedactionBox] = []
    for b in boxes:
        padded.append(
            RedactionBox(
                x=max(0, b.x - padding),
                y=max(0, b.y - padding),
                w=b.w + 2 * padding,
                h=b.h + 2 * padding,
                entity_type=b.entity_type,
                confidence=b.confidence,
                source=b.source,
                page=b.page,
                text_found=b.text_found,
            )
        )

    # Group by page
    from collections import defaultdict

    by_page: dict[int, list[RedactionBox]] = defaultdict(list)
    for b in padded:
        by_page[b.page].append(b)

    merged_all: list[RedactionBox] = []
    for page_num in sorted(by_page):
        page_boxes = by_page[page_num]

        # Iteratively merge until stable (handles chain overlaps)
        changed = True
        while changed:
            changed = False
            result: list[RedactionBox] = []
            used = [False] * len(page_boxes)
            for i, bi in enumerate(page_boxes):
                if used[i]:
                    continue
                current = bi
                for j in range(i + 1, len(page_boxes)):
                    if used[j]:
                        continue
                    if _boxes_overlap(current, page_boxes[j]):
                        current = _union_box(current, page_boxes[j])
                        used[j] = True
                        changed = True
                result.append(current)
            page_boxes = result

        merged_all.extend(page_boxes)

    merged_all.sort(key=lambda b: (b.page, b.y, b.x))
    return merged_all


# ---------------------------------------------------------------------------
# Image redaction
# ---------------------------------------------------------------------------


def redact_image(
    image: np.ndarray,
    boxes: list[RedactionBox],
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> np.ndarray:
    """Apply solid opaque rectangles to a copy of *image*.

    SAFETY: Uses cv2.rectangle with thickness=-1 (filled).  After each fill the
    pixel at the box centre is verified to equal fill_color — this assertion
    catches any colour-space mismatch or accidental transparency layer.

    Args:
        image:      BGR uint8 numpy array (OpenCV format).
        boxes:      Redaction boxes (page attribute is ignored here).
        fill_color: BGR tuple, default solid black.

    Returns:
        A new array — the input is NOT modified.
    """
    out = image.copy()
    h_img, w_img = out.shape[:2]

    for box in boxes:
        x1 = max(0, box.x)
        y1 = max(0, box.y)
        x2 = min(w_img, box.x + box.w)
        y2 = min(h_img, box.y + box.h)

        if x2 <= x1 or y2 <= y1:
            continue  # degenerate box — skip

        cv2.rectangle(out, (x1, y1), (x2 - 1, y2 - 1), fill_color, thickness=-1)

        # --- SAFETY ASSERTION ---
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        actual = tuple(int(v) for v in out[cy, cx, :3])
        assert actual == tuple(fill_color), (
            f"Solid-fill safety check failed at ({cx},{cy}): "
            f"expected {fill_color}, got {actual}. "
            "Reversible masking is NEVER acceptable for PII redaction."
        )

    return out


# ---------------------------------------------------------------------------
# PDF redaction
# ---------------------------------------------------------------------------


def redact_pdf_page(
    page: fitz.Page,
    boxes: list[RedactionBox],
    fill_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Permanently redact text and graphics within *boxes* on a PDF page.

    Uses PyMuPDF's two-phase approach:
      1. add_redact_annot() — marks the region.
      2. apply_redactions()  — removes the underlying text data from the PDF
                               stream (not just drawn over).

    This is critical: drawing a black rectangle without apply_redactions()
    leaves the underlying text recoverable via copy-paste or text extraction.

    Args:
        page:       PyMuPDF Page object (modified in-place).
        boxes:      Boxes whose page attribute matches this page's number.
        fill_color: RGB integers 0-255; normalised to 0-1 for fitz.
    """
    # Normalise 0-255 → 0.0-1.0
    r, g, b = (c / 255.0 for c in fill_color)

    page_rect = page.rect
    pw, ph = page_rect.width, page_rect.height

    for box in boxes:
        # Clamp to page bounds
        x0 = max(0.0, float(box.x))
        y0 = max(0.0, float(box.y))
        x1 = min(pw, float(box.x + box.w))
        y1 = min(ph, float(box.y + box.h))

        if x1 <= x0 or y1 <= y0:
            continue

        rect = fitz.Rect(x0, y0, x1, y1)
        page.add_redact_annot(rect, fill=(r, g, b))

    # apply_redactions removes the actual text bytes from the PDF content stream
    page.apply_redactions()


def save_redacted_pdf(doc: fitz.Document, output_path: Path) -> None:
    """Save *doc* with metadata stripped and aggressive compression.

    Args:
        doc:         Open fitz.Document with redactions already applied.
        output_path: Destination path (parent directory must exist).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Strip all PDF metadata fields
    doc.set_metadata({})

    doc.save(
        str(output_path),
        garbage=4,       # most aggressive object/xref compaction
        deflate=True,    # compress content streams
        clean=True,      # sanitise content streams
    )


# ---------------------------------------------------------------------------
# Image save with EXIF stripping
# ---------------------------------------------------------------------------


def save_redacted_image(
    image: np.ndarray,
    output_path: Path,
    original_format: str,
) -> None:
    """Save a redacted image, stripping EXIF metadata.

    Args:
        image:           BGR uint8 numpy array.
        output_path:     Destination path.
        original_format: "JPEG", "PNG", "TIFF", etc. (PIL format string).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = original_format.upper()

    if fmt in ("JPEG", "JPG"):
        cv2.imwrite(
            str(output_path),
            image,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        # Strip EXIF using piexif after writing
        try:
            piexif.remove(str(output_path))
        except Exception:
            # If piexif fails (e.g. no EXIF present), that's fine
            pass
    elif fmt == "PNG":
        cv2.imwrite(
            str(output_path),
            image,
            [cv2.IMWRITE_PNG_COMPRESSION, 9],  # lossless, max compression
        )
    else:
        # Fallback via Pillow for TIFF, BMP, etc.
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pil_img.save(str(output_path), format=fmt)


# ---------------------------------------------------------------------------
# High-level engine class
# ---------------------------------------------------------------------------


class RedactionEngine:
    """Orchestrates loading, redacting, and saving documents.

    Usage::

        engine = RedactionEngine(settings)
        stats = engine.process_pdf(input_path, boxes, output_path)
    """

    def __init__(self, settings) -> None:
        self._settings = settings
        cfg = settings.redaction
        raw_color = cfg.get("fill_color", [0, 0, 0])
        self._fill_color: tuple[int, int, int] = tuple(raw_color)  # type: ignore[assignment]
        self._padding: int = int(cfg.get("fill_padding_px", 4))

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------

    def process_pdf(
        self,
        input_path: Path,
        boxes: list[RedactionBox],
        output_path: Path,
    ) -> dict:
        """Open a PDF, apply solid-fill redactions page-by-page, save output.

        Args:
            input_path:  Source PDF.
            boxes:       All redaction boxes (may span multiple pages).
            output_path: Where to write the redacted PDF.

        Returns:
            Stats dict with keys: pages, total_boxes, pages_redacted,
            processing_ms.
        """
        t0 = time.monotonic()

        merged = merge_boxes(boxes, padding=self._padding)

        doc = fitz.open(str(input_path))
        pages_redacted = 0

        for page_num in range(len(doc)):
            page_boxes = [b for b in merged if b.page == page_num]
            if not page_boxes:
                continue
            page = doc[page_num]
            redact_pdf_page(page, page_boxes, fill_color=self._fill_color)
            pages_redacted += 1

        save_redacted_pdf(doc, output_path)
        doc.close()

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return {
            "pages": len(doc) if not doc.is_closed else doc.page_count,
            "total_boxes": len(merged),
            "pages_redacted": pages_redacted,
            "processing_ms": round(elapsed_ms, 1),
        }

    # ------------------------------------------------------------------
    # Image
    # ------------------------------------------------------------------

    def process_image(
        self,
        input_path: Path,
        boxes: list[RedactionBox],
        output_path: Path,
    ) -> dict:
        """Open an image, apply solid-fill redactions, save output.

        Args:
            input_path:  Source image file.
            boxes:       Redaction boxes (page=0 expected for images).
            output_path: Where to write the redacted image.

        Returns:
            Stats dict with keys: total_boxes, processing_ms.
        """
        t0 = time.monotonic()

        merged = merge_boxes(boxes, padding=self._padding)

        image = cv2.imread(str(input_path))
        if image is None:
            raise ValueError(f"Could not open image: {input_path}")

        redacted = redact_image(image, merged, fill_color=self._fill_color)

        # Detect original format from suffix
        suffix = input_path.suffix.lstrip(".").upper()
        fmt_map = {"JPG": "JPEG"}
        original_format = fmt_map.get(suffix, suffix) or "JPEG"

        save_redacted_image(redacted, output_path, original_format)

        elapsed_ms = (time.monotonic() - t0) * 1000.0
        return {
            "total_boxes": len(merged),
            "processing_ms": round(elapsed_ms, 1),
        }
