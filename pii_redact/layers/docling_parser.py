"""Layout-aware PDF text extraction using Docling (IBM Research, MIT license).

Provides field-level context: knows if a text span is in an "Employer" field
vs. an "Insurer" field, enabling downstream ORG disambiguation.

Docling API (v2.x):
    from docling.document_converter import DocumentConverter
    result = converter.convert(str(pdf_path))
    doc = result.document
    # doc.texts  — list[TextItem]
    # doc.tables — list[TableItem]
    # doc.pages  — dict[int, PageItem]  (1-indexed page numbers)
    # TextItem.prov — list[ProvenanceItem] with .page_no and .bbox (BOTTOMLEFT origin)
    # BoundingBox.to_top_left_origin(page_height) — converts to TOPLEFT coords
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LayoutSpan:
    """A text span with its structural context from Docling layout analysis."""

    text: str
    page: int
    x: int = 0
    y: int = 0
    w: int = 0
    h: int = 0
    block_type: str = "paragraph"  # "paragraph", "table_cell", "form_field_label", "form_field_value", "header"
    field_label: str = ""  # e.g. "Employer", "Patient Name", "Insurer" — from nearby label text


# Insurance form field labels that indicate PII values (must redact if PERSON/ORG detected)
HIGH_VALUE_FIELD_LABELS: frozenset[str] = frozenset(
    {
        "employer",
        "employer name",
        "employer's name",
        "place of employment",
        "patient name",
        "patient's name",
        "insured name",
        "insured's name",
        "policyholder",
        "claimant",
        "claimant name",
        "subscriber",
        "guarantor",
        "responsible party",
        "guarantor name",
        "referring physician",
        "attending physician",
        "rendering provider name",
        # Common shortened forms seen in real forms
        "employee",
        "insured",
        "patient",
        "claimant's name",
        "member name",
        "employee name",
    }
)

# Field labels that are context/institutional references (redact only if high confidence)
LOW_VALUE_FIELD_LABELS: frozenset[str] = frozenset(
    {
        "insurer",
        "insurance company",
        "payer",
        "payer name",
        "court",
        "agency",
        "department",
        "issuing authority",
        "insurance plan",
        "plan name",
        "carrier",
        "carrier name",
        "group name",
    }
)

# Regex: split a line like "Label: Value" or "Label Name: Value"
_LABEL_VALUE_RE = re.compile(
    r"^([A-Z][A-Za-z '’]+?):\s*(.+)$"
)


def is_high_value_field(field_label: str) -> bool:
    """Return True if this field label indicates PII that must be redacted."""
    return field_label.lower().strip() in HIGH_VALUE_FIELD_LABELS


def is_low_value_field(field_label: str) -> bool:
    """Return True if this field contains institutional context (not patient PII)."""
    return field_label.lower().strip() in LOW_VALUE_FIELD_LABELS


def extract_text_with_layout(pdf_path: Path) -> list[LayoutSpan]:
    """Parse *pdf_path* with Docling and return text spans with field context.

    Falls back to returning an empty list if Docling is unavailable or fails.
    The caller should fall back to PyMuPDF extraction on failure.

    Coordinate system: returned x, y, w, h are in PDF points using
    top-left origin (same as PyMuPDF / fitz.Rect convention).
    """
    try:
        from docling.document_converter import DocumentConverter  # type: ignore[import]
    except ImportError:
        logger.debug("Docling not installed — layout parsing unavailable")
        return []

    try:
        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))
        return _parse_docling_result(result)
    except Exception as exc:
        logger.warning("Docling extraction failed for %s: %s", pdf_path.name, exc)
        return []


def _parse_docling_result(result) -> list[LayoutSpan]:  # noqa: ANN001
    """Convert Docling's output structure to LayoutSpan objects.

    Docling v2.x API:
    - result.document.texts  — list of TextItem objects
    - result.document.tables — list of TableItem objects
    - result.document.pages  — dict {page_no (1-indexed): PageItem}
    - TextItem.text          — full text content (may contain multiple "Label: Value" pairs)
    - TextItem.label         — DocItemLabel enum (text, title, section_header, …)
    - TextItem.prov          — list of ProvenanceItem with .page_no and .bbox
    - BoundingBox            — BOTTOMLEFT origin; use .to_top_left_origin(page_height)

    Strategy:
    1. Build a page-height lookup from result.document.pages
    2. For each TextItem, convert bbox to top-left origin
    3. Split the text on "Label: Value" patterns to extract field context
    4. For spatially close items in the same row, try to associate label spans
       with their value spans (items at the same y, different x columns)
    5. Also handle table cells using result.document.tables
    """
    doc = result.document
    spans: list[LayoutSpan] = []

    # Build page height map (Docling pages are 1-indexed)
    page_heights: dict[int, float] = {}
    for page_no, page_item in doc.pages.items():
        if page_item.size:
            page_heights[page_no] = page_item.size.height
        else:
            page_heights[page_no] = 842.0  # A4 default

    # Collect all text items with their layout info
    raw_spans: list[LayoutSpan] = []

    for text_item in doc.texts:
        if not text_item.text:
            continue
        if not text_item.prov:
            continue

        prov = text_item.prov[0]
        page_no = prov.page_no
        ph = page_heights.get(page_no, 842.0)

        bbox = prov.bbox.to_top_left_origin(ph)
        x = int(bbox.l)
        y = int(bbox.t)
        w = max(1, int(bbox.r - bbox.l))
        h = max(1, int(bbox.b - bbox.t))

        label_str = str(text_item.label.value) if text_item.label else "text"
        block_type = _map_docling_label(label_str)

        raw_spans.append(
            LayoutSpan(
                text=text_item.text,
                page=page_no - 1,  # convert to 0-indexed
                x=x,
                y=y,
                w=w,
                h=h,
                block_type=block_type,
                field_label="",
            )
        )

    # Extract table cells
    for table_item in doc.tables:
        if not table_item.prov:
            continue
        prov = table_item.prov[0]
        page_no = prov.page_no
        ph = page_heights.get(page_no, 842.0)

        try:
            data = table_item.data
            if not data:
                continue
            rows = data.grid if hasattr(data, "grid") else []
            for row in rows:
                for cell in row:
                    if not cell or not cell.text:
                        continue
                    cell_bbox = cell.bbox
                    if cell_bbox is None:
                        continue
                    c_bbox = cell_bbox.to_top_left_origin(ph)
                    raw_spans.append(
                        LayoutSpan(
                            text=cell.text,
                            page=page_no - 1,
                            x=int(c_bbox.l),
                            y=int(c_bbox.t),
                            w=max(1, int(c_bbox.r - c_bbox.l)),
                            h=max(1, int(c_bbox.b - c_bbox.t)),
                            block_type="table_cell",
                            field_label="",
                        )
                    )
        except Exception as exc:
            logger.debug("Table cell extraction failed: %s", exc)

    # Step 1: Attempt to associate label/value pairs from same-row spans
    #   (Items on the same page at similar y-coordinates)
    spans = _associate_field_labels(raw_spans)

    # Step 2: For spans that still have no field_label, try splitting inline
    #   "Label: Value" patterns from merged text blocks
    final: list[LayoutSpan] = []
    for span in spans:
        if span.field_label or "\n" not in span.text and ":" not in span.text:
            final.append(span)
            continue
        # Try to split inline "Label: Value" pairs
        sub_spans = _split_label_value_text(span)
        if sub_spans:
            final.extend(sub_spans)
        else:
            final.append(span)

    return final


def _map_docling_label(label_value: str) -> str:
    """Map Docling DocItemLabel values to our block_type strings."""
    label_map = {
        "title": "header",
        "section_header": "header",
        "page_header": "header",
        "page_footer": "paragraph",
        "text": "paragraph",
        "paragraph": "paragraph",
        "list_item": "paragraph",
        "table": "table_cell",
        "caption": "paragraph",
        "form": "form_field_value",
        "key_value_region": "form_field_value",
        "field_region": "form_field_value",
        "field_heading": "form_field_label",
        "field_item": "form_field_value",
        "field_key": "form_field_label",
        "field_value": "form_field_value",
    }
    return label_map.get(label_value, "paragraph")


def _associate_field_labels(spans: list[LayoutSpan]) -> list[LayoutSpan]:
    """Try to associate value spans with their label spans on the same row.

    Heuristic: two spans on the same page at similar y-coordinates
    (within 5px) where one ends in ":" or is a known label pattern are
    treated as label/value pairs. The value span gets the field_label set.
    """
    if not spans:
        return spans

    # Group by page
    by_page: dict[int, list[LayoutSpan]] = {}
    for s in spans:
        by_page.setdefault(s.page, []).append(s)

    result: list[LayoutSpan] = []
    for page_spans in by_page.values():
        # Sort by y then x
        page_spans.sort(key=lambda s: (s.y, s.x))

        # Track which spans have been assigned a field label
        for i, span in enumerate(page_spans):
            if span.field_label:
                result.append(span)
                continue

            # Check if this span looks like a field label
            stripped = span.text.strip()
            if stripped.endswith(":") or _is_label_text(stripped):
                label_text = stripped.rstrip(":").strip()
                # Look for value spans to the right on the same row
                for j, other in enumerate(page_spans):
                    if i == j or other.field_label:
                        continue
                    # Same row: y within 6 pixels
                    if abs(other.y - span.y) > 6:
                        continue
                    # Value must be to the right
                    if other.x <= span.x:
                        continue
                    other.field_label = label_text
                span.block_type = "form_field_label"
                result.append(span)
            else:
                result.append(span)

    return result


def _is_label_text(text: str) -> bool:
    """Return True if text looks like a form field label (title-case, short, no sentence)."""
    # Short, no sentence-ending punctuation, title-case or ALL-CAPS
    if len(text) > 40 or len(text.split()) > 5:
        return False
    if text and (text[0].isupper() or text.isupper()):
        return True
    return False


def _split_label_value_text(span: LayoutSpan) -> list[LayoutSpan]:
    """Split a merged text span containing 'Label: Value' patterns into sub-spans.

    For text like 'Employer: GreenTech Inc. Patient Name: Jane Doe',
    returns two spans — one with field_label='Employer' and one with
    field_label='Patient Name'. Coordinates are the same as the parent
    (we can't determine exact sub-span coordinates from text alone).
    """
    text = span.text.strip()
    # Try matching multiple "Label: Value" segments separated by newlines or
    # by the start of the next capitalized label pattern
    #
    # Split on pattern: "  Some Label:" — two or more spaces followed by a label
    # Also split on newlines
    segments = re.split(r"\n+", text)
    if len(segments) == 1:
        # Try splitting on "  Title Case Word(s):" patterns
        segments = re.split(r"\s{2,}(?=[A-Z])", text)

    sub_spans: list[LayoutSpan] = []
    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue
        m = _LABEL_VALUE_RE.match(segment)
        if m:
            label_part = m.group(1).strip()
            value_part = m.group(2).strip()
            if value_part:
                sub_spans.append(
                    LayoutSpan(
                        text=value_part,
                        page=span.page,
                        x=span.x,
                        y=span.y,
                        w=span.w,
                        h=span.h,
                        block_type="form_field_value",
                        field_label=label_part,
                    )
                )
            else:
                # Label with no value — emit the label span itself
                sub_spans.append(
                    LayoutSpan(
                        text=label_part,
                        page=span.page,
                        x=span.x,
                        y=span.y,
                        w=span.w,
                        h=span.h,
                        block_type="form_field_label",
                        field_label="",
                    )
                )
        else:
            sub_spans.append(span)

    return sub_spans if len(sub_spans) > 1 else []
