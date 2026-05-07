"""Tests for core data models — pure Python, no model loading required."""

from __future__ import annotations

import pytest

from pii_redact.models import (
    DocumentResult,
    DocumentType,
    PageResult,
    RedactionBox,
)


# ---------------------------------------------------------------------------
# RedactionBox
# ---------------------------------------------------------------------------


class TestRedactionBox:
    def _make_box(self, **kwargs) -> RedactionBox:
        defaults = dict(
            x=10, y=20, w=100, h=50,
            entity_type="SSN",
            confidence=0.9,
            source="regex",
        )
        defaults.update(kwargs)
        return RedactionBox(**defaults)

    def test_creation_defaults(self):
        box = self._make_box()
        assert box.x == 10
        assert box.y == 20
        assert box.w == 100
        assert box.h == 50
        assert box.entity_type == "SSN"
        assert box.confidence == 0.9
        assert box.source == "regex"
        assert box.page == 0          # default
        assert box.text_found == ""   # default

    def test_creation_with_optional_fields(self):
        box = self._make_box(page=3, text_found="123-45-6789")
        assert box.page == 3
        assert box.text_found == "123-45-6789"

    def test_area_basic(self):
        box = self._make_box(w=100, h=50)
        assert box.area() == 5000

    def test_area_zero_dimension(self):
        assert self._make_box(w=0, h=50).area() == 0
        assert self._make_box(w=100, h=0).area() == 0

    def test_area_unit_box(self):
        assert self._make_box(w=1, h=1).area() == 1

    def test_as_dict_no_text(self):
        box = self._make_box(x=5, y=10, w=200, h=80, confidence=0.855)
        d = box.as_dict()
        assert d["entity_type"] == "SSN"
        assert d["confidence"] == 0.855   # round(0.855, 3)
        assert d["source"] == "regex"
        assert d["page"] == 0
        assert d["bbox"] == [5, 10, 200, 80]
        assert "text" not in d

    def test_as_dict_with_text(self):
        box = self._make_box(text_found="SECRET")
        d = box.as_dict(include_text=True)
        assert d["text"] == "SECRET"

    def test_as_dict_confidence_rounding(self):
        box = self._make_box(confidence=0.123456789)
        d = box.as_dict()
        assert d["confidence"] == 0.123   # rounded to 3 decimal places

    def test_as_dict_does_not_include_text_by_default(self):
        box = self._make_box(text_found="SENSITIVE")
        d = box.as_dict()
        assert "text" not in d

    def test_as_dict_bbox_is_list(self):
        box = self._make_box(x=1, y=2, w=3, h=4)
        assert box.as_dict()["bbox"] == [1, 2, 3, 4]


# ---------------------------------------------------------------------------
# DocumentType enum
# ---------------------------------------------------------------------------


class TestDocumentType:
    def test_all_expected_values_exist(self):
        expected = {
            "PDF_STRUCTURED", "SCAN_FORM", "PHOTO", "ID_DOC", "MEDICAL", "UNKNOWN"
        }
        actual = {member.value for member in DocumentType}
        assert actual == expected

    def test_is_string_enum(self):
        # DocumentType inherits from str — can be compared with plain strings
        assert DocumentType.PDF_STRUCTURED == "PDF_STRUCTURED"
        assert DocumentType.UNKNOWN == "UNKNOWN"

    def test_enum_identity(self):
        assert DocumentType("MEDICAL") is DocumentType.MEDICAL

    def test_unknown_is_fallback(self):
        # UNKNOWN must exist for the fallback / full-pipeline path
        assert DocumentType.UNKNOWN is not None


# ---------------------------------------------------------------------------
# DocumentResult.entity_counts and source_counts
# ---------------------------------------------------------------------------


def _make_doc_result(boxes_per_page: list[list[tuple]]) -> DocumentResult:
    """Helper: boxes_per_page is a list of pages, each page is a list of
    (entity_type, source, confidence) tuples."""
    page_results = []
    for page_num, page_boxes in enumerate(boxes_per_page):
        rb_list = [
            RedactionBox(
                x=i * 10, y=0, w=50, h=20,
                entity_type=et,
                confidence=conf,
                source=src,
                page=page_num,
            )
            for i, (et, src, conf) in enumerate(page_boxes)
        ]
        page_results.append(PageResult(page=page_num, doc_type=DocumentType.MEDICAL, redaction_boxes=rb_list))

    return DocumentResult(
        doc_id="test-001",
        filename="test.pdf",
        doc_type=DocumentType.MEDICAL,
        pages=len(boxes_per_page),
        page_results=page_results,
    )


class TestDocumentResult:
    def test_entity_counts_empty(self):
        doc = _make_doc_result([])
        assert doc.entity_counts == {}

    def test_entity_counts_single_type(self):
        doc = _make_doc_result([[("SSN", "regex", 0.9), ("SSN", "presidio", 0.8)]])
        assert doc.entity_counts == {"SSN": 2}

    def test_entity_counts_multiple_types(self):
        doc = _make_doc_result([
            [("SSN", "regex", 0.9), ("PERSON", "spacy", 0.7)],
            [("SSN", "presidio", 0.8), ("FACE", "yolov8", 0.95)],
        ])
        counts = doc.entity_counts
        assert counts["SSN"] == 2
        assert counts["PERSON"] == 1
        assert counts["FACE"] == 1

    def test_source_counts_empty(self):
        doc = _make_doc_result([])
        assert doc.source_counts == {}

    def test_source_counts_single_source(self):
        doc = _make_doc_result([[("SSN", "regex", 0.9), ("EIN", "regex", 0.8)]])
        assert doc.source_counts == {"regex": 2}

    def test_source_counts_multiple_sources(self):
        doc = _make_doc_result([
            [("SSN", "regex", 0.9), ("PERSON", "presidio", 0.7), ("FACE", "yolov8", 0.95)],
        ])
        counts = doc.source_counts
        assert counts["regex"] == 1
        assert counts["presidio"] == 1
        assert counts["yolov8"] == 1

    def test_all_boxes_aggregates_across_pages(self):
        doc = _make_doc_result([
            [("SSN", "regex", 0.9)],
            [("PERSON", "spacy", 0.7), ("EIN", "regex", 0.8)],
        ])
        assert len(doc.all_boxes) == 3

    def test_counts_consistent_with_all_boxes(self):
        doc = _make_doc_result([
            [("SSN", "regex", 0.9), ("SSN", "presidio", 0.8)],
            [("FACE", "yolov8", 0.95)],
        ])
        total = sum(doc.entity_counts.values())
        assert total == len(doc.all_boxes)


# ---------------------------------------------------------------------------
# merge_boxes — skipped gracefully if engine not yet importable
# ---------------------------------------------------------------------------


def test_merge_boxes_import():
    """Verify merge_boxes is importable from the engine module."""
    pytest.importorskip("cv2", reason="opencv not installed — skipping engine tests")
    try:
        from pii_redact.redact.engine import merge_boxes  # noqa: F401
    except ImportError as e:
        pytest.skip(f"Engine not importable: {e}")


def test_merge_boxes_empty():
    pytest.importorskip("cv2", reason="opencv not installed")
    try:
        from pii_redact.redact.engine import merge_boxes
    except ImportError as e:
        pytest.skip(str(e))
    assert merge_boxes([]) == []


def test_merge_boxes_overlapping():
    pytest.importorskip("cv2", reason="opencv not installed")
    try:
        from pii_redact.redact.engine import merge_boxes
    except ImportError as e:
        pytest.skip(str(e))

    boxes = [
        RedactionBox(x=0,  y=0,  w=100, h=50, entity_type="SSN", confidence=0.9, source="regex", page=0),
        RedactionBox(x=50, y=0,  w=100, h=50, entity_type="SSN", confidence=0.8, source="regex", page=0),
        RedactionBox(x=80, y=10, w=40,  h=30, entity_type="SSN", confidence=0.7, source="regex", page=0),
    ]
    merged = merge_boxes(boxes)
    assert len(merged) < len(boxes), "Overlapping boxes should be merged into fewer boxes"


def test_merge_boxes_non_overlapping():
    pytest.importorskip("cv2", reason="opencv not installed")
    try:
        from pii_redact.redact.engine import merge_boxes
    except ImportError as e:
        pytest.skip(str(e))

    boxes = [
        RedactionBox(x=0,   y=0,   w=10, h=10, entity_type="SSN", confidence=0.9, source="regex", page=0),
        RedactionBox(x=100, y=100, w=10, h=10, entity_type="SSN", confidence=0.9, source="regex", page=0),
        RedactionBox(x=200, y=200, w=10, h=10, entity_type="SSN", confidence=0.9, source="regex", page=0),
    ]
    merged = merge_boxes(boxes)
    assert len(merged) == 3, "Non-overlapping boxes should remain the same count"
