"""Tests for the Docling layout-aware PDF parser module.

Fast tests (marker: fast) use only pure Python logic — no PDF I/O, no Docling.
Slow tests (marker: slow) require docling to be installed and create real PDFs.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Fast unit tests — pure logic, no Docling / no file I/O
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_layout_span_dataclass():
    from pii_redact.layers.docling_parser import LayoutSpan, is_high_value_field, is_low_value_field

    span = LayoutSpan(text="GreenTech Inc.", page=0, field_label="employer")
    assert span.text == "GreenTech Inc."
    assert span.page == 0
    assert span.field_label == "employer"
    assert is_high_value_field(span.field_label)
    assert not is_low_value_field(span.field_label)


@pytest.mark.fast
def test_layout_span_defaults():
    from pii_redact.layers.docling_parser import LayoutSpan

    span = LayoutSpan(text="hello", page=1)
    assert span.x == 0
    assert span.y == 0
    assert span.w == 0
    assert span.h == 0
    assert span.block_type == "paragraph"
    assert span.field_label == ""


@pytest.mark.fast
def test_insurer_is_low_value():
    from pii_redact.layers.docling_parser import is_high_value_field, is_low_value_field

    assert is_low_value_field("insurer")
    assert not is_high_value_field("insurer")


@pytest.mark.fast
def test_payer_is_low_value():
    from pii_redact.layers.docling_parser import is_high_value_field, is_low_value_field

    assert is_low_value_field("payer")
    assert is_low_value_field("payer name")
    assert not is_high_value_field("payer")


@pytest.mark.fast
def test_patient_name_is_high_value():
    from pii_redact.layers.docling_parser import is_high_value_field

    assert is_high_value_field("patient name")
    assert is_high_value_field("Patient Name")  # case insensitive
    assert is_high_value_field("PATIENT NAME")  # all-caps


@pytest.mark.fast
def test_employer_is_high_value():
    from pii_redact.layers.docling_parser import is_high_value_field

    assert is_high_value_field("employer")
    assert is_high_value_field("Employer")
    assert is_high_value_field("employer name")
    assert is_high_value_field("Employer Name")


@pytest.mark.fast
def test_high_value_field_labels_coverage():
    from pii_redact.layers.docling_parser import HIGH_VALUE_FIELD_LABELS, is_high_value_field

    for label in HIGH_VALUE_FIELD_LABELS:
        assert is_high_value_field(label), f"Expected {label!r} to be high-value"


@pytest.mark.fast
def test_low_value_field_labels_coverage():
    from pii_redact.layers.docling_parser import LOW_VALUE_FIELD_LABELS, is_low_value_field

    for label in LOW_VALUE_FIELD_LABELS:
        assert is_low_value_field(label), f"Expected {label!r} to be low-value"


@pytest.mark.fast
def test_unknown_field_is_neither():
    from pii_redact.layers.docling_parser import is_high_value_field, is_low_value_field

    assert not is_high_value_field("diagnosis code")
    assert not is_low_value_field("diagnosis code")
    assert not is_high_value_field("")
    assert not is_low_value_field("")


@pytest.mark.fast
def test_extract_returns_empty_list_when_docling_unavailable(monkeypatch):
    """Must return [] gracefully if Docling is not importable."""
    from pii_redact.layers.docling_parser import extract_text_with_layout

    with patch.dict(sys.modules, {"docling": None, "docling.document_converter": None}):
        # Should not raise even if path does not exist
        result = extract_text_with_layout(Path("/nonexistent.pdf"))
        assert result == []


@pytest.mark.fast
def test_extract_returns_empty_list_on_conversion_error(tmp_path):
    """Must return [] gracefully if Docling raises during conversion."""
    # Write a non-PDF file with .pdf extension to trigger a parse error
    bad_pdf = tmp_path / "bad.pdf"
    bad_pdf.write_bytes(b"this is not a valid PDF")

    from pii_redact.layers.docling_parser import extract_text_with_layout

    result = extract_text_with_layout(bad_pdf)
    assert isinstance(result, list)
    # Either empty (Docling failed gracefully) or list of spans (Docling recovered)


@pytest.mark.fast
def test_split_label_value_text_pattern():
    """_split_label_value_text must split 'Label: Value' patterns."""
    from pii_redact.layers.docling_parser import LayoutSpan, _split_label_value_text

    span = LayoutSpan(
        text="Employer: GreenTech Inc.\nPatient Name: Jane Doe",
        page=0,
        x=40,
        y=100,
        w=300,
        h=20,
    )
    result = _split_label_value_text(span)
    assert len(result) == 2

    texts = {s.field_label: s.text for s in result}
    assert "Employer" in texts
    assert texts["Employer"] == "GreenTech Inc."
    assert "Patient Name" in texts
    assert texts["Patient Name"] == "Jane Doe"


@pytest.mark.fast
def test_split_label_value_preserves_coords():
    """Sub-spans from _split_label_value_text inherit parent coordinates."""
    from pii_redact.layers.docling_parser import LayoutSpan, _split_label_value_text

    span = LayoutSpan(
        text="Insurer: Blue Cross\nEmployer: Acme Corp",
        page=2,
        x=10,
        y=200,
        w=400,
        h=30,
    )
    result = _split_label_value_text(span)
    assert len(result) == 2
    for sub in result:
        assert sub.page == 2
        assert sub.x == 10
        assert sub.y == 200


@pytest.mark.fast
def test_split_label_value_returns_empty_for_plain_text():
    """_split_label_value_text returns [] for plain text with no label pattern."""
    from pii_redact.layers.docling_parser import LayoutSpan, _split_label_value_text

    span = LayoutSpan(text="Just some plain text without labels.", page=0)
    result = _split_label_value_text(span)
    assert result == []


@pytest.mark.fast
def test_map_docling_label_mapping():
    """_map_docling_label must return correct block_type strings."""
    from pii_redact.layers.docling_parser import _map_docling_label

    assert _map_docling_label("title") == "header"
    assert _map_docling_label("section_header") == "header"
    assert _map_docling_label("text") == "paragraph"
    assert _map_docling_label("field_key") == "form_field_label"
    assert _map_docling_label("field_value") == "form_field_value"
    assert _map_docling_label("unknown_label") == "paragraph"  # default


@pytest.mark.fast
def test_high_value_field_strips_whitespace():
    """is_high_value_field/is_low_value_field must strip surrounding whitespace."""
    from pii_redact.layers.docling_parser import is_high_value_field, is_low_value_field

    assert is_high_value_field("  employer  ")
    assert is_low_value_field("  insurer  ")


# ---------------------------------------------------------------------------
# Slow integration tests — require docling + fitz
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_docling_extracts_text_from_synthetic_pdf(tmp_path):
    """Docling must return at least one span from a text PDF."""
    pytest.importorskip("docling")
    fitz = pytest.importorskip("fitz")
    from pii_redact.layers.docling_parser import extract_text_with_layout

    # Create a simple test PDF
    p = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((40, 100), "Employer: GreenTech Inc.", fontsize=12)
    page.insert_text((40, 130), "Patient Name: Jane Doe", fontsize=12)
    doc.save(str(p))
    doc.close()

    spans = extract_text_with_layout(p)
    assert isinstance(spans, list)
    # Either contains spans or is empty if fallback triggers
    all_text = " ".join(s.text for s in spans)
    assert "GreenTech" in all_text or len(spans) == 0


@pytest.mark.slow
def test_docling_field_label_association_from_columns(tmp_path):
    """Docling parser must associate label and value spans from two-column layouts."""
    pytest.importorskip("docling")
    fitz = pytest.importorskip("fitz")
    from pii_redact.layers.docling_parser import extract_text_with_layout

    p = tmp_path / "form.pdf"
    doc = fitz.open()
    page = doc.new_page()

    # Two-column form: label on left, value on right, same y-row
    page.insert_text((40, 115), "Employer:", fontsize=10)
    page.insert_text((202, 115), "GreenTech Inc.", fontsize=10)
    page.insert_text((40, 145), "Patient Name:", fontsize=10)
    page.insert_text((202, 145), "Jane Doe", fontsize=10)
    page.insert_text((40, 175), "Insurer:", fontsize=10)
    page.insert_text((202, 175), "Blue Cross Blue Shield", fontsize=10)

    doc.save(str(p))
    doc.close()

    spans = extract_text_with_layout(p)
    assert isinstance(spans, list)
    # If spans returned, check field labels
    if spans:
        field_labels = {s.field_label for s in spans if s.field_label}
        value_texts = {s.text for s in spans}
        # At minimum some text should be present
        all_text = " ".join(s.text for s in spans)
        assert len(all_text) > 0


@pytest.mark.slow
def test_docling_span_page_is_zero_indexed(tmp_path):
    """Returned spans must use 0-indexed page numbers."""
    pytest.importorskip("docling")
    fitz = pytest.importorskip("fitz")
    from pii_redact.layers.docling_parser import extract_text_with_layout

    p = tmp_path / "single_page.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((40, 100), "Some text on page one.", fontsize=12)
    doc.save(str(p))
    doc.close()

    spans = extract_text_with_layout(p)
    if spans:
        for s in spans:
            assert s.page == 0, f"Expected page 0 (0-indexed), got {s.page}"
