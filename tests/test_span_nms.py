"""Tests for span deduplication (NMS) — text-span + coordinate IoU passes.

All tests are marked @pytest.mark.fast (no ML imports, pure logic only).
"""

import pytest

from pii_redact.models import RedactionBox
from pii_redact.redact.engine import deduplicate_boxes


def _box(
    *,
    entity_type: str,
    confidence: float,
    text_found: str = "",
    page: int = 0,
    x: int = 10,
    y: int = 10,
    w: int = 100,
    h: int = 20,
    source: str = "test",
) -> RedactionBox:
    return RedactionBox(
        x=x,
        y=y,
        w=w,
        h=h,
        entity_type=entity_type,
        confidence=confidence,
        source=source,
        page=page,
        text_found=text_found,
    )


# ---------------------------------------------------------------------------
# Pass A — text-span deduplication
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_exact_duplicate_same_entity_type_keeps_higher_confidence():
    """Two boxes with same text_found and entity_type → keep the one with higher confidence."""
    low = _box(entity_type="PERSON", confidence=0.6, text_found="John Doe")
    high = _box(entity_type="PERSON", confidence=0.9, text_found="John Doe")

    result = deduplicate_boxes([low, high])

    assert len(result) == 1
    assert result[0].confidence == 0.9


@pytest.mark.fast
def test_exact_duplicate_different_entity_types_keeps_more_specific():
    """US_SSN (score 100) beats PERSON (score 70) on same text_found, lower confidence."""
    ssn_box = _box(entity_type="US_SSN", confidence=0.7, text_found="523-67-4891")
    person_box = _box(entity_type="PERSON", confidence=0.9, text_found="523-67-4891")

    result = deduplicate_boxes([ssn_box, person_box])

    assert len(result) == 1
    assert result[0].entity_type == "US_SSN"


@pytest.mark.fast
def test_partial_text_overlap_not_deduplicated():
    """Substring match does NOT trigger Pass A — only exact text_found equality does."""
    full = _box(entity_type="PERSON", confidence=0.8, text_found="John Michael Doe")
    partial = _box(entity_type="PERSON", confidence=0.8, text_found="John Doe")

    result = deduplicate_boxes([full, partial])

    assert len(result) == 2


@pytest.mark.fast
def test_no_overlap_both_kept():
    """Two boxes with completely different text_found values are both retained."""
    box1 = _box(entity_type="PERSON", confidence=0.8, text_found="Alice Smith")
    box2 = _box(entity_type="PHONE_NUMBER", confidence=0.9, text_found="555-1234")

    result = deduplicate_boxes([box1, box2])

    assert len(result) == 2


@pytest.mark.fast
def test_empty_input_returns_empty():
    """Empty input list → empty output."""
    assert deduplicate_boxes([]) == []


@pytest.mark.fast
def test_type_specificity_us_ssn_beats_person_same_confidence():
    """With equal confidence, US_SSN (specificity 100) beats PERSON (specificity 70)."""
    ssn_box = _box(entity_type="US_SSN", confidence=0.85, text_found="523-67-4891")
    person_box = _box(entity_type="PERSON", confidence=0.85, text_found="523-67-4891")

    result = deduplicate_boxes([ssn_box, person_box])

    assert len(result) == 1
    assert result[0].entity_type == "US_SSN"


# ---------------------------------------------------------------------------
# Pass B — coordinate IoU deduplication
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_pass_b_high_iou_keeps_higher_confidence():
    """Two visual boxes (text_found='') on same page with IoU=0.8 → keep higher confidence."""
    # box1: x=0,y=0 w=100,h=100 → area 10000
    # box2: x=10,y=10 w=100,h=100 → area 10000
    # intersection: x=10..100, y=10..100 → 90x90=8100
    # union: 10000+10000-8100=11900 → IoU≈0.68 > 0.5
    high = _box(entity_type="FACE", confidence=0.9, text_found="", x=0, y=0, w=100, h=100)
    low = _box(entity_type="FACE", confidence=0.6, text_found="", x=10, y=10, w=100, h=100)

    result = deduplicate_boxes([high, low])

    assert len(result) == 1
    assert result[0].confidence == 0.9


@pytest.mark.fast
def test_pass_b_low_iou_both_kept():
    """Two visual boxes with IoU=0.3 (< 0.5) → both retained."""
    # box1: x=0,y=0 w=100,h=100 → area 10000
    # box2: x=70,y=70 w=100,h=100 → area 10000
    # intersection: x=70..100, y=70..100 → 30x30=900
    # union: 10000+10000-900=19100 → IoU≈0.047 < 0.5
    box1 = _box(entity_type="FACE", confidence=0.8, text_found="", x=0, y=0, w=100, h=100)
    box2 = _box(entity_type="FACE", confidence=0.7, text_found="", x=70, y=70, w=100, h=100)

    result = deduplicate_boxes([box1, box2])

    assert len(result) == 2


@pytest.mark.fast
def test_pass_b_different_pages_no_suppression():
    """Two visual boxes with high IoU but on different pages → both kept."""
    # Same coordinates but different page numbers → should never be compared
    box_p0 = _box(entity_type="FACE", confidence=0.9, text_found="", page=0, x=0, y=0, w=100, h=100)
    box_p1 = _box(entity_type="FACE", confidence=0.6, text_found="", page=1, x=0, y=0, w=100, h=100)

    result = deduplicate_boxes([box_p0, box_p1])

    assert len(result) == 2
    pages = {b.page for b in result}
    assert pages == {0, 1}


# ---------------------------------------------------------------------------
# Mixed: text-layer + visual boxes — both passes run independently
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_mixed_text_and_visual_boxes_both_passes_run():
    """Text-layer boxes go through Pass A; visual boxes go through Pass B independently."""
    # Pass A input: two boxes for same SSN text, different entity types
    ssn_box = _box(entity_type="US_SSN", confidence=0.8, text_found="523-67-4891", page=0)
    person_box = _box(entity_type="PERSON", confidence=0.9, text_found="523-67-4891", page=0)

    # Pass B input: two overlapping visual boxes
    vis_high = _box(entity_type="FACE", confidence=0.9, text_found="", page=0, x=0, y=0, w=100, h=100)
    vis_low = _box(entity_type="FACE", confidence=0.5, text_found="", page=0, x=5, y=5, w=100, h=100)

    result = deduplicate_boxes([ssn_box, person_box, vis_high, vis_low])

    # Pass A: 2 → 1 (US_SSN wins)
    # Pass B: 2 → 1 (vis_high wins, IoU of those boxes is high)
    assert len(result) == 2

    entity_types = {b.entity_type for b in result}
    assert "US_SSN" in entity_types
    assert "FACE" in entity_types

    # The kept visual box should be the higher confidence one
    face_box = next(b for b in result if b.entity_type == "FACE")
    assert face_box.confidence == 0.9


@pytest.mark.fast
def test_pass_b_exact_overlap_iou_equals_one():
    """Two identical visual boxes → IoU = 1.0 → suppress lower confidence."""
    high = _box(entity_type="PLATE", confidence=0.95, text_found="", x=50, y=50, w=200, h=50)
    low = _box(entity_type="PLATE", confidence=0.60, text_found="", x=50, y=50, w=200, h=50)

    result = deduplicate_boxes([high, low])

    assert len(result) == 1
    assert result[0].confidence == 0.95


@pytest.mark.fast
def test_pass_a_multiple_groups_independent():
    """Multiple distinct text_found groups each independently deduplicated."""
    # Group 1: SSN
    ssn_a = _box(entity_type="US_SSN", confidence=0.9, text_found="111-22-3333")
    ssn_b = _box(entity_type="PERSON", confidence=0.95, text_found="111-22-3333")
    # Group 2: email
    email_a = _box(entity_type="EMAIL_ADDRESS", confidence=0.8, text_found="foo@bar.com")
    email_b = _box(entity_type="PERSON", confidence=0.7, text_found="foo@bar.com")

    result = deduplicate_boxes([ssn_a, ssn_b, email_a, email_b])

    assert len(result) == 2
    types = {b.entity_type for b in result}
    assert "US_SSN" in types       # wins group 1 (specificity 100 vs 70, conf tie)
    assert "EMAIL_ADDRESS" in types  # wins group 2 (specificity 90 vs 70, higher conf)
