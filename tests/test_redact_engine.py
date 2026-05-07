"""Critical safety tests for the redaction engine.

All tests are designed to pass even in environments without GPU/model weights.
They require only: numpy, opencv-python-headless, PyMuPDF (fitz).
"""

from __future__ import annotations

import importlib
import io
import socket
import sys
from unittest import mock

import numpy as np
import pytest

from pii_redact.models import RedactionBox

pytestmark = pytest.mark.fast

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ENGINE_MODULE = "pii_redact.redact.engine"


def _import_engine():
    """Return the engine module or skip the test if unavailable."""
    try:
        return importlib.import_module(_ENGINE_MODULE)
    except ImportError as exc:
        pytest.skip(f"Engine not importable (missing dep?): {exc}")


def _white_image(h: int = 300, w: int = 300) -> np.ndarray:
    """Return a white BGR uint8 image of given size."""
    return np.full((h, w, 3), 255, dtype=np.uint8)


def _center_box(img: np.ndarray, size: int = 60) -> RedactionBox:
    """Return a RedactionBox centred in the image."""
    h, w = img.shape[:2]
    x = (w - size) // 2
    y = (h - size) // 2
    return RedactionBox(
        x=x, y=y, w=size, h=size,
        entity_type="TEST",
        confidence=1.0,
        source="test",
        page=0,
    )


# ---------------------------------------------------------------------------
# a) test_solid_fill_not_blur
# ---------------------------------------------------------------------------


class TestSolidFillNotBlur:
    """Prove that redact_image uses solid fill, not blur or gradient."""

    def test_solid_fill_not_blur(self):
        engine = _import_engine()
        img = _white_image()
        box = _center_box(img)
        fill = (0, 0, 0)

        result = engine.redact_image(img, [box], fill_color=fill)

        # Every pixel inside the box must be exactly fill_color
        roi = result[box.y : box.y + box.h, box.x : box.x + box.w]
        not_fill = roi[
            (roi[:, :, 0] != fill[0]) |
            (roi[:, :, 1] != fill[1]) |
            (roi[:, :, 2] != fill[2])
        ]
        assert len(not_fill) == 0, (
            f"Found {len(not_fill)} pixels inside the box that are NOT the fill color. "
            "Blur or gradient detected — only solid fill is acceptable."
        )

    def test_no_partial_coverage_pixels(self):
        """No pixel inside the box may be lighter than fill_color (no anti-aliasing)."""
        engine = _import_engine()
        img = _white_image()
        box = _center_box(img)
        fill = (0, 0, 0)

        result = engine.redact_image(img, [box], fill_color=fill)
        roi = result[box.y : box.y + box.h, box.x : box.x + box.w]

        # A "blur-like" pixel would be darker than original (white) but lighter than fill (black)
        # i.e. any channel value between 1 and 254 inclusive
        blur_like = roi[
            (roi[:, :, 0] > fill[0]) |
            (roi[:, :, 1] > fill[1]) |
            (roi[:, :, 2] > fill[2])
        ]
        assert len(blur_like) == 0, (
            f"{len(blur_like)} pixels found with partial coverage (blur-like). "
            "Only 100% opaque solid fill is acceptable for PII redaction."
        )


# ---------------------------------------------------------------------------
# b) test_fill_color_opaque
# ---------------------------------------------------------------------------


class TestFillColorOpaque:
    def test_fill_color_opaque_black(self):
        engine = _import_engine()
        img = _white_image()
        box = _center_box(img)
        fill = (0, 0, 0)

        result = engine.redact_image(img, [box], fill_color=fill)
        roi = result[box.y : box.y + box.h, box.x : box.x + box.w]

        assert roi.shape[0] > 0 and roi.shape[1] > 0, "ROI slice is empty"
        assert np.all(roi == np.array(fill, dtype=np.uint8)), (
            "Not every pixel inside the box equals fill_color. "
            "Expected 100% opaque solid fill."
        )

    def test_fill_color_opaque_custom_color(self):
        """Verify non-black fill colors also apply uniformly."""
        engine = _import_engine()
        img = _white_image()
        box = _center_box(img)
        fill = (128, 0, 200)   # arbitrary non-black BGR

        result = engine.redact_image(img, [box], fill_color=fill)
        roi = result[box.y : box.y + box.h, box.x : box.x + box.w]

        assert np.all(roi == np.array(fill, dtype=np.uint8)), (
            "Custom fill color not applied uniformly across the box."
        )

    def test_fill_does_not_bleed_outside_box(self):
        """Pixels immediately outside the box must be unchanged (white)."""
        engine = _import_engine()
        img = _white_image(300, 300)
        box = _center_box(img, size=60)
        fill = (0, 0, 0)

        result = engine.redact_image(img, [box], fill_color=fill)

        # Check a 1-pixel border outside all four edges of the box
        bx, by, bw, bh = box.x, box.y, box.w, box.h

        # Top row above box
        if by > 0:
            row = result[by - 1, bx : bx + bw]
            assert np.all(row == 255), "Fill bled above the box boundary"
        # Bottom row below box
        if by + bh < result.shape[0]:
            row = result[by + bh, bx : bx + bw]
            assert np.all(row == 255), "Fill bled below the box boundary"
        # Left column
        if bx > 0:
            col = result[by : by + bh, bx - 1]
            assert np.all(col == 255), "Fill bled to the left of the box boundary"
        # Right column
        if bx + bw < result.shape[1]:
            col = result[by : by + bh, bx + bw]
            assert np.all(col == 255), "Fill bled to the right of the box boundary"


# ---------------------------------------------------------------------------
# c) test_no_network_imports
# ---------------------------------------------------------------------------


class TestNoNetworkImports:
    """Core modules must be importable without making any network connections."""

    def _assert_no_network(self, module_name: str):
        """Import a module with socket.create_connection monkey-patched to raise."""
        # Remove from sys.modules to force re-import
        to_remove = [k for k in sys.modules if k == module_name or k.startswith(module_name + ".")]
        for k in to_remove:
            sys.modules.pop(k, None)

        def _no_network(*args, **kwargs):
            raise AssertionError(
                f"Network call attempted during import of {module_name!r}. "
                "All core modules must be importable offline."
            )

        with mock.patch("socket.create_connection", side_effect=_no_network):
            importlib.import_module(module_name)

    def test_models_no_network(self):
        self._assert_no_network("pii_redact.models")

    def test_config_no_network(self):
        self._assert_no_network("pii_redact.config")

    def test_engine_no_network(self):
        pytest.importorskip("cv2", reason="opencv not installed")
        pytest.importorskip("fitz", reason="PyMuPDF not installed")
        self._assert_no_network(_ENGINE_MODULE)


# ---------------------------------------------------------------------------
# d) test_merge_boxes_deduplication
# ---------------------------------------------------------------------------


class TestMergeBoxesDeduplication:
    def test_overlapping_boxes_are_merged(self):
        engine = _import_engine()

        boxes = [
            RedactionBox(x=0,  y=0,  w=100, h=50, entity_type="SSN", confidence=0.9, source="regex", page=0),
            RedactionBox(x=50, y=0,  w=100, h=50, entity_type="SSN", confidence=0.8, source="regex", page=0),
            RedactionBox(x=90, y=10, w=40,  h=30, entity_type="SSN", confidence=0.7, source="regex", page=0),
        ]
        merged = engine.merge_boxes(boxes)
        assert len(merged) < len(boxes), (
            f"Expected fewer than {len(boxes)} boxes after merging overlapping regions, got {len(merged)}"
        )

    def test_non_overlapping_boxes_unchanged_count(self):
        engine = _import_engine()

        boxes = [
            RedactionBox(x=0,   y=0,   w=10, h=10, entity_type="SSN", confidence=0.9, source="regex", page=0),
            RedactionBox(x=100, y=100, w=10, h=10, entity_type="SSN", confidence=0.9, source="regex", page=0),
            RedactionBox(x=200, y=200, w=10, h=10, entity_type="SSN", confidence=0.9, source="regex", page=0),
        ]
        merged = engine.merge_boxes(boxes)
        assert len(merged) == 3, (
            f"Non-overlapping boxes should not be merged. Expected 3, got {len(merged)}"
        )

    def test_different_pages_not_merged(self):
        engine = _import_engine()

        # Same coordinates but different pages — must NOT be merged
        boxes = [
            RedactionBox(x=0, y=0, w=100, h=50, entity_type="SSN", confidence=0.9, source="regex", page=0),
            RedactionBox(x=0, y=0, w=100, h=50, entity_type="SSN", confidence=0.9, source="regex", page=1),
        ]
        merged = engine.merge_boxes(boxes)
        assert len(merged) == 2, (
            "Boxes on different pages must never be merged together"
        )

    def test_empty_input(self):
        engine = _import_engine()
        assert engine.merge_boxes([]) == []

    def test_single_box_unchanged(self):
        engine = _import_engine()
        box = RedactionBox(x=10, y=10, w=50, h=30, entity_type="SSN", confidence=0.9, source="regex", page=0)
        merged = engine.merge_boxes([box])
        assert len(merged) == 1

    def test_merged_box_encompasses_all_inputs(self):
        """The merged box must cover all constituent box regions."""
        engine = _import_engine()

        boxes = [
            RedactionBox(x=10, y=10, w=50, h=30, entity_type="SSN", confidence=0.9, source="regex", page=0),
            RedactionBox(x=40, y=5,  w=60, h=50, entity_type="SSN", confidence=0.8, source="regex", page=0),
        ]
        merged = engine.merge_boxes(boxes)
        assert len(merged) == 1
        m = merged[0]
        # Left edge must be at most min(10, 40) = 10
        assert m.x <= 10
        # Top edge must be at most min(10, 5) = 5
        assert m.y <= 5
        # Right edge must be at least max(10+50, 40+60) = 100
        assert m.x + m.w >= 100
        # Bottom edge must be at least max(10+30, 5+50) = 55
        assert m.y + m.h >= 55


# ---------------------------------------------------------------------------
# e) test_pdf_redaction_removes_text
# ---------------------------------------------------------------------------


class TestPdfRedactionRemovesText:
    """Prove that PDF redaction removes the underlying text data, not just paints over it."""

    SECRET = "SECRET_SSN_123-45-6789"

    def _make_pdf_with_text(self) -> bytes:
        """Create an in-memory PDF containing SECRET_SSN text, return bytes."""
        fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)  # A4
        page.insert_text(
            (72, 100),
            self.SECRET,
            fontsize=12,
        )
        buf = io.BytesIO()
        doc.save(buf)
        doc.close()
        return buf.getvalue()

    def test_pdf_redaction_removes_text(self):
        fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")
        engine = _import_engine()

        pdf_bytes = self._make_pdf_with_text()

        # Verify the text is actually in the PDF before redaction
        pre_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pre_text = pre_doc[0].get_text()
        pre_doc.close()
        assert self.SECRET in pre_text, (
            "Pre-condition failed: SECRET text not found in original PDF. "
            "Test setup may be broken."
        )

        # Now redact: cover the text area with a box
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[0]

        # The text is inserted at y=100 with font size 12 — use a generous box
        box = RedactionBox(
            x=60, y=90, w=400, h=30,
            entity_type="SSN",
            confidence=1.0,
            source="test",
            page=0,
        )
        engine.redact_pdf_page(page, [box], fill_color=(0, 0, 0))

        # Save to BytesIO
        out_buf = io.BytesIO()
        doc.save(out_buf)
        doc.close()

        # Re-open and extract text
        out_buf.seek(0)
        post_doc = fitz.open(stream=out_buf.read(), filetype="pdf")
        post_text = post_doc[0].get_text()
        post_doc.close()

        assert self.SECRET not in post_text, (
            f"SECRET text still recoverable after PDF redaction! "
            f"Extracted: {post_text!r}. "
            "This is a critical safety failure — apply_redactions() must be called "
            "to remove text data from the PDF content stream, not just draw over it."
        )

    def test_pdf_non_redacted_text_survives(self):
        """Text outside the redaction box must remain readable."""
        fitz = pytest.importorskip("fitz", reason="PyMuPDF not installed")
        engine = _import_engine()

        doc = fitz.open()
        page = doc.new_page(width=595, height=842)
        page.insert_text((72, 100), self.SECRET, fontsize=12)
        page.insert_text((72, 400), "SAFE_TEXT_PRESERVED", fontsize=12)

        # Only redact a box covering the SECRET text (y=90..120), not y=400
        box = RedactionBox(
            x=60, y=90, w=400, h=30,
            entity_type="SSN", confidence=1.0, source="test", page=0,
        )
        engine.redact_pdf_page(page, [box], fill_color=(0, 0, 0))

        out_buf = io.BytesIO()
        doc.save(out_buf)
        doc.close()

        out_buf.seek(0)
        post_doc = fitz.open(stream=out_buf.read(), filetype="pdf")
        post_text = post_doc[0].get_text()
        post_doc.close()

        assert self.SECRET not in post_text, "SECRET text was not redacted"
        assert "SAFE_TEXT_PRESERVED" in post_text, (
            "Text outside the redaction box was inadvertently removed"
        )
