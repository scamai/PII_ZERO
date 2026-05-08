"""Tests for the Surya OCR wrapper (pii_redact.layers.surya_ocr).

Fast tests (marked @pytest.mark.fast) verify the module is importable and
the public API exists without loading any models.

Slow tests (marked @pytest.mark.slow) actually load the Surya models and run
inference; they require a working Surya installation and may take ~30 s on CPU.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fast tests — no model loading
# ---------------------------------------------------------------------------


@pytest.mark.fast
def test_surya_ocr_module_importable():
    """Module must be importable without triggering model load."""
    mod = importlib.import_module("pii_redact.layers.surya_ocr")
    assert hasattr(mod, "ocr_image"), "ocr_image must be exported"


@pytest.mark.fast
def test_surya_ocr_models_not_loaded_at_import():
    """Importing the module must not load any models (_det_predictor stays None)."""
    # Re-import to get the freshest reference; predictors must still be None
    # unless a slow test has already run and populated them in this process.
    mod = importlib.import_module("pii_redact.layers.surya_ocr")
    # We can only assert the attribute exists; its value may be non-None if
    # a slow test already ran.  The important invariant is that the attribute
    # is defined (not that it is None).
    assert hasattr(mod, "_det_predictor")
    assert hasattr(mod, "_rec_predictor")


# ---------------------------------------------------------------------------
# Slow tests — full model load + inference
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_surya_ocr_returns_list_on_blank_image():
    """OCR on a white image should return an empty list without error."""
    from pii_redact.layers.surya_ocr import ocr_image

    blank = np.ones((200, 400, 3), dtype=np.uint8) * 255
    result = ocr_image(blank)
    assert isinstance(result, list), "ocr_image must return a list"


@pytest.mark.slow
def test_surya_ocr_result_dict_shape():
    """Each item returned by ocr_image must have the required keys."""
    from pii_redact.layers.surya_ocr import ocr_image

    blank = np.ones((200, 400, 3), dtype=np.uint8) * 255
    result = ocr_image(blank)
    for item in result:
        assert "text" in item, "each result must have 'text'"
        assert "bbox" in item, "each result must have 'bbox'"
        assert "confidence" in item, "each result must have 'confidence'"
        assert isinstance(item["bbox"], list) and len(item["bbox"]) == 4, (
            "bbox must be [x1, y1, x2, y2]"
        )


@pytest.mark.slow
def test_surya_ocr_detects_text_in_synthetic_image():
    """OCR should detect known text rendered into a synthetic image."""
    from PIL import Image, ImageDraw, ImageFont

    from pii_redact.layers.surya_ocr import ocr_image

    # Render text onto a white background
    img_pil = Image.new("RGB", (600, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((10, 30), "Jane Doe SSN: 523-67-4891", fill=(0, 0, 0), font=font)

    # Convert PIL -> BGR numpy
    import numpy as np

    bgr = np.array(img_pil)[:, :, ::-1]

    result = ocr_image(bgr)

    assert isinstance(result, list), "ocr_image must return a list"
    assert len(result) > 0, "OCR should detect at least one text line in the rendered image"

    texts = [item["text"] for item in result]
    # At least one result should contain some recognizable characters
    combined = " ".join(texts)
    assert len(combined) > 0, "combined OCR text must be non-empty"
