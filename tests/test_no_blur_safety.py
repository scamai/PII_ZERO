"""Standalone pixel-perfect safety assertion.

Run with:
    python -m pytest tests/test_no_blur_safety.py -v

This file has ONE purpose: prove that the redaction engine NEVER produces
blur-like (partial-coverage) pixels inside any redaction box.

"Blur-like" is defined as:
    A pixel inside the box that is NOT exactly fill_color AND is lighter
    than the fill_color on at least one channel — which would indicate
    partial coverage (anti-aliasing, feathering, transparency blending,
    Gaussian blur artefact, etc.).

If this test fails, the tool has a critical security regression.
"""

from __future__ import annotations

import numpy as np
import pytest

from pii_redact.models import RedactionBox

pytestmark = pytest.mark.fast


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_FILL = (0, 0, 0)   # solid black, BGR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _import_redact_image():
    """Import redact_image or skip the test if dependencies not installed."""
    try:
        from pii_redact.redact.engine import redact_image
        return redact_image
    except ImportError as exc:
        pytest.skip(f"redact_image not importable: {exc}")


def _make_image(height: int, width: int, color: tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """Create a solid-color BGR uint8 image."""
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:, :] = color
    return img


def _assert_pixel_perfect_fill(
    result: np.ndarray,
    box: RedactionBox,
    fill_color: tuple[int, int, int],
    label: str = "",
) -> None:
    """Assert every pixel inside *box* equals exactly *fill_color*.

    Raises AssertionError with a detailed report on the first failure.
    """
    x1 = max(0, box.x)
    y1 = max(0, box.y)
    x2 = min(result.shape[1], box.x + box.w)
    y2 = min(result.shape[0], box.y + box.h)

    if x2 <= x1 or y2 <= y1:
        return  # degenerate / out-of-bounds box — nothing to check

    roi = result[y1:y2, x1:x2]
    fill_arr = np.array(fill_color, dtype=np.uint8)

    # Pixels that are NOT exactly fill_color
    mismatch_mask = np.any(roi != fill_arr, axis=2)
    num_mismatch = int(mismatch_mask.sum())

    if num_mismatch == 0:
        return  # all good

    # Gather details for the error message
    bad_pixels = roi[mismatch_mask]
    sample = bad_pixels[:5].tolist()

    # Blur-like vs. out-of-range classification
    blur_like = []
    for px in bad_pixels:
        # A pixel is "blur-like" if it's lighter than fill on any channel
        # (for black fill: any nonzero channel indicates partial coverage)
        if any(int(px[c]) > int(fill_color[c]) for c in range(3)):
            blur_like.append(px.tolist())

    prefix = f"[{label}] " if label else ""
    raise AssertionError(
        f"{prefix}SAFETY FAILURE: {num_mismatch} pixel(s) inside the redaction box "
        f"are NOT exactly fill_color {fill_color}.\n"
        f"  Box: x={box.x} y={box.y} w={box.w} h={box.h}\n"
        f"  Sample bad pixel BGR values: {sample}\n"
        f"  Blur-like pixels (lighter than fill): {len(blur_like)}\n"
        "  This indicates partial coverage — blur, anti-aliasing, or transparency blending.\n"
        "  Only 100% opaque solid fill is acceptable for PII redaction."
    )


# ---------------------------------------------------------------------------
# Parametrized box configurations
# ---------------------------------------------------------------------------


_BOX_CONFIGS = [
    # (label, image_size, box_params)
    ("centre_box_large",     (300, 300), dict(x=100, y=100, w=100, h=100)),
    ("centre_box_small",     (300, 300), dict(x=145, y=145, w=10,  h=10)),
    ("top_left_corner",      (300, 300), dict(x=0,   y=0,   w=50,  h=50)),
    ("bottom_right_corner",  (300, 300), dict(x=250, y=250, w=50,  h=50)),
    ("thin_horizontal_strip",(300, 300), dict(x=50,  y=150, w=200, h=2)),
    ("thin_vertical_strip",  (300, 300), dict(x=150, y=50,  w=2,   h=200)),
    ("1x1_pixel",            (300, 300), dict(x=150, y=150, w=1,   h=1)),
    ("full_image",           (100, 100), dict(x=0,   y=0,   w=100, h=100)),
    ("multi_channel_nonblack",(200, 200), dict(x=50, y=50,  w=80,  h=80)),
    ("very_wide",            (100, 500), dict(x=0,   y=0,   w=500, h=100)),
    ("very_tall",            (500, 100), dict(x=0,   y=0,   w=100, h=500)),
]


@pytest.mark.parametrize("label,img_size,box_params", _BOX_CONFIGS)
def test_zero_blur_pixels_parametrized(label: str, img_size: tuple, box_params: dict):
    """Given ANY RedactionBox, the engine must produce ZERO blur-like pixels inside it."""
    redact_image = _import_redact_image()

    h, w = img_size
    img = _make_image(h, w, color=(255, 255, 255))  # white canvas

    fill = DEFAULT_FILL if label != "multi_channel_nonblack" else (0, 128, 255)

    box = RedactionBox(
        entity_type="SSN",
        confidence=1.0,
        source="test",
        **box_params,
    )

    result = redact_image(img, [box], fill_color=fill)

    _assert_pixel_perfect_fill(result, box, fill, label=label)


def test_zero_blur_pixels_multiple_boxes():
    """Multiple boxes on the same image must each be pixel-perfect."""
    redact_image = _import_redact_image()

    img = _make_image(400, 400)
    boxes = [
        RedactionBox(x=10,  y=10,  w=80,  h=40,  entity_type="SSN",    confidence=0.9, source="regex"),
        RedactionBox(x=150, y=100, w=120, h=60,  entity_type="PERSON", confidence=0.8, source="spacy"),
        RedactionBox(x=300, y=300, w=50,  h=50,  entity_type="FACE",   confidence=0.95, source="yolov8"),
    ]
    fill = DEFAULT_FILL

    result = redact_image(img, boxes, fill_color=fill)

    for box in boxes:
        _assert_pixel_perfect_fill(result, box, fill, label=f"entity={box.entity_type}")


def test_original_image_not_mutated():
    """redact_image must return a new array; the original must be unchanged."""
    redact_image = _import_redact_image()

    img = _make_image(200, 200, color=(128, 64, 32))
    original_copy = img.copy()

    box = RedactionBox(x=50, y=50, w=100, h=100, entity_type="SSN", confidence=0.9, source="regex")
    _ = redact_image(img, [box], fill_color=DEFAULT_FILL)

    assert np.array_equal(img, original_copy), (
        "redact_image modified the input array in-place. "
        "It must return a new array and leave the original untouched."
    )


def test_pixels_outside_box_unchanged():
    """Pixels outside all boxes must be exactly the original colour."""
    redact_image = _import_redact_image()

    original_color = (200, 150, 100)
    img = _make_image(300, 300, color=original_color)

    box = RedactionBox(x=100, y=100, w=100, h=100, entity_type="SSN", confidence=0.9, source="regex")
    result = redact_image(img, [box], fill_color=DEFAULT_FILL)

    # Check all four regions outside the box
    top    = result[:100, :]
    bottom = result[200:, :]
    left   = result[100:200, :100]
    right  = result[100:200, 200:]

    for region_name, region in [("top", top), ("bottom", bottom), ("left", left), ("right", right)]:
        assert np.all(region == np.array(original_color, dtype=np.uint8)), (
            f"Pixels in region '{region_name}' (outside box) were modified. "
            "Only pixels inside the redaction box should change."
        )


def test_degenerate_box_does_not_crash():
    """Zero-size or out-of-bounds boxes must not raise — just be skipped."""
    redact_image = _import_redact_image()

    img = _make_image(100, 100)
    boxes = [
        RedactionBox(x=0,   y=0,   w=0,   h=100, entity_type="SSN", confidence=0.9, source="regex"),
        RedactionBox(x=0,   y=0,   w=100, h=0,   entity_type="SSN", confidence=0.9, source="regex"),
        RedactionBox(x=200, y=200, w=50,  h=50,  entity_type="SSN", confidence=0.9, source="regex"),  # OOB
        RedactionBox(x=-10, y=-10, w=50,  h=50,  entity_type="SSN", confidence=0.9, source="regex"),  # neg coords
    ]
    # Must not raise
    result = redact_image(img, boxes, fill_color=DEFAULT_FILL)
    assert result is not None
    assert result.shape == img.shape


def test_fill_color_variety():
    """Different fill colors must all apply with pixel-perfect accuracy."""
    redact_image = _import_redact_image()

    colors = [
        (0, 0, 0),       # black
        (255, 255, 255), # white
        (0, 0, 255),     # blue
        (255, 0, 0),     # red
        (0, 255, 0),     # green
        (127, 127, 127), # grey
    ]

    for fill in colors:
        # Use an image colour different from fill so we can tell them apart
        bg = tuple(255 - c for c in fill) if fill != (127, 127, 127) else (0, 0, 0)
        img = _make_image(100, 100, color=bg)
        box = RedactionBox(x=20, y=20, w=60, h=60, entity_type="SSN", confidence=0.9, source="regex")
        result = redact_image(img, [box], fill_color=fill)
        _assert_pixel_perfect_fill(result, box, fill, label=f"fill={fill}")
