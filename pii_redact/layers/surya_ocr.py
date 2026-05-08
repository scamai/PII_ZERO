"""Surya OCR layer — run full-page text detection + recognition on receipt images.

This module wraps surya-ocr (v0.17+) and exposes a single function:

    ocr_image(img_array) -> list[dict]

Each returned dict has the shape::

    {
        "text": str,
        "bbox": [x_min, y_min, x_max, y_max],   # pixel coords in input image
        "confidence": float,
        "source": "surya_ocr",
    }

The module is intentionally lazy: predictors are only instantiated on first
call so that importing this file does NOT trigger model downloads or GPU init.
Heavy callers should set CUDA_VISIBLE_DEVICES="" before importing when they
want CPU-only inference.

Usage::

    import numpy as np
    from PIL import Image
    from pii_redact.layers.surya_ocr import ocr_image

    pil_img = Image.open("receipt.png").convert("RGB")
    img_array = np.array(pil_img)
    lines = ocr_image(img_array)
    full_text = " ".join(ln["text"] for ln in lines if ln["text"].strip())
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy singleton predictors
# ---------------------------------------------------------------------------

_det_predictor: Any = None
_rec_predictor: Any = None


def _get_predictors():
    """Return (det_predictor, rec_predictor), loading on first call.

    Surya 0.17+ requires a ``FoundationPredictor`` to be passed into
    ``RecognitionPredictor`` at construction time.
    """
    global _det_predictor, _rec_predictor
    if _det_predictor is None or _rec_predictor is None:
        logger.info("[surya_ocr] Loading DetectionPredictor + RecognitionPredictor …")
        from surya.detection import DetectionPredictor
        from surya.recognition import FoundationPredictor, RecognitionPredictor

        _det_predictor = DetectionPredictor()
        _foundation = FoundationPredictor()
        _rec_predictor = RecognitionPredictor(_foundation)
        logger.info("[surya_ocr] Predictors ready.")
    return _det_predictor, _rec_predictor


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ocr_image(
    img: "np.ndarray | Any",  # numpy HxWx3 uint8 or PIL Image
    languages: list[str] | None = None,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Run Surya detection + recognition on a single image.

    Parameters
    ----------
    img:
        Either a numpy array (H×W×3, uint8, RGB) or a PIL Image.
    languages:
        Surya language hints, e.g. ``["en"]``.  Defaults to ``["en"]``.
    min_confidence:
        Drop lines whose recognition confidence is below this threshold.

    Returns
    -------
    list[dict]
        Each dict: ``{"text", "bbox", "confidence", "source"}``.
        ``bbox`` is ``[x_min, y_min, x_max, y_max]`` in pixel coordinates.
    """
    from PIL import Image as PILImage

    languages = languages or ["en"]

    # Normalise input to PIL Image (Surya expects PIL)
    if isinstance(img, np.ndarray):
        pil_img = PILImage.fromarray(img.astype(np.uint8), mode="RGB")
    elif hasattr(img, "convert"):
        pil_img = img.convert("RGB")
    else:
        raise TypeError(f"ocr_image: unsupported img type {type(img)}")

    det_predictor, rec_predictor = _get_predictors()

    # Step 1: detect text regions
    try:
        det_results = det_predictor([pil_img])
        det_result = det_results[0]
    except Exception as exc:
        logger.warning("[surya_ocr] Detection failed: %s", exc)
        return []

    # Build bbox list for recognition (list of [x1,y1,x2,y2])
    bboxes_for_img: list[list[int]] = []
    for pb in det_result.bboxes:
        poly = pb.polygon  # [[x,y], [x,y], [x,y], [x,y]]
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        bboxes_for_img.append([int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))])

    if not bboxes_for_img:
        logger.debug("[surya_ocr] No text regions detected.")
        return []

    # Step 2: recognise text in each detected region
    try:
        ocr_results = rec_predictor(
            images=[pil_img],
            det_predictor=None,  # we supply bboxes directly
            bboxes=[bboxes_for_img],
        )
        ocr_result = ocr_results[0]
    except Exception as exc:
        logger.warning("[surya_ocr] Recognition failed: %s", exc)
        return []

    lines: list[dict] = []
    for tl in ocr_result.text_lines:
        conf = float(tl.confidence or 0.0)
        if conf < min_confidence:
            continue
        poly = tl.polygon
        xs = [pt[0] for pt in poly]
        ys = [pt[1] for pt in poly]
        lines.append(
            {
                "text": tl.text,
                "bbox": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "confidence": round(conf, 4),
                "source": "surya_ocr",
            }
        )

    return lines


def ocr_pil(
    pil_img: Any,
    languages: list[str] | None = None,
    min_confidence: float = 0.0,
) -> list[dict]:
    """Convenience wrapper accepting a PIL Image directly."""
    import numpy as np

    return ocr_image(np.array(pil_img.convert("RGB")), languages=languages,
                     min_confidence=min_confidence)
