"""Surya OCR wrapper with lazy model loading.

Surya API (v0.17+):
  - DetectionPredictor: detects text regions, takes List[PIL.Image] -> List[TextDetectionResult]
  - FoundationPredictor + RecognitionPredictor: recognizes text, takes images + det_predictor
      -> List[OCRResult]

Models are loaded lazily on the first call to ocr_image() so that importing this module
does not trigger any heavy downloads or GPU initialization.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_det_predictor = None
_rec_predictor = None


def _load_models() -> None:
    """Load Surya detection and recognition predictors (once)."""
    global _det_predictor, _rec_predictor
    if _det_predictor is not None:
        return

    try:
        from surya.detection import DetectionPredictor
        from surya.recognition import FoundationPredictor, RecognitionPredictor

        logger.info("Loading Surya OCR models (first-time only)…")
        _det_predictor = DetectionPredictor(device="cpu")
        foundation = FoundationPredictor(device="cpu")
        _rec_predictor = RecognitionPredictor(foundation_predictor=foundation)
        logger.info("Surya OCR models loaded.")
    except Exception as exc:
        logger.error("Failed to load Surya OCR models: %s", exc)
        raise


def ocr_image(image: np.ndarray, langs: Optional[list[str]] = None) -> list[dict]:
    """Run OCR on a numpy BGR image array.

    Parameters
    ----------
    image:
        BGR uint8 numpy array (as returned by cv2.imread or cv2.cvtColor).
    langs:
        Reserved for future use. Surya auto-detects language.

    Returns
    -------
    list of dicts with keys:
        - "text": str
        - "bbox": [x1, y1, x2, y2]
        - "confidence": float
    """
    try:
        from PIL import Image

        _load_models()

        # Convert BGR numpy array -> RGB PIL Image
        if image.ndim == 2:
            # Grayscale
            pil_image = Image.fromarray(image, mode="L").convert("RGB")
        else:
            # BGR -> RGB
            rgb = image[:, :, ::-1]
            pil_image = Image.fromarray(rgb.astype(np.uint8), mode="RGB")

        results = _rec_predictor(
            images=[pil_image],
            det_predictor=_det_predictor,
            sort_lines=True,
            math_mode=False,
        )

        if not results:
            return []

        ocr_result = results[0]  # OCRResult for our single image
        output: list[dict] = []

        for line in ocr_result.text_lines:
            text = (line.text or "").strip()
            if not text:
                continue

            # polygon is List[List[float]], each inner list is [x, y]
            poly = line.polygon  # list of [x, y] pairs
            if poly and len(poly) >= 2:
                xs = [pt[0] for pt in poly]
                ys = [pt[1] for pt in poly]
                x1, y1 = int(min(xs)), int(min(ys))
                x2, y2 = int(max(xs)), int(max(ys))
            else:
                x1, y1, x2, y2 = 0, 0, 0, 0

            confidence = float(line.confidence) if line.confidence is not None else 0.0

            output.append({
                "text": text,
                "bbox": [x1, y1, x2, y2],
                "confidence": confidence,
            })

        return output

    except Exception as exc:
        logger.warning("surya_ocr.ocr_image failed: %s", exc)
        return []
