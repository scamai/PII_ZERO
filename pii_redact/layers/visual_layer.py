"""Visual detection layer — face, license plate, handwriting, QR/barcode.

All four detection passes are lazy-loaded and gracefully degrade when
optional dependencies or model weight files are unavailable.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from pii_redact.config import settings
from pii_redact.models import DocumentType, RedactionBox

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pad_box(
    x: int, y: int, w: int, h: int, padding: int, img_h: int, img_w: int
) -> tuple[int, int, int, int]:
    """Expand a bounding box by *padding* pixels, clamped to image bounds."""
    x2 = min(x + w + padding, img_w)
    y2 = min(y + h + padding, img_h)
    x1 = max(x - padding, 0)
    y1 = max(y - padding, 0)
    return x1, y1, x2 - x1, y2 - y1


# ---------------------------------------------------------------------------
# Detection passes
# ---------------------------------------------------------------------------


def _detect_faces(image: np.ndarray) -> list[RedactionBox]:
    """Pass A: face detection via deface, falling back to OpenCV Haar cascade."""
    img_h, img_w = image.shape[:2]
    padding: int = settings.redaction.fill_padding_px
    boxes: list[RedactionBox] = []

    # --- Try deface first ---
    try:
        from deface.deface import get_anonymized_frame  # type: ignore[import]

        # get_anonymized_frame returns the anonymised image; we need bounding boxes.
        # deface exposes its centerface detector as a separate import.
        from deface.centerface import CenterFace  # type: ignore[import]

        _cf = CenterFace()
        dets, _ = _cf(image, threshold=0.5)  # returns (N,5) array: x1,y1,x2,y2,score
        for det in dets:
            x1, y1, x2, y2, score = det
            x, y, w, h = int(x1), int(y1), int(x2 - x1), int(y2 - y1)
            x, y, w, h = _pad_box(x, y, w, h, padding, img_h, img_w)
            boxes.append(
                RedactionBox(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    entity_type="FACE",
                    confidence=float(score),
                    source="deface",
                )
            )
        logger.debug("deface detected %d face(s)", len(boxes))
        return boxes
    except Exception as exc:  # ImportError, model download error, etc.
        logger.debug("deface unavailable (%s); falling back to Haar cascade", exc)

    # --- Fallback: OpenCV Haar cascade (ships with opencv-python) ---
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )
        if len(detections) == 0:
            return []
        for x, y, w, h in detections:
            x, y, w, h = _pad_box(int(x), int(y), int(w), int(h), padding, img_h, img_w)
            boxes.append(
                RedactionBox(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    entity_type="FACE",
                    confidence=0.80,  # Haar cascade does not output probabilities
                    source="opencv_cascade",
                )
            )
        logger.debug("Haar cascade detected %d face(s)", len(boxes))
    except Exception as exc:
        logger.warning("Face detection failed entirely: %s", exc)

    return boxes


def _detect_license_plates(image: np.ndarray) -> list[RedactionBox]:
    """Pass B: license plate / logo detection via YOLOv8 (ultralytics)."""
    weights_path = Path(settings.models.yolov8_weights)
    if not weights_path.exists():
        logger.debug("YOLOv8 weights not found at %s; skipping plate detection", weights_path)
        return []

    try:
        from ultralytics import YOLO  # type: ignore[import]
    except ImportError:
        logger.debug("ultralytics not installed; skipping plate detection")
        return []

    img_h, img_w = image.shape[:2]
    padding: int = settings.redaction.fill_padding_px
    min_conf: float = settings.redaction.min_confidence
    boxes: list[RedactionBox] = []

    try:
        model = YOLO(str(weights_path))
        results = model(image, verbose=False)

        plate_keywords = {"license_plate", "plate", "numberplate", "licence_plate"}

        for result in results:
            if result.boxes is None:
                continue
            class_names: dict[int, str] = result.names  # {id: name}
            for box in result.boxes:
                cls_id = int(box.cls[0])
                cls_name = class_names.get(cls_id, "").lower()
                conf = float(box.conf[0])

                # Accept plate classes at any confidence above min_confidence;
                # accept other classes only if confidence > 0.7
                is_plate_class = any(kw in cls_name for kw in plate_keywords)
                if is_plate_class:
                    if conf < min_conf:
                        continue
                    entity_type = "LICENSE_PLATE"
                else:
                    if conf < 0.7:
                        continue
                    entity_type = "LICENSE_PLATE"  # redact unclassified high-conf objects too

                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
                x, y, w, h = _pad_box(x1, y1, x2 - x1, y2 - y1, padding, img_h, img_w)
                boxes.append(
                    RedactionBox(
                        x=x,
                        y=y,
                        w=w,
                        h=h,
                        entity_type=entity_type,
                        confidence=conf,
                        source="yolov8",
                    )
                )
    except Exception as exc:
        logger.warning("YOLOv8 plate detection error: %s", exc)

    logger.debug("YOLOv8 detected %d plate/object region(s)", len(boxes))
    return boxes


def _detect_handwriting(image: np.ndarray) -> list[RedactionBox]:
    """Pass C: handwriting / ink region detection using adaptive thresholding (pure OpenCV).

    Only runs when document_type is SCAN_FORM — caller is responsible for this check.
    """
    img_h, img_w = image.shape[:2]
    padding: int = settings.redaction.fill_padding_px
    min_area: int = settings.redaction.handwriting_min_area
    boxes: list[RedactionBox] = []

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        thresh = cv2.adaptiveThreshold(
            gray,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=8,
        )

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (12, 4))
        dilated = cv2.dilate(thresh, kernel, iterations=2)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            x, y, w, h = _pad_box(x, y, w, h, padding, img_h, img_w)
            boxes.append(
                RedactionBox(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    entity_type="HANDWRITING",
                    confidence=0.75,  # rule-based; no probabilistic score
                    source="opencv_ink",
                )
            )
    except Exception as exc:
        logger.warning("Handwriting detection error: %s", exc)

    logger.debug("opencv_ink detected %d ink region(s)", len(boxes))
    return boxes


def _detect_qr_barcodes(image: np.ndarray) -> list[RedactionBox]:
    """Pass D: QR code and barcode detection via pyzbar."""
    try:
        from pyzbar.pyzbar import decode as pyzbar_decode  # type: ignore[import]
    except ImportError:
        logger.debug("pyzbar not installed; skipping QR/barcode detection")
        return []

    img_h, img_w = image.shape[:2]
    padding: int = settings.redaction.fill_padding_px
    boxes: list[RedactionBox] = []

    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        decoded_objects = pyzbar_decode(gray)

        for obj in decoded_objects:
            rect = obj.rect  # pyzbar.Decoded: .rect is a Rect(left, top, width, height)
            x, y, w, h = rect.left, rect.top, rect.width, rect.height
            x, y, w, h = _pad_box(x, y, w, h, padding, img_h, img_w)

            obj_type = obj.type  # 'QRCODE', 'CODE128', 'EAN13', etc.
            entity_type = "QR_CODE" if "QR" in obj_type.upper() else "BARCODE"

            boxes.append(
                RedactionBox(
                    x=x,
                    y=y,
                    w=w,
                    h=h,
                    entity_type=entity_type,
                    confidence=1.0,  # pyzbar only returns fully decoded symbols
                    source="pyzbar",
                )
            )
    except Exception as exc:
        logger.warning("pyzbar QR/barcode detection error: %s", exc)

    logger.debug("pyzbar detected %d QR/barcode(s)", len(boxes))
    return boxes


# ---------------------------------------------------------------------------
# VisualLayer
# ---------------------------------------------------------------------------


class VisualLayer:
    """Run all four visual detection passes and return merged RedactionBox list."""

    def process(
        self,
        image: np.ndarray,
        document_type: DocumentType,
    ) -> list[RedactionBox]:
        """Detect PII-bearing visual regions in *image*.

        Parameters
        ----------
        image:
            BGR image as returned by cv2.imread or equivalent.
        document_type:
            Controls which detection passes are active (e.g., handwriting
            is only run for SCAN_FORM).

        Returns
        -------
        list[RedactionBox]
            All detected regions, after confidence filtering.
        """
        min_conf: float = settings.redaction.min_confidence
        all_boxes: list[RedactionBox] = []

        # Pass A — faces (always run)
        all_boxes.extend(_detect_faces(image))

        # Pass B — license plates / objects (always run)
        all_boxes.extend(_detect_license_plates(image))

        # Pass C — handwriting ink regions (SCAN_FORM only)
        if document_type is DocumentType.SCAN_FORM:
            all_boxes.extend(_detect_handwriting(image))

        # Pass D — QR codes and barcodes (always run)
        all_boxes.extend(_detect_qr_barcodes(image))

        # Filter by minimum confidence threshold
        filtered = [b for b in all_boxes if b.confidence >= min_conf]

        logger.info(
            "VisualLayer: %d raw box(es), %d after confidence filter (>= %.2f)",
            len(all_boxes),
            len(filtered),
            min_conf,
        )
        return filtered
