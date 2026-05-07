"""Document type classifier — routes each file to the correct sub-pipeline.

The classifier NEVER raises: every code path returns a DocumentType, even on errors.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pii_redact.models import DocumentType

logger = logging.getLogger(__name__)

# Extensions treated as raster images
_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}

# Keywords that identify structured insurance form PDFs
_STRUCTURED_KEYWORDS = {"CMS-1500", "ACORD", "CLAIM FORM", "CMS 1500"}


def _classify_pdf(file_path: Path) -> DocumentType:
    """Classify a PDF as PDF_STRUCTURED or SCAN_FORM."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(file_path))
        has_text_layer = False
        text_content = ""

        try:
            for page in doc:
                page_text = page.get_text()
                if len(page_text) > 100:
                    has_text_layer = True
                    text_content += page_text
        finally:
            doc.close()

        if has_text_layer:
            # Check for structured insurance form keywords
            upper_text = text_content.upper()
            for keyword in _STRUCTURED_KEYWORDS:
                if keyword.upper() in upper_text:
                    logger.debug("PDF classified as PDF_STRUCTURED (keyword: %r)", keyword)
                    return DocumentType.PDF_STRUCTURED
            # Has text layer but no insurance keywords — still structured
            logger.debug("PDF has text layer; classified as PDF_STRUCTURED (no keyword match)")
            return DocumentType.PDF_STRUCTURED
        else:
            logger.debug("PDF has no text layer; classified as SCAN_FORM")
            return DocumentType.SCAN_FORM

    except Exception as exc:
        logger.warning("PDF classification error for %s: %s", file_path, exc)
        return DocumentType.UNKNOWN


def _count_faces(image) -> int:  # image: np.ndarray
    """Count faces using deface or OpenCV Haar cascade fallback."""
    face_count = 0

    # Try deface / CenterFace first
    try:
        from deface.centerface import CenterFace  # type: ignore[import]

        cf = CenterFace()
        dets, _ = cf(image, threshold=0.5)
        face_count = len(dets) if dets is not None else 0
        logger.debug("deface found %d face(s)", face_count)
        return face_count
    except Exception:
        pass

    # Fallback: OpenCV Haar cascade
    try:
        import cv2

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(cascade_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_count = len(detections)
        logger.debug("Haar cascade found %d face(s)", face_count)
    except Exception as exc:
        logger.debug("Face counting failed: %s", exc)

    return face_count


def _classify_image(file_path: Path) -> DocumentType:
    """Classify a raster image as ID_DOC, PHOTO, or SCAN_FORM."""
    try:
        import cv2

        image = cv2.imread(str(file_path))
        if image is None:
            logger.warning("cv2.imread returned None for %s", file_path)
            return DocumentType.UNKNOWN

        img_h, img_w = image.shape[:2]

        # Face detection — needed to distinguish ID_DOC from PHOTO from SCAN
        face_count = _count_faces(image)

        if face_count > 0:
            # Portrait aspect ratio → ID document (driver's license, passport, insurance card)
            if img_h > img_w * 1.2:
                logger.debug("Image with face(s), portrait aspect → ID_DOC")
                return DocumentType.ID_DOC
            # Landscape with face(s) → could be scene photo
            logger.debug("Image with face(s), non-portrait → PHOTO")
            return DocumentType.PHOTO

        # No faces — use aspect ratio as heuristic
        if img_w > img_h * 1.5:
            logger.debug("Landscape image, no faces → PHOTO")
            return DocumentType.PHOTO

        # Portrait-shaped with no faces: likely a form / document scan
        logger.debug("Image with no faces → SCAN_FORM")
        return DocumentType.SCAN_FORM

    except Exception as exc:
        logger.warning("Image classification error for %s: %s", file_path, exc)
        return DocumentType.UNKNOWN


def classify_document(file_path: Path) -> DocumentType:
    """Classify *file_path* and return the appropriate DocumentType.

    This function never raises: any exception results in DocumentType.UNKNOWN.

    Parameters
    ----------
    file_path:
        Path to the document to classify.

    Returns
    -------
    DocumentType
        One of PDF_STRUCTURED, SCAN_FORM, PHOTO, ID_DOC, MEDICAL, UNKNOWN.
        MEDICAL is not auto-detected here — callers that know the document is
        medical should override after classification.
    """
    try:
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return _classify_pdf(path)

        if suffix in _IMAGE_SUFFIXES:
            return _classify_image(path)

        logger.info("Unknown file extension %r for %s → UNKNOWN", suffix, file_path)
        return DocumentType.UNKNOWN

    except Exception as exc:
        logger.error("Unexpected error in classify_document(%s): %s", file_path, exc)
        return DocumentType.UNKNOWN
