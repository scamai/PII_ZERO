"""Metadata stripping layer.

Removes EXIF, XMP, and PDF document metadata from the file on disk.
Preferred tool is ExifTool (system binary) when available; falls back to
piexif (JPEG/PNG) or PyMuPDF / fitz (PDF).
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

from pii_redact.models import MetadataStripResult

logger = logging.getLogger(__name__)

# GPS-related EXIF tag IDs that flag location data
_GPS_IFD_TAG = 0x8825  # GPSInfo pointer in IFD0
_GPS_LATITUDE_TAG = 2
_GPS_LONGITUDE_TAG = 4

# EXIF field names reported in MetadataStripResult.stripped_fields
_COMMON_EXIF_FIELDS = [
    "Make",
    "Model",
    "DateTime",
    "DateTimeOriginal",
    "DateTimeDigitized",
    "GPSInfo",
    "Software",
    "Artist",
    "Copyright",
    "ImageDescription",
    "UserComment",
    "XPAuthor",
    "XPComment",
    "XPKeywords",
]

# PDF metadata keys cleared by fitz
_PDF_META_KEYS = [
    "title",
    "author",
    "subject",
    "keywords",
    "creator",
    "producer",
    "creationDate",
    "modDate",
    "trapped",
    "encryption",
]


# ---------------------------------------------------------------------------
# GPS check (piexif-based)
# ---------------------------------------------------------------------------


def _check_gps_piexif(file_path: Path) -> bool:
    """Return True if the file contains GPS EXIF data."""
    try:
        import piexif  # type: ignore[import]

        exif_dict = piexif.load(str(file_path))
        gps_ifd = exif_dict.get("GPS", {})
        if gps_ifd:
            # GPS IFD present and non-empty — location data found
            return True
        # Also check for the GPSInfo pointer tag in IFD0
        ifd0 = exif_dict.get("0th", {})
        return _GPS_IFD_TAG in ifd0
    except Exception:
        return False


# ---------------------------------------------------------------------------
# EXIF stripping helpers
# ---------------------------------------------------------------------------


def _strip_exif_exiftool(file_path: Path) -> list[str]:
    """Strip all metadata with ExifTool (most complete approach)."""
    result = subprocess.run(
        ["exiftool", "-all=", "-overwrite_original", str(file_path)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        logger.warning("exiftool non-zero exit for %s: %s", file_path, result.stderr.strip())
    logger.debug("exiftool output: %s", result.stdout.strip())
    # ExifTool removes all fields; report the known common ones as stripped
    return _COMMON_EXIF_FIELDS[:]


def _strip_exif_piexif(file_path: Path) -> list[str]:
    """Strip EXIF from JPEG/PNG using piexif."""
    try:
        import piexif  # type: ignore[import]

        piexif.remove(str(file_path))
        return _COMMON_EXIF_FIELDS[:]
    except Exception as exc:
        logger.warning("piexif.remove failed for %s: %s", file_path, exc)
        return []


# ---------------------------------------------------------------------------
# PDF metadata stripping
# ---------------------------------------------------------------------------


def _strip_pdf_metadata(file_path: Path) -> list[str]:
    """Clear document metadata from a PDF using PyMuPDF (fitz)."""
    try:
        import fitz  # type: ignore[import]  # PyMuPDF

        doc = fitz.open(str(file_path))
        # Build list of fields that were non-empty before clearing
        current_meta = doc.metadata or {}
        stripped = [k for k, v in current_meta.items() if v]

        doc.set_metadata({})
        doc.save(str(file_path), incremental=True, encryption=fitz.PDF_ENCRYPT_KEEP)
        doc.close()
        return stripped
    except Exception as exc:
        logger.warning("fitz metadata strip failed for %s: %s", file_path, exc)
        return []


# ---------------------------------------------------------------------------
# MetaLayer
# ---------------------------------------------------------------------------


class MetaLayer:
    """Strip file-level metadata from images and PDFs.

    Parameters
    ----------
    qr_codes_found:
        Pass-through count from VisualLayer so the caller can aggregate
        MetadataStripResult without duplicating detection logic here.
    barcodes_found:
        Same as qr_codes_found but for barcodes.
    """

    def process(
        self,
        file_path: Path,
        qr_codes_found: int = 0,
        barcodes_found: int = 0,
    ) -> MetadataStripResult:
        """Strip metadata from *file_path* in-place and return a result record.

        Parameters
        ----------
        file_path:
            Path to the document on disk. Must be writable.
        qr_codes_found:
            Number of QR codes already detected by VisualLayer (pass-through).
        barcodes_found:
            Number of barcodes already detected by VisualLayer (pass-through).

        Returns
        -------
        MetadataStripResult
        """
        result = MetadataStripResult(
            qr_codes_found=qr_codes_found,
            barcodes_found=barcodes_found,
        )

        suffix = file_path.suffix.lower()
        is_image = suffix in {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".webp"}
        is_pdf = suffix == ".pdf"

        if not (is_image or is_pdf):
            logger.debug("MetaLayer: unsupported file type %s, skipping", suffix)
            return result

        # --- GPS check before stripping ---
        if is_image:
            result.exif_gps_found = _check_gps_piexif(file_path)
            if result.exif_gps_found:
                logger.warning("GPS EXIF data found in %s — will be stripped", file_path)

        # --- Metadata stripping ---
        if is_pdf:
            result.stripped_fields = _strip_pdf_metadata(file_path)
            logger.info(
                "MetaLayer (PDF): cleared %d metadata field(s) from %s",
                len(result.stripped_fields),
                file_path.name,
            )
            return result

        # Image: prefer ExifTool if present on PATH
        if shutil.which("exiftool") is not None:
            logger.debug("Using ExifTool for %s", file_path.name)
            result.stripped_fields = _strip_exif_exiftool(file_path)
        else:
            logger.debug("ExifTool not found; falling back to piexif for %s", file_path.name)
            result.stripped_fields = _strip_exif_piexif(file_path)

        logger.info(
            "MetaLayer: stripped %d EXIF field(s) from %s (GPS=%s)",
            len(result.stripped_fields),
            file_path.name,
            result.exif_gps_found,
        )
        return result
