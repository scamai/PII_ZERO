"""Template coordinate system for known insurance form types.

All (x_mm, y_mm, w_mm, h_mm) values are approximate bounding boxes measured
on standard letter-size (215.9 × 279.4 mm) form originals at 300 DPI.

IMPORTANT: These coordinates are best-effort approximations derived from
publicly available form specifications and must be calibrated with real
scanned form samples before use in production.  Render the template overlay
at 300 DPI against a clean form scan, visually verify each field, and update
the values accordingly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pii_redact.models import RedactionBox

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Conversion factor: mm → pixels at a given DPI
_MM_PER_INCH = 25.4


def _mm_to_px(mm: float, dpi: int) -> int:
    return round(mm * dpi / _MM_PER_INCH)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class FormTemplate:
    """A known insurance form type with field coordinate definitions."""

    name: str
    """Human-readable form name, e.g. 'CMS-1500' or 'ACORD 125'."""

    canonical_hash: str
    """Perceptual hash (imagehash.phash hex string) of a clean form image.
    Filled in during setup/calibration; empty string until calibrated."""

    fields: dict[str, tuple[int, int, int, int]] = field(default_factory=dict)
    """Mapping of field_name -> (x_mm, y_mm, w_mm, h_mm) at 300 DPI.

    Coordinate origin is the top-left corner of the page.
    All values are in millimetres.
    """


# ---------------------------------------------------------------------------
# CMS-1500 template
# ---------------------------------------------------------------------------
# Reference: https://www.cms.gov/medicare/billing/electronicbillingediforms/downloads/form1500.pdf
# Form size: 215.9 × 279.4 mm (US Letter)
# NOTE: values below are approximations — calibrate before production use.

CMS_1500 = FormTemplate(
    name="CMS-1500",
    canonical_hash="",  # populate after scanning a clean blank form
    fields={
        # Box 1a — Insured's ID number
        "insured_id":           (127, 25,  84,  7),
        # Box 2 — Patient's name (last, first, middle initial)
        "patient_name":         (  3, 35, 100,  7),
        # Box 3 — Patient's date of birth / sex
        "patient_dob":          (106, 35,  57,  7),
        "patient_sex":          (166, 35,  45,  7),
        # Box 4 — Insured's name
        "insured_name":         (  3, 46, 100,  7),
        # Box 5 — Patient's address
        "patient_address":      (  3, 56, 100,  7),
        "patient_city":         (  3, 64,  70,  7),
        "patient_state":        ( 76, 64,  27,  7),
        "patient_zip":          (106, 64,  45,  7),
        "patient_phone":        (154, 64,  57,  7),
        # Box 7 — Insured's address
        "insured_address":      (  3, 76, 100,  7),
        # Box 11 — Insured's group / policy number
        "insured_group":        (  3, 96, 100,  7),
        # Box 11c — Insured's DOB / sex
        "insured_dob":          (  3,106,  57,  7),
        # Box 17a — Referring provider NPI (used as proxy for physician_npi)
        "physician_npi":        (127,163,  84,  7),
        # Box 17b — Referring provider name
        "physician_name":       (  3,163, 120,  7),
        # Box 21 — Diagnosis codes (ICD-10) — 4 slots
        "diagnosis_1":          (  3,181,  48,  7),
        "diagnosis_2":          ( 55,181,  48,  7),
        "diagnosis_3":          (107,181,  48,  7),
        "diagnosis_4":          (159,181,  48,  7),
        # Box 24a — Date(s) of service
        "service_date_from":    (  3,196,  27,  7),
        "service_date_to":      ( 32,196,  27,  7),
        # Box 24b — Place of service
        "place_of_service":     ( 62,196,  14,  7),
        # Box 24d — Procedure code / modifier
        "procedure_code":       ( 79,196,  28,  7),
        "modifier":             (110,196,  25,  7),
        # Box 24e — Diagnosis pointer
        "diagnosis_pointer":    (138,196,  14,  7),
        # Box 24f — Charges
        "charges":              (155,196,  22,  7),
        # Box 24g — Days or units
        "days_units":           (180,196,  12,  7),
        # Box 25 — Federal tax ID
        "federal_tax_id":       (  3,237,  72,  7),
        # Box 26 — Patient's account number
        "patient_acct_num":     ( 79,237,  58,  7),
        # Box 27 — Accept assignment
        "accept_assignment":    (141,237,  30,  7),
        # Box 28 — Total charge
        "total_charge":         (  3,247,  58,  7),
        # Box 29 — Amount paid
        "amount_paid":          ( 65,247,  58,  7),
        # Box 31 — Physician/supplier signature (name area)
        "physician_signature":  (  3,257, 100,  9),
        # Box 32 — Service facility location
        "facility_name":        (  3,267,  90,  7),
        "facility_npi":         (  3,274,  90,  7),
        # Box 33 — Billing provider info / NPI
        "billing_npi":          (110,267,  97,  7),
        # Box 33a — Billing provider address
        "physician_address":    (110,274,  97,  7),
    },
)


# ---------------------------------------------------------------------------
# ACORD 125 template — Commercial General Liability Application
# ---------------------------------------------------------------------------
# Reference: ACORD 125 (2016/03) standard form
# Form size: 215.9 × 279.4 mm (US Letter), multi-page; coordinates below are page 1.
# NOTE: values below are approximations — calibrate before production use.

ACORD_125 = FormTemplate(
    name="ACORD 125",
    canonical_hash="",  # populate after scanning a clean blank form
    fields={
        # Section 1 — Applicant information
        "applicant_name":            (  3, 28, 130,  7),
        "applicant_mailing_address": (  3, 38, 130,  7),
        "applicant_city":            (  3, 46,  80,  7),
        "applicant_state":           ( 86, 46,  18,  7),
        "applicant_zip":             (107, 46,  28,  7),
        "applicant_phone":           (138, 46,  69,  7),
        "applicant_fax":             (138, 54,  69,  7),
        "applicant_website":         (  3, 54, 130,  7),
        "applicant_fein":            (  3, 62,  60,  7),   # Federal EIN
        "applicant_sic":             ( 66, 62,  30,  7),   # SIC code
        # Contact / producer info
        "contact_name":              (  3, 72, 100,  7),
        "contact_email":             (  3, 80, 100,  7),
        # Section 2 — Policy information
        "policy_effective_date":     (  3, 92,  45,  7),
        "policy_expiry_date":        ( 52, 92,  45,  7),
        "billing_plan":              (100, 92,  50,  7),
        # Section 3 — Nature of business
        "business_description":      (  3,106, 205, 10),
        "years_in_business":         (  3,119,  30,  7),
        "num_employees":             ( 37,119,  30,  7),
        "annual_revenue":            ( 70,119,  50,  7),
        # Section 4 — Locations
        "location_address":          (  3,133, 130,  7),
        "location_city":             (  3,141,  80,  7),
        "location_state":            ( 86,141,  18,  7),
        "location_zip":              (107,141,  28,  7),
        # Section 5 — Coverage limits
        "each_occurrence_limit":     (  3,155,  60,  7),
        "general_aggregate_limit":   ( 67,155,  60,  7),
        "products_limit":            (131,155,  60,  7),
        # Section 7 — Prior losses / claims
        "prior_losses_amount":       (  3,185,  80,  7),
        "prior_losses_description":  ( 87,185, 120,  7),
        # Signature block
        "applicant_signature":       (  3,255, 120,  9),
        "applicant_sign_date":       (127,255,  50,  7),
        "producer_name":             (  3,267, 100,  7),
        "producer_license":          (107,267,  50,  7),
    },
)

# Registry of all available templates, keyed by canonical name
TEMPLATE_REGISTRY: dict[str, FormTemplate] = {
    CMS_1500.name: CMS_1500,
    ACORD_125.name: ACORD_125,
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def get_template_boxes(template: FormTemplate, dpi: int = 300) -> list[RedactionBox]:
    """Convert a FormTemplate's mm field coordinates to pixel RedactionBoxes.

    Parameters
    ----------
    template:
        A FormTemplate instance.
    dpi:
        Rendering DPI — must match the DPI used to rasterise the document.
        Default 300 DPI is the pipeline standard (see settings.yaml: pipeline.ocr_dpi).

    Returns
    -------
    list[RedactionBox]
        One box per field, all with source="template" and confidence=1.0.
    """
    boxes: list[RedactionBox] = []
    for field_name, (x_mm, y_mm, w_mm, h_mm) in template.fields.items():
        x_px = _mm_to_px(x_mm, dpi)
        y_px = _mm_to_px(y_mm, dpi)
        w_px = _mm_to_px(w_mm, dpi)
        h_px = _mm_to_px(h_mm, dpi)
        boxes.append(
            RedactionBox(
                x=x_px,
                y=y_px,
                w=w_px,
                h=h_px,
                entity_type=field_name.upper(),
                confidence=1.0,
                source="template",
            )
        )
    return boxes


def classify_form_template(image: np.ndarray) -> FormTemplate | None:
    """Identify which known form template best matches *image* via perceptual hashing.

    Uses imagehash.phash to compute an 8×8 DCT hash of the image and compares it
    against the canonical_hash of each registered template.  Returns the closest
    matching template when the Hamming distance is within the configured threshold.

    Parameters
    ----------
    image:
        BGR or grayscale image (np.ndarray) of the form page.

    Returns
    -------
    FormTemplate | None
        The best-matching template, or None if no template is close enough.
    """
    try:
        import imagehash  # type: ignore[import]
        from PIL import Image  # type: ignore[import]
    except ImportError:
        logger.warning(
            "imagehash / Pillow not installed; form classification disabled"
        )
        return None

    threshold: int = 10
    try:
        threshold = int(settings_threshold())
    except Exception:
        pass

    # Compute hash for the input image
    if image.ndim == 3:
        pil_img = Image.fromarray(image[..., ::-1])  # BGR -> RGB
    else:
        pil_img = Image.fromarray(image)

    try:
        query_hash = imagehash.phash(pil_img)
    except Exception as exc:
        logger.warning("imagehash.phash failed: %s", exc)
        return None

    best_template: FormTemplate | None = None
    best_distance: int = threshold + 1

    for tmpl in TEMPLATE_REGISTRY.values():
        if not tmpl.canonical_hash:
            # Template not yet calibrated — skip
            continue
        try:
            ref_hash = imagehash.hex_to_hash(tmpl.canonical_hash)
            distance = query_hash - ref_hash
            if distance < best_distance:
                best_distance = distance
                best_template = tmpl
        except Exception as exc:
            logger.debug("Hash comparison failed for template %s: %s", tmpl.name, exc)
            continue

    if best_template is not None:
        logger.info(
            "Form classified as '%s' (hamming=%d, threshold=%d)",
            best_template.name,
            best_distance,
            threshold,
        )
    else:
        logger.debug(
            "No template matched within hamming threshold %d (best=%d)",
            threshold,
            best_distance,
        )

    return best_template


def settings_threshold() -> int:
    """Read perceptual_hash_threshold from settings, with a safe fallback."""
    from pii_redact.config import settings as _settings

    try:
        return int(_settings.templates.perceptual_hash_threshold)
    except AttributeError:
        return 10
