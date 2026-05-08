"""End-to-end smoke test: synthetic CMS-1500 insurance claim form.

Generates a realistic CMS-1500 PDF with a native text layer using PyMuPDF,
runs the full Presidio NER pipeline (text extraction path, no OCR needed),
and verifies that the critical PII fields are detected.

Marked `slow` because it loads presidio + spaCy.  Run with:
    pytest tests/test_smoke_claim_form.py -v

Note: this test does NOT exercise the VLM layer (too slow for CI).
The pipeline is invoked in inspect/dry-run mode — no files written.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Skip if fitz / presidio unavailable
# ---------------------------------------------------------------------------
fitz = pytest.importorskip("fitz", reason="PyMuPDF (fitz) not installed")
pytest.importorskip("presidio_analyzer", reason="presidio-analyzer not installed")

import fitz as _fitz  # noqa: E402

# ---------------------------------------------------------------------------
# Known PII values embedded in the synthetic form
# ---------------------------------------------------------------------------

CLAIM = {
    "patient_name": "Jane M. Doe",
    "dob": "03/14/1985",
    "ssn": "523-67-4891",
    "insured_name": "Robert A. Doe",
    "policy_number": "POL-7834521",
    "npi": "1234567893",       # valid NPI checksum
    "ein": "47-1234567",
    "phone": "(312) 555-0147",
    "email": "jane.doe@example.com",
    "address": "412 Maple Street, Chicago, IL 60601",
    "claim_ref": "CLM-2024-88801",
}

# Which entity types must appear in detected output (normalised)
REQUIRED_ENTITIES = {
    "PERSON",       # patient_name, insured_name
    "DATE_TIME",    # dob
    "US_SSN",       # ssn
    "PHONE_NUMBER", # phone
    "EMAIL_ADDRESS",# email
    "POLICY_NUM",   # policy_number
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_cms1500_pdf(out_path: Path) -> None:
    """Render a minimal CMS-1500 look-alike as a searchable PDF using PyMuPDF."""
    doc = _fitz.open()
    page = doc.new_page(width=612, height=792)   # US letter in points

    def row(label: str, value: str, y: float) -> None:
        page.insert_text((40, y), f"{label}:", fontsize=9, color=(0.4, 0.4, 0.4))
        page.insert_text((220, y), value, fontsize=11, color=(0, 0, 0))

    # Header
    page.draw_rect(_fitz.Rect(0, 0, 612, 36), color=None, fill=(0.86, 0.86, 0.86))
    page.insert_text((40, 24), "HEALTH INSURANCE CLAIM FORM (CMS-1500)",
                     fontsize=13, color=(0, 0, 0))

    # Patient section
    y = 60.0
    page.insert_text((40, y), "PATIENT & INSURED INFORMATION", fontsize=11, color=(0.1, 0.1, 0.1))
    y += 20

    row("Patient Name",           CLAIM["patient_name"], y); y += 20
    row("Date of Birth",          CLAIM["dob"],          y); y += 20
    row("Social Security Number", CLAIM["ssn"],          y); y += 20
    row("Patient Address",        CLAIM["address"],      y); y += 20
    row("Phone",                  CLAIM["phone"],        y); y += 20
    row("Email",                  CLAIM["email"],        y); y += 20

    # Insured section
    y += 10
    page.insert_text((40, y), "INSURED INFORMATION", fontsize=11, color=(0.1, 0.1, 0.1))
    y += 20

    row("Insured Name",              CLAIM["insured_name"],  y); y += 20
    row("Insurance Policy Number",   CLAIM["policy_number"], y); y += 20

    # Provider section
    y += 10
    page.insert_text((40, y), "PROVIDER INFORMATION", fontsize=11, color=(0.1, 0.1, 0.1))
    y += 20

    row("Rendering Provider NPI", CLAIM["npi"], y); y += 20
    row("Billing Provider EIN",   CLAIM["ein"], y); y += 20

    # Claim section
    y += 10
    page.insert_text((40, y), "CLAIM INFORMATION", fontsize=11, color=(0.1, 0.1, 0.1))
    y += 20

    row("Claim Reference", CLAIM["claim_ref"], y); y += 20

    # Footer
    page.draw_rect(_fitz.Rect(0, 756, 612, 792), color=None, fill=(0.86, 0.86, 0.86))
    page.insert_text((40, 776), "FORM CMS-1500 (02-12)", fontsize=8, color=(0.4, 0.4, 0.4))

    doc.save(str(out_path))
    doc.close()


def _run_inspect(pdf_path: Path) -> list:
    """Run pipeline.inspect() and return all detected RedactionBoxes."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from pii_redact.config import load_settings
    from pii_redact.pipeline import PIIRedactionPipeline

    settings = load_settings()
    pipeline = PIIRedactionPipeline(settings, use_vlm=False)
    result = pipeline.inspect(pdf_path)
    return result.all_boxes


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCMS1500SmokeSuite:
    """Full pipeline smoke test on a synthetic CMS-1500 form."""

    @pytest.fixture(scope="class")
    def claim_pdf(self, tmp_path_factory) -> Path:
        p = tmp_path_factory.mktemp("smoke") / "cms1500.pdf"
        _build_cms1500_pdf(p)
        return p

    @pytest.fixture(scope="class")
    def detections(self, claim_pdf) -> list:
        return _run_inspect(claim_pdf)

    def test_pipeline_runs_without_error(self, claim_pdf, detections):
        """Pipeline must complete and return a non-empty list."""
        assert isinstance(detections, list), "inspect() must return a list"

    def test_at_least_one_detection(self, detections):
        """Must detect something on a PII-rich form."""
        assert len(detections) > 0, (
            f"Pipeline returned 0 detections on a CMS-1500 form with {len(CLAIM)} PII fields"
        )

    def test_entity_type_coverage(self, detections):
        """All required entity types must appear in detections."""
        found_types = {box.entity_type for box in detections}
        missing = REQUIRED_ENTITIES - found_types
        assert not missing, (
            f"Missing entity types: {missing}\n"
            f"Detected: {sorted(found_types)}"
        )

    def test_ssn_detected(self, detections):
        """SSN 523-67-4891 must be flagged."""
        ssn_hits = [b for b in detections if b.entity_type == "US_SSN"]
        assert ssn_hits, "SSN not detected"
        texts = [b.text_found or "" for b in ssn_hits]
        assert any(CLAIM["ssn"] in t or t in CLAIM["ssn"] for t in texts), (
            f"SSN value not matched. Hits: {texts}"
        )

    def test_person_name_detected(self, detections):
        """At least one of the two person names must be detected."""
        person_hits = [b for b in detections if b.entity_type == "PERSON"]
        assert person_hits, "No PERSON entities detected"

    def test_policy_number_detected(self, detections):
        """POL-prefixed policy number must be detected by regex recognizer."""
        policy_hits = [b for b in detections if b.entity_type == "POLICY_NUM"]
        assert policy_hits, (
            f"POLICY_NUM not detected. All types: {[b.entity_type for b in detections]}"
        )

    def test_no_unreasonably_high_fp_rate(self, detections):
        """Sanity check: FP rate should not exceed 10x the number of PII fields."""
        max_reasonable = len(CLAIM) * 10
        assert len(detections) <= max_reasonable, (
            f"Suspiciously many detections: {len(detections)} "
            f"(max expected ~{max_reasonable} for {len(CLAIM)} PII fields)"
        )

    def test_confidence_scores_in_range(self, detections):
        """All confidence scores must be in [0.0, 1.0]."""
        bad = [b for b in detections if not (0.0 <= b.confidence <= 1.0)]
        assert not bad, f"Out-of-range confidence scores: {bad}"

    def test_boxes_have_valid_coordinates(self, detections):
        """All boxes must have non-negative dimensions."""
        bad = [b for b in detections if b.w < 0 or b.h < 0]
        assert not bad, f"Boxes with negative dimensions: {bad}"


# ---------------------------------------------------------------------------
# Standalone report (python tests/test_smoke_claim_form.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        pdf_path = Path(tmpdir) / "cms1500.pdf"
        _build_cms1500_pdf(pdf_path)
        print(f"[smoke] Form written to {pdf_path}")

        boxes = _run_inspect(pdf_path)
        print(f"\n[smoke] {len(boxes)} detection(s):\n")

        found_types: set[str] = set()
        for box in sorted(boxes, key=lambda b: (b.entity_type, b.page)):
            found_types.add(box.entity_type)
            txt = (box.text_found or "")[:60]
            print(f"  {box.entity_type:<22} conf={box.confidence:.2f}  text={txt!r}")

        print(f"\n[smoke] Entity types found: {sorted(found_types)}")
        missing = REQUIRED_ENTITIES - found_types
        if missing:
            print(f"[smoke] MISSING required types: {missing}")
            sys.exit(1)
        else:
            print("[smoke] All required entity types present. PASS")
