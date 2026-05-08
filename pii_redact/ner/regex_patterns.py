"""Presidio PatternRecognizer list for insurance-domain PII patterns."""

from __future__ import annotations

from presidio_analyzer import Pattern, PatternRecognizer

# ---------------------------------------------------------------------------
# SSN  — XXX-XX-XXXX  or  XXXXXXXXX
# ---------------------------------------------------------------------------
_ssn_recognizer = PatternRecognizer(
    supported_entity="SSN",
    name="SSN_Recognizer",
    patterns=[
        Pattern(
            name="ssn_dashes",
            regex=r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
            score=0.85,
        ),
        Pattern(
            name="ssn_nodashes",
            regex=r"\b(?!000|666|9\d{2})\d{3}(?!00)\d{2}(?!0000)\d{4}\b",
            score=0.5,
        ),
    ],
    context=[
        "ssn", "social security", "social security number", "taxpayer",
        "tin", "tax id",
    ],
)

# ---------------------------------------------------------------------------
# NPI — National Provider Identifier (10 digits, starts 1 or 2)
# ---------------------------------------------------------------------------
_npi_recognizer = PatternRecognizer(
    supported_entity="NPI",
    name="NPI_Recognizer",
    patterns=[
        Pattern(
            name="npi_10digit",
            regex=r"\b[12]\d{9}\b",
            score=0.65,
        ),
    ],
    context=[
        "npi", "national provider", "provider id", "provider identifier",
        "rendering provider", "billing provider",
    ],
)

# ---------------------------------------------------------------------------
# EIN — Employer Identification Number  XX-XXXXXXX
# ---------------------------------------------------------------------------
_ein_recognizer = PatternRecognizer(
    supported_entity="EIN",
    name="EIN_Recognizer",
    patterns=[
        Pattern(
            name="ein_standard",
            regex=r"\b(?!00)\d{2}-\d{7}\b",
            score=0.75,
        ),
    ],
    context=[
        "ein", "employer identification", "federal tax", "fein",
        "tax id", "employer id",
    ],
)

# ---------------------------------------------------------------------------
# ICD-10 codes  — e.g.  M54.5  Z00.00  A00-B99 (diagnosis)
# ---------------------------------------------------------------------------
_icd10_recognizer = PatternRecognizer(
    supported_entity="ICD10_CODE",
    name="ICD10_Recognizer",
    patterns=[
        Pattern(
            name="icd10_full",
            regex=r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?\b",
            score=0.70,
        ),
    ],
    context=[
        "icd", "icd-10", "icd10", "diagnosis", "dx", "primary diagnosis",
        "secondary diagnosis", "principal diagnosis", "condition code",
    ],
)

# ---------------------------------------------------------------------------
# CPT codes — 5-digit numeric (with optional modifier)
# ---------------------------------------------------------------------------
_cpt_recognizer = PatternRecognizer(
    supported_entity="CPT_CODE",
    name="CPT_Recognizer",
    patterns=[
        Pattern(
            name="cpt_5digit",
            regex=r"\b[0-9]{5}(?:-[A-Z0-9]{2})?\b",
            score=0.60,
        ),
    ],
    context=[
        "cpt", "procedure code", "procedure", "hcpcs", "service code",
        "billing code", "revenue code",
    ],
)

# ---------------------------------------------------------------------------
# Policy number — POL-XXXXXXX  (7 alphanumeric chars after prefix)
# ---------------------------------------------------------------------------
_policy_recognizer = PatternRecognizer(
    supported_entity="POLICY_NUM",
    name="PolicyNumber_Recognizer",
    patterns=[
        Pattern(
            name="policy_pol_prefix",
            regex=r"\bPOL-[A-Z0-9]{7,}\b",
            score=0.90,
        ),
        Pattern(
            name="policy_generic",
            regex=r"\b[A-Z]{2,4}[0-9]{6,10}\b",
            score=0.55,
        ),
    ],
    context=[
        "policy", "policy number", "policy no", "policy #", "pol no",
        "insurance policy", "policy id",
    ],
)

# ---------------------------------------------------------------------------
# ABA routing number — 9 digits starting with 0-3 (or 6-7 for certain)
# ---------------------------------------------------------------------------
_routing_recognizer = PatternRecognizer(
    supported_entity="ROUTING_NUM",
    name="RoutingNumber_Recognizer",
    patterns=[
        Pattern(
            name="routing_9digit",
            regex=r"\b(?:0[0-9]|1[0-2]|2[1-9]|3[0-2]|6[1-2]|7[1-2])\d{7}\b",
            score=0.70,
        ),
    ],
    context=[
        "routing", "routing number", "aba", "aba number", "aba routing",
        "bank routing", "transit number", "wire transfer",
    ],
)

# ---------------------------------------------------------------------------
# Adjuster ID — typically ADJ followed by alphanumerics
# ---------------------------------------------------------------------------
_adjuster_id_recognizer = PatternRecognizer(
    supported_entity="ADJUSTER_ID",
    name="AdjusterID_Recognizer",
    patterns=[
        Pattern(
            name="adjuster_adj_prefix",
            regex=r"\bADJ-?[A-Z0-9]{4,10}\b",
            score=0.85,
        ),
        Pattern(
            name="adjuster_generic",
            regex=r"\b[A-Z]{1,3}\d{5,8}\b",
            score=0.50,
        ),
    ],
    context=[
        "adjuster", "adjuster id", "claims adjuster", "adjuster number",
        "assigned adjuster", "handling adjuster",
    ],
)

# ---------------------------------------------------------------------------
# Claim reference number — CLM- or CLAIM- prefix patterns
# ---------------------------------------------------------------------------
_claim_ref_recognizer = PatternRecognizer(
    supported_entity="CLAIM_REF",
    name="ClaimRef_Recognizer",
    patterns=[
        Pattern(
            name="claim_clm_prefix",
            regex=r"\b(?:CLM|CLA|CLAIM)-[A-Z0-9]{6,15}\b",
            score=0.90,
        ),
        Pattern(
            name="claim_numeric",
            regex=r"\b\d{10,14}\b",
            score=0.45,
        ),
    ],
    context=[
        "claim", "claim number", "claim no", "claim #", "claim ref",
        "claim reference", "clm", "loss ref", "file number",
    ],
)

# ---------------------------------------------------------------------------
# DEA number — 2 letters + 7 digits with Luhn-like check structure
# ---------------------------------------------------------------------------
_dea_recognizer = PatternRecognizer(
    supported_entity="DEA_NUM",
    name="DEA_Recognizer",
    patterns=[
        Pattern(
            name="dea_standard",
            regex=r"\b[A-Z]{2}\d{7}\b",
            score=0.75,
        ),
    ],
    context=[
        "dea", "dea number", "dea registration", "drug enforcement",
        "prescriber", "controlled substance",
    ],
)

# ---------------------------------------------------------------------------
# CREDIT_CARD — 13-19 digit cards (Visa, Mastercard, Amex, Maestro, Discover)
# Extends coverage beyond Presidio's built-in CreditCardRecognizer to include
# 19-digit Maestro/Visa Electron cards; validates Luhn checksum.
# ---------------------------------------------------------------------------
class CreditCard19Recognizer(PatternRecognizer):
    """Covers 13-19 digit credit/debit cards (Visa, Mastercard, Amex, Maestro, Discover)."""

    PATTERNS = [
        Pattern("cc_16_spaced", r"\b(?:\d[ -]?){12,18}\d\b", score=0.5),
        Pattern("cc_compact", r"\b\d{13,19}\b", score=0.5),
    ]
    CONTEXT = [
        "credit", "debit", "card", "visa", "mastercard", "maestro", "amex",
        "discover", "payment", "card number", "cc", "cvv",
    ]

    def __init__(self) -> None:
        super().__init__(
            supported_entity="CREDIT_CARD",
            patterns=self.PATTERNS,
            context=self.CONTEXT,
            supported_language="en",
        )

    def invalidate_result(self, pattern_text: str) -> bool:  # type: ignore[override]
        digits = pattern_text.replace(" ", "").replace("-", "")
        if not digits.isdigit():
            return True
        if not (13 <= len(digits) <= 19):
            return True
        # Luhn check
        total = 0
        for i, d in enumerate(reversed(digits)):
            n = int(d)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return total % 10 != 0  # True = invalid


_credit_card_recognizer = CreditCard19Recognizer()

# ---------------------------------------------------------------------------
# ADDRESS — US-style street addresses  (e.g. "123 Maple Street")
# ---------------------------------------------------------------------------
_address_recognizer = PatternRecognizer(
    supported_entity="LOCATION",
    name="Address_Recognizer",
    patterns=[
        Pattern(
            name="us_street_address",
            regex=(
                r"\b\d{1,5}\s+(?:[A-Z][A-Za-z]*\s+){1,4}"
                r"(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|"
                r"Lane|Ln|Court|Ct|Way|Place|Pl|Circle|Trail|Terrace|"
                r"Parkway|Pkwy|Highway|Hwy|Square|Estates|Lodge|Plaza|Commons)\b"
            ),
            score=0.65,
        ),
    ],
    context=[
        "address", "addr", "street", "residence", "located at",
        "resides at", "home address", "mailing address", "billing address",
    ],
)

# ---------------------------------------------------------------------------
# Exported list consumed by presidio_setup.py
# ---------------------------------------------------------------------------
PRESIDIO_RECOGNIZER_LIST: list[PatternRecognizer] = [
    _ssn_recognizer,
    _npi_recognizer,
    _ein_recognizer,
    _icd10_recognizer,
    _cpt_recognizer,
    _policy_recognizer,
    _routing_recognizer,
    _adjuster_id_recognizer,
    _claim_ref_recognizer,
    _dea_recognizer,
    _credit_card_recognizer,
    _address_recognizer,
]
