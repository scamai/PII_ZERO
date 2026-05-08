"""Presidio PatternRecognizer list for insurance-domain PII patterns."""

from __future__ import annotations

import re
from typing import Optional

from presidio_analyzer import Pattern, PatternRecognizer, RecognizerResult

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
# ABA routing number — 9 digits with 3-7-1 checksum validation
# ABA check: sum(digit[i] * [3,7,1,3,7,1,3,7,1][i]) % 10 == 0
# Without this, 9-digit phone/zip fragments produce P≈0.29; with it P rises sharply.
# ---------------------------------------------------------------------------
class _AbaRoutingRecognizer(PatternRecognizer):
    """ABA routing recognizer that gates matches on the 3-7-1 Luhn-like checksum."""

    _WEIGHTS = [3, 7, 1, 3, 7, 1, 3, 7, 1]

    def validate_result(self, pattern_text: str) -> Optional[bool]:
        digits = re.sub(r"\D", "", pattern_text)
        if len(digits) != 9:
            return False
        total = sum(int(d) * w for d, w in zip(digits, self._WEIGHTS))
        return total % 10 == 0


_routing_recognizer = _AbaRoutingRecognizer(
    supported_entity="US_BANK_NUMBER",
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
    """Covers 13-19 digit credit/debit cards (Visa, Mastercard, Amex, Maestro, Discover).

    Base score 0.65: Luhn validation provides strong signal; context enhancer
    adds 0.15 more when card-related keywords appear nearby (→ 0.8).
    """

    PATTERNS = [
        Pattern("cc_16_spaced", r"\b(?:\d[ -]?){12,18}\d\b", score=0.65),
        Pattern("cc_compact", r"\b\d{13,19}\b", score=0.65),
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
# SWIFT/BIC code — 8 or 11 character bank identifier code
# Format: AAAA BB CC [DDD]  where A=bank, B=country, C=location, D=branch
# ---------------------------------------------------------------------------
_swift_bic_recognizer = PatternRecognizer(
    supported_entity="SWIFT_BIC_CODE",
    name="SwiftBic_Recognizer",
    patterns=[
        # Base score 0.40: requires context keywords to reach ≥0.6 threshold.
        # SWIFT codes (8/11 chars) collide with company abbreviations and
        # all-caps words — context is mandatory to avoid FP explosion.
        Pattern(
            name="swift_bic_11",
            regex=r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}[A-Z0-9]{3}\b",
            score=0.40,
        ),
        Pattern(
            name="swift_bic_8",
            regex=r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}\b",
            score=0.40,
        ),
    ],
    context=[
        "swift", "bic", "swift code", "bic code", "bank identifier",
        "international transfer", "wire", "iban", "bank code",
    ],
)

# ---------------------------------------------------------------------------
# CVV / CVC — 3-4 digit card security code, always near card keywords
# ---------------------------------------------------------------------------
_cvv_recognizer = PatternRecognizer(
    supported_entity="CREDIT_CARD_SECURITY_CODE",
    name="CVV_Recognizer",
    patterns=[
        # Base score 0.40: below the 0.6 detection threshold on its own.
        # Context enhancer adds ~0.35 when cvv/cvc keywords appear → total 0.75.
        # This ensures we only flag CVV when explicitly labeled, avoiding FPs
        # on amounts ($123), zip codes, or SSN substrings.
        Pattern(
            name="cvv_3_4_digit",
            regex=r"\b\d{3,4}\b",
            score=0.40,
        ),
    ],
    context=[
        "cvv", "cvc", "cvc2", "cvv2", "cid", "security code",
        "card verification", "card security",
    ],
)

# ---------------------------------------------------------------------------
# PHONE_NUMBER — US and international phone numbers
#
# Presidio's built-in PhoneRecognizer scores at 0.4 without context keywords,
# which falls below the 0.6 threshold for bare numbers in financial documents.
# This recognizer sets base score 0.65 for common US formats (deterministic
# digit-count + area-code validation) and 0.60 for E.164 international format.
# ---------------------------------------------------------------------------
_phone_recognizer = PatternRecognizer(
    supported_entity="PHONE_NUMBER",
    name="Phone_Recognizer",
    patterns=[
        # US 10-digit: (800) 555-1234 / 800-555-1234 / 800.555.1234
        # Area code must start 2-9; exchange must start 2-9
        Pattern(
            name="phone_us_10digit",
            # Lookbehind blocks numbers embedded in EDI (+9876543210), SWIFT (:5248632145), IBAN (Q5899676300)
            regex=r"(?<![+:\w])(?:\+1[\s.\-]?)?(?:\([2-9]\d{2}\)|[2-9]\d{2})[\s.\-]?[2-9]\d{2}[\s.\-]?\d{4}(?!\d)",
            score=0.65,
        ),
        # E.164 international: +44 20 7946 0958 / +33-1-42-86-83-26
        # Lookbehind blocks EDI field separators (:12+999999, SU+9X:12+123)
        Pattern(
            name="phone_international",
            regex=r"(?<![a-zA-Z0-9])\+[1-9]\d{0,2}[\s.\-]?\(?\d{1,4}\)?[\s.\-]?\d{2,4}[\s.\-]?\d{2,4}(?:[\s.\-]?\d{2,4})?(?!\d|:)\b",
            score=0.60,
        ),
    ],
    context=[
        "phone", "mobile", "cell", "tel", "telephone", "fax",
        "contact", "call", "number", "ph", "ph.", "ext",
    ],
)

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
# ITIN — Individual Taxpayer Identification Number (9xx-xx-xxxx)
# Built-in UsItinRecognizer scores 0.5, which falls below our min_confidence=0.6.
# This replacement scores 0.65 and adds context gates to suppress FPs.
# ---------------------------------------------------------------------------
_itin_recognizer = PatternRecognizer(
    supported_entity="US_ITIN",
    name="ITIN_Recognizer",
    patterns=[
        Pattern(
            name="itin_dashes",
            regex=r"\b9\d{2}-\d{2}-\d{4}\b",
            score=0.65,
        ),
        Pattern(
            name="itin_spaces",
            regex=r"\b9\d{2} \d{2} \d{4}\b",
            score=0.55,
        ),
    ],
    context=[
        "itin", "individual taxpayer", "tax id", "taxpayer identification",
        "taxpayer id", "irs", "w-7", "form w7",
    ],
)

# ---------------------------------------------------------------------------
# US Passport — letter + 8 digits (pre-2021) or 9 alphanumeric (post-2021)
# Built-in UsPassportRecognizer scores 0.45, below our min_confidence=0.6.
# This replacement scores 0.65 with context gates to prevent license-plate FPs.
# ---------------------------------------------------------------------------
_us_passport_recognizer = PatternRecognizer(
    supported_entity="US_PASSPORT",
    name="Passport_Recognizer",
    patterns=[
        Pattern(
            name="us_passport_letter_digits",
            regex=r"\b[A-Z]\d{8}\b",
            score=0.65,
        ),
        Pattern(
            name="us_passport_alphanumeric",
            regex=r"\b[A-Z][A-Z0-9]{8}\b",
            score=0.45,
        ),
    ],
    context=[
        "passport", "passport number", "passport no", "passport #",
        "us passport", "american passport", "travel document",
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
    _swift_bic_recognizer,
    _cvv_recognizer,
    _phone_recognizer,
    _address_recognizer,
    _itin_recognizer,
    _us_passport_recognizer,
]
