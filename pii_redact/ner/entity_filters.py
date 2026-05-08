"""Post-detection entity filters for reducing NER false positives.

These filters operate on individual detected spans and their surrounding
context. They are language-agnostic and apply after Presidio analysis.

Each filter function returns True if the span should be KEPT, False to drop.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# ORGANIZATION / ORG filters
# ---------------------------------------------------------------------------

# Generic role words and contract labels that spaCy misclassifies as ORG.
# All lowercase — compare against span.lower().
_ORG_ROLE_BLOCKLIST: frozenset[str] = frozenset({
    # Contract party labels
    "client", "vendor", "service provider", "service providers",
    "contractor", "subcontractor", "counterparty", "counterparties",
    "party", "parties", "provider", "providers", "supplier", "suppliers",
    "recipient", "recipients", "grantee", "grantor",
    "lessor", "lessee", "borrower", "lender", "creditor", "debtor",
    "payer", "payee", "issuer", "obligor",
    # Generic business nouns that aren't company names
    "customer service", "customer support", "support",
    "department", "division", "branch", "unit", "group",
    "healthcare service", "healthcare services",
    "property", "utilities", "service", "services",
    # Document / form structural labels
    "report", "report - renewal options", "summary",
    "agreement", "contract", "policy",
    # Technical field names (CSV / XML headers)
    "sender wallet", "sender country", "receiver wallet",
    "swapdetails", "tradedetails",
    # Acronyms handled separately, but catch a few common ones here
    "ai", "pii", "api", "edi", "xml", "fpml",
})

# Technical characters that indicate an artifact, not a company name.
_TECH_CHARS_RE = re.compile(r'[=<>{}/\\@]')

# Looks like an XML/URL schema string (very long with dots and slashes).
_SCHEMA_RE = re.compile(r'https?://|xsi:|xmlns:|schemaLocation', re.IGNORECASE)

# All-uppercase acronyms: ≤6 letters, no lowercase, no digits.
# Captures CMT, TIO, AI, UTC, HMRC, PII, etc.
_ACRONYM_RE = re.compile(r'^[A-Z]{2,6}$')

# Minimum word count: single-word spans that start with lowercase are role refs.
_FINANCIAL_CONTEXT_TERMS = frozenset({
    "employer", "employed", "employment", "employee", "workplace",
    "works at", "worked at", "working at", "place of employment",
    "lender", "borrower", "bank", "banking", "financial institution",
    "policyholder", "insured", "insurer", "insurance company",
    "inc.", "llc", "ltd.", "corp.", "gmbh", "s.a.", "plc", "hb",
    "inc,", "llc,", "ltd,",
})


def is_valid_org(span: str, context_window: str = "") -> bool:
    """Return True if *span* looks like a real organization name (not a FP).

    Parameters
    ----------
    span:
        The detected entity span text.
    context_window:
        Text surrounding the span (±100 chars recommended). Used for the
        financial-context gate on short/ambiguous spans.
    """
    if not span or not span.strip():
        return False

    span = span.strip()

    # 1. Blocklist: generic role words and contract labels
    if span.lower() in _ORG_ROLE_BLOCKLIST:
        return False

    # 2. All-caps acronym (2-6 letters, no punctuation)
    if _ACRONYM_RE.match(span):
        return False

    # 3. Technical artifacts: XML schema, URL patterns, special chars
    if _TECH_CHARS_RE.search(span) or _SCHEMA_RE.search(span):
        return False

    # 4. Starts with lowercase → role reference, not a proper noun
    if span[0].islower():
        return False

    # 5. Suspiciously long and contains no uppercase letters beyond first char
    #    (catches phrases like "the Loan Amount", "a risk diversification strategy")
    words = span.split()
    if len(words) > 6:
        # Long spans are usually document titles or sentences, not company names.
        # Allow if they contain a known legal suffix (Inc., Ltd., etc.)
        has_suffix = any(w.lower().rstrip(',') in {"inc", "llc", "ltd", "corp",
                                                    "gmbh", "plc", "hb", "sa",
                                                    "s.a", "co"} for w in words)
        if not has_suffix:
            return False

    # 6. Short ambiguous spans (1-2 words, no legal suffix) require financial context.
    has_legal_suffix = any(
        span.lower().endswith(s) for s in ("inc.", "llc", "ltd.", "corp.", "gmbh",
                                            "s.a.", "plc", "hb", " inc", " llc",
                                            " ltd", " corp", " co")
    )
    if not has_legal_suffix and len(words) <= 2 and context_window:
        ctx_lower = context_window.lower()
        has_financial_context = any(term in ctx_lower for term in _FINANCIAL_CONTEXT_TERMS)
        if not has_financial_context:
            return False

    return True


# ---------------------------------------------------------------------------
# PERSON filters (complement to langdetect filter)
# ---------------------------------------------------------------------------

# Single-word "person" FPs — common English nouns that spaCy misclassifies.
_PERSON_SINGLE_WORD_BLOCKLIST: frozenset[str] = frozenset({
    "email", "phone", "address", "date", "name", "vendor", "client",
    "provider", "department", "service", "agent", "admin", "user",
    "customer", "contact", "manager", "officer", "director",
    "representative", "rep", "staff", "team",
})


def is_valid_person(span: str) -> bool:
    """Return True if *span* is a plausible person name (not a label FP)."""
    if not span or not span.strip():
        return False
    span = span.strip()
    words = span.split()
    # Single-word person FPs from form field labels
    if len(words) == 1 and span.lower() in _PERSON_SINGLE_WORD_BLOCKLIST:
        return False
    # Starts with lowercase → not a proper name
    if span[0].islower():
        return False
    return True
