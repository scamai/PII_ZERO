"""Redaction engine, AES-256 vault, and audit trail."""

from pii_redact.redact.engine import (
    RedactionEngine,
    merge_boxes,
    redact_image,
    redact_pdf_page,
    save_redacted_image,
    save_redacted_pdf,
)
from pii_redact.redact.vault import PIIVault
from pii_redact.redact.audit import write_audit_json

__all__ = [
    "RedactionEngine",
    "merge_boxes",
    "redact_image",
    "redact_pdf_page",
    "save_redacted_image",
    "save_redacted_pdf",
    "PIIVault",
    "write_audit_json",
]
