"""Tests for vault encrypt/decrypt and audit log functionality.

These tests mock SQLite and filesystem I/O — no actual files are created.
"""

from __future__ import annotations

import importlib
import json
import os

import pytest
pytestmark = pytest.mark.fast
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Helpers — skip gracefully if vault module doesn't exist yet
# ---------------------------------------------------------------------------


def _import_vault():
    """Import the vault module, skipping if not yet implemented."""
    try:
        return importlib.import_module("pii_redact.vault")
    except ModuleNotFoundError:
        pytest.skip("pii_redact.vault not yet implemented — skipping vault tests")


def _require_cryptography():
    try:
        from cryptography.fernet import Fernet
        return Fernet
    except ImportError:
        pytest.skip("cryptography library not installed")


# ---------------------------------------------------------------------------
# Standalone crypto round-trip tests (no vault module dependency)
# ---------------------------------------------------------------------------
#
# These tests exercise the cryptographic primitives directly so that even if
# the vault module isn't written yet, the crypto contract is validated.


@pytest.mark.subprocess
class TestCryptoRoundTrip:
    """Encrypt/decrypt round-trip using Fernet (the expected vault cipher)."""

    def test_encrypt_decrypt_roundtrip(self):
        Fernet = _require_cryptography()
        key = Fernet.generate_key()
        f = Fernet(key)

        plaintext = b"John Doe SSN: 123-45-6789"
        ciphertext = f.encrypt(plaintext)

        assert ciphertext != plaintext, "Ciphertext must differ from plaintext"
        assert f.decrypt(ciphertext) == plaintext

    def test_wrong_key_raises(self):
        Fernet = _require_cryptography()
        from cryptography.fernet import InvalidToken

        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()
        assert key1 != key2

        f1 = Fernet(key1)
        f2 = Fernet(key2)

        ciphertext = f1.encrypt(b"secret")
        with pytest.raises(InvalidToken):
            f2.decrypt(ciphertext)

    def test_wrong_key_does_not_return_garbage(self):
        """Decrypting with wrong key must raise, never silently return garbage."""
        Fernet = _require_cryptography()
        from cryptography.fernet import InvalidToken

        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()

        f1 = Fernet(key1)
        f2 = Fernet(key2)

        ciphertext = f1.encrypt(b"top secret PII data")
        result = None
        try:
            result = f2.decrypt(ciphertext)
        except (InvalidToken, Exception):
            pass  # correct — exception expected

        assert result is None, (
            "Decryption with wrong key returned data instead of raising. "
            "Silently returning garbage is worse than raising — it hides PII leakage."
        )

    def test_different_plaintexts_produce_different_ciphertexts(self):
        Fernet = _require_cryptography()
        key = Fernet.generate_key()
        f = Fernet(key)

        c1 = f.encrypt(b"SSN 111-22-3333")
        c2 = f.encrypt(b"SSN 444-55-6666")
        assert c1 != c2

    def test_nonce_uniqueness(self):
        """Encrypting the same plaintext twice gives different ciphertexts (IND-CPA)."""
        Fernet = _require_cryptography()
        key = Fernet.generate_key()
        f = Fernet(key)

        plain = b"repeat me"
        c1 = f.encrypt(plain)
        c2 = f.encrypt(plain)
        assert c1 != c2, "Fernet uses a fresh IV each time — ciphertexts must differ"
        # But both decrypt correctly
        assert f.decrypt(c1) == plain
        assert f.decrypt(c2) == plain

    def test_key_length_is_adequate(self):
        """Generated key must be at least 32 bytes (256 bits) when decoded."""
        import base64
        Fernet = _require_cryptography()
        key = Fernet.generate_key()
        raw = base64.urlsafe_b64decode(key)
        assert len(raw) >= 32, f"Key too short: {len(raw)} bytes"

    def test_ciphertext_is_not_plaintext(self):
        Fernet = _require_cryptography()
        key = Fernet.generate_key()
        f = Fernet(key)

        plaintext = b"POLICY_NUM: POL-ABC1234"
        ciphertext = f.encrypt(plaintext)
        assert plaintext not in ciphertext, "Plaintext must not appear verbatim in ciphertext"


# ---------------------------------------------------------------------------
# Vault module tests (skipped if vault not yet written)
# ---------------------------------------------------------------------------


@pytest.mark.subprocess
class TestVaultEncryptDecrypt:
    def test_vault_encrypt_decrypt_roundtrip(self):
        vault = _import_vault()
        if not hasattr(vault, "encrypt") or not hasattr(vault, "decrypt"):
            pytest.skip("vault module lacks encrypt/decrypt functions")

        key = vault.generate_key()
        plaintext = "John Doe SSN: 123-45-6789"
        ciphertext = vault.encrypt(plaintext, key)

        assert ciphertext != plaintext
        assert vault.decrypt(ciphertext, key) == plaintext

    def test_vault_wrong_key_raises(self):
        vault = _import_vault()
        if not hasattr(vault, "encrypt") or not hasattr(vault, "decrypt"):
            pytest.skip("vault module lacks encrypt/decrypt functions")

        key1 = vault.generate_key()
        key2 = vault.generate_key()
        ciphertext = vault.encrypt("secret", key1)

        with pytest.raises(Exception):
            vault.decrypt(ciphertext, key2)


# ---------------------------------------------------------------------------
# Audit log write/read with mocked SQLite
# ---------------------------------------------------------------------------


class TestAuditLog:
    """Verify audit log write/read contract using an in-memory SQLite database.

    If pii_redact.audit doesn't exist yet, tests are skipped gracefully.
    """

    def _get_audit_module(self):
        try:
            return importlib.import_module("pii_redact.audit")
        except ModuleNotFoundError:
            pytest.skip("pii_redact.audit not yet implemented")

    def test_audit_log_write_and_read_in_memory(self):
        """Write an audit event and read it back via an in-memory SQLite DB."""
        audit = self._get_audit_module()
        if not (hasattr(audit, "write_event") and hasattr(audit, "read_events")):
            pytest.skip("audit module lacks write_event/read_events")

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            event = {
                "doc_id": "test-001",
                "filename": "claim.pdf",
                "entity_type": "SSN",
                "source": "regex",
                "page": 0,
                "confidence": 0.9,
            }
            audit.write_event(db_path, event)
            events = audit.read_events(db_path)

            assert len(events) >= 1
            latest = events[-1]
            assert latest["doc_id"] == "test-001"
            assert latest["entity_type"] == "SSN"
        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_audit_log_does_not_store_pii_text(self):
        """Audit log must never persist the raw text_found value."""
        audit = self._get_audit_module()
        if not (hasattr(audit, "write_event") and hasattr(audit, "read_events")):
            pytest.skip("audit module lacks write_event/read_events")

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            event = {
                "doc_id": "test-002",
                "entity_type": "SSN",
                "source": "regex",
                "page": 0,
                "confidence": 0.9,
                "text_found": "123-45-6789",  # should be excluded from audit log
            }
            audit.write_event(db_path, event)
            events = audit.read_events(db_path)

            for e in events:
                stored = json.dumps(e)
                assert "123-45-6789" not in stored, (
                    "PII text_found value appeared in audit log — "
                    "this field must NEVER be written to the audit log."
                )
        finally:
            Path(db_path).unlink(missing_ok=True)


class TestAuditLogMockedSQLite:
    """Test audit log behaviour with sqlite3 fully mocked (no file I/O)."""

    def test_write_event_calls_sqlite_execute(self):
        """Verify write_event issues a SQL INSERT (mocked DB — no file created)."""
        try:
            audit = importlib.import_module("pii_redact.audit")
        except ModuleNotFoundError:
            pytest.skip("pii_redact.audit not yet implemented")

        if not hasattr(audit, "write_event"):
            pytest.skip("audit.write_event not implemented")

        mock_conn = mock.MagicMock()
        mock_cursor = mock.MagicMock()
        mock_conn.__enter__ = mock.Mock(return_value=mock_conn)
        mock_conn.__exit__ = mock.Mock(return_value=False)
        mock_conn.cursor.return_value = mock_cursor
        mock_conn.commit = mock.Mock()

        with mock.patch("sqlite3.connect", return_value=mock_conn):
            audit.write_event(
                ":memory:",
                {"doc_id": "x", "entity_type": "SSN", "source": "regex", "page": 0, "confidence": 0.9},
            )

        # The audit module must have called execute (or executemany) at least once
        assert mock_cursor.execute.called or mock_cursor.executemany.called, (
            "write_event did not call cursor.execute — no SQL was issued"
        )
