"""Reversible vault: Fernet encryption + stable token map.

Each redaction session creates a VaultSession. The session:
  - generates a per-session Fernet key
  - maintains a plaintext → token mapping so identical PII gets the same token
  - encrypts and stores plaintext → ciphertext for authorized de-anonymization

The vault key is never written to disk alongside the redacted document.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Low-level crypto helpers (thin wrappers around Fernet)
# ---------------------------------------------------------------------------


def generate_key() -> bytes:
    """Generate a new Fernet key (32-byte random, base64url-encoded)."""
    from cryptography.fernet import Fernet
    return Fernet.generate_key()


def encrypt(plaintext: str, key: bytes) -> str:
    """Encrypt a plaintext string, returning a base64url ciphertext string."""
    from cryptography.fernet import Fernet
    f = Fernet(key)
    return f.encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str, key: bytes) -> str:
    """Decrypt a ciphertext string produced by :func:`encrypt`."""
    from cryptography.fernet import Fernet
    f = Fernet(key)
    return f.decrypt(ciphertext.encode()).decode()


# ---------------------------------------------------------------------------
# VaultSession — per-document session object
# ---------------------------------------------------------------------------


@dataclass
class VaultSession:
    """Holds the encryption key and token map for one redaction session."""

    key: bytes = field(default_factory=generate_key)

    # plaintext → stable token (e.g. "123-45-6789" → "[SSN_1]")
    _token_map: dict[str, str] = field(default_factory=dict)

    # plaintext → encrypted ciphertext (for de-anonymization)
    _cipher_store: dict[str, str] = field(default_factory=dict)

    # counter per entity type for stable token numbering
    _counters: dict[str, int] = field(default_factory=dict)

    def redact(self, plaintext: str, entity_type: str) -> str:
        """Return the stable replacement token for *plaintext*.

        First call for a given plaintext assigns a new token and encrypts the
        value.  Subsequent calls return the same token (idempotent).
        """
        if plaintext in self._token_map:
            return self._token_map[plaintext]

        # Assign next counter for this entity type
        n = self._counters.get(entity_type, 0) + 1
        self._counters[entity_type] = n
        token = f"[{entity_type}_{n}]"

        self._token_map[plaintext] = token
        self._cipher_store[plaintext] = encrypt(plaintext, self.key)
        return token

    def recover(self, token: str) -> Optional[str]:
        """Return the original plaintext for a token, or None if unknown."""
        reverse = {v: k for k, v in self._token_map.items()}
        plaintext = reverse.get(token)
        return plaintext

    def export(self, path: Path) -> None:
        """Write the encrypted vault to *path* as JSON.

        The file contains only ciphertexts — the key is NOT included and must
        be stored separately (e.g. printed to stdout for the operator).
        """
        payload = {
            "tokens": self._token_map,
            "store": self._cipher_store,
        }
        path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def load(cls, path: Path, key: bytes) -> "VaultSession":
        """Reconstruct a VaultSession from an exported vault file + key."""
        payload = json.loads(path.read_text())
        session = cls(key=key)
        session._token_map = payload["tokens"]
        session._cipher_store = payload["store"]
        # Rebuild counters from existing tokens
        for token in payload["tokens"].values():
            # token format: [ENTITY_TYPE_N]
            inner = token.strip("[]")
            parts = inner.rsplit("_", 1)
            if len(parts) == 2 and parts[1].isdigit():
                etype, n = parts[0], int(parts[1])
                session._counters[etype] = max(session._counters.get(etype, 0), n)
        return session
