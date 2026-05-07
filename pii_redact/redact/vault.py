"""Encrypted vault for original pixel regions and audit log.

Stores encrypted copies of redacted image patches so that authorised
parties can restore the original content if needed.  All encryption uses
AES-256-GCM (authenticated encryption) via the ``cryptography`` library.

Key management
--------------
The 32-byte AES key must be supplied as a base64-encoded environment variable
whose name is ``settings.vault.key_env`` (default ``PII_VAULT_KEY``).

Generate a key once::

    python -c "import os, base64; print(base64.b64encode(os.urandom(32)).decode())"

Then::

    export PII_VAULT_KEY=<output>
"""

from __future__ import annotations

import base64
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from pii_redact.models import DocumentResult, RedactionBox


class PIIVault:
    """AES-256-GCM encrypted vault backed by SQLite.

    Args:
        settings: Loaded ``Settings`` object from ``pii_redact.config``.

    Raises:
        RuntimeError: If the required key environment variable is not set.
        ValueError:   If the key is not exactly 32 bytes after base64 decoding.
    """

    def __init__(self, settings) -> None:
        vault_cfg = settings.vault
        key_env: str = vault_cfg.key_env
        db_path: Path = Path(vault_cfg.db_path)

        # --- Load and validate key ---
        raw_key = os.environ.get(key_env)
        if raw_key is None:
            raise RuntimeError(
                f"PII vault key not found. "
                f"Set the environment variable '{key_env}' to a base64-encoded "
                f"32-byte AES-256 key.  "
                f"Generate one with: "
                f"python -c \"import os,base64; print(base64.b64encode(os.urandom(32)).decode())\""
            )
        key_bytes = base64.b64decode(raw_key)
        if len(key_bytes) != 32:
            raise ValueError(
                f"AES-256 key must be 32 bytes; got {len(key_bytes)} bytes "
                f"(check that {key_env} is a valid base64-encoded 32-byte key)."
            )
        self._aesgcm = AESGCM(key_bytes)

        # --- Open / create SQLite database ---
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS regions (
                region_id   TEXT PRIMARY KEY,
                doc_id      TEXT NOT NULL,
                page        INTEGER NOT NULL,
                x           INTEGER,
                y           INTEGER,
                w           INTEGER,
                h           INTEGER,
                entity_type TEXT,
                nonce       BLOB NOT NULL,
                ciphertext  BLOB NOT NULL,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id            TEXT NOT NULL,
                filename          TEXT,
                doc_type          TEXT,
                processed_at      TEXT NOT NULL,
                total_redactions  INTEGER,
                entity_counts     TEXT,   -- JSON
                source_counts     TEXT,   -- JSON
                processing_ms     REAL
            );
            """
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Region storage & retrieval
    # ------------------------------------------------------------------

    def store_region(
        self,
        doc_id: str,
        box: RedactionBox,
        pixel_data: bytes,
    ) -> str:
        """Encrypt and store a pixel region.

        Args:
            doc_id:     Unique document identifier (used as AAD).
            box:        The redaction box describing this region.
            pixel_data: Raw pixel bytes (e.g. PNG-encoded patch).

        Returns:
            A UUID string identifying this stored region.
        """
        region_id = str(uuid.uuid4())
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        ciphertext = self._aesgcm.encrypt(
            nonce,
            pixel_data,
            doc_id.encode(),  # authenticated additional data
        )
        created_at = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT INTO regions
                (region_id, doc_id, page, x, y, w, h, entity_type,
                 nonce, ciphertext, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                region_id,
                doc_id,
                box.page,
                box.x,
                box.y,
                box.w,
                box.h,
                box.entity_type,
                nonce,
                ciphertext,
                created_at,
            ),
        )
        self._conn.commit()
        return region_id

    def restore_region(self, doc_id: str, region_id: str) -> bytes:
        """Decrypt and return stored pixel bytes.

        Args:
            doc_id:    Must match the doc_id used during store_region().
            region_id: UUID returned by store_region().

        Returns:
            Original pixel bytes.

        Raises:
            KeyError:                        If region_id not found for doc_id.
            cryptography.exceptions.InvalidTag: If the key is wrong or data
                                             has been tampered with.
        """
        row = self._conn.execute(
            "SELECT nonce, ciphertext FROM regions WHERE region_id=? AND doc_id=?",
            (region_id, doc_id),
        ).fetchone()

        if row is None:
            raise KeyError(
                f"Region '{region_id}' not found for document '{doc_id}'."
            )

        return self._aesgcm.decrypt(
            bytes(row["nonce"]),
            bytes(row["ciphertext"]),
            doc_id.encode(),
        )

    # ------------------------------------------------------------------
    # Audit log
    # ------------------------------------------------------------------

    def log_document(self, result: DocumentResult) -> None:
        """Record a processed document in the audit log.

        Args:
            result: Completed DocumentResult from the pipeline.
        """
        self._conn.execute(
            """
            INSERT INTO audit_log
                (doc_id, filename, doc_type, processed_at,
                 total_redactions, entity_counts, source_counts, processing_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.doc_id,
                result.filename,
                result.doc_type.value,
                datetime.now(timezone.utc).isoformat(),
                len(result.all_boxes),
                json.dumps(result.entity_counts),
                json.dumps(result.source_counts),
                result.total_processing_ms,
            ),
        )
        self._conn.commit()

    def export_audit(self, doc_id: str | None = None) -> list[dict]:
        """Return audit log entries, optionally filtered by document ID.

        Args:
            doc_id: If given, return only entries for this document.

        Returns:
            List of row dicts (JSON-safe values; entity_counts and
            source_counts are parsed back to dicts).
        """
        if doc_id is not None:
            rows = self._conn.execute(
                "SELECT * FROM audit_log WHERE doc_id=? ORDER BY id",
                (doc_id,),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM audit_log ORDER BY id"
            ).fetchall()

        result = []
        for row in rows:
            entry = dict(row)
            # Parse JSON columns back to Python dicts
            for col in ("entity_counts", "source_counts"):
                if entry.get(col):
                    try:
                        entry[col] = json.loads(entry[col])
                    except (json.JSONDecodeError, TypeError):
                        pass
            result.append(entry)
        return result

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._conn.close()

    def __enter__(self) -> "PIIVault":
        return self

    def __exit__(self, *args) -> None:
        self.close()
