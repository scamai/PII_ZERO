"""SQLite-backed audit log for redaction events.

Each event records WHAT was redacted (entity type, source, confidence) but
NEVER the raw text_found — that stays in the vault only.

Schema:
    audit_events(
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        ts          TEXT NOT NULL,          -- ISO-8601 UTC timestamp
        doc_id      TEXT NOT NULL,
        filename    TEXT,
        entity_type TEXT NOT NULL,
        source      TEXT NOT NULL,          -- "regex" | "presidio" | "insurance_ner" | ...
        page        INTEGER NOT NULL,
        confidence  REAL NOT NULL
    )
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS audit_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT    NOT NULL,
    doc_id      TEXT    NOT NULL,
    filename    TEXT,
    entity_type TEXT    NOT NULL,
    source      TEXT    NOT NULL,
    page        INTEGER NOT NULL,
    confidence  REAL    NOT NULL
)
"""


def _connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def write_event(db_path: str, event: dict[str, Any]) -> None:
    """Append one redaction event to the audit log at *db_path*.

    ``event`` must contain: doc_id, entity_type, source, page, confidence.
    ``filename`` is optional.  ``text_found`` is explicitly EXCLUDED — it must
    never appear in the audit log.
    """
    ts = datetime.now(timezone.utc).isoformat()
    row = (
        ts,
        event["doc_id"],
        event.get("filename"),
        event["entity_type"],
        event["source"],
        int(event["page"]),
        float(event["confidence"]),
    )

    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(_CREATE_TABLE)
        cursor.execute(
            """INSERT INTO audit_events
               (ts, doc_id, filename, entity_type, source, page, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            row,
        )
        conn.commit()


def read_events(db_path: str) -> list[dict[str, Any]]:
    """Return all audit events as a list of dicts, ordered by id."""
    with _connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(_CREATE_TABLE)
        cursor.execute("SELECT * FROM audit_events ORDER BY id")
        rows = cursor.fetchall()
    return [dict(row) for row in rows]
