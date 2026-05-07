"""Per-document audit trail writer.

Writes one JSON file per processed document to the configured audit log
directory.  PII text values are NEVER written unless ``include_text=True``
is explicitly passed (development use only).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pii_redact.models import DocumentResult


def write_audit_json(
    result: DocumentResult,
    log_dir: Path,
    include_text: bool = False,
) -> Path:
    """Write a structured audit JSON file for *result*.

    The file is written atomically (write to a temp name, then rename) to
    avoid partial reads if the process is interrupted.

    Args:
        result:       Completed pipeline result for one document.
        log_dir:      Directory to write the audit file into (created if
                      absent).
        include_text: If True, raw PII text is included in each box entry.
                      NEVER enable in production — for developer debugging
                      only.

    Returns:
        Path to the written audit JSON file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Build the audit record
    page_summaries = []
    for pr in result.page_results:
        page_summaries.append(
            {
                "page": pr.page,
                "doc_type": pr.doc_type.value,
                "redaction_count": len(pr.redaction_boxes),
                "processing_ms": round(pr.processing_ms, 2),
                "boxes": [b.as_dict(include_text=include_text) for b in pr.redaction_boxes],
            }
        )

    audit_record = {
        "schema_version": "1.0",
        "doc_id": result.doc_id,
        "filename": result.filename,
        "doc_type": result.doc_type.value,
        "pages": result.pages,
        "processed_at": datetime.now(timezone.utc).isoformat(),
        "total_redactions": len(result.all_boxes),
        "total_processing_ms": round(result.total_processing_ms, 2),
        "entity_counts": result.entity_counts,
        "source_counts": result.source_counts,
        "page_results": page_summaries,
        "_include_text": include_text,  # flag so reviewers know what was logged
    }

    output_path = log_dir / f"audit_{result.doc_id}.json"
    tmp_path = output_path.with_suffix(".json.tmp")

    tmp_path.write_text(
        json.dumps(audit_record, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # Atomic rename
    tmp_path.replace(output_path)

    return output_path
