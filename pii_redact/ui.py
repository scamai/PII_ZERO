"""Gradio local web UI for pii-redact.

Runs on http://127.0.0.1:7860 — fully local, never shares to Gradio cloud.

Tabs:
  1. Redact       — upload → redact → preview + download
  2. Inspect      — dry-run with annotated detection overlay
  3. Vault/Restore — restore original from the encrypted vault
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from pii_redact.pipeline import PIIRedactionPipeline
    from pii_redact.models import DocumentResult, RedactionBox

logger = logging.getLogger(__name__)

# Entity type → BGR color for annotation overlay
_ENTITY_COLORS: dict[str, tuple[int, int, int]] = {
    "PERSON": (0, 0, 255),
    "SSN": (0, 128, 255),
    "PHONE_NUMBER": (0, 200, 255),
    "EMAIL_ADDRESS": (0, 255, 200),
    "CREDIT_CARD": (200, 0, 255),
    "POLICY_NUM": (255, 128, 0),
    "FACE": (255, 0, 0),
    "LICENSE_PLATE": (0, 255, 0),
    "QR_CODE": (255, 255, 0),
    "BARCODE": (180, 255, 0),
    "HANDWRITING": (255, 180, 0),
    "LOCATION": (100, 255, 100),
    "DATE_TIME": (100, 100, 255),
    "NRP": (200, 200, 0),
}
_DEFAULT_COLOR = (200, 100, 200)


def _color_for(entity_type: str) -> tuple[int, int, int]:
    return _ENTITY_COLORS.get(entity_type, _DEFAULT_COLOR)


def _first_page_image(file_path: Path) -> np.ndarray | None:
    """Return a BGR numpy array of the first page of *file_path*, or None on failure."""
    import cv2

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        try:
            import fitz

            doc = fitz.open(str(file_path))
            page = doc[0]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
            arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
            doc.close()
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception as exc:
            logger.warning("PDF preview failed: %s", exc)
            return None
    else:
        img = cv2.imread(str(file_path))
        return img


def _annotate_image(image: np.ndarray, boxes: list[RedactionBox]) -> np.ndarray:
    """Draw colored bounding boxes with entity labels onto *image*."""
    import cv2

    annotated = image.copy()
    for box in boxes:
        color = _color_for(box.entity_type)
        x1, y1 = box.x, box.y
        x2, y2 = box.x + box.w, box.y + box.h
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness=2)
        label = f"{box.entity_type} {box.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - th - 4)), (x1 + tw + 2, y1), color, -1)
        cv2.putText(
            annotated,
            label,
            (x1 + 1, max(th, y1 - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return annotated


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _boxes_to_dataframe_rows(result: DocumentResult) -> list[list[Any]]:
    """Convert all RedactionBoxes in *result* to rows for gr.Dataframe."""
    rows = []
    for box in result.all_boxes:
        rows.append(
            [
                box.entity_type,
                round(box.confidence, 3),
                box.source,
                box.page,
                f"{box.x},{box.y},{box.w},{box.h}",
            ]
        )
    return rows


# ---------------------------------------------------------------------------
# Tab handlers
# ---------------------------------------------------------------------------


def _handle_redact(pipeline: PIIRedactionPipeline, upload):
    """Gradio handler for the Redact tab."""
    if upload is None:
        return None, None, {"error": "No file uploaded."}, None

    input_path = Path(upload.name)

    # Write redacted copy to a temp file so we can offer it for download
    with tempfile.NamedTemporaryFile(
        suffix=input_path.suffix, delete=False, prefix="redacted_"
    ) as tmp:
        output_path = Path(tmp.name)

    try:
        result = pipeline.process(input_path, output_path)
    except Exception as exc:
        logger.error("Redact handler error: %s", exc, exc_info=True)
        return None, None, {"error": str(exc)}, None

    # Build preview images (first page only)
    orig_img = _first_page_image(input_path)
    orig_rgb = _bgr_to_rgb(orig_img) if orig_img is not None else None

    redacted_img = _first_page_image(output_path)
    redacted_rgb = _bgr_to_rgb(redacted_img) if redacted_img is not None else None

    entities_json = {
        "doc_type": result.doc_type,
        "pages": result.pages,
        "total_detections": len(result.all_boxes),
        "entity_counts": result.entity_counts,
        "detections": [b.as_dict() for b in result.all_boxes],
    }

    return orig_rgb, redacted_rgb, entities_json, str(output_path)


def _handle_inspect(pipeline: PIIRedactionPipeline, upload):
    """Gradio handler for the Inspect tab."""
    if upload is None:
        return None, []

    input_path = Path(upload.name)

    try:
        result = pipeline.inspect(input_path)
    except Exception as exc:
        logger.error("Inspect handler error: %s", exc, exc_info=True)
        return None, []

    # Annotated preview (first page)
    img = _first_page_image(input_path)
    annotated_rgb = None
    if img is not None:
        page0_boxes = [b for b in result.all_boxes if b.page == 0]
        annotated = _annotate_image(img, page0_boxes)
        annotated_rgb = _bgr_to_rgb(annotated)

    rows = _boxes_to_dataframe_rows(result)
    return annotated_rgb, rows


def _handle_restore(vault, upload, key_hex: str):
    """Gradio handler for the Vault/Restore tab."""
    if upload is None:
        return None, []
    if not key_hex or not key_hex.strip():
        return None, [["Error", "Vault key is required", "", "", ""]]

    input_path = Path(upload.name)

    with tempfile.NamedTemporaryFile(
        suffix=input_path.suffix, delete=False, prefix="restored_"
    ) as tmp:
        output_path = Path(tmp.name)

    try:
        if vault is None:
            from pii_redact.config import load_settings

            settings = load_settings()
            vault_db = Path(settings.vault.get("db_path", "./vault/vault.db"))
            from pii_redact.vault import PIIVault  # type: ignore[import]

            _vault = PIIVault(vault_db, key_hex=key_hex.strip())
        else:
            _vault = vault

        entries = _vault.restore(input_path, output_path)
        rows = [
            [e.get("entity_type", ""), e.get("page", 0), e.get("bbox", ""), "", ""]
            for e in (entries or [])
        ]
        return str(output_path), rows
    except Exception as exc:
        logger.error("Restore handler error: %s", exc, exc_info=True)
        return None, [["Error", str(exc), "", "", ""]]


# ---------------------------------------------------------------------------
# UI builder
# ---------------------------------------------------------------------------


def launch_ui(
    pipeline: PIIRedactionPipeline,
    vault=None,
    port: int = 7860,
) -> None:
    """Build and launch the Gradio local web UI.

    Parameters
    ----------
    pipeline:
        A fully-initialised PIIRedactionPipeline.
    vault:
        Optional PIIVault instance for the restore tab.
    port:
        Local TCP port to bind. Default 7860.
    """
    import gradio as gr

    # ------------------------------------------------------------------
    # Tab 1 — Redact
    # ------------------------------------------------------------------
    with gr.Blocks(title="pii-redact") as app:
        gr.Markdown("# pii-redact — Offline PII Anonymization")

        with gr.Tab("Redact"):
            gr.Markdown("Upload a document to detect and redact all PII regions.")
            with gr.Row():
                upload_redact = gr.File(label="Upload document")
                btn_redact = gr.Button("Redact", variant="primary")

            with gr.Row():
                img_original = gr.Image(label="Original preview", type="numpy")
                img_redacted = gr.Image(label="Redacted output", type="numpy")

            json_entities = gr.JSON(label="Detected entities")
            file_download = gr.File(label="Download redacted file")

            btn_redact.click(
                fn=lambda f: _handle_redact(pipeline, f),
                inputs=[upload_redact],
                outputs=[img_original, img_redacted, json_entities, file_download],
            )

        # ------------------------------------------------------------------
        # Tab 2 — Inspect (dry run)
        # ------------------------------------------------------------------
        with gr.Tab("Inspect (dry run)"):
            gr.Markdown(
                "Upload a document to see what *would* be redacted — no files are modified."
            )
            with gr.Row():
                upload_inspect = gr.File(label="Upload document")
                btn_inspect = gr.Button("Inspect", variant="secondary")

            img_annotated = gr.Image(label="Annotated detections", type="numpy")
            df_detections = gr.Dataframe(
                label="Detections",
                headers=["Entity Type", "Confidence", "Source", "Page", "Location"],
                datatype=["str", "number", "str", "number", "str"],
                interactive=False,
            )

            btn_inspect.click(
                fn=lambda f: _handle_inspect(pipeline, f),
                inputs=[upload_inspect],
                outputs=[img_annotated, df_detections],
            )

        # ------------------------------------------------------------------
        # Tab 3 — Vault / Restore
        # ------------------------------------------------------------------
        with gr.Tab("Vault / Restore"):
            gr.Markdown(
                "Restore original PII values from the encrypted vault.\n\n"
                "The vault key must match the key used during redaction "
                "(`PII_VAULT_KEY` environment variable)."
            )
            with gr.Row():
                upload_restore = gr.File(label="Upload redacted file")
                key_input = gr.Textbox(
                    label="Vault key (hex)", type="password", placeholder="64-character hex key"
                )
            btn_restore = gr.Button("Restore original", variant="stop")

            file_restored = gr.File(label="Restored original")
            df_regions = gr.Dataframe(
                label="Stored regions",
                headers=["Entity Type", "Page", "BBox", "", ""],
                interactive=False,
            )

            btn_restore.click(
                fn=lambda f, k: _handle_restore(vault, f, k),
                inputs=[upload_restore, key_input],
                outputs=[file_restored, df_regions],
            )

    app.launch(
        server_name="127.0.0.1",
        server_port=port,
        share=False,
        show_error=True,
    )
