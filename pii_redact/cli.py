"""Click-based CLI for pii-redact.

Entry point: pii-redact  (configured in pyproject.toml as pii_redact.cli:main)

Commands:
  redact           — redact a single file or entire directory
  inspect          — dry-run: show what would be redacted
  restore          — restore original values from the encrypted vault
  download-models  — print instructions for downloading model weights
  doctor           — check all required components are present
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import click


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_pipeline(config_path: str | None, vault_path: str | None):
    """Build a PIIRedactionPipeline from CLI options."""
    from pii_redact.config import load_settings
    from pii_redact.pipeline import PIIRedactionPipeline

    settings = load_settings(config_path) if config_path else load_settings()

    vault = None
    if vault_path:
        try:
            from pii_redact.vault import PIIVault  # type: ignore[import]

            vault = PIIVault(Path(vault_path))
        except Exception as exc:
            click.echo(f"Warning: could not open vault at {vault_path}: {exc}", err=True)

    return PIIRedactionPipeline(settings, vault=vault)


def _print_summary_table(results) -> None:
    """Print a compact results table to stdout."""
    header = f"{'File':<40} {'Type':<16} {'Boxes':>5} {'Time(ms)':>9} {'Status':<8}"
    click.echo(header)
    click.echo("-" * len(header))
    for r in results:
        status = "ok" if r.pages >= 0 else "ERROR"
        click.echo(
            f"{r.filename:<40} {r.doc_type:<16} {len(r.all_boxes):>5}"
            f" {r.total_processing_ms:>9.0f} {status:<8}"
        )


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """pii-redact — 100% offline PII anonymization for insurance claim documents."""


# ---------------------------------------------------------------------------
# redact
# ---------------------------------------------------------------------------


@main.command()
@click.argument("file_or_dir", type=click.Path(exists=True))
@click.option("--output", "-o", "output_path", default=None, help="Output file or directory.")
@click.option("--vault", "-v", "vault_path", default=None, help="Path to the PII vault database.")
@click.option("--dry-run", is_flag=True, default=False, help="Detect PII but do not write files.")
@click.option("--config", "-c", "config_path", default=None, help="Path to settings.yaml.")
def redact(
    file_or_dir: str,
    output_path: str | None,
    vault_path: str | None,
    dry_run: bool,
    config_path: str | None,
) -> None:
    """Redact PII from FILE_OR_DIR.

    When FILE_OR_DIR is a file, write the redacted version to --output (default:
    same directory with '_redacted' suffix).  When it is a directory, process all
    supported files and write results to --output directory.
    """
    pipeline = _make_pipeline(config_path, vault_path)
    path = Path(file_or_dir)
    any_error = False

    if path.is_dir():
        out_dir = Path(output_path) if output_path else path.parent / (path.name + "_redacted")
        results = pipeline.process_batch(path, out_dir)
        _print_summary_table(results)
        if any(r.pages == 0 for r in results):
            any_error = True
    else:
        if output_path:
            out = Path(output_path)
        else:
            out = path.parent / (path.stem + "_redacted" + path.suffix)

        result = pipeline.process(path, out, dry_run=dry_run)
        _print_summary_table([result])
        if result.pages == 0:
            any_error = True

    sys.exit(1 if any_error else 0)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@main.command()
@click.argument("file", type=click.Path(exists=True))
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "table"], case_sensitive=False),
    default="table",
    show_default=True,
    help="Output format.",
)
@click.option("--config", "-c", "config_path", default=None, help="Path to settings.yaml.")
def inspect(file: str, output_format: str, config_path: str | None) -> None:
    """Dry-run: detect PII in FILE without modifying anything."""
    from pii_redact.config import load_settings
    from pii_redact.pipeline import PIIRedactionPipeline

    settings = load_settings(config_path) if config_path else load_settings()
    pipeline = PIIRedactionPipeline(settings)
    result = pipeline.inspect(Path(file))

    if output_format == "json":
        click.echo(
            json.dumps(
                {
                    "doc_id": result.doc_id,
                    "filename": result.filename,
                    "doc_type": result.doc_type,
                    "pages": result.pages,
                    "detections": [b.as_dict() for b in result.all_boxes],
                },
                indent=2,
                default=str,
            )
        )
    else:
        # Pretty table
        click.echo(f"\nFile   : {result.filename}")
        click.echo(f"Type   : {result.doc_type}")
        click.echo(f"Pages  : {result.pages}")
        click.echo(f"Boxes  : {len(result.all_boxes)}\n")

        if result.all_boxes:
            col = f"{'Entity Type':<20} {'Conf':>5} {'Source':<16} {'Page':>4} {'BBox (x,y,w,h)'}"
            click.echo(col)
            click.echo("-" * len(col))
            for box in result.all_boxes:
                click.echo(
                    f"{box.entity_type:<20} {box.confidence:>5.2f} {box.source:<16}"
                    f" {box.page:>4}  {box.x},{box.y},{box.w},{box.h}"
                )
        else:
            click.echo("No PII detected.")


# ---------------------------------------------------------------------------
# restore
# ---------------------------------------------------------------------------


@main.command()
@click.argument("redacted_file", type=click.Path(exists=True))
@click.option("--vault", "-v", "vault_path", default=None, help="Path to vault database.")
@click.option("--output", "-o", "output_path", default=None, help="Destination for restored file.")
def restore(redacted_file: str, vault_path: str | None, output_path: str | None) -> None:
    """Restore the original PII values in REDACTED_FILE from the vault.

    Requires the PII_VAULT_KEY environment variable (AES-256-GCM hex key).
    """
    key_hex = os.environ.get("PII_VAULT_KEY")
    if not key_hex:
        click.echo(
            "Error: PII_VAULT_KEY environment variable is not set.\n"
            "Export the vault key as a 64-character hex string:\n"
            "  export PII_VAULT_KEY=<your-key>",
            err=True,
        )
        sys.exit(1)

    from pii_redact.config import load_settings

    settings = load_settings()
    if vault_path is None:
        vault_path = settings.vault.get("db_path", "./vault/vault.db")

    vault = None
    try:
        from pii_redact.vault import PIIVault  # type: ignore[import]

        vault = PIIVault(Path(vault_path), key_hex=key_hex)
    except ImportError:
        click.echo("Error: vault module not found.", err=True)
        sys.exit(1)
    except Exception as exc:
        click.echo(f"Error opening vault: {exc}", err=True)
        sys.exit(1)

    in_path = Path(redacted_file)
    out_path = Path(output_path) if output_path else in_path.parent / (in_path.stem + "_restored" + in_path.suffix)

    try:
        vault.restore(in_path, out_path)
        click.echo(f"Restored → {out_path}")
    except Exception as exc:
        click.echo(f"Restore failed: {exc}", err=True)
        sys.exit(1)


# ---------------------------------------------------------------------------
# download-models
# ---------------------------------------------------------------------------


@main.command("download-models")
def download_models() -> None:
    """Print instructions for downloading model weights and run the download script."""
    click.echo(
        "\nModel download instructions\n"
        "===========================\n"
        "\nRun the bundled download script to fetch all required model weights:\n"
        "\n    python scripts/download_models.py\n"
        "\nThis will download:\n"
        "  • PaddleOCR models (text recognition)\n"
        "  • TrOCR (handwriting recognition)\n"
        "  • YOLOv8n weights (license plate / object detection)\n"
        "  • SAM checkpoint (Segment Anything Model)\n"
        "  • spaCy en_core_web_lg (NER)\n"
        "  • scispaCy en_core_sci_lg (medical NER)\n"
        "  • Insurance NER fine-tuned model\n"
        "\nAll models are stored under ./models/ and used fully offline.\n"
        "\nTo run the download script now:\n"
    )
    import subprocess  # noqa: S404

    script = Path(__file__).parent.parent / "scripts" / "download_models.py"
    if script.exists():
        click.echo(f"Running {script} …\n")
        result = subprocess.run([sys.executable, str(script)], check=False)
        sys.exit(result.returncode)
    else:
        click.echo(f"Script not found at {script}. Create it or run manually.")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

_COMPONENTS = [
    # (label, import_check, weight_path_setting_key, critical)
    ("PyMuPDF (fitz)", "fitz", None, True),
    ("OpenCV", "cv2", None, True),
    ("PaddleOCR", "paddleocr", None, True),
    ("presidio-analyzer", "presidio_analyzer", None, True),
    ("spaCy NER", "spacy", "models.spacy_model", True),
    ("scispaCy model", "spacy", "models.scispacy_model", False),
    ("Insurance NER", "pii_redact.ner.insurance_ner", "models.insurance_ner_model", False),
    ("YOLOv8 weights", "ultralytics", "models.yolov8_weights", False),
    ("SAM checkpoint", None, "models.sam_checkpoint", False),
    ("TrOCR model dir", None, "models.trocr_dir", False),
    ("pyzbar", "pyzbar", None, False),
    ("piexif", "piexif", None, False),
    ("Gradio UI", "gradio", None, False),
    ("cryptography", "cryptography", None, True),
    ("click", "click", None, True),
]


@main.command()
def doctor() -> None:
    """Check that all required components and model files are present."""
    from pii_redact.config import load_settings

    settings = load_settings()

    header = f"{'Component':<30} {'Status':<12} Notes"
    click.echo("\n" + header)
    click.echo("-" * 72)

    all_critical_ok = True

    for label, import_name, weight_key, critical in _COMPONENTS:
        status_str = ""
        notes = ""
        ok = True

        # Check importability
        if import_name:
            try:
                __import__(import_name)
                status_str = "ok"
            except ImportError:
                status_str = "MISSING"
                notes = f"pip install {import_name.split('.')[0]}"
                ok = False

        # Check model weight / directory path
        if weight_key and ok:
            try:
                # Navigate nested keys like "models.yolov8_weights"
                parts = weight_key.split(".")
                val = settings
                for part in parts:
                    val = getattr(val, part)
                weight_path = Path(str(val))
                if weight_path.exists():
                    status_str = "ok"
                    notes = str(weight_path)
                else:
                    status_str = "missing weights"
                    notes = f"Run download-models  ({weight_path})"
                    ok = False
            except AttributeError:
                status_str = "config missing"
                notes = f"No key {weight_key!r} in settings"
                ok = False

        if not ok and critical:
            all_critical_ok = False

        icon = "✓" if ok else "✗"
        click.echo(f"{icon} {label:<28} {status_str:<12} {notes}")

    click.echo()
    if all_critical_ok:
        click.echo("All critical components are ready.")
        sys.exit(0)
    else:
        click.echo("One or more critical components are missing. See above.")
        sys.exit(1)
