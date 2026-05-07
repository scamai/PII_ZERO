#!/usr/bin/env python3
"""Pre-fetch all model weights once (requires internet).

Run this script ONCE before switching to an offline environment:
    python scripts/download_models.py

After this script completes successfully, set these env vars and the
full pipeline runs 100% offline:
    TRANSFORMERS_OFFLINE=1
    HF_DATASETS_OFFLINE=1
    HF_HUB_OFFLINE=1
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Environment — point all HuggingFace caches to ./models/trocr so they're
# inside the repo and survive a Docker COPY instruction.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "models"

os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR / "trocr")
os.environ["HF_HOME"] = str(MODELS_DIR / "trocr")
os.environ["HF_HUB_CACHE"] = str(MODELS_DIR / "trocr")

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Track outcomes
_downloaded: list[str] = []
_manual: list[str] = []
_failed: list[str] = []


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# a) PaddleOCR
# ---------------------------------------------------------------------------

_section("PaddleOCR (auto-download on first instantiation)")
try:
    from paddleocr import PaddleOCR  # type: ignore

    ocr_dir = MODELS_DIR / "paddleocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)

    print("Instantiating PaddleOCR — this triggers weight download …")
    _ = PaddleOCR(
        lang="en",
        use_gpu=False,
        show_log=False,
        det_model_dir=str(ocr_dir / "det"),
        rec_model_dir=str(ocr_dir / "rec"),
        cls_model_dir=str(ocr_dir / "cls"),
    )
    print("PaddleOCR weights downloaded.")
    _downloaded.append("PaddleOCR (det + rec + cls, English)")
except Exception as exc:
    print(f"WARNING: PaddleOCR download failed: {exc}")
    _failed.append(f"PaddleOCR: {exc}")


# ---------------------------------------------------------------------------
# b) TrOCR (microsoft/trocr-base-handwritten)
# ---------------------------------------------------------------------------

_section("TrOCR — microsoft/trocr-base-handwritten")
try:
    from huggingface_hub import snapshot_download  # type: ignore

    trocr_dir = MODELS_DIR / "trocr" / "microsoft--trocr-base-handwritten"
    trocr_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading TrOCR snapshot → {trocr_dir} …")
    snapshot_download(
        repo_id="microsoft/trocr-base-handwritten",
        local_dir=str(trocr_dir),
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )
    print("TrOCR downloaded.")
    _downloaded.append("TrOCR microsoft/trocr-base-handwritten (PyTorch weights)")
except Exception as exc:
    print(f"WARNING: TrOCR download failed: {exc}")
    _failed.append(f"TrOCR: {exc}")


# ---------------------------------------------------------------------------
# c) spaCy — en_core_web_lg
# ---------------------------------------------------------------------------

_section("spaCy — en_core_web_lg")
try:
    result = subprocess.run(
        [sys.executable, "-m", "spacy", "download", "en_core_web_lg"],
        check=True,
        capture_output=False,
    )
    print("en_core_web_lg downloaded.")
    _downloaded.append("spaCy en_core_web_lg")
except subprocess.CalledProcessError as exc:
    print(f"WARNING: spaCy en_core_web_lg download failed (exit {exc.returncode}).")
    _failed.append("spaCy en_core_web_lg")
except FileNotFoundError:
    print("WARNING: spacy not installed — skipping.")
    _failed.append("spaCy en_core_web_lg (spacy not installed)")


# ---------------------------------------------------------------------------
# d) YOLOv8n
# ---------------------------------------------------------------------------

_section("YOLOv8n — ultralytics/assets")
try:
    from ultralytics import YOLO  # type: ignore

    yolo_dir = MODELS_DIR / "yolov8"
    yolo_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading yolov8n.pt (triggers auto-download to CWD) …")
    # YOLO() downloads to CWD if not found; point it at models dir
    yolo_target = yolo_dir / "yolov8n.pt"
    if not yolo_target.exists():
        _ = YOLO("yolov8n.pt")  # downloads to CWD
        # Move the downloaded file into models/yolov8/
        cwd_pt = Path("yolov8n.pt")
        if cwd_pt.exists():
            shutil.copy(str(cwd_pt), str(yolo_target))
            cwd_pt.unlink()
            print(f"Moved yolov8n.pt → {yolo_target}")
        else:
            print("yolov8n.pt already downloaded to cache by ultralytics.")
    else:
        print(f"yolov8n.pt already exists at {yolo_target} — skipping.")
    _downloaded.append("YOLOv8n (yolov8n.pt)")
except Exception as exc:
    print(f"WARNING: YOLOv8 download failed: {exc}")
    _failed.append(f"YOLOv8n: {exc}")


# ---------------------------------------------------------------------------
# e) SAM (Segment Anything Model) — manual download required
# ---------------------------------------------------------------------------

_section("SAM — Segment Anything Model (MANUAL download required)")

SAM_MODELS = {
    "sam_vit_h_4b8939.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "sam_vit_l_0b3195.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "sam_vit_b_01ec64.pth": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}
sam_dir = MODELS_DIR / "sam"
sam_dir.mkdir(parents=True, exist_ok=True)

print("SAM weights are large (~2.4 GB for ViT-H). Download manually with wget:")
print()
for filename, url in SAM_MODELS.items():
    dest = sam_dir / filename
    if dest.exists():
        print(f"  [ALREADY PRESENT] {dest}")
        _downloaded.append(f"SAM {filename}")
    else:
        print(f"  wget -P {sam_dir}/ {url}")
        _manual.append(f"SAM {filename}  →  wget -P {sam_dir}/ {url}")

print()
print("After downloading, place files in:", sam_dir)


# ---------------------------------------------------------------------------
# f) scispaCy wheel — pip install instruction
# ---------------------------------------------------------------------------

_section("scispaCy — manual pip install")

SCISPACY_WHEEL = (
    "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/"
    "en_core_sci_lg-0.5.3.tar.gz"
)
print("scispaCy models must be installed via pip (not huggingface_hub).")
print()
print("Run the following commands to install scispaCy + the large biomedical model:")
print()
print("  pip install scispacy")
print(f"  pip install {SCISPACY_WHEEL}")
print()
print("Other available scispaCy models:")
print("  en_core_sci_sm  — small (~100 MB)")
print("  en_core_sci_md  — medium (~200 MB)")
print("  en_core_sci_lg  — large (~700 MB, recommended)")
print()
print("Visit https://allenai.github.io/scispacy/ for the full list.")
_manual.append(f"scispaCy en_core_sci_lg — pip install {SCISPACY_WHEEL}")


# ---------------------------------------------------------------------------
# Final checklist
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("  DOWNLOAD SUMMARY")
print("=" * 60)

if _downloaded:
    print("\nDOWNLOADED SUCCESSFULLY:")
    for item in _downloaded:
        print(f"  [x] {item}")

if _manual:
    print("\nREQUIRES MANUAL ACTION:")
    for item in _manual:
        print(f"  [ ] {item}")

if _failed:
    print("\nFAILED (check error messages above):")
    for item in _failed:
        print(f"  [!] {item}")

print()
if not _failed and not _manual:
    print("All models downloaded. Set these env vars for offline operation:")
    print("  export TRANSFORMERS_OFFLINE=1")
    print("  export HF_DATASETS_OFFLINE=1")
    print("  export HF_HUB_OFFLINE=1")
elif _failed:
    print("Some downloads failed. Re-run after fixing the errors above.")
    sys.exit(1)
else:
    print("Core models downloaded. Complete the manual steps above, then set:")
    print("  export TRANSFORMERS_OFFLINE=1")
    print("  export HF_DATASETS_OFFLINE=1")
    print("  export HF_HUB_OFFLINE=1")
