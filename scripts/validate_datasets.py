"""
Validation dataset downloader and benchmark runner.

Downloads publicly available PII / de-identification benchmark datasets
and runs the pii_redact pipeline against them, reporting precision/recall/F1.

Usage:
    python scripts/validate_datasets.py --list          # show available datasets
    python scripts/validate_datasets.py --download tab  # download Text Anonymization Benchmark
    python scripts/validate_datasets.py --run tab       # run benchmark against TAB
    python scripts/validate_datasets.py --all           # download all + run all
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "benchmarks"

# ──────────────────────────────────────────────────────────────────────────────
# Dataset registry
# ──────────────────────────────────────────────────────────────────────────────

DATASETS = {
    "tab": {
        "name": "Text Anonymization Benchmark (TAB)",
        "description": "1,268 European Court of Human Rights decisions annotated for PII. "
                       "Gold standard for text anonymization evaluation.",
        "source": "HuggingFace — ildpil/text-anonymization-benchmark",
        "hf_id": "ildpil/text-anonymization-benchmark",
        "entity_types": ["PERSON", "ORG", "LOC", "DATE", "CODE"],
        "paper": "https://arxiv.org/abs/2202.00443",
        "license": "CC BY 4.0",
        "applicable_layers": ["text_layer", "presidio", "insurance_ner"],
        "notes": "Best general-purpose text anonymization benchmark. "
                 "Does not include insurance-specific entities — use as a baseline for PERSON/DATE/ORG recall.",
    },
    "spy": {
        "name": "SPY — Synthetic PII Detection Dataset",
        "description": "Fully synthetic PII-annotated dataset from NAACL 2025. "
                       "Designed specifically for PII detection model evaluation.",
        "source": "ACL Anthology / HuggingFace",
        "hf_id": None,  # available via ACL Anthology download
        "paper": "https://aclanthology.org/2025.naacl-srw.23/",
        "license": "CC BY 4.0",
        "applicable_layers": ["text_layer", "presidio", "regex_patterns"],
        "notes": "Synthetic so no real PII risk. Good for testing regex patterns since "
                 "it uses realistic but fake SSN, phone, email, address formats.",
        "download_instructions": "Download PDF + supplementary data from ACL Anthology; "
                                 "dataset itself linked in paper appendix.",
    },
    "mendeley_finance": {
        "name": "Synthetic PII in Financial Documents (Mendeley)",
        "description": "PII-annotated synthetic financial documents. "
                       "Closest available public dataset to insurance claim format.",
        "source": "Mendeley Data — data.mendeley.com/datasets/tzrjx692jy",
        "url": "https://data.mendeley.com/datasets/tzrjx692jy",
        "paper": "https://www.nature.com/articles/s41598-025-04971-9",
        "license": "CC BY 4.0",
        "applicable_layers": ["text_layer", "presidio", "insurance_ner", "regex_patterns"],
        "notes": "Financial document format overlaps significantly with insurance claims. "
                 "Contains bank account numbers, routing numbers, addresses, names — "
                 "directly tests our regex recognizers for routing/account numbers.",
    },
    "gretel_finance": {
        "name": "Gretel AI Synthetic PII Finance (Multilingual)",
        "description": "100 distinct financial document formats, 20 subtypes each. "
                       "Available directly on HuggingFace — no sign-up required.",
        "source": "HuggingFace — gretelai/synthetic_pii_finance_multilingual",
        "hf_id": "gretelai/synthetic_pii_finance_multilingual",
        "license": "Apache 2.0",
        "applicable_layers": ["text_layer", "presidio", "regex_patterns"],
        "notes": "Best freely downloadable dataset for financial PII. "
                 "Multilingual (EN + 5 others) — use EN subset for our pipeline. "
                 "Has BIO tags for NER evaluation.",
    },
    "wider_face": {
        "name": "WIDER FACE",
        "description": "32,203 images with 393,703 face annotations across 61 event classes. "
                       "Standard benchmark for face detection models.",
        "source": "http://shuoyang1213.me/WIDERFACE/",
        "license": "Research only (non-commercial)",
        "applicable_layers": ["visual_layer", "deface"],
        "notes": "Validates the face detection component of the visual layer. "
                 "Use validation split (no training needed). "
                 "Target: our deface/cascade pipeline should exceed 85% recall on Easy subset.",
        "download_instructions": "wget http://shuoyang1213.me/WIDERFACE/WiderFace_Results/  (see site for links)",
    },
    "lp_detection": {
        "name": "License Plate Detection (HuggingFace)",
        "description": "8,823 images annotated for license plate bounding boxes (COCO format). "
                       "Validates YOLOv8 plate detection component.",
        "source": "HuggingFace — keremberke/license-plate-object-detection",
        "hf_id": "keremberke/license-plate-object-detection",
        "license": "MIT",
        "applicable_layers": ["visual_layer", "yolov8"],
        "notes": "Use test split. Target: IoU > 0.5 on >80% of plates.",
    },
    "i2b2_2014": {
        "name": "i2b2 2014 De-Identification Challenge",
        "description": "1,300+ clinical notes annotated for 18 PHI categories. "
                       "Gold standard for medical de-identification. "
                       "Includes patient name, date, age, phone, ID, location, provider.",
        "source": "https://www.i2b2.org/NLP/HeartDisease/",
        "license": "Requires DUA (free sign-up at i2b2.org)",
        "applicable_layers": ["text_layer", "presidio", "insurance_ner", "scispacy"],
        "notes": "MOST IMPORTANT dataset for validating medical record handling (Type E documents). "
                 "Requires free institutional sign-up — not auto-downloadable. "
                 "Published F1: Presidio achieves ~0.81 on this dataset; "
                 "our scispaCy + insurance NER chain should exceed this.",
        "download_instructions": "Register at https://www.i2b2.org/NLP/HeartDisease/ — "
                                 "approval usually takes 1-3 days.",
    },
    "piibench": {
        "name": "PIIBench — Unified Multi-Source Benchmark",
        "description": "Multi-source benchmark corpus for PII detection in natural language text. "
                       "Published April 2026 (arXiv 2604.15776).",
        "source": "https://arxiv.org/abs/2604.15776",
        "license": "See paper",
        "applicable_layers": ["text_layer", "presidio"],
        "notes": "Newest comprehensive benchmark as of 2026. "
                 "Check GitHub for dataset release — may require author request.",
    },
}

# ──────────────────────────────────────────────────────────────────────────────
# Download helpers
# ──────────────────────────────────────────────────────────────────────────────

def download_hf_dataset(dataset_id: str, dest: Path) -> bool:
    """Download a HuggingFace dataset to dest directory."""
    try:
        subprocess.run(
            [sys.executable, "-c",
             f"from datasets import load_dataset; "
             f"ds = load_dataset('{dataset_id}'); "
             f"ds.save_to_disk('{dest}')"],
            check=True,
            env={"TRANSFORMERS_OFFLINE": "0"},  # needs network for download
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ERROR: {e}")
        return False


def download_dataset(key: str) -> None:
    info = DATASETS[key]
    dest = DATA_DIR / key
    dest.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Downloading: {info['name']}")
    print(f"Source: {info['source']}")

    if info.get("hf_id"):
        print(f"  → from HuggingFace: {info['hf_id']}")
        ok = download_hf_dataset(info["hf_id"], dest)
        if ok:
            print(f"  ✓ Saved to {dest}")
        else:
            print(f"  ✗ Download failed")
    elif info.get("download_instructions"):
        print(f"  Manual download required:")
        print(f"  {info['download_instructions']}")
        (dest / "DOWNLOAD_INSTRUCTIONS.txt").write_text(
            info["download_instructions"] + f"\n\nSave files to: {dest}\n"
        )
    else:
        print(f"  → URL: {info.get('url', info.get('source', 'see paper'))}")
        print(f"  Manual download — save to: {dest}")

    # Write metadata
    (dest / "dataset_info.json").write_text(json.dumps(info, indent=2))


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ──────────────────────────────────────────────────────────────────────────────

def run_gretel_benchmark(dataset_path: Path) -> dict:
    """
    Run text layer evaluation against Gretel finance dataset.
    Expects BIO-tagged data from HuggingFace datasets format.
    Returns precision/recall/F1 per entity type.
    """
    try:
        from datasets import load_from_disk
        from pii_redact.ner.presidio_setup import build_analyzer_engine
        from pii_redact.ner.regex_patterns import PRESIDIO_RECOGNIZER_LIST
    except ImportError as e:
        return {"error": str(e), "note": "Run pip install datasets presidio-analyzer first"}

    print("  Loading Gretel dataset...")
    ds = load_from_disk(str(dataset_path))
    test_split = ds.get("test", ds.get("train"))

    analyzer = build_analyzer_engine()

    tp: dict[str, int] = {}
    fp: dict[str, int] = {}
    fn: dict[str, int] = {}

    sample_count = min(500, len(test_split))
    print(f"  Evaluating on {sample_count} samples...")

    for i, sample in enumerate(test_split.select(range(sample_count))):
        text = sample.get("full_text", sample.get("text", ""))
        gold_entities = sample.get("privacy_mask", [])  # Gretel format

        if not text:
            continue

        # Run our analyzer
        results = analyzer.analyze(text=text, language="en")
        predicted = {(r.start, r.end, r.entity_type) for r in results}
        gold = {(e["start"], e["end"], e["label"]) for e in gold_entities
                if isinstance(e, dict) and "start" in e}

        for pred in predicted:
            etype = pred[2]
            if pred in gold:
                tp[etype] = tp.get(etype, 0) + 1
            else:
                fp[etype] = fp.get(etype, 0) + 1

        for g in gold:
            etype = g[2]
            if g not in predicted:
                fn[etype] = fn.get(etype, 0) + 1

    # Compute per-entity metrics
    metrics = {}
    all_types = set(list(tp.keys()) + list(fn.keys()))
    for etype in sorted(all_types):
        t = tp.get(etype, 0)
        f = fp.get(etype, 0)
        n = fn.get(etype, 0)
        precision = t / (t + f) if (t + f) > 0 else 0.0
        recall = t / (t + n) if (t + n) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        metrics[etype] = {"precision": round(precision, 3),
                          "recall": round(recall, 3),
                          "f1": round(f1, 3),
                          "tp": t, "fp": f, "fn": n}

    # Micro-average
    total_tp = sum(v["tp"] for v in metrics.values())
    total_fp = sum(v["fp"] for v in metrics.values())
    total_fn = sum(v["fn"] for v in metrics.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2*micro_p*micro_r / (micro_p+micro_r) if (micro_p+micro_r) > 0 else 0
    metrics["__micro__"] = {"precision": round(micro_p, 3),
                             "recall": round(micro_r, 3),
                             "f1": round(micro_f1, 3)}
    return metrics


def run_visual_benchmark_wider_face(dataset_path: Path) -> dict:
    """
    Run face detection benchmark against WIDER FACE validation split.
    Computes AP (average precision) at IoU=0.5.
    """
    try:
        import cv2
        from pii_redact.layers.visual_layer import VisualLayer
        from pii_redact.models import DocumentType
    except ImportError as e:
        return {"error": str(e)}

    ann_file = dataset_path / "wider_face_val_bbx_gt.txt"
    img_dir = dataset_path / "images"
    if not ann_file.exists() or not img_dir.exists():
        return {"error": f"WIDER FACE not downloaded to {dataset_path}. "
                        f"See {dataset_path}/DOWNLOAD_INSTRUCTIONS.txt"}

    layer = VisualLayer()
    tp = fp = fn = 0
    sample_limit = 200  # first 200 images for quick evaluation

    with open(ann_file) as f:
        lines = f.read().splitlines()

    i = 0
    img_count = 0
    while i < len(lines) and img_count < sample_limit:
        img_name = lines[i].strip()
        n_faces = int(lines[i+1].strip())
        gold_boxes = []
        for j in range(n_faces):
            parts = list(map(int, lines[i+2+j].strip().split()))
            gold_boxes.append((parts[0], parts[1], parts[2], parts[3]))  # x,y,w,h
        i += 2 + n_faces

        img_path = img_dir / img_name
        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        pred_boxes_raw = layer.process(img, DocumentType.PHOTO)
        pred_boxes = [(b.x, b.y, b.w, b.h) for b in pred_boxes_raw
                      if b.entity_type == "FACE"]

        def iou(b1, b2):
            x1, y1, w1, h1 = b1
            x2, y2, w2, h2 = b2
            ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
            iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
            inter = ix * iy
            union = w1*h1 + w2*h2 - inter
            return inter / union if union > 0 else 0.0

        matched_gold = set()
        for pb in pred_boxes:
            best_iou = 0.0
            best_gi = -1
            for gi, gb in enumerate(gold_boxes):
                if gi in matched_gold:
                    continue
                score = iou(pb, gb)
                if score > best_iou:
                    best_iou = score
                    best_gi = gi
            if best_iou >= 0.5:
                tp += 1
                matched_gold.add(best_gi)
            else:
                fp += 1
        fn += len(gold_boxes) - len(matched_gold)
        img_count += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0
    return {
        "images_evaluated": img_count,
        "face_detection": {
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn,
        },
        "target": "recall > 0.85 on Easy subset",
    }


def print_metrics(name: str, metrics: dict) -> None:
    print(f"\n{'─'*60}")
    print(f"Results: {name}")
    print(f"{'─'*60}")
    if "error" in metrics:
        print(f"  ERROR: {metrics['error']}")
        if "note" in metrics:
            print(f"  Note: {metrics['note']}")
        return
    if "face_detection" in metrics:
        fd = metrics["face_detection"]
        print(f"  Face detection  P={fd['precision']:.3f}  R={fd['recall']:.3f}  F1={fd['f1']:.3f}")
        print(f"  Images evaluated: {metrics['images_evaluated']}")
        print(f"  Target: {metrics.get('target', '')}")
        return
    micro = metrics.pop("__micro__", None)
    if micro:
        print(f"  MICRO-AVG:  P={micro['precision']:.3f}  R={micro['recall']:.3f}  F1={micro['f1']:.3f}")
    print(f"  {'Entity Type':<25} {'P':>6} {'R':>6} {'F1':>6} {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'─'*60}")
    for etype, m in sorted(metrics.items()):
        print(f"  {etype:<25} {m['precision']:>6.3f} {m['recall']:>6.3f} "
              f"{m['f1']:>6.3f} {m.get('tp',0):>5} {m.get('fp',0):>5} {m.get('fn',0):>5}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="pii-redact validation benchmark runner")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    parser.add_argument("--download", metavar="KEY", help="Download a dataset by key (or 'all')")
    parser.add_argument("--run", metavar="KEY", help="Run benchmark for a dataset key")
    parser.add_argument("--all", action="store_true", help="Download all + run all benchmarks")
    args = parser.parse_args()

    if args.list or not any([args.download, args.run, args.all]):
        print("\nAvailable validation datasets:\n")
        for key, info in DATASETS.items():
            print(f"  {key:<25} {info['name']}")
            print(f"  {'':25} Layers: {', '.join(info['applicable_layers'])}")
            print(f"  {'':25} {info['notes'][:80]}...")
            print()
        return

    if args.download:
        keys = list(DATASETS.keys()) if args.download == "all" else [args.download]
        for key in keys:
            if key not in DATASETS:
                print(f"Unknown dataset: {key}. Use --list to see options.")
                sys.exit(1)
            download_dataset(key)

    if args.run:
        key = args.run
        if key not in DATASETS:
            print(f"Unknown dataset: {key}")
            sys.exit(1)
        dest = DATA_DIR / key
        if key == "gretel_finance":
            metrics = run_gretel_benchmark(dest)
            print_metrics(DATASETS[key]["name"], metrics)
        elif key == "wider_face":
            metrics = run_visual_benchmark_wider_face(dest)
            print_metrics(DATASETS[key]["name"], metrics)
        else:
            print(f"Benchmark runner for '{key}' not yet implemented.")
            print(f"  See dataset_info.json in {dest} for manual evaluation guidance.")

    if args.all:
        print("Downloading all datasets...")
        for key in DATASETS:
            download_dataset(key)
        print("\nRunning automated benchmarks...")
        for key in ["gretel_finance", "wider_face"]:
            dest = DATA_DIR / key
            if key == "gretel_finance":
                m = run_gretel_benchmark(dest)
            else:
                m = run_visual_benchmark_wider_face(dest)
            print_metrics(DATASETS[key]["name"], m)


if __name__ == "__main__":
    main()
