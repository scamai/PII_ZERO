"""SROIE / CORD receipt image benchmark: Surya OCR → Presidio NER.

Measures OCR quality and NER entity detection on scanned receipt images.
Supports two datasets:
  - priyank-m/SROIE_2019_text_recognition  (word-level images; OCR-quality metric)
  - naver-clova-ix/cord-v2                 (full receipt images; NER metric)

WARNING: Loads Surya OCR models + spaCy. CPU-only is fine but slow.
Set CUDA_VISIBLE_DEVICES="" to prevent GPU driver init.

Usage::

    CUDA_VISIBLE_DEVICES="" python scripts/run_benchmark_sroie.py \\
        --dataset sroie --max-docs 30 \\
        --output data/benchmarks/results_sroie.json

    CUDA_VISIBLE_DEVICES="" python scripts/run_benchmark_sroie.py \\
        --dataset cord --max-docs 20 \\
        --output data/benchmarks/results_cord.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

# Disable GPU — prevent driver init and OOM surprises
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Label maps
# ---------------------------------------------------------------------------

SROIE_LABEL_MAP: dict[str, str] = {
    "company": "ORG",
    "date": "DATE_TIME",
    "address": "LOCATION",
    "total": "TOTAL",
}

CORD_LABEL_MAP: dict[str, str] = {
    "store_name": "ORG",
    "store_addr": "LOCATION",
    "total_price": "TOTAL",
    "tax_price": "DATE_TIME",   # not ideal, but lets us test numeric NER
}


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------


def _overlaps(a: str, b: str) -> bool:
    """True if either string is a non-empty substring of the other."""
    a, b = a.strip(), b.strip()
    return bool(a) and bool(b) and (a in b or b in a)


def compute_f1(
    predicted: list[str],
    gold: list[str],
    partial: bool = True,
) -> dict:
    """Compute precision, recall, F1.

    Parameters
    ----------
    predicted, gold:
        Lists of span text strings.
    partial:
        If True use substring containment as a match criterion.
    """
    if partial:
        tp = sum(1 for p in predicted if any(_overlaps(p, g) for g in gold))
        fn = sum(1 for g in gold if not any(_overlaps(g, p) for p in predicted))
    else:
        tp = sum(1 for p in predicted if p in gold)
        fn = sum(1 for g in gold if g not in predicted)
    fp = len(predicted) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "precision": round(prec, 3),
        "recall": round(rec, 3),
        "f1": round(f1, 3),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


# ---------------------------------------------------------------------------
# OCR helper (lazy-loaded)
# ---------------------------------------------------------------------------

_ocr_fn = None


def _get_ocr():
    """Return the ocr_image function, loading surya on first call."""
    global _ocr_fn
    if _ocr_fn is None:
        print("[OCR] Loading Surya OCR predictors …")
        from pii_redact.layers.surya_ocr import ocr_image
        _ocr_fn = ocr_image
        print("[OCR] Surya OCR ready.")
    return _ocr_fn


# ---------------------------------------------------------------------------
# NER helper (lazy-loaded)
# ---------------------------------------------------------------------------

_ner_analyzer = None
_vlm_extractor = None


def _get_analyzer():
    """Return the Presidio AnalyzerEngine, loading on first call."""
    global _ner_analyzer
    if _ner_analyzer is None:
        print("[NER] Loading Presidio AnalyzerEngine …")
        from pii_redact.ner.presidio_setup import build_analyzer_engine
        _ner_analyzer = build_analyzer_engine()
        print("[NER] Presidio ready.")
    return _ner_analyzer


def _get_vlm():
    """Return the VLMExtractor instance, loading Qwen3-VL-8B on first call."""
    global _vlm_extractor
    if _vlm_extractor is None:
        print("[VLM] Loading VLMExtractor (Qwen3-VL-8B) …")
        from pii_redact.ner.vlm_extractor import VLMExtractor
        _vlm_extractor = VLMExtractor()
        _vlm_extractor._load()
        device = getattr(_vlm_extractor, "_device", "unknown")
        print(f"[VLM] VLMExtractor ready. device={device}")
    return _vlm_extractor


def run_ner(text: str, min_score: float = 0.5) -> list[str]:
    """Return list of detected span strings."""
    analyzer = _get_analyzer()
    try:
        results = analyzer.analyze(text=text, language="en")
    except Exception as exc:
        logger.warning("Presidio error: %s", exc)
        return []
    return [text[r.start:r.end] for r in results if r.score >= min_score]


def run_ner_vlm(
    pil_image,
    ocr_text: str,
    min_score: float = 0.5,
    tmp_dir: Path | None = None,
) -> list[str]:
    """Run VLMExtractor + Presidio NER, merge results, dedup by text value.

    VLM spans are listed first (visual detection takes priority over OCR-text NER).
    Deduplication is case-insensitive so "TOTAL 99.00" and "total 99.00" count once.
    Temp PNG is cleaned up even if the VLM call raises.
    """
    import tempfile

    tmp_dir = tmp_dir or Path(tempfile.gettempdir())
    tmp_path = tmp_dir / f"_vlm_tmp_{os.getpid()}.png"

    try:
        pil_image.save(str(tmp_path), format="PNG")

        extractor = _get_vlm()
        try:
            vlm_boxes = extractor.extract(str(tmp_path), page=0)
        except Exception as exc:
            logger.warning("VLMExtractor.extract failed: %s", exc)
            vlm_boxes = []

        vlm_spans = [b.text_found for b in vlm_boxes if b.text_found.strip()]
        ner_spans = run_ner(ocr_text, min_score=min_score)

        seen: set[str] = set()
        merged: list[str] = []
        for span in vlm_spans + ner_spans:
            key = span.strip().lower()
            if key and key not in seen:
                seen.add(key)
                merged.append(span)
        return merged

    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# OCR quality metrics for word-level images (SROIE text recognition)
# ---------------------------------------------------------------------------


def _ocr_word_image(pil_img, ocr_fn) -> str:
    """OCR a small word-level image, return best recognised text."""
    import numpy as np

    img_array = np.array(pil_img.convert("RGB"))
    try:
        lines = ocr_fn(img_array)
    except Exception as exc:
        logger.warning("OCR failed: %s", exc)
        return ""
    # For word-level images there should be only 1 line
    texts = [ln["text"].strip() for ln in lines if ln["text"].strip()]
    return " ".join(texts)


def _cer(ref: str, hyp: str) -> float:
    """Character Error Rate (Levenshtein / len(ref)), returns 0 if ref empty."""
    if not ref:
        return 0.0
    import difflib
    ops = difflib.SequenceMatcher(None, ref, hyp).get_opcodes()
    errors = sum(
        max(i2 - i1, j2 - j1) if tag != "equal" else 0
        for tag, i1, i2, j1, j2 in ops
    )
    return errors / len(ref)


# ---------------------------------------------------------------------------
# SROIE benchmark (word-level text recognition)
# ---------------------------------------------------------------------------


def run_sroie(
    data_dir: Path,
    max_docs: int,
    min_score: float = 0.5,
    partial: bool = True,
    save_cache: bool = True,
) -> dict:
    """Benchmark OCR quality on SROIE word-level images.

    Since the SROIE text recognition dataset provides individual word images +
    ground-truth text (no named entity annotations), this benchmark measures:
      - CER (Character Error Rate) of Surya OCR vs ground-truth text
      - NER detection rate on the OCR'd text (what entity types are found)

    Uses streaming to avoid downloading the full 33k-example dataset before
    iterating.  The dataset is NOT saved to disk in streaming mode (too large);
    pass ``save_cache=False`` to skip the save step.
    """
    from datasets import load_dataset, load_from_disk

    cache_path = data_dir / "sroie"
    ds = None

    if cache_path.exists():
        print(f"[SROIE] Loading from cache: {cache_path}")
        try:
            ds = load_from_disk(str(cache_path))
        except Exception as exc:
            print(f"[SROIE] Cache load failed ({exc}), re-downloading with streaming …")

    if ds is None:
        print("[SROIE] Streaming from HuggingFace (no full download) …")
        try:
            # Use streaming=True so we iterate without downloading 33k images first.
            # save_to_disk is skipped for streaming datasets (not directly supported).
            ds = load_dataset(
                "priyank-m/SROIE_2019_text_recognition",
                split="train",
                streaming=True,
            )
        except Exception as exc:
            return {"error": f"Dataset load failed: {exc}"}

    ocr_fn = _get_ocr()

    n_docs = 0
    total_cer = 0.0
    ocr_errors = 0
    all_pred_spans: list[str] = []
    ner_entity_counts: dict[str, int] = {}
    t0 = time.monotonic()

    for row in ds:
        if n_docs >= max_docs:
            break

        pil_img = row.get("image")
        gold_text = (row.get("text") or "").strip()

        if pil_img is None:
            print(f"  [WARN] Row {n_docs}: missing image, skipping")
            continue

        # OCR the word image
        pred_text = _ocr_word_image(pil_img, ocr_fn)
        if not pred_text:
            ocr_errors += 1
            n_docs += 1
            continue

        # CER
        cer = _cer(gold_text.lower(), pred_text.lower())
        total_cer += cer

        # NER on the OCR'd text (contextual — limited for single words)
        pred_spans = run_ner(pred_text, min_score=min_score)
        all_pred_spans.extend(pred_spans)
        for span_text in pred_spans:
            ner_entity_counts["(any)"] = ner_entity_counts.get("(any)", 0) + 1

        n_docs += 1
        if n_docs % 10 == 0:
            avg_cer = total_cer / max(n_docs - ocr_errors, 1)
            print(f"  [{n_docs}/{max_docs}] avg_CER={avg_cer:.3f}  ocr_errors={ocr_errors}")

    elapsed = time.monotonic() - t0
    avg_cer = total_cer / max(n_docs - ocr_errors, 1)

    print(
        f"\n[SROIE] {n_docs} docs | {elapsed:.1f}s | avg_CER={avg_cer:.4f} "
        f"| ocr_errors={ocr_errors}"
    )

    return {
        "dataset": "SROIE_word_recognition",
        "n_docs": n_docs,
        "elapsed_s": round(elapsed, 1),
        "ocr_errors": ocr_errors,
        "avg_cer": round(avg_cer, 4),
        "ner_entity_counts": ner_entity_counts,
    }


# ---------------------------------------------------------------------------
# CORD benchmark (full receipt images with structured entity annotations)
# ---------------------------------------------------------------------------


def _extract_cord_gold_entities(gt: dict) -> list[tuple[str, str]]:
    """Extract gold (entity_type, value) pairs from CORD ground truth dict."""
    entities: list[tuple[str, str]] = []
    parse = gt.get("gt_parse", {})

    # total.total_price → TOTAL
    total = parse.get("total", {})
    if isinstance(total, dict):
        for k, v in total.items():
            if "total_price" in k and v:
                entities.append(("TOTAL", str(v)))
            elif "cash" in k.lower() and v:
                entities.append(("TOTAL", str(v)))

    # sub_total.* → TOTAL (financial amounts)
    sub_total = parse.get("sub_total", {})
    if isinstance(sub_total, dict):
        for k, v in sub_total.items():
            if v and ("price" in k or "total" in k):
                entities.append(("TOTAL", str(v)))

    return entities


def run_cord(
    data_dir: Path,
    max_docs: int,
    min_score: float = 0.5,
    partial: bool = True,
    save_cache: bool = True,
    use_vlm: bool = False,
) -> dict:
    """Benchmark OCR + NER on CORD full-receipt images.

    CORD provides full receipt images + structured entity labels
    (menus, totals, taxes).  This benchmark:
      - Runs Surya OCR on each receipt image
      - Runs Presidio NER on the OCR'd text
      - Compares detected amounts against gold total_price values
    """
    from datasets import load_dataset, load_from_disk

    cache_path = data_dir / "cord"
    ds = None

    if cache_path.exists():
        print(f"[CORD] Loading from cache: {cache_path}")
        try:
            ds = load_from_disk(str(cache_path))
        except Exception as exc:
            print(f"[CORD] Cache load failed ({exc}), re-downloading …")

    if ds is None:
        print("[CORD] Streaming from HuggingFace (no full download) …")
        try:
            ds = load_dataset("naver-clova-ix/cord-v2", split="train", streaming=True)
        except Exception as exc:
            return {"error": f"Dataset load failed: {exc}"}

    ocr_fn = _get_ocr()

    import numpy as np

    n_docs = 0
    ocr_errors = 0
    all_gold: list[str] = []
    all_pred: list[str] = []
    entity_stats: dict[str, dict] = {}
    ocr_line_counts: list[int] = []
    t0 = time.monotonic()

    for row in ds:
        if n_docs >= max_docs:
            break

        pil_img = row.get("image")
        if pil_img is None:
            print(f"  [WARN] Row {n_docs}: missing image, skipping")
            ocr_errors += 1
            n_docs += 1
            continue

        # Parse ground truth
        gt_raw = row.get("ground_truth", "{}")
        if isinstance(gt_raw, str):
            try:
                gt = json.loads(gt_raw)
            except Exception:
                gt = {}
        else:
            gt = gt_raw or {}

        gold_entities = _extract_cord_gold_entities(gt)

        # OCR the full receipt image
        try:
            img_array = np.array(pil_img.convert("RGB"))
            ocr_lines = ocr_fn(img_array)
        except Exception as exc:
            logger.warning("OCR failed on doc %d: %s", n_docs, exc)
            ocr_errors += 1
            n_docs += 1
            continue

        if not ocr_lines:
            logger.debug("No OCR lines for doc %d", n_docs)
            ocr_errors += 1
            n_docs += 1
            continue

        ocr_line_counts.append(len(ocr_lines))
        full_text = " ".join(
            ln["text"] for ln in ocr_lines if ln["text"].strip()
        )

        # NER on OCR'd text (+ VLM grounding if enabled)
        if use_vlm:
            pred_spans = run_ner_vlm(pil_img, full_text, min_score=min_score, tmp_dir=data_dir)
        else:
            pred_spans = run_ner(full_text, min_score=min_score)

        # Accumulate metrics
        gold_texts = [v for _, v in gold_entities]
        all_gold.extend(gold_texts)
        all_pred.extend(pred_spans)

        for etype, etext in gold_entities:
            entity_stats.setdefault(etype, {"gold": [], "pred": []})
            entity_stats[etype]["gold"].append(etext)

        # Map pred spans to TOTAL if they look like prices
        for span in pred_spans:
            entity_stats.setdefault("TOTAL", {"gold": [], "pred": []})
            entity_stats["TOTAL"]["pred"].append(span)

        n_docs += 1
        if n_docs % 5 == 0:
            avg_lines = sum(ocr_line_counts) / len(ocr_line_counts) if ocr_line_counts else 0
            print(f"  [{n_docs}/{max_docs}] avg_ocr_lines={avg_lines:.1f}  ocr_errors={ocr_errors}")

    elapsed = time.monotonic() - t0
    overall = compute_f1(all_pred, all_gold, partial=partial)
    per_entity = {
        et: compute_f1(s["pred"], s["gold"], partial=partial)
        for et, s in entity_stats.items()
    }
    avg_lines = sum(ocr_line_counts) / len(ocr_line_counts) if ocr_line_counts else 0
    mode = "partial" if partial else "exact"

    print(
        f"\n[CORD] {n_docs} docs | {elapsed:.1f}s | "
        f"F1={overall['f1']:.3f} ({mode}) | avg_ocr_lines={avg_lines:.1f}"
    )

    return {
        "dataset": "CORD_receipt",
        "n_docs": n_docs,
        "elapsed_s": round(elapsed, 1),
        "ocr_errors": ocr_errors,
        "avg_ocr_lines": round(avg_lines, 1),
        "match_mode": mode,
        "vlm_mode": use_vlm,
        "overall": overall,
        "per_entity": per_entity,
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------


def print_table(results: dict) -> None:
    name = results.get("dataset", "?")
    vlm_mode = results.get("vlm_mode", False)
    mode_label = "  [VLM+NER]" if vlm_mode else "  [NER]" if "overall" in results else ""
    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  {name}{mode_label}")
    print(sep)

    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return

    n = results.get("n_docs", 0)
    elapsed = results.get("elapsed_s", 0)
    errs = results.get("ocr_errors", 0)

    # SROIE word-recognition report
    if "avg_cer" in results:
        cer = results["avg_cer"]
        print(f"  Docs processed : {n}")
        print(f"  OCR errors     : {errs}")
        print(f"  Avg CER        : {cer:.4f}  ({(1 - cer) * 100:.1f}% char accuracy)")
        print(f"  Elapsed        : {elapsed}s")
        ner = results.get("ner_entity_counts", {})
        if ner:
            print(f"  NER detections : {sum(ner.values())}")
        print(sep)
        return

    # CORD / full-receipt report
    ov = results.get("overall", {})
    avg_lines = results.get("avg_ocr_lines", 0)
    print(f"  Docs processed : {n}")
    print(f"  OCR errors     : {errs}  |  Avg OCR lines : {avg_lines:.1f}")
    if ov:
        print(
            f"  Overall        : P={ov['precision']:.3f}  R={ov['recall']:.3f}"
            f"  F1={ov['f1']:.3f}  (TP={ov['tp']} FP={ov['fp']} FN={ov['fn']})"
        )
    print(f"  Elapsed        : {elapsed}s")
    per = results.get("per_entity", {})
    if per:
        print()
        print(f"  {'Entity':<22} {'P':>6} {'R':>6} {'F1':>6}  TP  FP  FN")
        print(f"  {'-' * 56}")
        for et, m in sorted(per.items(), key=lambda x: -x[1]["f1"]):
            if m["tp"] + m["fp"] + m["fn"] > 0:
                print(
                    f"  {et:<22} {m['precision']:>6.3f} {m['recall']:>6.3f} "
                    f"{m['f1']:>6.3f}  {m['tp']:>2}  {m['fp']:>2}  {m['fn']:>2}"
                )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark Surya OCR + Presidio NER on receipt images."
    )
    parser.add_argument(
        "--dataset",
        choices=["sroie", "cord", "all"],
        default="cord",
        help="Which dataset to benchmark (default: cord)",
    )
    parser.add_argument(
        "--data-dir",
        default="./data/benchmarks",
        help="Root directory for cached datasets and results",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=20,
        help="Maximum number of documents to process (default: 20)",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.5,
        help="Presidio minimum confidence score (default: 0.5)",
    )
    parser.add_argument(
        "--partial",
        action="store_true",
        default=True,
        help="Use partial/overlap span matching (default: True)",
    )
    parser.add_argument(
        "--exact",
        dest="partial",
        action="store_false",
        help="Use exact span matching instead of partial",
    )
    parser.add_argument(
        "--output",
        default="./data/benchmarks/results_sroie.json",
        help="Path to write JSON results",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Do not save dataset to disk cache",
    )
    parser.add_argument(
        "--vlm",
        action="store_true",
        default=False,
        help=(
            "Enable VLM-augmented NER on receipt images (CORD only). "
            "Requires Qwen3-VL-8B weights. Adds VLMExtractor grounding results "
            "to Presidio NER spans for A/B comparison."
        ),
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    data_root.mkdir(parents=True, exist_ok=True)
    save_cache = not args.no_cache

    all_results: list[dict] = []

    if args.dataset in ("sroie", "all"):
        r = run_sroie(
            data_root,
            max_docs=args.max_docs,
            min_score=args.min_score,
            partial=args.partial,
            save_cache=save_cache,
        )
        print_table(r)
        all_results.append(r)

    if args.dataset in ("cord", "all"):
        if args.vlm:
            os.environ["PII_USE_VLM"] = "1"
        r = run_cord(
            data_root,
            max_docs=args.max_docs,
            min_score=args.min_score,
            partial=args.partial,
            save_cache=save_cache,
            use_vlm=args.vlm,
        )
        print_table(r)
        all_results.append(r)

    if all_results:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, indent=2))
        print(f"\nResults written → {out}")


if __name__ == "__main__":
    main()
