"""NLP-layer benchmark: Presidio (spaCy NER) + regex on TAB and Gretel Finance.

WARNING: This script loads spaCy + torch. Run standalone, never import from
the fast test suite or combine with CUDA-sensitive code in the same process.

Usage:
    CUDA_VISIBLE_DEVICES="" python scripts/run_benchmark_nlp.py --dataset gretel
    CUDA_VISIBLE_DEVICES="" python scripts/run_benchmark_nlp.py --dataset tab
    CUDA_VISIBLE_DEVICES="" python scripts/run_benchmark_nlp.py --dataset all

Outputs JSON results and a per-entity F1 table.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Disable GPU — this benchmark does not need CUDA and we don't want driver init
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, str(Path(__file__).parent.parent))

# ---------------------------------------------------------------------------
# Label normalisation: map Gretel/TAB entity labels to Presidio entity types
# ---------------------------------------------------------------------------

LABEL_MAP: dict[str, str] = {
    # Gretel Finance labels → Presidio equivalents
    "NAME": "PERSON",
    "FIRST_NAME": "PERSON",
    "LAST_NAME": "PERSON",
    "COMPANY": "ORG",
    "STREET_ADDRESS": "LOCATION",
    "LOCAL_LATLNG": "LOCATION",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "EMAIL": "EMAIL_ADDRESS",
    "DATE": "DATE_TIME",
    "DATE_OF_BIRTH": "DATE_TIME",
    "DATE_TIME": "DATE_TIME",
    "SSN": "US_SSN",
    "BANK_ROUTING_NUMBER": "US_BANK_NUMBER",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "ZIP": "ZIP_CODE",
    "IPV4": "IP_ADDRESS",
    "IPV6": "IP_ADDRESS",
    "EIN": "NRP",      # closest Presidio type; custom recognizer covers EIN
    "NPI": "NRP",
    # TAB labels
    "PERSON": "PERSON",
    "ORG": "ORG",
    "ORGANIZATION": "ORG",   # Gretel sometimes uses full word
    "LOC": "LOCATION",
    "DEM": "NRP",
    "CODE": "NRP",
    "DATETIME": "DATE_TIME",
    "QUANTITY": "NRP",
    "MISC": "NRP",
}


def normalise_label(raw: str) -> str:
    return LABEL_MAP.get(raw.upper(), raw.upper())


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_f1(predicted: list[str], gold: list[str]) -> dict:
    tp = sum(1 for p in predicted if p in gold)
    fp = len(predicted) - tp
    fn = sum(1 for g in gold if g not in predicted)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f1, 3),
            "tp": tp, "fp": fp, "fn": fn}


# ---------------------------------------------------------------------------
# Presidio + regex combined NER
# ---------------------------------------------------------------------------

_analyzer = None


def _get_analyzer():
    global _analyzer
    if _analyzer is None:
        print("[NLP] Loading Presidio AnalyzerEngine (first call) ...")
        from pii_redact.ner.presidio_setup import build_analyzer_engine
        _analyzer = build_analyzer_engine()
        print("[NLP] Presidio ready.")
    return _analyzer


def run_nlp_ner(text: str, min_score: float = 0.6) -> list[tuple[str, str]]:
    """Run Presidio NER + custom insurance recognizers on text.

    Returns [(normalised_entity_type, span_text), ...]
    """
    analyzer = _get_analyzer()
    try:
        results = analyzer.analyze(text=text, language="en")
    except Exception as exc:
        print(f"  [WARN] Presidio error: {exc}", file=sys.stderr)
        results = []

    findings: list[tuple[str, str]] = []
    for r in results:
        if r.score < min_score:
            continue
        span = text[r.start:r.end]
        findings.append((r.entity_type, span))
    return findings


# ---------------------------------------------------------------------------
# TAB benchmark
# ---------------------------------------------------------------------------


def run_tab(data_dir: Path, max_docs: int, min_score: float = 0.6) -> dict:
    from datasets import load_from_disk

    print(f"\n[TAB/NLP] Loading from {data_dir} ...")
    try:
        ds = load_from_disk(str(data_dir))
        split = ds["test"] if "test" in ds else ds
    except Exception as exc:
        return {"error": str(exc)}

    all_pred: list[str] = []
    all_gold: list[str] = []
    entity_stats: dict[str, dict] = {}
    n_docs = 0
    t0 = time.monotonic()

    for example in split:
        if n_docs >= max_docs:
            break
        text = example.get("text") or ""
        if not text:
            continue

        gold_entities: list[tuple[str, str]] = []
        for em in example.get("entity_mentions", []):
            if em.get("confidential_status", "") != "NOT_CONFIDENTIAL":
                span = em.get("span_text") or text[em["start_offset"]:em["end_offset"]]
                gold_entities.append((normalise_label(em.get("entity_type", "UNKNOWN")), span))

        pred_entities = [(normalise_label(et), sp) for et, sp in run_nlp_ner(text, min_score)]

        gold_texts = [t for _, t in gold_entities]
        pred_texts = [t for _, t in pred_entities]
        all_gold.extend(gold_texts)
        all_pred.extend(pred_texts)

        for etype, etext in gold_entities:
            entity_stats.setdefault(etype, {"gold": [], "pred": []})
            entity_stats[etype]["gold"].append(etext)
        for etype, etext in pred_entities:
            entity_stats.setdefault(etype, {"gold": [], "pred": []})
            entity_stats[etype]["pred"].append(etext)

        n_docs += 1
        if n_docs % 50 == 0:
            print(f"  [{n_docs}/{min(max_docs, len(split))} docs]")

    elapsed = time.monotonic() - t0
    overall = compute_f1(all_pred, all_gold)
    per_entity = {et: compute_f1(s["pred"], s["gold"]) for et, s in entity_stats.items()}

    print(f"[TAB/NLP] {n_docs} docs | {elapsed:.1f}s | F1={overall['f1']:.3f}")
    return {"dataset": "TAB_NLP", "n_docs": n_docs, "elapsed_s": round(elapsed, 1),
            "overall": overall, "per_entity": per_entity}


# ---------------------------------------------------------------------------
# Gretel benchmark
# ---------------------------------------------------------------------------


def run_gretel(data_dir: Path, max_docs: int, min_score: float = 0.6) -> dict:
    from datasets import load_from_disk

    print(f"\n[Gretel/NLP] Loading from {data_dir} ...")
    try:
        ds = load_from_disk(str(data_dir))
        split = ds if not hasattr(ds, "keys") else ds[list(ds.keys())[0]]
    except Exception as exc:
        return {"error": str(exc)}

    all_pred: list[str] = []
    all_gold: list[str] = []
    entity_stats: dict[str, dict] = {}
    n_docs = 0
    t0 = time.monotonic()

    for example in split:
        if n_docs >= max_docs:
            break
        text = example.get("generated_text") or example.get("text") or ""
        if not text:
            continue

        raw_spans = example.get("pii_spans", [])
        if isinstance(raw_spans, str):
            try:
                raw_spans = json.loads(raw_spans)
            except Exception:
                raw_spans = []

        gold_entities: list[tuple[str, str]] = []
        for span in raw_spans:
            s, e = span.get("start", 0), span.get("end", 0)
            label = normalise_label(span.get("label", "UNKNOWN"))
            gold_entities.append((label, text[s:e]))

        pred_entities = [(normalise_label(et), sp) for et, sp in run_nlp_ner(text, min_score)]

        gold_texts = [t for _, t in gold_entities]
        pred_texts = [t for _, t in pred_entities]
        all_gold.extend(gold_texts)
        all_pred.extend(pred_texts)

        for etype, etext in gold_entities:
            entity_stats.setdefault(etype, {"gold": [], "pred": []})
            entity_stats[etype]["gold"].append(etext)
        for etype, etext in pred_entities:
            entity_stats.setdefault(etype, {"gold": [], "pred": []})
            entity_stats[etype]["pred"].append(etext)

        n_docs += 1
        if n_docs % 50 == 0:
            print(f"  [{n_docs}/{max_docs} docs]")

    elapsed = time.monotonic() - t0
    overall = compute_f1(all_pred, all_gold)
    per_entity = {et: compute_f1(s["pred"], s["gold"]) for et, s in entity_stats.items()}

    print(f"[Gretel/NLP] {n_docs} docs | {elapsed:.1f}s | F1={overall['f1']:.3f}")
    return {"dataset": "Gretel_NLP", "n_docs": n_docs, "elapsed_s": round(elapsed, 1),
            "overall": overall, "per_entity": per_entity}


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------


def print_table(results: dict) -> None:
    name = results.get("dataset", "?")
    print(f"\n{'='*62}")
    print(f"  {name} Results")
    print(f"{'='*62}")
    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return
    ov = results["overall"]
    print(f"  Overall  P={ov['precision']:.3f}  R={ov['recall']:.3f}  F1={ov['f1']:.3f}")
    print(f"  TP={ov['tp']}  FP={ov['fp']}  FN={ov['fn']}  ({results['elapsed_s']}s)")
    print()
    print(f"  {'Entity':<22} {'P':>6} {'R':>6} {'F1':>6}")
    print(f"  {'-'*44}")
    for et, m in sorted(results.get("per_entity", {}).items(),
                        key=lambda x: -x[1]["f1"]):
        if m["f1"] > 0 or m["tp"] + m["fp"] + m["fn"] > 0:
            print(f"  {et:<22} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")
    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["tab", "gretel", "all"], default="gretel")
    parser.add_argument("--data-dir", default="./data/benchmarks")
    parser.add_argument("--max-docs", type=int, default=100)
    parser.add_argument("--min-score", type=float, default=0.6,
                        help="Presidio minimum confidence score (default 0.6)")
    parser.add_argument("--output", default="./data/benchmarks/results_nlp.json")
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    all_results: list[dict] = []

    if args.dataset in ("tab", "all"):
        r = run_tab(data_root / "tab", args.max_docs, min_score=args.min_score)
        print_table(r)
        all_results.append(r)

    if args.dataset in ("gretel", "all"):
        r = run_gretel(data_root / "gretel_finance", args.max_docs, min_score=args.min_score)
        print_table(r)
        all_results.append(r)

    if all_results:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(all_results, indent=2))
        print(f"Results → {out}")


if __name__ == "__main__":
    main()
