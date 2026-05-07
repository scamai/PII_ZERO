"""Offline PII redaction benchmark against TAB and Gretel Finance datasets.

Usage:
    python scripts/run_benchmark.py --dataset tab
    python scripts/run_benchmark.py --dataset gretel
    python scripts/run_benchmark.py --dataset all

Outputs a JSON results file and prints a per-entity F1 table.
Safe to run: no heavy ML libraries loaded at module level.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Keep project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------


def _compute_metrics(
    predicted: list[str], gold: list[str]
) -> dict[str, float]:
    """Compute precision, recall, F1 for entity token lists."""
    tp = sum(1 for p in predicted if p in gold)
    fp = len(predicted) - tp
    fn = sum(1 for g in gold if g not in predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _run_regex_ner(text: str) -> list[tuple[str, str]]:
    """Run regex-only NER (no presidio/torch). Returns [(entity_type, text), ...]."""
    import re

    # Inline patterns to avoid importing pii_redact.ner.regex_patterns (loads torch)
    PATTERNS: list[tuple[str, str]] = [
        ("SSN", r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b"),
        ("SSN", r"\b(?!000|666|9\d{2})\d{3}(?!00)\d{2}(?!0000)\d{4}\b"),
        ("PHONE", r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
        ("EMAIL", r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"),
        ("DATE", r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{1,2}[-/]\d{1,2})\b"),
        ("ZIP", r"\b\d{5}(?:-\d{4})?\b"),
        ("POLICY_NUM", r"\bPOL-[A-Z0-9]{7,}\b"),
        ("CLAIM_REF", r"\b(?:CLM|CLA|CLAIM)-[A-Z0-9]{6,15}\b"),
        ("NPI", r"\b[12]\d{9}\b"),
        ("EIN", r"\b(?!00)\d{2}-\d{7}\b"),
        ("ICD10_CODE", r"\b[A-TV-Z][0-9][0-9AB](?:\.[0-9A-TV-Z]{1,4})?\b"),
        ("DEA_NUM", r"\b[A-Z]{2}\d{7}\b"),
    ]

    findings: list[tuple[str, str]] = []
    for entity_type, pattern in PATTERNS:
        for m in re.finditer(pattern, text):
            findings.append((entity_type, m.group()))
    return findings


# ---------------------------------------------------------------------------
# TAB dataset benchmark
# ---------------------------------------------------------------------------


def _run_tab_benchmark(data_dir: Path) -> dict:
    """Run against the TAB (Text Anonymization Benchmark) dataset.

    TAB schema: text, entity_mentions[{start_offset, end_offset, entity_type,
    confidential_status, span_text}]
    We only evaluate confidential entities (the ones that should be redacted).
    """
    from datasets import load_from_disk  # type: ignore[import]

    print(f"\n[TAB] Loading from {data_dir} ...")
    try:
        ds = load_from_disk(str(data_dir))
        split = ds["test"] if "test" in ds else ds
    except Exception as exc:
        return {"error": str(exc)}

    all_predicted: list[str] = []
    all_gold: list[str] = []
    entity_stats: dict[str, dict] = {}
    n_docs = 0
    t0 = time.monotonic()

    for example in split:
        text = example.get("text") or ""
        if not text:
            continue

        # Gold: all entity mentions except NOT_CONFIDENTIAL
        # TAB statuses: ETHNIC, HEALTH, SEX, BELIEF, POLITICS = sensitive
        gold_entities: list[tuple[str, str]] = []
        for em in example.get("entity_mentions", []):
            if em.get("confidential_status", "") != "NOT_CONFIDENTIAL":
                span = em.get("span_text") or text[em["start_offset"]:em["end_offset"]]
                gold_entities.append((em.get("entity_type", "UNKNOWN"), span))

        predicted_entities = _run_regex_ner(text)

        gold_texts = [t for _, t in gold_entities]
        pred_texts = [t for _, t in predicted_entities]
        all_gold.extend(gold_texts)
        all_predicted.extend(pred_texts)

        for etype, etext in gold_entities:
            entity_stats.setdefault(etype, {"gold": [], "pred": []})
            entity_stats[etype]["gold"].append(etext)
        for etype, etext in predicted_entities:
            entity_stats.setdefault(etype, {"gold": [], "pred": []})
            entity_stats[etype]["pred"].append(etext)

        n_docs += 1

    elapsed = time.monotonic() - t0
    overall = _compute_metrics(all_predicted, all_gold)
    per_entity = {
        et: _compute_metrics(stats["pred"], stats["gold"])
        for et, stats in entity_stats.items()
    }

    print(f"[TAB] {n_docs} docs | {elapsed:.1f}s | overall F1={overall['f1']:.3f}")
    return {
        "dataset": "TAB",
        "n_docs": n_docs,
        "elapsed_s": round(elapsed, 2),
        "overall": overall,
        "per_entity": per_entity,
    }


# ---------------------------------------------------------------------------
# Gretel Finance benchmark
# ---------------------------------------------------------------------------


def _run_gretel_benchmark(data_dir: Path, max_docs: int = 200) -> dict:
    """Run against the Gretel AI finance PII dataset.

    Gretel schema: generated_text, pii_spans[{start, end, label}]
    Gold spans are string offsets into generated_text.
    """
    import json as _json
    from datasets import load_from_disk  # type: ignore[import]

    print(f"\n[Gretel] Loading from {data_dir} ...")
    try:
        ds = load_from_disk(str(data_dir))
        split = ds if not hasattr(ds, "keys") else ds[list(ds.keys())[0]]
    except Exception as exc:
        return {"error": str(exc)}

    all_predicted: list[str] = []
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

        # pii_spans is a JSON string or list
        raw_spans = example.get("pii_spans", [])
        if isinstance(raw_spans, str):
            try:
                raw_spans = _json.loads(raw_spans)
            except Exception:
                raw_spans = []

        gold_entities: list[tuple[str, str]] = []
        for span in raw_spans:
            s, e = span.get("start", 0), span.get("end", 0)
            label = span.get("label", "UNKNOWN").upper()
            gold_entities.append((label, text[s:e]))

        predicted_entities = _run_regex_ner(text)

        gold_texts = [t for _, t in gold_entities]
        pred_texts = [t for _, t in predicted_entities]
        all_gold.extend(gold_texts)
        all_predicted.extend(pred_texts)

        for etype, etext in gold_entities:
            entity_stats.setdefault(etype, {"gold": [], "pred": []})
            entity_stats[etype]["gold"].append(etext)
        for etype, etext in predicted_entities:
            entity_stats.setdefault(etype, {"gold": [], "pred": []})
            entity_stats[etype]["pred"].append(etext)

        n_docs += 1

    elapsed = time.monotonic() - t0
    overall = _compute_metrics(all_predicted, all_gold)
    per_entity = {
        et: _compute_metrics(stats["pred"], stats["gold"])
        for et, stats in entity_stats.items()
    }

    print(f"[Gretel] {n_docs} docs | {elapsed:.1f}s | overall F1={overall['f1']:.3f}")
    return {
        "dataset": "Gretel",
        "n_docs": n_docs,
        "elapsed_s": round(elapsed, 2),
        "overall": overall,
        "per_entity": per_entity,
    }


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------


def _print_table(results: dict) -> None:
    dataset = results.get("dataset", "?")
    print(f"\n{'='*60}")
    print(f"  {dataset} Benchmark Results")
    print(f"{'='*60}")

    if "error" in results:
        print(f"  ERROR: {results['error']}")
        return

    if "overall" in results:
        ov = results["overall"]
        print(f"  Overall  P={ov['precision']:.3f}  R={ov['recall']:.3f}  F1={ov['f1']:.3f}")
        print(f"  TP={ov['tp']}  FP={ov['fp']}  FN={ov['fn']}")
        print()
        print(f"  {'Entity':<20} {'P':>6} {'R':>6} {'F1':>6}")
        print(f"  {'-'*42}")
        for et, m in sorted(results.get("per_entity", {}).items()):
            print(f"  {et:<20} {m['precision']:>6.3f} {m['recall']:>6.3f} {m['f1']:>6.3f}")
    else:
        print(f"  Docs processed: {results.get('n_docs', '?')}")
        print(f"  Entities detected: {results.get('entities_detected', '?')}")
        print(f"  Per-doc avg: {results.get('entities_per_doc', '?')}")

    print(f"\n  Time: {results.get('elapsed_s', '?')}s")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="PII redaction benchmark runner")
    parser.add_argument(
        "--dataset",
        choices=["tab", "gretel", "all"],
        default="tab",
        help="Which dataset to benchmark against",
    )
    parser.add_argument(
        "--data-dir",
        default="./data/benchmarks",
        help="Root directory containing benchmark datasets",
    )
    parser.add_argument(
        "--output",
        default="./data/benchmarks/results.json",
        help="Where to write JSON results",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=200,
        help="Max docs to process per dataset (default 200)",
    )
    args = parser.parse_args()

    data_root = Path(args.data_dir)
    all_results: list[dict] = []

    if args.dataset in ("tab", "all"):
        tab_dir = data_root / "tab"
        if tab_dir.exists():
            r = _run_tab_benchmark(tab_dir)
            _print_table(r)
            all_results.append(r)
        else:
            print(f"[TAB] Data not found at {tab_dir} — skipping")

    if args.dataset in ("gretel", "all"):
        gretel_dir = data_root / "gretel_finance"
        if gretel_dir.exists():
            r = _run_gretel_benchmark(gretel_dir, max_docs=args.max_docs)
            _print_table(r)
            all_results.append(r)
        else:
            print(f"[Gretel] Data not found at {gretel_dir} — skipping")

    if all_results:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(all_results, indent=2))
        print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
