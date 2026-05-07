#!/usr/bin/env python3
"""Fine-tune a spaCy NER model on labeled insurance documents.

Prerequisites:
    1. Install spaCy:            pip install spacy
    2. Install spaCy training:   python -m spacy download en_core_web_lg
    3. Annotate data with Label Studio (instructions printed below).
    4. Export annotations as train.spacy and dev.spacy into ./data/

Usage:
    python scripts/train_insurance_ner.py [--config config.cfg] [--output ./models/insurance_ner]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
MODELS_DIR = REPO_ROOT / "models"
DEFAULT_OUTPUT = MODELS_DIR / "insurance_ner"
DEFAULT_CONFIG = REPO_ROOT / "config.cfg"

TRAIN_SPACY = DATA_DIR / "train.spacy"
DEV_SPACY = DATA_DIR / "dev.spacy"


# ---------------------------------------------------------------------------
# Label Studio annotation instructions
# ---------------------------------------------------------------------------

LABEL_STUDIO_INSTRUCTIONS = """
╔══════════════════════════════════════════════════════════════════╗
║           ANNOTATION INSTRUCTIONS — LABEL STUDIO               ║
╚══════════════════════════════════════════════════════════════════╝

1. Install Label Studio:
       pip install label-studio
       label-studio start

2. Create a new project with the following Labeling Configuration (NER):

   <View>
     <Labels name="label" toName="text">
       <Label value="PERSON"      background="#FF6B6B"/>
       <Label value="ORG"         background="#4ECDC4"/>
       <Label value="SSN"         background="#FFE66D"/>
       <Label value="POLICY_NUM"  background="#A8E6CF"/>
       <Label value="CLAIM_REF"   background="#DDA0DD"/>
       <Label value="NPI"         background="#98FB98"/>
       <Label value="EIN"         background="#F0E68C"/>
       <Label value="ICD10_CODE"  background="#87CEEB"/>
       <Label value="CPT_CODE"    background="#FFA500"/>
       <Label value="DEA_NUM"     background="#DEB887"/>
       <Label value="ADJUSTER_ID" background="#B0C4DE"/>
       <Label value="ROUTING_NUM" background="#BC8F8F"/>
       <Label value="DATE_OF_BIRTH" background="#90EE90"/>
       <Label value="ADDRESS"     background="#FFD700"/>
       <Label value="PHONE"       background="#E6E6FA"/>
       <Label value="EMAIL"       background="#ADD8E6"/>
       <Label value="PLATE"       background="#F5DEB3"/>
     </Labels>
     <Text name="text" value="$text"/>
   </View>

3. Import your insurance documents as plain text (one per task).
   Use the Label Studio API or UI to upload a JSONL file:
       [{"text": "Patient: John Doe SSN: 123-45-6789 ..."}]

4. Annotate spans by clicking and dragging over text, then
   selecting the entity type from the popup.

5. Export annotations:
   - Go to Export → DocBin (spaCy v3)
   - Or use the Label Studio → spaCy converter:
         label-studio-converter export --type spacy --input annotations.json \\
             --output data/ --split 0.8

6. Place the resulting files as:
       data/train.spacy
       data/dev.spacy

RECOMMENDED MINIMUM CORPUS SIZE:
   - 200 annotated documents for initial baseline
   - 500+ for production-quality NER
   - Each entity type should appear in at least 20 training examples

TIP: Use the existing regex patterns (pii_redact/ner/regex_patterns.py)
     to pre-annotate documents and speed up human review in Label Studio.
"""


def print_label_studio_instructions() -> None:
    print(LABEL_STUDIO_INSTRUCTIONS)


# ---------------------------------------------------------------------------
# Step a) Generate spaCy config
# ---------------------------------------------------------------------------


def generate_spacy_config(output_path: Path) -> None:
    """Run `python -m spacy init config` to scaffold a training config."""
    if output_path.exists():
        print(f"Config already exists at {output_path} — skipping init.")
        return

    print(f"Generating spaCy config → {output_path} …")
    cmd = [
        sys.executable, "-m", "spacy", "init", "config",
        str(output_path),
        "--lang", "en",
        "--pipeline", "ner",
        "--optimize", "accuracy",
        "--gpu", "-1",  # CPU
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        print("ERROR: spacy init config failed:")
        print(result.stderr)
        sys.exit(1)
    print(f"Config written to {output_path}")
    print()
    print("IMPORTANT: Review config.cfg and set vectors to 'en_core_web_lg' for")
    print("better initialisation. Look for [initialize.vectors] and set:")
    print("    vectors = ${paths.vectors}")
    print("    paths.vectors = 'en_core_web_lg'")


# ---------------------------------------------------------------------------
# Step b) Validate training data
# ---------------------------------------------------------------------------


def validate_data() -> None:
    """Check that train.spacy and dev.spacy exist and are non-empty."""
    missing = [p for p in [TRAIN_SPACY, DEV_SPACY] if not p.exists()]
    if missing:
        print("ERROR: Training data not found:")
        for p in missing:
            print(f"  {p}")
        print()
        print_label_studio_instructions()
        sys.exit(1)

    # Quick sanity-check: open with spaCy and count docs
    try:
        import spacy
        from spacy.tokens import DocBin

        for path in [TRAIN_SPACY, DEV_SPACY]:
            db = DocBin().from_disk(path)
            nlp_blank = spacy.blank("en")
            docs = list(db.get_docs(nlp_blank.vocab))
            print(f"  {path.name}: {len(docs)} documents")
            if len(docs) == 0:
                print(f"WARNING: {path.name} contains 0 documents.")
    except ImportError:
        print("spaCy not installed — skipping data validation.")


# ---------------------------------------------------------------------------
# Step c) Run training
# ---------------------------------------------------------------------------


def run_training(config_path: Path, output_dir: Path) -> None:
    """Invoke spaCy's CLI trainer."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nStarting training …")
    print(f"  Config:  {config_path}")
    print(f"  Output:  {output_dir}")
    print(f"  Train:   {TRAIN_SPACY}")
    print(f"  Dev:     {DEV_SPACY}")
    print()

    cmd = [
        sys.executable, "-m", "spacy", "train",
        str(config_path),
        "--output", str(output_dir),
        "--paths.train", str(TRAIN_SPACY),
        "--paths.dev", str(DEV_SPACY),
        "--gpu-id", "-1",   # CPU; set to 0 for GPU
    ]

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\nERROR: Training failed with exit code {result.returncode}.")
        sys.exit(result.returncode)

    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# Step d) Evaluate and print per-entity F1 scores
# ---------------------------------------------------------------------------


def evaluate_model(model_dir: Path) -> None:
    """Evaluate the best model on the dev set and print per-entity F1."""
    best_model = model_dir / "model-best"
    if not best_model.exists():
        print(f"WARNING: No model-best found at {best_model}. Skipping evaluation.")
        return

    print(f"\nEvaluating {best_model} on dev set …")

    try:
        import spacy
        from spacy.tokens import DocBin
        from spacy.scorer import Scorer

        nlp = spacy.load(str(best_model))
        db = DocBin().from_disk(DEV_SPACY)
        dev_docs = list(db.get_docs(nlp.vocab))

        # Run inference
        examples = []
        from spacy.training import Example
        for gold_doc in dev_docs:
            pred_doc = nlp(gold_doc.text)
            examples.append(Example(pred_doc, gold_doc))

        scorer = Scorer()
        scores = scorer.score(examples)

        # Per-entity scores live in scores["ents_per_type"]
        per_entity = scores.get("ents_per_type", {})

        print("\nPer-entity F1 scores (dev set):")
        print(f"  {'Entity':<20} {'P':>7} {'R':>7} {'F1':>7}")
        print(f"  {'-'*20} {'-'*7} {'-'*7} {'-'*7}")
        for entity, metrics in sorted(per_entity.items()):
            p  = metrics.get("p",  0.0)
            r  = metrics.get("r",  0.0)
            f1 = metrics.get("f",  0.0)
            print(f"  {entity:<20} {p:7.3f} {r:7.3f} {f1:7.3f}")

        overall_f1 = scores.get("ents_f", 0.0)
        overall_p  = scores.get("ents_p", 0.0)
        overall_r  = scores.get("ents_r", 0.0)
        print(f"\n  {'OVERALL':<20} {overall_p:7.3f} {overall_r:7.3f} {overall_f1:7.3f}")

        if overall_f1 < 0.7:
            print()
            print("WARNING: Overall F1 < 0.70. Consider:")
            print("  - Adding more training examples (aim for 500+)")
            print("  - Reviewing annotation consistency in Label Studio")
            print("  - Increasing training epochs in config.cfg (max_steps)")
            print("  - Using en_core_web_lg vectors for better initialisation")

    except ImportError as exc:
        print(f"Could not run evaluation (missing dependency): {exc}")
        print("Run manually:")
        print(f"  python -m spacy evaluate {best_model} {DEV_SPACY} --output eval.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune spaCy NER on labeled insurance documents."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to spaCy config file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output directory for trained model (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--label-studio",
        action="store_true",
        help="Print Label Studio annotation instructions and exit.",
    )
    args = parser.parse_args()

    if args.label_studio:
        print_label_studio_instructions()
        return

    print("=" * 60)
    print("  Insurance NER Fine-Tuning Pipeline")
    print("=" * 60)

    # a) Config
    print("\n[1/4] Generating spaCy config …")
    generate_spacy_config(args.config)

    # b) Data validation
    print("\n[2/4] Validating training data …")
    validate_data()

    # c) Training
    print("\n[3/4] Training …")
    run_training(args.config, args.output)

    # d) Evaluation
    print("\n[4/4] Evaluation …")
    evaluate_model(args.output)

    print("\nDone. Best model saved to:", args.output / "model-best")
    print("To use in the pipeline, update config/settings.yaml:")
    print(f"  ner:")
    print(f"    insurance_ner_model: {args.output / 'model-best'}")
    print()
    print("To view Label Studio annotation instructions, run:")
    print("  python scripts/train_insurance_ner.py --label-studio")


if __name__ == "__main__":
    main()
