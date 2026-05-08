# PII Removal Tool — Build Progress

_Last verified: 2026-05-08, S6 report_

---

## Architecture (current)

Four detection layers in sequence. Results unioned, deduplicated, rendered as solid black fills.

```
Document Image / PDF
  ├── Layer 1: Surya OCR  — line-level text + bounding boxes from scanned images
  ├── Layer 2: Text NER   — Presidio (spaCy + 17 custom PatternRecognizers) + optional GLiNER
  ├── Layer 3: Layout     — Docling field-label context (high-value / low-value field routing)
  └── Layer 4: Visual     — Qwen3-VL-8B grounding (bbox_2d JSON) + face/plate/QR detection
Post: Span NMS (text dedup + coordinate IoU) → confidence gate → solid-fill redaction
```

---

## Benchmark Results (canonical)

### Gretel Finance — text-layer NER path
Run: `CUDA_VISIBLE_DEVICES="" python scripts/run_benchmark_nlp.py --dataset gretel --max-docs 100 --partial`

#### Partial match (primary metric for redaction compliance)
> Partial match is correct for PII coverage: detecting "Springfield" when gold is "Springfield, IL" counts as a hit.

| Sprint | Change | Partial F1 |
|--------|--------|-----------|
| S0 | Regex-only baseline | 0.178 |
| S1 | +Presidio NLP (spaCy NER) | 0.604 |
| S2a | +ORG/PERSON entity filters (entity_filters.py) | 0.730 |
| S2b | +Surya OCR, GLiNER, Docling, NMS (4 parallel agents) | 0.732 |
| S3 | +CREDIT_CARD (0.65), SWIFT_BIC, US_BANK_NUMBER entities | 0.732 |
| S4 | +PHONE_NUMBER custom recognizer (R: 0.267→0.611) | 0.738 |
| S5 | +GLiNER benchmarked (PII_USE_GLINER=1) + phone precision fix + _validate_span | 0.772 |
| **S6** | **+ABA checksum (US_BANK P: 0.286→1.000) + ITIN/Passport recognizers + presidio-structured + name heuristic** | **0.771** |

Run command: `PII_USE_GLINER=1 CUDA_VISIBLE_DEVICES="" python scripts/run_benchmark_nlp.py --dataset gretel --max-docs 100 --partial`

**Latest run (2026-05-08, partial match, S6 with GLiNER):**

| Metric | Value |
|--------|-------|
| Precision | 0.798 |
| Recall | 0.745 |
| **F1** | **0.771** |
| TP / FP / FN | ~613 / ~156 / ~210 |
| Runtime | 1280s (100 docs, CPU GLiNER) |

> S6 overall F1 ≈ flat vs S5 (within noise). US_BANK_NUMBER precision 0.286→1.000 is the headline S6 gain.
> TransformersNlpEngine (`obi/deid_roberta_i2b2`) benchmarked but not adopted — clinical model, net F1 neutral on financial text.

**Latest run without GLiNER** (faster, 7.8s): P=0.811, R=0.677, F1=0.733

#### Per-entity breakdown (partial, S6 current)

_With `PII_USE_GLINER=1` (canonical / best quality):_

| Entity | P | R | F1 | Status |
|--------|---|---|----|--------|
| US_BANK_NUMBER | 1.000 | 1.000 | **1.000** | Fixed S6: ABA 3-7-1 checksum gate |
| EMAIL_ADDRESS | 1.000 | 0.933 | **0.966** | Excellent |
| IBAN_CODE | 1.000 | 0.917 | **0.957** | Excellent |
| IP_ADDRESS | 0.875 | 1.000 | **0.933** | Excellent |
| DATE_TIME | 0.891 | 0.895 | **0.893** | Excellent |
| PHONE_NUMBER | 0.739 | 0.850 | **0.791** | Strong (S5: precision fixed) |
| CREDIT_CARD | 0.667 | 0.667 | **0.667** | Good |
| LOCATION | 0.564 | 0.558 | 0.561 | Via GLiNER (was 0.382) |
| ORG | 0.556 | 0.532 | 0.543 | Via GLiNER (was 0.253) |
| PERSON | 0.664 | 0.452 | 0.538 | Recall gap; name heuristic added for Docling path |
| SWIFT_BIC_CODE | 0.200 | 0.500 | 0.286 | Context-gated (low base) |
| CREDIT_CARD_SECURITY_CODE | 0.000 | 0.000 | 0.000 | No Gretel examples with context keywords |
| US_ITIN | — | — | — | New S6 recognizer (not in Gretel dataset) |
| US_PASSPORT | — | — | — | New S6 recognizer (not in Gretel dataset) |

#### Exact match (secondary, for comparison)
| Run | Exact F1 |
|-----|---------|
| Baseline (cron agent) | 0.243 |
| S4 (2026-05-08, no GLiNER) | 0.359 (+48%) |
| S5 (2026-05-08, GLiNER) | ~0.41 (est.) |
| **S6 (2026-05-08, GLiNER + ABA)** | **TBD** |

### SROIE / CORD — image OCR path
Run: `python scripts/run_benchmark_sroie.py --dataset [sroie|cord] --max-docs N`

| Dataset | Metric | Value | Notes |
|---------|--------|-------|-------|
| SROIE (word crops) | CER | **8.49%** | Surya OCR on English receipt words |
| CORD (full receipts, NER only) | F1 | 0.18 | Indonesian receipts — expected low |
| CORD (full receipts, VLM+NER) | F1 | **0.099** | **Confirmed stable (2 runs, same result). VLM adds 47 FP, 6 TP. Dataset mismatch: CORD = Indonesian store receipts; VLM prompted for person names/SSNs not present.** |

---

## Test Suite

| File | Tests | Mark | Notes |
|------|-------|------|-------|
| test_models.py | ~15 | fast | RedactionBox / DocumentResult contracts |
| test_no_blur_safety.py | ~10 | fast | Pixel-level fill safety |
| test_redact_engine.py | ~30 | fast | Engine safety + PII-in-output guard |
| test_vault.py | ~12 | fast | Fernet encrypt/decrypt + audit log |
| test_entity_filters.py | ~34 | fast | ORG/PERSON structural quality filters |
| test_span_nms.py | ~10 | fast | Text-span + IoU dedup |
| test_gliner_recognizer.py | ~20 | fast | GLiNER zero-shot NER (mocked) |
| test_surya_ocr.py | ~10 | fast | Surya OCR wrapper (mocked predictors) |
| test_docling_parser.py | ~15 | fast | Docling layout parser (mocked) |
| test_benchmark_sroie.py | ~30 | fast | SROIE/CORD benchmark runner (mocked) |
| test_vlm_extractor.py | 20 | fast | VLM grounding parser (no model load) |
| test_benchmark_vlm.py | 15 | fast | VLM+NER benchmark merge/dedup logic |
| test_presidio_setup.py | 12 | slow | AnalyzerEngine singleton, reset, ITIN/Passport, StructuredEngine |
| **Total fast** | **206** | | `pytest -m fast` — safe anytime, ~8s |
| test_regex_patterns.py | ~17 | slow | Presidio import → spaCy → torch |
| test_smoke_claim_form.py | ~11 | slow | End-to-end CMS-1500 PDF pipeline |

**Test tier rules (critical — prevents OOM):**
- `pytest -m fast` — pure Python, no ML imports. Run anytime.
- `pytest -m slow` — loads torch+spaCy+Presidio. CI/Docker only, never background.
- Never run `pytest tests/` unmarked — loads all ML libraries simultaneously.

---

## Sprint History

### Sprint 0 — Scaffold
- Project structure, pyproject.toml, Docker, core data models
- 10 insurance-domain regex recognizers (SSN, NPI, EIN, ICD10, CPT, POLICY_NUM, ROUTING_NUM, ADJUSTER_ID, CLAIM_REF, DEA_NUM)
- Presidio AnalyzerEngine singleton builder with spaCy fallback cascade
- Click CLI skeleton, Gradio UI skeleton, Pydantic config
- Fernet vault + SQLite audit log
- Benchmark runner (scripts/run_benchmark.py) — TAB + Gretel Finance

### Sprint 1 — NER baseline
- Presidio NLP layer enabled: Gretel F1 0.178 → 0.604
- Label normalization (ORGANIZATION→ORG, TIME→DATE_TIME, IBAN→IBAN_CODE, etc.)
- IBAN_CODE F1=0.957 after mapping fix
- langdetect filter: drop PERSON/ORG/LOCATION in non-English ±100-char context
  - PERSON P: 0.273 → 0.653 (+139%)
- ORG structural quality filter (entity_filters.py):
  - Blocklist (role words, acronyms, XML artifacts, lowercase, overlong phrases)
  - ORG P: 0.056 → 0.245 exact, 0.129 → 0.415 partial
  - Overall F1: 0.604 → 0.730
- PERSON form-label filter (is_valid_person): drops "Email", "Phone", "Address" FPs
- End-to-end smoke test on synthetic CMS-1500 form (11 PII fields, 9 assertions)
- Hourly verification cron job (fast tests + Gretel benchmark + git log)

### Sprint 2 — OCR, GLiNER, Layout, NMS (4 parallel agents, merged)
- **Surya OCR** (`pii_redact/layers/surya_ocr.py`): replaces dead PaddleOCR path
  - API: DetectionPredictor + FoundationPredictor + RecognitionPredictor
  - Lazy model loading, `[]` on any failure
  - SROIE benchmark: CER=8.49% on English receipt words
- **GLiNER** (`pii_redact/ner/gliner_recognizer.py`): zero-shot NER, opt-in via `PII_USE_GLINER=1`
  - 18-entry label→entity map; model: `urchade/gliner_medium-v2.1`
  - Registered after spaCy in Presidio registry
- **Span NMS** (`pii_redact/redact/engine.py::deduplicate_boxes`): two-pass dedup
  - Pass A: text-span exact dedup by (page, text_found), entity specificity ranking
  - Pass B: coordinate IoU > 0.5 → suppress lower-confidence box
  - Wired into all 6 sub-pipeline exit points in pipeline.py
- **Docling layout parser** (`pii_redact/layers/docling_parser.py`)
  - LayoutSpan dataclass with text, bbox, page, field_label, is_high_value, is_low_value
  - Spatial heuristic: same-row span ending with ":" → field label
  - `is_high_value_field()`: Employer/Patient → bypass ORG gate; confidence floor 0.75
  - `is_low_value_field()`: Insurer/Payer → keep context gate
  - Docling BOTTOMLEFT → PyMuPDF TOPLEFT coordinate conversion
  - Falls back to PyMuPDF on parse failure

### Sprint 3 — Financial entity coverage
- **CREDIT_CARD**: base score 0.5 → 0.65 (passes min_score=0.6 without context)
  - F1: 0 → 0.667
- **SWIFT_BIC_CODE**: new recognizer, base score 0.40 (context-gated)
  - 8-char and 11-char patterns; initial 0.75/0.60 caused P=0.002 FP explosion → fixed to 0.40
  - F1: 0 → 0.286
- **CREDIT_CARD_SECURITY_CODE (CVV)**: new recognizer, base score 0.40 (context-gated)
  - Requires "cvv/cvc/security code" keywords; initial 0.60 fired on SSN substrings → fixed
- **US_BANK_NUMBER**: entity type renamed from ROUTING_NUM (LABEL_MAP alignment)
  - F1: 0 → 0.444 (R=1.000; precision limited by 9-digit pattern breadth)
- LABEL_MAP additions in run_benchmark_nlp.py: ROUTING_NUM, CREDIT_CARD_NUMBER, SWIFT_BIC_CODE, CREDIT_CARD_SECURITY_CODE, BBAN, ACCOUNT_PIN

### Sprint 6 — Presidio utilization + ABA checksum + ITIN/Passport (current)
- **ABA routing checksum** (`pii_redact/ner/regex_patterns.py`)
  - `_AbaRoutingRecognizer` subclass overrides `validate_result()` with 3-7-1 weighted sum
  - US_BANK_NUMBER P: 0.286 → **1.000**, F1: 0.444 → **1.000**
- **ITIN recognizer** — `ITIN_Recognizer` (pattern `9\d{2}-\d{2}-\d{4}`, score 0.65)
  - Replaces built-in `UsItinRecognizer` (base 0.5 < min_confidence=0.6 — was silently suppressed)
- **US Passport recognizer** — `Passport_Recognizer` (pattern `[A-Z]\d{8}`, score 0.65)
  - Replaces built-in `UsPassportRecognizer` (base 0.45 < min_confidence=0.6)
- **presidio-structured** — `build_structured_engine()` in presidio_setup.py
  - Wraps `AnalyzerEngine` in `StructuredEngine` for pandas DataFrame PII analysis
  - `PandasAnalysisBuilder().generate_analysis(df)` → `engine.anonymize(df, analysis, operators=...)`
- **TransformersNlpEngine** — `PII_USE_TRANSFORMERS=1` opt-in + `.load()` fix
  - Model: `obi/deid_roberta_i2b2` (clinical PHI, i2b2 2014 challenge)
  - Benchmarked: F1=0.770, no gain vs spaCy+GLiNER on financial text — not adopted as default
- **PERSON name heuristic** — `looks_like_name()` fallback in `_boxes_from_layout_spans()`
  - Fires when NER produces no PERSON in a high-value Docling field (Patient Name, Insured, etc.)
  - Accepts 2-60 char, letters/spaces/hyphens/periods, starts uppercase, not in field-label blocklist
- **VLM GPU fix** — `dtype=` → `torch_dtype=`, `device_map="auto"` → `device_map={"": 0}`
  - Root cause: wrong kwarg name caused float32 load (32 GB) → silent OOM → CPU fallback
  - Fixed: bfloat16 explicit, single-GPU placement bypasses accelerate heuristic
- **Test suite** — `test_presidio_setup.py` added (12 tests: singleton, reset, ITIN/Passport, StructuredEngine)
  - Total: 197 tests passing, 9 skipped
- **Overall S6 F1: 0.771** (≈ S5 0.772; headline gain is US_BANK_NUMBER 0.444→1.000)

### Sprint 5 — GLiNER validated + PHONE_NUMBER precision
- **GLiNER benchmarked for first time** (`PII_USE_GLINER=1`)
  - First run revealed EMAIL regression: P 1.000→0.636 (GLiNER returning form labels like "Email:")
  - Fix: `_validate_span()` in `gliner_recognizer.py` — format gate per entity type
    - EMAIL_ADDRESS: requires `@` in span
    - PHONE_NUMBER: requires ≥7 digits in span
    - US_SSN: requires SSN pattern match
    - NPI/EIN: requires at least one digit
  - After fix: EMAIL restored 0.966, PHONE P lifted 0.362→0.739
  - Net gains from GLiNER: LOCATION 0.382→0.556, ORG 0.253→0.545, PHONE F1 0.423→0.791
  - **Overall F1: 0.738 → 0.772** (+0.034; 534.6s vs 7.8s without GLiNER)
- **PHONE_NUMBER precision fix** (`pii_redact/ner/regex_patterns.py`)
  - Root cause: EDI/EDIFACT field separators (`NAD+SU+9X:12+9876543210`) and SWIFT/IBAN embedded numbers
  - FP categories: 17 international FPs (EDI `+` prefix), 8 bare 10-digit FPs (SWIFT/IBAN)
  - US 10-digit: `(?<!\d)` → `(?<![+:\w])` (blocks after `+`, `:`, word chars)
  - International: `(?<![a-zA-Z0-9])` before `\+` + `(?!\d|:)` lookahead
  - **Result: PHONE_NUMBER P: 0.324→0.739** (with GLiNER)
- **VLM grounding A/B benchmark** (CORD, CPU run in progress)

### Sprint 4 — VLM grounding + PHONE_NUMBER
- **VLM grounding mode** (`pii_redact/ner/vlm_extractor.py` rewritten)
  - Old: pipe-delimited text output, `(x=0, y=0, w=0, h=0)` placeholder coordinates
  - New: JSON `bbox_2d` grounding output — Qwen3-VL's native coordinate format
  - Format: `[{"bbox_2d": [x1, y1, x2, y2], "label": "PERSON", "text": "Jane", "confidence": 0.9}]`
  - Coords normalized 0-1000 → pixel by `(coord / 1000) * img_dim`
  - Parser handles: markdown fences, reversed coords, out-of-bounds, JSON in prose
  - 20 fast unit tests in `tests/test_vlm_extractor.py`
- **VLM benchmark A/B** (`scripts/run_benchmark_sroie.py`)
  - `--vlm` flag: routes each CORD receipt image through `run_ner_vlm()` (VLM + NER merged)
  - `_get_vlm()` lazy loader, temp PNG lifecycle managed in `finally`
  - `print_table()` shows `[VLM+NER]` vs `[NER]`
  - 15 fast mocked tests in `tests/test_benchmark_vlm.py`
  - **GPU run pending**: `python scripts/run_benchmark_sroie.py --dataset cord --max-docs 50 --vlm`
- **PHONE_NUMBER custom recognizer** (`pii_redact/ner/regex_patterns.py`)
  - Root cause: built-in PhoneRecognizer base 0.4 → filtered at min_score=0.6 without context
  - US 10-digit: `(?<!\d)(?:\+1[\s.-]?)?(?:\([2-9]\d{2}\)|[2-9]\d{2})[\s.-]?[2-9]\d{2}[\s.-]?\d{4}(?!\d)` score=0.65
  - International E.164: `\+[1-9]\d{0,2}[\s.-]?\(?\d{1,4}\)?[\s.-]?\d{2,4}...` score=0.60
  - Removed `PhoneRecognizer` from Presidio built-ins → 14 custom recognizers total
  - **Result: PHONE_NUMBER R: 0.267 → 0.611 (+129%), F1: 0.348 → 0.423 (+22%)**
  - Overall partial F1: 0.732 → **0.738**

---

## Known Gaps (priority order)

| Gap | Entity | Metric | Root cause | Difficulty |
|-----|--------|--------|------------|------------|
| PERSON recall | PERSON | R=0.452 | NER misses single-word names in short text; name heuristic added for Docling path only | Medium |
| SWIFT_BIC precision | SWIFT_BIC | P=0.200 | 8-char pattern matches company abbreviations | Hard |
| CREDIT_CARD_SECURITY_CODE | CVV | F1=0.000 | Gretel test set lacks "cvv/cvc" context keywords near 3-digit values | Low (dataset issue) |
| VLM+NER on receipts (GPU) | image path | unknown | CPU run done (F1=0.099, degraded); GPU run in progress | Run on GPU |
| WildReceipt benchmark | image path | unknown | 1,765 real receipt photos not yet ingested | Medium |
| GLiNER runtime | all | 1280s vs 8s | CPU inference; acceptable for batch, slow for interactive | GPU / quantize |
| TransformersNlpEngine | PERSON/ORG | neutral | `obi/deid_roberta_i2b2` trained on clinical PHI, not financial text; no gain on Gretel | Domain gap |

---

## Immediate Next Steps

1. **CORD VLM conclusion** — F1=0.099 confirmed stable across 2 independent runs. VLM+NER is harmful on CORD (47 FP, 6 TP vs NER-only 0.18). Root cause: dataset mismatch — CORD = Indonesian store receipts with no personal PII. VLM GPU fix applied (`torch_dtype`, `device_map={"": 0}`); device confirmation print added to benchmark. VLM is better validated on insurance forms where personal PII is dense.

2. **PERSON recall on NLP benchmark** — R=0.452. Name heuristic (`looks_like_name`) added for Docling path (high-value fields). No NLP benchmark impact since it only fires on form fields; PERSON recall on flat text still needs a better NER model for single-word names.

3. **SWIFT_BIC precision** — P=0.200. 8-char pattern fires on company abbreviations. Needs BIC format gate (first 4 chars alpha, next 2 = ISO country code).

4. **GLiNER runtime** — 1280s/100 docs on CPU. Consider GPU inference for interactive mode.

5. **WildReceipt** — 1,765 real receipt photos. Dataset setup: `scripts/download_wildreceipt.py`.

---

## Infrastructure Notes

- **Test tier rules (CRITICAL — prevents OOM):**
  - `pytest -m fast` — pure Python, no ML imports. Always safe. (~8s)
  - `pytest -m slow` — loads torch+spaCy+Presidio. CI/Docker only, never background.
  - Never `pytest tests/` unmarked.

- **Segfault isolation:** `test_regex_patterns.py` excluded from default collection (presidio import chain loads torch + bundled OpenSSL, corrupts `cryptography` AES state in same process). Use `pytest tests/test_regex_patterns.py` standalone.

- **Vault tests:** Run separately (`pytest tests/test_vault.py`) due to same OpenSSL isolation issue.

- **Hourly cron:** job `db335ee6`, fires :13 past every hour, 7-day TTL.
  - Checks: fast tests + Gretel NLP benchmark (exact) + git log
  - Regression threshold: Gretel exact F1 < 0.240

- **GPU notes:**
  - FP8 inference requires SM ≥ 8.9 (RTX 4090, H100); auto-detected
  - RTX 3090 (SM 8.6): auto-falls back to bfloat16 base model
  - VLM inference: ~12s/page on RTX 3090
