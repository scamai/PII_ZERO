# PII Removal Tool — Build Progress

## Architecture
Thin Python orchestration layer (Path B):
- **Presidio** (regex + spaCy NER) as 3-layer chained recognizer
- **PyMuPDF** for PDF extraction
- **PaddleOCR** for printed text OCR
- **Transformers (TrOCR)** for handwriting
- **Fernet (AES-128-CBC + HMAC)** for reversible vault encryption
- **Gradio** UI + CLI

## Source Tree (implemented)
```
pii_redact/
  __init__.py          — package root, exports RedactionBox / DocumentType / MetadataStripResult
  models.py            — core dataclasses: RedactionBox, DocumentResult, PageResult, etc.
  config.py            — Pydantic settings loaded from config/settings.yaml
  pipeline.py          — RedactionPipeline orchestrator (stub)
  classifier.py        — document type classifier (stub)
  cli.py               — Click CLI: redact / batch / review / vault / audit / train
  ui.py                — Gradio 3-tab UI (stub)
  layers/              — (empty, reserved for redaction layers)
  ner/
    regex_patterns.py  — 10 PatternRecognizers: SSN, NPI, EIN, ICD10, CPT, POLICY_NUM,
                         ROUTING_NUM, ADJUSTER_ID, CLAIM_REF, DEA_NUM
    presidio_setup.py  — singleton AnalyzerEngine builder with spaCy fallback cascade
    insurance_ner.py   — insurance-domain spaCy NER stub
  redact/              — (empty, reserved for redaction writers)
  templates/           — (empty, coordinate template store)

config/settings.yaml   — runtime config (thresholds, paths, entity list)
scripts/
  validate_datasets.py — offline dataset validator (fetches i2b2 PHI challenge data)
  download_models.py   — model downloader
  train_insurance_ner.py — training stub

tests/
  test_models.py         — RedactionBox / DocumentResult data contract tests
  test_no_blur_safety.py — pixel-level redaction blur safety
  test_redact_engine.py  — engine safety + PII-in-output guard tests
  test_regex_patterns.py — determinism tests for all 10 regex recognizers
  test_vault.py          — Fernet encrypt/decrypt + audit log (5 skipped: vault module pending)
```

## Test Suite Status

| File | Tests | Status |
|------|-------|--------|
| test_models.py | ~15 | PASS |
| test_no_blur_safety.py | ~10 | PASS |
| test_redact_engine.py | ~30 | PASS |
| test_regex_patterns.py | ~17 | PASS |
| test_vault.py | 7 pass / 5 skip | PASS (isolated) |

**Known issue**: Full `pytest` run segfaults on vault crypto tests (#72+) because
`presidio_analyzer` import chain loads torch + bundled OpenSSL, which corrupts
`cryptography`'s OpenSSL AES state.  
**Fix applied**: `@pytest.mark.forked` on `TestCryptoRoundTrip` — but `fork()` is
deprecated for multi-threaded Python 3.13.  
**Current workaround**: Run vault tests separately: `pytest tests/test_vault.py`

## What Works
- [x] Full project scaffolding + pyproject.toml + Docker
- [x] Core data models (RedactionBox, DocumentResult, etc.)
- [x] 10 insurance-domain regex recognizers (SSN, NPI, EIN, ICD10, CPT, POLICY_NUM, ROUTING_NUM, ADJUSTER_ID, CLAIM_REF, DEA_NUM)
- [x] Presidio AnalyzerEngine singleton builder with spaCy fallback cascade
- [x] Click CLI skeleton (redact / batch / review / vault / audit / train)
- [x] Gradio UI skeleton (3 tabs)
- [x] Config system (Pydantic settings + settings.yaml)
- [x] 84+ passing tests (85 pass, 5 skip in isolated run)

## Benchmark Results

### Layer 1: Regex-only baseline
Run: `python scripts/run_benchmark.py --dataset all --max-docs 200`

| Dataset | P | R | F1 | Notes |
|---------|---|---|----|-------|
| TAB (200 docs) | 0.000 | 0.000 | 0.000 | TAB targets PERSON/ORG/LOC — no regex match |
| Gretel Finance (200 docs) | 0.302 | 0.126 | 0.178 | EMAIL=0.867, DATE=0.412, IPV4=1.0 |

### Layer 2: Presidio NER (spaCy + regex)
Run: `CUDA_VISIBLE_DEVICES="" python scripts/run_benchmark_nlp.py --dataset all --max-docs 100 --min-score 0.6`

| Dataset | P | R | F1 | Notes |
|---------|---|---|----|-------|
| TAB (100 docs) | 0.025 | 0.604 | 0.049 | High FP: legal text has many non-confidential entities |
| Gretel Finance (100 docs) | 0.163 | 0.532 | 0.251 | +41% vs regex |

**Gretel NLP per-entity highlights (min_score=0.6):**

| Entity | P | R | F1 |
|--------|---|---|----|
| IP_ADDRESS | 0.875 | 1.000 | 0.933 |
| EMAIL_ADDRESS | 0.929 | 0.867 | 0.897 |
| DATE_TIME | 0.522 | 0.726 | 0.607 |
| PHONE_NUMBER | 0.312 | 0.882 | 0.462 |
| PERSON | 0.137 | 0.401 | 0.204 |
| ORG | 0.056 | 0.365 | 0.098 |

**TAB structural problem:** Court documents contain thousands of non-confidential ORG/LOC mentions (court names, agencies) that NER cannot distinguish from confidential ones. This requires document-level context, not span-level detection.

**Key gaps remaining:**
- CREDIT_CARD: 0.0 — Presidio recognizer exists but Gretel format may not match
- LOCATION: 0.0 — span mismatch (Gretel annotates full addresses, spaCy detects city names)
- ORG precision very low — needs context-aware filtering
- PERSON precision 0.137 — spaCy finds many non-PII person references

## What's Next (priority order)
1. [x] Fix full-suite segfault — excluded test_regex_patterns.py from default collection
2. [x] Fix infinite subprocess recursion in conftest.py (_PYTEST_SUBPROCESS_WORKER sentinel)
3. [x] Implement `pii_redact.vault` — Fernet key gen + encrypt/decrypt + VaultSession token map
4. [x] Implement `pii_redact.audit` — SQLite write_event / read_events
5. [x] Integrate Qwen3-VL-8B-Instruct-FP8 (GPU-first, CPU fallback) into pipeline
6. [x] Benchmark runner (scripts/run_benchmark.py) — TAB + Gretel Finance
7. [x] Fix label mapping (ORGANIZATION→ORG, etc.) to improve Gretel F1
8. [x] Enable Presidio NLP layer in benchmark (run_benchmark_nlp.py) — Gretel F1 +41%
9. [x] Download Qwen3-VL model weights:
   - FP8 (SM≥8.9): Qwen/Qwen3-VL-8B-Instruct-FP8
   - bfloat16 (RTX 3090 / SM 8.6): Qwen/Qwen3-VL-8B-Instruct
10. [x] Test VLM on sample document — working on GPU (16s inference, 10/10 PII detected)
    - Fixed: Qwen3VLForConditionalGeneration class name
    - Fixed: system message as messages dict, not apply_chat_template kwarg
    - Fixed: SM-aware model selection (FP8 needs SM≥8.9; RTX 3090 uses bf16 base)
11. [x] Add label mappings: TIME→DATE_TIME, IBAN→IBAN_CODE, PASSPORT_NUMBER, DRIVER_LICENSE_NUMBER
    - IBAN_CODE: F1=0.957 after fix
12. [ ] Improve CREDIT_CARD detection (Presidio requires Luhn validation — Gretel uses fake nums)
13. [ ] Improve LOCATION by routing addresses through VLM instead of spaCy (VLM detects full address)
14. [x] Wire VLM into full pipeline — pipeline._run_visual_layer writes temp PNG, passes to VLM
15. [ ] Wire Gradio UI to actual pipeline
16. [x] Hourly verification agent — cron job db335ee6 (fires :13 past every hour, 7-day TTL)
    - Runs: fast tests + Gretel NLP benchmark + git log check
    - Baseline: Gretel F1=0.243, EMAIL=0.897, IBAN=0.957, IP=0.933

## Current Benchmark Baselines
| Dataset | Layer | P | R | F1 | Notes |
|---------|-------|---|---|----|-------|
| Gretel (100 docs) | Regex only | 0.302 | 0.126 | 0.178 | email, ip only |
| Gretel (100 docs) | Presidio NLP | 0.164 | 0.469 | 0.243 | +37% vs regex |
| TAB (100 docs) | Presidio NLP | 0.025 | 0.604 | 0.049 | structural FP problem |

## Known Limitations
- CREDIT_CARD: Gretel uses 19-digit Maestro numbers; Presidio regex doesn't cover them
- LOCATION: Full address span mismatch (Gretel: full street; spaCy NER: city name only)
- TAB precision: Court documents have thousands of non-confidential ORG/LOC mentions
- VLM inference: ~12s per page on RTX 3090 bfloat16; model loads in ~4s

## Test Tier Rules (CRITICAL — prevents machine OOM/kill)
- **`pytest -m fast`** — pure Python only, no ML imports. Safe to run anytime. (<10s)
- **`pytest -m slow`** — loads torch+spacy+paddle+presidio. CI/Docker only. Never background.
- **Never run `pytest tests/` (unmarked full suite)** — loads all ML libraries simultaneously.
- test_regex_patterns.py is `slow` (presidio import → spacy → torch chain)
- test_models, test_vault, test_no_blur_safety, test_redact_engine are `fast`

## Attempted / Known Issues
- **scispaCy install**: version conflicts with spaCy 3.x; skipped in favor of spaCy `en_core_web_sm`
- **PaddleOCR**: heavy install, not yet exercised in tests (guarded by `importlib` in engine tests)
- **pytest-forked on Python 3.13**: `fork()` in multi-threaded process causes deprecation warning and failures; subprocess spawn approach needed instead
