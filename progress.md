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

## What's Next (priority order)
1. [ ] Fix full-suite segfault (subprocess isolation in conftest.py)
2. [ ] Implement `pii_redact.vault` — Fernet key gen + encrypt/decrypt + token map
3. [ ] Implement `pii_redact.audit` — SQLite-backed audit log write/read
4. [ ] Implement `pii_redact.redact.pdf_writer` — PyMuPDF black-box redaction
5. [ ] Implement `pii_redact.pipeline` — full orchestration pipeline
6. [ ] Validate against i2b2 PHI dataset (dataset_validation script)
7. [ ] Wire Gradio UI to actual pipeline
8. [ ] Docker image + CI

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
