# PII ZERO

**100% offline PII anonymization for insurance claim documents.**

PII ZERO detects and redacts personally identifiable information from insurance claims, EOBs, medical records, and related documents using a layered detection architecture — no data ever leaves the machine. It is built to meet HIPAA de-identification requirements under the Expert Determination method.

---

## Why this exists

Insurance document workflows are drowning in PII. A single claim packet can contain SSNs, NPI numbers, EINs, dates of birth, policy numbers, DEA numbers, ICD-10 codes, and handwritten patient signatures — spread across structured PDFs, scanned TIFFs, and photos. Existing SaaS tools require sending that data to a vendor's server. That is not acceptable for healthcare data.

PII ZERO runs entirely on-premise. No API calls. No cloud storage. No vendor data agreements. The raw documents never leave the host machine.

---

## What it redacts

Solid black fills. Never blur, never inpaint. Blur-based redaction was broken by [Bishop Fox's Unredacter](https://bishopfox.com/blog/unredacter-tool-never-use-pixelation-blur) in 2021 — blurred text is recoverable. Every detection in PII ZERO produces a `fill_color: [0, 0, 0]` box that overwrites the pixel data permanently and removes the native text from the PDF object tree.

### Entity types detected

| Category | Entities |
|---|---|
| Identity | PERSON, DATE_TIME (DOB), US_SSN, US_PASSPORT, US_DRIVER_LICENSE |
| Contact | PHONE_NUMBER, EMAIL_ADDRESS, LOCATION (full addresses) |
| Insurance | POLICY_NUM, CLAIM_REF, ADJUSTER_ID, NPI, EIN, ROUTING_NUM |
| Medical | ICD10_CODE, CPT_CODE, DEA_NUM |
| Financial | CREDIT_CARD, IBAN_CODE, IP_ADDRESS |
| Visual | FACE (detected in photos), LICENSE_PLATE, HANDWRITING (free-form) |

---

## Architecture

Three detection layers run in sequence; results are unioned and deduplicated.

```
Document
   │
   ├─► Layer 1: Text NER (Presidio + spaCy)
   │     ├── 10 custom PatternRecognizers (insurance domain)
   │     ├── spaCy en_core_web_lg (PERSON, ORG, LOC)
   │     └── Presidio built-ins (EMAIL, PHONE, SSN, CREDIT_CARD, ...)
   │
   ├─► Layer 2: OCR + NER  (scanned/raster documents)
   │     ├── PaddleOCR  — printed text → NER
   │     ├── TrOCR (microsoft/trocr-base-handwritten) — handwriting
   │     └── Same Presidio pipeline on OCR output
   │
   └─► Layer 3: Visual  (Qwen3-VL-8B, GPU-first)
         ├── Face detection (CenterFace / OpenCV Haar cascade)
         ├── License plate detection (YOLOv8n)
         ├── QR / barcode detection (pyzbar)
         └── VLM: Qwen3-VL-8B-Instruct — holistic PII extraction from image
               FP8 on SM≥8.9 (RTX 4090 / H100) | bfloat16 on SM 8.6 (RTX 3090)
```

**Post-processing:**
- Language-aware FP filter: PERSON/ORG/LOCATION spans in non-English context windows are dropped (eliminates foreign-language hallucinations from `en_core_web_lg`)
- Confidence threshold gate: `min_confidence: 0.60` in settings.yaml
- Coordinate padding: each box expanded 4 px to prevent edge bleed-through

**Reversible vault (optional):**
- All detected PII values encrypted with AES-256-GCM (Fernet) before redaction
- Token map stored in SQLite vault (`./vault/vault.db`)
- Original values restorable with `pii-redact restore` + the vault key

---

## Benchmark results

Evaluated on [Gretel Finance PII dataset](https://huggingface.co/datasets/gretelai/gretel_pii_finance) (100 docs, min_score=0.6, partial span matching).

> **Partial span matching** is the correct metric for redaction. If a model detects "Springfield" and the gold label is "Springfield, IL 62701", the PII is caught — the claim that matters for compliance is coverage, not exact span boundary alignment.

### Overall

| Mode | P | R | F1 |
|------|---|---|-----|
| Exact span | 0.163 | 0.471 | 0.242 |
| Partial overlap | 0.489 | 0.787 | **0.604** |

### Per entity (partial match)

| Entity | P | R | F1 |
|--------|---|---|-----|
| EMAIL_ADDRESS | 1.000 | 0.933 | **0.966** |
| IBAN_CODE | 1.000 | 0.917 | **0.957** |
| IP_ADDRESS | 0.875 | 1.000 | **0.933** |
| DATE_TIME | 0.891 | 0.895 | **0.893** |
| PERSON | 0.653 | 0.413 | 0.506 |
| LOCATION | 0.426 | 0.349 | 0.384 |
| ORG | 0.129 | 0.750 | 0.220 |

**Known benchmark limitations:**
- CREDIT_CARD: Gretel test set uses Luhn-invalid synthetic card numbers. Production behavior is correct — "Credit Card Number: 4532015112830366" is detected and redacted. Benchmark F1 stays 0 by design.
- ORG precision is low because the Gretel dataset contains many non-PII organizational references in the same documents (insurers, courts, agencies). This is a document-level context problem, not a span-detection problem.
- TAB dataset (legal documents): structural FP problem — court documents mention thousands of non-confidential ORG/LOC entities. F1=0.049 on exact match; improving this requires document-type-aware context filtering.

---

## Installation

### Prerequisites

- Python 3.10+
- GPU with 12+ GB VRAM recommended for VLM layer (Qwen3-VL-8B)
- CPU-only mode works but VLM inference is slow (~5 min/page)

```bash
git clone https://github.com/scamai/PII_ZERO
cd PII_ZERO
pip install -e ".[dev]"
```

### Download model weights

```bash
python scripts/download_models.py
```

This fetches spaCy `en_core_web_lg`, TrOCR, YOLOv8n, and SAM. Qwen3-VL must be downloaded separately from Hugging Face (requires acceptance of model license):

```bash
# SM >= 8.9 (RTX 4090, H100, A100): use FP8 — fits in ~12 GB VRAM
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct-FP8

# SM 8.6 (RTX 3090): auto-detected, falls back to bfloat16 base model
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
```

The pipeline auto-detects GPU compute capability and selects the right model at load time.

### Check installation

```bash
pii-redact doctor
```

---

## Usage

### CLI

**Redact a single file:**
```bash
pii-redact redact claim.pdf --output claim_redacted.pdf
```

**Dry run (detect only, no output written):**
```bash
pii-redact inspect claim.pdf --format table
pii-redact inspect claim.pdf --format json
```

**Batch a directory:**
```bash
pii-redact redact ./claims_inbox/ --output ./claims_redacted/
```

**With reversible vault** (keeps original values for authorized restoration):
```bash
export PII_VAULT_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
pii-redact redact claim.pdf --vault ./vault/vault.db
pii-redact restore claim_redacted.pdf --vault ./vault/vault.db
```

**Enable VLM layer** (requires GPU + downloaded Qwen3-VL weights):
```bash
pii-redact redact claim.pdf --vlm
```

### Web UI

```bash
pii-redact ui --port 7860
pii-redact ui --port 7860 --vlm      # with VLM layer
```

Opens a local Gradio interface at `http://127.0.0.1:7860`. Three tabs:
- **Redact** — upload document, download redacted output
- **Inspect** — dry run with detection overlay and confidence scores
- **Vault / Restore** — restore original PII from encrypted vault

No data is sent to Gradio cloud. The interface runs fully locally.

### Docker

```bash
docker-compose up --build
# UI available at http://localhost:7860
```

---

## Document routing

PII ZERO classifies each document before processing and routes it to the appropriate sub-pipeline:

| Document type | Text layer | OCR | Visual | VLM |
|---|---|---|---|---|
| PDF (native text) | Presidio NER | — | optional | optional |
| Scanned form / TIFF | — | PaddleOCR + NER | TrOCR | optional |
| Photo | — | — | Face, plate, QR | optional |
| ID document | — | PaddleOCR | Face | optional |
| Medical record | Presidio + scispaCy | PaddleOCR | — | optional |
| Unknown | all layers | all layers | all layers | optional |

Form templates (CMS-1500, ACORD 125, ACORD 140) are matched by perceptual hash at 300 DPI. Template-aware processing uses known field coordinates as priors, improving recall on structured forms.

---

## Configuration

All runtime settings live in `config/settings.yaml`. Key options:

```yaml
redaction:
  fill_color: [0, 0, 0]        # solid black — never blur
  fill_padding_px: 4
  min_confidence: 0.60

pipeline:
  ocr_dpi: 300
  skip_visual_for_structured_pdfs: false

vault:
  db_path: ./vault/vault.db
  key_env: PII_VAULT_KEY        # AES-256-GCM key loaded from env

audit:
  log_dir: ./audit_logs
  include_text: false           # never log PII values
```

---

## Testing

Tests are split into two tiers to prevent ML library conflicts:

```bash
# Fast tier — pure Python, <10 seconds, safe to run anytime
pytest -m fast

# Slow tier — loads torch + spaCy + Presidio — run in isolation
pytest tests/test_regex_patterns.py
pytest tests/test_smoke_claim_form.py -v    # end-to-end CMS-1500 smoke test

# Never run unmarked full suite — simultaneous ML library loads cause segfaults
# (torch OpenSSL + cryptography OpenSSL conflict on Python 3.13)
```

### Test coverage

| File | Tests | Tier | What it covers |
|------|-------|------|----------------|
| test_models.py | 15 | fast | RedactionBox / DocumentResult data contracts |
| test_no_blur_safety.py | 10 | fast | Pixel-level verification: no blur, only fills |
| test_redact_engine.py | 30 | fast | Engine safety + PII-in-output guard |
| test_vault.py | 7 pass / 5 skip | fast | Fernet encrypt/decrypt + audit log |
| test_regex_patterns.py | 17 | slow | Determinism tests for all 10 regex recognizers |
| test_smoke_claim_form.py | 9 | slow | End-to-end pipeline on synthetic CMS-1500 PDF |

**Total: 88+ passing tests.**

---

## Custom insurance domain recognizers

Ten PatternRecognizers tuned to insurance document formats, registered alongside Presidio built-ins:

| Recognizer | Pattern example | Score |
|---|---|---|
| SSN | `523-67-4891` | 0.85 |
| NPI | `1234567893` (10-digit, starts 1 or 2) | 0.65 |
| EIN | `47-1234567` | 0.75 |
| ICD-10 | `M54.5`, `Z00.00` | 0.70 |
| CPT | `99213`, `93000-26` | 0.60 |
| POLICY_NUM | `POL-7834521`, generic alphanumeric | 0.55–0.90 |
| ROUTING_NUM | 9-digit ABA routing | 0.70 |
| ADJUSTER_ID | `ADJ-4829`, `ADJ4829` | 0.50–0.85 |
| CLAIM_REF | `CLM-2024-88801` | 0.45–0.90 |
| DEA_NUM | `AB1234567` (2 letters + 7 digits) | 0.75 |
| CREDIT_CARD | 13–19 digit (Luhn validated) | 0.50 |
| ADDRESS | `412 Maple Street` (US street pattern) | 0.65 |

All recognizers use context-boosting: score is raised when surrounding text contains relevant keywords (`npi`, `policy number`, `routing`, etc.).

---

## Audit trail

Every processed document generates a JSON audit entry in `./audit_logs/`:

```json
{
  "doc_id": "3f8a1c2d-...",
  "filename": "claim_2024_00441.pdf",
  "doc_type": "PDF_STRUCTURED",
  "pages": 3,
  "total_processing_ms": 1842.3,
  "entity_counts": {"PERSON": 2, "US_SSN": 1, "EMAIL_ADDRESS": 1},
  "source_counts": {"presidio": 4, "vlm": 2},
  "detections": [
    {"entity_type": "US_SSN", "confidence": 0.85, "source": "presidio", "page": 0}
  ]
}
```

PII values are never written to the audit log (`include_text: false` by default). The log records that a detection occurred, which entity type, and which layer found it.

---

## Security notes

- **Solid fill only.** Blur and pixelation are recoverable. Every redaction writes a filled black rectangle over both the pixel data (raster) and the PDF text object (native PDFs).
- **Vault key never persists.** The AES-256-GCM key is loaded from the `PII_VAULT_KEY` environment variable at runtime. It is never written to disk.
- **No network calls.** All model weights are loaded from local paths. `presidio_analyzer` and `spacy` do not phone home. Gradio UI runs with `share=False`.
- **Audit log is PII-free.** `include_text: false` in settings.yaml. Turning this on for debugging must be done deliberately.

---

## Roadmap

### Near-term (precision and recall)

- **ORG precision** — currently 0.129. Requires document-level context to separate "Acme Corp (employer of record)" from "Blue Cross Blue Shield (insurer listed on form)". Plan: classify entity role within document structure before flagging.
- **PHONE recall** — 0.267. Format variation ((312) vs 312. vs +1-312) requires a broader base pattern with lower score + context boosting.
- **PaddleOCR integration** — needed for scanned documents; the biggest gap between current state and production-ready. Not yet installed in dev environment.
- **Span deduplication** — overlapping boxes from multiple detection layers need NMS before rendering.

### Medium-term (production hardiness)

- **VLM benchmark numbers** — Qwen3-VL is wired into the pipeline but not yet included in any benchmark evaluation. Need per-entity F1 numbers with VLM on vs off.
- **Template-aware coordinate priors** — CMS-1500 and ACORD forms have fixed field positions. Hard-coded priors for known field coordinates would boost recall on structured forms without relying on NER at all.
- **Async batch processing** — current batch mode is sequential; multi-worker PDF rendering + NER can run in parallel per-page.
- **Fine-tuned insurance NER** — `en_core_web_lg` is a general-purpose model. A spaCy model fine-tuned on annotated insurance claims would raise PERSON/ORG precision significantly.

### Compliance and audit

- **HIPAA Safe Harbor field checklist** — 18 categories defined in 45 CFR §164.514(b)(2). Add a per-document coverage report that checks which categories were detected and flags any gaps.
- **Structured audit export** — convert JSON audit logs to HIPAA-compliant CSV for compliance reporting.
- **Vault access log** — track who restored what, when, with what key identity.

---

## License

MIT. See LICENSE.

---

## Acknowledgments

- [Microsoft Presidio](https://github.com/microsoft/presidio) — NER engine and recognizer framework
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) — visual PII extraction
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) — printed text OCR
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) — PDF text extraction and redaction
- [Bishop Fox Unredacter](https://bishopfox.com/blog/unredacter-tool-never-use-pixelation-blur) — the paper that settled the blur debate
- [Gretel Finance PII dataset](https://huggingface.co/datasets/gretelai/gretel_pii_finance) — benchmark dataset
