# PII ZERO

**100% offline PII redaction for financial and insurance document images.**

PII ZERO detects and redacts personally identifiable information from scanned receipts, invoices, bank statements, insurance claim forms, and any financial document — whether it arrives as a native PDF, a scanned TIFF, or a phone photo. No data ever leaves the machine.

---

## The problem it solves

Financial workflows are full of document images. A single invoice or bank statement can carry account numbers, routing numbers, SWIFT codes, names, addresses, SSNs, and credit card digits — embedded in a photograph, a scanned form, or a rasterized PDF. Sending those images to a SaaS redaction API means handing your customers' financial data to a third party.

PII ZERO runs entirely on-premise. Every model runs locally. No API calls. No cloud storage. No vendor data agreements.

---

## Document types

The pipeline handles the full range of real-world financial documents:

| Document | Format | Examples |
|---|---|---|
| **Receipts** | Photo / scanned image | Grocery, retail, restaurant, ATM receipts |
| **Invoices** | PDF / scanned / image | Vendor invoices, utility bills, medical bills |
| **Bank statements** | PDF (native text) / scanned | Account statements, wire transfer records |
| **Insurance claim forms** | Scanned TIFF / PDF | CMS-1500, ACORD 125, ACORD 140, EOBs |
| **ID documents** | Photo | Driving licenses, passports (face + fields) |
| **Medical records** | PDF / scanned | Referral letters, lab reports, discharge summaries |

For **receipt and invoice images** specifically — the most common financial document format — the pipeline runs:

1. Surya OCR to extract printed text from the image with bounding boxes
2. Presidio + spaCy NER with 17 custom recognizers on the OCR output
3. Visual layer for face detection, QR codes, and barcodes
4. Optional Qwen3-VL-8B for visual PII grounding with pixel-accurate bounding boxes

---

## What it redacts

Solid black fills. Never blur, never inpaint. Blur-based redaction was broken by [Bishop Fox's Unredacter](https://bishopfox.com/blog/unredacter-tool-never-use-pixelation-blur) in 2021 — pixelated text is recoverable. Every detection in PII ZERO writes a `fill_color: [0, 0, 0]` box that overwrites pixel data permanently and removes the native text from the PDF object tree.

### Entity types detected

| Category | Entities |
|---|---|
| **Identity** | PERSON, DATE_TIME (DOB), US_SSN, US_PASSPORT, US_DRIVER_LICENSE |
| **Contact** | PHONE_NUMBER, EMAIL_ADDRESS, LOCATION (addresses) |
| **Financial** | CREDIT_CARD (13–19 digit, Luhn validated), CREDIT_CARD_SECURITY_CODE (CVV/CVC), IBAN_CODE, SWIFT_BIC_CODE, US_BANK_NUMBER (ABA routing) |
| **Insurance** | POLICY_NUM, CLAIM_REF, ADJUSTER_ID, NPI, EIN |
| **Medical** | ICD10_CODE, CPT_CODE, DEA_NUM |
| **Network** | IP_ADDRESS, MAC_ADDRESS |
| **Visual** | FACE (photos), LICENSE_PLATE, QR/barcode, HANDWRITING (free-form) |

---

## Architecture

Four detection layers run in sequence. Results are unioned, deduplicated by IoU and text-span NMS, and rendered as solid fills.

```
Document Image / PDF
        │
        ├─► [Route] Document classifier → selects sub-pipeline
        │
        ├─► Layer 1: Surya OCR  (scanned images and raster documents)
        │     ├── DetectionPredictor — text region bounding boxes
        │     ├── RecognitionPredictor — printed text with per-line confidence
        │     └── TrOCR (microsoft/trocr-base-handwritten) — handwriting regions
        │
        ├─► Layer 2: Text NER  (native PDF text + OCR output)
        │     ├── Presidio AnalyzerEngine (singleton)
        │     │     ├── spaCy en_core_web_lg → PERSON, ORG, LOCATION
        │     │     ├── 17 custom PatternRecognizers (insurance + financial domain)
        │     │     └── Presidio built-ins (EMAIL, IBAN, IP, SSN, ...)
        │     ├── Post-NER filters (entity_filters.py)
        │     │     ├── ORG: blocklist + acronym gate + tech-char filter + context gate
        │     │     └── PERSON: form-label blocklist + lowercase-start filter
        │     ├── Language filter (langdetect) — drops non-English NER FPs
        │     ├── GLiNER zero-shot NER  [opt-in: PII_USE_GLINER=1]
        │     │     └── urchade/gliner_medium-v2.1 (or nvidia/gliner-PII)
        │     └── StructuredEngine  [opt-in: PII_USE_STRUCTURED=1]
        │           └── presidio-structured: PandasAnalysisBuilder → DataFrame column PII
        │
        ├─► Layer 3: Layout Context  (Docling PDF parser)
        │     ├── Extracts text blocks with field-label associations
        │     ├── "Employer: GreenTech Inc." → is_high_value=True → bypass ORG gate
        │     ├── "Insurer: Blue Cross" → is_low_value=True → keep context gate
        │     ├── Name heuristic: high-value fields with no NER PERSON hit → looks_like_name() fallback
        │     └── Falls back to PyMuPDF on parse failure
        │
        └─► Layer 4: Visual  (Qwen3-VL-8B, GPU-first)
              ├── Face detection (CenterFace / OpenCV Haar cascade)
              ├── License plate detection (YOLOv8n)
              ├── QR / barcode detection (pyzbar)
              └── VLM: Qwen3-VL-8B-Instruct — grounding mode (bbox_2d JSON output)
                    Output: pixel-accurate bounding boxes (0-1000 normalized → pixel)
                    FP8 on SM≥8.9 (RTX 4090 / H100) | bfloat16 on SM 8.6 (RTX 3090)
                    Benchmark: python scripts/run_benchmark_sroie.py --dataset cord --vlm

Post-processing (all layers):
  → Span NMS: text-span exact dedup (page + text, keep highest specificity)
  → Coordinate IoU dedup: suppress visual boxes with IoU > 0.5
  → Confidence threshold gate: min_confidence = 0.60 (per settings.yaml)
  → Coordinate padding: 4 px expansion to prevent edge bleed-through
```

**Reversible vault (optional):**
- All detected PII values encrypted with AES-256-GCM (Fernet) before redaction
- Token map stored in SQLite vault (`./vault/vault.db`)
- Original values restorable with `pii-redact restore` + the vault key

---

## Benchmark results

Evaluated on [Gretel Finance PII dataset](https://huggingface.co/datasets/gretelai/gretel_pii_finance) (100 docs, text-layer PDFs).

> **Partial span matching** is the correct metric for redaction. If the system detects "Springfield" and the gold label is "Springfield, IL 62701", the PII is caught — what matters for compliance is coverage, not exact boundary alignment.

### Overall (Gretel Finance, 100 docs)

| Mode | Match | P | R | F1 |
|---|---|---|---|---|
| No GLiNER (fast, 7.8s) | Partial | 0.811 | 0.677 | 0.733 |
| **GLiNER enabled (canonical)** | **Partial** | **0.798** | **0.745** | **0.771** |
| No GLiNER (fast, 7.8s) | Exact | 0.343 | 0.376 | 0.359 |

Run with GLiNER: `PII_USE_GLINER=1 CUDA_VISIBLE_DEVICES="" python scripts/run_benchmark_nlp.py --dataset gretel --max-docs 100 --partial`

### Per entity (partial match, with GLiNER — S6)

| Entity | P | R | F1 | Notes |
|---|---|---|---|---|
| US_BANK_NUMBER | 1.000 | 1.000 | **1.000** | S6: ABA 3-7-1 checksum gate |
| EMAIL_ADDRESS | 1.000 | 0.933 | **0.966** | |
| IBAN_CODE | 1.000 | 0.917 | **0.957** | |
| IP_ADDRESS | 0.875 | 1.000 | **0.933** | |
| DATE_TIME | 0.891 | 0.895 | **0.893** | |
| PHONE_NUMBER | 0.739 | 0.850 | **0.791** | S5: EDI/SWIFT lookbehind precision fix |
| CREDIT_CARD | 0.667 | 0.667 | **0.667** | Luhn-validated, 13–19 digit |
| LOCATION | 0.564 | 0.558 | 0.561 | GLiNER lift (was 0.382) |
| ORG | 0.556 | 0.532 | 0.543 | GLiNER lift (was 0.253) |
| PERSON | 0.664 | 0.452 | 0.538 | Recall gap: NER misses single-word names in short text |
| SWIFT_BIC_CODE | 0.200 | 0.500 | 0.286 | Precision gap: 8-char pattern matches abbreviations |

**Sprint history:**

| Sprint | Change | Partial F1 |
|---|---|---|
| S0 | Regex-only baseline | 0.178 |
| S1 | +spaCy NER (Presidio) + langdetect filter | 0.604 |
| S2a | +ORG/PERSON structural quality filters | 0.730 |
| S2b | +Surya OCR, GLiNER, Docling, NMS | 0.732 |
| S3 | +CREDIT_CARD, SWIFT_BIC, US_BANK_NUMBER | 0.732 |
| S4 | +PHONE_NUMBER custom recognizer (R: 0.267→0.611) | 0.738 |
| S5 | +GLiNER benchmarked + PHONE precision fix + _validate_span | 0.772 |
| **S6** | **+ABA checksum (US_BANK P: 0.286→1.000) + ITIN/Passport + StructuredEngine + name heuristic** | **0.771** |

> S6 overall F1 is within noise of S5. The headline gain is US_BANK_NUMBER F1: 0.444 → **1.000** from the ABA checksum gate.

> **Image-path benchmark:** Gretel measures the text-layer NER path. SROIE and CORD measure the OCR+image path:
>
> | Dataset | Mode | Result | Notes |
> |---------|------|--------|-------|
> | SROIE (word crops) | OCR | CER=8.49% | English receipts, Surya |
> | CORD (20 docs) | NER only | F1=0.18 | Indonesian receipts — expected low |
> | CORD (20 docs) | VLM+NER | F1=0.099 | Confirmed across 2 runs. VLM adds FPs without TP gain. Dataset mismatch: store receipts have no personal PII for VLM to ground. |
>
> VLM is better suited to insurance forms and medical records where personal PII is dense and visually structured.

---

## Installation

### Prerequisites

- Python 3.10+
- GPU with 12+ GB VRAM recommended for VLM layer (Qwen3-VL-8B)
- CPU-only mode works; VLM inference is ~5 min/page without GPU

```bash
git clone https://github.com/scamai/PII_ZERO
cd PII_ZERO
pip install -e ".[dev]"
```

### Download model weights

```bash
python scripts/download_models.py
```

Downloads: spaCy `en_core_web_lg`, Surya OCR (auto-downloads on first use), TrOCR, YOLOv8n.

Qwen3-VL must be downloaded separately (requires HuggingFace license acceptance):

```bash
# SM >= 8.9 (RTX 4090, H100, A100): FP8 — ~12 GB VRAM
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct-FP8

# SM 8.6 (RTX 3090): auto-detected, falls back to bfloat16
huggingface-cli download Qwen/Qwen3-VL-8B-Instruct
```

**GLiNER zero-shot NER (optional):**
```bash
pip install gliner
export PII_USE_GLINER=1   # activates at runtime
```

### Verify installation

```bash
pii-redact doctor
```

---

## Usage

### CLI — receipt and invoice images

**Redact a scanned receipt (JPEG/PNG/TIFF):**
```bash
pii-redact redact receipt.jpg --output receipt_redacted.jpg
```

**Redact a scanned invoice PDF:**
```bash
pii-redact redact invoice_scan.pdf --output invoice_redacted.pdf
```

**Dry run — see what would be redacted:**
```bash
pii-redact inspect receipt.jpg --format table
pii-redact inspect invoice.pdf --format json
```

**Batch a folder of receipts:**
```bash
pii-redact redact ./receipts_inbox/ --output ./receipts_redacted/
```

**With VLM layer** (best recall on complex images, requires GPU):
```bash
pii-redact redact receipt.jpg --vlm
```

**With reversible vault** (encrypted backup of original PII for authorized recovery):
```bash
export PII_VAULT_KEY=$(python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())")
pii-redact redact invoice.pdf --vault ./vault/vault.db
pii-redact restore invoice_redacted.pdf --vault ./vault/vault.db
```

### Web UI

```bash
pii-redact ui --port 7860
pii-redact ui --port 7860 --vlm      # with VLM layer
```

Opens a local Gradio interface at `http://127.0.0.1:7860`. Three tabs:
- **Redact** — upload document image or PDF, download redacted output
- **Inspect** — dry run with detection overlay and entity labels
- **Vault / Restore** — restore original PII from encrypted vault

No data is sent to Gradio cloud. The interface runs fully locally.

### Docker

```bash
docker-compose up --build
# UI available at http://localhost:7860
```

---

## Document routing

The pipeline classifies each input before processing and routes it to the appropriate sub-pipeline:

| Document type | OCR | Text NER | Layout | Visual | VLM |
|---|---|---|---|---|---|
| **Receipt / invoice image** | Surya OCR | Presidio | — | face, QR | optional |
| **Scanned bank statement** | Surya OCR | Presidio | — | — | optional |
| PDF (native text layer) | — | Presidio + Docling | Docling | optional | optional |
| Scanned form / TIFF | Surya OCR | Presidio | — | TrOCR | optional |
| ID document photo | Surya OCR | Presidio | — | face | optional |
| Medical record | — | Presidio + scispaCy | Docling | TrOCR | optional |
| Unknown | all layers | all layers | all layers | all | optional |

Form templates (CMS-1500, ACORD 125, ACORD 140) are matched by perceptual hash at 300 DPI. Template-aware processing uses known field coordinates as priors, improving recall on structured forms.

---

## Custom recognizers

17 PatternRecognizers tuned for financial and insurance document formats, registered alongside Presidio built-ins (replaces `PhoneRecognizer`, `UsBankRecognizer`, `CreditCardRecognizer`, `UsItinRecognizer`, `UsPassportRecognizer`):

| Recognizer | Example | Score | Context-gated |
|---|---|---|---|
| SSN | `523-67-4891` | 0.85 | No (prefix gate: never 000/666/9xx) |
| NPI | `1234567893` (10-digit, starts 1 or 2) | 0.65 | Yes |
| EIN | `47-1234567` | 0.75 | Yes |
| ICD-10 | `M54.5`, `Z00.00` | 0.70 | Yes |
| CPT | `99213`, `93000-26` | 0.60 | Yes |
| POLICY_NUM | `POL-7834521` | 0.55–0.90 | Yes |
| US_BANK_NUMBER | `021000021` (ABA routing) | 0.70 | Yes + **ABA checksum** (3-7-1 digit-weighted mod 10) |
| ADJUSTER_ID | `ADJ-4829` | 0.50–0.85 | Yes |
| CLAIM_REF | `CLM-2024-88801` | 0.45–0.90 | Yes |
| DEA_NUM | `AB1234567` | 0.75 | Yes |
| CREDIT_CARD | `4111 1111 1111 1111` (Luhn validated, 13–19 digit) | 0.65 | No (Luhn sufficient) |
| SWIFT_BIC_CODE | `BOFAUS3N` | 0.40 | **Yes — requires "SWIFT/BIC" keyword** |
| CREDIT_CARD_SECURITY_CODE | `CVV: 123` | 0.40 | **Yes — requires "CVV/CVC" keyword** |
| PHONE_NUMBER | `(800) 555-1234`, `+44 20 7946 0958` | 0.60–0.65 | No (area-code validation + EDI/SWIFT lookbehind) |
| ADDRESS | `412 Maple Street` | 0.65 | Yes |
| US_ITIN | `912-34-5678` (starts 9xx) | 0.65 | Yes (replaces built-in score 0.5) |
| US_PASSPORT | `A12345678` (letter + 8 digits) | 0.65 | Yes (replaces built-in score 0.45) |

Context-gating means the base score is below the 0.6 detection threshold; Presidio's context enhancer only passes the threshold when domain keywords appear nearby. The built-in `UsItinRecognizer` (0.5) and `UsPassportRecognizer` (0.45) were silently suppressed by the `min_confidence=0.6` gate — these custom replacements score 0.65 to pass it.

---

## Post-detection entity filters

Structural quality filters in `pii_redact/ner/entity_filters.py` run after Presidio NER to remove false positives:

**ORG filter (`is_valid_org`)** — drops spaCy ORG spans that are:
- Generic role words: "Client", "Vendor", "Service Provider", "Borrower", etc.
- All-caps acronyms 2–6 letters: "CMT", "AI", "HMRC", "PII"
- Technical artifacts: XML schemas, URLs, email addresses, slash-delimited strings
- Short ambiguous names (1–2 words, no legal suffix) without financial context nearby

**PERSON filter (`is_valid_person`)** — drops single-word form field labels misclassified as names: "Email", "Phone", "Address", "Vendor", "Agent", "Manager", etc.

**Name heuristic (`looks_like_name`)** — fallback for Docling high-value fields (Patient Name, Insured, Employer) where NER produces no PERSON hit. Accepts 2–60 character strings composed of letters/spaces/hyphens/periods/apostrophes starting with an uppercase letter. Fires at confidence 0.75.

**Language filter** — drops PERSON/ORG/LOCATION spans where the ±100-character context window is detected as non-English (eliminates foreign-language hallucinations from `en_core_web_lg`).

**Field-context override (Docling)** — when Docling detects a span as the value of a labeled field:
- `is_high_value_field` (Employer, Patient Name, SSN, DOB): bypasses ORG context gate — always redact
- `is_low_value_field` (Insurer, Payer, Court): keeps context gate active — institutional reference, not subject data

---

## Span deduplication (NMS)

Multiple detection layers can flag the same span. Before rendering, two deduplication passes run:

**Pass A — text-span dedup:** Groups boxes by `(page, text_found)`. Keeps the highest-specificity detection per span. Specificity ranking: `US_SSN=100 > CREDIT_CARD=90 > EMAIL_ADDRESS=85 > ... > NRP=40`.

**Pass B — coordinate IoU dedup:** For visual bounding boxes without text anchors, suppresses any box with IoU > 0.5 against a kept box (greedy, confidence-descending).

---

## Configuration

```yaml
# config/settings.yaml
redaction:
  fill_color: [0, 0, 0]        # solid black — never blur or pixelate
  fill_padding_px: 4
  min_confidence: 0.60

pipeline:
  ocr_dpi: 300
  skip_visual_for_structured_pdfs: false

vault:
  db_path: ./vault/vault.db
  key_env: PII_VAULT_KEY        # AES-256-GCM key from env, never written to disk

audit:
  log_dir: ./audit_logs
  include_text: false           # never log PII values — only entity type and confidence
```

---

## Testing

```bash
# Fast tier — pure Python, < 10 seconds, no ML libraries
pytest -m fast

# Slow tier — loads torch + spaCy + Presidio; run in isolation
pytest tests/test_regex_patterns.py
pytest tests/test_smoke_claim_form.py -v    # end-to-end CMS-1500 smoke test

# Never run the unmarked full suite — simultaneous ML library loads cause
# segfaults (torch + cryptography OpenSSL conflict on Python 3.13)
```

### Test coverage

| File | Tests | Tier | What it covers |
|---|---|---|---|
| test_models.py | 15 | fast | RedactionBox / DocumentResult data contracts |
| test_no_blur_safety.py | 10 | fast | Pixel-level: only fills, zero blur output |
| test_redact_engine.py | 30 | fast | Engine safety + PII-in-output guard |
| test_vault.py | 12 | fast | Fernet encrypt/decrypt + audit log |
| test_entity_filters.py | 34 | fast | ORG/PERSON FP filter: 20 TP/FP cases each |
| test_span_nms.py | 10 | fast | IoU dedup + text-span dedup (all edge cases) |
| test_gliner_recognizer.py | 20 | fast | GLiNER interface + _validate_span (mocked model) |
| test_surya_ocr.py | 10 | fast | Surya OCR wrapper (mocked predictors) |
| test_docling_parser.py | 15 | fast | Docling bbox conversion + field classification |
| test_benchmark_sroie.py | 30 | fast | SROIE/CORD benchmark runner (mocked) |
| test_vlm_extractor.py | 20 | fast | VLM bbox_2d grounding parser (no model load) |
| test_benchmark_vlm.py | 15 | fast | VLM+NER benchmark merge/dedup logic |
| test_presidio_setup.py | 12 | slow | AnalyzerEngine singleton + reset + ITIN/Passport + StructuredEngine |
| test_regex_patterns.py | 17 | slow | Determinism tests for all 17 regex recognizers |
| test_smoke_claim_form.py | 11 | slow | End-to-end pipeline on synthetic CMS-1500 PDF |

**Total: 206 passing fast tests, 0 failures.** (`pytest -m fast` — safe anytime, ~15s)

---

## Audit trail

Every processed document generates a JSON audit entry in `./audit_logs/`:

```json
{
  "doc_id": "3f8a1c2d-...",
  "filename": "receipt_2024_0441.jpg",
  "doc_type": "SCAN_FORM",
  "pages": 1,
  "total_processing_ms": 2340.1,
  "entity_counts": {
    "PERSON": 1,
    "CREDIT_CARD": 1,
    "DATE_TIME": 1,
    "LOCATION": 1
  },
  "source_counts": {"surya_ocr": 3, "presidio": 1},
  "detections": [
    {"entity_type": "CREDIT_CARD", "confidence": 0.65, "source": "presidio", "page": 0},
    {"entity_type": "PERSON", "confidence": 0.85, "source": "presidio", "page": 0}
  ]
}
```

PII values are never written to the audit log. The log records that a detection occurred, which entity type, and which layer found it.

---

## Security

- **Solid fill only.** Blur and pixelation are recoverable ([Bishop Fox, 2021](https://bishopfox.com/blog/unredacter-tool-never-use-pixelation-blur)). Every redaction writes a filled black rectangle over both pixel data (raster) and the PDF text object (native PDFs).
- **Vault key never persists.** The AES-256-GCM key is loaded from the `PII_VAULT_KEY` environment variable at runtime. It is never written to disk.
- **No network calls.** All model weights are loaded from local paths. `presidio_analyzer` and `spacy` do not phone home. Gradio UI runs with `share=False`. Surya OCR loads models from the local HuggingFace cache.
- **Audit log is PII-free.** `include_text: false` in settings.yaml. Turning this on for debugging must be done deliberately.

---

## What's been built

Six sprints from blank repo to a 4-layer detection pipeline with F1=0.771 on real financial PII data.

| Sprint | What shipped | F1 |
|---|---|---|
| S0 | Project scaffold: 10 regex recognizers, Presidio setup, Fernet vault, audit log, Click CLI, Gradio UI | 0.178 |
| S1 | Presidio NLP layer, label normalization, langdetect filter, ORG/PERSON structural quality filters | 0.604 → 0.730 |
| S2 | Surya OCR (replaced PaddleOCR), GLiNER integration, Span NMS (2-pass dedup), Docling layout parser | 0.732 |
| S3 | CREDIT_CARD (Luhn, 13–19 digit), SWIFT_BIC_CODE, CVV, US_BANK_NUMBER — financial entity coverage | 0.732 |
| S4 | VLM grounding rewrite (bbox_2d JSON coords), VLM A/B benchmark infra, PHONE_NUMBER custom recognizer (R: 0.267→0.611) | 0.738 |
| S5 | GLiNER benchmarked (first time), PHONE precision fix (EDI/SWIFT lookbehind), _validate_span guard | 0.772 |
| **S6** | **ABA routing checksum (US_BANK_NUMBER P: 0.286→1.000), ITIN/Passport recognizers, StructuredEngine, name heuristic for form fields, VLM GPU fix** | **0.771** |

**4.3× improvement over the regex-only baseline.** The full pipeline is:

- **Surya OCR** — multilingual, MIT-licensed, replaces PaddleOCR. CER=8.49% on SROIE English receipts.
- **Presidio + 17 custom recognizers** — spaCy NER + domain-specific patterns for insurance, financial, and medical entities. Replaces 5 noisy built-ins (PhoneRecognizer, CreditCardRecognizer, UsBankRecognizer, UsItinRecognizer, UsPassportRecognizer).
- **GLiNER zero-shot NER** (`PII_USE_GLINER=1`) — BERT-sized model with plain-English prompts. +0.034 F1 over spaCy alone; lifts ORG 0.253→0.543, LOCATION 0.382→0.561.
- **StructuredEngine** (`PII_USE_STRUCTURED=1`) — presidio-structured wrapping AnalyzerEngine for pandas DataFrame / tabular PII analysis.
- **Docling layout parser** — field-label context routing. High-value fields (Patient Name, Insured) bypass ORG gate and apply name heuristic fallback. Low-value fields (Insurer, Payer) keep context gate.
- **Span NMS** — 2-pass deduplication: text-span exact dedup + coordinate IoU > 0.5 suppression.
- **Qwen3-VL-8B grounding** — visual PII with pixel-accurate bbox_2d coordinates. GPU-first with explicit `device_map={"": 0}` and `torch_dtype=bfloat16` (16 GB on RTX 3090).
- **Fernet vault** — AES-256-GCM encrypted PII backup with SQLite token map; reversible with `pii-redact restore`.
- **206 fast tests** — all pure Python, no ML imports, run in ~15s. Safety-critical: blur guard, PII-in-output guard, vault encrypt/decrypt.

---

## Gaps to final goal

The system reliably catches most structured PII (emails, IBANs, IPs, phone numbers, credit cards). The remaining gaps are in unstructured entity types and image-path coverage.

### Precision gaps

| Entity | P | Root cause | Status |
|---|---|---|---|
| ~~US_BANK_NUMBER~~ | ~~0.286~~ | ~~9-digit ABA pattern too broad~~ | **Fixed S6**: ABA 3-7-1 checksum gate → P=1.000 |
| SWIFT_BIC_CODE | 0.200 | 8-char pattern matches company abbreviations and product codes | Open: needs BIC format gate (first 4 alpha, chars 5-6 = ISO country code) |
| CREDIT_CARD_SECURITY_CODE | 0.000 | 3-digit pattern needs "cvv/cvc" label nearby — Gretel has none | Dataset issue; test on real forms with labeled CVV fields |

### Recall gaps

| Entity | R | Root cause | Status |
|---|---|---|---|
| PERSON | 0.452 | NER misses single-word names on short isolated text; `is_valid_person` filter then blocks them | Partially addressed: `looks_like_name` heuristic added for Docling high-value fields. Flat-text recall still needs a financial-domain NER model. |
| ORG | 0.532 | Quality filter over-suppresses short orgs without legal suffixes | GLiNER helps (0.253→0.543); further gains expected from `nvidia/gliner-PII` |
| LOCATION | 0.558 | spaCy detects city names; gold labels often include full "City, State ZIP" | Partial match already counted; address recognizer covers street addresses |

### Image-path gaps

| Gap | Status | What's needed |
|---|---|---|
| VLM on insurance forms | Not benchmarked | CORD confirmed wrong domain. Validate VLM on CMS-1500 / ACORD insurance form images — personal PII is dense there. |
| WildReceipt | Not started | 1,765 real-world receipt photos; measures robustness to angle, lighting, blur |
| Template-aware forms | Not started | CMS-1500 and ACORD forms have fixed field coordinates — coordinate priors = zero-miss on structured fields |

### Coverage gaps

| Gap | What's missing |
|---|---|
| HIPAA Safe Harbor | 18 identifier categories per 45 CFR §164.514(b)(2) — no per-document coverage report yet |
| Handwriting | TrOCR wired into routing table but not benchmarked; handwritten names/numbers on forms |
| Driver's license fields | US_DRIVER_LICENSE pattern exists but no layout-aware detection for ID document photos |
| MAC_ADDRESS | Listed in entity table; no PatternRecognizer implemented |

### Infrastructure gaps

| Gap | Notes |
|---|---|
| GLiNER runtime | ~1280s/100 docs on CPU; blocks interactive use. Fix: INT8 via `optimum-intel`, or GPU inference |
| Async batch | Sequential per-document. Per-page NER is independent; parallelizable |
| Financial-domain NER | `en_core_web_lg` is news-domain; fine-tuned model on insurance claims would lift PERSON/ORG ~0.10–0.15 |
| TransformersNlpEngine | `obi/deid_roberta_i2b2` (clinical) benchmarked — neutral on financial text. A financial-domain model needed. |

---

## Roadmap

### Active (Sprint 6 complete — targeting Sprint 7)

- **SWIFT_BIC precision** — P=0.200. Add BIC format gate: chars 1–4 must be alpha (bank code), chars 5–6 must match a valid ISO 3166-1 country code. Should raise P without touching recall.
- **`nvidia/gliner-PII`** — swap `urchade/gliner_medium-v2.1` for the purpose-fine-tuned PII model. Expected PERSON/ORG recall lift on insurance forms where generic GLiNER misses domain-specific names.
- **VLM on insurance forms** — CORD benchmark shows VLM adds FPs on store receipts (wrong domain). Validate `--vlm` on CMS-1500 and ACORD form image samples where personal PII is dense.

### Near-term

- **WildReceipt benchmark** — 1,765 real-world receipt photos. More varied than SROIE (angle, lighting, partial occlusion). Measures OCR+NER robustness.
- **GLiNER GPU inference** — INT8 quantization via `optimum-intel`, or GPU runtime. Brings ~1280s/100 docs to <60s for interactive use.
- **Financial-domain NER model** — `en_core_web_lg` is news-domain. A transformer fine-tuned on financial/insurance text (or `PII_USE_TRANSFORMERS=1` with a financial model) would lift PERSON/ORG recall ~0.10–0.15.

### Medium-term

- **HIPAA Safe Harbor report** — per-document coverage report flagging which of the 18 §164.514(b)(2) identifier categories were detected.
- **Template-aware coordinate priors** — CMS-1500 and ACORD forms have fixed field positions. Hard-coded priors = zero-miss on structured forms without needing NER.
- **Async batch processing** — parallel per-page NER; independent pages don't need to wait for each other.
- **Driver's license layout detection** — US_DRIVER_LICENSE pattern exists; add layout-aware detection for ID document photos (MRZ line, field positions).

---

## Acknowledgments

- [Microsoft Presidio](https://github.com/microsoft/presidio) — NER engine and recognizer framework
- [Surya OCR](https://github.com/VikParuchuri/surya) — MIT-licensed multilingual OCR (replaced PaddleOCR)
- [GLiNER](https://github.com/urchade/GLiNER) — zero-shot NER with plain-English entity prompts (NAACL 2024)
- [Docling](https://github.com/DS4SD/docling) — IBM Research layout-aware PDF parser (MIT)
- [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) — visual PII extraction
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) — PDF text extraction and redaction rendering
- [Bishop Fox Unredacter](https://bishopfox.com/blog/unredacter-tool-never-use-pixelation-blur) — the paper that settled the blur debate
- [Gretel Finance PII dataset](https://huggingface.co/datasets/gretelai/gretel_pii_finance) — NLP benchmark dataset
- [SROIE (ICDAR 2019)](https://rrc.cvc.uab.es/?ch=13) — scanned receipt image benchmark
