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
2. Presidio + spaCy NER with 13 custom recognizers on the OCR output
3. Visual layer for face detection, QR codes, and barcodes
4. Optional Qwen3-VL-8B for holistic image-level PII extraction

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
        │     │     ├── 13 custom PatternRecognizers (insurance + financial domain)
        │     │     └── Presidio built-ins (EMAIL, PHONE, IBAN, IP, SSN, ...)
        │     ├── Post-NER filters (entity_filters.py)
        │     │     ├── ORG: blocklist + acronym gate + tech-char filter + context gate
        │     │     └── PERSON: form-label blocklist + lowercase-start filter
        │     ├── Language filter (langdetect) — drops non-English NER FPs
        │     └── GLiNER zero-shot NER  [opt-in: PII_USE_GLINER=1]
        │           └── urchade/gliner_medium-v2.1 (or nvidia/gliner-PII)
        │
        ├─► Layer 3: Layout Context  (Docling PDF parser)
        │     ├── Extracts text blocks with field-label associations
        │     ├── "Employer: GreenTech Inc." → is_high_value=True → bypass ORG gate
        │     ├── "Insurer: Blue Cross" → is_low_value=True → keep context gate
        │     └── Falls back to PyMuPDF on parse failure
        │
        └─► Layer 4: Visual  (Qwen3-VL-8B, GPU-first)
              ├── Face detection (CenterFace / OpenCV Haar cascade)
              ├── License plate detection (YOLOv8n)
              ├── QR / barcode detection (pyzbar)
              └── VLM: Qwen3-VL-8B-Instruct — holistic PII extraction from image
                    FP8 on SM≥8.9 (RTX 4090 / H100) | bfloat16 on SM 8.6 (RTX 3090)

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

| Match mode | P | R | F1 |
|---|---|---|---|
| Exact span | 0.356 | 0.376 | 0.366 |
| Partial overlap | 0.810 | 0.667 | **0.732** |

### Per entity (partial match)

| Entity | P | R | F1 | Notes |
|---|---|---|---|---|
| EMAIL_ADDRESS | 1.000 | 0.933 | **0.966** | |
| IBAN_CODE | 1.000 | 0.917 | **0.957** | |
| IP_ADDRESS | 0.875 | 1.000 | **0.933** | |
| DATE_TIME | 0.891 | 0.895 | **0.893** | |
| CREDIT_CARD | 0.667 | 0.667 | **0.667** | Luhn-validated; score boosted from 0.5→0.65 |
| PERSON | 0.677 | 0.420 | 0.519 | Recall limited by form-label filter |
| US_BANK_NUMBER | 0.286 | 1.000 | **0.444** | High recall; precision limited by 9-digit pattern |
| LOCATION | 0.426 | 0.349 | 0.384 | |
| PHONE_NUMBER | 0.500 | 0.267 | 0.348 | Format variation (see roadmap) |
| SWIFT_BIC_CODE | 0.200 | 0.500 | **0.286** | Context-gated to avoid all-caps FPs |
| ORG | 0.423 | 0.182 | 0.254 | Context gate trades recall for precision |

**Baseline history:**

| Sprint | Event | Partial F1 |
|---|---|---|
| S0 | Regex-only baseline | 0.178 |
| S1a | +spaCy NER (Presidio) | 0.604 |
| S1b | +langdetect foreign-language filter | 0.604 |
| S2a | +ORG/PERSON structural quality filters | **0.730** |
| S2b | +Surya OCR, GLiNER, Docling, NMS (merged) | 0.732 |
| S3 | +CREDIT_CARD, SWIFT_BIC, US_BANK_NUMBER fixed | **0.732** + new entities |

> **Note on image-path benchmark:** The Gretel dataset measures the *text-layer* NER path. A separate image-path benchmark using SROIE (973 scanned receipt JPEGs) is in progress and will measure Surya OCR + NER end-to-end F1 on real receipt scans.

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

13 PatternRecognizers tuned for financial and insurance document formats, registered alongside Presidio built-ins:

| Recognizer | Example | Score | Context-gated |
|---|---|---|---|
| SSN | `523-67-4891` | 0.85 | No (Luhn-like prefix) |
| NPI | `1234567893` (10-digit, starts 1 or 2) | 0.65 | Yes |
| EIN | `47-1234567` | 0.75 | Yes |
| ICD-10 | `M54.5`, `Z00.00` | 0.70 | Yes |
| CPT | `99213`, `93000-26` | 0.60 | Yes |
| POLICY_NUM | `POL-7834521` | 0.55–0.90 | Yes |
| US_BANK_NUMBER | `021000021` (ABA routing) | 0.70 | Yes |
| ADJUSTER_ID | `ADJ-4829` | 0.50–0.85 | Yes |
| CLAIM_REF | `CLM-2024-88801` | 0.45–0.90 | Yes |
| DEA_NUM | `AB1234567` | 0.75 | Yes |
| CREDIT_CARD | `4111 1111 1111 1111` (Luhn validated) | 0.65 | No (Luhn sufficient) |
| SWIFT_BIC_CODE | `BOFAUS3N` | 0.40 | **Yes — requires "SWIFT/BIC" keyword** |
| CREDIT_CARD_SECURITY_CODE | `CVV: 123` | 0.40 | **Yes — requires "CVV/CVC" keyword** |
| ADDRESS | `412 Maple Street` | 0.65 | Yes |

Context-gating means the base score is below the 0.6 detection threshold; Presidio's context enhancer only passes the threshold when domain keywords appear nearby. This prevents SWIFT codes (which look like ticker symbols) and CVV codes (which look like amounts) from firing on unrelated text.

---

## Post-detection entity filters

Structural quality filters in `pii_redact/ner/entity_filters.py` run after Presidio NER to remove false positives:

**ORG filter (`is_valid_org`)** — drops spaCy ORG spans that are:
- Generic role words: "Client", "Vendor", "Service Provider", "Borrower", etc.
- All-caps acronyms 2–6 letters: "CMT", "AI", "HMRC", "PII"
- Technical artifacts: XML schemas, URLs, email addresses, slash-delimited strings
- Short ambiguous names (1–2 words, no legal suffix) without financial context nearby

**PERSON filter (`is_valid_person`)** — drops single-word form field labels misclassified as names: "Email", "Phone", "Address", "Vendor", "Agent", "Manager", etc.

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
| test_entity_filters.py | 30 | fast | ORG/PERSON FP filter: 20 TP/FP cases each |
| test_span_nms.py | 12 | fast | IoU dedup + text-span dedup (all edge cases) |
| test_docling_parser.py | 16 | fast/slow | Docling bbox conversion + field classification |
| test_gliner_recognizer.py | 13 | fast | GLiNER interface (mocked model) |
| test_surya_ocr.py | 5 | fast/slow | Surya OCR wrapper (blank image + dict shape) |
| test_regex_patterns.py | 26 | slow | Determinism tests for all 13 regex recognizers |
| test_smoke_claim_form.py | 9 | slow | End-to-end pipeline on synthetic CMS-1500 PDF |

**Total: 138 passing fast tests, 0 failures.**

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

## Roadmap

### Active (Sprint 3)

- **SROIE image benchmark** — 973 scanned receipt JPEGs from ICDAR 2019, run through the full Surya OCR → Presidio NER stack. First real measurement of image-path F1 on receipts. In progress.
- **US_BANK_NUMBER precision** — P=0.286 means 3.5× FPs on 9-digit ABA patterns. Context gate tightening or checksum validation needed.
- **PERSON recall** — R=0.420. The ORG context gate is correct but the PERSON filter may be too conservative on single-word given names without context.

### Near-term

- **WildReceipt benchmark** — 1,765 real-world receipt photos under natural lighting and camera angle. More realistic than SROIE (which is clean scans). Measures robustness to image quality variation.
- **PHONE recall** — R=0.267. Format variation: `(312) 555-1234` vs `312.555.1234` vs `+1-312-555-1234`. Broader base pattern + context boosting.
- **VLM benchmark** — Qwen3-VL-8B is wired into the pipeline and confirmed running at 16 s/page on RTX 3090 bfloat16. No F1 numbers yet with VLM on vs off.
- **nvidia/gliner-PII** — currently using `urchade/gliner_medium-v2.1` as fallback. The purpose-fine-tuned PII model should significantly improve PERSON/ORG recall.

### Medium-term

- **HIPAA Safe Harbor coverage report** — 18 identifier categories per 45 CFR §164.514(b)(2). Add per-document coverage report flagging which categories were detected and which are absent.
- **Template-aware coordinate priors** — CMS-1500 and ACORD forms have fixed field positions. Hard-coded coordinate priors would boost recall on structured forms without relying on NER.
- **Async batch processing** — current batch mode is sequential. Multi-worker PDF rendering + NER can run in parallel per-page.
- **Fine-tuned insurance NER** — `en_core_web_lg` is a general-purpose model. A spaCy model fine-tuned on annotated insurance claims would raise PERSON/ORG F1 by ~0.10–0.15.

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
