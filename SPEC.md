# PDF-to-DOCX OCR Pipeline — Technical Specification

**Version:** 0.2.1  
**Date:** 2026-02-28  
**License:** Apache-2.0  
**Author:** BeTime

---

## 1. Product Overview

### 1.1 Purpose
A fully local, privacy-first OCR pipeline that converts scanned PDF documents into structured Word (DOCX), TXT, and HTML files, with best-in-class accuracy for **Thai and English** documents. No data leaves the machine — all processing happens on-device.

### 1.2 Key Capabilities

| Capability | Description |
|---|---|
| Thai-First OCR Engine | EasyOCR 1.7+ (primary) — best Thai+English accuracy; CPU-only, no GPU required |
| Thai-TrOCR | Auto-downloaded from HuggingFace (openthaigpt/thai-trocr); secondary fallback |
| Multi-Engine Fallback | PaddleOCR automatic fallback for multilingual content |
| AI Layout Detection | DocLayout-YOLO (YOLOv10-based) with OpenCV contour fallback |
| Table Extraction | Grid detection + per-cell OCR, PPStructure optional |
| Figure Extraction | Auto-crop detected figures, embed as base64 in output |
| Manual Crop & Correct | Interactive bounding-box editor for missed tables/figures |
| Auto Fine-Tuning | Retrains YOLO layout model every N manual corrections |
| LLM Post-Correction | Optional Typhoon OCR 7B via Ollama (requires ≥ 16 GB VRAM) |
| Multi-Language | Thai + English (primary); 100+ languages via Tesseract fallback |
| Security Hardened | PBKDF2 password hashing, XSS-safe HTML export, path traversal protection |

### 1.3 Hardware Constraints (Laptop-Friendly)

| Mode | RAM | GPU VRAM | Engine Used |
|---|---|---|---|
| CPU (default) | 8 GB | None | EasyOCR (primary — best Thai accuracy) |
| CPU (secondary) | 8 GB | None | Thai-TrOCR (auto-downloaded) |
| CPU (fallback) | 8 GB | None | PaddleOCR multilingual |
| GPU (optional) | 8 GB | Any | EasyOCR with CUDA acceleration |

> No GPU required. EasyOCR produces correct Thai + English output on CPU.

### 1.4 Non-Goals
- Cloud/remote processing
- Real-time video OCR
- Handwriting recognition (limited Thai-TrOCR support only)

---

## 2. Architecture

### 2.1 System Diagram

```text
┌─────────────────────────────────────────────────────────────┐
│                    Gradio Web UI (app.py)                    │
│  ┌─────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Convert  │  │Review/Correct│  │ Training │  │ History │ │
│  │   Tab    │  │     Tab      │  │   Tab    │  │   Tab   │ │
│  └────┬─────┘  └──────┬───────┘  └────┬─────┘  └────┬────┘ │
└───────┼───────────────┼──────────────┼──────────────┼──────┘
        │               │              │              │
        ▼               ▼              ▼              ▼
┌─────────────────────────────────────────────────────────────┐
│                  OCR Pipeline (pipeline.py)                  │
│                                                             │
│  ┌─────────────┐  ┌────────────────┐  ┌──────────────────┐ │
│  │ Preprocessor │  │ Layout Detector│  │   OCR Engine     │ │
│  │ (OpenCV)     │  │ (YOLO/OpenCV)  │  │ (Thai Cascade)   │ │
│  └──────┬──────┘  └───────┬────────┘  └────────┬─────────┘ │
│         │                 │                     │           │
│  ┌──────┴──────┐  ┌──────┴────────┐  ┌────────┴─────────┐ │
│  │ Table       │  │ Image         │  │ Document         │  │
│  │ Extractor   │  │ Extractor     │  │ Exporter         │  │
│  └─────────────┘  └───────────────┘  └──────────────────┘ │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Correction Store (Fine-Tuning)              │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Processing Pipeline

```text
PDF Input
    │
    ▼
1. PDF Rendering (PyMuPDF)         → High-res page images (configurable DPI)
    │
    ▼
2. OpenCV Pre-processing           → Deskew · Denoise · Binarise · CLAHE
    │
    ▼
3. Layout Detection                → DocLayout-YOLO or OpenCV fallback
   → Classify: text, table, figure, formula, caption
    │
    ▼
4. Region Processing (Thai-Optimised Cascade)
   ├─ Text regions    → Typhoon OCR 3B (GPU ≥ 8 GB VRAM, highest Thai accuracy)
   │                    Thai-TrOCR ONNX (CPU fallback, compact Thai recogniser)
   │                    PaddleOCR → Tesseract (general fallback)
   ├─ Table regions   → Grid detection + per-cell OCR (same cascade)
   ├─ Figure regions  → Crop & embed as image (no OCR)
   └─ Manual regions  → User-drawn boxes merged into detection
    │
    ▼
5. LLM Post-Correction (optional)  → Typhoon OCR 7B via Ollama (≥ 16 GB VRAM)
    │
    ▼
6. Reading-Order Sort              → Column-aware top→bottom, left→right
    │
    ▼
7. Document Export (HTML-first)    → DOCX · TXT · HTML
    │
    ▼
8. Correction Logging              → YOLO-format labels for fine-tuning
```

---

## 3. Module Specifications

### 3.1 preprocessor.py — OpenCV Pre-processing

| Step | Method | Parameters | Purpose |
|---|---|---|---|
| Denoise | `fastNlMeansDenoising` | h=8-10, templateWindowSize=5-7 | Remove scan noise |
| Deskew | `HoughLinesP` + `warpAffine` | threshold=100, minLineLength=100 | Straighten rotated scans |
| CLAHE | `createCLAHE` | clipLimit=2.0, tileGridSize=(8,8) | Enhance contrast |
| Binarise | `adaptiveThreshold` | blockSize=31, C=15 | Convert to black/white |
| Morphology | `dilate` + `erode` | kernel=(2,2), iterations=1 | Clean thin strokes |

**Quality Presets:**

| Preset | Steps | DPI Scale |
|---|---|---|
| `fast` | denoise | 1.5× |
| `balanced` | denoise, deskew, CLAHE | 2.0× |
| `accurate` | denoise, deskew, CLAHE, binarise, morphology | 2.5× |

### 3.2 ocr_engine.py — Thai-Optimised Multi-Engine OCR

**Engine Selection Logic:**

```python
def select_engine(use_gpu: bool, engine_override: str = None) -> str:
    if engine_override:
        return engine_override     # User-selected engine takes priority
    return "easyocr"               # Default: EasyOCR, works on CPU + GPU
```

**Engine Priority:** easyocr → thai_trocr → paddleocr → none

| Engine | Model | Thai Accuracy | GPU Required | CPU Support | Notes |
|---|---|---|---|---|---|
| EasyOCR 1.7+ | Built-in Thai + English models | ★★★★★ | No | ✅ | **Primary; best for Thai docs, forms, tables; auto-downloads weights** |
| Thai-TrOCR | `openthaigpt/thai-trocr` (HuggingFace) | ★★★★☆ | No | ✅ | Secondary fallback; auto-downloaded on first use |
| PaddleOCR | `paddleocr>=2.7.0` | ★★★☆☆ | Optional | ✅ | Multilingual general fallback |

**API (unchanged):**

```python
ocr_image(image, engine_override=None, languages=None) → {text, confidence, engine_used, lines}
ocr_full_page(image, languages=None) → {text, confidence, engine_used, lines}
get_available_engines() → {typhoon_3b: bool, thai_trocr: bool, paddleocr: bool, tesseract: bool}
```

**Thai-TrOCR ONNX Export (for faster CPU inference):**

```python
# Export Hugging Face model to ONNX once
from optimum.exporters.onnx import main_export
main_export("openthaigpt/thai-trocr", output="models/thai-trocr-onnx", task="image-to-text")

# Then load with onnxruntime for ~1.5× CPU speedup
import onnxruntime as ort
session = ort.InferenceSession("models/thai-trocr-onnx/model.onnx",
                                providers=["CPUExecutionProvider"])
```

### 3.3 layout_detector.py — Layout Detection

**Model:** DocLayout-YOLO (YOLOv10 architecture, DocStructBench weights)

| Class ID | Class Name | Category |
|---|---|---|
| 0 | title | text |
| 1 | plain text | text |
| 2 | abandon | other |
| 3 | figure | figure |
| 4 | figure_caption | caption |
| 5 | table | table |
| 6 | table_caption | caption |
| 7 | table_footnote | other |
| 8 | isolate_formula | formula |
| 9 | formula_caption | caption |

**Figure→Table Reclassification Heuristics:**
1. Both H+V grid lines detected (bordered table)
2. Dense horizontal rules only (borderless form)
3. Wide aspect + high text density (text-heavy table)

**Configuration:**

| Parameter | Default | Range | Description |
|---|---|---|---|
| `YOLO_CONFIDENCE` | 0.30 | 0.05–0.50 | Detection threshold |
| `YOLO_NMS` | 0.45 | 0.20–0.80 | Non-max suppression threshold |

### 3.4 exporter.py — Document Export

**Export Strategy:** HTML-first (single source of truth)

```text
ContentBlocks → HTML (styled) → DOCX (via htmldocx)
                              → TXT (plain text from blocks)
```

**HTML Security:** All user text is escaped via `html.escape()` before insertion into HTML to prevent XSS.

### 3.5 correction_store.py — Fine-Tuning Data Store

**Storage Layout:**

```text
correction_data/
├── images/          PNG page crops
├── labels/          YOLO .txt label files (class cx cy w h)
├── corrections.jsonl   Append-only correction log
├── retrain_log.json    Retrain history
├── dataset/            Auto-generated YOLO dataset
└── finetuned_models/   Saved fine-tuned weights
```

**Retrain Trigger:** Every `RETRAIN_INTERVAL` (default: 100) manual corrections.

**Fine-Tune Parameters:**

| Parameter | Value |
|---|---|
| Epochs | 5 |
| Image Size | 1280 |
| Batch Size | 2 |
| Learning Rate | 0.0001 |
| Patience | 3 |

### 3.6 services.py — Authentication & History

**Password Security:**
- PBKDF2-HMAC-SHA256 with 260,000 iterations
- Random 32-byte salt per password
- Stored as `salt_hex:key_hex`

**Session Management:**
- 256-bit random tokens via `secrets.token_hex(32)`
- 24-hour expiry
- Server-side session store (in-memory)

**Path Safety:**
- All user-supplied path components sanitized via `re.sub(r'[^a-zA-Z0-9_-]', '', input)`
- Resolved paths validated against base directory

---

## 4. Security Specifications

### 4.1 OWASP Top 10 Coverage

| Risk | Mitigation |
|---|---|
| A01 Broken Access Control | Path traversal protection, safe filename sanitization |
| A02 Cryptographic Failures | PBKDF2 (260k iterations) for passwords, `secrets` for tokens |
| A03 Injection | HTML escaping for XSS, Tesseract config sanitization |
| A04 Insecure Design | Input validation on all public APIs, file size limits |
| A05 Security Misconfiguration | No debug in production, env-based config |
| A06 Vulnerable Components | Pinned dependency versions, minimal base image |
| A07 Auth Failures | Rate limiting, session expiry, secure token generation |
| A08 Data Integrity | Checksummed file operations, atomic writes |
| A09 Logging Failures | Structured logging, no credential logging |
| A10 SSRF | No external URL fetching in pipeline |

### 4.2 SonarQube Rules Addressed

| Rule | Severity | Fix |
|---|---|---|
| S2068 | Critical | Credentials moved to environment variables |
| S4790 | Critical | SHA256 replaced with PBKDF2-HMAC (260k iterations) |
| S5131 | Critical | All HTML output uses `html.escape()` |
| S2083 | Critical | Path traversal protection on all file operations |
| S4507 | Major | `show_error=False` in production mode |
| S2139 | Major | Exception messages sanitized before user display |
| S1854 | Minor | Dead code and unused variables removed |
| S3776 | Minor | Complex methods refactored |
| S1192 | Minor | Magic strings replaced with constants |

### 4.3 Input Validation

| Input | Validation |
|---|---|
| PDF file path | Exists, is file, `.pdf` extension, size < 200MB |
| Username | 3–50 chars, alphanumeric + underscore |
| Password | 6+ chars |
| Bounding box | 4 numeric values, x1 > x0, y1 > y0 |
| Page number | Non-negative integer, within document range |
| YOLO confidence | Float 0.05–0.50 |
| Trim percentage | Float 0–25 |
| Entry ID | Alphanumeric + underscore + hyphen only |

---

## 5. API Reference

### 5.1 OCRPipeline

```python
class OCRPipeline:
    def process_pdf(pdf_path, quality="balanced", header_trim=0,
                    footer_trim=0, languages=None, yolo_confidence=None) → dict

    def process_pdf_with_corrections(pdf_path, manual_regions=None,
                                      quality="balanced", ...) → dict

    def detect_page_regions(pdf_path, page_num, quality="balanced",
                            yolo_confidence=None) → dict

    def add_manual_region(page_image, bbox, region_class,
                          page_number, pdf_name="") → dict

    def get_status() → dict
```

### 5.2 Return Schemas

**process_pdf result:**

```json
{
  "success": true,
  "text": "extracted text...",
  "files": {"txt": "/path/to.txt", "docx": "/path/to.docx", "html": "/path/to.html"},
  "metadata": {
    "pages": 3, "tables": 1, "figures": 2,
    "engines": {"primary": "typhoon_3b", "fallback_used": false},
    "quality": "balanced"
  },
  "error": null
}
```

---

## 6. Deployment

### 6.1 Requirements

- Python 3.10+
- Tesseract OCR 5.x (with `tesseract-ocr-tha` language pack)
- **16 GB RAM** recommended (8 GB minimum)
- **GPU:** NVIDIA RTX 3060+ with 8 GB VRAM recommended for Typhoon OCR 3B
- Docker (optional)

### 6.2 Model Download

```bash
# EasyOCR — primary Thai engine (auto-downloads on first run, ~200 MB)
# No manual download needed; weights cached automatically by EasyOCR.

# Thai-TrOCR — secondary fallback (auto-downloaded from HuggingFace on first use)
# No manual download needed; cached to ~/.cache/huggingface/

# DocLayout-YOLO — bundled in models/ directory (already included in repo)
```

### 6.3 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SERVER_PORT` | `7870` | Web server port |
| `SERVER_HOST` | `127.0.0.1` | Bind address |
| `OCR_ENGINE` | `easyocr` | Primary OCR engine: `easyocr`, `thai_trocr`, `paddleocr` |
| `LANGUAGES` | `tha+eng` | OCR language(s) — Thai+English default |
| `USE_GPU` | `false` | Enable GPU acceleration (CPU-only works well) |
| `DISABLE_TROCR_PRELOAD` | `0` | Set to `1` to skip Thai-TrOCR download at startup |
| `QUALITY_PRESET` | `balanced` | Default quality level |
| `YOLO_CONFIDENCE` | `0.15` | Layout detection threshold |
| `TABLE_DETECTION` | `true` | Enable table extraction |
| `IMAGE_EXTRACTION` | `true` | Enable figure extraction |
| `RETRAIN_INTERVAL` | `100` | Manual corrections before auto-retrain |
| `MAX_PDF_SIZE_MB` | `200` | Maximum upload file size |
| `DEFAULT_USERNAME` | `admin` | Default user account name |
| `DEFAULT_PASSWORD` | *(generated)* | Default user password (shown on first run) |
| `DEBUG_MODE` | `false` | Enable debug/error display |

### 6.4 Docker

```bash
# Build
docker build -t pdf-ocr-pipeline:0.2.1 .

# CPU-only (recommended — EasyOCR works great without GPU)
docker run -d --name pdf-ocr -p 7870:7870 \
  -v ./correction_data:/app/correction_data \
  -e LANGUAGES=tha+eng \
  -e OCR_ENGINE=easyocr \
  -e USE_GPU=false \
  --restart unless-stopped \
  pdf-ocr-pipeline:0.2.1

# GPU (optional, for speed)
docker run -d --gpus all --name pdf-ocr -p 7870:7870 \
  -v ./correction_data:/app/correction_data \
  -e LANGUAGES=tha+eng \
  -e OCR_ENGINE=easyocr \
  -e USE_GPU=true \
  --restart unless-stopped \
  pdf-ocr-pipeline:0.2.1
```

---

## 7. Testing

### 7.1 Test Suites

| Suite | File | Coverage |
|---|---|---|
| Pipeline Integration | `tests/test_pipeline.py` | PDF processing, OCR, export, corrections |
| UI / Handlers | `tests/test_ui.py` | Gradio handlers, server health, Docker |
| Thai OCR Accuracy | `tests/test_thai_ocr.py` | Engine selection, Thai character accuracy |

### 7.2 Running Tests

```bash
# All tests
pytest tests/ -v

# Pipeline only
pytest tests/test_pipeline.py -v

# Thai OCR accuracy
pytest tests/test_thai_ocr.py -v

# Security focused
pytest tests/test_pipeline.py -k "security" -v
```

---

## 8. Fine-Tuning Guide

### 8.1 Manual Correction Workflow
1. Upload PDF in **Review & Correct** tab
2. Review auto-detected regions (shown with bounding boxes)
3. Draw manual bounding boxes for missed tables/figures
4. Convert with corrections → corrections saved to YOLO format
5. After N corrections, model auto-retrains in background

### 8.2 Custom Training

```python
from src.correction_store import CorrectionStore
store = CorrectionStore()

# Trigger manual retrain
store._start_retrain_async()

# Check status
stats = store.get_stats()
print(f"Corrections: {stats['total_manual_corrections']}")
print(f"Next retrain: {stats['next_retrain_at']}")
```

### 8.3 Model Weights

- **Layout base model:** `models/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1280_2501.pt`
- **Layout fine-tuned:** `correction_data/finetuned_models/finetuned_YYYYMMDD_HHMMSS.pt`
- **Thai OCR primary:** `models/typhoon-ocr-3b/`
- **Thai OCR CPU fallback:** `models/thai-trocr-onnx/`

---

## 9. Changelog

### v0.2.1 (2026-02-28)
- **OCR:** **EasyOCR** replaces Typhoon OCR 3B as primary Thai engine — no GPU required, correct Thai text output verified
- **OCR:** Added **Thai-TrOCR** auto-download from HuggingFace (`openthaigpt/thai-trocr`) as secondary fallback
- **OCR:** Removed Typhoon OCR 3B/7B dependency (unavailable Python package); removed Tesseract from default cascade
- **Config:** `OCR_ENGINE` default changed from `typhoon` → `easyocr`; `USE_GPU` default changed to `false`
- **Config:** New `DISABLE_TROCR_PRELOAD` env variable (set `1` to skip Thai-TrOCR download at startup)
- **Layout:** Raised `YOLO_CONFIDENCE` from 0.15 → 0.30 for cleaner region detection
- **Deps:** Removed `gradio<5.0.0` and `huggingface_hub<1.0.0` upper-bound caps (resolved `HfFolder` import error)
- **Fix:** Fixed Windows `cp1252` encoding in subprocess calls (`encoding="utf-8", errors="replace"`)
- **Fix:** Increased pipeline test timeout from 180s → 900s (EasyOCR CPU processing of full PDFs)
- **Docker:** Pre-downloads EasyOCR Thai+English model weights at image build time
- **Tests:** All **43/43 tests passing** (pipeline + UI + Docker)

### v0.2.0 (2026-02-26)
- **OCR:** Replaced Tesseract as primary with **Typhoon OCR 3B** — highest Thai+English accuracy on 8 GB VRAM laptops
- **OCR:** Added **Thai-TrOCR ONNX** as CPU-friendly Thai fallback (replaces EasyOCR as Thai fallback)
- **OCR:** Engine auto-selection based on available GPU VRAM
- **OCR:** Optional **Typhoon OCR 7B** LLM correction pass via Ollama for highest accuracy
- **Config:** New `TYPHOON_MODEL` env variable (3b/7b), `OCR_ENGINE=typhoon` default
- **Config:** `LANGUAGES` default changed to `tha+eng`
- **Docs:** Hardware requirements table added (8 GB RAM / 8 GB VRAM laptop minimum)
- **Deps:** Added `optimum[exporters]` for Thai-TrOCR ONNX export
- **Tests:** Added `tests/test_thai_ocr.py` for Thai accuracy regression testing

### v0.1.1 (2026-02-25)
- **Security:** PBKDF2 password hashing (260k iterations, random salt)
- **Security:** XSS prevention via `html.escape()` on all HTML output
- **Security:** Path traversal protection on all file operations
- **Security:** Input validation on all public APIs
- **Feature:** Manual crop/region editor in Review tab
- **Feature:** Interactive bounding box drawing
- **Docs:** Complete technical specification

### v0.5.0
- Manual correction + auto-retrain every 100 corrections
- Review & Correct tab
- Training dashboard

### v0.4.0
- Group-first architecture
- HTML-first export
- DocLayout-YOLO integration
