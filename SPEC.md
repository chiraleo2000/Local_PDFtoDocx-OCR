# PDF-to-DOCX OCR Pipeline — Technical Specification

**Version:** 0.1.1  
**Date:** 2026-02-25  
**License:** Apache-2.0  
**Author:** BeTime

---

## 1. Product Overview

### 1.1 Purpose
A fully local, privacy-first OCR pipeline that converts scanned PDF documents into structured Word (DOCX), TXT, and HTML files. No data leaves the machine — all processing happens on-device.

### 1.2 Key Capabilities
| Capability | Description |
|---|---|
| Multi-Engine OCR | Tesseract 5 (primary), PaddleOCR, EasyOCR with automatic fallback |
| AI Layout Detection | DocLayout-YOLO (YOLOv10-based) with OpenCV contour fallback |
| Table Extraction | Grid detection + per-cell OCR, PPStructure optional |
| Figure Extraction | Auto-crop detected figures, embed as base64 in output |
| Manual Crop & Correct | Interactive bounding-box editor for missed tables/figures |
| Auto Fine-Tuning | Retrains YOLO layout model every N manual corrections |
| Multi-Language | 100+ languages via Tesseract; Thai, Chinese, Japanese, Korean, Arabic built-in |
| Security Hardened | PBKDF2 password hashing, XSS-safe HTML export, path traversal protection |

### 1.3 Non-Goals
- Cloud/remote processing
- Real-time video OCR
- Handwriting recognition (limited EasyOCR support only)

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
│  │ (OpenCV)     │  │ (YOLO/OpenCV)  │  │ (Multi-engine)   │ │
│  └──────┬──────┘  └───────┬────────┘  └────────┬─────────┘ │
│         │                 │                     │           │
│  ┌──────┴──────┐  ┌──────┴────────┐  ┌────────┴─────────┐ │
│  │ Table       │  │ Image         │  │ Document         │ │
│  │ Extractor   │  │ Extractor     │  │ Exporter         │ │
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
4. Region Processing
   ├─ Text regions    → OCR (Tesseract/PaddleOCR/EasyOCR)
   ├─ Table regions   → Grid detection + per-cell OCR
   ├─ Figure regions  → Crop & embed as image (no OCR)
   └─ Manual regions  → User-drawn boxes merged into detection
    │
    ▼
5. Reading-Order Sort              → Column-aware top→bottom, left→right
    │
    ▼
6. Document Export (HTML-first)    → DOCX · TXT · HTML
    │
    ▼
7. Correction Logging              → YOLO-format labels for fine-tuning
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

### 3.2 ocr_engine.py — Multi-Engine OCR

**Engine Priority:** primary → fallback → none

| Engine | Strengths | Languages | GPU Support |
|---|---|---|---|
| Tesseract 5 | Best general accuracy, 100+ languages | `--oem 1 --psm 3` | No |
| PaddleOCR | Superior CJK & multilingual | ch, en, th, ja, ko, ar | Yes |
| EasyOCR | Robust on distorted/low-res text | 80+ languages | Yes |

**API:**
```python
ocr_image(image, engine_override=None, languages=None) → {text, confidence, engine_used, lines}
ocr_full_page(image, languages=None) → {text, confidence, engine_used, lines}
get_available_engines() → {tesseract: bool, paddleocr: bool, easyocr: bool}
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
| `YOLO_CONFIDENCE` | 0.15 | 0.05–0.50 | Detection threshold |
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
  "metadata": {"pages": 3, "tables": 1, "figures": 2, "engines": {}, "quality": "balanced"},
  "error": null
}
```

---

## 6. Deployment

### 6.1 Requirements
- Python 3.10+
- Tesseract OCR 5.x
- 4 GB RAM minimum (8 GB recommended for YOLO)
- Docker (optional)

### 6.2 Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SERVER_PORT` | `7870` | Web server port |
| `SERVER_HOST` | `127.0.0.1` | Bind address |
| `OCR_ENGINE` | `tesseract` | Primary OCR engine |
| `LANGUAGES` | `eng` | OCR language(s) |
| `USE_GPU` | `false` | Enable GPU acceleration |
| `QUALITY_PRESET` | `balanced` | Default quality level |
| `YOLO_CONFIDENCE` | `0.15` | Layout detection threshold |
| `TABLE_DETECTION` | `true` | Enable table extraction |
| `IMAGE_EXTRACTION` | `true` | Enable figure extraction |
| `RETRAIN_INTERVAL` | `100` | Manual corrections before auto-retrain |
| `MAX_PDF_SIZE_MB` | `200` | Maximum upload file size |
| `DEFAULT_USERNAME` | `admin` | Default user account name |
| `DEFAULT_PASSWORD` | *(generated)* | Default user password (shown on first run) |
| `DEBUG_MODE` | `false` | Enable debug/error display |

### 6.3 Docker

```bash
docker build -t pdf-ocr-pipeline:0.1.1 .
docker run -d -p 7870:7870 \
  -v ./correction_data:/app/correction_data \
  -e LANGUAGES=tha+eng \
  pdf-ocr-pipeline:0.1.1
```

---

## 7. Testing

### 7.1 Test Suites

| Suite | File | Coverage |
|---|---|---|
| Pipeline Integration | `tests/test_pipeline.py` | PDF processing, OCR, export, corrections |
| UI / Handlers | `tests/test_ui.py` | Gradio handlers, server health, Docker |

### 7.2 Running Tests

```bash
# All tests
pytest tests/ -v

# Pipeline only
pytest tests/test_pipeline.py -v

# UI only (includes server start/stop)
pytest tests/test_ui.py -v

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
- **Base model:** `models/DocLayout-YOLO-DocStructBench/doclayout_yolo_docstructbench_imgsz1280_2501.pt`
- **Fine-tuned:** `correction_data/finetuned_models/finetuned_YYYYMMDD_HHMMSS.pt`

---

## 9. Changelog

### v0.1.1 (2026-02-25)
- **Security:** PBKDF2 password hashing (260k iterations, random salt)
- **Security:** XSS prevention via `html.escape()` on all HTML output
- **Security:** Path traversal protection on all file operations
- **Security:** Input validation on all public APIs
- **Security:** Tesseract config sanitization
- **Security:** Filename sanitization for correction store
- **Security:** Debug mode disabled by default
- **Code Quality:** SonarQube-compliant code structure
- **Code Quality:** Reduced cognitive complexity across all modules
- **Code Quality:** Type hints on all public methods
- **Code Quality:** Constants extracted from magic strings
- **Feature:** Manual crop/region editor in Review tab
- **Feature:** Interactive bounding box drawing
- **Feature:** Configurable default credentials via environment
- **Feature:** File size validation (200MB default limit)
- **UI:** Improved Gradio theme and layout
- **Docs:** Complete technical specification (this document)

### v0.5.0
- Manual correction + auto-retrain every 100 corrections
- Review & Correct tab
- Training dashboard

### v0.4.0
- Group-first architecture
- HTML-first export
- DocLayout-YOLO integration
