# PDF-to-DOCX OCR Pipeline

> **v0.2.1** — Open-source, privacy-first OCR pipeline optimised for **Thai + English** documents. Converts scanned PDFs into structured Word (DOCX), TXT, and HTML files — 100% local, no data leaves your machine.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)

---

## Hardware Requirements

| Config | Minimum | Recommended |
| --- | --- | --- |
| **RAM** | 8 GB | 16 GB |
| **GPU VRAM** | None (CPU-only works great) | Optional (CUDA for speed) |
| **Disk** | 10 GB | 20 GB (models + data) |

> **No GPU required.** EasyOCR runs on CPU and produces accurate Thai + English results.

---

## Features

| Feature | Description |
| --- | --- |
| **EasyOCR (Primary Thai)** | EasyOCR 1.7+ — best Thai+English accuracy, no GPU needed, zero setup |
| **Thai-TrOCR (Secondary)** | Auto-downloaded from HuggingFace (`openthaigpt/thai-trocr`) — line-level Thai recogniser |
| **PaddleOCR (Fallback)** | Multilingual fallback for general-purpose documents (CJK, Arabic, etc.) |
| **OpenCV Pre-processing** | Deskew, denoise, binarise, CLAHE contrast enhancement |
| **Layout Detection** | DocLayout-YOLO (AI) or OpenCV contour-based fallback — detects text blocks, tables, figures |
| **Table Extraction** | Grid detection with per-cell OCR, exports as HTML tables in DOCX |
| **Figure Extraction** | Isolates images/charts and embeds them into output documents |
| **Multi-Language** | Thai, English, Chinese, Japanese, Korean, Arabic + more via EasyOCR |
| **Output Formats** | DOCX (styled, with tables & images), TXT, HTML |
| **Web UI** | Gradio interface — upload PDF, configure settings, preview, download |
| **Auth & History** | Simple login system with per-user processing history |
| **Docker Ready** | Single-command deployment with `docker run` |

---

## OCR Engine Strategy

The pipeline uses a **Thai-optimised cascade** — engines tried in order:

```text
Input Region
    │
    ├─ EasyOCR          → Primary (Thai + English, CPU-first, accurate)
    ├─ Thai-TrOCR       → Secondary fallback (auto-downloaded from HuggingFace)
    └─ PaddleOCR        → Final multilingual fallback
```

| Engine | Thai Accuracy | GPU Required | CPU OK | Best For |
| --- | --- | --- | --- | --- |
| **EasyOCR 1.7+** | ★★★★★ | No | ✅ | **Primary — Thai docs, forms, government PDFs** |
| **Thai-TrOCR** | ★★★★☆ | No | ✅ | Line-level Thai text, secondary fallback |
| **PaddleOCR** | ★★★☆☆ | Optional | ✅ | Multilingual general fallback |

---

## Architecture

```text
PDF Input
    │
    ▼
┌──────────────────────────────────────────────┐
│  1. PDF Rendering (PyMuPDF)                  │  High-res page images (configurable DPI)
├──────────────────────────────────────────────┤
│  2. OpenCV Pre-processing                    │  Deskew · Denoise · Binarise · CLAHE
├──────────────────────────────────────────────┤
│  3. Layout Detection                         │  DocLayout-YOLO or OpenCV fallback
│     → text, table, figure, formula regions   │
├──────────────────────────────────────────────┤
│  4. OCR per Region (Thai-Optimised Cascade)  │
│     ├─ EasyOCR          (primary, CPU)        │  Best Thai accuracy, no GPU needed
│     ├─ Thai-TrOCR       (secondary fallback)  │  Line-level, auto-downloaded
│     └─ PaddleOCR                             │  General multilingual fallback
├──────────────────────────────────────────────┤
│  5. Table & Figure Extraction                │  Grid OCR · Image crop & embed
├──────────────────────────────────────────────┤
│  6. Document Export                          │  DOCX · TXT · HTML
└──────────────────────────────────────────────┘
```

---

## Project Structure

```text
├── app.py                 # Gradio web application
├── Dockerfile             # Docker deployment
├── requirements.txt       # Python dependencies
├── models/
│   └── DocLayout-YOLO/    # Layout detection weights (bundled)
├── src/
│   ├── __init__.py        # Package init (version)
│   ├── preprocessor.py    # OpenCV image pre-processing
│   ├── ocr_engine.py      # Multi-engine OCR (EasyOCR / Thai-TrOCR / PaddleOCR)
│   ├── layout_detector.py # Layout detection + table extraction
│   ├── exporter.py        # DOCX/TXT/HTML export + figure embedding
│   ├── pipeline.py        # Main pipeline orchestrator
│   └── services.py        # Authentication + history management
└── tests/
    ├── test_pipeline.py   # Pipeline integration tests
    └── test_ui.py         # UI / server / Docker tests (43 tests total)
```

---

## Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/chiraleo2000/Local_PDFtoDocx-OCR.git
cd Local_PDFtoDocx-OCR

# Install Python dependencies (EasyOCR + all engines included)
pip install -r requirements.txt

# Run the application
python app.py
```

Open <http://127.0.0.1:7870> in your browser.

Default login: `guest` / `guest123`

EasyOCR will automatically download Thai + English model weights on first run (~200 MB).
Thai-TrOCR weights are auto-downloaded from HuggingFace on first use.

### Docker (CPU — recommended)

```bash
# Build
docker build -t pdf-ocr-pipeline:0.2.1 .

# Run (CPU, EasyOCR primary — no GPU needed)
docker run -d --name pdf-ocr -p 7870:7870 \
  -v ./correction_data:/app/correction_data \
  -e LANGUAGES=tha+eng \
  -e OCR_ENGINE=easyocr \
  -e USE_GPU=false \
  --restart unless-stopped \
  pdf-ocr-pipeline:0.2.1
```

### Docker (GPU — optional for speed)

```bash
docker run -d --gpus all --name pdf-ocr -p 7870:7870 \
  -v ./correction_data:/app/correction_data \
  -e LANGUAGES=tha+eng \
  -e OCR_ENGINE=easyocr \
  -e USE_GPU=true \
  --restart unless-stopped \
  pdf-ocr-pipeline:0.2.1
```

---

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `SERVER_PORT` | `7870` | Web server port |
| `SERVER_HOST` | `0.0.0.0` | Bind address (`127.0.0.1` for local only) |
| `OCR_ENGINE` | `easyocr` | Primary OCR engine: `easyocr`, `thai_trocr`, `paddleocr` |
| `LANGUAGES` | `tha+eng` | OCR languages (Thai+English default) |
| `USE_GPU` | `false` | Enable GPU acceleration (CPU works well without it) |
| `QUALITY_PRESET` | `balanced` | Quality level (`fast`, `balanced`, `accurate`) |
| `YOLO_CONFIDENCE` | `0.30` | Layout detection confidence threshold |
| `TABLE_DETECTION` | `true` | Enable table detection and extraction |
| `IMAGE_EXTRACTION` | `true` | Enable figure/image extraction |
| `RETRAIN_INTERVAL` | `100` | Manual corrections before auto-retrain |
| `MAX_PDF_SIZE_MB` | `200` | Maximum upload file size |
| `DEFAULT_USERNAME` | `admin` | Default login username |
| `DEFAULT_PASSWORD` | *(generated)* | Default login password (shown on first run) |
| `DISABLE_TROCR_PRELOAD` | `0` | Set to `1` to skip Thai-TrOCR download at startup |

---

## Testing

All 43 tests pass (as of v0.2.1):

```bash
# Run all tests
pytest tests/ -v

# Pipeline integration tests only
pytest tests/test_pipeline.py -v

# UI / server / Docker tests
pytest tests/test_ui.py -v
```

---

## Usage

1. **Upload** a PDF file via the web UI
2. **Select** quality level (Fast / Balanced / Accurate)
3. **Optionally** set header/footer trim percentages
4. Click **Convert to DOCX**
5. **Download** the Word file, or view extracted text

The pipeline processes each page:

- Renders to high-resolution image
- Applies OpenCV preprocessing
- Detects layout regions (text, tables, figures)
- Runs Thai-optimised OCR cascade on each region
- Exports structured DOCX with tables and embedded images

---

## Development

```bash
# Quick pipeline smoke test
python -c "from src.pipeline import OCRPipeline; p = OCRPipeline(); r = p.process_pdf('tests/testocrtor.pdf'); print(r['success'], len(r['text']), 'chars')"

# Check available OCR engines
python -c "from src.ocr_engine import OCREngine; e = OCREngine(); print(e.get_available_engines())"
```

---

## Changelog

### v0.2.1 (2026-02-28)
- **EasyOCR** replaces Typhoon OCR 3B as primary Thai engine — no GPU required, correct Thai text output verified
- Added **Thai-TrOCR** auto-download from HuggingFace as secondary fallback
- Raised YOLO layout detection confidence from 0.15 → 0.30 for cleaner region detection
- Fixed `gradio` + `huggingface_hub` version incompatibility (removed upper-bound caps)
- Fixed Windows `cp1252` encoding in subprocess calls
- Added `DISABLE_TROCR_PRELOAD` environment variable for faster local server startup
- All **43/43 tests passing** (pipeline + UI + Docker)
- Docker image pre-downloads EasyOCR model weights at build time

### v0.2.0 (2026-02-22)
- Initial release with Thai-optimised multi-engine OCR cascade
- DocLayout-YOLO layout detection
- Gradio web UI with review/correction workflow
- Docker support

---

## License

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for details.

```text
Copyright 2026 BeTime

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
