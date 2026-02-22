# PDF-to-DOCX OCR Pipeline

> **v0.1.0-dev** — Open-source, multi-language OCR pipeline that converts scanned PDF documents into structured Word (DOCX), TXT, and HTML files.

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](Dockerfile)

---

## Features

| Feature | Description |
| --- | --- |
| **Multi-Engine OCR** | Tesseract 5 (primary) with optional PaddleOCR and EasyOCR fallback |
| **OpenCV Pre-processing** | Deskew, denoise, binarise, CLAHE contrast enhancement |
| **Layout Detection** | DocLayout-YOLO (AI) or OpenCV contour-based fallback — detects text blocks, tables, figures |
| **Table Extraction** | Grid detection with per-cell OCR, exports as HTML tables in DOCX |
| **Figure Extraction** | Isolates images/charts and embeds them into output documents |
| **LLM Post-Correction** | Optional Qwen2.5-VL or Ollama-based text correction (GPU recommended) |
| **Multi-Language** | Thai, English, Chinese, Japanese, Korean, Arabic + 100 more via Tesseract |
| **Output Formats** | DOCX (styled, with tables & images), TXT, HTML |
| **Web UI** | Gradio interface — upload PDF, configure settings, preview, download |
| **Auth & History** | Simple login system with per-user processing history |
| **Docker Ready** | Single-command deployment with `docker run` |

---

## Architecture

```text
PDF Input
    │
    ▼
┌──────────────────────────────────┐
│  1. PDF Rendering (PyMuPDF)      │  High-res page images
├──────────────────────────────────┤
│  2. OpenCV Pre-processing        │  Deskew · Denoise · Binarise · CLAHE
├──────────────────────────────────┤
│  3. Layout Detection             │  DocLayout-YOLO or OpenCV fallback
│     → text, table, figure regions│
├──────────────────────────────────┤
│  4. OCR per Region               │  Tesseract / PaddleOCR / EasyOCR
├──────────────────────────────────┤
│  5. LLM Correction (optional)    │  Qwen2.5-VL / Ollama
├──────────────────────────────────┤
│  6. Table & Figure Extraction    │  Grid OCR · Image crop & embed
├──────────────────────────────────┤
│  7. Document Export               │  DOCX · TXT · HTML
└──────────────────────────────────┘
```

---

## Project Structure

```text
├── app.py                 # Gradio web application
├── Dockerfile             # Docker deployment
├── requirements.txt       # Python dependencies
├── .env.example           # Environment configuration template
├── src/
│   ├── __init__.py        # Package init (version)
│   ├── preprocessor.py    # OpenCV image pre-processing
│   ├── ocr_engine.py      # Multi-engine OCR (Tesseract/PaddleOCR/EasyOCR)
│   ├── layout_detector.py # Layout detection + table extraction
│   ├── exporter.py        # DOCX/TXT/HTML export + figure embedding
│   ├── pipeline.py        # Main pipeline orchestrator
│   └── services.py        # Authentication + history management
└── tests/
    └── testocrtor.pdf     # Sample test PDF
```

---

## Quick Start

### Local Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Local_PDFtoDocx-OCR.git
cd Local_PDFtoDocx-OCR

# Install Python dependencies
pip install -r requirements.txt

# Ensure Tesseract is installed
# Windows: https://github.com/tesseract-ocr/tesseract/wiki
# Linux:   sudo apt install tesseract-ocr tesseract-ocr-eng tesseract-ocr-tha

# Configure environment
cp .env.example .env
# Edit .env to set LANGUAGES, OCR_ENGINE, etc.

# Run the application
python app.py
```

Open <http://127.0.0.1:7870> in your browser.

Default login: `guest` / `guest123`

### Docker

```bash
# Build
docker build -t pdf-ocr-pipeline:0.1.0-dev .

# Run
docker run -d --name pdf-ocr -p 7870:7870 pdf-ocr-pipeline:0.1.0-dev

# Open http://localhost:7870
```

---

## Configuration

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Description |
| --- | --- | --- |
| `SERVER_PORT` | `7870` | Web server port |
| `OCR_ENGINE` | `tesseract` | Primary OCR engine (`tesseract`, `paddleocr`, `easyocr`) |
| `LANGUAGES` | `eng` | OCR languages (e.g. `tha+eng`, `chi_sim+eng`) |
| `USE_GPU` | `false` | Enable GPU acceleration |
| `LLM_CORRECTION` | `false` | Enable LLM post-correction |
| `QUALITY_PRESET` | `balanced` | Quality level (`fast`, `balanced`, `accurate`) |
| `TABLE_DETECTION` | `true` | Enable table detection and extraction |
| `IMAGE_EXTRACTION` | `true` | Enable figure/image extraction |

---

## Optional Dependencies

The core pipeline uses Tesseract + OpenCV. For enhanced capabilities:

```bash
# PaddleOCR — better multilingual & handwriting OCR
pip install paddleocr>=2.7.0

# EasyOCR — fallback for distorted text
pip install easyocr>=1.7.0

# DocLayout-YOLO — AI layout detection
pip install doclayout-yolo>=0.0.2

# LLM correction (requires GPU with ≥6 GB VRAM)
pip install torch>=2.0.0 transformers>=4.37.0 accelerate>=0.25.0
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
- Runs OCR on each region
- Exports structured DOCX with tables and embedded images

---

## Development

```bash
# Run tests
python -c "from src.pipeline import OCRPipeline; p = OCRPipeline(); r = p.process_pdf('tests/testocrtor.pdf'); print(r['success'], len(r['text']), 'chars')"

# Check available OCR engines
python -c "from src.ocr_engine import OCREngine; e = OCREngine(); print(e.get_available_engines())"
```

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
