# LocalOCR — PDF to DOCX Converter

**Thai-optimised OCR pipeline** that converts scanned PDF documents into editable DOCX, TXT, and HTML files — all running locally on your machine.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Version](https://img.shields.io/badge/version-0.5.5-purple)

## Features

- **Strict OCR policy**: Thai pages → **Thai-TrOCR** (line-level, per-line positions); all other languages → **PaddleOCR** (PP-OCRv5). EasyOCR/Tesseract available only as explicit overrides.
- **Layout-faithful output**: every OCR line carries its bbox, so DOCX/HTML output matches the structure, spacing, and alignment of the source pages (`LAYOUT_MODE=absolute`)
- **YOLO Layout Detection**: Tables, figures, titles, and text regions via DocLayout-YOLO
- **Multiple Output Formats**: DOCX, TXT, HTML with preserved structure
- **Dark-themed Desktop GUI**: Modern tkinter interface with live PDF preview
- **Web Interface**: Gradio-based UI with Convert / Review / History tabs
- **Correction Learning**: Logs corrections and supports auto-retraining
- **Docker Support**: Full Dockerfile included

## Quick Start

### Windows Installer

```bat
# Download and run install.bat
install.bat
```

It will ask for an installation path, install dependencies, create a desktop shortcut, and launch the app.

### Linux / macOS Installer

```bash
chmod +x install.sh
./install.sh
```

### Manual Run

```bash
# Install dependencies
pip install -r requirements.txt

# Desktop GUI
python gui_app.py

# Web Interface
python app.py
```

### Docker Deploy

The Docker setup is CPU-only by default and exposes the Gradio web app at <http://localhost:7870>.
Default layout backend is **Docling** (TableFormer) with Thai-TrOCR + PaddleOCR for text.

```bash
# CPU-only build and deploy
docker compose up --build -d localocr

# View logs
docker compose logs -f localocr

# Stop
docker compose down
```

Set `LAYOUT_BACKEND=yolo` to roll back to DocLayout-YOLO if needed.

Optional accelerator profiles are included:

```bash
# NVIDIA CUDA GPU. Requires Docker Desktop/Engine with NVIDIA Container Toolkit.
docker compose --profile gpu up --build -d localocr-gpu

# Experimental Intel OpenVINO/NPU path for ONNX Thai-TrOCR models.
# Requires a Linux host with the NPU/iGPU devices exposed to Docker.
docker compose --profile npu up --build -d localocr-npu
```

Direct Docker commands are also supported:

```bash
docker build -t localocr:cpu --build-arg ACCELERATOR=cpu .
docker run --rm -p 7870:7870 -v localocr_correction_data:/app/correction_data localocr:cpu

docker build -t localocr:cuda --build-arg ACCELERATOR=cuda .
docker run --rm --gpus all -p 7870:7870 -e USE_GPU=true localocr:cuda
```

Notes:

- CPU mode is the supported default and works without special hardware.
- GPU mode accelerates PyTorch-based components such as EasyOCR and DocLayout-YOLO when CUDA is available. PaddleOCR GPU support may require a host/CUDA-specific `paddlepaddle-gpu` wheel.
- NPU mode is experimental. It installs OpenVINO ONNX Runtime and sets `ONNX_PROVIDERS=OpenVINOExecutionProvider,CPUExecutionProvider`, but only ONNX-backed models can use that provider.

## Requirements

- Python 3.10+
- CUDA GPU (optional, for faster OCR)
- ~4 GB disk space for models

## Project Structure

```text
├── gui_app.py              # Desktop GUI (tkinter, dark theme)
├── app.py                  # Web GUI (Gradio)
├── install.bat             # Windows installer
├── install.sh              # Linux/macOS installer
├── build_exe.py            # PyInstaller exe builder
├── requirements.txt        # Python dependencies
├── src/
│   ├── pipeline.py         # Main OCR pipeline
│   ├── ocr_engine.py       # Multi-engine OCR manager
│   ├── layout_detector.py  # YOLO + OpenCV layout detection
│   ├── exporter.py         # DOCX/TXT/HTML export
│   ├── preprocessor.py     # Image preprocessing
│   ├── correction_store.py # Correction logging
│   └── services.py         # Auth & history management
├── models/                 # YOLO model weights
├── correction_data/        # User corrections
└── tests/                  # Test suite
```

## License

Apache-2.0
