# LocalOCR — PDF to DOCX Converter

**Thai-optimised OCR pipeline** that converts scanned PDF documents into editable DOCX, TXT, and HTML files — all running locally on your machine.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue)
![License](https://img.shields.io/badge/license-Apache--2.0-green)
![Version](https://img.shields.io/badge/version-0.3.0-purple)

## Features

- **Multi-engine OCR**: EasyOCR, PaddleOCR, Tesseract, Thai-TrOCR, Typhoon OCR
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

## Requirements

- Python 3.10+
- CUDA GPU (optional, for faster OCR)
- ~4 GB disk space for models

## Project Structure

```
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
