# syntax=docker/dockerfile:1
# ============================================================
# LocalOCR - PDF to DOCX Web App
# CPU-first Docker image with optional CUDA and OpenVINO/NPU builds.
# ============================================================
FROM python:3.12-slim-bookworm

LABEL maintainer="chiraleo2000"
LABEL org.opencontainers.image.title="LocalOCR"
LABEL org.opencontainers.image.description="Thai+English PDF OCR, DocLayout-YOLO, Gradio web UI"

WORKDIR /app

# ACCELERATOR values:
#   cpu  - default, CPU-only PyTorch and OCR runtime
#   cuda - NVIDIA CUDA PyTorch wheels; run with Docker NVIDIA runtime
#   npu  - CPU PyTorch plus OpenVINO ONNX Runtime provider for compatible ONNX models
ARG ACCELERATOR=cpu
ARG TORCH_CUDA_INDEX=https://download.pytorch.org/whl/cu121
ARG TORCH_CPU_INDEX=https://download.pytorch.org/whl/cpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    ACCELERATOR=${ACCELERATOR} \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=7870 \
    OCR_ENGINE=easyocr \
    OCR_FALLBACK=paddleocr \
    DISABLE_TROCR_PRELOAD=1 \
    CUDA_DEVICE=0 \
    LANGUAGES=tha+eng \
    YOLO_CONFIDENCE=0.25 \
    YOLO_NMS=0.40 \
    TABLE_DETECTION=true \
    TABLE_ENGINE=paddleocr \
    IMAGE_EXTRACTION=true \
    IMAGE_MIN_AREA=4000 \
    QUALITY_PRESET=accurate \
    MAX_PDF_SIZE_MB=200 \
    HISTORY_RETENTION_DAYS=30 \
    CORRECTION_DATA_DIR=/app/correction_data \
    RETRAIN_INTERVAL=100 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        fontconfig \
        fonts-noto-core \
        fonts-noto-cjk \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-tha \
    && rm -rf /var/lib/apt/lists/* \
    && python -m pip install --upgrade pip setuptools wheel && \
    if [ "$ACCELERATOR" = "cuda" ] || [ "$ACCELERATOR" = "gpu" ]; then \
        python -m pip install torch torchvision --index-url "$TORCH_CUDA_INDEX"; \
    else \
        python -m pip install torch torchvision --index-url "$TORCH_CPU_INDEX"; \
    fi

COPY requirements.txt ./
RUN python -m pip install -r requirements.txt dill && \
    if [ "$ACCELERATOR" = "npu" ]; then \
        python -m pip uninstall -y onnxruntime && \
        python -m pip install onnxruntime-openvino openvino; \
    fi && \
    if [ "$ACCELERATOR" = "cuda" ] || [ "$ACCELERATOR" = "gpu" ]; then \
        printf '%s\n' \
          'export USE_GPU=${USE_GPU:-true}' \
          'export ONNX_PROVIDERS=${ONNX_PROVIDERS:-CUDAExecutionProvider,CPUExecutionProvider}' \
          > /etc/profile.d/localocr-accelerator.sh; \
    elif [ "$ACCELERATOR" = "npu" ]; then \
        printf '%s\n' \
          'export USE_GPU=${USE_GPU:-false}' \
          'export ONNX_PROVIDERS=${ONNX_PROVIDERS:-OpenVINOExecutionProvider,CPUExecutionProvider}' \
          > /etc/profile.d/localocr-accelerator.sh; \
    else \
        printf '%s\n' \
          'export USE_GPU=${USE_GPU:-false}' \
          'export ONNX_PROVIDERS=${ONNX_PROVIDERS:-CPUExecutionProvider}' \
          > /etc/profile.d/localocr-accelerator.sh; \
    fi

COPY src/ ./src/
COPY app.py ./
COPY .env.example ./.env.example
COPY models/ ./models/

RUN python - <<'PY'
from pathlib import Path
import shutil

model_dir = Path('/app/models/DocLayout-YOLO-DocStructBench')
model_dir.mkdir(parents=True, exist_ok=True)
model_path = model_dir / 'doclayout_yolo_docstructbench_imgsz1280_2501.pt'
if model_path.exists():
    print('DocLayout-YOLO model already present:', model_path)
else:
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(
        'juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501',
        'doclayout_yolo_docstructbench_imgsz1280_2501.pt',
    )
    shutil.copy2(cached, model_path)
    print('DocLayout-YOLO model downloaded:', model_path)

import easyocr
easyocr.Reader(['th', 'en'], gpu=False)
print('EasyOCR Thai/English models ready')
PY

ENV LOCALOCR_USER=appuser

RUN groupadd --gid 1001 appuser \
    && useradd --uid 1001 --gid 1001 --create-home appuser \
    && mkdir -p /tmp/pdf_ocr_history /tmp/pdf_ocr_images \
    && mkdir -p /app/correction_data/images /app/correction_data/labels \
    && chown -R appuser:appuser /app /tmp/pdf_ocr_history /tmp/pdf_ocr_images

VOLUME ["/app/correction_data", "/tmp/pdf_ocr_history"]

USER appuser

EXPOSE 7870

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -sf http://localhost:7870/ || exit 1

CMD ["sh", "-lc", ". /etc/profile.d/localocr-accelerator.sh && exec python app.py"]