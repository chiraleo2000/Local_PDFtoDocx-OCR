# ============================================================
# LocalOCR — PDF to DOCX Web App  v0.3.2
# Thai-optimised OCR | DocLayout-YOLO | EasyOCR + PaddleOCR
# ============================================================
FROM python:3.12-slim

LABEL maintainer="chiraleo2000"
LABEL version="0.3.2"
LABEL org.opencontainers.image.description="LocalOCR v0.3.2 — Thai+English PDF OCR, DocLayout-YOLO, Gradio web UI"

WORKDIR /app

# GPU build arg: --build-arg USE_GPU=true for CUDA support
ARG USE_GPU=false

# ── Runtime environment ───────────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=7870 \
    OCR_ENGINE=easyocr \
    OCR_FALLBACK=paddleocr \
    DISABLE_TROCR_PRELOAD=0 \
    USE_GPU=${USE_GPU} \
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

# ── System dependencies ───────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── PyTorch (CPU or GPU) ──────────────────────────────────────────────────
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$USE_GPU" = "true" ]; then \
        pip install --no-cache-dir torch torchvision \
            --index-url https://download.pytorch.org/whl/cu121; \
    else \
        pip install --no-cache-dir torch torchvision \
            --index-url https://download.pytorch.org/whl/cpu; \
    fi

# ── Python dependencies ───────────────────────────────────────────────────
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir dill

# ── Pre-download EasyOCR models (Thai + English) ──────────────────────────
RUN python -c "\
import easyocr; \
r = easyocr.Reader(['th', 'en'], gpu=False); \
print('EasyOCR models ready')"

# ── Download DocLayout-YOLO model from HuggingFace ────────────────────────
RUN python -c "\
import os, shutil; \
from huggingface_hub import hf_hub_download; \
dest = '/app/models/DocLayout-YOLO-DocStructBench'; \
os.makedirs(dest, exist_ok=True); \
pt = os.path.join(dest, 'doclayout_yolo_docstructbench_imgsz1280_2501.pt'); \
cached = hf_hub_download( \
    'juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501', \
    'doclayout_yolo_docstructbench_imgsz1280_2501.pt'); \
shutil.copy2(cached, pt); \
print('DocLayout-YOLO model ready:', pt)"

# ── Application source ────────────────────────────────────────────────────
COPY src/       ./src/
COPY app.py     ./
COPY .env.example ./.env

# ── Non-root user + runtime directories ──────────────────────────────────
RUN groupadd --gid 1001 appuser \
    && useradd --uid 1001 --gid 1001 --create-home appuser \
    && mkdir -p /tmp/pdf_ocr_history /tmp/pdf_ocr_images \
    && mkdir -p /app/correction_data/images /app/correction_data/labels \
    && chown -R appuser:appuser /app /tmp/pdf_ocr_history /tmp/pdf_ocr_images

VOLUME ["/app/correction_data"]

USER appuser

EXPOSE 7870

HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
    CMD curl -sf http://localhost:7870/ || exit 1

CMD ["python", "app.py"]