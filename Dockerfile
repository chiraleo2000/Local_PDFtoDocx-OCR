# PDF-to-DOCX OCR Pipeline — Docker Image
# v1.0.0-beta  |  Thai-optimised OCR (EasyOCR + Thai-TrOCR + Tesseract fallback)  |  Apache-2.0
FROM python:3.12-slim

LABEL maintainer="BeTime"
LABEL version="1.0.0-beta"
LABEL license="Apache-2.0"
LABEL org.opencontainers.image.description="PDF OCR Pipeline v1.0.0-beta — Thai EasyOCR+TrOCR+Tesseract, DOCX layout config, security-hardened"

WORKDIR /app

# GPU build arg: set --build-arg USE_GPU=true for CUDA support
ARG USE_GPU=false

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=7870 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    USE_GPU=${USE_GPU} \
    OCR_ENGINE=easyocr \
    LANGUAGES=tha+eng

# System deps + PyTorch in one layer (S7031 — merge consecutive RUN instructions)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    libgomp1 \
    curl \
    tesseract-ocr \
    tesseract-ocr-tha \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip \
    && if [ "$USE_GPU" = "true" ]; then \
           pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121; \
       else \
           pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
       fi

# Python deps + EasyOCR model pre-download (S7031 — merge consecutive RUN instructions)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir dill \
    && python -c "\
import easyocr; \
reader = easyocr.Reader(['th', 'en'], gpu=False); \
print('EasyOCR Thai+English models downloaded successfully')"

# Application source + YOLO model weights
COPY src/ ./src/
COPY app.py ./
COPY tests/ ./tests/
COPY .env.example ./.env.example
COPY models/ ./models/

# Create non-root user + setup runtime dirs (S7031 — merge consecutive RUN instructions)
RUN groupadd --gid 1001 appuser \
    && useradd --uid 1001 --gid 1001 --create-home appuser \
    && cp .env.example .env \
    && mkdir -p /tmp/pdf_ocr_history /tmp/pdf_ocr_images \
    && mkdir -p /app/correction_data/images /app/correction_data/labels \
    && chown -R appuser:appuser /tmp/pdf_ocr_history /tmp/pdf_ocr_images \
       /app/models /app/correction_data /app

# Correction & training data — mount a volume here for persistence
VOLUME ["/app/correction_data"]

# Switch to non-root user
USER appuser

EXPOSE 7870

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://localhost:7870/ || exit 1

CMD ["python", "app.py"]