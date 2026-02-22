# PDF-to-DOCX OCR Pipeline — Docker Image
# v0.1.0-dev  |  Apache-2.0
FROM python:3.12-slim

LABEL maintainer="BeTime"
LABEL version="0.1.0-dev"
LABEL license="Apache-2.0"

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=7870

# System deps: Tesseract 5 + multi-language packs, OpenCV libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng tesseract-ocr-tha \
    tesseract-ocr-chi-sim tesseract-ocr-jpn tesseract-ocr-kor \
    libgl1 libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Application source
COPY src/ ./src/
COPY app.py ./
COPY .env.example ./.env.example
# Setup env and runtime directories
RUN cp .env.example .env \
    && mkdir -p /app/models /tmp/pdf_ocr_history /tmp/pdf_ocr_images \
    && chmod -R 777 /tmp/pdf_ocr_history /tmp/pdf_ocr_images /app/models

EXPOSE 7870

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:7870/ || exit 1

CMD ["python", "app.py"]