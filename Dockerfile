# PDF-to-DOCX OCR Pipeline — Docker Image
# v0.1.1  |  Security-hardened, non-root, CPU-only, YOLO layout  |  Apache-2.0
FROM python:3.12-slim

LABEL maintainer="BeTime"
LABEL version="0.1.1"
LABEL license="Apache-2.0"
LABEL org.opencontainers.image.description="PDF OCR Pipeline — security-hardened"

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
    tesseract-ocr-ara \
    libgl1 libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch CPU-only first (smaller, no CUDA)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu

# Python deps (includes doclayout-yolo)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Application source + YOLO model weights
COPY src/ ./src/
COPY app.py ./
COPY tests/ ./tests/
COPY .env.example ./.env.example
COPY models/ ./models/

# Create non-root user for security (SonarQube S4823)
RUN groupadd --gid 1001 appuser && \
    useradd --uid 1001 --gid 1001 --create-home appuser

# Setup env and runtime directories
RUN cp .env.example .env \
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