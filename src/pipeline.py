"""
OCR Pipeline Orchestrator
PDF -> Page Images -> OpenCV -> Layout Detection -> OCR -> Table -> Image -> LLM -> Export

Includes inline:
  - Text processing (Thai spacing fix, OCR artifact cleanup)
  - LLM post-correction (optional Ollama / Qwen backend)
"""
import os
import re
import logging
import base64
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import fitz  # PyMuPDF

from .preprocessor import OpenCVPreprocessor
from .ocr_engine import OCREngine
from .layout_detector import LayoutDetector, TableExtractor
from .exporter import ImageExtractor, DocumentExporter

logger = logging.getLogger(__name__)

# ── Optional imports for LLM correction ───────────────────────────────────────
_REQUESTS_AVAILABLE = False
try:
    import requests as _requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    pass


# ══════════════════════════════════════════════════════════════════════════════
# Text Processing Helpers
# ══════════════════════════════════════════════════════════════════════════════
_THAI_CHAR_RE = re.compile(r"[\u0E00-\u0E7F]")
_THAI_SPACE_RE = re.compile(r"([\u0E00-\u0E7F])\s+([\u0E00-\u0E7F])")


def _fix_thai_spacing(text: str) -> str:
    """Remove artificial spaces between Thai characters (Tesseract artifact)."""
    lines = []
    for line in text.split("\n"):
        prev = None
        while prev != line:
            prev = line
            line = _THAI_SPACE_RE.sub(r"\1\2", line)
        lines.append(line)
    return "\n".join(lines)


def clean_text(text: str, languages: str = "eng") -> str:
    """Remove OCR artifacts, fix Thai spacing, normalise whitespace."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    if "tha" in languages or _THAI_CHAR_RE.search(text):
        text = _fix_thai_spacing(text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


# ══════════════════════════════════════════════════════════════════════════════
# LLM Post-Correction (optional — disabled by default)
# ══════════════════════════════════════════════════════════════════════════════
class LLMCorrector:
    """Correct OCR text using a vision-language model (optional).
    Supports Ollama and Qwen2.5-VL backends.  Pass-through when disabled."""

    def __init__(self):
        self.enabled = os.getenv("LLM_CORRECTION", "false").lower() == "true"
        self.backend = os.getenv("LLM_BACKEND", "disabled").lower()
        self.model_name = os.getenv("LLM_MODEL", "")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._loaded = False

        if self.enabled and self.backend == "qwen":
            self._try_load_qwen()
        logger.info(f"LLMCorrector — enabled={self.enabled}, backend={self.backend}")

    def _try_load_qwen(self):
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            quantize = os.getenv("LLM_QUANTIZE", "4bit")
            load_kwargs: Dict[str, Any] = {"device_map": "auto", "trust_remote_code": True}
            if quantize == "4bit":
                from transformers import BitsAndBytesConfig
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                load_kwargs["torch_dtype"] = torch.float16
            self._qwen_model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name, **load_kwargs)
            self._qwen_processor = AutoProcessor.from_pretrained(
                self.model_name, trust_remote_code=True)
            self._loaded = True
            logger.info("Qwen2.5-VL loaded")
        except Exception as exc:
            logger.warning(f"Qwen not available: {exc}")

    def correct_text(self, raw_text: str,
                     image_crop: Optional[np.ndarray] = None,
                     ocr_confidence: float = 1.0) -> Dict[str, Any]:
        if not self.enabled or self.backend == "disabled":
            return {"corrected_text": raw_text, "was_corrected": False,
                    "backend_used": "disabled"}
        if ocr_confidence >= 0.85 and len(raw_text.strip()) > 3:
            return {"corrected_text": raw_text, "was_corrected": False,
                    "backend_used": "skipped"}
        if self.backend == "ollama":
            return self._correct_ollama(raw_text, image_crop)
        return {"corrected_text": raw_text, "was_corrected": False,
                "backend_used": "none"}

    def _correct_ollama(self, raw_text: str,
                        image_crop: Optional[np.ndarray]) -> Dict[str, Any]:
        if not _REQUESTS_AVAILABLE:
            return {"corrected_text": raw_text, "was_corrected": False,
                    "backend_used": "unavailable"}
        try:
            prompt = (
                "Fix OCR errors in this text. Return ONLY the corrected text:\n\n"
                f"{raw_text}"
            )
            payload: Dict[str, Any] = {
                "model": self.model_name or "llava",
                "prompt": prompt, "stream": False,
            }
            if image_crop is not None:
                _, buf = cv2.imencode(".png", image_crop)
                payload["images"] = [base64.b64encode(buf.tobytes()).decode()]
            resp = _requests.post(f"{self.ollama_url}/api/generate",
                                  json=payload, timeout=60)
            if resp.status_code == 200:
                corrected = resp.json().get("response", "").strip()
                if corrected:
                    return {"corrected_text": corrected, "was_corrected": True,
                            "backend_used": "ollama"}
        except Exception as exc:
            logger.warning(f"Ollama correction failed: {exc}")
        return {"corrected_text": raw_text, "was_corrected": False,
                "backend_used": "ollama_error"}

    def is_available(self) -> bool:
        if not self.enabled:
            return False
        if self.backend == "qwen":
            return self._loaded
        if self.backend == "ollama":
            return _REQUESTS_AVAILABLE
        return False


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════
class OCRPipeline:
    """Full PDF -> DOCX/TXT/HTML pipeline."""

    QUALITY_MAP = {
        "fast":     {"dpi_scale": 1.5, "quality": "fast"},
        "balanced": {"dpi_scale": 2.0, "quality": "balanced"},
        "accurate": {"dpi_scale": 2.5, "quality": "accurate"},
    }

    def __init__(self):
        quality = os.getenv("QUALITY_PRESET", "balanced")
        self.languages = os.getenv("LANGUAGES", "eng")
        self.preprocessor = OpenCVPreprocessor(quality=quality)
        self.ocr = OCREngine()
        self.layout = LayoutDetector()
        self.table_extractor = TableExtractor()
        self.image_extractor = ImageExtractor()
        self.llm = LLMCorrector()
        self.exporter = DocumentExporter()
        logger.info("OCR Pipeline initialised")

    def process_pdf(self, pdf_path: str, quality: str = "balanced",
                    header_trim: float = 0, footer_trim: float = 0) -> Dict[str, Any]:
        """Process a PDF end-to-end.
        Returns dict with keys: success, text, files, metadata, error."""
        try:
            cfg = self.QUALITY_MAP.get(quality, self.QUALITY_MAP["balanced"])
            scale = cfg["dpi_scale"]
            q = cfg["quality"]

            doc = fitz.open(pdf_path)
            page_count = len(doc)

            all_text_parts: List[str] = []
            all_tables_html: List[str] = []
            all_figures: List[Dict[str, Any]] = []
            total_tables = 0
            total_figures = 0

            for page_num in range(page_count):
                logger.info(f"Processing page {page_num + 1}/{page_count}")
                page = doc[page_num]

                # 1. Render page to high-res image
                mat = fitz.Matrix(scale, scale)
                pix = page.get_pixmap(matrix=mat)
                img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                    pix.h, pix.w, pix.n)
                if img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
                elif img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                h, w = img.shape[:2]

                # Apply header/footer trim
                top_cut = int((header_trim / 100) * h) if header_trim > 0 else 0
                bot_cut = int((footer_trim / 100) * h) if footer_trim > 0 else 0
                if top_cut > 0 or bot_cut > 0:
                    img = img[top_cut:h - max(bot_cut, 0), :]
                h, w = img.shape[:2]

                # 2. OpenCV pre-processing
                preprocessed = self.preprocessor.preprocess(img, quality=q)

                # 3. Layout detection (on original colour image)
                layout_result = self.layout.detect_layout(img, page_num)
                detections = layout_result.get("detections", {})

                # 4. OCR per detected text region
                text_regions = detections.get("text_regions", [])
                page_text_parts: List[str] = []

                if text_regions:
                    text_regions.sort(key=lambda r: r["bbox"][1])
                    for region in text_regions:
                        x0, y0, x1, y1 = [int(v) for v in region["bbox"]]
                        x0, y0 = max(0, x0), max(0, y0)
                        x1, y1 = min(w, x1), min(h, y1)
                        if x1 <= x0 or y1 <= y0:
                            continue
                        crop = preprocessed[y0:y1, x0:x1]
                        if crop.size == 0:
                            continue
                        ocr_result = self.ocr.ocr_image(
                            crop, region_type=region.get("class", "plain text"))
                        raw_text = ocr_result.get("text", "")
                        conf = ocr_result.get("confidence", 0)
                        original_crop = img[y0:y1, x0:x1]
                        corrected = self.llm.correct_text(raw_text, original_crop, conf)
                        text = corrected["corrected_text"]
                        if text.strip():
                            page_text_parts.append(text)
                else:
                    # No layout regions — OCR the full page
                    ocr_result = self.ocr.ocr_full_page(preprocessed)
                    raw_text = ocr_result.get("text", "")
                    conf = ocr_result.get("confidence", 0)
                    corrected = self.llm.correct_text(raw_text, img, conf)
                    text = corrected["corrected_text"]
                    if text.strip():
                        page_text_parts.append(text)

                page_text = "\n".join(page_text_parts)
                page_text = clean_text(page_text, self.languages)
                if page_text.strip():
                    all_text_parts.append(page_text)

                # 5. Table extraction
                table_dets = detections.get("tables", [])
                if table_dets:
                    tables = self.table_extractor.extract_tables(img, table_dets)
                    for t in tables:
                        if t.get("html"):
                            all_tables_html.append(t["html"])
                        if t.get("text"):
                            all_text_parts.append(f"\n[Table]\n{t['text']}\n")
                    total_tables += len(table_dets)

                # 6. Figure extraction
                figure_dets = detections.get("figures", [])
                if figure_dets:
                    figures = self.image_extractor.extract_figures(
                        img, figure_dets, page_num)
                    all_figures.extend(figures)
                    total_figures += len(figure_dets)

            doc.close()

            full_text = "\n\n".join(all_text_parts)

            # Build metadata
            engines = self.ocr.get_available_engines()
            active_engines = [k for k, v in engines.items() if v]
            meta_str = (
                f"Pages: {page_count}\n"
                f"Quality: {quality}\n"
                f"OCR Engine: {', '.join(active_engines)}\n"
                f"Tables: {total_tables}\n"
                f"Figures: {total_figures}\n"
                f"LLM correction: {'enabled' if self.llm.enabled else 'disabled'}"
            )

            files = self.exporter.create_all(
                full_text, all_tables_html, all_figures, meta_str)

            return {
                "success": True,
                "text": full_text,
                "files": files,
                "metadata": {
                    "pages": page_count,
                    "tables": total_tables,
                    "figures": total_figures,
                    "engines": engines,
                    "llm_enabled": self.llm.enabled,
                    "quality": quality,
                },
                "error": None,
            }

        except Exception as exc:
            logger.error(f"Pipeline error: {exc}", exc_info=True)
            return {
                "success": False, "text": "", "files": {},
                "metadata": {}, "error": str(exc),
            }

    def get_status(self) -> Dict[str, Any]:
        return {
            "ocr_engines": self.ocr.get_available_engines(),
            "layout_detector": self.layout.model_loaded,
            "table_extraction": self.table_extractor.enabled,
            "image_extraction": self.image_extractor.enabled,
            "llm_correction": self.llm.is_available(),
            "llm_backend": self.llm.backend,
        }
