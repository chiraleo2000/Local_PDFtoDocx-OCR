"""
Multi-Engine OCR Module — v0.3
Supports Tesseract 5 (primary), with optional PaddleOCR and EasyOCR.
Each engine is loaded lazily and fails gracefully.
Runtime language override supported via `languages` parameter.
"""
import os
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ── Optional imports ──────────────────────────────────────────────────────────
TESSERACT_AVAILABLE = False
try:
    import pytesseract
    pytesseract.get_tesseract_version()
    TESSERACT_AVAILABLE = True
    logger.info("Tesseract OCR available")
except Exception as e:
    logger.warning(f"Tesseract not available: {e}")

PADDLE_AVAILABLE = False
try:
    from paddleocr import PaddleOCR as _PaddleOCR
    PADDLE_AVAILABLE = True
    logger.info("PaddleOCR available")
except Exception:
    pass

EASYOCR_AVAILABLE = False
_easyocr = None  # imported lazily to avoid torchvision DLL crash on Windows

def _check_easyocr():
    global EASYOCR_AVAILABLE, _easyocr
    if _easyocr is not None:
        return
    try:
        import easyocr as _er
        _easyocr = _er
        EASYOCR_AVAILABLE = True
        logger.info("EasyOCR available")
    except Exception:
        EASYOCR_AVAILABLE = False


class OCREngine:
    """Unified multi-engine OCR with automatic fallback and runtime language override."""

    def __init__(self):
        self.use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        self.primary_engine = os.getenv("OCR_ENGINE", "tesseract").lower()
        self.fallback_engine = os.getenv("OCR_FALLBACK", "tesseract").lower()
        self.languages = os.getenv("LANGUAGES", "eng")
        self.tess_oem = int(os.getenv("TESSERACT_OEM", "1"))
        self.tess_psm = int(os.getenv("TESSERACT_PSM", "3"))

        self._paddle_instance = None
        self._easyocr_instance = None

        logger.info(f"OCREngine — primary={self.primary_engine}, "
                     f"fallback={self.fallback_engine}, gpu={self.use_gpu}")

    # ── Lazy loaders ──

    def _get_paddle(self, languages: Optional[str] = None):
        if self._paddle_instance is None and PADDLE_AVAILABLE:
            try:
                lang = self._paddle_lang(languages)
                self._paddle_instance = _PaddleOCR(
                    use_angle_cls=True, lang=lang,
                    use_gpu=self.use_gpu, show_log=False,
                )
            except Exception as e:
                logger.warning(f"PaddleOCR init failed: {e}")
        return self._paddle_instance

    def _get_easyocr(self, languages: Optional[str] = None):
        _check_easyocr()
        if self._easyocr_instance is None and EASYOCR_AVAILABLE:
            try:
                langs = self._easyocr_langs(languages)
                self._easyocr_instance = _easyocr.Reader(langs, gpu=self.use_gpu)
            except Exception as e:
                logger.warning(f"EasyOCR init failed: {e}")
        return self._easyocr_instance

    # ── Public API ──

    def ocr_image(self, image: np.ndarray,
                  engine_override: Optional[str] = None,
                  languages: Optional[str] = None) -> Dict[str, Any]:
        """
        Run OCR on an image crop.
        Returns: {"text", "confidence", "engine_used", "lines"}
        """
        engine = engine_override or self.primary_engine
        lang = languages or self.languages

        result = self._run_engine(image, engine, lang)
        if result and result.get("text", "").strip():
            return result

        # Fallback
        fb = self.fallback_engine if engine != self.fallback_engine else "tesseract"
        result = self._run_engine(image, fb, lang)
        if result and result.get("text", "").strip():
            return result

        return {"text": "", "confidence": 0.0, "engine_used": "none", "lines": []}

    def ocr_full_page(self, image: np.ndarray,
                      languages: Optional[str] = None) -> Dict[str, Any]:
        """Run OCR on a full page image."""
        return self.ocr_image(image, languages=languages)

    # ── Engine runners ──

    def _run_engine(self, image: np.ndarray, engine: str,
                    languages: Optional[str] = None) -> Optional[Dict[str, Any]]:
        try:
            lang = languages or self.languages
            if engine == "tesseract":
                return self._run_tesseract(image, lang)
            elif engine == "paddleocr":
                return self._run_paddle(image, lang)
            elif engine == "easyocr":
                return self._run_easyocr(image, lang)
        except Exception as exc:
            logger.error(f"Engine '{engine}' error: {exc}")
        return None

    def _run_tesseract(self, image: np.ndarray,
                       languages: Optional[str] = None) -> Optional[Dict[str, Any]]:
        if not TESSERACT_AVAILABLE:
            return None

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        lang = self._tess_lang(languages)
        config = f"--oem {self.tess_oem} --psm {self.tess_psm}"

        try:
            data = pytesseract.image_to_data(
                gray, lang=lang, config=config,
                output_type=pytesseract.Output.DICT,
            )
        except Exception as exc:
            logger.warning(f"Tesseract image_to_data failed: {exc}")
            try:
                text = pytesseract.image_to_string(gray, lang=lang, config=config)
                return {"text": text.strip(), "confidence": 0.5,
                        "engine_used": "tesseract", "lines": []}
            except Exception:
                return None

        lines_map: Dict[int, List[str]] = {}
        confidences = []
        n = len(data.get("text", []))
        for i in range(n):
            conf = int(data["conf"][i])
            word = data["text"][i].strip()
            if conf < 0 or not word:
                continue
            block = data["block_num"][i]
            par = data["par_num"][i]
            line_num = data["line_num"][i]
            key = block * 10000 + par * 100 + line_num
            lines_map.setdefault(key, []).append(word)
            confidences.append(conf / 100.0)

        text_lines = [" ".join(words) for words in lines_map.values()]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {
            "text": "\n".join(text_lines),
            "confidence": avg_conf,
            "engine_used": "tesseract",
            "lines": [{"text": t, "confidence": avg_conf} for t in text_lines],
        }

    def _run_paddle(self, image: np.ndarray,
                    languages: Optional[str] = None) -> Optional[Dict[str, Any]]:
        paddle = self._get_paddle(languages)
        if paddle is None:
            return None
        result = paddle.ocr(image, cls=True)
        if not result or not result[0]:
            return None
        lines = []
        all_text = []
        confidences = []
        for line_info in result[0]:
            text = line_info[1][0]
            conf = float(line_info[1][1])
            lines.append({"text": text, "confidence": conf})
            all_text.append(text)
            confidences.append(conf)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {
            "text": "\n".join(all_text),
            "confidence": avg_conf,
            "engine_used": "paddleocr",
            "lines": lines,
        }

    def _run_easyocr(self, image: np.ndarray,
                     languages: Optional[str] = None) -> Optional[Dict[str, Any]]:
        reader = self._get_easyocr(languages)
        if reader is None:
            return None
        results = reader.readtext(image)
        if not results:
            return None
        lines = []
        all_text = []
        confidences = []
        for _, text, conf in results:
            lines.append({"text": text, "confidence": float(conf)})
            all_text.append(text)
            confidences.append(float(conf))
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {
            "text": "\n".join(all_text),
            "confidence": avg_conf,
            "engine_used": "easyocr",
            "lines": lines,
        }

    # ── Language helpers ──

    def _paddle_lang(self, languages: Optional[str] = None) -> str:
        lang = languages or self.languages
        if lang in ("auto", "eng"):
            return "en"
        mapping = {"tha": "th", "chi_sim": "ch", "jpn": "japan",
                   "kor": "korean", "ara": "ar"}
        first = lang.split("+")[0]
        return mapping.get(first, "en")

    def _tess_lang(self, languages: Optional[str] = None) -> str:
        lang = languages or self.languages
        return lang if lang != "auto" else "eng"

    def _easyocr_langs(self, languages: Optional[str] = None) -> List[str]:
        lang = languages or self.languages
        if lang in ("auto", "eng"):
            return ["en"]
        mapping = {"eng": "en", "tha": "th", "chi_sim": "ch_sim",
                   "jpn": "ja", "kor": "ko", "ara": "ar"}
        return [mapping.get(l, l) for l in lang.split("+")]

    # ── Status ──

    def get_available_engines(self) -> Dict[str, bool]:
        return {
            "tesseract": TESSERACT_AVAILABLE,
            "paddleocr": PADDLE_AVAILABLE,
            "easyocr": EASYOCR_AVAILABLE,
        }

    def is_available(self) -> bool:
        return any(self.get_available_engines().values())
