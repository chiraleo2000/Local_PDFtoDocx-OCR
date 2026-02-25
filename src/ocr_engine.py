"""
Multi-Engine OCR Module — v1.0
Supports Tesseract 5 (primary), with optional PaddleOCR and EasyOCR.
Each engine is loaded lazily and fails gracefully.
Runtime language override supported via ``languages`` parameter.

Security:
    - Tesseract config parameters sanitised (no shell injection)
    - Image inputs validated before processing
    - Error messages do not expose internal paths
"""
import os
import re
import logging
from typing import Optional, List, Dict, Any
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Config sanitisation: only allow safe characters in Tesseract flags
_SAFE_TESS_CONFIG_RE = re.compile(r"^[a-zA-Z0-9_ .=-]+$")
_MAX_IMAGE_PIXELS = 100_000_000  # 100 megapixels


def _validate_image(image: np.ndarray) -> bool:
    """Return True if *image* is a valid, reasonably sized numpy array."""
    if image is None or not isinstance(image, np.ndarray):
        return False
    if image.size == 0 or image.ndim < 2:
        return False
    h, w = image.shape[:2]
    if h * w > _MAX_IMAGE_PIXELS:
        logger.warning("Image too large: %d x %d pixels", w, h)
        return False
    return True

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
    """Unified multi-engine OCR with automatic fallback and runtime language override.

    Security: OEM/PSM values are range-checked and config strings are sanitised.
    """

    _VALID_OEM_RANGE = range(0, 4)  # 0–3
    _VALID_PSM_RANGE = range(0, 14)  # 0–13

    def __init__(self) -> None:
        self.use_gpu = os.getenv("USE_GPU", "false").lower() == "true"
        self.primary_engine = os.getenv("OCR_ENGINE", "tesseract").lower()
        self.fallback_engine = os.getenv("OCR_FALLBACK", "tesseract").lower()
        self.languages = os.getenv("LANGUAGES", "eng")

        oem_raw = int(os.getenv("TESSERACT_OEM", "1"))
        psm_raw = int(os.getenv("TESSERACT_PSM", "3"))
        self.tess_oem = oem_raw if oem_raw in self._VALID_OEM_RANGE else 1
        self.tess_psm = psm_raw if psm_raw in self._VALID_PSM_RANGE else 3

        self._paddle_instance = None
        self._easyocr_instance = None

        logger.info(
            "OCREngine — primary=%s, fallback=%s, gpu=%s",
            self.primary_engine, self.fallback_engine, self.use_gpu,
        )

    # ── Lazy loaders ──

    def _get_paddle(self, languages: Optional[str] = None):
        """Lazily initialise PaddleOCR."""
        if self._paddle_instance is None and PADDLE_AVAILABLE:
            try:
                lang = self._paddle_lang(languages)
                self._paddle_instance = _PaddleOCR(
                    use_angle_cls=True, lang=lang,
                    use_gpu=self.use_gpu, show_log=False,
                )
            except (ImportError, OSError, RuntimeError) as exc:
                logger.warning("PaddleOCR init failed: %s", type(exc).__name__)
        return self._paddle_instance

    def _get_easyocr(self, languages: Optional[str] = None):
        """Lazily initialise EasyOCR."""
        _check_easyocr()
        if self._easyocr_instance is None and EASYOCR_AVAILABLE:
            try:
                langs = self._easyocr_langs(languages)
                self._easyocr_instance = _easyocr.Reader(langs, gpu=self.use_gpu)
            except (ImportError, OSError, RuntimeError) as exc:
                logger.warning("EasyOCR init failed: %s", type(exc).__name__)
        return self._easyocr_instance

    # ── Public API ──

    def ocr_image(self, image: np.ndarray,
                  engine_override: Optional[str] = None,
                  languages: Optional[str] = None) -> Dict[str, Any]:
        """Run OCR on an image crop.

        Returns: {"text", "confidence", "engine_used", "lines"}
        """
        if not _validate_image(image):
            return {"text": "", "confidence": 0.0, "engine_used": "none", "lines": []}

        engine = (engine_override or self.primary_engine).lower()
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
        if not _validate_image(image):
            return {"text": "", "confidence": 0.0, "engine_used": "none", "lines": []}
        return self.ocr_image(image, languages=languages)

    # ── Engine runners ──

    def _run_engine(self, image: np.ndarray, engine: str,
                    languages: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Dispatch to the named engine with error containment."""
        try:
            lang = languages or self.languages
            if engine == "tesseract":
                return self._run_tesseract(image, lang)
            if engine == "paddleocr":
                return self._run_paddle(image, lang)
            if engine == "easyocr":
                return self._run_easyocr(image, lang)
            logger.warning("Unknown engine: %s", engine)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Engine '%s' error: %s", engine, type(exc).__name__)
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
        # Sanitise config to prevent command injection
        config = f"--oem {self.tess_oem} --psm {self.tess_psm}"
        if not _SAFE_TESS_CONFIG_RE.match(config):
            logger.warning("Unsafe Tesseract config rejected")
            config = "--oem 1 --psm 3"

        try:
            data = pytesseract.image_to_data(
                gray, lang=lang, config=config,
                output_type=pytesseract.Output.DICT,
            )
        except (OSError, RuntimeError) as exc:
            logger.warning("Tesseract image_to_data failed: %s", type(exc).__name__)
            try:
                text = pytesseract.image_to_string(gray, lang=lang, config=config)
                return {"text": text.strip(), "confidence": 0.5,
                        "engine_used": "tesseract", "lines": []}
            except (OSError, RuntimeError):
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
