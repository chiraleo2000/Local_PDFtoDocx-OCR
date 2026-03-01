"""
Multi-Engine OCR Module — v2.2
Thai-optimised cascade: EasyOCR (Thai+English) → Thai-TrOCR (line-level)
  → PaddleOCR (non-Thai fallback) → Tesseract.

Engine priority for Thai text:
    EasyOCR          → Best Thai+English accuracy, built-in line detection
    Thai-TrOCR       → Line-level Thai OCR (ONNX or transformers)
    PaddleOCR        → General multilingual fallback (NO Thai support)
    Tesseract        → Last-resort fallback

Security:
    - Image inputs validated before processing
    - Error messages do not expose internal paths
"""
import os
import sys
import re
import logging
from typing import Optional, List, Dict, Any

# Guard stdout/stderr — PaddlePaddle & EasyOCR read sys.stdout.encoding
if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# --- Tesseract (pytesseract) ---
TESSERACT_AVAILABLE = False
try:
    import pytesseract as _pytesseract
    TESSERACT_AVAILABLE = True
    logger.info("pytesseract available")
except ImportError:
    _pytesseract = None  # type: ignore[assignment]

_MAX_IMAGE_PIXELS = 100_000_000  # 100 megapixels
_THAI_TROCR_HF_REPO = "openthaigpt/thai-trocr"


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


# ── Optional imports ──────────────────────────────────────────────────────────

# --- Thai-TrOCR ONNX ---
THAI_TROCR_AVAILABLE = False
_trocr_session = None
_trocr_processor = None


def _check_thai_trocr():
    """Try to load Thai-TrOCR ONNX model, or auto-download from HuggingFace."""
    global THAI_TROCR_AVAILABLE, _trocr_session, _trocr_processor
    if _trocr_session is not None:
        return
    if os.getenv("DISABLE_TROCR_PRELOAD", "").strip() == "1":
        logger.info("Thai-TrOCR preload disabled (DISABLE_TROCR_PRELOAD=1)")
        THAI_TROCR_AVAILABLE = False
        return
    try:
        import onnxruntime as ort
        model_dir = os.getenv("THAI_TROCR_PATH",
                              os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                           "models", "thai-trocr-onnx"))
        onnx_path = os.path.join(model_dir, "model.onnx")
        if os.path.exists(onnx_path):
            _trocr_session = ort.InferenceSession(
                onnx_path, providers=["CPUExecutionProvider"])
            try:
                from transformers import TrOCRProcessor
                hf_dir = os.getenv("THAI_TROCR_HF_PATH",
                                   os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                "models", "thai-trocr"))
                if os.path.isdir(hf_dir):
                    _trocr_processor = TrOCRProcessor.from_pretrained(hf_dir)
                else:
                    _trocr_processor = TrOCRProcessor.from_pretrained(
                        _THAI_TROCR_HF_REPO)
            except Exception:
                _trocr_processor = None
            THAI_TROCR_AVAILABLE = _trocr_processor is not None
            if THAI_TROCR_AVAILABLE:
                logger.info("Thai-TrOCR ONNX loaded")
            return
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("Thai-TrOCR ONNX load failed: %s", type(exc).__name__)

    # Fallback: try transformers VisionEncoderDecoderModel (local or HuggingFace)
    try:
        from transformers import VisionEncoderDecoderModel, TrOCRProcessor
        hf_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              "models", "thai-trocr")
        if os.path.isdir(hf_dir):
            _trocr_processor = TrOCRProcessor.from_pretrained(hf_dir)
            _trocr_session = VisionEncoderDecoderModel.from_pretrained(hf_dir)
            THAI_TROCR_AVAILABLE = True
            logger.info("Thai-TrOCR loaded (transformers, local)")
        else:
            # Auto-download from HuggingFace
            logger.info("Downloading Thai-TrOCR from HuggingFace (openthaigpt/thai-trocr)...")
            _trocr_processor = TrOCRProcessor.from_pretrained(_THAI_TROCR_HF_REPO)
            _trocr_session = VisionEncoderDecoderModel.from_pretrained(_THAI_TROCR_HF_REPO)
            THAI_TROCR_AVAILABLE = True
            logger.info("Thai-TrOCR downloaded and loaded (transformers)")
    except Exception as exc:
        logger.warning("Thai-TrOCR init failed: %s — %s", type(exc).__name__, exc)
        THAI_TROCR_AVAILABLE = False


# --- PaddleOCR ---
PADDLE_AVAILABLE = False
try:
    from paddleocr import PaddleOCR as _PaddleOCR
    PADDLE_AVAILABLE = True
    logger.info("PaddleOCR available")
except Exception:
    pass


# --- EasyOCR (Thai + multilingual) ---
EASYOCR_AVAILABLE = False
_easyocr_reader = None
try:
    import easyocr as _easyocr_mod
    EASYOCR_AVAILABLE = True
    logger.info("EasyOCR available")
except Exception:
    pass


class OCREngine:
    """Unified multi-engine OCR — Thai-optimised cascade (v2.2).

    Engine priority for Thai text:
        1. easyocr    — EasyOCR Thai+English (best Thai full-page)
        2. thai_trocr  — Thai-TrOCR (line-level Thai)
        3. paddleocr   — PaddleOCR multilingual (no Thai, general fallback)
        4. tesseract   — Last-resort fallback
    """

    def __init__(self) -> None:
        self.use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
        self.primary_engine = os.getenv("OCR_ENGINE", "easyocr").lower()
        self.languages = os.getenv("LANGUAGES", "tha+eng")
        self._paddle_instance = None
        self._easyocr_reader = None
        self._engines_checked = False

        logger.info(
            "OCREngine v2.1 — primary=%s, gpu=%s, lang=%s",
            self.primary_engine, self.use_gpu, self.languages,
        )

    def _ensure_engines(self):
        """Lazy check for available engines."""
        if self._engines_checked:
            return
        _check_thai_trocr()
        self._engines_checked = True

    # ── Lazy loaders ──

    def _get_easyocr(self, languages: Optional[str] = None):
        """Lazily initialise EasyOCR with Thai+English."""
        if self._easyocr_reader is None and EASYOCR_AVAILABLE:
            try:
                lang_list = self._easyocr_langs(languages)
                self._easyocr_reader = _easyocr_mod.Reader(
                    lang_list, gpu=self.use_gpu)
                logger.info("EasyOCR reader created: langs=%s, gpu=%s",
                            lang_list, self.use_gpu)
            except Exception as exc:
                logger.warning("EasyOCR init failed: %s — %s",
                               type(exc).__name__, exc)
        return self._easyocr_reader

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

    # ── Public API ──

    def ocr_image(self, image: np.ndarray,
                  engine_override: Optional[str] = None,
                  languages: Optional[str] = None) -> Dict[str, Any]:
        """Run OCR on an image crop.

        Returns: {"text", "confidence", "engine_used", "lines"}
        """
        if not _validate_image(image):
            return {"text": "", "confidence": 0.0, "engine_used": "none", "lines": []}

        self._ensure_engines()
        engine = (engine_override or self.primary_engine).lower()
        lang = languages or self.languages

        cascade = self._build_cascade(engine)
        for eng in cascade:
            result = self._run_engine(image, eng, lang)
            if result and result.get("text", "").strip():
                return result

        return {"text": "", "confidence": 0.0, "engine_used": "none", "lines": []}

    def ocr_full_page(self, image: np.ndarray,
                      languages: Optional[str] = None) -> Dict[str, Any]:
        """Run OCR on a full page image."""
        if not _validate_image(image):
            return {"text": "", "confidence": 0.0, "engine_used": "none", "lines": []}
        return self.ocr_image(image, languages=languages)

    def _build_cascade(self, requested: str) -> List[str]:
        """Build ordered list of engines to try.

        Thai-optimised: EasyOCR (Thai native) → Thai-TrOCR → PaddleOCR → Tesseract.
        """
        if requested in ("tesseract", "pytesseract"):
            return ["tesseract", "easyocr", "thai_trocr", "paddleocr"]
        elif requested in ("easyocr", "easy"):
            return ["easyocr", "thai_trocr", "paddleocr", "tesseract"]
        elif requested in ("thai_trocr", "trocr"):
            return ["thai_trocr", "easyocr", "paddleocr", "tesseract"]
        elif requested in ("paddleocr", "paddle"):
            return ["paddleocr", "easyocr", "thai_trocr", "tesseract"]
        else:
            # Default: EasyOCR first (best Thai), then Thai-TrOCR, then PaddleOCR, then tesseract
            return ["easyocr", "thai_trocr", "paddleocr", "tesseract"]

    # ── Engine runners ──

    def _run_engine(self, image: np.ndarray, engine: str,
                    languages: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Dispatch to the named engine with error containment."""
        try:
            lang = languages or self.languages
            if engine == "easyocr":
                return self._run_easyocr(image, lang)
            if engine == "thai_trocr":
                return self._run_thai_trocr(image, lang)
            if engine == "paddleocr":
                return self._run_paddle_engine(image, lang)
            if engine == "tesseract":
                return self._run_tesseract(image, lang)
            logger.debug("Unknown engine: %s", engine)
        except (OSError, ValueError, RuntimeError) as exc:
            logger.error("Engine '%s' error: %s — %s", engine, type(exc).__name__, exc)
        return None

    # ── EasyOCR (Thai + multilingual) ──

    def _run_easyocr(self, image: np.ndarray,
                     languages: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Run EasyOCR with Thai+English support."""
        reader = self._get_easyocr(languages)
        if reader is None:
            return None
        try:
            # EasyOCR works on numpy images directly (BGR or grayscale)
            results = reader.readtext(image)
            if not results:
                return None

            lines = []
            all_text = []
            confidences = []
            for (bbox, text, conf) in results:
                lines.append({"text": text, "confidence": float(conf), "bbox": bbox})
                all_text.append(text)
                confidences.append(float(conf))

            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            return {
                "text": "\n".join(all_text),
                "confidence": avg_conf,
                "engine_used": "easyocr",
                "lines": lines,
            }
        except Exception as exc:
            logger.warning("EasyOCR error: %s — %s", type(exc).__name__, exc)
            return None

    # ── Thai-TrOCR ONNX ──

    def _run_thai_trocr(self, image: np.ndarray,
                        _languages: Optional[str] = None) -> Optional[Dict[str, Any]]:
        _check_thai_trocr()
        if not THAI_TROCR_AVAILABLE or _trocr_processor is None:
            return None
        try:
            from PIL import Image as PILImage

            if len(image.shape) == 2:
                pil_img = PILImage.fromarray(image).convert("RGB")
            else:
                pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            pixel_values = _trocr_processor(
                images=pil_img, return_tensors="pt"
            ).pixel_values

            if _trocr_session is not None and hasattr(_trocr_session, 'run'):
                # ONNX runtime needs numpy
                np_values = pixel_values.numpy()
                input_name = _trocr_session.get_inputs()[0].name
                outputs = _trocr_session.run(None, {input_name: np_values})
                if outputs and len(outputs) > 0:
                    text = _trocr_processor.batch_decode(
                        outputs[0], skip_special_tokens=True)[0]
                else:
                    text = ""
            elif _trocr_session is not None and hasattr(_trocr_session, 'generate'):
                # Already a PyTorch tensor from return_tensors="pt"
                generated = _trocr_session.generate(pixel_values, max_new_tokens=512)
                text = _trocr_processor.batch_decode(
                    generated, skip_special_tokens=True)[0]
            else:
                return None

            return {
                "text": text.strip(),
                "confidence": 0.8,
                "engine_used": "thai_trocr",
                "lines": [{"text": text.strip(), "confidence": 0.8}],
            }
        except Exception as exc:
            logger.warning("Thai-TrOCR error: %s", exc)
            return None

    # ── Tesseract (pytesseract) ──

    @staticmethod
    def _parse_tesseract_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse pytesseract image_to_data output into structured result."""
        line_texts: Dict[int, List[str]] = {}
        line_confs: Dict[int, List[float]] = {}

        for i, text in enumerate(data["text"]):
            conf = float(data["conf"][i])
            if conf < 0:  # -1 means no text detected
                continue
            text = text.strip()
            if not text:
                continue
            line_num = data["line_num"][i]
            line_texts.setdefault(line_num, []).append(text)
            line_confs.setdefault(line_num, []).append(conf / 100.0)

        lines: List[Dict[str, Any]] = []
        all_text_parts: List[str] = []
        all_confs: List[float] = []
        for ln in sorted(line_texts.keys()):
            line_str = " ".join(line_texts[ln])
            avg_c = sum(line_confs[ln]) / len(line_confs[ln]) if line_confs[ln] else 0.0
            lines.append({"text": line_str, "confidence": avg_c})
            all_text_parts.append(line_str)
            all_confs.extend(line_confs[ln])

        full_text = "\n".join(all_text_parts)
        if not full_text.strip():
            return None

        avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
        return {
            "text": full_text,
            "confidence": avg_conf,
            "engine_used": "tesseract",
            "lines": lines,
        }

    def _run_tesseract(self, image: np.ndarray,
                       languages: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Run tesseract OCR via pytesseract."""
        if not TESSERACT_AVAILABLE or _pytesseract is None:
            return None
        try:
            from PIL import Image as PILImage

            if len(image.shape) == 2:
                pil_img = PILImage.fromarray(image)
            else:
                pil_img = PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            lang = languages or self.languages
            tess_lang = lang.replace(" ", "")

            data = _pytesseract.image_to_data(
                pil_img, lang=tess_lang, output_type=_pytesseract.Output.DICT
            )
            return self._parse_tesseract_data(data)
        except Exception as exc:
            logger.warning("Tesseract error: %s — %s", type(exc).__name__, exc)
            return None

    # ── PaddleOCR ──

    def _run_paddle_engine(self, image: np.ndarray,
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
            bbox = line_info[0]
            lines.append({"text": text, "confidence": conf, "bbox": bbox})
            all_text.append(text)
            confidences.append(conf)
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {
            "text": "\n".join(all_text),
            "confidence": avg_conf,
            "engine_used": "paddleocr",
            "lines": lines,
        }

    # ── PaddleOCR with position data (for table cell OCR) ──

    def ocr_image_with_positions(self, image: np.ndarray,
                                 languages: Optional[str] = None
                                 ) -> List[Dict[str, Any]]:
        """Return OCR results with bbox positions for each text line.

        Each item: {"text": str, "confidence": float,
                    "bbox": [[x0,y0],[x1,y1],[x2,y2],[x3,y3]]}
        """
        self._ensure_engines()
        paddle = self._get_paddle(languages)
        if paddle is not None:
            try:
                result = paddle.ocr(image, cls=True)
                if result and result[0]:
                    items = []
                    for line_info in result[0]:
                        items.append({
                            "text": line_info[1][0],
                            "confidence": float(line_info[1][1]),
                            "bbox": line_info[0],
                        })
                    return items
            except Exception:
                pass

        res = self.ocr_image(image, languages=languages)
        text = res.get("text", "")
        if text.strip():
            return [{"text": text, "confidence": res.get("confidence", 0.5),
                     "bbox": None}]
        return []

    # ── Language helpers ──

    def _easyocr_langs(self, languages: Optional[str] = None) -> List[str]:
        """Convert pipeline language string to EasyOCR language list."""
        lang = languages or self.languages
        lang_list = []
        mapping = {
            "tha": "th", "eng": "en", "chi_sim": "ch_sim", "chi_tra": "ch_tra",
            "jpn": "ja", "kor": "ko", "ara": "ar", "hin": "hi",
        }
        for part in lang.split("+"):
            part = part.strip().lower()
            if part in mapping:
                lang_list.append(mapping[part])
            elif part in ("th", "en", "ch_sim", "ch_tra", "ja", "ko", "ar"):
                lang_list.append(part)
            elif part == "auto":
                lang_list = ["th", "en"]
                break
        if not lang_list:
            lang_list = ["th", "en"]
        return lang_list

    def _paddle_lang(self, languages: Optional[str] = None) -> str:
        lang = languages or self.languages
        if lang in ("auto", "eng"):
            return "en"
        # PaddleOCR v2.x supported: ch, ch_doc, en, korean, japan,
        # chinese_cht, ta, te, ka, latin, arabic, cyrillic, devanagari
        # NOTE: Thai (tha/th) is NOT supported by PaddleOCR — fallback to 'en'
        mapping = {"tha": "en", "chi_sim": "ch", "jpn": "japan",
                   "kor": "korean", "ara": "arabic"}
        first = lang.split("+")[0]
        return mapping.get(first, "en")

    # ── Status ──

    def get_available_engines(self) -> Dict[str, bool]:
        self._ensure_engines()
        return {
            "tesseract": TESSERACT_AVAILABLE,
            "easyocr": EASYOCR_AVAILABLE,
            "thai_trocr": THAI_TROCR_AVAILABLE,
            "paddleocr": PADDLE_AVAILABLE,
        }

    def is_available(self) -> bool:
        return any(self.get_available_engines().values())
