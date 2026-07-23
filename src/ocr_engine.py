# pylint: disable=no-member,broad-exception-caught,global-statement,wrong-import-position
# pylint: disable=invalid-name,import-outside-toplevel,missing-function-docstring
"""
Multi-Engine OCR Module — v3.1 (one model per language selection)

    Thai / Both  ("tha"/"th" in LANGUAGES, including tha+eng):
        Thai-TrOCR ONLY — no Paddle cascade. Region crops are segmented
        into single text lines (CV projection), each line recognised by
        TrOCR, with bboxes (+ optional bold) for the exporter.

    English / other (no Thai in LANGUAGES):
        PaddleOCR ONLY — PP-OCRv5 with native per-line bboxes.

No silent fallback to EasyOCR/Tesseract/Typhoon: those engines run only
when explicitly requested via OCR_ENGINE=<name> (or the UI dropdown).
OCR_ENGINE=auto (default) applies the strict policy above.

Security:
    - Image inputs validated before processing
    - Error messages do not expose internal paths
"""
import os
import sys
import logging
import threading
from typing import Optional, List, Dict, Any

from .runtime import model_slot

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
# Pin revision so Hub downloads are reproducible (bandit B615 / supply-chain).
_THAI_TROCR_HF_REVISION = os.getenv(
    "THAI_TROCR_HF_REVISION",
    "ce167339d2eaa48137cd6ed3efd65bac1ec45430",
)
_SKIP_HEAVY_IMPORTS = os.getenv("LOCALOCR_SKIP_HEAVY_IMPORTS", "").strip() == "1"


def _onnx_providers() -> List[str]:
    """Return ONNX Runtime providers requested by the deployment environment."""
    configured = os.getenv("ONNX_PROVIDERS", "").strip()
    if configured:
        providers = [provider.strip() for provider in configured.split(",")]
        return [provider for provider in providers if provider]

    accelerator = os.getenv("ACCELERATOR", "cpu").strip().lower()
    if accelerator in {"cuda", "gpu"}:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if accelerator in {"npu", "openvino"}:
        return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


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


def _ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Return a contiguous 3-channel BGR image for PaddleOCR / OpenCV models.

    PaddleX doc-unwarping / Normalize crashes with
    ``IndexError: tuple index out of range`` on grayscale ``(H, W)`` crops
    (and on ``(H, W, 1)``) because it indexes ``img.shape[2]``.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        return image
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3:
        channels = image.shape[2]
        if channels == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if channels == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if channels == 3:
            return np.ascontiguousarray(image)
    return image


def _segment_text_lines(image: np.ndarray) -> List[List[int]]:  # NOSONAR
    """Segment a (possibly multi-line) text crop into single-line bboxes.

    TrOCR is a SINGLE-LINE recognition model — feeding it a multi-line
    crop produces garbage. This splits the crop into horizontal line
    bands using morphological dilation + contours, then merges fragments
    (Thai tone marks / vowels above & below the baseline) that belong to
    the same visual line.

    Returns a list of ``[x0, y0, x1, y1]`` boxes in crop coordinates,
    sorted top→bottom. Empty list when nothing was found.
    """
    if image is None or image.size == 0:
        return []
    gray = (cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if image.ndim == 3 else image.copy())
    h, w = gray.shape[:2]
    if h < 4 or w < 4:
        return []

    # Binarise: text → white on black (Otsu handles scans + renders)
    _, binary = cv2.threshold(
        gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(binary) == 0:
        return []

    # Connect characters within a line: wide horizontal kernel, slight
    # vertical reach so Thai upper/lower vowel marks join their line.
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (max(12, w // 30), 3))
    dilated = cv2.dilate(binary, kernel, iterations=2)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 8 or bh < 6:                       # specks
            continue
        boxes.append([x, y, x + bw, y + bh])
    if not boxes:
        return []

    # Merge boxes whose vertical ranges overlap (same visual line)
    boxes.sort(key=lambda b: b[1])
    merged: List[List[int]] = [boxes[0]]
    for b in boxes[1:]:
        prev = merged[-1]
        overlap = min(prev[3], b[3]) - max(prev[1], b[1])
        min_h = max(1, min(prev[3] - prev[1], b[3] - b[1]))
        if overlap > 0.45 * min_h:
            prev[0] = min(prev[0], b[0])
            prev[1] = min(prev[1], b[1])
            prev[2] = max(prev[2], b[2])
            prev[3] = max(prev[3], b[3])
        else:
            merged.append(list(b))

    # Pad each line a little so ascenders/descenders aren't clipped
    out = []
    for x0, y0, x1, y1 in merged:
        pad = max(2, int((y1 - y0) * 0.15))
        out.append([max(0, x0 - 2), max(0, y0 - pad),
                    min(w, x1 + 2), min(h, y1 + pad)])
    out.sort(key=lambda b: (b[1], b[0]))
    return out


_MAX_CHUNK_RATIO = 8.0      # max chunk width as a multiple of line height
_MIN_CHUNK_GAP_FRAC = 0.45  # horizontal gap (× line height) that splits chunks
_TROCR_MIN_HEIGHT = 32      # upscale crops shorter than this before TrOCR


def _speed_mode() -> bool:
    return os.getenv("SPEED_MODE", "0").strip().lower() in (
        "1", "true", "yes", "on")


def _join_trocr_chunks(parts: List[str]) -> str:
    """Join TrOCR chunk texts without inserting spaces between Thai glyphs."""
    out = ""
    for part in parts:
        part = (part or "").strip()
        if not part:
            continue
        if not out:
            out = part
            continue
        prev_ch, next_ch = out[-1], part[0]
        if ("\u0e00" <= prev_ch <= "\u0e7f"
                and "\u0e00" <= next_ch <= "\u0e7f"):
            out += part
        else:
            out += " " + part
    return out


def _split_line_into_chunks(gray: np.ndarray,  # NOSONAR
                            box: List[int],
                            max_ratio: Optional[float] = None) -> List[List[int]]:
    """Split one text-line box into SMALL word/phrase chunks for TrOCR.

    TrOCR resizes its input to a fixed square patch grid — wide lines get
    horizontally squashed and mis-read. Splitting each line on horizontal
    whitespace gaps (and force-splitting over-wide chunks at the emptiest
    column) keeps every crop at a readable aspect ratio.

    Returns chunk boxes ``[x0, y0, x1, y1]`` in page coordinates,
    ordered left→right.
    """
    x0, y0, x1, y1 = [int(v) for v in box]
    h = max(1, y1 - y0)
    line = gray[y0:y1, x0:x1]
    if line.size == 0:
        return [list(box)]
    _, binary = cv2.threshold(line, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink = (binary > 0).sum(axis=0)

    # Ink runs — column ranges that contain text
    runs: List[List[int]] = []
    start = None
    for i, v in enumerate(ink):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(ink)])
    if not runs:
        return [list(box)]

    # Merge runs separated by small gaps (intra-word/character gaps)
    gap_thr = max(int(h * _MIN_CHUNK_GAP_FRAC), 6)
    chunks = [runs[0][:]]
    for s, e in runs[1:]:
        if s - chunks[-1][1] <= gap_thr:
            chunks[-1][1] = e
        else:
            chunks.append([s, e])

    # Force-split chunks wider than max_ratio × height,
    # cutting at the column with the least ink near the split point
    ratio = float(max_ratio) if max_ratio is not None else _MAX_CHUNK_RATIO
    max_w = max(int(h * ratio), 24)
    final: List[List[int]] = []
    for s, e in chunks:
        while e - s > max_w:
            target = s + max_w
            lo = max(s + h, target - h)
            hi = min(e - h, target + h)
            if lo >= hi:
                cut = target
            else:
                cut = lo + int(np.argmin(ink[lo:hi]))
            final.append([s, cut])
            s = cut
        final.append([s, e])

    pad = max(2, h // 10)
    out: List[List[int]] = []
    for s, e in final:
        if e - s < 3:                      # specks
            continue
        out.append([max(x0, x0 + s - pad), y0,
                    min(x1, x0 + e + pad), y1])
    return out or [list(box)]


# ── Optional imports ──────────────────────────────────────────────────────────

# --- Thai-TrOCR ONNX ---
THAI_TROCR_AVAILABLE = False
_trocr_session = None
_trocr_processor = None


def _check_thai_trocr(preload: bool = False):  # NOSONAR
    """Try to load Thai-TrOCR ONNX model, or auto-download from HuggingFace.

    DISABLE_TROCR_PRELOAD=1 only skips EAGER loading at startup
    (``preload=True``); lazy loading at first use always proceeds —
    previously this flag disabled Thai-TrOCR entirely, which silently
    broke Thai OCR in the web app.
    """
    global THAI_TROCR_AVAILABLE, _trocr_session, _trocr_processor
    if _trocr_session is not None:
        return
    if preload and os.getenv("DISABLE_TROCR_PRELOAD", "").strip() == "1":
        logger.info("Thai-TrOCR preload skipped (DISABLE_TROCR_PRELOAD=1) — "
                    "will load lazily on first Thai page")
        return
    try:
        import onnxruntime as ort
        model_dir = os.getenv("THAI_TROCR_PATH",
                              os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                           "models", "thai-trocr-onnx"))
        onnx_path = os.path.join(model_dir, "model.onnx")
        if os.path.exists(onnx_path):
            providers = _onnx_providers()
            _trocr_session = ort.InferenceSession(
                onnx_path, providers=providers)
            try:
                from transformers import TrOCRProcessor
                hf_dir = os.getenv("THAI_TROCR_HF_PATH",
                                   os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                                "models", "thai-trocr"))
                if os.path.isdir(hf_dir):
                    _trocr_processor = TrOCRProcessor.from_pretrained(  # nosec B615
                        hf_dir, local_files_only=True)
                else:
                    _trocr_processor = TrOCRProcessor.from_pretrained(
                        _THAI_TROCR_HF_REPO,
                        revision=_THAI_TROCR_HF_REVISION)
            except Exception:
                _trocr_processor = None
            THAI_TROCR_AVAILABLE = _trocr_processor is not None
            if THAI_TROCR_AVAILABLE:
                logger.info("Thai-TrOCR ONNX loaded with providers=%s", providers)
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
            _trocr_processor = TrOCRProcessor.from_pretrained(  # nosec B615
                hf_dir, local_files_only=True)
            _trocr_session = VisionEncoderDecoderModel.from_pretrained(  # nosec B615
                hf_dir, local_files_only=True)
            THAI_TROCR_AVAILABLE = True
            logger.info("Thai-TrOCR loaded (transformers, local)")
        else:
            # Auto-download from HuggingFace (pinned revision)
            logger.info(
                "Downloading Thai-TrOCR from HuggingFace "
                "(openthaigpt/thai-trocr@%s)...",
                _THAI_TROCR_HF_REVISION[:12])
            _trocr_processor = TrOCRProcessor.from_pretrained(
                _THAI_TROCR_HF_REPO, revision=_THAI_TROCR_HF_REVISION)
            _trocr_session = VisionEncoderDecoderModel.from_pretrained(
                _THAI_TROCR_HF_REPO, revision=_THAI_TROCR_HF_REVISION)
            THAI_TROCR_AVAILABLE = True
            logger.info("Thai-TrOCR downloaded and loaded (transformers)")
        # Optional CUDA for transformers TrOCR (keep CPU under 1.5GB budgets)
        _maybe_move_trocr_to_device()
    except Exception as exc:
        logger.warning("Thai-TrOCR init failed: %s — %s", type(exc).__name__, exc)
        THAI_TROCR_AVAILABLE = False


def _trocr_torch_device() -> str:
    """Device for transformers Thai-TrOCR.

    Explicit ``TROCR_DEVICE`` wins. Otherwise CUDA when ``USE_GPU=true``.
    Docling layout typically holds ~0.4GB; TrOCR fits under a 1.5GB cap.
    """
    raw = (os.getenv("TROCR_DEVICE") or "").strip().lower()
    if raw in ("cpu", "cuda", "gpu"):
        return "cpu" if raw == "cpu" else "cuda"
    use_gpu = (os.getenv("USE_GPU") or "").strip().lower() == "true"
    return "cuda" if use_gpu else "cpu"


def _maybe_move_trocr_to_device() -> None:
    global _trocr_session
    if _trocr_session is None or not hasattr(_trocr_session, "to"):
        return
    device = _trocr_torch_device()
    if device != "cuda":
        logger.info("Thai-TrOCR device=cpu")
        return
    ensure_trocr_on_cuda()


def ensure_trocr_on_cuda() -> bool:
    """Move Thai-TrOCR to CUDA if configured; return True when on CUDA."""
    global _trocr_session
    if _trocr_session is None or not hasattr(_trocr_session, "to"):
        return False
    if _trocr_torch_device() != "cuda":
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        cur = next(_trocr_session.parameters()).device
        if cur.type == "cuda":
            return True
        torch.cuda.empty_cache()
        _trocr_session = _trocr_session.to("cuda")
        _trocr_session.eval()
        logger.info("Thai-TrOCR moved to CUDA")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("Thai-TrOCR CUDA move failed: %s", type(exc).__name__)
        return False


def cascade_trocr_to_cuda() -> bool:
    """After Docling finishes, free VRAM and put Thai-TrOCR on CUDA.

    Sharing Docling+TrOCR on a 1.5GB card makes Thai garbage. Cascade
    (layout first, then OCR) keeps both under the cap and is far faster
    than CPU TrOCR bands.
    """
    global _trocr_session
    _check_thai_trocr(preload=True)
    if _trocr_session is None or not hasattr(_trocr_session, "to"):
        return False
    try:
        import torch
        if not torch.cuda.is_available():
            return False
        # Drop Docling / other CUDA caches before loading TrOCR
        torch.cuda.empty_cache()
        cur = next(_trocr_session.parameters()).device
        if cur.type == "cuda":
            logger.info("Thai-TrOCR already on CUDA (cascade)")
            return True
        _trocr_session = _trocr_session.to("cuda")
        _trocr_session.eval()
        logger.info("Thai-TrOCR cascade → CUDA")
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("TrOCR cascade CUDA failed: %s", type(exc).__name__)
        return False


def trocr_to_cpu() -> None:
    """Move Thai-TrOCR back to CPU to free VRAM for Docling."""
    global _trocr_session
    if _trocr_session is None or not hasattr(_trocr_session, "to"):
        return
    try:
        import torch
        cur = next(_trocr_session.parameters()).device
        if cur.type == "cpu":
            return
        _trocr_session = _trocr_session.to("cpu")
        torch.cuda.empty_cache()
        logger.info("Thai-TrOCR moved back to CPU")
    except Exception as exc:  # noqa: BLE001
        logger.debug("TrOCR→CPU failed: %s", type(exc).__name__)


# --- PaddleOCR ---
PADDLE_AVAILABLE = False
_PaddleOCR = None
if not _SKIP_HEAVY_IMPORTS:
    try:
        from paddleocr import PaddleOCR as _PaddleOCR
        PADDLE_AVAILABLE = True
        logger.info("PaddleOCR available")
    except Exception:
        pass


# --- EasyOCR (Thai + multilingual) ---
EASYOCR_AVAILABLE = False
_easyocr_reader = None
_easyocr_mod = None
if not _SKIP_HEAVY_IMPORTS:
    try:
        import easyocr as _easyocr_mod
        EASYOCR_AVAILABLE = True
        logger.info("EasyOCR available")
    except Exception:
        pass


# --- Typhoon OCR (SCB10X — best Thai document OCR) ---
TYPHOON_AVAILABLE = False
_typhoon_ocr_document = None
if not _SKIP_HEAVY_IMPORTS:
    try:
        from typhoon_ocr import ocr_document as _typhoon_ocr_document
        TYPHOON_AVAILABLE = True
        logger.info("Typhoon OCR package available")
    except Exception:
        pass


def _typhoon_configured() -> bool:
    """Typhoon OCR is usable when the package is installed AND either an
    API key (api.opentyphoon.ai) or a self-hosted endpoint is configured."""
    return TYPHOON_AVAILABLE and bool(
        os.getenv("TYPHOON_OCR_API_KEY")
        or os.getenv("TYPHOON_BASE_URL")
        or os.getenv("OPENAI_API_KEY"))


def _paddle_version_major() -> int:
    """Return the installed paddleocr major version (0 when unknown)."""
    try:
        import paddleocr
        return int(str(getattr(paddleocr, "__version__", "0")).split(".")[0])
    except Exception:
        return 0


def _paddle_supports_thai() -> bool:
    """PP-OCRv5 Thai model (lang="th") requires paddleocr >= 3.x."""
    return _paddle_version_major() >= 3


def _detect_line_bold(gray_crop: np.ndarray) -> bool:
    """OpenCV stroke-width / ink-density heuristic for bold glyphs."""
    if gray_crop is None or not isinstance(gray_crop, np.ndarray):
        return False
    if gray_crop.size == 0:
        return False
    if gray_crop.ndim == 3:
        gray = cv2.cvtColor(gray_crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = gray_crop
    h, w = gray.shape[:2]
    if h < 4 or w < 8:
        return False
    try:
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = int(cv2.countNonZero(binary))
        if ink < 12:
            return False
        density = ink / float(max(binary.size, 1))
        dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        strokes = dist[binary > 0]
        if strokes.size == 0:
            return False
        mean_sw = float(np.mean(strokes))
        rel = mean_sw / float(max(h, 1))
        # Bold: denser ink and thicker strokes relative to line height
        return density >= 0.20 and rel >= 0.075
    except Exception:  # noqa: BLE001
        return False


class OCREngine:
    """Unified OCR — one model per language selection (v3.1).

    Thai / Both:  Thai-TrOCR only (line-segmented, per-line bboxes + bold)
    Other langs:  PaddleOCR only (PP-OCRv5, native per-line bboxes)

    EasyOCR / Tesseract / Typhoon run ONLY when explicitly requested
    via OCR_ENGINE=<name> or an engine_override.
    """

    def __init__(self) -> None:
        self.use_gpu = os.getenv("USE_GPU", "true").lower() == "true"
        self.primary_engine = os.getenv("OCR_ENGINE", "auto").lower()
        self.languages = os.getenv("LANGUAGES", "tha+eng")
        self._paddle_instances: Dict[str, Any] = {}
        self._easyocr_reader = None
        self._engines_checked = False
        self._init_lock = threading.Lock()

        logger.info(
            "OCREngine v3.1 — primary=%s, gpu=%s, lang=%s "
            "(Thai/Both→Thai-TrOCR only, other→PaddleOCR only)",
            self.primary_engine, self.use_gpu, self.languages,
        )

    def _ensure_engines(self):
        """Lazy check for available engines (thread-safe)."""
        if self._engines_checked:
            return
        with self._init_lock:
            if self._engines_checked:
                return
            _check_thai_trocr()
            self._engines_checked = True

    # ── Lazy loaders ──

    def _get_easyocr(self, languages: Optional[str] = None):
        """Lazily initialise EasyOCR with Thai+English."""
        if self._easyocr_reader is not None or not EASYOCR_AVAILABLE:
            return self._easyocr_reader
        with self._init_lock:
            if self._easyocr_reader is not None or not EASYOCR_AVAILABLE:
                return self._easyocr_reader
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

    def _paddle_device(self) -> str:
        """Pick Paddle device. Low VRAM budgets keep Paddle on CPU so Docling
        can use the GPU; CPU-only paddle wheels also force cpu."""
        if not self.use_gpu:
            return "cpu"
        try:
            import paddle
            if not paddle.is_compiled_with_cuda():
                logger.warning(
                    "paddlepaddle is CPU-only — install paddlepaddle-gpu "
                    "for CUDA builds (Paddle stays on CPU for now)")
                return "cpu"
        except Exception:  # noqa: BLE001
            return "cpu"
        raw = os.getenv("MAX_VRAM_MB", "").strip()
        if raw:
            try:
                if float(raw) <= 2048:
                    logger.info(
                        "MAX_VRAM_MB=%s — keeping PaddleOCR on CPU so "
                        "Docling/TableFormer can use the GPU", raw)
                    return "cpu"
            except ValueError:
                pass
        return "gpu"

    def _get_paddle(self, languages: Optional[str] = None):
        """Lazily initialise PaddleOCR per language (2.x and 3.x APIs).

        Skipped when SKIP_PADDLE_PRELOAD=1 and languages include Thai
        (Thai/Both use Thai-TrOCR only — keeps 1.5GB VRAM free).
        """
        if not PADDLE_AVAILABLE:
            return None
        lang_sel = languages or self.languages
        if (self._is_thai(lang_sel)
                and os.getenv("SKIP_PADDLE_PRELOAD", "1").strip() != "0"
                and self.primary_engine in ("auto", "thai_trocr", "trocr")):
            logger.debug(
                "PaddleOCR skipped for Thai/Both (Thai-TrOCR only policy)")
            return None
        lang = self._paddle_lang(languages)
        with self._init_lock:
            if lang in self._paddle_instances:
                return self._paddle_instances[lang]
            instance = None
            device = self._paddle_device()
            # CPU oneDNN/PIR path crashes on some paddle wheels during OCR
            # (ConvertPirAttribute2RuntimeAttribute). Force classic executor.
            os.environ.setdefault("FLAGS_use_mkldnn", "0")
            os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")
            try:
                if _paddle_version_major() >= 3:
                    # 3.x: disable doc unwarping/orientation — they crash on
                    # grayscale/small region crops and we already deskew in
                    # OpenCVPreprocessor.
                    kwargs = {
                        "lang": lang,
                        "use_textline_orientation": True,
                        "device": device,
                        "use_doc_unwarping": False,
                        "use_doc_orientation_classify": False,
                        "enable_mkldnn": False,
                    }
                    try:
                        instance = _PaddleOCR(**kwargs)
                    except TypeError:
                        # Older 3.x without those kwargs
                        kwargs.pop("use_doc_unwarping", None)
                        kwargs.pop("use_doc_orientation_classify", None)
                        kwargs.pop("enable_mkldnn", None)
                        instance = _PaddleOCR(**kwargs)
                else:
                    instance = _PaddleOCR(
                        use_angle_cls=True, lang=lang,
                        use_gpu=(device == "gpu"), show_log=False,
                    )
                logger.info(
                    "PaddleOCR initialised (v%d.x, lang=%s, device=%s)",
                    max(_paddle_version_major(), 2), lang, device)
            except (ImportError, OSError, RuntimeError, TypeError) as exc:
                logger.warning("PaddleOCR init failed: %s", type(exc).__name__)
            if instance is not None:
                self._paddle_instances[lang] = instance
            return instance

    @staticmethod
    def _parse_paddle_result(result) -> List[Dict[str, Any]]:  # NOSONAR
        """Normalise PaddleOCR output (2.x nested lists OR 3.x OCRResult)."""
        lines: List[Dict[str, Any]] = []
        if not result:
            return lines
        first = result[0]
        if first is None:
            return lines
        # 3.x: list of dict-like OCRResult with rec_texts/rec_scores/rec_polys
        if hasattr(first, "get") and first.get("rec_texts") is not None:
            for res in result:
                texts = res.get("rec_texts") or []
                scores = res.get("rec_scores") or []
                polys = res.get("rec_polys")
                if polys is None:
                    polys = res.get("dt_polys")
                for i, text in enumerate(texts):
                    bbox = None
                    if polys is not None and i < len(polys):
                        try:
                            bbox = [[float(p[0]), float(p[1])]
                                    for p in polys[i]]
                        except (TypeError, ValueError, IndexError):
                            bbox = None
                    conf = float(scores[i]) if i < len(scores) else 0.0
                    lines.append({"text": text, "confidence": conf,
                                  "bbox": bbox})
            return lines
        # 2.x: [[ [bbox, (text, conf)], ... ]]
        for line_info in first:
            try:
                lines.append({"text": line_info[1][0],
                              "confidence": float(line_info[1][1]),
                              "bbox": line_info[0]})
            except (TypeError, IndexError):
                continue
        return lines

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

        cascade = self._build_cascade(engine, lang)
        # Shared model slot — allows N parallel inferences across users/pages
        with model_slot():
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

    @staticmethod
    def _is_thai(languages: str) -> bool:
        lang = (languages or "").lower()
        parts = [p.strip() for p in lang.split("+")]
        return "tha" in parts or "th" in parts or "tha" in lang

    def _build_cascade(self, requested: str,
                       languages: Optional[str] = None) -> List[str]:
        """One model per language selection (v3.1).

        auto:  Thai/Both → ["thai_trocr"], English/other → ["paddleocr"].
        An explicitly named engine runs alone — no silent fallback.
        """
        explicit = {
            "typhoon": "typhoon", "typhoon_ocr": "typhoon",
            "typhoon-ocr": "typhoon",
            "tesseract": "tesseract", "pytesseract": "tesseract",
            "easyocr": "easyocr", "easy": "easyocr",
            "thai_trocr": "thai_trocr", "trocr": "thai_trocr",
            "paddleocr": "paddleocr", "paddle": "paddleocr",
        }
        if requested in explicit:
            return [explicit[requested]]
        lang = languages or self.languages
        if self._is_thai(lang):
            return ["thai_trocr"]
        return ["paddleocr"]

    # ── Engine runners ──

    def _run_engine(self, image: np.ndarray, engine: str,
                    languages: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Dispatch to the named engine with error containment."""
        try:
            lang = languages or self.languages
            if engine == "typhoon":
                return self._run_typhoon(image, lang)
            if engine == "easyocr":
                return self._run_easyocr(image, lang)
            if engine == "thai_trocr":
                return self._run_thai_trocr(image, lang)
            if engine == "paddleocr":
                return self._run_paddle_engine(image, lang)
            if engine == "tesseract":
                return self._run_tesseract(image, lang)
            logger.debug("Unknown engine: %s", engine)
        except (OSError, ValueError, RuntimeError, IndexError, TypeError):
            logger.exception("Engine '%s' error", engine)
        return None

    # ── Typhoon OCR (SCB10X Thai document VLM) ──

    def _run_typhoon(self, image: np.ndarray,
                     _languages: Optional[str] = None
                     ) -> Optional[Dict[str, Any]]:
        """Run Typhoon OCR via the typhoon-ocr package.

        Uses api.opentyphoon.ai with TYPHOON_OCR_API_KEY, or a self-hosted
        endpoint (vLLM / Ollama) via TYPHOON_BASE_URL. Returns markdown-ish
        text; no per-line bboxes (the pipeline falls back to block bbox).
        """
        if not _typhoon_configured() or _typhoon_ocr_document is None:
            return None
        import tempfile
        import uuid
        tmp = os.path.join(tempfile.gettempdir(),
                           f"typhoon_{uuid.uuid4().hex}.png")
        try:
            cv2.imwrite(tmp, image)
            kwargs: Dict[str, Any] = {}
            base_url = os.getenv("TYPHOON_BASE_URL", "").strip()
            if base_url:
                kwargs["base_url"] = base_url
                kwargs["api_key"] = os.getenv("TYPHOON_OCR_API_KEY", "no-key")
            task_type = os.getenv("TYPHOON_TASK", "").strip()
            if task_type:
                kwargs["task_type"] = task_type
            text = _typhoon_ocr_document(
                tmp, model=os.getenv("TYPHOON_MODEL", "typhoon-ocr"),
                **kwargs)
            text = (text or "").strip()
            if not text:
                return None
            lines = [{"text": ln, "confidence": 0.95}
                     for ln in text.split("\n") if ln.strip()]
            return {"text": text, "confidence": 0.95,
                    "engine_used": "typhoon", "lines": lines}
        except Exception as exc:
            logger.warning("Typhoon OCR error: %s — %s",
                           type(exc).__name__, exc)
            return None
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass

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

    _TROCR_BATCH = 8
    _TROCR_BATCH_SPEED = 16

    @staticmethod
    def _trocr_recognize(pil_images: List) -> List[str]:
        """Recognise a list of single-line PIL images with Thai-TrOCR."""
        if not pil_images or _trocr_processor is None:
            return []
        texts: List[str] = []
        if _trocr_session is not None and hasattr(_trocr_session, "run"):
            # ONNX runtime — one image at a time
            for img in pil_images:
                pixel_values = _trocr_processor(
                    images=img, return_tensors="pt").pixel_values
                input_name = _trocr_session.get_inputs()[0].name
                outputs = _trocr_session.run(
                    None, {input_name: pixel_values.numpy()})
                if outputs and len(outputs) > 0:
                    texts.append(_trocr_processor.batch_decode(
                        outputs[0], skip_special_tokens=True)[0])
                else:
                    texts.append("")
            return texts
        if _trocr_session is not None and hasattr(_trocr_session, "generate"):
            # transformers — batched generation (CUDA when model is on GPU)
            import torch
            speed = _speed_mode()
            batch_size = (OCREngine._TROCR_BATCH_SPEED if speed
                          else OCREngine._TROCR_BATCH)
            # Prefer max_new_tokens only (avoid dual max_length warning)
            # Slightly higher than 96 — long Thai duty lines were truncating.
            max_tokens = 128 if speed else 160
            model_dev = next(_trocr_session.parameters()).device
            # Clear conflicting max_length on generation_config
            try:
                gc = getattr(_trocr_session, "generation_config", None)
                if gc is not None and getattr(gc, "max_length", None):
                    gc.max_length = None
            except Exception:
                pass
            for i in range(0, len(pil_images), batch_size):
                batch = pil_images[i:i + batch_size]
                pixel_values = _trocr_processor(
                    images=batch, return_tensors="pt").pixel_values
                pixel_values = pixel_values.to(model_dev)
                with torch.inference_mode():
                    generated = _trocr_session.generate(
                        pixel_values,
                        max_new_tokens=max_tokens,
                        do_sample=False,
                        num_beams=1,
                    )
                texts.extend(_trocr_processor.batch_decode(
                    generated, skip_special_tokens=True))
            return texts
        return ["" for _ in pil_images]

    def ocr_table_cell(self, image: np.ndarray,
                       languages: Optional[str] = None) -> Dict[str, Any]:
        """OCR one table cell with whole-cell TrOCR (no aggressive chunking).

        Docling cell crops are already tight; over-chunking them produces
        garbled Thai (ตาแทนง / ด้านข้างราชการ). Prefer a single crop.
        """
        if not _validate_image(image):
            return {"text": "", "confidence": 0.0, "engine_used": "none",
                    "lines": []}
        self._ensure_engines()
        lang = languages or self.languages
        with model_slot():
            if self._is_thai(lang):
                res = self._run_thai_trocr(
                    image, lang, whole_cell=True)
                if res and (res.get("text") or "").strip():
                    return res
            return self.ocr_image(image, languages=lang)

    def _run_thai_trocr(self, image: np.ndarray,  # NOSONAR
                        _languages: Optional[str] = None,
                        *, whole_cell: bool = False) -> Optional[Dict[str, Any]]:
        """Chunked Thai-TrOCR recognition (v3.1).

        TrOCR is a single-line, fixed-input model. The crop is segmented
        in two passes BEFORE recognition:

            1. visual lines (horizontal bands), then
            2. small word/phrase chunks inside each line (gap-based,
               with force-splitting of over-wide chunks)

        Each small chunk is recognised separately (upscaled when tiny),
        then chunk texts are reassembled left→right per line. Every chunk
        keeps its bbox so the exporter reproduces spacing and alignment.

        ``whole_cell=True`` skips force-splitting (table cells / short labels).
        """
        _check_thai_trocr()
        if not THAI_TROCR_AVAILABLE or _trocr_processor is None:
            logger.error("Thai-TrOCR unavailable — Thai pages cannot be OCRed "
                         "(check models/thai-trocr or network access)")
            return None
        try:
            from PIL import Image as PILImage

            if len(image.shape) == 2:
                gray = image
                rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            else:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Table cells: one (or few) visual lines, no force-split chunks
            if whole_cell:
                h, w = image.shape[:2]
                min_h = max(_TROCR_MIN_HEIGHT, 48)
                if 0 < h < min_h:
                    scale = min_h / h
                    rgb = cv2.resize(
                        rgb,
                        (max(1, int(w * scale)), min_h),
                        interpolation=cv2.INTER_CUBIC)
                    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
                line_boxes = _segment_text_lines(rgb)
                if not line_boxes:
                    h, w = gray.shape[:2]
                    line_boxes = [[0, 0, w, h]]
                # Keep each line intact — only gap-split on large whitespace
                chunks_per_line = [
                    _split_line_into_chunks(gray, lb, max_ratio=64.0)
                    for lb in line_boxes]
            else:
                line_boxes = _segment_text_lines(image)
                if not line_boxes:
                    h, w = image.shape[:2]
                    line_boxes = [[0, 0, w, h]]

                # Pass 2: phrase chunks (required for Thai — whole lines squash glyphs)
                # SPEED_MODE: wider chunks → fewer generates
                # CUDA cascade: still chunk, but allow slightly denser crops
                on_cuda = False
                try:
                    if (_trocr_session is not None
                            and hasattr(_trocr_session, "parameters")):
                        on_cuda = next(
                            _trocr_session.parameters()).device.type == "cuda"
                except Exception:
                    on_cuda = False
                if _speed_mode():
                    chunk_ratio = 8.0 if on_cuda else 12.0
                else:
                    chunk_ratio = None
                chunks_per_line = [
                    _split_line_into_chunks(gray, lb, max_ratio=chunk_ratio)
                    for lb in line_boxes]

            crops: List[Any] = []
            flat_boxes: List[List[int]] = []
            for chunks in chunks_per_line:
                for x0, y0, x1, y1 in chunks:
                    crop = rgb[y0:y1, x0:x1]
                    ch = y1 - y0
                    if 0 < ch < _TROCR_MIN_HEIGHT:   # tiny text → upscale
                        scale = _TROCR_MIN_HEIGHT / ch
                        crop = cv2.resize(
                            crop,
                            (max(1, int((x1 - x0) * scale)),
                             _TROCR_MIN_HEIGHT),
                            interpolation=cv2.INTER_CUBIC)
                    crops.append(PILImage.fromarray(crop))
                    flat_boxes.append([x0, y0, x1, y1])
            if not crops:
                return None

            texts = self._trocr_recognize(crops)

            # Per-chunk segments (pipeline clusters them back into lines)
            segments: List[Dict[str, Any]] = []
            full_lines: List[str] = []
            idx = 0
            for line_i, chunks in enumerate(chunks_per_line):
                parts: List[str] = []
                for _ in chunks:
                    text = (texts[idx] if idx < len(texts) else "").strip()
                    box = flat_boxes[idx]
                    idx += 1
                    if not text:
                        continue
                    parts.append(text)
                    segments.append({
                        "text": text,
                        "confidence": 0.85,
                        "bbox": [float(v) for v in box],
                        # Bold is decided at export (headers only), not OCR
                        "bold": False,
                    })
                if parts:
                    full_lines.append(_join_trocr_chunks(parts))
            if not segments:
                return None
            return {
                "text": "\n".join(full_lines),
                "confidence": 0.85,
                "engine_used": "thai_trocr",
                "lines": segments,
            }
        except Exception as exc:
            logger.warning("Thai-TrOCR error: %s — %s", type(exc).__name__, exc)
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
        lang = languages or self.languages
        # CRITICAL GUARD: never let an English-only Paddle model "read"
        # Thai input — it returns ASCII garbage. (Thai normally routes to
        # Thai-TrOCR; this guard protects explicit OCR_ENGINE=paddleocr
        # use with Thai input on paddleocr<3.x.)
        if "tha" in lang and self._paddle_lang(lang) != "th":
            logger.debug("PaddleOCR skipped: no Thai model on paddleocr<3.x")
            return None
        paddle = self._get_paddle(lang)
        if paddle is None:
            return None
        # PaddleX Normalize/unwarp requires HxWx3 — preprocess often yields gray
        bgr = _ensure_bgr(image)
        if bgr is None or bgr.size == 0 or bgr.ndim != 3 or bgr.shape[2] != 3:
            return None
        if min(bgr.shape[:2]) < 8:
            return None
        try:
            result = paddle.ocr(bgr, cls=True)
        except TypeError:           # 3.x removed the cls kwarg
            try:
                result = paddle.ocr(
                    bgr,
                    use_doc_unwarping=False,
                    use_doc_orientation_classify=False,
                )
            except TypeError:
                result = paddle.ocr(bgr)
        lines = self._parse_paddle_result(result)
        if not lines:
            return None
        all_text = [ln["text"] for ln in lines]
        confidences = [ln["confidence"] for ln in lines]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return {
            "text": "\n".join(all_text),
            "confidence": avg_conf,
            "engine_used": "paddleocr",
            "lines": lines,
        }

    # ── OCR with position data (for table cell OCR) ──

    def ocr_image_with_positions(self, image: np.ndarray,
                                 languages: Optional[str] = None
                                 ) -> List[Dict[str, Any]]:
        """Return OCR results with bbox positions for each text segment.

        Each item: {"text": str, "confidence": float,
                    "bbox": [x0,y0,x1,y1] or [[x,y]..] quad}
        """
        self._ensure_engines()
        lang = languages or self.languages

        # Strict policy: Thai → Thai-TrOCR (chunk bboxes from segmentation)
        if self._is_thai(lang):
            res = self._run_thai_trocr(image, lang)
            if res and res.get("lines"):
                return res["lines"]
            return []

        # Other languages → PaddleOCR (native per-line bboxes)
        res = self._run_paddle_engine(image, lang)
        if res and res.get("lines"):
            return res["lines"]
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
        first = lang.split("+")[0].strip().lower()
        # paddleocr>=3.x ships the PP-OCRv5 Thai model (lang="th").
        # On 2.x Thai is unsupported — return "en" and let
        # _run_paddle_engine skip Thai input entirely.
        thai_target = "th" if _paddle_supports_thai() else "en"
        mapping = {"tha": thai_target, "th": thai_target,
                   "chi_sim": "ch", "jpn": "japan",
                   "kor": "korean", "ara": "arabic"}
        return mapping.get(first, "en")

    # ── Status ──

    def get_available_engines(self) -> Dict[str, bool]:
        self._ensure_engines()
        return {
            "typhoon": _typhoon_configured(),
            "tesseract": TESSERACT_AVAILABLE,
            "easyocr": EASYOCR_AVAILABLE,
            "thai_trocr": THAI_TROCR_AVAILABLE,
            "paddleocr": PADDLE_AVAILABLE,
            "paddleocr_thai": PADDLE_AVAILABLE and _paddle_supports_thai(),
        }

    def is_available(self) -> bool:
        return any(self.get_available_engines().values())
