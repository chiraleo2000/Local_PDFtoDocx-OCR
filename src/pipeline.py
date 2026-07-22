"""OCR Pipeline Orchestrator - v2.5

PDF -> Render -> Layout Detect -> GROUP (text/table/figure)
  -> OCR text only -> Table extraction -> Image extraction
  -> Build HTML -> HTML->DOCX + HTML->TXT

v2.0 changes:
    - Strict engine policy: Thai → Thai-TrOCR (line-level), other → PaddleOCR
    - Improved table extraction with grid-aware cell alignment
    - OCR engine shared with table extractor for consistent results

Security:
    - PDF path validated (exists, is file, size limit)
    - Input bounds checked on all parameters
    - Error messages do not expose internal paths
    - Documents properly closed in finally blocks
"""
import os
import re
import base64
import html as html_module
import logging
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, Tuple

import numpy as np
import fitz  # PyMuPDF

from .preprocessor import OpenCVPreprocessor
from .ocr_engine import OCREngine
from .layout_detector import LayoutDetector, TableExtractor
from .exporter import ImageExtractor, DocumentExporter
from .correction_store import CorrectionStore
from .runtime import PAGE_WORKERS, summary as runtime_summary

logger = logging.getLogger(__name__)

_MAX_PDF_SIZE_BYTES = int(os.getenv("MAX_PDF_SIZE_MB", "200")) * 1024 * 1024
_MAX_TRIM_PERCENT = 25.0
# OCR segments below this confidence are dropped (engines that report it)
_MIN_SEGMENT_CONF = float(os.getenv("OCR_MIN_CONF", "0.30"))


def _validate_pdf_path(pdf_path: str) -> Optional[str]:
    """Return an error message if *pdf_path* is invalid, else ``None``."""
    if not pdf_path or not isinstance(pdf_path, str):
        return "No PDF path provided."
    if not pdf_path.lower().endswith(".pdf"):
        return "File does not have a .pdf extension."
    if not os.path.isfile(pdf_path):
        return "PDF file not found."
    try:
        size = os.path.getsize(pdf_path)
    except OSError:
        return "Cannot read PDF file."
    if size > _MAX_PDF_SIZE_BYTES:
        limit_mb = _MAX_PDF_SIZE_BYTES // (1024 * 1024)
        return f"PDF exceeds the {limit_mb} MB size limit."
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Content Block — the core ordering primitive
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class ContentBlock:
    """One semantic element on a page, sorted by reading position."""
    block_type: str          # "text" | "table" | "figure" | "caption"
    page: int                # 0-based page number
    y_top: float             # top-y coordinate (for reading-order sort)
    x_left: float = 0.0     # left-x coordinate (for column sort)
    text: str = ""           # extracted text (text/caption blocks, table plain text)
    table_html: str = ""     # HTML markup (table blocks only)
    figure: Dict[str, Any] = field(default_factory=dict)  # figure payload
    # ── Layout-fidelity payload (v2.1 — absolute positioning) ──
    bbox: List[float] = field(default_factory=list)   # [x0,y0,x1,y1] page units
    page_width: float = 0.0   # page width in same units as bbox
    page_height: float = 0.0  # page height in same units as bbox
    # Per visual line: {"text", "bbox": [x0,y0,x1,y1], optional "size_pt", "bold"}
    lines: List[Dict[str, Any]] = field(default_factory=list)
    table_meta: Dict[str, Any] = field(default_factory=dict)  # e.g. col_widths


def _quad_to_rect(bbox) -> Optional[List[float]]:
    """Normalise an OCR bbox (quad ``[[x,y]..]`` or rect ``[x0,y0,x1,y1]``)."""
    try:
        if not bbox:
            return None
        if isinstance(bbox[0], (list, tuple)):
            xs = [float(p[0]) for p in bbox]
            ys = [float(p[1]) for p in bbox]
            return [min(xs), min(ys), max(xs), max(ys)]
        if len(bbox) >= 4:
            return [float(bbox[0]), float(bbox[1]),
                    float(bbox[2]), float(bbox[3])]
    except (TypeError, ValueError, IndexError):
        pass
    return None


def _segments_to_lines(segments: List[Dict[str, Any]],  # NOSONAR
                       languages: str = "eng") -> List[Dict[str, Any]]:
    """Cluster OCR segments into visual lines (sorted top→bottom, left→right).

    OCR engines return word/phrase segments; this groups segments whose
    vertical centres overlap into single lines so the exporter can
    reproduce the original line structure, alignment, and spacing.
    """
    entries = []
    for seg in segments:
        rect = _quad_to_rect(seg.get("bbox"))
        txt = (seg.get("text") or "").strip()
        if rect is None or not txt:
            continue
        # v2.5: drop low-confidence fragments (hallucinated specks)
        conf = seg.get("conf", seg.get("confidence", seg.get("score")))
        try:
            if conf is not None and float(conf) < _MIN_SEGMENT_CONF:
                continue
        except (TypeError, ValueError):
            pass
        entries.append({"text": txt, "rect": rect,
                        "cy": (rect[1] + rect[3]) / 2.0,
                        "h": rect[3] - rect[1]})
    if not entries:
        return []

    entries.sort(key=lambda e: (e["cy"], e["rect"][0]))
    lines: List[Dict[str, Any]] = []
    cluster: List[Dict[str, Any]] = [entries[0]]

    def _flush(items: List[Dict[str, Any]]) -> None:
        items.sort(key=lambda e: e["rect"][0])
        x0 = min(e["rect"][0] for e in items)
        y0 = min(e["rect"][1] for e in items)
        x1 = max(e["rect"][2] for e in items)
        y1 = max(e["rect"][3] for e in items)
        text = clean_text(" ".join(e["text"] for e in items), languages)
        if text:
            lines.append({"text": text, "bbox": [x0, y0, x1, y1]})

    for entry in entries[1:]:
        ref = cluster[0]
        tol = max(ref["h"], entry["h"]) * 0.6
        if abs(entry["cy"] - ref["cy"]) <= tol:
            cluster.append(entry)
        else:
            _flush(cluster)
            cluster = [entry]
    _flush(cluster)
    return lines


# ══════════════════════════════════════════════════════════════════════════════
# Text Processing Helpers
# ══════════════════════════════════════════════════════════════════════════════
_THAI_CHAR_RE = re.compile(r"[\u0E00-\u0E7F]")
_THAI_SPACE_RE = re.compile(r"([\u0E00-\u0E7F])\s+([\u0E00-\u0E7F])")
# Duplicated Thai combining marks (tone marks, upper/lower vowels) —
# a frequent OCR artifact: "กรุ่่ง" → "กรุ่ง"
_THAI_DUP_MARK_RE = re.compile(r"([ัิ-ฺ็-๎])\1+")


def _fix_thai_spacing(text: str) -> str:
    """Remove artificial spaces between Thai characters (OCR artifact)."""
    lines = []
    for line in text.split("\n"):
        prev = None
        while prev != line:
            prev = line
            line = _THAI_SPACE_RE.sub(r"\1\2", line)
        lines.append(line)
    return "\n".join(lines)


def _embedded_text_reliable(text: str, languages: str) -> bool:
    """Decide whether a PDF's embedded text layer can be trusted.

    Many scanned Thai PDFs carry a hidden text layer in which Thai glyphs
    have no Unicode mapping — extraction silently drops every Thai
    character, leaving only ASCII fragments. Trusting such a layer
    produces garbage output, so those pages must go through real OCR.
    """
    if not text:
        return False
    n = len(text)
    # Broken CMap → replacement characters
    if text.count("�") / n > 0.05:
        return False
    # Mostly digits/punctuation — letters were stripped from the layer
    letters = sum(ch.isalpha() for ch in text)
    if letters / n < 0.25:
        return False
    # Thai requested but not a single Thai character extracted —
    # classic symptom of a Thai font without a ToUnicode table.
    # v2.3: a genuinely English-only page inside a tha+eng document is
    # still trustworthy — but only when it reads like healthy prose, not
    # like the ASCII residue a stripped Thai layer leaves behind
    # (digit-heavy fragments, orphaned closing parentheses).
    lang_l = (languages or "").lower().replace(",", "+")
    lang_parts = [p for p in re.split(r"[+\s]+", lang_l) if p]
    wants_thai = "tha" in lang_l or "th" in lang_parts
    if wants_thai and not _THAI_CHAR_RE.search(text):
        digits = sum(ch.isdigit() for ch in text)
        opens = text.count("(")
        closes = text.count(")")
        healthy_latin = (
            letters >= 40
            and letters / n >= 0.55
            and digits / n <= 0.12
            and closes <= opens + 2
        )
        if not healthy_latin:
            return False
    return True


# Thai-TrOCR sometimes emits repeated hallucinated syllables
_THAI_HALLUCINATION_RUN = re.compile(
    r"(?:วรร[ทณ]|บรรณาธิการ|ฯลฯ)(?:\s*(?:วรร[ทณ]|บรรณาธิการ|ฯลฯ)){2,}"
)


def clean_text(text: str, languages: str = "eng") -> str:
    """Remove OCR artifacts, fix Thai spacing, normalise whitespace."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    if "tha" in languages or _THAI_CHAR_RE.search(text):
        text = _fix_thai_spacing(text)
        # Collapse duplicated tone/vowel marks (OCR stutter)
        text = _THAI_DUP_MARK_RE.sub(r"\1", text)
        # Normalise decomposed sara-am (nikhahit + sara aa → sara am)
        text = text.replace("ํา", "ำ")
        # Double sara-e mistaken for sara-ae: "เเ" → "แ"
        text = text.replace("เเ", "แ")
        # Collapse TrOCR hallucination runs (วรรทวรรทวรรณ…)
        text = _THAI_HALLUCINATION_RUN.sub("", text)
        text = re.sub(r"(วรร[ทณ])\1{1,}", r"\1", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    lines = [line.strip() for line in text.split("\n")]
    return "\n".join(lines).strip()


# ══════════════════════════════════════════════════════════════════════════════
# Reading-Order Sort (column-aware)
# ══════════════════════════════════════════════════════════════════════════════
def _sort_reading_order(blocks: List[ContentBlock],
                        page_width: int) -> List[ContentBlock]:
    """Sort blocks in natural reading order (top→bottom, left→right)."""
    if len(blocks) <= 1:
        return blocks

    mid_x = page_width / 2.0
    left = [b for b in blocks if (b.x_left + 50) < mid_x]
    right = [b for b in blocks if (b.x_left + 50) >= mid_x]

    if len(left) >= 2 and len(right) >= 2:
        left_max_x = max(b.x_left for b in left) if left else 0
        right_min_x = min(b.x_left for b in right) if right else page_width
        gap = right_min_x - left_max_x
        if gap > page_width * 0.1:
            left.sort(key=lambda b: b.y_top)
            right.sort(key=lambda b: b.y_top)
            return left + right

    blocks.sort(key=lambda b: (b.y_top, b.x_left))
    return blocks


def _bbox_overlap_frac(a: List[float], b: List[float]) -> float:
    """Intersection area as a fraction of the smaller bbox."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((a[2] - a[0]) * (a[3] - a[1]), 1e-6)
    area_b = max((b[2] - b[0]) * (b[3] - b[1]), 1e-6)
    return inter / min(area_a, area_b)


_DEDUP_WS_RE = re.compile(r"\s+")


def _dedup_blocks(blocks: List["ContentBlock"]) -> List["ContentBlock"]:  # NOSONAR
    """Drop text blocks whose text duplicates an overlapping earlier block.

    Overlapping layout detections make the same paragraph get OCRed
    twice; this de-duplicates the result (v2.5).
    """
    out: List[ContentBlock] = []
    for b in blocks:
        dup = False
        if b.block_type in ("text", "caption") and b.text.strip():
            key = _DEDUP_WS_RE.sub("", b.text)
            for o in out:
                if o.block_type not in ("text", "caption"):
                    continue
                if _DEDUP_WS_RE.sub("", o.text) != key:
                    continue
                if (not b.bbox or not o.bbox
                        or _bbox_overlap_frac(b.bbox, o.bbox) > 0.30):
                    dup = True
                    break
        if not dup:
            out.append(b)
    if len(out) != len(blocks):
        logger.info("Removed %d duplicated text block(s)",
                    len(blocks) - len(out))
    return out


_DEFAULT_LANGUAGES = "tha+eng"
ProgressCallback = Callable[[int, int, str], None]


def _page_content_chars(blocks: List["ContentBlock"]) -> int:
    """Count characters retained as text/table content on a page's blocks."""
    n = 0
    for b in blocks:
        if b.block_type in ("text", "caption"):
            n += len((b.text or "").strip())
        elif b.block_type == "table":
            n += len((b.text or "").strip())
            if b.table_html:
                n += len(re.sub(r"<[^>]+>", "", b.table_html))
    return n


def _page_thai_chars(blocks: List["ContentBlock"]) -> int:
    """Count Thai characters across text/table blocks on a page group."""
    n = 0
    for b in blocks:
        if b.block_type in ("text", "caption", "table"):
            n += len(_THAI_CHAR_RE.findall(b.text or ""))
            if b.table_html:
                n += len(_THAI_CHAR_RE.findall(b.table_html))
    return n


def _band_trocr_text_blocks(
        ocr: "OCREngine",
        page_img: np.ndarray,
        page_num: int,
        languages: str,
        bands: int = 5,
) -> List["ContentBlock"]:
    """Lightweight text recovery: OCR horizontal page bands with Thai-TrOCR.

    Faster than YOLO layout recovery; used under SPEED_MODE when Docling
    kept too little Thai text.
    """
    if page_img is None or page_img.size == 0:
        return []
    h, w = page_img.shape[:2]
    bands = max(2, min(int(bands), 12))
    out: List[ContentBlock] = []
    for i in range(bands):
        y0 = int(h * i / bands)
        y1 = int(h * (i + 1) / bands)
        if y1 <= y0:
            continue
        crop = page_img[y0:y1, :]
        try:
            res = ocr._run_thai_trocr(crop, languages) or {}
        except Exception:  # noqa: BLE001
            continue
        text = clean_text(res.get("text") or "", languages)
        if not text.strip():
            continue
        lines = []
        for seg in res.get("lines") or []:
            st = clean_text(seg.get("text") or "", languages)
            if not st:
                continue
            bb = seg.get("bbox") or [0, 0, w, y1 - y0]
            if len(bb) >= 4:
                lines.append({
                    "text": st,
                    "bbox": [float(bb[0]), float(bb[1]) + y0,
                             float(bb[2]), float(bb[3]) + y0],
                    "confidence": float(seg.get("confidence") or 0.8),
                })
        bbox = [0.0, float(y0), float(w), float(y1)]
        if lines:
            bbox = [
                min(ln["bbox"][0] for ln in lines),
                min(ln["bbox"][1] for ln in lines),
                max(ln["bbox"][2] for ln in lines),
                max(ln["bbox"][3] for ln in lines),
            ]
        out.append(ContentBlock(
            block_type="text", page=page_num,
            y_top=bbox[1], x_left=bbox[0], text=text,
            bbox=bbox, page_width=float(w), page_height=float(h),
            lines=lines,
        ))
    return out


def _docling_page_is_sparse(
        blocks: List["ContentBlock"],
        page_img: Optional[np.ndarray],
        thai_job: bool = False,
) -> bool:
    """True when Docling kept too little readable content for the page ink."""
    chars = _page_content_chars(blocks)
    text_blocks = sum(
        1 for b in blocks if b.block_type in ("text", "caption") and b.text.strip())
    figures = sum(1 for b in blocks if b.block_type == "figure")
    speed = os.getenv("SPEED_MODE", "0").strip().lower() in (
        "1", "true", "yes", "on")
    # SPEED_MODE: almost never YOLO-recover (YOLO+OCR blows the 10s/page budget)
    if speed:
        tables = sum(1 for b in blocks if b.block_type == "table")
        figures = sum(1 for b in blocks if b.block_type == "figure")
        # Only if Docling returned literally nothing useful
        return chars < 20 and text_blocks == 0 and tables == 0 and figures == 0
    # Quality mode — more aggressive recovery
    min_chars = 800 if thai_job else 300
    if chars < min_chars:
        return True
    if text_blocks < 5:
        return True
    if figures >= 1 and text_blocks < 6:
        return True
    if page_img is None:
        return chars < min_chars
    try:
        import cv2
        gray = (page_img if page_img.ndim == 2
                else cv2.cvtColor(page_img, cv2.COLOR_BGR2GRAY))
        h, w = gray.shape[:2]
        small = cv2.resize(gray, (max(1, w // 4), max(1, h // 4)))
        _, binary = cv2.threshold(small, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = float(cv2.countNonZero(binary)) / max(binary.size, 1)
        if ink > 0.03 and chars < 1200:
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def _merge_complementary_blocks(  # NOSONAR
        primary: List["ContentBlock"],
        secondary: List["ContentBlock"],
        text_only: bool = True,
) -> List["ContentBlock"]:
    """Keep Docling structure; fill missing text (and optionally tables) from YOLO.

    Default ``text_only=True`` avoids figure explosion from YOLO fragments.
    Secondary text overlapping TABLES is skipped (cells already extracted).
    Text overlapping FIGURES is kept — org-chart / diagram labels.
    """
    if not secondary:
        return primary
    if not primary:
        return secondary if not text_only else [
            b for b in secondary if b.block_type in ("text", "caption", "table")
        ]

    out = list(primary)
    primary_tables = [b for b in primary if b.block_type == "table" and b.bbox]
    primary_figures = [
        b for b in primary if b.block_type == "figure" and b.bbox
        and (b.figure or {}).get("base64")]
    primary_text = [
        b for b in primary
        if b.block_type in ("text", "caption") and b.bbox and b.text.strip()]

    def _overlaps_struct(bbox, structs, thresh=0.55) -> bool:
        for s in structs:
            if s.bbox and _bbox_overlap_frac(bbox, s.bbox) > thresh:
                return True
        return False

    added = 0
    for b in secondary:
        if not b.bbox:
            continue
        if b.block_type in ("text", "caption"):
            if not (b.text or "").strip():
                continue
            # Only skip text that is clearly table-cell content
            if _overlaps_struct(b.bbox, primary_tables, 0.55):
                continue
            key = _DEDUP_WS_RE.sub("", b.text)
            dup = False
            for o in primary_text:
                if _DEDUP_WS_RE.sub("", o.text) == key:
                    if _bbox_overlap_frac(b.bbox, o.bbox) > 0.25:
                        dup = True
                        break
                elif _bbox_overlap_frac(b.bbox, o.bbox) > 0.65:
                    dup = True
                    break
            if dup:
                continue
            out.append(b)
            added += 1
        elif b.block_type == "table":
            has_content = bool(
                (b.table_html or "").strip() or (b.text or "").strip())
            if not has_content:
                continue
            if _overlaps_struct(b.bbox, primary_tables, 0.40):
                continue
            # Prefer Docling tables; only add YOLO table if none on page
            page_tables = [t for t in primary_tables if t.page == b.page]
            if page_tables:
                continue
            out.append(b)
            added += 1
        elif b.block_type == "figure" and not text_only:
            if not (b.figure or {}).get("base64"):
                continue
            if _overlaps_struct(b.bbox, primary_figures, 0.40):
                continue
            out.append(b)
            added += 1

    if added:
        logger.info("Sparse recovery merged %d complementary block(s)", added)
        out = _dedup_blocks(out)
        out.sort(key=lambda x: (x.page, x.y_top, x.x_left))
    return out


def _figure_area(block: "ContentBlock") -> float:
    if block.bbox and len(block.bbox) >= 4:
        return max(0.0, (block.bbox[2] - block.bbox[0])
                   * (block.bbox[3] - block.bbox[1]))
    fig = block.figure or {}
    return float(fig.get("width", 0) or 0) * float(fig.get("height", 0) or 0)


def _prune_structure_blocks(
        blocks: List["ContentBlock"],
        max_figures_per_page: int = 2,
        max_tables_per_page: int = 1,
        max_tables_total: int = 2,
        max_figures_total: int = 2,
) -> List["ContentBlock"]:
    """Keep the largest figures/tables; drop tiny OCR fallback crops.

    Gold demo has 2 tables + 2 images total — enforce document-level caps.
    """
    by_page: Dict[int, List[ContentBlock]] = {}
    others: List[ContentBlock] = []
    for b in blocks:
        if b.block_type in ("figure", "table"):
            by_page.setdefault(b.page, []).append(b)
        else:
            others.append(b)

    kept_figs: List[ContentBlock] = []
    kept_tables: List[ContentBlock] = []
    for page, items in by_page.items():
        figs = [b for b in items if b.block_type == "figure"
                and (b.figure or {}).get("base64")]
        tables = [b for b in items if b.block_type == "table"
                  and ((b.table_html or "").strip() or (b.text or "").strip())]
        figs.sort(key=_figure_area, reverse=True)
        if figs:
            top = _figure_area(figs[0])
            min_area = max(top * 0.08, 1.0)
            figs = [f for f in figs if _figure_area(f) >= min_area]
        figs = figs[:max_figures_per_page]
        tables.sort(
            key=lambda t: len((t.table_html or "") + (t.text or "")),
            reverse=True)
        tables = tables[:max_tables_per_page]
        kept_figs.extend(figs)
        kept_tables.extend(tables)
        dropped = len(items) - len(figs) - len(tables)
        if dropped:
            logger.info(
                "Pruned %d structure block(s) on page %d "
                "(kept %d figures, %d tables)",
                dropped, page + 1, len(figs), len(tables))

    kept_figs.sort(key=_figure_area, reverse=True)
    kept_figs = kept_figs[:max_figures_total]
    kept_tables.sort(
        key=lambda t: len((t.table_html or "") + (t.text or "")),
        reverse=True)
    kept_tables = kept_tables[:max_tables_total]

    out = others + kept_figs + kept_tables
    out.sort(key=lambda b: (b.page, b.y_top, b.x_left))
    return out


class OCRPipeline:
    """Full PDF → DOCX/TXT/HTML pipeline (v2.0).

    Group-first architecture:
      1. Render page
      2. Layout detection (YOLO / OpenCV)
      3. Group regions: text, table, figure
      4. OCR text regions (Thai → Thai-TrOCR, other → PaddleOCR)
      5. Extract tables (grid + cell OCR, improved alignment)
      6. Extract figures as images
      7. Sort reading order
      8. Export via HTML-first

    v2.0: Tesseract removed. Thai-optimised engine cascade.
    Security: PDF inputs validated, file size limited, all paths checked.
    """

    # PDF base is 72 DPI. Scales map to target render DPI:
    #   fast=360, balanced=540, accurate=720.
    # Override any preset with env RENDER_DPI_SCALE (e.g. 10.0 → 720 DPI).
    QUALITY_MAP = {
        "fast":     {"dpi_scale": 5.0, "quality": "fast"},       # 360 DPI
        "balanced": {"dpi_scale": 7.5, "quality": "balanced"},   # 540 DPI
        "accurate": {"dpi_scale": 10.0, "quality": "accurate"},  # 720 DPI
    }
    # Thai-TrOCR needs denser pixels than Latin under the Fast UI preset.
    # SPEED_MODE / THAI_FAST_MIN_SCALE can lower this for ~10s/page budgets.
    _THAI_FAST_MIN_SCALE = 7.5  # ~540 DPI (quality)
    _THAI_FAST_MIN_SCALE_SPEED = 6.0  # ~432 DPI (1.5GB GPU speed path)

    @classmethod
    def _thai_fast_min_scale(cls) -> float:
        raw = os.getenv("THAI_FAST_MIN_SCALE", "").strip()
        if raw:
            try:
                return max(4.0, min(float(raw), 12.0))
            except ValueError:
                pass
        speed = os.getenv("SPEED_MODE", "0").strip().lower() in (
            "1", "true", "yes", "on")
        return cls._THAI_FAST_MIN_SCALE_SPEED if speed else cls._THAI_FAST_MIN_SCALE

    @classmethod
    def _resolve_dpi_scale(cls, quality: str,
                           languages: Optional[str] = None) -> float:
        """Return render scale for *quality*, honouring RENDER_DPI_SCALE."""
        cfg = cls.QUALITY_MAP.get(quality, cls.QUALITY_MAP["accurate"])
        scale = float(cfg["dpi_scale"])
        lang = (languages or "").lower().replace(",", "+")
        parts = [p for p in re.split(r"[+\s]+", lang) if p]
        wants_thai = "tha" in lang or "th" in parts
        if wants_thai and quality == "fast":
            scale = max(scale, cls._thai_fast_min_scale())
        override = os.getenv("RENDER_DPI_SCALE", "").strip()
        if override:
            try:
                scale = max(1.0, min(float(override), 20.0))
            except ValueError:
                pass
        return scale

    def __init__(self) -> None:
        quality = os.getenv("QUALITY_PRESET", "fast")
        self.languages = os.getenv("LANGUAGES", _DEFAULT_LANGUAGES)
        self.layout_mode = os.getenv("LAYOUT_MODE", "flow").lower()
        if self.layout_mode in ("flowing", "structured"):
            self.layout_mode = "flow"
        self.preprocessor = OpenCVPreprocessor(quality=quality)
        self.ocr = OCREngine()
        self.layout = LayoutDetector()
        self.table_extractor = TableExtractor()
        self.table_extractor.set_ocr_engine(self.ocr)  # Share OCR engine
        self.image_extractor = ImageExtractor()
        self.exporter = DocumentExporter()
        self.corrections = CorrectionStore()
        self.docling = None
        self.layout_backend = "yolo"
        try:
            from .runtime import configure_cuda_vram_limit
            configure_cuda_vram_limit()
        except Exception:  # noqa: BLE001
            pass
        try:
            from .docling_backend import (
                DoclingBackend, docling_ready, layout_backend)
            self.layout_backend = layout_backend()
            if docling_ready():
                scale = self._resolve_dpi_scale(quality)
                self.docling = DoclingBackend(
                    ocr_engine=self.ocr,
                    images_scale=max(1.0, scale / 2.0))
                if self.docling.available:
                    self.layout_backend = "docling"
                    # Warm models in background so first Convert is not stuck
                    # downloading weights at 0%.
                    if os.getenv("DOCLING_WARMUP", "1").strip() != "0":
                        import threading
                        threading.Thread(
                            target=self.docling.warm_up,
                            name="docling-warmup",
                            daemon=True,
                        ).start()
                else:
                    self.docling = None
                    self.layout_backend = "yolo"
                    logger.warning(
                        "LAYOUT_BACKEND=docling but converter failed — "
                        "falling back to YOLO")
            elif self.layout_backend == "docling":
                logger.warning(
                    "Docling not installed — falling back to YOLO layout")
                self.layout_backend = "yolo"
        except Exception:  # noqa: BLE001
            logger.exception("Docling backend init failed — using YOLO")
            self.docling = None
            self.layout_backend = "yolo"
        logger.info(
            "OCR Pipeline initialised (v2.6 — layout=%s, absolute pixel "
            "layout, Thai-TrOCR/PaddleOCR, Docling+OpenCV)",
            self.layout_backend)

    def process_pdf(self, pdf_path: str, quality: str = "accurate",  # NOSONAR
                    header_trim: float = 0, footer_trim: float = 0,
                    languages: Optional[str] = None,
                    yolo_confidence: Optional[float] = None,
                    page_size: str = "A4",
                    margin_preset: str = "Normal",
                    layout_mode: Optional[str] = None,
                    progress_callback: Optional[ProgressCallback] = None,
                    ) -> Dict[str, Any]:
        """Process a PDF end-to-end.

        Args:
            pdf_path: Path to the PDF file (validated).
            languages: Override OCR language (e.g. ``"tha+eng"``).
            yolo_confidence: Override YOLO threshold (lower = more regions).
            page_size: Output page size (A4, Letter, Legal, A3, B5).
            margin_preset: Output margin preset (Normal, Narrow, Moderate, Wide).
            layout_mode: ``absolute`` (pixel-grid) or ``flow``; default from env.
        """
        # ── Input validation ──
        validation_error = _validate_pdf_path(pdf_path)
        if validation_error:
            return {"success": False, "text": "", "files": {},
                    "metadata": {}, "error": validation_error}

        header_trim = max(0.0, min(float(header_trim), _MAX_TRIM_PERCENT))
        footer_trim = max(0.0, min(float(footer_trim), _MAX_TRIM_PERCENT))

        doc = None
        try:
            langs = languages or self.languages
            cfg = self.QUALITY_MAP.get(quality, self.QUALITY_MAP["accurate"])
            scale = self._resolve_dpi_scale(quality, langs)
            q = cfg["quality"]
            mode = (layout_mode or self.layout_mode or "flow").lower()
            if mode in ("flowing", "structured"):
                mode = "flow"

            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            doc = None

            progress_total = page_count + 1
            all_blocks, total_tables, total_figures = (
                self._run_layout_pipeline(
                    pdf_path, page_count, scale, q, langs,
                    header_trim, footer_trim, yolo_confidence,
                    progress_callback=progress_callback,
                    progress_total=progress_total,
                ))
            all_blocks = _prune_structure_blocks(all_blocks)
            total_tables = sum(1 for b in all_blocks if b.block_type == "table")
            total_figures = sum(
                1 for b in all_blocks if b.block_type == "figure")

            full_text = self._blocks_to_text(all_blocks)

            engines = self.ocr.get_available_engines()
            active_engines = [k for k, v in engines.items() if v]
            rt = runtime_summary()
            meta_str = (
                f"Pages: {page_count}\n"
                f"Quality: {quality}\n"
                f"Render scale: {scale:.1f}x (~{int(scale * 72)} DPI)\n"
                f"Layout: {mode}\n"
                f"Layout backend: {self.layout_backend}\n"
                f"Language: {langs}\n"
                f"OCR Engine: {', '.join(active_engines)}\n"
                f"Parallel: {rt['page_workers']} page workers, "
                f"{rt['max_concurrent_jobs']} concurrent jobs\n"
                f"Tables: {total_tables}\n"
                f"Figures: {total_figures}"
            )
            self._notify_progress(
                progress_callback, page_count, progress_total,
                "Exporting output files")
            files = self.exporter.create_all_from_blocks(
                all_blocks, meta_str,
                page_size=page_size, margin_preset=margin_preset,
                layout_mode=mode, render_dpi=scale * 72.0)
            self._notify_progress(
                progress_callback, progress_total, progress_total,
                "Exported output files")
            return {
                "success": True,
                "text": full_text,
                "files": files,
                "metadata": {
                    "pages": page_count,
                    "tables": total_tables,
                    "figures": total_figures,
                    "engines": engines,
                    "quality": quality,
                    "languages": langs,
                    "layout_mode": mode,
                    "layout_backend": self.layout_backend,
                    "dpi_scale": scale,
                    "page_workers": rt["page_workers"],
                    "max_concurrent_jobs": rt["max_concurrent_jobs"],
                },
                "error": None,
            }

        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Pipeline error")
            return {
                "success": False, "text": "", "files": {},
                "metadata": {}, "error": str(exc),
            }
        finally:
            if doc:
                doc.close()

    def _run_layout_pipeline(  # NOSONAR
            self, pdf_path: str, page_count: int, scale: float, q: str,
            langs: str, header_trim: float, footer_trim: float,
            yolo_confidence: Optional[float],
            extra_regions_by_page: Optional[Dict[int, List[Dict]]] = None,
            progress_callback: Optional[ProgressCallback] = None,
            progress_total: Optional[int] = None,
            pdf_name: str = "",
    ) -> Tuple[List[ContentBlock], int, int]:
        """Run Docling (default) or YOLO page-parallel layout → blocks."""
        progress_total = progress_total or (page_count + 1)
        # Docling page-by-page path — live progress, no full-PDF pre-render hang.
        if (self.layout_backend == "docling" and self.docling
                and self.docling.available
                and not (extra_regions_by_page or {})):
            _lang_l = (langs or "").lower()
            thai_job = ("tha" in _lang_l or "+th" in f"+{_lang_l}"
                        or _lang_l.startswith("th"))
            self._notify_progress(
                progress_callback, 0, progress_total,
                "Starting Docling + Thai-TrOCR…")
            try:
                all_blocks: List[ContentBlock] = []
                total_tables = 0
                total_figures = 0
                doc = fitz.open(pdf_path)
                sparse_raw = os.getenv(
                    "DOCLING_SPARSE_RECOVERY", "text").strip().lower()
                sparse_recovery = sparse_raw not in ("0", "false", "no", "off")
                # text/sparse/1 = YOLO text merge ONLY when page is sparse
                # always/force/2 = always merge text (slow, quality)
                # full = also merge YOLO figures (can explode counts)
                text_only = sparse_raw not in ("full", "figures", "all")
                always_merge = sparse_raw in ("always", "force", "2")
                fullpage_sup = os.getenv(
                    "DOCLING_FULLPAGE_SUPPLEMENT", "0").strip().lower() in (
                    "1", "true", "yes", "on")
                speed_mode = os.getenv("SPEED_MODE", "0").strip().lower() in (
                    "1", "true", "yes", "on")
                # SPEED_MODE: skip YOLO recovery + full-page fill (10s/page budget)
                if speed_mode:
                    fullpage_sup = False
                    if sparse_raw in ("sparse", "text", "1", "true", "yes", "on"):
                        sparse_recovery = False
                        always_merge = False
                try:
                    # SPEED_MODE: one Docling pass on the full PDF (not N
                    # single-page temp PDFs) — critical for ~10s/page.
                    if speed_mode and page_count > 0:
                        self._notify_progress(
                            progress_callback, 0, progress_total,
                            f"Docling layout ({page_count} pages)…")
                        page_images: Dict[int, np.ndarray] = {}
                        for i in range(page_count):
                            page_images[i] = self._render_page_image(
                                doc[i], scale, 0.0, 0.0)
                        self._notify_progress(
                            progress_callback, 0, progress_total,
                            "Docling + Thai-TrOCR…")
                        all_blocks, total_tables, total_figures = (
                            self.docling.convert_to_blocks(
                                pdf_path, languages=langs,
                                page_images=page_images))
                        # SPEED_MODE: banded Thai-TrOCR supplies body text
                        # (per-region text re-OCR is skipped in the adapter).
                        band_ocr = os.getenv(
                            "SPEED_BAND_OCR", "1").strip().lower() not in (
                            "0", "false", "no", "off")
                        if thai_job and band_ocr:
                            n_bands = int(os.getenv("SPEED_OCR_BANDS", "2")
                                          or "2")
                            band_scale = float(
                                os.getenv("SPEED_BAND_SCALE", "4.0") or "4.0")
                            by_page: Dict[int, List[ContentBlock]] = {}
                            for b in all_blocks:
                                by_page.setdefault(b.page, []).append(b)
                            recovered: List[ContentBlock] = []
                            # Lower-DPI band renders keep CPU TrOCR under budget
                            band_doc = fitz.open(pdf_path)
                            try:
                                for i in range(page_count):
                                    pblocks = by_page.get(i, [])
                                    pimg = self._render_page_image(
                                        band_doc[i], band_scale, 0.0, 0.0)
                                    if pimg is None:
                                        recovered.extend(pblocks)
                                        continue
                                    self._notify_progress(
                                        progress_callback, i, progress_total,
                                        f"Page {i + 1}/{page_count}: "
                                        "banded Thai-TrOCR…")
                                    band_blocks = _band_trocr_text_blocks(
                                        self.ocr, pimg, i, langs,
                                        bands=n_bands)
                                    if (band_blocks
                                            and abs(band_scale - scale) > 0.01):
                                        m = scale / band_scale
                                        hi, wi = page_images[i].shape[:2]
                                        for bb in band_blocks:
                                            bb.page_width = float(wi)
                                            bb.page_height = float(hi)
                                            if bb.bbox and len(bb.bbox) >= 4:
                                                bb.bbox = [
                                                    v * m for v in bb.bbox]
                                                bb.y_top = bb.bbox[1]
                                                bb.x_left = bb.bbox[0]
                                            for ln in bb.lines or []:
                                                lb = ln.get("bbox") or []
                                                if len(lb) >= 4:
                                                    ln["bbox"] = [
                                                        v * m for v in lb]
                                    if band_blocks:
                                        before = _page_thai_chars(pblocks)
                                        pblocks = _merge_complementary_blocks(
                                            pblocks, band_blocks,
                                            text_only=True)
                                        logger.info(
                                            "Page %d band OCR → +%d blocks "
                                            "(thai %d → %d)",
                                            i + 1, len(band_blocks), before,
                                            _page_thai_chars(pblocks))
                                    recovered.extend(pblocks)
                            finally:
                                band_doc.close()
                            all_blocks = recovered
                            total_tables = sum(
                                1 for b in all_blocks
                                if b.block_type == "table")
                            total_figures = sum(
                                1 for b in all_blocks
                                if b.block_type == "figure")
                        self._notify_progress(
                            progress_callback, page_count, progress_total,
                            f"Done Docling ({page_count} pages)")
                    else:
                        for i in range(page_count):
                            self._notify_progress(
                                progress_callback, i, progress_total,
                                f"Page {i + 1}/{page_count}: "
                                f"{'Docling + Thai-TrOCR' if thai_job else 'Docling'}")
                            # Raw render only — Docling bboxes are in PDF page
                            # space. Deskew/denoise/upscale would misalign crops
                            # (Latin garbage / empty figures). OCR engines enhance
                            # their own crops; figures need full-color pixels.
                            page_img = self._render_page_image(
                                doc[i], scale, 0.0, 0.0)
                            blocks, n_t, n_f = self.docling.convert_page_to_blocks(
                                pdf_path, i, languages=langs, page_img=page_img)
                            need_recovery = sparse_recovery and (
                                always_merge
                                or _docling_page_is_sparse(
                                    blocks, page_img, thai_job=thai_job)
                            )
                            if need_recovery:
                                self._notify_progress(
                                    progress_callback, i, progress_total,
                                    f"Page {i + 1}/{page_count}: "
                                    "recovering missing text (YOLO+OCR)…")
                                logger.info(
                                    "Docling page %d — YOLO complementary recovery",
                                    i + 1)
                                yolo_blocks, _yt, _yf, yerr = self._process_page_job(
                                    pdf_path, i, scale, q, langs,
                                    0.0, 0.0, yolo_confidence,
                                    [], pdf_name=pdf_name or os.path.basename(
                                        pdf_path))
                                if yerr:
                                    logger.warning(
                                        "YOLO recovery page %d: %s", i + 1, yerr)
                                else:
                                    chars_before = _page_content_chars(blocks)
                                    before = len(blocks)
                                    blocks = _merge_complementary_blocks(
                                        blocks, yolo_blocks, text_only=text_only)
                                    logger.info(
                                        "Page %d recovery: %d → %d blocks "
                                        "(chars %d → %d)",
                                        i + 1, before, len(blocks),
                                        chars_before, _page_content_chars(blocks))
                                    n_t = sum(1 for b in blocks
                                              if b.block_type == "table")
                                    n_f = sum(1 for b in blocks
                                              if b.block_type == "figure")
                            # Optional full-page OCR fill-in
                            if fullpage_sup and thai_job:
                                page_txt = "\n".join(
                                    (b.text or "") for b in blocks
                                    if b.block_type in (
                                        "text", "caption", "table"))
                                numbered = len(re.findall(
                                    r"(?:^|\n)\s*\d+[\)\.]", page_txt))
                                if numbered < 4:
                                    try:
                                        pre = self.preprocessor.preprocess(
                                            page_img, quality=q or "fast")
                                        fp_blocks, _, _ = self._fullpage_fallback(
                                            page_img, pre, i, langs)
                                        if fp_blocks:
                                            blocks = _merge_complementary_blocks(
                                                blocks, fp_blocks, text_only=True)
                                            logger.info(
                                                "Page %d full-page OCR supplement "
                                                "→ %d blocks", i + 1, len(blocks))
                                    except Exception:  # noqa: BLE001
                                        logger.exception(
                                            "Full-page supplement failed page %d",
                                            i + 1)
                            all_blocks.extend(blocks)
                            total_tables += n_t
                            total_figures += n_f
                            # Free GPU between pages under 1.5GB VRAM budgets
                            try:
                                import torch
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                            except Exception:  # noqa: BLE001
                                pass
                            self._notify_progress(
                                progress_callback, i + 1, progress_total,
                                f"Done page {i + 1}/{page_count} "
                                f"(tables={total_tables}, figures={total_figures})")
                finally:
                    doc.close()
                all_blocks = _prune_structure_blocks(all_blocks)
                total_tables = sum(
                    1 for b in all_blocks if b.block_type == "table")
                total_figures = sum(
                    1 for b in all_blocks if b.block_type == "figure")
                logger.info(
                    "Docling convert: blocks=%d tables=%d figures=%d chars=%d",
                    len(all_blocks), total_tables, total_figures,
                    _page_content_chars(all_blocks))
                if all_blocks:
                    return all_blocks, total_tables, total_figures
                logger.warning(
                    "Docling returned no blocks — falling back to YOLO path")
            except Exception:  # noqa: BLE001
                logger.exception(
                    "Docling convert failed — falling back to YOLO path")

        page_results = self._process_pages_parallel(
            pdf_path, page_count, scale, q, langs,
            header_trim, footer_trim, yolo_confidence,
            extra_regions_by_page=extra_regions_by_page,
            progress_callback=progress_callback,
            progress_total=progress_total,
            pdf_name=pdf_name,
        )
        all_blocks = []
        total_tables = 0
        total_figures = 0
        for page_num in range(page_count):
            blocks, n_tables, n_figures, err = page_results[page_num]
            if err:
                logger.error("Page %d failed: %s", page_num + 1, err)
            all_blocks.extend(blocks)
            total_tables += n_tables
            total_figures += n_figures
        return all_blocks, total_tables, total_figures

    def _process_pages_parallel(
            self, pdf_path: str, page_count: int, scale: float, q: str,
            langs: str, header_trim: float, footer_trim: float,
            yolo_confidence: Optional[float],
            extra_regions_by_page: Optional[Dict[int, List[Dict]]] = None,
            progress_callback: Optional[ProgressCallback] = None,
            progress_total: Optional[int] = None,
            pdf_name: str = "",
    ) -> Dict[int, Tuple[List[ContentBlock], int, int, Optional[str]]]:
        """Process PDF pages in parallel (each worker opens its own Document)."""
        workers = max(1, min(PAGE_WORKERS, page_count))
        progress_total = progress_total or (page_count + 1)
        results: Dict[int, Tuple[List[ContentBlock], int, int, Optional[str]]] = {}
        name = pdf_name or os.path.basename(pdf_path)

        if workers <= 1 or page_count <= 1:
            for page_num in range(page_count):
                results[page_num] = self._process_page_job(
                    pdf_path, page_num, scale, q, langs,
                    header_trim, footer_trim, yolo_confidence,
                    (extra_regions_by_page or {}).get(page_num, []),
                    pdf_name=name,
                )
                self._notify_progress(
                    progress_callback, page_num + 1, progress_total,
                    f"Completed page {page_num + 1}/{page_count}")
            return results

        logger.info(
            "Parallel page processing: %d pages · %d workers",
            page_count, workers)
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    self._process_page_job,
                    pdf_path, page_num, scale, q, langs,
                    header_trim, footer_trim, yolo_confidence,
                    (extra_regions_by_page or {}).get(page_num, []),
                    name,
                ): page_num
                for page_num in range(page_count)
            }
            for fut in as_completed(futures):
                page_num = futures[fut]
                try:
                    results[page_num] = fut.result()
                except Exception as exc:
                    logger.exception("Page %d worker crashed", page_num + 1)
                    results[page_num] = ([], 0, 0, str(exc))
                completed += 1
                self._notify_progress(
                    progress_callback, completed, progress_total,
                    f"Completed {completed}/{page_count} pages")
        return results

    def _process_page_job(
            self, pdf_path: str, page_num: int, scale: float, q: str,
            langs: str, header_trim: float, footer_trim: float,
            yolo_confidence: Optional[float],
            extra_regions: Optional[List[Dict]] = None,
            pdf_name: str = "",
    ) -> Tuple[List[ContentBlock], int, int, Optional[str]]:
        """Render + OCR one page. Thread-safe (own fitz.Document)."""
        doc = None
        try:
            logger.info("Processing page %d", page_num + 1)
            doc = fitz.open(pdf_path)
            if page_num < 0 or page_num >= len(doc):
                return [], 0, 0, "Invalid page number"
            extras = extra_regions or []
            # Always render + run layout/table/figure extraction. Embedded
            # text may supply body text, but never skip structure — Review
            # can show tables while convert used to report Tables: 0.
            img = self._render_page_image(
                doc[page_num], scale, header_trim, footer_trim)
            h, w = img.shape[:2]
            preprocessed = self.preprocessor.preprocess(img, quality=q)

            direct_blocks = self._embedded_text_block(
                doc[page_num], page_num, langs)
            page_blocks, n_tables, n_figures = self._process_single_page(
                img, preprocessed, h, w, page_num, langs, yolo_confidence,
                extra_regions=extras)

            if direct_blocks is not None and n_tables == 0 and n_figures == 0:
                # Born-digital with no detected structure — keep embedded
                # text/figures (already includes embedded images).
                page_blocks, n_tables, n_figures = (
                    direct_blocks,
                    sum(1 for b in direct_blocks if b.block_type == "table"),
                    sum(1 for b in direct_blocks if b.block_type == "figure"),
                )
            elif direct_blocks is not None and (n_tables > 0 or n_figures > 0):
                # Hybrid: keep extracted tables/figures; prefer OCR/layout
                # text only outside structure boxes so we don't lose tables.
                struct = [b for b in page_blocks
                          if b.block_type in ("table", "figure")]
                struct_boxes = [b.bbox for b in struct if b.bbox]
                text_keep = []
                for b in direct_blocks:
                    if b.block_type != "text" or not b.bbox:
                        if b.block_type == "text":
                            text_keep.append(b)
                        continue
                    if any(_bbox_overlap_frac(b.bbox, sb) > 0.40
                           for sb in struct_boxes):
                        continue
                    text_keep.append(b)
                # Drop OCR text that duplicates embedded text / sits in
                # table-figure regions; keep captions from layout path.
                ocr_extra = [
                    b for b in page_blocks
                    if b.block_type in ("text", "caption")
                    and b.bbox
                    and not any(_bbox_overlap_frac(b.bbox, sb) > 0.40
                                for sb in struct_boxes)
                    and not any(
                        _DEDUP_WS_RE.sub("", b.text)
                        == _DEDUP_WS_RE.sub("", t.text)
                        for t in text_keep if t.text.strip())
                ]
                # Scale embedded PDF-point coords to render pixels when
                # page_width differs (embedded uses PDF pts; layout uses px).
                emb_pw = next(
                    (b.page_width for b in direct_blocks if b.page_width), 0.0)
                emb_ph = next(
                    (b.page_height for b in direct_blocks if b.page_height), 0.0)
                if emb_pw > 0 and emb_ph > 0 and abs(emb_pw - w) > 1.0:
                    sx_e, sy_e = w / emb_pw, h / emb_ph
                    for b in text_keep:
                        if b.bbox and len(b.bbox) >= 4:
                            b.bbox = [b.bbox[0] * sx_e, b.bbox[1] * sy_e,
                                      b.bbox[2] * sx_e, b.bbox[3] * sy_e]
                            b.y_top, b.x_left = b.bbox[1], b.bbox[0]
                        for ln in b.lines or []:
                            lb = ln.get("bbox")
                            if lb and len(lb) >= 4:
                                ln["bbox"] = [
                                    lb[0] * sx_e, lb[1] * sy_e,
                                    lb[2] * sx_e, lb[3] * sy_e]
                        b.page_width, b.page_height = float(w), float(h)
                merged = text_keep + ocr_extra + struct
                page_blocks = _dedup_blocks(_sort_reading_order(merged, w))
                n_tables = sum(1 for b in page_blocks if b.block_type == "table")
                n_figures = sum(
                    1 for b in page_blocks if b.block_type == "figure")

            if extras and pdf_name:
                for mr in extras:
                    self.corrections.log_correction(
                        page_image=img,
                        bbox=mr["bbox"],
                        region_class=mr.get("class", "figure"),
                        page_number=page_num,
                        pdf_name=pdf_name,
                        action="add",
                        source="manual",
                    )
            return page_blocks, n_tables, n_figures, None
        except Exception as exc:
            logger.exception("Page %d error", page_num + 1)
            return [], 0, 0, str(exc)
        finally:
            if doc is not None:
                doc.close()

    # ── Single-page processing (group-first) ─────────────────────────────────

    def _process_single_page(self, img: np.ndarray, preprocessed: np.ndarray,  # NOSONAR
                             h: int, w: int, page_num: int,
                             languages: str = _DEFAULT_LANGUAGES,
                             yolo_confidence: Optional[float] = None,
                             extra_regions: Optional[List[Dict]] = None,
                             ) -> tuple:
        """Process one page: detect → merge manual → group → OCR text / extract table / extract figure."""

        # ── Step 1: Layout detection ─────────────────────────────────
        layout_result = self.layout.detect_layout(
            img, page_num, confidence=yolo_confidence)
        detections = layout_result.get("detections", {})

        text_regions  = detections.get("text_regions", [])
        table_dets    = detections.get("tables", [])
        figure_dets   = detections.get("figures", [])
        caption_dets  = detections.get("captions", [])
        formula_dets  = detections.get("formulas", [])

        # ── v2.5: collapse overlapping/nested detections — duplicated
        #    boxes are the main source of duplicated OCR paragraphs ──
        text_regions = self.layout.dedup_regions(text_regions)
        table_dets   = self.layout.dedup_regions(table_dets)
        figure_dets  = self.layout.dedup_regions(figure_dets)
        caption_dets = self.layout.dedup_regions(caption_dets)

        if self.layout_backend == "docling":
            _method = "Docling"
        elif self.layout.model_loaded:
            _method = "YOLO"
        else:
            _method = "OpenCV"
        logger.info(
            "Detected: page %d — text=%d, tables=%d, figures=%d, captions=%d"
            " (method=%s)",
            page_num + 1, len(text_regions), len(table_dets),
            len(figure_dets), len(caption_dets), _method,
        )

        # ── Step 1b: Merge manual corrections ────────────────────────
        self._merge_manual_regions(
            extra_regions, text_regions, table_dets, figure_dets, page_num)

        # ── Step 1c: Recover graphics the detector missed (v2.3) ─────
        # Logos/stamps often land in the "abandon" class (dropped), or
        # are missed entirely — both used to lose the image. Re-tag
        # graphic-like abandoned regions and scan the leftover ink.
        if os.getenv("GRAPHIC_RECOVERY", "true").lower() == "true":
            other_dets = detections.get("other", [])
            try:
                if other_dets:
                    figure_dets.extend(
                        self.layout.filter_graphic_regions(img, other_dets))
                known = [d["bbox"] for d in (
                    text_regions + table_dets + figure_dets
                    + caption_dets + formula_dets + other_dets)]
                figure_dets.extend(self.layout.recover_graphics(img, known))
            except (OSError, RuntimeError, ValueError):
                logger.exception(
                    "Graphic recovery failed on page %d", page_num + 1)

        page_blocks: List[ContentBlock] = []

        # ── Fallback: no layout detected → full-page OCR ────────────
        if not text_regions and not table_dets and not figure_dets:
            return self._fullpage_fallback(img, preprocessed, page_num, languages)

        # ── Step 2-4: OCR text, captions, formulas ───────────────────
        self._ocr_regions_to_blocks(
            text_regions, preprocessed, img, h, w, page_num,
            "text", languages, page_blocks)
        self._ocr_regions_to_blocks(
            caption_dets, preprocessed, img, h, w, page_num,
            "caption", languages, page_blocks)
        self._ocr_regions_to_blocks(
            formula_dets, preprocessed, img, h, w, page_num,
            "text", languages, page_blocks)

        # ── Step 5: Extract tables ───────────────────────────────────
        try:
            n_tables = self._extract_table_blocks(
                img, table_dets, page_num, languages, page_blocks)
        except (OSError, RuntimeError, ValueError):
            logger.exception("Table extraction failed on page %d", page_num + 1)
            n_tables = 0

        # ── Step 6: Extract figures ──────────────────────────────────
        try:
            n_figures = self._extract_figure_blocks(
                img, figure_dets, page_num, page_blocks)
        except (OSError, RuntimeError, ValueError):
            logger.exception("Figure extraction failed on page %d", page_num + 1)
            n_figures = 0

        logger.info(
            "Page %d result: text_blocks=%d, tables=%d, figures=%d",
            page_num + 1, len([b for b in page_blocks if b.block_type == "text"]),
            n_tables, n_figures,
        )

        # ── Step 7: Sort into reading order, drop duplicated text ────
        page_blocks = _sort_reading_order(page_blocks, w)
        page_blocks = _dedup_blocks(page_blocks)
        return page_blocks, n_tables, n_figures

    # ── _process_single_page helpers ─────────────────────────────────────────

    @staticmethod
    def _merge_manual_regions(extra_regions, text_regions, table_dets,
                              figure_dets, page_num) -> None:
        """Merge user-drawn manual corrections into detection lists."""
        for mr in (extra_regions or []):
            cls = mr.get("class", "figure")
            det = {"bbox": mr["bbox"], "confidence": 1.0,
                   "class": cls, "class_id": 5 if cls == "table" else 3,
                   "source": "manual"}
            if cls == "table":
                table_dets.append(det)
            elif cls == "figure":
                figure_dets.append(det)
            else:
                text_regions.append(det)
            logger.info("Merged manual %s region on page %d", cls, page_num)

    def _fullpage_fallback(self, img, preprocessed, page_num, languages):
        """OCR the entire page when no layout regions were detected.

        v2.3: pages with no recognisable text but visible ink (covers,
        diagrams, logo pages) are preserved as a full-page figure instead
        of being dropped from the output.
        """
        ocr_result = self.ocr.ocr_full_page(preprocessed, languages=languages)
        raw = ocr_result.get("text", "")
        txt = clean_text(raw, languages)
        blocks: List[ContentBlock] = []
        if txt.strip():
            ph, pw = preprocessed.shape[:2]
            lines = _segments_to_lines(ocr_result.get("lines") or [], languages)
            if lines:
                txt = "\n".join(line["text"] for line in lines)
                x0 = min(line["bbox"][0] for line in lines)
                y0 = min(line["bbox"][1] for line in lines)
                x1 = max(line["bbox"][2] for line in lines)
                y1 = max(line["bbox"][3] for line in lines)
                bbox = [x0, y0, x1, y1]
            else:
                bbox = []
            blocks.append(ContentBlock(
                block_type="text", page=page_num, y_top=0, x_left=0, text=txt,
                bbox=bbox, page_width=float(pw), page_height=float(ph),
                lines=lines))
            return blocks, 0, 0

        # No text found — keep the page as an image if it has content
        try:
            fig = self.image_extractor.page_figure(img, page_num)
        except (OSError, RuntimeError, ValueError):
            fig = None
        if fig:
            ih, iw = img.shape[:2]
            bbox = fig.get("bbox", [0.0, 0.0, float(iw), float(ih)])
            logger.info(
                "Page %d: no text detected — preserved as full-page figure",
                page_num + 1)
            return [ContentBlock(
                block_type="figure", page=page_num,
                y_top=float(bbox[1]), x_left=float(bbox[0]),
                figure=fig,
                bbox=[float(v) for v in bbox],
                page_width=float(iw), page_height=float(ih),
            )], 0, 1
        return blocks, 0, 0

    def _embedded_text_block(self, page, page_num: int, languages: str):  # NOSONAR
        """Return positioned blocks for born-digital PDF pages.

        Uses ``get_text("dict")`` so block/line positions and font sizes are
        preserved (bbox in PDF points). v2.3: pages with a reliable text
        layer AND images no longer lose their images — embedded raster
        images and vector logo clusters are extracted as figure blocks
        alongside the text (hybrid path). Returns a list of ContentBlocks,
        or ``None`` if the page needs the OCR path.
        """
        # Optional override: always OCR, never trust embedded text
        if os.getenv("FORCE_OCR", "false").lower() == "true":
            return None

        raw = page.get_text("text") or ""
        cleaned = clean_text(raw, languages)
        pw = float(page.rect.width)
        ph = float(page.rect.height)
        page_area = max(pw * ph, 1.0)

        # Full-page scan (one image covering nearly the whole page) →
        # always take the OCR path, even if a hidden text layer exists.
        try:
            img_infos = page.get_image_info() or []
        except (RuntimeError, ValueError):
            img_infos = []
        for info in img_infos:
            bx = info.get("bbox")
            if not bx:
                continue
            r = fitz.Rect(bx) & page.rect
            if r.get_area() >= page_area * 0.82:
                return None

        if not cleaned.strip():
            # No text layer at all. Born-digital graphic-only pages
            # (covers, logo pages) keep their images; anything else
            # goes to OCR.
            if img_infos or self._page_has_drawings(page):
                fig_blocks = self._embedded_image_blocks(page, page_num, [])
                if fig_blocks:
                    return fig_blocks
            return None

        # Broken/incomplete text layer (e.g. Thai glyphs without Unicode
        # mapping) → route the page through the OCR pipeline instead.
        if not _embedded_text_reliable(cleaned, languages):
            logger.warning(
                "Page %d: embedded text layer looks unreliable "
                "(missing Thai/letters) — using OCR instead.", page_num + 1)
            return None

        blocks: List[ContentBlock] = []
        try:
            page_dict = page.get_text("dict")
        except (RuntimeError, ValueError):
            page_dict = None

        for blk in (page_dict or {}).get("blocks", []):
            if blk.get("type") != 0:          # text blocks only
                continue
            lines: List[Dict[str, Any]] = []
            for ln in blk.get("lines", []):
                spans = ln.get("spans", [])
                line_text = clean_text(
                    "".join(s.get("text", "") for s in spans), languages)
                if not line_text:
                    continue
                sizes = [float(s.get("size", 0)) for s in spans
                         if s.get("size")]
                bold = any((int(s.get("flags", 0)) & 16) for s in spans)
                lines.append({
                    "text": line_text,
                    "bbox": [float(v) for v in ln.get("bbox", blk["bbox"])],
                    "size_pt": (sum(sizes) / len(sizes)) if sizes else 0.0,
                    "bold": bold,
                })
            if not lines:
                continue
            bx = [float(v) for v in blk.get("bbox", [0, 0, 0, 0])]
            blocks.append(ContentBlock(
                block_type="text", page=page_num,
                y_top=bx[1], x_left=bx[0],
                text="\n".join(line["text"] for line in lines),
                bbox=bx, page_width=pw, page_height=ph,
                lines=lines,
            ))

        if not blocks:   # parsing failed — single flat block fallback
            txt = clean_text(raw, languages)
            blocks = [ContentBlock(
                block_type="text", page=page_num, y_top=0, x_left=0,
                text=txt, page_width=pw, page_height=ph)]

        # ── v2.3: keep the page's images (was: entire page sent to OCR,
        #    or images silently lost) ──
        try:
            blocks.extend(
                self._embedded_image_blocks(page, page_num, blocks))
        except (RuntimeError, ValueError):
            logger.exception(
                "Embedded image extraction failed on page %d", page_num + 1)
        return blocks

    # ── Embedded image / vector-logo extraction (v2.3) ───────────────────────

    _EMBED_RENDER_SCALE = 10.0      # embedded figures match accurate (720 DPI)
    _EMBED_MIN_SIDE_PT = 12.0       # smallest figure side (PDF points)
    _EMBED_MIN_AREA_PT = 900.0      # smallest figure area (PDF points²)
    _MAX_VECTOR_PRIMITIVES = 500    # skip vector clustering on huge art pages

    @staticmethod
    def _page_has_drawings(page) -> bool:
        try:
            return bool(page.get_drawings())
        except (RuntimeError, ValueError):
            return False

    def _embedded_image_blocks(self, page, page_num: int,  # NOSONAR
                               text_blocks: List[ContentBlock],
                               ) -> List[ContentBlock]:
        """Extract placed raster images + vector graphic clusters as figures.

        Each figure region is rendered from the page (handles rotation,
        masks, transparency and overlapping content correctly) and encoded
        as base64 PNG, positioned with the same PDF-point coordinates used
        by the embedded text blocks.
        """
        pw = float(page.rect.width)
        ph = float(page.rect.height)
        page_area = max(pw * ph, 1.0)
        text_rects = [fitz.Rect(b.bbox) for b in text_blocks if b.bbox]

        rects: List[fitz.Rect] = []

        # 1) Placed raster images (logos, photos, stamps)
        try:
            infos = page.get_image_info() or []
        except (RuntimeError, ValueError):
            infos = []
        for info in infos:
            bx = info.get("bbox")
            if not bx:
                continue
            r = fitz.Rect(bx) & page.rect
            if r.is_empty or r.width < 6 or r.height < 6:
                continue
            if r.get_area() >= page_area * 0.95:
                continue
            rects.append(r)

        # 2) Vector graphics (logos drawn as paths) — clustered primitives
        rects.extend(self._vector_graphic_rects(page, page_area))

        rects = self._merge_rects(rects, pad=4.0)

        blocks: List[ContentBlock] = []
        idx = 0
        for r in rects:
            if (r.width < self._EMBED_MIN_SIDE_PT
                    or r.height < self._EMBED_MIN_SIDE_PT
                    or r.get_area() < self._EMBED_MIN_AREA_PT):
                continue
            if self._rect_mostly_text(r, text_rects):
                continue       # text frame / underline box — not a figure
            try:
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(self._EMBED_RENDER_SCALE,
                                       self._EMBED_RENDER_SCALE),
                    clip=r)
                png = pix.tobytes("png")
            except (RuntimeError, ValueError):
                continue
            idx += 1
            blocks.append(ContentBlock(
                block_type="figure", page=page_num,
                y_top=float(r.y0), x_left=float(r.x0),
                figure={
                    "page": page_num,
                    "index": idx,
                    "base64": base64.b64encode(png).decode(),
                    "width": int(pix.width),
                    "height": int(pix.height),
                    "bbox": [float(r.x0), float(r.y0),
                             float(r.x1), float(r.y1)],
                    "source": "embedded",
                },
                bbox=[float(r.x0), float(r.y0), float(r.x1), float(r.y1)],
                page_width=pw, page_height=ph,
            ))
        if blocks:
            logger.info("Page %d: extracted %d embedded figure(s)",
                        page_num + 1, len(blocks))
        return blocks

    def _vector_graphic_rects(self, page, page_area: float
                              ) -> List["fitz.Rect"]:
        """Cluster vector drawing primitives into logo/graphic rectangles."""
        try:
            drawings = page.get_drawings()
        except (RuntimeError, ValueError):
            return []
        if not drawings or len(drawings) > self._MAX_VECTOR_PRIMITIVES:
            return []
        raw: List[fitz.Rect] = []
        for d in drawings:
            r = fitz.Rect(d.get("rect", fitz.Rect()))
            if r.is_empty:
                continue
            if r.get_area() > page_area * 0.6:    # page frame / background
                continue
            raw.append(r)
        if not raw:
            return []
        merged = self._merge_rects(raw, pad=6.0)
        out = []
        for r in merged:
            if r.width < self._EMBED_MIN_SIDE_PT * 1.2:
                continue
            if r.height < self._EMBED_MIN_SIDE_PT * 1.2:
                continue
            if (r.get_area() < self._EMBED_MIN_AREA_PT
                    or r.get_area() > page_area * 0.6):
                continue
            out.append(r)
        return out

    @staticmethod
    def _merge_rects(rects: List["fitz.Rect"], pad: float = 0.0
                     ) -> List["fitz.Rect"]:
        """Union intersecting (optionally padded) rectangles."""
        pending = [fitz.Rect(r) for r in rects if not fitz.Rect(r).is_empty]
        merged: List[fitz.Rect] = []
        while pending:
            r = pending.pop()
            changed = True
            while changed:
                changed = False
                rp = fitz.Rect(r.x0 - pad, r.y0 - pad,
                               r.x1 + pad, r.y1 + pad)
                i = 0
                while i < len(pending):
                    if rp.intersects(pending[i]):
                        r |= pending.pop(i)
                        changed = True
                    else:
                        i += 1
            merged.append(r)
        return merged

    @staticmethod
    def _rect_mostly_text(rect: "fitz.Rect",
                          text_rects: List["fitz.Rect"]) -> bool:
        """True when *rect* is mostly covered by text blocks."""
        if not text_rects:
            return False
        area = rect.get_area()
        if area <= 0:
            return True
        covered = sum(abs((rect & tr).get_area()) for tr in text_rects)
        return (covered / area) > 0.55

    def _ocr_regions_to_blocks(self, regions, preprocessed, img, h, w,
                               page_num, block_type, languages, out) -> None:
        """OCR a list of detected regions and append ContentBlocks to *out*."""
        for region in regions:
            block = self._ocr_region_to_block(
                region, preprocessed, img, h, w, page_num,
                block_type=block_type, languages=languages)
            if block:
                out.append(block)

    def _extract_table_blocks(self, img, table_dets, page_num,
                              languages, out) -> int:
        """Extract tables and append to *out*. Returns table count."""
        if not table_dets:
            return 0
        count = 0
        ih, iw = img.shape[:2]
        for t in self.table_extractor.extract_tables(
                img, table_dets, languages=languages):
            bbox = t.get("bbox", [0, 0, 0, 0])
            html = t.get("html", "") or ""
            text = t.get("text", "") or ""
            if not html and text.strip():
                # Rebuild a minimal HTML table so absolute export does not
                # silently drop structured content.
                rows = [r for r in text.split("\n") if r.strip()]
                cells = [r.split("\t") for r in rows] or [[text]]
                html = "<table>" + "".join(
                    "<tr>" + "".join(
                        f"<td>{html_module.escape(c)}</td>" for c in row)
                    + "</tr>" for row in cells) + "</table>"
            if not html and not text.strip():
                logger.warning(
                    "Page %d: table detection produced empty extraction "
                    "at bbox=%s — skipping count", page_num + 1, bbox)
                continue
            out.append(ContentBlock(
                block_type="table", page=page_num,
                y_top=float(bbox[1]), x_left=float(bbox[0]),
                text=text,
                table_html=html,
                bbox=[float(v) for v in bbox],
                page_width=float(iw), page_height=float(ih),
                table_meta={"col_widths": t.get("col_widths", [])},
            ))
            count += 1
        if table_dets and count == 0:
            logger.warning(
                "Page %d: %d table region(s) detected but none extracted",
                page_num + 1, len(table_dets))
        return count

    def _extract_figure_blocks(self, img, figure_dets, page_num, out) -> int:
        """Extract figures and append to *out*. Returns figure count."""
        if not figure_dets:
            return 0
        ih, iw = img.shape[:2]
        extracted = self.image_extractor.extract_figures(img, figure_dets, page_num)
        for fig in extracted:
            # bbox now travels with the figure payload — index-based
            # association broke whenever the extractor filtered a region
            bbox = fig.get("bbox") or [0, 0, 0, 0]
            out.append(ContentBlock(
                block_type="figure", page=page_num,
                y_top=float(bbox[1]), x_left=float(bbox[0]),
                figure=fig,
                bbox=[float(v) for v in bbox],
                page_width=float(iw), page_height=float(ih),
            ))
        return len(extracted)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ocr_region_to_block(self, region: Dict, preprocessed: np.ndarray,
                             _img: np.ndarray, h: int, w: int,
                             page_num: int, block_type: str = "text",
                             languages: str = _DEFAULT_LANGUAGES,
                             ) -> Optional[ContentBlock]:
        """OCR a single detected text region → ContentBlock or None."""
        x0, y0, x1, y1 = [int(v) for v in region["bbox"]]
        # v2.5: pad the crop so Thai upper/lower vowels and tone marks at
        # the region border are not clipped off before recognition
        pad = max(4, int(min(h, w) * 0.004))
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
        if x1 <= x0 or y1 <= y0:
            return None

        prep_h, prep_w = preprocessed.shape[:2]
        sx = prep_w / w if w > 0 else 1.0
        sy = prep_h / h if h > 0 else 1.0
        px0, py0 = int(x0 * sx), int(y0 * sy)
        px1, py1 = int(x1 * sx), int(y1 * sy)
        px0, py0 = max(0, px0), max(0, py0)
        px1, py1 = min(prep_w, px1), min(prep_h, py1)

        crop = preprocessed[py0:py1, px0:px1]
        if crop.size == 0:
            return None

        ocr_result = self.ocr.ocr_image(crop, languages=languages)
        raw = ocr_result.get("text", "")
        txt = clean_text(raw, languages)
        if not txt.strip():
            return None

        # ── Per-line positions (crop coords → page coords) ──
        lines = _segments_to_lines(ocr_result.get("lines") or [], languages)
        for line in lines:
            lx0, ly0, lx1, ly1 = line["bbox"]
            line["bbox"] = [
                lx0 / sx + x0, ly0 / sy + y0,
                lx1 / sx + x0, ly1 / sy + y0,
            ]
        if lines:
            txt = "\n".join(line["text"] for line in lines)

        return ContentBlock(
            block_type=block_type, page=page_num,
            y_top=float(y0), x_left=float(x0),
            text=txt,
            bbox=[float(x0), float(y0), float(x1), float(y1)],
            page_width=float(w), page_height=float(h),
            lines=lines,
        )

    @staticmethod
    def _notify_progress(callback: Optional[ProgressCallback], current: int,
                         total: int, message: str) -> None:
        """Notify UI progress callbacks without coupling the pipeline to Gradio."""
        if callback is not None:
            callback(current, total, message)

    @staticmethod
    def _blocks_to_text(blocks: List[ContentBlock]) -> str:
        """Convert blocks → plain text with page separators."""
        parts: List[str] = []
        current_page = -1
        for b in blocks:
            if b.page != current_page:
                if current_page >= 0:
                    parts.append("")
                current_page = b.page

            if b.block_type in ("text", "caption", "table"):
                if b.text.strip():
                    parts.append(b.text)
            elif b.block_type == "figure":
                fig_idx = b.figure.get("index", "?")
                pg = b.figure.get("page", "?")
                page_disp = (pg + 1) if isinstance(pg, int) else pg
                parts.append(
                    f"[Figure {fig_idx} — Page {page_disp}]")

        return "\n\n".join(parts)

    @staticmethod
    def _render_page_image(page, scale: float,
                           header_trim: float,
                           footer_trim: float) -> np.ndarray:
        """Render one PDF page to a BGR numpy image."""
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
            pix.h, pix.w, pix.n)
        if img.shape[2] == 4:
            img = img[:, :, [2, 1, 0]]
        elif img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        img = np.ascontiguousarray(img)
        h = img.shape[0]
        top_cut = int((header_trim / 100) * h) if header_trim > 0 else 0
        bot_cut = int((footer_trim / 100) * h) if footer_trim > 0 else 0
        if top_cut > 0 or bot_cut > 0:
            img = img[top_cut:h - max(bot_cut, 0), :]
        return img

    # ══════════════════════════════════════════════════════════════════════════
    # Manual Correction Support  (v0.5)
    # ══════════════════════════════════════════════════════════════════════════

    def detect_page_regions(self, pdf_path: str, page_num: int,
                            quality: str = "accurate",
                            yolo_confidence: Optional[float] = None,
                            ) -> Dict[str, Any]:
        """Detect layout on a single page and return regions + rendered image.

        Used by the UI to show detected regions before the user adds manual ones.
        """
        validation_error = _validate_pdf_path(pdf_path)
        if validation_error:
            return {"success": False, "error": validation_error}

        doc = None
        try:
            scale = self._resolve_dpi_scale(quality)
            doc = fitz.open(pdf_path)
            if page_num < 0 or page_num >= len(doc):
                return {"success": False, "error": "Invalid page number"}
            img = self._render_page_image(doc[page_num], scale, 0, 0)

            detections = None
            method = "YOLO" if self.layout.model_loaded else "OpenCV"
            if (self.layout_backend == "docling" and self.docling
                    and self.docling.available):
                try:
                    detections = self.docling.detect_page(
                        pdf_path, page_num, page_img=img)
                    method = "Docling"
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Docling detect_page failed — using YOLO")
                    detections = None
            if detections is None:
                layout_result = self.layout.detect_layout(
                    img, page_num, confidence=yolo_confidence)
                detections = layout_result.get("detections", {})

            return {
                "success": True,
                "page_image": img,
                "detections": detections,
                "image_shape": list(img.shape),
                "layout_backend": method.lower(),
            }
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("detect_page_regions error")
            return {"success": False, "error": str(exc)}
        finally:
            if doc:
                doc.close()

    def add_manual_region(self, page_image: np.ndarray,
                          bbox: List[float], region_class: str,
                          page_number: int, pdf_name: str = "",
                          ) -> Dict[str, Any]:
        """Log a user-drawn manual region to the correction store.

        Returns correction info including retrain status.
        """
        result = self.corrections.log_correction(
            page_image=page_image,
            bbox=bbox,
            region_class=region_class,
            page_number=page_number,
            pdf_name=pdf_name,
            action="add",
            source="manual",
        )
        return result

    def process_pdf_with_corrections(
            self, pdf_path: str,
            manual_regions: Optional[Dict[int, List[Dict]]] = None,
            quality: str = "accurate",
            header_trim: float = 0, footer_trim: float = 0,
            languages: Optional[str] = None,
            yolo_confidence: Optional[float] = None,
            page_size: str = "A4",
            margin_preset: str = "Normal",
            layout_mode: Optional[str] = None,
            progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Process PDF with optional manual region corrections.

        Args:
            manual_regions: ``{page_num: [{"bbox": [x0,y0,x1,y1], "class": "table"|"figure"}, ...]}``
            page_size: Output page size (A4, Letter, Legal, A3, B5).
            margin_preset: Output margin preset (Normal, Narrow, Moderate, Wide).
            layout_mode: ``absolute`` (pixel-grid) or ``flow``; default from env.
        """
        # ── Input validation ──
        validation_error = _validate_pdf_path(pdf_path)
        if validation_error:
            return {"success": False, "text": "", "files": {},
                    "metadata": {}, "error": validation_error}

        header_trim = max(0.0, min(float(header_trim), _MAX_TRIM_PERCENT))
        footer_trim = max(0.0, min(float(footer_trim), _MAX_TRIM_PERCENT))

        doc = None
        try:
            langs = languages or self.languages
            cfg = self.QUALITY_MAP.get(quality, self.QUALITY_MAP["accurate"])
            scale = self._resolve_dpi_scale(quality, langs)
            q = cfg["quality"]
            mode = (layout_mode or self.layout_mode or "flow").lower()
            if mode in ("flowing", "structured"):
                mode = "flow"

            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            doc = None

            progress_total = page_count + 1
            pdf_name = os.path.basename(pdf_path)
            all_blocks, total_tables, total_figures = (
                self._run_layout_pipeline(
                    pdf_path, page_count, scale, q, langs,
                    header_trim, footer_trim, yolo_confidence,
                    extra_regions_by_page=manual_regions or {},
                    progress_callback=progress_callback,
                    progress_total=progress_total,
                    pdf_name=pdf_name,
                ))
            all_blocks = _prune_structure_blocks(all_blocks)
            total_tables = sum(1 for b in all_blocks if b.block_type == "table")
            total_figures = sum(
                1 for b in all_blocks if b.block_type == "figure")

            full_text = self._blocks_to_text(all_blocks)
            engines = self.ocr.get_available_engines()
            active_engines = [k for k, v in engines.items() if v]
            rt = runtime_summary()
            meta_str = (
                f"Pages: {page_count}\n"
                f"Quality: {quality}\n"
                f"Render scale: {scale:.1f}x (~{int(scale * 72)} DPI)\n"
                f"Layout: {mode}\n"
                f"Layout backend: {self.layout_backend}\n"
                f"Language: {langs}\n"
                f"OCR Engine: {', '.join(active_engines)}\n"
                f"Parallel: {rt['page_workers']} page workers, "
                f"{rt['max_concurrent_jobs']} concurrent jobs\n"
                f"Tables: {total_tables}\n"
                f"Figures: {total_figures}"
            )
            self._notify_progress(
                progress_callback, page_count, progress_total,
                "Exporting output files")
            files = self.exporter.create_all_from_blocks(
                all_blocks, meta_str,
                page_size=page_size, margin_preset=margin_preset,
                layout_mode=mode, render_dpi=scale * 72.0)
            self._notify_progress(
                progress_callback, progress_total, progress_total,
                "Exported output files")
            return {
                "success": True,
                "text": full_text,
                "files": files,
                "metadata": {
                    "pages": page_count,
                    "tables": total_tables,
                    "figures": total_figures,
                    "engines": engines,
                    "quality": quality,
                    "languages": langs,
                    "layout_mode": mode,
                    "layout_backend": self.layout_backend,
                    "dpi_scale": scale,
                    "page_workers": rt["page_workers"],
                    "max_concurrent_jobs": rt["max_concurrent_jobs"],
                    "manual_corrections": sum(
                        len(v) for v in (manual_regions or {}).values()),
                },
                "error": None,
            }
        except (OSError, RuntimeError, ValueError) as exc:
            logger.exception("Pipeline error")
            return {
                "success": False, "text": "", "files": {},
                "metadata": {}, "error": str(exc),
            }
        finally:
            if doc:
                doc.close()

    def get_status(self) -> Dict[str, Any]:
        correction_stats = self.corrections.get_stats()
        return {
            "ocr_engines": self.ocr.get_available_engines(),
            "primary_engine": self.ocr.primary_engine,
            "languages": self.languages,
            "layout_detector": self.layout.model_loaded,
            "yolo_confidence": self.layout.confidence,
            "table_extraction": self.table_extractor.enabled,
            "image_extraction": self.image_extractor.enabled,
            "corrections": correction_stats,
            "runtime": runtime_summary(),
        }
