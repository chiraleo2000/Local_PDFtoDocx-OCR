"""OCR Pipeline Orchestrator - v2.3

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
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable

import numpy as np
import fitz  # PyMuPDF

from .preprocessor import OpenCVPreprocessor
from .ocr_engine import OCREngine
from .layout_detector import LayoutDetector, TableExtractor
from .exporter import ImageExtractor, DocumentExporter
from .correction_store import CorrectionStore

logger = logging.getLogger(__name__)

_MAX_PDF_SIZE_BYTES = int(os.getenv("MAX_PDF_SIZE_MB", "200")) * 1024 * 1024
_MAX_TRIM_PERCENT = 25.0


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


def _segments_to_lines(segments: List[Dict[str, Any]],
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
    if "tha" in languages and not _THAI_CHAR_RE.search(text):
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


def clean_text(text: str, languages: str = "eng") -> str:
    """Remove OCR artifacts, fix Thai spacing, normalise whitespace."""
    if not text:
        return ""
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    if "tha" in languages or _THAI_CHAR_RE.search(text):
        text = _fix_thai_spacing(text)
        # Collapse duplicated tone/vowel marks (OCR stutter)
        text = _THAI_DUP_MARK_RE.sub(r"\1", text)
        # Normalise decomposed sara-am (nikhahit + sara aa → sara am)
        text = text.replace("ํา", "ำ")
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


_DEFAULT_LANGUAGES = "tha+eng"
ProgressCallback = Callable[[int, int, str], None]


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

    QUALITY_MAP = {
        "fast":     {"dpi_scale": 1.5, "quality": "fast"},
        "balanced": {"dpi_scale": 2.0, "quality": "balanced"},
        "accurate": {"dpi_scale": 2.5, "quality": "accurate"},
    }

    def __init__(self) -> None:
        quality = os.getenv("QUALITY_PRESET", "balanced")
        self.languages = os.getenv("LANGUAGES", _DEFAULT_LANGUAGES)
        self.preprocessor = OpenCVPreprocessor(quality=quality)
        self.ocr = OCREngine()
        self.layout = LayoutDetector()
        self.table_extractor = TableExtractor()
        self.table_extractor.set_ocr_engine(self.ocr)  # Share OCR engine
        self.image_extractor = ImageExtractor()
        self.exporter = DocumentExporter()
        self.corrections = CorrectionStore()
        logger.info("OCR Pipeline initialised (v2.0 — Thai-optimised, HTML-first, trainable)")

    def process_pdf(self, pdf_path: str, quality: str = "balanced",
                    header_trim: float = 0, footer_trim: float = 0,
                    languages: Optional[str] = None,
                    yolo_confidence: Optional[float] = None,
                    page_size: str = "A4",
                    margin_preset: str = "Normal",
                    progress_callback: Optional[ProgressCallback] = None,
                    ) -> Dict[str, Any]:
        """Process a PDF end-to-end.

        Args:
            pdf_path: Path to the PDF file (validated).
            languages: Override OCR language (e.g. ``"tha+eng"``).
            yolo_confidence: Override YOLO threshold (lower = more regions).
            page_size: Output page size (A4, Letter, Legal, A3, B5).
            margin_preset: Output margin preset (Normal, Narrow, Moderate, Wide).
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
            cfg = self.QUALITY_MAP.get(quality, self.QUALITY_MAP["balanced"])
            scale = cfg["dpi_scale"]
            q = cfg["quality"]

            doc = fitz.open(pdf_path)
            page_count = len(doc)
            progress_total = page_count + 1
            all_blocks: List[ContentBlock] = []
            total_tables = 0
            total_figures = 0

            for page_num in range(page_count):
                logger.info("Processing page %d/%d", page_num + 1, page_count)
                direct_blocks = self._embedded_text_block(doc[page_num], page_num, langs)
                if direct_blocks is not None:
                    all_blocks.extend(direct_blocks)
                    total_figures += sum(
                        1 for b in direct_blocks if b.block_type == "figure")
                    self._notify_progress(
                        progress_callback, page_num + 1, progress_total,
                        f"Completed page {page_num + 1}/{page_count}")
                    continue

                img = self._render_page_image(
                    doc[page_num], scale, header_trim, footer_trim)
                h, w = img.shape[:2]
                preprocessed = self.preprocessor.preprocess(img, quality=q)

                page_blocks, n_tables, n_figures = self._process_single_page(
                    img, preprocessed, h, w, page_num, langs, yolo_confidence)

                all_blocks.extend(page_blocks)
                total_tables += n_tables
                total_figures += n_figures
                self._notify_progress(
                    progress_callback, page_num + 1, progress_total,
                    f"Completed page {page_num + 1}/{page_count}")

            full_text = self._blocks_to_text(all_blocks)

            engines = self.ocr.get_available_engines()
            active_engines = [k for k, v in engines.items() if v]
            meta_str = (
                f"Pages: {page_count}\n"
                f"Quality: {quality}\n"
                f"Language: {langs}\n"
                f"OCR Engine: {', '.join(active_engines)}\n"
                f"Tables: {total_tables}\n"
                f"Figures: {total_figures}"
            )
            self._notify_progress(
                progress_callback, page_count, progress_total,
                "Exporting output files")
            files = self.exporter.create_all_from_blocks(
                all_blocks, meta_str,
                page_size=page_size, margin_preset=margin_preset)
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

    # ── Single-page processing (group-first) ─────────────────────────────────

    def _process_single_page(self, img: np.ndarray, preprocessed: np.ndarray,
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

        logger.info(
            "Detected: page %d — text=%d, tables=%d, figures=%d, captions=%d"
            " (method=%s)",
            page_num + 1, len(text_regions), len(table_dets),
            len(figure_dets), len(caption_dets),
            "YOLO" if self.layout.model_loaded else "OpenCV",
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

        # ── Step 7: Sort into reading order ──────────────────────────
        page_blocks = _sort_reading_order(page_blocks, w)
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

    def _embedded_text_block(self, page, page_num: int, languages: str):
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

    _EMBED_RENDER_SCALE = 3.0       # render embedded figures at 3× for quality
    _EMBED_MIN_SIDE_PT = 12.0       # smallest figure side (PDF points)
    _EMBED_MIN_AREA_PT = 900.0      # smallest figure area (PDF points²)
    _MAX_VECTOR_PRIMITIVES = 500    # skip vector clustering on huge art pages

    @staticmethod
    def _page_has_drawings(page) -> bool:
        try:
            return bool(page.get_drawings())
        except (RuntimeError, ValueError):
            return False

    def _embedded_image_blocks(self, page, page_num: int,
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
            out.append(ContentBlock(
                block_type="table", page=page_num,
                y_top=float(bbox[1]), x_left=float(bbox[0]),
                text=t.get("text", ""),
                table_html=t.get("html", ""),
                bbox=[float(v) for v in bbox],
                page_width=float(iw), page_height=float(ih),
                table_meta={"col_widths": t.get("col_widths", [])},
            ))
            count += 1
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
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(w, x1), min(h, y1)
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
                            quality: str = "balanced",
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
            cfg = self.QUALITY_MAP.get(quality, self.QUALITY_MAP["balanced"])
            scale = cfg["dpi_scale"]
            doc = fitz.open(pdf_path)
            if page_num < 0 or page_num >= len(doc):
                return {"success": False, "error": "Invalid page number"}
            img = self._render_page_image(doc[page_num], scale, 0, 0)

            layout_result = self.layout.detect_layout(
                img, page_num, confidence=yolo_confidence)
            detections = layout_result.get("detections", {})

            return {
                "success": True,
                "page_image": img,
                "detections": detections,
                "image_shape": list(img.shape),
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
            quality: str = "balanced",
            header_trim: float = 0, footer_trim: float = 0,
            languages: Optional[str] = None,
            yolo_confidence: Optional[float] = None,
            page_size: str = "A4",
            margin_preset: str = "Normal",
            progress_callback: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """Process PDF with optional manual region corrections.

        Args:
            manual_regions: ``{page_num: [{"bbox": [x0,y0,x1,y1], "class": "table"|"figure"}, ...]}``
            page_size: Output page size (A4, Letter, Legal, A3, B5).
            margin_preset: Output margin preset (Normal, Narrow, Moderate, Wide).
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
            cfg = self.QUALITY_MAP.get(quality, self.QUALITY_MAP["balanced"])
            scale = cfg["dpi_scale"]
            q = cfg["quality"]

            doc = fitz.open(pdf_path)
            page_count = len(doc)
            progress_total = page_count + 1
            all_blocks: List[ContentBlock] = []
            total_tables = 0
            total_figures = 0
            pdf_name = os.path.basename(pdf_path)

            for page_num in range(page_count):
                logger.info("Processing page %d/%d", page_num + 1, page_count)
                extra = (manual_regions or {}).get(page_num, [])
                direct_blocks = self._embedded_text_block(doc[page_num], page_num, langs)
                if direct_blocks is not None:
                    all_blocks.extend(direct_blocks)
                    total_figures += sum(
                        1 for b in direct_blocks if b.block_type == "figure")
                    page_rect = doc[page_num].rect
                    correction_image = np.ones(
                        (max(1, int(page_rect.height)),
                         max(1, int(page_rect.width)), 3),
                        dtype=np.uint8) * 255
                    for mr in extra:
                        self.corrections.log_correction(
                            page_image=correction_image,
                            bbox=mr["bbox"],
                            region_class=mr.get("class", "figure"),
                            page_number=page_num,
                            pdf_name=pdf_name,
                            action="add",
                            source="manual",
                        )
                    self._notify_progress(
                        progress_callback, page_num + 1, progress_total,
                        f"Completed page {page_num + 1}/{page_count}")
                    continue

                img = self._render_page_image(
                    doc[page_num], scale, header_trim, footer_trim)
                h, w = img.shape[:2]
                preprocessed = self.preprocessor.preprocess(img, quality=q)

                # Merge manual regions into detection
                page_blocks, n_tables, n_figures = self._process_single_page(
                    img, preprocessed, h, w, page_num, langs,
                    yolo_confidence, extra_regions=extra)

                # Log manual corrections to the store
                for mr in extra:
                    self.corrections.log_correction(
                        page_image=img,
                        bbox=mr["bbox"],
                        region_class=mr.get("class", "figure"),
                        page_number=page_num,
                        pdf_name=pdf_name,
                        action="add",
                        source="manual",
                    )

                all_blocks.extend(page_blocks)
                total_tables += n_tables
                total_figures += n_figures
                self._notify_progress(
                    progress_callback, page_num + 1, progress_total,
                    f"Completed page {page_num + 1}/{page_count}")

            full_text = self._blocks_to_text(all_blocks)
            engines = self.ocr.get_available_engines()
            active_engines = [k for k, v in engines.items() if v]
            meta_str = (
                f"Pages: {page_count}\n"
                f"Quality: {quality}\n"
                f"Language: {langs}\n"
                f"OCR Engine: {', '.join(active_engines)}\n"
                f"Tables: {total_tables}\n"
                f"Figures: {total_figures}"
            )
            self._notify_progress(
                progress_callback, page_count, progress_total,
                "Exporting output files")
            files = self.exporter.create_all_from_blocks(
                all_blocks, meta_str,
                page_size=page_size, margin_preset=margin_preset)
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
        }
