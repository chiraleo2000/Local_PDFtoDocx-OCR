"""
OCR Pipeline Orchestrator — v0.4
PDF → Render → Layout Detect → GROUP (text/table/figure)
  → OCR text only → Table extraction → Image extraction
  → Build HTML → HTML→DOCX + HTML→TXT

v0.4 changes:
  - Group-first: detect layout, group regions, then process each group type
  - OCR only on text regions (figures are extracted as images, not OCR'd)
  - No LLM correction (CPU-only, simple ML pipeline)
  - 1-based figure numbering
  - Simplified API: fewer parameters, best defaults
"""
import os
import re
import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import fitz  # PyMuPDF

from .preprocessor import OpenCVPreprocessor
from .ocr_engine import OCREngine
from .layout_detector import LayoutDetector, TableExtractor
from .exporter import ImageExtractor, DocumentExporter
from .correction_store import CorrectionStore

logger = logging.getLogger(__name__)


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


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════
class OCRPipeline:
    """Full PDF → DOCX/TXT/HTML pipeline (v0.4).

    Group-first architecture:
      1. Render page
      2. Layout detection (YOLO / OpenCV)
      3. Group regions: text, table, figure
      4. OCR text regions only (not figures)
      5. Extract tables (grid + cell OCR)
      6. Extract figures as images
      7. Sort reading order
      8. Export via HTML-first
    """

    QUALITY_MAP = {
        "fast":     {"dpi_scale": 1.5, "quality": "fast"},
        "balanced": {"dpi_scale": 2.0, "quality": "balanced"},
        "accurate": {"dpi_scale": 2.5, "quality": "accurate"},
    }

    def __init__(self):
        quality = os.getenv("QUALITY_PRESET", "balanced")
        self.languages = os.getenv("LANGUAGES", "tha+eng")
        self.preprocessor = OpenCVPreprocessor(quality=quality)
        self.ocr = OCREngine()
        self.layout = LayoutDetector()
        self.table_extractor = TableExtractor()
        self.image_extractor = ImageExtractor()
        self.exporter = DocumentExporter()
        self.corrections = CorrectionStore()
        logger.info("OCR Pipeline initialised (v0.5 — group-first, HTML-first, trainable)")

    def process_pdf(self, pdf_path: str, quality: str = "balanced",
                    header_trim: float = 0, footer_trim: float = 0,
                    languages: Optional[str] = None,
                    yolo_confidence: Optional[float] = None) -> Dict[str, Any]:
        """Process a PDF end-to-end.

        Args:
            languages: Override OCR language (e.g. ``"tha+eng"``).
            yolo_confidence: Override YOLO threshold (lower = more regions).
        """
        try:
            langs = languages or self.languages
            cfg = self.QUALITY_MAP.get(quality, self.QUALITY_MAP["balanced"])
            scale = cfg["dpi_scale"]
            q = cfg["quality"]

            doc = fitz.open(pdf_path)
            page_count = len(doc)
            all_blocks: List[ContentBlock] = []
            total_tables = 0
            total_figures = 0

            for page_num in range(page_count):
                logger.info("Processing page %d/%d", page_num + 1, page_count)
                img = self._render_page_image(
                    doc[page_num], scale, header_trim, footer_trim)
                h, w = img.shape[:2]
                preprocessed = self.preprocessor.preprocess(img, quality=q)

                page_blocks, n_tables, n_figures = self._process_single_page(
                    img, preprocessed, h, w, page_num, q, langs, yolo_confidence)

                all_blocks.extend(page_blocks)
                total_tables += n_tables
                total_figures += n_figures

            doc.close()

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
            files = self.exporter.create_all_from_blocks(all_blocks, meta_str)
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

        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
            return {
                "success": False, "text": "", "files": {},
                "metadata": {}, "error": str(exc),
            }

    # ── Single-page processing (group-first) ─────────────────────────────────

    def _process_single_page(self, img: np.ndarray, preprocessed: np.ndarray,
                             h: int, w: int, page_num: int,
                             quality: str, languages: str = "tha+eng",
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

        # ── Step 1b: Merge manual corrections ────────────────────────
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

        page_blocks: List[ContentBlock] = []

        # ── Fallback: no layout detected → full-page OCR ────────────
        if not text_regions and not table_dets and not figure_dets:
            ocr_result = self.ocr.ocr_full_page(preprocessed, languages=languages)
            raw = ocr_result.get("text", "")
            txt = clean_text(raw, languages)
            if txt.strip():
                page_blocks.append(ContentBlock(
                    block_type="text", page=page_num, y_top=0, x_left=0,
                    text=txt))
            return page_blocks, 0, 0

        # ── Step 2: OCR TEXT regions only ────────────────────────────
        for region in text_regions:
            block = self._ocr_region_to_block(
                region, preprocessed, img, h, w, page_num,
                languages=languages)
            if block:
                page_blocks.append(block)

        # ── Step 3: OCR captions (also text) ─────────────────────────
        for region in caption_dets:
            block = self._ocr_region_to_block(
                region, preprocessed, img, h, w, page_num,
                block_type="caption", languages=languages)
            if block:
                page_blocks.append(block)

        # ── Step 4: OCR formulas (treat as text) ─────────────────────
        for region in formula_dets:
            block = self._ocr_region_to_block(
                region, preprocessed, img, h, w, page_num,
                block_type="text", languages=languages)
            if block:
                page_blocks.append(block)

        # ── Step 5: Extract tables (grid detect + cell OCR) ──────────
        n_tables = 0
        if table_dets:
            for t in self.table_extractor.extract_tables(
                    img, table_dets, languages=languages):
                bbox = t.get("bbox", [0, 0, 0, 0])
                page_blocks.append(ContentBlock(
                    block_type="table", page=page_num,
                    y_top=float(bbox[1]), x_left=float(bbox[0]),
                    text=t.get("text", ""),
                    table_html=t.get("html", ""),
                ))
                n_tables += 1

        # ── Step 6: Extract figures AS IMAGES (no OCR) ───────────────
        n_figures = 0
        if figure_dets:
            extracted = self.image_extractor.extract_figures(
                img, figure_dets, page_num)
            for i, fig in enumerate(extracted):
                bbox = (figure_dets[i]["bbox"]
                        if i < len(figure_dets) else [0, 0, 0, 0])
                page_blocks.append(ContentBlock(
                    block_type="figure", page=page_num,
                    y_top=float(bbox[1]), x_left=float(bbox[0]),
                    figure=fig,
                ))
                n_figures += 1

        # ── Step 7: Sort into reading order ──────────────────────────
        page_blocks = _sort_reading_order(page_blocks, w)

        return page_blocks, n_tables, n_figures

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _ocr_region_to_block(self, region: Dict, preprocessed: np.ndarray,
                             img: np.ndarray, h: int, w: int,
                             page_num: int, block_type: str = "text",
                             languages: str = "tha+eng",
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

        return ContentBlock(
            block_type=block_type, page=page_num,
            y_top=float(y0), x_left=float(x0),
            text=txt,
        )

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

            if b.block_type in ("text", "caption"):
                if b.text.strip():
                    parts.append(b.text)
            elif b.block_type == "table":
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
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
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
        try:
            cfg = self.QUALITY_MAP.get(quality, self.QUALITY_MAP["balanced"])
            scale = cfg["dpi_scale"]
            doc = fitz.open(pdf_path)
            if page_num >= len(doc):
                doc.close()
                return {"success": False, "error": "Invalid page number"}
            img = self._render_page_image(doc[page_num], scale, 0, 0)
            doc.close()

            layout_result = self.layout.detect_layout(
                img, page_num, confidence=yolo_confidence)
            detections = layout_result.get("detections", {})

            return {
                "success": True,
                "page_image": img,
                "detections": detections,
                "image_shape": list(img.shape),
            }
        except Exception as exc:
            logger.error("detect_page_regions error: %s", exc)
            return {"success": False, "error": str(exc)}

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
    ) -> Dict[str, Any]:
        """Process PDF with optional manual region corrections.

        Args:
            manual_regions: ``{page_num: [{"bbox": [x0,y0,x1,y1], "class": "table"|"figure"}, ...]}``
        """
        try:
            langs = languages or self.languages
            cfg = self.QUALITY_MAP.get(quality, self.QUALITY_MAP["balanced"])
            scale = cfg["dpi_scale"]
            q = cfg["quality"]

            doc = fitz.open(pdf_path)
            page_count = len(doc)
            all_blocks: List[ContentBlock] = []
            total_tables = 0
            total_figures = 0
            pdf_name = os.path.basename(pdf_path)

            for page_num in range(page_count):
                logger.info("Processing page %d/%d", page_num + 1, page_count)
                img = self._render_page_image(
                    doc[page_num], scale, header_trim, footer_trim)
                h, w = img.shape[:2]
                preprocessed = self.preprocessor.preprocess(img, quality=q)

                # Merge manual regions into detection
                extra = (manual_regions or {}).get(page_num, [])

                page_blocks, n_tables, n_figures = self._process_single_page(
                    img, preprocessed, h, w, page_num, q, langs,
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

            doc.close()

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
            files = self.exporter.create_all_from_blocks(all_blocks, meta_str)
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
        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
            return {
                "success": False, "text": "", "files": {},
                "metadata": {}, "error": str(exc),
            }

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
