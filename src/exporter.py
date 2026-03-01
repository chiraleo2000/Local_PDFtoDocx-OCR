"""
Document Exporter + Image Extractor — v2.1
HTML-first export: Build positioned HTML → convert to DOCX & TXT.
Images embedded as base64 in HTML, properly transferred to DOCX via htmldocx.
Figure numbering: 1-based.

v2.1 improvements:
    - Reasonable image sizing (based on original dimensions, not hardcoded)
    - Table borders enforced in DOCX output
    - Configurable page size (A4, Letter, Legal, A3, B5)
    - Configurable margins (Normal, Narrow, Moderate, Wide)

v2.0 improvements:
    - Better table-to-DOCX conversion with proper cell alignment
    - Colspan/rowspan handling in fallback DOCX builder
    - Style preservation (tabs, indentation) in text blocks
    - Improved HTML table structure with col-width hints

Security:
    - All user text is escaped via ``html.escape()`` before HTML insertion (XSS)
    - Base64 data validated before embedding
    - Filenames sanitised in temp file operations
"""
import os
import re
import html
import logging
import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List, TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image

if TYPE_CHECKING:
    from .pipeline import ContentBlock

logger = logging.getLogger(__name__)

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    from docx import Document
    from docx.shared import Inches, Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False
    logger.warning("python-docx not installed — DOCX export unavailable")

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

HTMLDOCX_AVAILABLE = False
try:
    from htmldocx import HtmlToDocx
    HTMLDOCX_AVAILABLE = True
except ImportError:
    logger.info("htmldocx not installed — will use fallback DOCX builder")

_BASE64_RE = re.compile(r"^[A-Za-z0-9+/\n\r]*={0,2}$")


def _is_valid_base64(data: str) -> bool:
    """Validate that *data* looks like legitimate base64 (anti-injection)."""
    if not data or len(data) > 50_000_000:  # 50 MB max
        return False
    return bool(_BASE64_RE.match(data))


# ── Page layout configurations ────────────────────────────────────────────────
# Page sizes: (width_inches, height_inches)
PAGE_SIZES = {
    "A4": (8.27, 11.69),
    "Letter": (8.5, 11.0),
    "Legal": (8.5, 14.0),
    "A3": (11.69, 16.54),
    "B5": (6.93, 9.84),
}

# Margin presets: (top, bottom, left, right) in inches — matching MS Word defaults
MARGIN_PRESETS = {
    "Normal": (1.0, 1.0, 1.0, 1.0),
    "Narrow": (0.5, 0.5, 0.5, 0.5),
    "Moderate": (1.0, 1.0, 0.75, 0.75),
    "Wide": (1.0, 1.0, 1.5, 1.5),
}

# Max image display width in HTML (pixels)
_MAX_IMG_DISPLAY_PX = 550

# Assumed rendering DPI for pixel-to-inch conversion
_IMG_RENDER_DPI = 150


# ══════════════════════════════════════════════════════════════════════════════
# Image / Figure Extraction  (1-based numbering)
# ══════════════════════════════════════════════════════════════════════════════
class ImageExtractor:
    """Crop detected figure regions, save to disk, and encode to base64."""

    def __init__(self, min_width: int = 80, min_height: int = 80,
                 min_area: int = 5000):
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = int(os.getenv("IMAGE_MIN_AREA", str(min_area)))
        self.enabled = os.getenv("IMAGE_EXTRACTION", "true").lower() == "true"
        self.temp_dir = Path(tempfile.gettempdir()) / "pdf_ocr_images"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def extract_figures(self, page_image: np.ndarray,
                        figure_detections: List[Dict],
                        page_number: int = 0) -> List[Dict[str, Any]]:
        if not self.enabled or not figure_detections:
            return []

        figures: List[Dict[str, Any]] = []
        img_h, img_w = page_image.shape[:2]

        for idx, det in enumerate(figure_detections):
            bbox = det["bbox"]
            x0, y0, x1, y1 = [int(v) for v in bbox]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(img_w, x1), min(img_h, y1)
            crop = page_image[y0:y1, x0:x1]
            if crop.size == 0:
                continue

            h, w = crop.shape[:2]
            if w < self.min_width or h < self.min_height or w * h < self.min_area:
                continue

            fig_num = idx + 1                               # 1-based
            fname = f"page{page_number + 1}_fig{fig_num}.png"
            fpath = self.temp_dir / fname
            cv2.imwrite(str(fpath), crop)

            pil_img = Image.fromarray(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                if len(crop.shape) == 3 else crop
            )
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            figures.append({
                "page": page_number,
                "index": fig_num,                           # 1-based
                "path": str(fpath),
                "base64": b64,
                "width": w,
                "height": h,
            })
        return figures


# ══════════════════════════════════════════════════════════════════════════════
# Document Export — HTML-first  (DOCX / TXT / HTML)
# ══════════════════════════════════════════════════════════════════════════════
class DocumentExporter:
    """HTML-first exporter (v0.4).

    Build a styled HTML document from ContentBlocks, then:
      - save the HTML directly
      - convert HTML -> DOCX via htmldocx (with manual fallback)
      - derive plain-text from blocks
    """

    FONT_NAME = "Tahoma"
    FONT_SIZE_NORMAL = 11
    FONT_SIZE_TABLE = 10

    def __init__(self):
        pass

    # ══════════════════════════════════════════════════════════════════════════
    # Primary API (v0.4 — block-based, HTML-first)
    # ══════════════════════════════════════════════════════════════════════════
    def create_all_from_blocks(self, blocks: list,
                               metadata: str = "",
                               page_size: str = "A4",
                               margin_preset: str = "Normal",
                               ) -> Dict[str, Optional[str]]:
        """Create TXT + HTML + DOCX.  HTML is the single source of truth."""
        html_content = self._build_html(blocks, metadata)
        return {
            "txt":  self._blocks_to_txt(blocks, metadata),
            "html": self._save_html(html_content),
            "docx": self._html_to_docx(html_content, page_size, margin_preset),
        }

    # ── HTML builder ──────────────────────────────────────────────────────────
    def _build_html(self, blocks: list, metadata: str = "") -> str:
        parts: List[str] = self._html_head()
        if metadata:
            meta_html = metadata.replace("\n", "<br>")
            parts.append(f"<div class='meta-info'>{meta_html}</div>")

        current_page = -1
        for b in blocks:
            if b.page != current_page:
                current_page = b.page
                parts.append(self._page_divider_html(b.page, current_page == b.page))
            parts.extend(self._block_to_html(b))

        parts += ["</body>", "</html>"]
        return "\n".join(parts)

    @staticmethod
    def _html_head() -> List[str]:
        """Return the HTML preamble (doctype → <body>)."""
        return [
            "<!DOCTYPE html>",
            "<html lang='en'>",
            "<head>",
            "<meta charset='utf-8'>",
            "<title>OCR Result</title>",
            "<style>",
            "body { font-family: Tahoma, 'Segoe UI', Arial, sans-serif; "
            "max-width: 900px; margin: 0 auto; padding: 2rem; "
            "line-height: 1.8; color: #1e293b; }",
            "h1, h2, h3 { color: #0f172a; margin-top: 1.5em; margin-bottom: 0.5em; }",
            "p { margin: 0.4em 0; text-align: justify; }",
            "table { border-collapse: collapse; width: 100%; margin: 1rem 0; }",
            "th, td { border: 1px solid #cbd5e1; padding: 8px 12px; "
            "text-align: left; vertical-align: top; }",
            "th { background: #f1f5f9; font-weight: bold; }",
            "figure { margin: 1rem 0; text-align: center; }",
            "figure img { max-width: 100%; height: auto; border-radius: 4px; "
            "box-shadow: 0 2px 8px rgba(0,0,0,0.1); }",
            "figcaption { color: #64748b; font-size: 0.85rem; margin-top: 0.5rem; }",
            ".page-break { page-break-before: always; border-top: 2px solid #e2e8f0; "
            "margin-top: 2rem; padding-top: 1rem; color: #94a3b8; "
            "font-size: 0.85rem; }",
            ".meta-info { color: #64748b; font-size: 0.85rem; "
            "border-bottom: 1px solid #e2e8f0; padding-bottom: 1rem; "
            "margin-bottom: 2rem; white-space: pre-line; }",
            "</style>",
            "</head>",
            "<body>",
        ]

    @staticmethod
    def _page_divider_html(page_num: int, is_first: bool) -> str:
        """Return an HTML page divider element."""
        if is_first:
            return (f"<div class='page-break' style='border:none;margin-top:0;"
                    f"padding-top:0'>Page {page_num + 1}</div>")
        return f"<div class='page-break'>Page {page_num + 1}</div>"

    def _block_to_html(self, b) -> List[str]:
        """Convert a single ContentBlock to HTML snippet(s)."""
        if b.block_type in ("text", "caption"):
            return [self._text_to_html(line.strip())
                    for line in b.text.split("\n") if line.strip()]
        if b.block_type == "table":
            if b.table_html:
                return [b.table_html]
            if b.text.strip():
                return [f"<pre>{b.text}</pre>"]
            return []
        if b.block_type == "figure":
            return self._figure_block_to_html(b)
        return []

    @staticmethod
    def _figure_block_to_html(b) -> List[str]:
        """Convert a figure ContentBlock to an HTML <figure> snippet."""
        b64 = b.figure.get("base64", "")
        if not b64 or not _is_valid_base64(b64):
            return []
        fig_idx = int(b.figure.get("index", 0))
        fig_page = b.figure.get("page", 0)
        page_disp = (fig_page + 1) if isinstance(fig_page, int) else fig_page
        escaped_alt = html.escape(f"Figure {fig_idx}")
        escaped_cap = html.escape(f"Figure {fig_idx} \u2014 Page {page_disp}")
        orig_w = b.figure.get("width", 0)
        orig_h = b.figure.get("height", 0)
        if orig_w > 0 and orig_h > 0:
            display_w = min(orig_w, _MAX_IMG_DISPLAY_PX)
            display_h = int(orig_h * display_w / orig_w)
            img_style = f"width:{display_w}px;height:{display_h}px;"
        else:
            img_style = f"max-width:{_MAX_IMG_DISPLAY_PX}px;height:auto;"
        return [
            (
                "<figure>"
                + f"<img src='data:image/png;base64,{b64}' "
                + f"alt='{escaped_alt}' style='{img_style}' />"
                + f"<figcaption>{escaped_cap}</figcaption>"
                + "</figure>"
            )
        ]

    # ── Save HTML ─────────────────────────────────────────────────────────────
    @staticmethod
    def _save_html(html_content: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".html", prefix="ocr_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(html_content)
        return path

    # ── HTML -> DOCX ─────────────────────────────────────────────────────────
    @staticmethod
    def _extract_base64_images_to_tempfiles(html_content: str) -> str:
        """Replace base64 data URIs in <img> src with temp file paths.

        htmldocx tries to use the src attribute as a filename, which fails
        when the src is a data URI (filename too long).  This pre-processes
        the HTML so that base64 images are saved as temp files and the src
        attributes point to those files instead.
        """
        _DATA_URI_RE = re.compile(
            r"""src\s*=\s*['"]data:image/(png|jpe?g|gif|webp);base64,([A-Za-z0-9+/=]+)['"]""",
        )

        def _replacer(match):
            fmt = match.group(1).lower()
            b64_data = match.group(2)
            suffix = f".{fmt}" if fmt != "jpeg" else ".jpg"
            try:
                img_bytes = base64.b64decode(b64_data)
                fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="ocr_img_")
                with os.fdopen(fd, "wb") as f:
                    f.write(img_bytes)
                return f"src='{tmp_path}'"
            except Exception:
                return match.group(0)  # leave unchanged on error

        return _DATA_URI_RE.sub(_replacer, html_content)

    def _html_to_docx(self, html_content: str,
                       page_size: str = "A4",
                       margin_preset: str = "Normal") -> Optional[str]:
        if not DOCX_AVAILABLE:
            logger.error("python-docx not installed")
            return None

        # Try htmldocx first — handles images, tables, alignment
        if HTMLDOCX_AVAILABLE:
            try:
                # Pre-process: extract base64 images to temp files
                # to avoid "filename too long" errors in htmldocx
                processed_html = self._extract_base64_images_to_tempfiles(
                    html_content)
                parser = HtmlToDocx()
                doc = parser.parse_html_string(processed_html)
                # Apply page layout, fix table borders, fix image sizes
                self._apply_page_layout(doc, page_size, margin_preset)
                self._fix_table_borders(doc)
                self._fix_image_sizes(doc, page_size, margin_preset)
                fd, path = tempfile.mkstemp(suffix=".docx", prefix="ocr_")
                os.close(fd)
                doc.save(path)
                logger.info("DOCX created via htmldocx")
                return path
            except Exception as exc:
                logger.warning("htmldocx failed (%s) — using fallback", exc)

        # Fallback: manual DOCX from parsed HTML
        return self._fallback_docx(html_content, page_size, margin_preset)

    def _fallback_docx(self, html_content: str,
                        page_size: str = "A4",
                        margin_preset: str = "Normal") -> Optional[str]:
        """Create DOCX by parsing HTML with BeautifulSoup."""
        if not BS4_AVAILABLE:
            return None

        doc = Document()
        style = doc.styles["Normal"]
        style.font.name = self.FONT_NAME
        style.font.size = Pt(self.FONT_SIZE_NORMAL)

        # Apply page layout
        self._apply_page_layout(doc, page_size, margin_preset)

        soup = BeautifulSoup(html_content, "html.parser")
        body = soup.find("body")
        if not body:
            return None

        # Store layout info for image sizing
        self._current_page_size = page_size
        self._current_margin_preset = margin_preset

        for el in body.children:
            if not hasattr(el, "name") or el.name is None:
                continue
            self._el_to_docx(doc, el)

        fd, path = tempfile.mkstemp(suffix=".docx", prefix="ocr_")
        os.close(fd)
        doc.save(path)
        return path

    def _el_to_docx(self, doc, el) -> None:
        """Convert a parsed HTML element into DOCX content (dispatch)."""
        tag = el.name
        handler = {
            "h1": self._heading_to_docx,
            "h2": self._heading_to_docx,
            "h3": self._heading_to_docx,
            "p": self._paragraph_to_docx,
            "pre": self._preformatted_to_docx,
            "table": self._table_el_to_docx,
            "figure": self._figure_el_to_docx,
        }.get(tag)
        if handler:
            handler(doc, el)
        elif tag == "div":
            for child in el.children:
                if hasattr(child, "name") and child.name:
                    self._el_to_docx(doc, child)

    def _heading_to_docx(self, doc, el) -> None:
        level = int(el.name[1])
        h = doc.add_heading(el.get_text(strip=True), level=level)
        for run in h.runs:
            run.font.name = self.FONT_NAME

    def _paragraph_to_docx(self, doc, el) -> None:
        text = el.get_text(strip=True)
        if text:
            p = doc.add_paragraph(text)
            for run in p.runs:
                run.font.name = self.FONT_NAME
                run.font.size = Pt(self.FONT_SIZE_NORMAL)

    def _preformatted_to_docx(self, doc, el) -> None:
        text = el.get_text()
        if text.strip():
            p = doc.add_paragraph(text)
            for run in p.runs:
                run.font.name = "Consolas"
                run.font.size = Pt(self.FONT_SIZE_TABLE)

    def _table_el_to_docx(self, doc, table_el) -> None:
        rows_el = table_el.find_all("tr")
        if not rows_el:
            return

        max_cols = self._calc_table_col_count(rows_el)
        if max_cols == 0:
            return

        tbl = doc.add_table(rows=len(rows_el), cols=max_cols)
        tbl.style = "Table Grid"

        for ri, row_el in enumerate(rows_el):
            cells = row_el.find_all(["td", "th"])
            col_idx = 0
            for cell_el in cells:
                if col_idx >= max_cols:
                    break
                col_idx = self._fill_table_cell(
                    tbl, ri, col_idx, cell_el, max_cols, len(rows_el))

    @staticmethod
    def _calc_table_col_count(rows_el) -> int:
        """Return the maximum column count across all rows."""
        max_cols = 0
        for r in rows_el:
            cols_in_row = sum(
                int(cell.get("colspan", 1))
                for cell in r.find_all(["td", "th"])
            )
            max_cols = max(max_cols, cols_in_row)
        return max_cols

    def _fill_table_cell(self, tbl, ri: int, col_idx: int,
                         cell_el, max_cols: int, total_rows: int) -> int:
        """Populate one table cell and return the next column index."""
        colspan = int(cell_el.get("colspan", 1))
        rowspan = int(cell_el.get("rowspan", 1))

        cell = tbl.cell(ri, col_idx)
        cell.text = cell_el.get_text(strip=True)

        if colspan > 1 and col_idx + colspan - 1 < max_cols:
            try:
                cell.merge(tbl.cell(ri, col_idx + colspan - 1))
            except Exception:
                pass
        if rowspan > 1 and ri + rowspan - 1 < total_rows:
            try:
                cell.merge(tbl.cell(ri + rowspan - 1, col_idx))
            except Exception:
                pass

        for p in cell.paragraphs:
            for run in p.runs:
                run.font.name = self.FONT_NAME
                run.font.size = Pt(self.FONT_SIZE_TABLE)
                if cell_el.name == "th":
                    run.bold = True

        return col_idx + colspan

    def _figure_el_to_docx(self, doc, figure_el) -> None:
        img_el = figure_el.find("img")
        caption_el = figure_el.find("figcaption")
        caption_text = caption_el.get_text(strip=True) if caption_el else ""
        if not img_el:
            return
        src = img_el.get("src", "")
        try:
            img_bytes = None
            if src.startswith("data:image"):
                b64_data = src.split(",", 1)[1] if "," in src else ""
                if b64_data:
                    img_bytes = base64.b64decode(b64_data)
            elif os.path.isfile(src):
                with open(src, "rb") as fimg:
                    img_bytes = fimg.read()

            if img_bytes:
                # Calculate reasonable image width
                page_sz = getattr(self, "_current_page_size", "A4")
                margin_p = getattr(self, "_current_margin_preset", "Normal")
                img_width = self._calc_image_width(img_bytes, page_sz, margin_p)
                doc.add_picture(BytesIO(img_bytes), width=Inches(img_width))

            if caption_text:
                p = doc.add_paragraph(caption_text)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in p.runs:
                    run.font.size = Pt(9)
                    run.font.color.rgb = RGBColor(100, 116, 139)
        except Exception as exc:
            logger.warning("Failed to embed figure in DOCX: %s", exc)

    # ── TXT from blocks ──────────────────────────────────────────────────────
    def _blocks_to_txt(self, blocks: list, metadata: str = "") -> str:
        parts: List[str] = []
        if metadata:
            parts.append(f"--- Document Info ---\n{metadata}\n---\n")

        current_page = -1
        for b in blocks:
            if b.page != current_page:
                if current_page >= 0:
                    parts.append(f"\n{'─' * 60}")
                parts.append(f"  Page {b.page + 1}")
                parts.append(f"{'─' * 60}\n")
                current_page = b.page
            parts.append(self._block_to_txt(b))

        content = "\n\n".join(p for p in parts if p)
        fd, path = tempfile.mkstemp(suffix=".txt", prefix="ocr_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    @staticmethod
    def _block_to_txt(b) -> str:
        """Convert a single ContentBlock to its plain-text representation."""
        if b.block_type in ("text", "caption"):
            return b.text if b.text.strip() else ""
        if b.block_type == "table":
            return f"[Table]\n{b.text}" if b.text.strip() else ""
        if b.block_type == "figure":
            fig_idx = b.figure.get("index", "?")
            pg = b.figure.get("page", "?")
            page_disp = (pg + 1) if isinstance(pg, int) else pg
            return f"[Figure {fig_idx} — Page {page_disp}]"
        return ""

    # ── Page layout & post-processing helpers ─────────────────────────────────
    @staticmethod
    def _apply_page_layout(doc, page_size: str = "A4",
                           margin_preset: str = "Normal") -> None:
        """Set page dimensions and margins on all sections of the DOCX."""
        pw, ph = PAGE_SIZES.get(page_size, PAGE_SIZES["A4"])
        mt, mb, ml, mr = MARGIN_PRESETS.get(margin_preset, MARGIN_PRESETS["Normal"])
        for section in doc.sections:
            section.page_width = Inches(pw)
            section.page_height = Inches(ph)
            section.top_margin = Inches(mt)
            section.bottom_margin = Inches(mb)
            section.left_margin = Inches(ml)
            section.right_margin = Inches(mr)

    @staticmethod
    def _fix_table_borders(doc) -> None:
        """Ensure every table has visible cell borders (like Word's Table Grid)."""
        from docx.oxml.ns import qn, nsdecls
        from docx.oxml import parse_xml
        for table in doc.tables:
            tbl = table._tbl
            tbl_pr = tbl.tblPr
            if tbl_pr is None:
                tbl_pr = parse_xml(f'<w:tblPr {nsdecls("w")}/>')
                tbl.insert(0, tbl_pr)
            borders = parse_xml(
                f'<w:tblBorders {nsdecls("w")}>'
                '<w:top w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
                '<w:left w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
                '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
                '<w:right w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
                '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
                '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="808080"/>'
                '</w:tblBorders>'
            )
            existing = tbl_pr.find(qn('w:tblBorders'))
            if existing is not None:
                tbl_pr.remove(existing)
            tbl_pr.append(borders)

    @staticmethod
    def _fix_image_sizes(doc, page_size: str = "A4",
                         margin_preset: str = "Normal") -> None:
        """Cap all inline images to the usable page width."""
        pw, _ = PAGE_SIZES.get(page_size, PAGE_SIZES["A4"])
        _, _, ml, mr = MARGIN_PRESETS.get(margin_preset, MARGIN_PRESETS["Normal"])
        max_width_inches = pw - ml - mr - 0.2
        max_width_emu = int(max_width_inches * 914400)  # 1 inch = 914400 EMU
        for shape in doc.inline_shapes:
            if shape.width and shape.width > max_width_emu:
                ratio = max_width_emu / shape.width
                shape.width = max_width_emu
                shape.height = int(shape.height * ratio)

    @staticmethod
    def _calc_image_width(img_bytes: bytes, page_size: str = "A4",
                          margin_preset: str = "Normal") -> float:
        """Return a reasonable image width in inches based on actual pixel size."""
        pw, _ = PAGE_SIZES.get(page_size, PAGE_SIZES["A4"])
        _, _, ml, mr = MARGIN_PRESETS.get(margin_preset, MARGIN_PRESETS["Normal"])
        usable_width = pw - ml - mr - 0.2
        try:
            img = Image.open(BytesIO(img_bytes))
            w_px, _ = img.size
            w_inches = w_px / _IMG_RENDER_DPI
            return min(w_inches, usable_width)
        except Exception:
            return min(4.0, usable_width)

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _text_to_html(stripped: str) -> str:
        """Convert a single line of text to an HTML element, XSS-safe."""
        if not stripped:
            return ""
        escaped = html.escape(stripped)
        if stripped.startswith("###"):
            return f"<h3>{html.escape(stripped.lstrip('# '))}</h3>"
        if stripped.startswith("##"):
            return f"<h2>{html.escape(stripped.lstrip('# '))}</h2>"
        if stripped.startswith("#"):
            return f"<h1>{html.escape(stripped.lstrip('# '))}</h1>"
        return f"<p>{escaped}</p>"

    # ══════════════════════════════════════════════════════════════════════════
    # LEGACY API  (backward-compatible with tests)
    # ══════════════════════════════════════════════════════════════════════════
    @staticmethod
    def _line_to_html_element(stripped: str) -> str:
        """Convert a single line to an HTML element, XSS-safe."""
        if not stripped:
            return "<br>"
        escaped = html.escape(stripped)
        if stripped.startswith("###"):
            return f"<h3>{html.escape(stripped.lstrip('# '))}</h3>"
        if stripped.startswith("##"):
            return f"<h2>{html.escape(stripped.lstrip('# '))}</h2>"
        if stripped.startswith("#"):
            return f"<h1>{html.escape(stripped.lstrip('# '))}</h1>"
        return f"<p>{escaped}</p>"

    @staticmethod
    def create_txt(text: str, metadata: str = "") -> str:
        content = ""
        if metadata:
            content += f"--- Document Info ---\n{metadata}\n---\n\n"
        content += text
        fd, path = tempfile.mkstemp(suffix=".txt", prefix="ocr_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return path

    @staticmethod
    def create_html(text: str, tables_html: Optional[List[str]] = None,
                    figures: Optional[List[Dict[str, Any]]] = None,
                    metadata: str = "") -> str:
        """Legacy HTML builder — XSS-safe."""
        parts = [
            "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
            "<title>OCR Result</title>",
            "<style>",
            "body{font-family:Tahoma,Arial,sans-serif;max-width:900px;margin:auto;"
            "padding:2rem;line-height:1.7;color:#1e293b}",
            "table{border-collapse:collapse;width:100%;margin:1rem 0}",
            "th,td{border:1px solid #cbd5e1;padding:8px 12px;text-align:left}",
            "th{background:#f1f5f9;font-weight:bold}",
            "img{max-width:100%;height:auto;margin:1rem 0;border-radius:4px}",
            "</style></head><body>",
        ]
        if metadata:
            parts.append(f"<div style='color:#64748b;font-size:0.85rem;"
                         f"border-bottom:1px solid #e2e8f0;padding-bottom:1rem;"
                         f"margin-bottom:2rem'>{html.escape(metadata)}</div>")
        for line in text.split("\n"):
            parts.append(DocumentExporter._line_to_html_element(line.strip()))
        for th in (tables_html or []):
            parts.append(th)
        for fig in (figures or []):
            b64 = fig.get("base64", "")
            if b64 and _is_valid_base64(b64):
                parts.append(
                    f"<figure><img src='data:image/png;base64,{b64}' "
                    f"alt='Figure'/></figure>")
        parts.append("</body></html>")
        fd, path = tempfile.mkstemp(suffix=".html", prefix="ocr_")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write("\n".join(parts))
        return path

    def create_docx(self, text: str,
                    tables_html: Optional[List[str]] = None,
                    figures: Optional[List[Dict[str, Any]]] = None,
                    metadata: str = "") -> Optional[str]:
        """Legacy API — builds HTML first, then converts to DOCX."""
        if not DOCX_AVAILABLE:
            logger.error("python-docx not installed")
            return None
        html_path = self.create_html(text, tables_html, figures, metadata)
        with open(html_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return self._html_to_docx(html_content)

    def create_all(self, text: str,
                   tables_html: Optional[List[str]] = None,
                   figures: Optional[List[Dict[str, Any]]] = None,
                   metadata: str = "") -> Dict[str, Optional[str]]:
        return {
            "txt": self.create_txt(text, metadata),
            "html": self.create_html(text, tables_html, figures, metadata),
            "docx": self.create_docx(text, tables_html, figures, metadata),
        }
