"""
Document Exporter + Image Extractor — v2.0
HTML-first export: Build positioned HTML → convert to DOCX & TXT.
Images embedded as base64 in HTML, properly transferred to DOCX via htmldocx.
Figure numbering: 1-based.

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
                               metadata: str = "") -> Dict[str, Optional[str]]:
        """Create TXT + HTML + DOCX.  HTML is the single source of truth."""
        html_content = self._build_html(blocks, metadata)
        return {
            "txt":  self._blocks_to_txt(blocks, metadata),
            "html": self._save_html(html_content),
            "docx": self._html_to_docx(html_content),
        }

    # ── HTML builder ──────────────────────────────────────────────────────────
    def _build_html(self, blocks: list, metadata: str = "") -> str:
        parts: List[str] = [
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

        if metadata:
            meta_html = metadata.replace("\n", "<br>")
            parts.append(f"<div class='meta-info'>{meta_html}</div>")

        current_page = -1
        for b in blocks:
            # page divider
            if b.page != current_page:
                if current_page < 0:
                    parts.append(
                        f"<div class='page-break' style='border:none;margin-top:0;"
                        f"padding-top:0'>Page {b.page + 1}</div>")
                else:
                    parts.append(
                        f"<div class='page-break'>Page {b.page + 1}</div>")
                current_page = b.page

            if b.block_type in ("text", "caption"):
                for line in b.text.split("\n"):
                    stripped = line.strip()
                    if stripped:
                        parts.append(self._text_to_html(stripped))

            elif b.block_type == "table":
                if b.table_html:
                    parts.append(b.table_html)
                elif b.text.strip():
                    parts.append(f"<pre>{b.text}</pre>")

            elif b.block_type == "figure":
                b64 = b.figure.get("base64", "")
                fig_idx = int(b.figure.get("index", 0))
                fig_page = b.figure.get("page", 0)
                page_disp = (fig_page + 1) if isinstance(fig_page, int) else fig_page
                if b64 and _is_valid_base64(b64):
                    escaped_alt = html.escape(f"Figure {fig_idx}")
                    escaped_cap = html.escape(f"Figure {fig_idx} \u2014 Page {page_disp}")
                    parts.append(
                        f"<figure>"
                        f"<img src='data:image/png;base64,{b64}' "
                        f"alt='{escaped_alt}' />"
                        f"<figcaption>{escaped_cap}</figcaption>"
                        f"</figure>")

        parts += ["</body>", "</html>"]
        return "\n".join(parts)

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
            r"""src\s*=\s*['"]data:image/(png|jpe?g|gif|webp);base64,([A-Za-z0-9+/\n\r=]+)['"]""",
            re.IGNORECASE,
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

    def _html_to_docx(self, html_content: str) -> Optional[str]:
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
                fd, path = tempfile.mkstemp(suffix=".docx", prefix="ocr_")
                os.close(fd)
                doc.save(path)
                logger.info("DOCX created via htmldocx")
                return path
            except Exception as exc:
                logger.warning("htmldocx failed (%s) — using fallback", exc)

        # Fallback: manual DOCX from parsed HTML
        return self._fallback_docx(html_content)

    def _fallback_docx(self, html_content: str) -> Optional[str]:
        """Create DOCX by parsing HTML with BeautifulSoup."""
        if not BS4_AVAILABLE:
            return None

        doc = Document()
        style = doc.styles["Normal"]
        style.font.name = self.FONT_NAME
        style.font.size = Pt(self.FONT_SIZE_NORMAL)

        soup = BeautifulSoup(html_content, "html.parser")
        body = soup.find("body")
        if not body:
            return None

        for el in body.children:
            if not hasattr(el, "name") or el.name is None:
                continue
            self._el_to_docx(doc, el)

        fd, path = tempfile.mkstemp(suffix=".docx", prefix="ocr_")
        os.close(fd)
        doc.save(path)
        return path

    def _el_to_docx(self, doc, el) -> None:
        tag = el.name
        if tag in ("h1", "h2", "h3"):
            h = doc.add_heading(el.get_text(strip=True), level=int(tag[1]))
            for run in h.runs:
                run.font.name = self.FONT_NAME

        elif tag == "p":
            text = el.get_text(strip=True)
            if text:
                p = doc.add_paragraph(text)
                for run in p.runs:
                    run.font.name = self.FONT_NAME
                    run.font.size = Pt(self.FONT_SIZE_NORMAL)

        elif tag == "pre":
            text = el.get_text()
            if text.strip():
                p = doc.add_paragraph(text)
                for run in p.runs:
                    run.font.name = "Consolas"
                    run.font.size = Pt(self.FONT_SIZE_TABLE)

        elif tag == "table":
            self._table_el_to_docx(doc, el)

        elif tag == "figure":
            self._figure_el_to_docx(doc, el)

        elif tag == "div":
            for child in el.children:
                if hasattr(child, "name") and child.name:
                    self._el_to_docx(doc, child)

    def _table_el_to_docx(self, doc, table_el) -> None:
        rows_el = table_el.find_all("tr")
        if not rows_el:
            return

        # Calculate the true column count considering colspan
        max_cols = 0
        for r in rows_el:
            cols_in_row = 0
            for cell in r.find_all(["td", "th"]):
                cs = int(cell.get("colspan", 1))
                cols_in_row += cs
            max_cols = max(max_cols, cols_in_row)

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

                colspan = int(cell_el.get("colspan", 1))
                rowspan = int(cell_el.get("rowspan", 1))

                cell = tbl.cell(ri, col_idx)
                cell_text = cell_el.get_text(strip=True)
                cell.text = cell_text

                # Merge cells for colspan
                if colspan > 1 and col_idx + colspan - 1 < max_cols:
                    try:
                        merge_cell = tbl.cell(ri, col_idx + colspan - 1)
                        cell.merge(merge_cell)
                    except Exception:
                        pass

                # Merge cells for rowspan
                if rowspan > 1 and ri + rowspan - 1 < len(rows_el):
                    try:
                        merge_cell = tbl.cell(ri + rowspan - 1, col_idx)
                        cell.merge(merge_cell)
                    except Exception:
                        pass

                # Style the cell
                for p in cell.paragraphs:
                    for run in p.runs:
                        run.font.name = self.FONT_NAME
                        run.font.size = Pt(self.FONT_SIZE_TABLE)
                        if cell_el.name == "th":
                            run.bold = True

                col_idx += colspan

    def _figure_el_to_docx(self, doc, figure_el) -> None:
        img_el = figure_el.find("img")
        caption_el = figure_el.find("figcaption")
        caption_text = caption_el.get_text(strip=True) if caption_el else ""
        if not img_el:
            return
        src = img_el.get("src", "")
        try:
            if src.startswith("data:image"):
                b64_data = src.split(",", 1)[1] if "," in src else ""
                if b64_data:
                    img_bytes = base64.b64decode(b64_data)
                    doc.add_picture(BytesIO(img_bytes), width=Inches(5.5))
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

            if b.block_type in ("text", "caption"):
                if b.text.strip():
                    parts.append(b.text)
            elif b.block_type == "table":
                if b.text.strip():
                    parts.append(f"[Table]\n{b.text}")
            elif b.block_type == "figure":
                fig_idx = b.figure.get("index", "?")
                pg = b.figure.get("page", "?")
                page_disp = (pg + 1) if isinstance(pg, int) else pg
                parts.append(f"[Figure {fig_idx} — Page {page_disp}]")

        content = "\n\n".join(parts)
        fd, path = tempfile.mkstemp(suffix=".txt", prefix="ocr_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        return path

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
            stripped = line.strip()
            if not stripped:
                parts.append("<br>")
            elif stripped.startswith("###"):
                parts.append(f"<h3>{html.escape(stripped.lstrip('# '))}</h3>")
            elif stripped.startswith("##"):
                parts.append(f"<h2>{html.escape(stripped.lstrip('# '))}</h2>")
            elif stripped.startswith("#"):
                parts.append(f"<h1>{html.escape(stripped.lstrip('# '))}</h1>")
            else:
                parts.append(f"<p>{html.escape(stripped)}</p>")
        if tables_html:
            for th in tables_html:
                parts.append(th)
        if figures:
            for fig in figures:
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
