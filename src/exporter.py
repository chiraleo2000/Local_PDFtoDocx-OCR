"""
Document Exporter + Image Extractor
Export OCR results to DOCX, TXT, HTML with embedded figures and tables.
"""
import os
import logging
import base64
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Dict, Any, Optional, List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

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


# ══════════════════════════════════════════════════════════════════════════════
# Image / Figure Extraction
# ══════════════════════════════════════════════════════════════════════════════
class ImageExtractor:
    """Crop detected figure regions, save to disk, and encode to base64."""

    def __init__(self, min_width: int = 80, min_height: int = 80,
                 min_area: int = 10000):
        self.min_width = min_width
        self.min_height = min_height
        self.min_area = min_area
        self.enabled = os.getenv("IMAGE_EXTRACTION", "true").lower() == "true"
        self.temp_dir = Path(tempfile.gettempdir()) / "pdf_ocr_images"
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def extract_figures(self, page_image: np.ndarray,
                        figure_detections: List[Dict],
                        page_number: int = 0) -> List[Dict[str, Any]]:
        if not self.enabled or not figure_detections:
            return []

        figures = []
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

            fname = f"page{page_number}_fig{idx}.png"
            fpath = self.temp_dir / fname
            cv2.imwrite(str(fpath), crop)

            pil_img = Image.fromarray(
                cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if len(crop.shape) == 3 else crop
            )
            buf = BytesIO()
            pil_img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()

            figures.append({
                "page": page_number, "index": idx,
                "path": str(fpath), "base64": b64,
                "width": w, "height": h,
            })
        return figures


# ══════════════════════════════════════════════════════════════════════════════
# Document Export (DOCX / TXT / HTML)
# ══════════════════════════════════════════════════════════════════════════════
class DocumentExporter:
    """Export pipeline results to DOCX / TXT / HTML."""

    FONT_NAME = "Tahoma"
    FONT_SIZE_NORMAL = 11
    FONT_SIZE_TABLE = 10

    def __init__(self):
        pass

    # ── TXT ───────────────────────────────────────────────────────────────────
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

    # ── HTML ──────────────────────────────────────────────────────────────────
    @staticmethod
    def create_html(text: str, tables_html: Optional[List[str]] = None,
                    figures: Optional[List[Dict[str, Any]]] = None,
                    metadata: str = "") -> str:
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
                         f"margin-bottom:2rem'>{metadata}</div>")
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                parts.append("<br>")
            elif stripped.startswith("###"):
                parts.append(f"<h3>{stripped.lstrip('# ')}</h3>")
            elif stripped.startswith("##"):
                parts.append(f"<h2>{stripped.lstrip('# ')}</h2>")
            elif stripped.startswith("#"):
                parts.append(f"<h1>{stripped.lstrip('# ')}</h1>")
            else:
                parts.append(f"<p>{stripped}</p>")
        if tables_html:
            for th in tables_html:
                parts.append(th)
        if figures:
            for fig in figures:
                b64 = fig.get("base64", "")
                if b64:
                    parts.append(
                        f"<figure><img src='data:image/png;base64,{b64}' "
                        f"alt='Figure'/></figure>"
                    )
        parts.append("</body></html>")
        fd, path = tempfile.mkstemp(suffix=".html", prefix="ocr_")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write("\n".join(parts))
        return path

    # ── DOCX ──────────────────────────────────────────────────────────────────
    def create_docx(self, text: str,
                    tables_html: Optional[List[str]] = None,
                    figures: Optional[List[Dict[str, Any]]] = None,
                    metadata: str = "") -> Optional[str]:
        if not DOCX_AVAILABLE:
            logger.error("python-docx not installed")
            return None

        doc = Document()
        style = doc.styles["Normal"]
        font = style.font
        font.name = self.FONT_NAME
        font.size = Pt(self.FONT_SIZE_NORMAL)

        if metadata:
            p = doc.add_paragraph()
            run = p.add_run(metadata)
            run.font.size = Pt(9)
            run.font.color.rgb = RGBColor(100, 116, 139)
            doc.add_paragraph("")

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                doc.add_paragraph("")
                continue
            if stripped.startswith("###"):
                h = doc.add_heading(stripped.lstrip("# "), level=3)
                for run in h.runs:
                    run.font.name = self.FONT_NAME
                continue
            if stripped.startswith("##"):
                h = doc.add_heading(stripped.lstrip("# "), level=2)
                for run in h.runs:
                    run.font.name = self.FONT_NAME
                continue
            if stripped.startswith("#"):
                h = doc.add_heading(stripped.lstrip("# "), level=1)
                for run in h.runs:
                    run.font.name = self.FONT_NAME
                continue
            if stripped.startswith(("\u2022 ", "- ", "* ")):
                p = doc.add_paragraph(stripped[2:], style="List Bullet")
                for run in p.runs:
                    run.font.name = self.FONT_NAME
                    run.font.size = Pt(self.FONT_SIZE_NORMAL)
                continue
            p = doc.add_paragraph(stripped)
            for run in p.runs:
                run.font.name = self.FONT_NAME
                run.font.size = Pt(self.FONT_SIZE_NORMAL)

        if tables_html and BS4_AVAILABLE:
            for th in tables_html:
                self._add_html_table(doc, th)
        if figures:
            for fig in figures:
                self._add_figure(doc, fig)

        fd, path = tempfile.mkstemp(suffix=".docx", prefix="ocr_")
        os.close(fd)
        doc.save(path)
        return path

    def _add_html_table(self, doc, html: str):
        soup = BeautifulSoup(html, "html.parser")
        table_el = soup.find("table")
        if not table_el:
            return
        rows_el = table_el.find_all("tr")
        if not rows_el:
            return
        max_cols = max(len(row.find_all(["td", "th"])) for row in rows_el)
        if max_cols == 0:
            return
        tbl = doc.add_table(rows=len(rows_el), cols=max_cols)
        tbl.style = "Table Grid"
        for ri, row_el in enumerate(rows_el):
            cells = row_el.find_all(["td", "th"])
            for ci, cell_el in enumerate(cells):
                if ci < max_cols:
                    cell = tbl.cell(ri, ci)
                    cell.text = cell_el.get_text(strip=True)
                    for p in cell.paragraphs:
                        for run in p.runs:
                            run.font.name = self.FONT_NAME
                            run.font.size = Pt(self.FONT_SIZE_TABLE)
                            if cell_el.name == "th":
                                run.bold = True

    def _add_figure(self, doc, fig: Dict[str, Any]):
        b64 = fig.get("base64", "")
        path = fig.get("path", "")
        try:
            if path and os.path.exists(path):
                doc.add_picture(path, width=Inches(5.5))
            elif b64:
                img_bytes = base64.b64decode(b64)
                doc.add_picture(BytesIO(img_bytes), width=Inches(5.5))
            page = fig.get("page", "?")
            idx = fig.get("index", "?")
            p = doc.add_paragraph(f"Figure {idx} — Page {page}")
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(100, 116, 139)
        except Exception as exc:
            logger.warning(f"Failed to embed figure: {exc}")

    # ── Convenience ───────────────────────────────────────────────────────────
    def create_all(self, text: str,
                   tables_html: Optional[List[str]] = None,
                   figures: Optional[List[Dict[str, Any]]] = None,
                   metadata: str = "") -> Dict[str, Optional[str]]:
        return {
            "txt": self.create_txt(text, metadata),
            "html": self.create_html(text, tables_html, figures, metadata),
            "docx": self.create_docx(text, tables_html, figures, metadata),
        }
