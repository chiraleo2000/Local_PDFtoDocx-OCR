"""Map DoclingDocument items → LocalOCR ContentBlock list.

Normalises Docling provenance bboxes to top-left page-pixel space matching
the absolute DOCX exporter, re-OCRs text with OCREngine when page images
are available, and builds table_html from TableFormer structure.
"""
from __future__ import annotations

import base64
import html as html_module
import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _pipeline_helpers():
    """Lazy import to avoid circular dependency with pipeline.py."""
    from . import pipeline as pl
    return pl


def _bbox_from_prov(prov, page_h: float) -> Optional[List[float]]:
    """Convert Docling provenance bbox → [x0,y0,x1,y1] top-left origin."""
    if prov is None:
        return None
    try:
        bb = prov.bbox
        if hasattr(bb, "to_top_left_origin"):
            bb = bb.to_top_left_origin(page_h)
        l = float(bb.l)
        t = float(bb.t)
        r = float(bb.r)
        b = float(bb.b)
        if b < t:
            t, b = page_h - t, page_h - b
        return [min(l, r), min(t, b), max(l, r), max(t, b)]
    except Exception:  # noqa: BLE001
        return None


def _page_size(docling_doc, page_no: int) -> Tuple[float, float]:
    """Return (width, height) for 1-based page_no."""
    try:
        pages = getattr(docling_doc, "pages", None) or {}
        page = pages.get(page_no) or pages.get(str(page_no))
        if page is not None and getattr(page, "size", None):
            return float(page.size.width), float(page.size.height)
    except Exception:  # noqa: BLE001
        pass
    return 0.0, 0.0


def _export_table_html(table_item) -> Tuple[str, str, List[float]]:
    """Return (html, plain_text, col_width_fractions) from a Docling table."""
    html = ""
    text = ""
    col_widths: List[float] = []
    try:
        if hasattr(table_item, "export_to_html"):
            html = table_item.export_to_html() or ""
    except Exception:  # noqa: BLE001
        html = ""
    try:
        data = getattr(table_item, "data", None)
        if data is not None:
            grid = getattr(data, "grid", None) or getattr(data, "table_cells", None)
            if grid and not html:
                rows_html = []
                plain_rows = []
                for row in grid:
                    cells = []
                    plain = []
                    for cell in row:
                        t = getattr(cell, "text", None)
                        if t is None and isinstance(cell, dict):
                            t = cell.get("text", "")
                        t = (t or "").strip()
                        cells.append(f"<td>{html_module.escape(t)}</td>")
                        plain.append(t)
                    rows_html.append("<tr>" + "".join(cells) + "</tr>")
                    plain_rows.append("\t".join(plain))
                html = "<table>" + "".join(rows_html) + "</table>"
                text = "\n".join(plain_rows)
            num_cols = getattr(data, "num_cols", None)
            if num_cols and int(num_cols) > 0:
                col_widths = [1.0 / int(num_cols)] * int(num_cols)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to export Docling table structure")
    if not text and html:
        text = re.sub(r"<[^>]+>", "\t", html)
        text = re.sub(r"\t+", "\t", text).strip()
    return html, text, col_widths


def _figure_from_image(page_img: Optional[np.ndarray], bbox: List[float],
                       page_num: int, index: int) -> Dict[str, Any]:
    """Crop page image to bbox and encode as figure payload."""
    if page_img is None or not bbox:
        return {"page": page_num, "index": index, "bbox": bbox}
    h, w = page_img.shape[:2]
    x0 = max(0, int(bbox[0]))
    y0 = max(0, int(bbox[1]))
    x1 = min(w, int(bbox[2]))
    y1 = min(h, int(bbox[3]))
    if x1 <= x0 or y1 <= y0:
        return {"page": page_num, "index": index, "bbox": bbox}
    crop = page_img[y0:y1, x0:x1]
    try:
        from PIL import Image
        if crop.ndim == 2:
            pil = Image.fromarray(crop)
        else:
            pil = Image.fromarray(crop[:, :, ::-1])
        buf = BytesIO()
        pil.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {
            "page": page_num,
            "index": index,
            "base64": b64,
            "width": int(pil.width),
            "height": int(pil.height),
            "bbox": bbox,
            "source": "docling",
        }
    except Exception:  # noqa: BLE001
        logger.exception("Figure encode failed")
        return {"page": page_num, "index": index, "bbox": bbox}


def _ocr_text_block(ocr, page_img: Optional[np.ndarray], bbox: List[float],
                    page_num: int, pw: float, ph: float,
                    languages: str, fallback_text: str = ""):
    """OCR a text region crop → ContentBlock with line bboxes."""
    pl = _pipeline_helpers()
    text = pl.clean_text(fallback_text or "", languages)
    lines: List[Dict[str, Any]] = []
    if ocr is not None and page_img is not None and bbox:
        h, w = page_img.shape[:2]
        pad = max(2, int(min(h, w) * 0.003))
        x0 = max(0, int(bbox[0]) - pad)
        y0 = max(0, int(bbox[1]) - pad)
        x1 = min(w, int(bbox[2]) + pad)
        y1 = min(h, int(bbox[3]) + pad)
        if x1 > x0 and y1 > y0:
            crop = page_img[y0:y1, x0:x1]
            try:
                result = ocr.ocr_full_page(crop, languages=languages)
                segs = result.get("lines") or []
                shifted = []
                for seg in segs:
                    s = dict(seg)
                    bb = s.get("bbox")
                    if not bb:
                        continue
                    if isinstance(bb[0], (list, tuple)):
                        s["bbox"] = [[p[0] + x0, p[1] + y0] for p in bb]
                    elif len(bb) >= 4:
                        s["bbox"] = [bb[0] + x0, bb[1] + y0,
                                     bb[2] + x0, bb[3] + y0]
                    shifted.append(s)
                lines = pl._segments_to_lines(shifted, languages)
                if lines:
                    text = "\n".join(ln["text"] for ln in lines)
                elif result.get("text"):
                    text = pl.clean_text(result["text"], languages)
            except Exception:  # noqa: BLE001
                logger.exception("Region OCR failed; using Docling text")
    return pl.ContentBlock(
        block_type="text", page=page_num,
        y_top=float(bbox[1]) if bbox else 0.0,
        x_left=float(bbox[0]) if bbox else 0.0,
        text=text,
        bbox=list(bbox) if bbox else [],
        page_width=pw, page_height=ph,
        lines=lines,
    )


def _suppress_text_in_structure(blocks: List) -> List:
    """Drop text/caption blocks that mostly sit inside a table or figure."""
    struct = [b for b in blocks if b.block_type in ("table", "figure") and b.bbox]
    if not struct:
        return blocks
    out = []
    for b in blocks:
        if b.block_type not in ("text", "caption") or not b.bbox:
            out.append(b)
            continue
        covered = False
        for s in struct:
            ax0, ay0, ax1, ay1 = b.bbox
            bx0, by0, bx1, by1 = s.bbox
            ix0, iy0 = max(ax0, bx0), max(ay0, by0)
            ix1, iy1 = min(ax1, bx1), min(ay1, by1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            area = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
            if inter / area > 0.55:
                covered = True
                break
        if not covered:
            out.append(b)
    return out


def docling_to_blocks(
        docling_doc,
        ocr=None,
        page_images: Optional[Dict[int, np.ndarray]] = None,
        languages: str = "tha+eng",
) -> List:
    """Convert a DoclingDocument into ContentBlocks for DocumentExporter."""
    pl = _pipeline_helpers()
    page_images = page_images or {}
    blocks: List = []
    fig_idx = 0

    iterate = getattr(docling_doc, "iterate_items", None)
    items: List[Any] = []
    if callable(iterate):
        try:
            for item, _level in iterate():
                items.append(item)
        except Exception:  # noqa: BLE001
            items = []
    if not items:
        for attr in ("texts", "tables", "pictures"):
            seq = getattr(docling_doc, attr, None) or []
            items.extend(list(seq))

    def _label(item) -> str:
        lab = getattr(item, "label", None)
        if lab is None:
            return type(item).__name__.lower()
        return str(getattr(lab, "value", lab)).lower()

    for item in items:
        label = _label(item)
        provs = getattr(item, "prov", None) or []
        if not provs:
            continue
        prov = provs[0]
        page_no = int(getattr(prov, "page_no", 1) or 1)
        page_idx = page_no - 1
        pw, ph = _page_size(docling_doc, page_no)
        page_img = page_images.get(page_idx)
        if page_img is not None and (pw <= 0 or ph <= 0):
            ph, pw = float(page_img.shape[0]), float(page_img.shape[1])
        bbox = _bbox_from_prov(prov, ph if ph > 0 else 1.0)
        if bbox is None:
            continue
        if page_img is not None and pw > 0 and ph > 0:
            ih, iw = page_img.shape[:2]
            if abs(iw - pw) > 1.0 or abs(ih - ph) > 1.0:
                sx, sy = iw / pw, ih / ph
                bbox = [bbox[0] * sx, bbox[1] * sy,
                        bbox[2] * sx, bbox[3] * sy]
                pw, ph = float(iw), float(ih)

        if "table" in label:
            html, text, col_w = _export_table_html(item)
            if not html and not text:
                logger.warning(
                    "Docling table on page %d has empty structure", page_no)
                continue
            blocks.append(pl.ContentBlock(
                block_type="table", page=page_idx,
                y_top=bbox[1], x_left=bbox[0],
                text=text, table_html=html,
                bbox=bbox, page_width=pw, page_height=ph,
                table_meta={"col_widths": col_w},
            ))
        elif "picture" in label or "figure" in label or "image" in label:
            fig_idx += 1
            fig = _figure_from_image(page_img, bbox, page_idx, fig_idx)
            if not fig.get("base64"):
                pil_img = getattr(item, "image", None)
                if pil_img is not None and hasattr(pil_img, "pil_image"):
                    try:
                        buf = BytesIO()
                        pil_img.pil_image.save(buf, format="PNG")
                        fig["base64"] = base64.b64encode(
                            buf.getvalue()).decode()
                        fig["width"] = pil_img.pil_image.width
                        fig["height"] = pil_img.pil_image.height
                    except Exception:  # noqa: BLE001
                        pass
            blocks.append(pl.ContentBlock(
                block_type="figure", page=page_idx,
                y_top=bbox[1], x_left=bbox[0],
                figure=fig, bbox=bbox,
                page_width=pw, page_height=ph,
            ))
        elif any(k in label for k in (
                "text", "section", "title", "caption", "paragraph",
                "list", "formula", "code", "checkbox")):
            raw = getattr(item, "text", None) or getattr(item, "orig", "") or ""
            btype = "caption" if "caption" in label else "text"
            block = _ocr_text_block(
                ocr, page_img, bbox, page_idx, pw, ph, languages,
                fallback_text=str(raw))
            block.block_type = btype
            if block.text.strip() or block.lines:
                blocks.append(block)

    blocks = _suppress_text_in_structure(blocks)
    blocks.sort(key=lambda b: (b.page, b.y_top, b.x_left))
    return blocks


def detections_from_docling(docling_doc, page_idx: int = 0,
                            page_img: Optional[np.ndarray] = None,
                            ) -> Dict[str, List[Dict[str, Any]]]:
    """Build Review-UI detection dict from a DoclingDocument page."""
    out: Dict[str, List[Dict[str, Any]]] = {
        "figures": [], "tables": [], "text_regions": [],
        "formulas": [], "captions": [], "other": [],
    }
    images = {page_idx: page_img} if page_img is not None else {}
    blocks = docling_to_blocks(docling_doc, ocr=None, page_images=images,
                               languages="eng")
    for b in blocks:
        if b.page != page_idx or not b.bbox:
            continue
        det = {"bbox": b.bbox, "confidence": 1.0, "class": b.block_type,
               "source": "docling"}
        if b.block_type == "table":
            det["class"] = "table"
            out["tables"].append(det)
        elif b.block_type == "figure":
            det["class"] = "figure"
            out["figures"].append(det)
        elif b.block_type == "caption":
            det["class"] = "caption"
            out["captions"].append(det)
        else:
            det["class"] = "text"
            out["text_regions"].append(det)
    return out
