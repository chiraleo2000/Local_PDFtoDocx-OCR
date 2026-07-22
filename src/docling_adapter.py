"""Map DoclingDocument items → LocalOCR ContentBlock list.

Normalises Docling provenance bboxes to top-left page-pixel space matching
the absolute DOCX exporter, re-OCRs text with OCREngine when page images
are available, and builds table_html from TableFormer structure.
"""
from __future__ import annotations

import base64
import html as html_module
import logging
import os
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _strip_html_tags(text: str, repl: str = " ") -> str:
    """Replace HTML tags with *repl* (used for plain-text / garbled checks)."""
    return _HTML_TAG_RE.sub(repl, text or "")


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


def _grid_to_table_html(grid) -> Tuple[str, str]:
    """Build (html, plain_text) from a Docling table grid."""
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
    return "<table>" + "".join(rows_html) + "</table>", "\n".join(plain_rows)


def _export_table_html(table_item) -> Tuple[str, str, List[float]]:  # NOSONAR
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
                html, text = _grid_to_table_html(grid)
            num_cols = getattr(data, "num_cols", None)
            if num_cols and int(num_cols) > 0:
                col_widths = [1.0 / int(num_cols)] * int(num_cols)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to export Docling table structure")
    if not text and html:
        text = re.sub(r"\t+", "\t", _strip_html_tags(html, "\t")).strip()
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


def _wants_thai(languages: str) -> bool:
    lang = (languages or "").lower().replace(",", "+")
    parts = [p.strip() for p in re.split(r"[+\s]+", lang) if p.strip()]
    return "tha" in lang or "th" in parts


def _thai_reocr_enabled(languages: str) -> bool:
    """Thai jobs always re-OCR with Thai-TrOCR unless DOCLING_REOCR=force-off.

    RapidOCR/EasyOCR produce Latin gibberish on Thai scans — never trust
    them as final text when Thai is requested.
    """
    raw = (os.getenv("DOCLING_REOCR", "1") or "1").strip().lower()
    if raw in ("force-off", "force_off", "never"):
        return False
    if raw in ("0", "false", "no", "off"):
        # Still force Thai TrOCR when Thai is in the language set
        if _wants_thai(languages):
            logger.info(
                "DOCLING_REOCR=0 ignored — Thai requested; "
                "forcing Thai-TrOCR re-OCR")
            return True
        return False
    return True


# RapidOCR-on-Thai residue tokens seen in broken runs
_RAPID_GARBAGE_TOKENS = (
    "COMMSSUBLMACLUNGMUNEUSLUOBLUMLABEMUI",
    "ENUCSH",
    "BESUSSU",
    "CUMMLEUBLMUML",
    "OBLUMLABEMUI",
    "MACLUNGMUNEUS",
)

# Long ALL-CAPS consonant soup with almost no vowels/spaces (RapidOCR junk)
_RAPID_CAPS_SOUP = re.compile(
    r"(?:[BCDFGHJKLMNPQRSTVWXYZ]{6,}){2,}|"
    r"\b[BCDFGHJKLMNPQRSTVWXYZ]{10,}\b"
)


def _thai_char_count(text: str) -> int:
    return len(re.findall(r"[\u0E00-\u0E7F]", text or ""))


def _thai_density(text: str) -> float:
    """Fraction of letters that are Thai (0..1)."""
    if not text:
        return 0.0
    thai = _thai_char_count(text)
    letters = sum(ch.isalpha() for ch in text)
    if letters <= 0:
        return 1.0 if thai > 0 else 0.0
    return thai / letters


def _has_usable_thai(text: str, min_chars: int = 3,
                     min_density: float = 0.15) -> bool:
    """True when text has enough real Thai — not a single stray glyph."""
    if _thai_char_count(text) < min_chars:
        return False
    return _thai_density(text) >= min_density


def _looks_like_healthy_latin(text: str) -> bool:
    """Space-separated English / product names — keep on tha+eng jobs."""
    if not text or not text.strip():
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 2:
        return False
    spaces = text.count(" ") + text.count("\n") + text.count("\t")
    vowels = sum(ch.lower() in "aeiou" for ch in text)
    upper = sum(ch.isupper() for ch in text if ch.isalpha())
    # Real English has vowels and word breaks; RapidOCR soup does not
    if vowels >= max(1, letters // 8) and (spaces >= 1 or letters <= 24):
        if upper / max(letters, 1) < 0.85 or spaces >= 2:
            return True
    # Short labels: PC, MS SQL, VMware, email-ish tokens
    if letters <= 40 and re.search(
            r"[a-z]", text) and not _RAPID_CAPS_SOUP.search(text.upper()):
        return True
    return False


def _looks_garbled_for_thai(text: str) -> bool:  # NOSONAR
    """True ONLY for RapidOCR-on-Thai Latin gibberish — not real English.

    Previous versions treated any Latin (≥2 letters) as garbage on Thai jobs,
    which deleted most page content when TrOCR missed a crop.
    """
    if not text or not text.strip():
        return False  # empty is "no text", not garbled — caller decides
    if _has_usable_thai(text):
        return False
    compact = re.sub(r"\s+", "", text).upper()
    for tok in _RAPID_GARBAGE_TOKENS:
        if tok in compact:
            return True
    if _looks_like_healthy_latin(text):
        return False
    letters = sum(ch.isalpha() for ch in text)
    if letters < 6:
        return False
    # Consonant-heavy ALLCAPS soup without Thai
    if _RAPID_CAPS_SOUP.search(text.upper()):
        vowels = sum(ch.lower() in "aeiou" for ch in text)
        spaces = text.count(" ")
        if vowels <= letters * 0.12 and spaces <= 1:
            return True
    # Digit/punct residue from stripped Thai ToUnicode layers
    if letters >= 8 and _thai_char_count(text) == 0:
        digits = sum(ch.isdigit() for ch in text)
        if digits / max(len(text.strip()), 1) > 0.45 and letters < 12:
            return True
    return False


def _refuse_latin_fallback(text: str, languages: str,
                           context: str = "text") -> str:
    """Drop RapidOCR Latin gibberish when Thai was requested; keep English."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""
    if _wants_thai(languages) and _looks_garbled_for_thai(cleaned):
        logger.warning(
            "Dropping RapidOCR gibberish %s (not healthy Thai/English)",
            context)
        return ""
    return cleaned


def _shift_seg_bbox(bb, x0: float, y0: float):
    """Offset a segment bbox (quad or rect) by crop origin."""
    if isinstance(bb[0], (list, tuple)):
        return [[p[0] + x0, p[1] + y0] for p in bb]
    if len(bb) >= 4:
        return [bb[0] + x0, bb[1] + y0, bb[2] + x0, bb[3] + y0]
    return None


def _ocr_crop_lines(ocr, crop, x0: float, y0: float, languages: str, pl):
    """Run OCR cascade on a crop; return (text, lines)."""
    result = ocr.ocr_image(crop, languages=languages) or {}
    if not (result.get("text") or result.get("lines")):
        result = ocr.ocr_full_page(crop, languages=languages) or {}
    shifted = []
    for seg in result.get("lines") or []:
        s = dict(seg)
        bb = s.get("bbox")
        if not bb:
            continue
        shifted_bb = _shift_seg_bbox(bb, x0, y0)
        if shifted_bb is None:
            continue
        s["bbox"] = shifted_bb
        shifted.append(s)
    lines = pl._segments_to_lines(shifted, languages)
    if lines:
        return "\n".join(ln["text"] for ln in lines), lines
    if result.get("text"):
        return pl.clean_text(result["text"], languages), []
    return "", []


def _filter_garbled_lines(text: str, lines: List[Dict[str, Any]],
                          languages: str, bbox: List[float]):
    """Drop garbled Latin lines; rebuild text/lines for Thai jobs."""
    text = _refuse_latin_fallback(text, languages, context="text block")
    if text and lines:
        keep = []
        for ln in lines:
            t = _refuse_latin_fallback(
                ln.get("text", ""), languages, context="text line")
            if t:
                keep.append({**ln, "text": t})
        if keep:
            return "\n".join(ln["text"] for ln in keep), keep
        return text, ([{"text": text, "bbox": list(bbox)}] if text else [])
    if text:
        return text, [{"text": text, "bbox": list(bbox)}]
    return "", []


def _ocr_text_block(ocr, page_img: Optional[np.ndarray], bbox: List[float],
                    page_num: int, pw: float, ph: float,
                    languages: str, fallback_text: str = ""):  # NOSONAR
    """OCR a text region crop → ContentBlock (auto cascade: TrOCR→Paddle)."""
    pl = _pipeline_helpers()
    # Never seed Thai jobs with RapidOCR gibberish — keep healthy English seeds
    seed = fallback_text or ""
    if _wants_thai(languages) and _looks_garbled_for_thai(seed):
        seed = ""
    text = pl.clean_text(seed, languages)
    lines: List[Dict[str, Any]] = []
    if ocr is not None and page_img is not None and bbox:
        h, w = page_img.shape[:2]
        pad = max(2, int(min(h, w) * 0.003))
        x0 = max(0, int(bbox[0]) - pad)
        y0 = max(0, int(bbox[1]) - pad)
        x1 = min(w, int(bbox[2]) + pad)
        y1 = min(h, int(bbox[3]) + pad)
        if x1 > x0 and y1 > y0:
            try:
                text, lines = _ocr_crop_lines(
                    ocr, page_img[y0:y1, x0:x1], x0, y0, languages, pl)
            except Exception:  # noqa: BLE001
                logger.exception("Region OCR failed")
    text, lines = _filter_garbled_lines(text, lines, languages, bbox or [])
    return pl.ContentBlock(
        block_type="text", page=page_num,
        y_top=float(bbox[1]) if bbox else 0.0,
        x_left=float(bbox[0]) if bbox else 0.0,
        text=text,
        bbox=list(bbox) if bbox else [],
        page_width=pw, page_height=ph,
        lines=lines,
    )


def _table_content_block(pl, page_num, bbox, pw, ph, languages, text, html):
    """Build a table ContentBlock with cleaned text."""
    return pl.ContentBlock(
        block_type="table", page=page_num,
        y_top=bbox[1], x_left=bbox[0],
        text=pl.clean_text(text, languages), table_html=html,
        bbox=list(bbox), page_width=pw, page_height=ph,
    )


def _reocr_table_via_extractor(ocr, crop, languages: str):
    """Try TableExtractor re-OCR; raise if result is still garbled."""
    from .layout_detector import TableExtractor
    extractor = TableExtractor()
    extractor.set_ocr_engine(ocr)
    dets = [{"bbox": [0, 0, crop.shape[1], crop.shape[0]],
             "confidence": 1.0, "class": "table"}]
    tables = extractor.extract_tables(crop, dets, languages=languages)
    if not tables:
        return None, None
    t = tables[0]
    cand_html = t.get("html") or ""
    cand_text = t.get("text") or ""
    cand_plain = _strip_html_tags(cand_html) if cand_html else cand_text
    if cand_plain and not _looks_garbled_for_thai(cand_plain):
        return cand_html, cand_text or cand_plain
    raise RuntimeError("garbled table OCR")


def _reocr_table_via_cascade(ocr, crop, languages: str, pl):
    """Fallback: region OCR cascade → single-column HTML table."""
    result = ocr.ocr_image(crop, languages=languages) or {}
    text = _refuse_latin_fallback(
        pl.clean_text(result.get("text") or "", languages),
        languages, context="table")
    if not text:
        return "", ""
    rows = [r for r in text.split("\n") if r.strip()]
    html = "<table>" + "".join(
        "<tr><td>" + html_module.escape(r) + "</td></tr>"
        for r in rows) + "</table>"
    return html, text


def _reocr_table_block(ocr, page_img, bbox, page_num, pw, ph, languages,
                       fallback_html: str = "", fallback_text: str = ""):  # NOSONAR
    """Re-OCR a table crop with Thai-capable engines; keep Docling HTML if Thai OK."""
    pl = _pipeline_helpers()
    html = fallback_html or ""
    text = fallback_text or ""
    if _wants_thai(languages):
        # Strip Latin-only RapidOCR seeds before any keep/re-OCR decision
        if html and _looks_garbled_for_thai(_strip_html_tags(html)):
            html = ""
        if text and _looks_garbled_for_thai(text):
            text = ""
    if (ocr is None or page_img is None or not bbox
            or not _wants_thai(languages)):
        return _table_content_block(
            pl, page_num, bbox, pw, ph, languages, text, html)
    # Keep Docling HTML only when it has meaningful Thai density
    plain_html = _strip_html_tags(html) if html else ""
    if html and _has_usable_thai(plain_html, min_chars=6, min_density=0.25):
        return _table_content_block(
            pl, page_num, bbox, pw, ph, languages, text or plain_html, html)
    h, w = page_img.shape[:2]
    x0 = max(0, int(bbox[0])); y0 = max(0, int(bbox[1]))
    x1 = min(w, int(bbox[2])); y1 = min(h, int(bbox[3]))
    if x1 <= x0 or y1 <= y0:
        return _table_content_block(
            pl, page_num, bbox, pw, ph, languages, text, html)
    crop = page_img[y0:y1, x0:x1]
    speed = os.getenv("SPEED_MODE", "0").strip().lower() in (
        "1", "true", "yes", "on")
    try:
        # SPEED_MODE: one crop OCR (cell-by-cell TableExtractor is too slow)
        if speed:
            html, text = _reocr_table_via_cascade(ocr, crop, languages, pl)
            logger.info(
                "Table SPEED re-OCR (cascade) on page %d", page_num + 1)
        else:
            cand_html, cand_text = _reocr_table_via_extractor(
                ocr, crop, languages)
            if cand_html is not None:
                html = cand_html or html
                text = cand_text or text
                logger.info(
                    "Table re-OCR with Thai engines on page %d", page_num + 1)
    except Exception:  # noqa: BLE001
        logger.exception("Table Thai re-OCR failed — trying region OCR cascade")
        try:
            html, text = _reocr_table_via_cascade(ocr, crop, languages, pl)
        except Exception:  # noqa: BLE001
            html, text = "", ""
    # Final gate — never export RapidOCR gibberish table HTML on Thai jobs
    if html and _looks_garbled_for_thai(_strip_html_tags(html)):
        html = ""
    text = _refuse_latin_fallback(text, languages, context="table text")
    return _table_content_block(
        pl, page_num, bbox, pw, ph, languages, text, html)


def _struct_has_content(block) -> bool:
    """Only suppress overlapping text when structure actually carries content."""
    if block.block_type == "table":
        if (block.table_html or "").strip() or (block.text or "").strip():
            return True
        return False
    if block.block_type == "figure":
        fig = block.figure or {}
        return bool(fig.get("base64"))
    return False


def _bbox_cover_frac(inner, outer) -> float:
    """Fraction of *inner* area covered by intersection with *outer*."""
    ax0, ay0, ax1, ay1 = inner
    bx0, by0, bx1, by1 = outer
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max((ax1 - ax0) * (ay1 - ay0), 1e-6)
    return inter / area


def _suppress_text_in_structure(blocks: List) -> List:  # NOSONAR
    """Drop text that mostly sits inside a filled TABLE only.

    Do NOT suppress text inside figures — org charts / diagrams keep their
    OCR'd labels; the figure image alone is not enough for searchable DOCX.
    """
    struct = [
        b for b in blocks
        if b.block_type == "table" and b.bbox and _struct_has_content(b)
    ]
    if not struct:
        return blocks
    out = []
    for b in blocks:
        if b.block_type not in ("text", "caption") or not b.bbox:
            out.append(b)
            continue
        if not any(_bbox_cover_frac(b.bbox, s.bbox) > 0.72 for s in struct):
            out.append(b)
    return out


def _empty_text_as_figure(page_img, bbox, page_num, pw, ph, fig_idx: int):
    """Preserve unread text regions as cropped images so content is not lost."""
    pl = _pipeline_helpers()
    fig = _figure_from_image(page_img, bbox, page_num, fig_idx)
    if not fig.get("base64"):
        return None
    return pl.ContentBlock(
        block_type="figure", page=page_num,
        y_top=float(bbox[1]), x_left=float(bbox[0]),
        figure=fig, bbox=list(bbox),
        page_width=pw, page_height=ph,
    )


def _iter_docling_items(docling_doc) -> List[Any]:
    """Collect Docling items via iterate_items or attribute fallbacks."""
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
    return items


def _item_label(item) -> str:
    lab = getattr(item, "label", None)
    if lab is None:
        return type(item).__name__.lower()
    return str(getattr(lab, "value", lab)).lower()


def _item_geometry(docling_doc, prov, page_images):
    """Return (page_idx, page_no, page_img, bbox, pw, ph) or None."""
    page_no = int(getattr(prov, "page_no", 1) or 1)
    page_idx = page_no - 1
    pw, ph = _page_size(docling_doc, page_no)
    page_img = page_images.get(page_idx)
    if page_img is not None and (pw <= 0 or ph <= 0):
        ph, pw = float(page_img.shape[0]), float(page_img.shape[1])
    bbox = _bbox_from_prov(prov, ph if ph > 0 else 1.0)
    if bbox is None:
        return None
    if page_img is not None and pw > 0 and ph > 0:
        ih, iw = page_img.shape[:2]
        if abs(iw - pw) > 1.0 or abs(ih - ph) > 1.0:
            sx, sy = iw / pw, ih / ph
            bbox = [bbox[0] * sx, bbox[1] * sy,
                    bbox[2] * sx, bbox[3] * sy]
            pw, ph = float(iw), float(ih)
    return page_idx, page_no, page_img, bbox, pw, ph


def _append_table_item(pl, item, ocr, page_img, bbox, page_idx, page_no,
                       pw, ph, languages, blocks, fig_idx: int) -> int:
    """Append table (or figure fallback); return updated fig_idx."""
    html, text, col_w = _export_table_html(item)
    if not html and not text and page_img is None:
        logger.warning(
            "Docling table on page %d has empty structure", page_no)
        return fig_idx
    speed = os.getenv("SPEED_MODE", "0").strip().lower() in (
        "1", "true", "yes", "on")
    # SPEED_MODE: keep Docling table geometry; band OCR supplies Thai text
    if (_thai_reocr_enabled(languages) and page_img is not None
            and not speed):
        block = _reocr_table_block(
            ocr, page_img, bbox, page_idx, pw, ph, languages,
            fallback_html=html, fallback_text=text)
        if col_w:
            block.table_meta = {"col_widths": col_w}
        if (not (block.table_html or "").strip()
                and not (block.text or "").strip()):
            area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            if area / max(pw * ph, 1.0) >= 0.05:
                fig_idx += 1
                fig_block = _empty_text_as_figure(
                    page_img, bbox, page_idx, pw, ph, fig_idx)
                if fig_block is not None:
                    blocks.append(fig_block)
            return fig_idx
        blocks.append(block)
        return fig_idx
    if speed and _wants_thai(languages):
        # Drop RapidOCR gibberish; band OCR carries searchable Thai text
        if html and _looks_garbled_for_thai(_strip_html_tags(html)):
            html, text = "", ""
        elif text and _looks_garbled_for_thai(text):
            text = ""
    if not html and not text:
        # Keep empty table shell so layout/figures stay aligned
        if speed and page_img is not None and bbox:
            blocks.append(pl.ContentBlock(
                block_type="table", page=page_idx,
                y_top=bbox[1], x_left=bbox[0],
                text="", table_html="<table></table>",
                bbox=bbox, page_width=pw, page_height=ph,
                table_meta={"col_widths": col_w} if col_w else {},
            ))
            return fig_idx
        return fig_idx
    blocks.append(pl.ContentBlock(
        block_type="table", page=page_idx,
        y_top=bbox[1], x_left=bbox[0],
        text=text, table_html=html,
        bbox=bbox, page_width=pw, page_height=ph,
        table_meta={"col_widths": col_w},
    ))
    return fig_idx


def _figure_bytes_from_item(item, fig: Dict[str, Any]) -> Dict[str, Any]:
    """Fill fig base64 from Docling item.image when crop bytes are missing."""
    if fig.get("base64"):
        return fig
    pil_img = getattr(item, "image", None)
    if pil_img is None or not hasattr(pil_img, "pil_image"):
        return fig
    try:
        buf = BytesIO()
        pil_img.pil_image.save(buf, format="PNG")
        fig["base64"] = base64.b64encode(buf.getvalue()).decode()
        fig["width"] = pil_img.pil_image.width
        fig["height"] = pil_img.pil_image.height
    except Exception:  # noqa: BLE001
        pass
    return fig


def _append_figure_item(pl, item, page_img, bbox, page_idx, pw, ph,
                        blocks, fig_idx: int) -> int:
    """Append a figure block; return updated fig_idx."""
    fig_idx += 1
    fig = _figure_bytes_from_item(
        item, _figure_from_image(page_img, bbox, page_idx, fig_idx))
    if not fig.get("base64") and page_img is not None:
        logger.warning(
            "Skipping empty figure on page %d (no crop/image bytes)",
            page_idx + 1)
        return fig_idx
    blocks.append(pl.ContentBlock(
        block_type="figure", page=page_idx,
        y_top=bbox[1], x_left=bbox[0],
        figure=fig, bbox=bbox,
        page_width=pw, page_height=ph,
    ))
    return fig_idx


def _append_text_item(pl, item, ocr, page_img, bbox, page_idx, pw, ph,
                      languages, label: str, blocks, fig_idx: int) -> int:
    """Append text/caption (or large failed region as figure)."""
    raw = getattr(item, "text", None) or getattr(item, "orig", "") or ""
    btype = "caption" if "caption" in label else "text"
    # SPEED_MODE uses banded page TrOCR for body text — skip per-region
    # TrOCR here (keeps ~10s/page). Tables still re-OCR below.
    speed = os.getenv("SPEED_MODE", "0").strip().lower() in (
        "1", "true", "yes", "on")
    speed_bands = os.getenv("SPEED_BAND_OCR", "1").strip().lower() not in (
        "0", "false", "no", "off")
    do_reocr = (
        _thai_reocr_enabled(languages)
        and ocr is not None
        and page_img is not None
        and not (speed and speed_bands)
    )
    if do_reocr:
        block = _ocr_text_block(
            ocr, page_img, bbox, page_idx, pw, ph, languages,
            fallback_text=str(raw))
    else:
        text = pl.clean_text(str(raw), languages)
        if _wants_thai(languages) and _looks_garbled_for_thai(text):
            logger.warning(
                "Dropping RapidOCR gibberish Docling text on page %d",
                page_idx + 1)
            text = ""
        block = pl.ContentBlock(
            block_type="text", page=page_idx,
            y_top=bbox[1], x_left=bbox[0],
            text=text, bbox=list(bbox),
            page_width=pw, page_height=ph,
            lines=[{"text": text, "bbox": list(bbox)}] if text else [],
        )
    block.block_type = btype
    if block.text.strip() or block.lines:
        blocks.append(block)
        return fig_idx
    if page_img is None or not bbox:
        return fig_idx
    area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    if area / max(pw * ph, 1.0) >= 0.04:
        fig_idx += 1
        fig_block = _empty_text_as_figure(
            page_img, bbox, page_idx, pw, ph, fig_idx)
        if fig_block is not None:
            blocks.append(fig_block)
    return fig_idx


_TEXT_LABEL_KEYS = (
    "text", "section", "title", "caption", "paragraph",
    "list", "formula", "code", "checkbox",
)


def docling_to_blocks(  # NOSONAR
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

    for item in _iter_docling_items(docling_doc):
        label = _item_label(item)
        provs = getattr(item, "prov", None) or []
        if not provs:
            continue
        geom = _item_geometry(docling_doc, provs[0], page_images)
        if geom is None:
            continue
        page_idx, page_no, page_img, bbox, pw, ph = geom

        if "table" in label:
            fig_idx = _append_table_item(
                pl, item, ocr, page_img, bbox, page_idx, page_no,
                pw, ph, languages, blocks, fig_idx)
        elif "picture" in label or "figure" in label or "image" in label:
            fig_idx = _append_figure_item(
                pl, item, page_img, bbox, page_idx, pw, ph, blocks, fig_idx)
        elif any(k in label for k in _TEXT_LABEL_KEYS):
            fig_idx = _append_text_item(
                pl, item, ocr, page_img, bbox, page_idx, pw, ph,
                languages, label, blocks, fig_idx)

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
