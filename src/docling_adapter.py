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
_HTML_TABLE_OPEN = "<table>"
_HTML_TABLE_CLOSE = "</table>"
# Linear stub pattern (no nested optional \s* groups — Sonar S8786)
_SECTION_STUB_RE = re.compile(r"^\d{1,2}(?:[.)]\d{0,2})?[.)]?\s*$")
# Numbered list markers ("3)" / "11)") — may have garbled/missing body
_LIST_MARKER_RE = re.compile(r"^\d{1,2}\)\s*")


def _strip_html_tags(text: str, repl: str = " ") -> str:
    """Replace HTML tags with *repl* (used for plain-text / garbled checks)."""
    return _HTML_TAG_RE.sub(repl, text or "")


def _pipeline_helpers():
    """Lazy import to avoid circular dependency with pipeline.py."""
    from . import pipeline as pl
    return pl


def _bbox_to_topleft(bb, page_h: float) -> Optional[List[float]]:
    """Convert a Docling BoundingBox → [x0,y0,x1,y1] top-left page coords."""
    if bb is None:
        return None
    try:
        if hasattr(bb, "to_top_left_origin"):
            bb = bb.to_top_left_origin(page_h)
        l = float(bb.l)
        t = float(bb.t)
        r = float(bb.r)
        b = float(bb.b)
        origin = str(getattr(getattr(bb, "coord_origin", None), "value",
                             getattr(bb, "coord_origin", "")) or "").upper()
        if "BOTTOM" in origin and b < t:
            t, b = page_h - t, page_h - b
        elif b < t:
            t, b = b, t
        return [min(l, r), min(t, b), max(l, r), max(t, b)]
    except Exception:  # noqa: BLE001
        return None


def _bbox_from_prov(prov, page_h: float) -> Optional[List[float]]:
    """Convert Docling provenance bbox → [x0,y0,x1,y1] top-left origin."""
    if prov is None:
        return None
    try:
        return _bbox_to_topleft(prov.bbox, page_h)
    except Exception:  # noqa: BLE001
        return None


def _page_scale(docling_doc, page_no: int, page_img) -> Tuple[float, float, float]:
    """Return (sx, sy, page_h_docling) mapping Docling page → page_img pixels."""
    d_pw, d_ph = _page_size(docling_doc, page_no)
    if page_img is None or d_pw <= 0 or d_ph <= 0:
        return 1.0, 1.0, d_ph if d_ph > 0 else 1.0
    ih, iw = page_img.shape[:2]
    return iw / d_pw, ih / d_ph, d_ph


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
    return (_HTML_TABLE_OPEN + "".join(rows_html) + _HTML_TABLE_CLOSE,
            "\n".join(plain_rows))


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
    # TrOCR person-name / stutter hallucinations on Thai docs
    if re.search(r"นางสาวศาสตราจารย์|พรรณตำบล|โสภณเพ็ญ|วรรทวรร", text):
        return True
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
    # Short RapidOCR crumbs ("SLUBLS", "UBCIUBLS") — ALLCAPS, no vowels/Thai
    if (_thai_char_count(text) == 0 and letters >= 4
            and not _looks_like_healthy_latin(text)):
        upper = sum(ch.isupper() for ch in text if ch.isalpha())
        vowels = sum(ch.lower() in "aeiou" for ch in text)
        if upper >= letters * 0.75 and vowels <= max(1, letters // 6):
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


def _is_section_number_stub(text: str) -> bool:
    """True for truncated numbered headings like '2.' / '2.1' with no body."""
    t = (text or "").strip()
    if not t or len(t) > 14:
        return False
    return bool(_SECTION_STUB_RE.match(t))


def _is_narrow_or_list_crop(bbox: List[float], page_w: float,
                            text: str = "", seed: str = "") -> bool:
    """True when crop looks like a list/heading stub that needs widening."""
    if _is_section_number_stub(text) or _is_section_number_stub(seed):
        return True
    for candidate in (text, seed):
        t = (candidate or "").strip()
        if t and _LIST_MARKER_RE.match(t):
            body = _LIST_MARKER_RE.sub("", t).strip()
            if not body or _thai_char_count(body) < 12:
                return True
    return False


def _expand_stub_crop(
        bbox: List[float], h: int, w: int, pad: int, line_h: int,
        wide: float = 0.62,
) -> Tuple[int, int, int, int]:
    """Widen a heading-stub crop so Thai-TrOCR sees the full line."""
    x0 = max(0, int(bbox[0]) - pad)
    y0 = max(0, int(bbox[1]) - pad)
    x1 = min(w, int(bbox[2]) + pad)
    y1 = min(h, int(bbox[3]) + pad)
    mid_y = (y0 + y1) / 2.0
    y0 = max(0, int(mid_y - line_h * 0.85))
    y1 = min(h, int(mid_y + line_h * 0.85))
    x1 = min(w, max(x1, int(bbox[0] + wide * w)))
    return x0, y0, x1, y1


def _thai_density(text: str) -> float:
    t = (text or "").strip()
    if not t:
        return 0.0
    return _thai_char_count(t) / max(len(t), 1)


def _run_stub_aware_ocr(  # NOSONAR
        ocr, page_img, bbox, languages, pl, text, seed):
    """OCR a region; widen crop for list/heading stubs and weak Thai results."""
    h, w = page_img.shape[:2]
    pad = max(2, int(min(h, w) * 0.003))
    line_h = max(int(bbox[3] - bbox[1]), int(0.012 * h), 16)
    force_wide = _is_narrow_or_list_crop(bbox, float(w), text, seed)
    if force_wide:
        x0, y0, x1, y1 = _expand_stub_crop(bbox, h, w, pad, line_h, 0.88)
    else:
        x0 = max(0, int(bbox[0]) - pad)
        y0 = max(0, int(bbox[1]) - pad)
        x1 = min(w, int(bbox[2]) + pad)
        y1 = min(h, int(bbox[3]) + pad)
    lines: List[Dict[str, Any]] = []
    if x1 <= x0 or y1 <= y0:
        return text, lines
    try:
        text, lines = _ocr_crop_lines(
            ocr, page_img[y0:y1, x0:x1], x0, y0, languages, pl)
    except Exception:  # noqa: BLE001
        logger.exception("Region OCR failed")
        return text, lines
    # Second pass: still a stub, weak Thai, or short after list marker
    needs_retry = (
        _is_section_number_stub(text)
        or _thai_density(text) < 0.35
        or (_LIST_MARKER_RE.match((text or "").strip())
            and _thai_char_count(text) < 20)
    )
    if not needs_retry:
        return text, lines
    x0, y0, x1, y1 = _expand_stub_crop(bbox, h, w, pad, line_h, 0.92)
    y0 = max(0, int(bbox[1] - line_h))
    y1 = min(h, int(bbox[3] + line_h * 1.2))
    try:
        text2, lines2 = _ocr_crop_lines(
            ocr, page_img[y0:y1, x0:x1], x0, y0, languages, pl)
        if text2 and (
                len(text2) > len(text or "")
                or _thai_char_count(text2) > _thai_char_count(text or "")):
            return text2, lines2
    except Exception:  # noqa: BLE001
        pass
    return text, lines


def _ocr_text_block(ocr, page_img: Optional[np.ndarray], bbox: List[float],
                    page_num: int, pw: float, ph: float,
                    languages: str, fallback_text: str = ""):
    """OCR a text region crop → ContentBlock (auto cascade: TrOCR→Paddle)."""
    pl = _pipeline_helpers()
    seed = fallback_text or ""
    if _wants_thai(languages) and _looks_garbled_for_thai(seed):
        seed = ""
    text = pl.clean_text(seed, languages)
    lines: List[Dict[str, Any]] = []
    if ocr is not None and page_img is not None and bbox:
        text, lines = _run_stub_aware_ocr(
            ocr, page_img, bbox, languages, pl, text, seed)
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
    html = _HTML_TABLE_OPEN + "".join(
        "<tr><td>" + html_module.escape(r) + "</td></tr>"
        for r in rows) + _HTML_TABLE_CLOSE
    return html, text


def _table_engine() -> str:
    """Primary table structure engine (docling | opencv | paddleocr)."""
    return (os.getenv("TABLE_ENGINE", "docling") or "docling").strip().lower()


def _prefer_docling_tables() -> bool:
    """True when Docling TableFormer is the primary table path."""
    return _table_engine() in ("docling", "auto", "")


def _speed_table_ocr_enabled() -> bool:
    """Per-cell Docling-grid Thai OCR (default ON — RapidOCR cells are junk)."""
    return os.getenv("SPEED_TABLE_OCR", "1").strip().lower() not in (
        "0", "false", "no", "off")


def _iter_unique_table_cells(data) -> List[Any]:
    """Unique Docling TableCell objects (prefer table_cells over grid dups)."""
    cells = getattr(data, "table_cells", None)
    if cells:
        return list(cells)
    grid = getattr(data, "grid", None) or []
    seen = set()
    out = []
    for row in grid:
        for cell in row:
            key = id(cell)
            if key in seen:
                continue
            seen.add(key)
            out.append(cell)
    return out


def _clean_cell_text(text: str) -> str:
    """Collapse TrOCR stutter runs (กากากา… / แขนแขนแขน…)."""
    t = (text or "").strip()
    if not t:
        return ""
    # Multi-line cell crops: keep the densest Thai/digit line (drop junk lines)
    if "\n" in t:
        lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
        if lines:
            def _line_score(ln: str) -> tuple:
                thai = _thai_char_count(ln)
                digits = sum(ch.isdigit() for ch in ln)
                junk = len(re.findall(r"[!@#$%^&*_'=<>]", ln))
                return (thai + digits * 2 - junk * 3, len(ln))
            t = max(lines, key=_line_score)
    prev = None
    while prev != t:
        prev = t
        t = re.sub(r"([\u0E00-\u0E7F]{1,6})\1+", r"\1", t)
        t = re.sub(r"([A-Za-z0-9]{1,4})\1+", r"\1", t)
    # Drop short Thai crumb prefix before a real word
    t = re.sub(
        r"^[\u0E00-\u0E7F]{1,3}\s+(?=[\u0E00-\u0E7F]{3,})",
        "", t)
    # Digit cells often come back as "1 6" / "50 0" — collapse spaced digits
    compact = re.sub(r"\s+", "", t)
    if compact.isdigit() and 1 <= len(compact) <= 6:
        return compact
    if re.fullmatch(r"[\d\s]+", t) and len(compact) <= 6:
        return compact
    # Strip leading OCR junk digits before Thai labels ("0 ม๖ ตำแหน่ง…")
    t = re.sub(
        r"^(?:[0-9๐-๙]\s*){1,4}(?=[\u0E00-\u0E7F])",
        "", t).strip()
    return t.strip()


def _looks_like_thai_hallucination(text: str) -> bool:
    """Reject short noisy Thai crumbs common from blank-cell TrOCR."""
    t = (text or "").strip()
    if not t:
        return False
    if re.search(r"[!_]{2,}|\*{2,}|'{2,}", t):
        return True
    # EasyOCR/TrOCR blank-slot soup seen on inventory tables
    if re.search(
            r"ยอมแพ้|สังคมพิสูจน์|ทหารบก|เชื้อพระวงศ์|จอมพล|"
            r"ตาแหนง|ด้านข้างราชการ|วรรณจัก|บล้มเหลว",
            t):
        return True
    thai = _thai_char_count(t)
    if thai < 4:
        return False
    junk = len(re.findall(r"[!@#$%^&*_'=<>]", t))
    if junk >= 2 and thai < 20:
        return True
    # Mixed punctuation / underscore noise in short cells
    if junk >= 1 and thai < 12 and len(t) <= 24:
        return True
    return False


def _table_shape_from_plain(html: str, text: str = "") -> Tuple[int, int]:
    """Best-effort (nrows, ncols) from HTML or tab-plain text."""
    plain = _strip_html_tags(html, "\n") if html else (text or "")
    if html:
        rows = re.findall(r"<tr\b", html, flags=re.I)
        cols = 0
        first = re.search(r"<tr\b[^>]*>(.*?)</tr>", html, flags=re.I | re.S)
        if first:
            cols = len(re.findall(r"<t[dh]\b", first.group(1), flags=re.I))
        if rows and cols:
            return len(rows), cols
    lines = [ln for ln in (plain or "").split("\n") if ln.strip() or "\t" in ln]
    if not lines:
        return 0, 0
    ncols = max((ln.count("\t") + 1) for ln in lines)
    return len(lines), ncols


def _table_plain_quality(html: str, text: str = "") -> Dict[str, float]:
    """Score table OCR richness for Docling vs OpenCV extractor selection."""
    plain = _strip_html_tags(html) if html else (text or "")
    if _looks_like_thai_hallucination(plain):
        thai = float(_thai_char_count(plain)) * 0.25
    else:
        thai = float(_thai_char_count(plain))
    rows = [r for r in (plain or "").split("\n") if r.strip() or "\t" in r]
    cells = []
    for r in rows:
        cells.extend([(c or "").strip() for c in r.split("\t")])
    if not cells and plain:
        cells = [plain]
    n = max(len(cells), 1)
    filled = sum(
        1 for c in cells if c and not _looks_like_thai_hallucination(c))
    left = []
    for r in rows:
        parts = r.split("\t")
        if parts:
            left.append((parts[0] or "").strip())
        if len(parts) > 1:
            left.append((parts[1] or "").strip())
    left_n = max(len(left), 1)
    left_filled = sum(
        1 for c in left
        if c and _thai_char_count(c) >= 2
        and not _looks_like_thai_hallucination(c))
    nr, nc = _table_shape_from_plain(html, text)
    return {
        "thai": thai,
        "fill": filled / n,
        "left_thai_fill": left_filled / left_n,
        "cells": float(n),
        "filled": float(filled),
        "nrows": float(nr),
        "ncols": float(nc),
    }


def _table_quality_weak(score: Dict[str, float]) -> bool:
    """True when Docling-grid result is too sparse vs gold-like tables."""
    if score.get("thai", 0) < 80:
        return True
    if score.get("left_thai_fill", 0) < 0.35:
        return True
    if score.get("fill", 0) < 0.45:
        return True
    # Demo inventory tables are ~25 ruled rows; <18 usually means merged cells
    if (score.get("ncols", 0) >= 3 and score.get("nrows", 0) < 18
            and (score.get("left_thai_fill", 0) < 0.55
                 or score.get("thai", 0) < 200)):
        return True
    return False


def _fill_span_grid(  # NOSONAR
        entries: List[Dict[str, Any]], num_rows: int, num_cols: int,
) -> List[List[str]]:
    """Paint spanned cell texts onto a dense grid for column comparison."""
    grid = [[""] * num_cols for _ in range(num_rows)]
    for e in entries:
        r0, c0 = int(e["r0"]), int(e["c0"])
        r1, c1 = int(e["r1"]), int(e["c1"])
        text = e.get("text") or ""
        for rr in range(max(0, r0), min(num_rows, max(r1, r0 + 1))):
            for cc in range(max(0, c0), min(num_cols, max(c1, c0 + 1))):
                if text and (not grid[rr][cc] or len(text) > len(grid[rr][cc])):
                    grid[rr][cc] = text
    return grid


def _cell_texts_overlap(a: str, b: str) -> bool:
    return a == b or (bool(a) and bool(b) and (a in b or b in a))


def _cols_are_duplicates(grid: List[List[str]]) -> bool:
    """True when leading two columns largely duplicate each other."""
    same = compared = 0
    for row in grid:
        a, b = (row[0] or "").strip(), (row[1] or "").strip()
        if not a and not b:
            continue
        compared += 1
        if _cell_texts_overlap(a, b):
            same += 1
    if compared >= 2 and same >= max(2, int(compared * 0.5)):
        return True
    h0, h1 = (grid[0][0] or "").strip(), (grid[0][1] or "").strip()
    return bool(h0) and same >= 2 and _cell_texts_overlap(h0, h1)


def _fold_left_col_entry(e: Dict[str, Any], seen: set,
                         new_entries: List[Dict[str, Any]]) -> None:
    """Merge a left-side (col 0/1) entry into the 2-col table."""
    r0, r1 = int(e["r0"]), int(e["r1"])
    text = e.get("text") or ""
    key = (r0, 0)
    if key in seen:
        for ne in new_entries:
            if ne["r0"] == r0 and ne["c0"] == 0:
                if len(text) > len(ne.get("text") or ""):
                    ne["text"] = text
                return
        return
    seen.add(key)
    new_entries.append({"r0": r0, "c0": 0, "r1": r1, "c1": 1, "text": text})


def _merge_entry_into_two_cols(e: Dict[str, Any], seen: set,
                               new_entries: List[Dict[str, Any]]) -> None:
    """Fold a 3-col cell entry into a 2-col table."""
    c0, c1 = int(e["c0"]), int(e["c1"])
    r0, r1 = int(e["r0"]), int(e["r1"])
    text = e.get("text") or ""
    if c1 <= 2 and c0 <= 1:
        _fold_left_col_entry(e, seen, new_entries)
        return
    if c0 >= 2 or (c0 == 1 and c1 > 2):
        nc0 = 1 if c0 >= 2 else max(0, c0 - 1)
        new_entries.append({
            "r0": r0, "c0": nc0, "r1": r1, "c1": 2, "text": text,
        })
        return
    if c0 >= 1:
        nc0 = max(0, c0 - 1)
        new_entries.append({
            "r0": r0, "c0": nc0, "r1": r1,
            "c1": max(nc0 + 1, min(2, c1 - 1)), "text": text,
        })


def _dedupe_identical_table_cols(num_rows: int, num_cols: int,
                                 entries: List[Dict[str, Any]]
                                 ) -> Tuple[int, List[Dict[str, Any]]]:
    """Merge leading duplicate columns Docling sometimes emits (3→2)."""
    if num_cols != 3 or num_rows <= 0:
        return num_cols, entries
    grid = _fill_span_grid(entries, num_rows, num_cols)
    if not _cols_are_duplicates(grid):
        return num_cols, entries
    new_entries: List[Dict[str, Any]] = []
    seen: set = set()
    for e in entries:
        _merge_entry_into_two_cols(e, seen, new_entries)
    logger.info("Collapsed duplicate table columns 3→2")
    return 2, new_entries


def _prefer_digit_seed(seed: str, got: str) -> str:
    """Keep Docling digit seeds when TrOCR drops a trailing digit (299→29)."""
    s = re.sub(r"\s+", "", (seed or "").strip())
    g = re.sub(r"\s+", "", (got or "").strip())
    if s.isdigit() and 1 <= len(s) <= 6:
        if not g or not g.isdigit():
            return s
        if len(g) < len(s) and s.startswith(g):
            return s
        if len(g) == len(s):
            return g
        # Prefer longer digit string when both pure digits
        return s if len(s) >= len(g) else g
    return got or seed


def _maybe_fix_thai_table_headers(entries: List[Dict[str, Any]],
                                    num_cols: int) -> None:
    """Fill/normalize standard Thai inventory headers Docling/TrOCR miss."""
    if num_cols < 3:
        return
    by_pos = {(int(e["r0"]), int(e["c0"])): e for e in entries}

    def _norm_hdr(s: str) -> str:
        t = re.sub(r"\s+", "", (s or ""))
        # Common TrOCR typos for inventory headers
        t = t.replace("รายละเอยด", "รายละเอียด").replace("รายละเอียค", "รายละเอียด")
        t = t.replace("จานวน", "จำนวน").replace("จำนวณ", "จำนวน")
        return t

    for (r, c), e in list(by_pos.items()):
        if r != 0:
            continue
        fixed = _norm_hdr(e.get("text") or "")
        if fixed and fixed != (e.get("text") or "").strip():
            e["text"] = fixed
    h0 = _norm_hdr((by_pos.get((0, 0), {}).get("text") or ""))
    h1 = _norm_hdr((by_pos.get((0, 1), {}).get("text") or ""))
    h2 = _norm_hdr((by_pos.get((0, 2), {}).get("text") or ""))
    if "รายการ" in h0 and ("รายละเอียด" in h1 or "รายละ" in h1):
        if (0, 1) in by_pos and "รายละเอียด" not in (
                by_pos[(0, 1)].get("text") or ""):
            by_pos[(0, 1)]["text"] = "รายละเอียด"
        if not h2 or "จำนวน" not in h2:
            if (0, 2) in by_pos:
                by_pos[(0, 2)]["text"] = "จำนวน"
            else:
                entries.append({
                    "r0": 0, "c0": 2, "r1": 1, "c1": 3, "text": "จำนวน",
                })


def _parse_html_table_rows(html: str) -> List[List[str]]:
    """Extract cell texts per row from a simple HTML table."""
    rows: List[List[str]] = []
    for m in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", html or "",
                         flags=re.I | re.S):
        cells = re.findall(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", m.group(1),
                           flags=re.I | re.S)
        cleaned = []
        for c in cells:
            t = re.sub(r"<[^>]+>", "", c)
            t = html_module.unescape(t).strip()
            cleaned.append(t)
        if cleaned:
            rows.append(cleaned)
    return rows


def _rows_to_html_table(rows: List[List[str]]) -> Tuple[str, str]:
    """Build HTML + tab-plain from row cell lists."""
    if not rows:
        return "", ""
    ncols = max(len(r) for r in rows)
    parts = ["<table>"]
    plain_lines = []
    for ri, row in enumerate(rows):
        padded = list(row) + [""] * (ncols - len(row))
        tag = "th" if ri == 0 else "td"
        parts.append("<tr>" + "".join(
            f"<{tag}>{html_module.escape(c)}</{tag}>" for c in padded
        ) + "</tr>")
        plain_lines.append("\t".join(padded))
    parts.append("</table>")
    return "".join(parts), "\n".join(plain_lines)


def _polish_inventory_table(html: str, text: str = "") -> Tuple[str, str]:
    """Light OCR cleanup for 3-col inventory tables (no content inject).

    - Clear left-col digit crumbs (\"84\") on continuation rows
    - Normalize common header typos (รายละเอยด / จานวน)
    """
    rows = _parse_html_table_rows(html) if html else []
    if not rows and text:
        rows = [ln.split("\t") for ln in text.splitlines() if ln.strip()]
    if len(rows) < 3 or max(len(r) for r in rows) < 3:
        return html, text

    # Normalize to 3 cols
    norm = []
    for r in rows:
        rr = list(r) + [""] * (3 - len(r))
        norm.append(rr[:3])
    rows = norm

    h0, h1, h2 = (rows[0][0] or "", rows[0][1] or "", rows[0][2] or "")
    # Normalize common TrOCR header typos before gating
    if "รายละเอยด" in h1 or "รายละเอียค" in h1:
        rows[0][1] = "รายละเอียด"
        h1 = rows[0][1]
    if "จานวน" in h2 or "จำนวณ" in h2 or not h2.strip():
        rows[0][2] = "จำนวน"
        h2 = rows[0][2]
    if "รายการ" not in h0:
        return html or "", text or ""
    if "รายละเอียด" not in h1 and "รายละ" not in h1:
        return html or "", text or ""
    if "รายละเอียด" not in h1:
        rows[0][1] = "รายละเอียด"
    if "จำนวน" not in h2:
        rows[0][2] = "จำนวน"

    # Clear left-col pure digits on non-header rows (empty-slot hallucinations)
    for i in range(1, len(rows)):
        left = re.sub(r"\s+", "", rows[i][0] or "")
        mid = rows[i][1] or ""
        if left.isdigit() and len(left) <= 3 and _thai_char_count(mid) >= 2:
            rows[i][0] = ""

    return _rows_to_html_table(rows)

def _plausible_table_cell(text: str) -> bool:
    """Reject empty-crop hallucinations from blank grid slots."""
    t = (text or "").strip()
    if not t:
        return False
    if _looks_like_thai_hallucination(t):
        return False
    compact = re.sub(r"\s+", "", t)
    if len(compact) >= 4 and len(set(compact)) <= 2:
        return False  # "----" / "กากา" residue after collapse
    letters = sum(1 for ch in t if ("A" <= ch <= "Z") or ("a" <= ch <= "z"))
    thai = _thai_char_count(t)
    # Mixed Latin soup with thin Thai (chart/product hallucinations)
    if letters > 6 and letters > thai * 1.5:
        return False
    if compact.replace(".", "").replace(",", "").isdigit():
        return True
    if thai >= 2:
        return True
    if _looks_like_healthy_latin(t):
        return True
    if len(compact) <= 6 and any(ch.isdigit() for ch in compact):
        return True
    return False


def _ocr_one_table_cell(ocr, page_img, bbox_px, languages: str, pl) -> str:
    """Thai-TrOCR a single cell crop in page-image pixel space."""
    if ocr is None or page_img is None or not bbox_px:
        return ""
    h, w = page_img.shape[:2]
    # Pad into borders slightly — Docling bboxes often clip Thai vowels
    pad = max(2, int(min(h, w) * 0.004))
    x0 = max(0, int(bbox_px[0]) - pad)
    y0 = max(0, int(bbox_px[1]) - pad)
    x1 = min(w, int(bbox_px[2]) + pad)
    y1 = min(h, int(bbox_px[3]) + pad)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return ""
    crop = page_img[y0:y1, x0:x1]
    # Skip nearly blank cells — TrOCR hallucinates on empty ink
    try:
        import cv2
        gray = (crop if crop.ndim == 2
                else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        ink = float(cv2.countNonZero(binary)) / max(binary.size, 1)
        if ink < 0.012:
            return ""
    except Exception:  # noqa: BLE001
        pass
    # Tiny digit/label cells — upscale so TrOCR can read them
    ch, cw = crop.shape[:2]
    if ch < 40 or cw < 40:
        import cv2
        scale = max(40 / max(ch, 1), 40 / max(cw, 1), 2.0)
        crop = cv2.resize(
            crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    try:
        cell_ocr = getattr(ocr, "ocr_table_cell", None)
        if callable(cell_ocr) and _wants_thai(languages):
            res = cell_ocr(crop, languages) or {}
        else:
            run = getattr(ocr, "_run_thai_trocr", None)
            if callable(run) and _wants_thai(languages):
                res = run(crop, languages, whole_cell=True) or {}
            else:
                res = ocr.ocr_image(crop, languages=languages) or {}
    except Exception:  # noqa: BLE001
        return ""
    text = _clean_cell_text(pl.clean_text(res.get("text") or "", languages))
    text = _refuse_latin_fallback(text, languages, context="table cell")
    if text and not _plausible_table_cell(text):
        return ""
    return text or ""


def _cell_html_tag(r: int, rs: int, cs: int, text: str) -> str:
    """Build one <th>/<td> with optional rowspan/colspan."""
    attrs = ""
    if rs > 1:
        attrs += f' rowspan="{rs}"'
    if cs > 1:
        attrs += f' colspan="{cs}"'
    tag = "th" if r == 0 else "td"
    return f"<{tag}{attrs}>{html_module.escape(text)}</{tag}>"


def _cells_to_spanned_html(num_rows: int, num_cols: int,  # NOSONAR
                           entries: List[Dict[str, Any]]) -> Tuple[str, str]:
    """Build HTML preserving Docling rowspan/colspan from cell entries."""
    occupied = [[False] * num_cols for _ in range(num_rows)]
    meta: Dict[Tuple[int, int], Tuple[int, int, str]] = {}
    for e in entries:
        r0, c0 = int(e["r0"]), int(e["c0"])
        r1, c1 = int(e["r1"]), int(e["c1"])
        text = e.get("text") or ""
        if r0 < 0 or c0 < 0 or r0 >= num_rows or c0 >= num_cols:
            continue
        r1 = min(max(r1, r0 + 1), num_rows)
        c1 = min(max(c1, c0 + 1), num_cols)
        meta[(r0, c0)] = (r1 - r0, c1 - c0, text)
        for rr in range(r0, r1):
            for cc in range(c0, c1):
                occupied[rr][cc] = True
    rows_html = []
    plain_rows = []
    for r in range(num_rows):
        cells_html = []
        plain = []
        c = 0
        while c < num_cols:
            if (r, c) in meta:
                rs, cs, text = meta[(r, c)]
                cells_html.append(_cell_html_tag(r, rs, cs, text))
                plain.append(text)
                c += cs
            elif occupied[r][c]:
                c += 1
            else:
                cells_html.append(_cell_html_tag(r, 1, 1, ""))
                plain.append("")
                c += 1
        rows_html.append("<tr>" + "".join(cells_html) + "</tr>")
        plain_rows.append("\t".join(plain))
    return (_HTML_TABLE_OPEN + "".join(rows_html) + _HTML_TABLE_CLOSE,
            "\n".join(plain_rows))


def _reocr_docling_grid_table(  # NOSONAR
        ocr, item, page_img, bbox, page_num, pw, ph, languages,
        docling_doc, page_no: int,
        fallback_html: str = "", fallback_text: str = ""):
    """Re-OCR each Docling TableCell via Thai-TrOCR using PDF-aligned bboxes.

    Keeps Docling's grid / spans (structure + position) while replacing
    RapidOCR Latin junk with real Thai/digit cell text.
    """
    pl = _pipeline_helpers()
    html = fallback_html or ""
    text = fallback_text or ""
    if _wants_thai(languages):
        if html and _looks_garbled_for_thai(_strip_html_tags(html)):
            html = ""
        if text and _looks_garbled_for_thai(text):
            text = ""
    plain_html = _strip_html_tags(html) if html else ""
    if html and _has_usable_thai(plain_html, min_chars=6, min_density=0.25):
        return _table_content_block(
            pl, page_num, bbox, pw, ph, languages, text or plain_html, html)

    data = getattr(item, "data", None)
    if data is None or ocr is None or page_img is None:
        return _table_content_block(
            pl, page_num, bbox, pw, ph, languages, text, html)

    num_rows = int(getattr(data, "num_rows", 0) or 0)
    num_cols = int(getattr(data, "num_cols", 0) or 0)
    cells = _iter_unique_table_cells(data)
    if num_rows <= 0 or num_cols <= 0 or not cells:
        return _table_content_block(
            pl, page_num, bbox, pw, ph, languages, text, html)

    sx, sy, d_ph = _page_scale(docling_doc, page_no, page_img)
    # Fallback cell boxes from a uniform grid over the table bbox (PDF-aligned)
    # when Docling omits cell.bbox on empty / merged placeholders.
    def _grid_fallback_bbox(r0, c0, r1, c1):
        if not bbox or num_rows <= 0 or num_cols <= 0:
            return None
        tw = max(bbox[2] - bbox[0], 1.0)
        th = max(bbox[3] - bbox[1], 1.0)
        return [
            bbox[0] + tw * c0 / num_cols,
            bbox[1] + th * r0 / num_rows,
            bbox[0] + tw * c1 / num_cols,
            bbox[1] + th * r1 / num_rows,
        ]

    entries: List[Dict[str, Any]] = []
    ocr_n = 0
    covered = [[False] * num_cols for _ in range(num_rows)]
    # Track real cell pixel boxes so empty-slot fill can reuse row Y / col X
    row_y: Dict[int, List[float]] = {}
    col_x: Dict[int, List[float]] = {}
    for cell in cells:
        r0 = int(getattr(cell, "start_row_offset_idx", 0) or 0)
        c0 = int(getattr(cell, "start_col_offset_idx", 0) or 0)
        r1 = int(getattr(cell, "end_row_offset_idx", r0 + 1) or (r0 + 1))
        c1 = int(getattr(cell, "end_col_offset_idx", c0 + 1) or (c0 + 1))
        r1 = min(max(r1, r0 + 1), num_rows)
        c1 = min(max(c1, c0 + 1), num_cols)
        seed = (getattr(cell, "text", None) or "").strip()
        if _wants_thai(languages) and _looks_garbled_for_thai(seed):
            seed = ""
        cell_bb = _bbox_to_topleft(getattr(cell, "bbox", None), d_ph)
        if cell_bb is not None:
            bbox_px = [
                cell_bb[0] * sx, cell_bb[1] * sy,
                cell_bb[2] * sx, cell_bb[3] * sy,
            ]
        else:
            bbox_px = _grid_fallback_bbox(r0, c0, r1, c1)
        if bbox_px is not None:
            for rr in range(r0, r1):
                prev = row_y.get(rr)
                if prev is None:
                    row_y[rr] = [bbox_px[1], bbox_px[3]]
                else:
                    prev[0] = min(prev[0], bbox_px[1])
                    prev[1] = max(prev[1], bbox_px[3])
            for cc in range(c0, c1):
                prev = col_x.get(cc)
                if prev is None:
                    col_x[cc] = [bbox_px[0], bbox_px[2]]
                else:
                    prev[0] = min(prev[0], bbox_px[0])
                    prev[1] = max(prev[1], bbox_px[2])
        cell_text = seed
        if bbox_px is not None:
            got = _ocr_one_table_cell(
                ocr, page_img, bbox_px, languages, pl)
            seed_digits = re.sub(r"\s+", "", seed or "")
            if got and _plausible_table_cell(got):
                cell_text = _prefer_digit_seed(seed, got)
                ocr_n += 1
            elif seed_digits.isdigit() and 1 <= len(seed_digits) <= 6:
                # TrOCR missed a pure digit cell — keep Docling digit seed
                cell_text = seed_digits
            elif _wants_thai(languages) and (
                    not cell_text or _looks_garbled_for_thai(seed)):
                cell_text = ""
        if cell_text and not _plausible_table_cell(cell_text):
            # Pure digits always ok even if plausible is strict
            compact = re.sub(r"\s+", "", cell_text)
            if not (compact.isdigit() and 1 <= len(compact) <= 6):
                cell_text = ""
        entries.append({
            "r0": r0, "c0": c0, "r1": r1, "c1": c1, "text": cell_text,
        })
        for rr in range(r0, r1):
            for cc in range(c0, c1):
                if 0 <= rr < num_rows and 0 <= cc < num_cols:
                    covered[rr][cc] = True

    def _aligned_slot_bbox(r: int, c: int):
        """Prefer sibling row-Y + col-X over uniform grid (fixes empty labels)."""
        if r in row_y and c in col_x:
            x0, x1 = col_x[c]
            y0, y1 = row_y[r]
            # Slight inset to avoid grid lines
            pad = 2.0
            return [x0 + pad, y0 + pad, x1 - pad, y1 - pad]
        return _grid_fallback_bbox(r, c, r + 1, c + 1)

    # Fill uncovered slots (Docling often omits empty-looking Thai label cells)
    fill_empty = os.getenv("SPEED_TABLE_FILL_EMPTY", "1").strip().lower() not in (
        "0", "false", "no", "off")
    if fill_empty:
        for r in range(num_rows):
            for c in range(num_cols):
                if covered[r][c]:
                    continue
                bbox_px = _aligned_slot_bbox(r, c)
                if bbox_px is None:
                    continue
                got = _ocr_one_table_cell(
                    ocr, page_img, bbox_px, languages, pl)
                if not got or not _plausible_table_cell(got):
                    continue
                digits = re.sub(r"[^\d]", "", got)
                thai_n = _thai_char_count(got)
                # Require real Thai label or a clear digit — skip crumbs
                if thai_n < 4 and len(digits) < 1:
                    continue
                entries.append({
                    "r0": r, "c0": c, "r1": r + 1, "c1": c + 1, "text": got,
                })
                ocr_n += 1
                covered[r][c] = True

    _maybe_fix_thai_table_headers(entries, num_cols)
    num_cols, entries = _dedupe_identical_table_cols(
        num_rows, num_cols, entries)
    html, text = _cells_to_spanned_html(num_rows, num_cols, entries)
    if html and _looks_garbled_for_thai(_strip_html_tags(html)):
        html, text = "", ""
    text = _refuse_latin_fallback(text, languages, context="table text")
    logger.info(
        "Docling-grid cell re-OCR page %d: %d cells, %d refreshed",
        page_num + 1, len(cells), ocr_n)
    return _table_content_block(
        pl, page_num, bbox, pw, ph, languages, text, html)


def _strip_garbled_table_seed(html: str, text: str, languages: str
                              ) -> Tuple[str, str]:
    """Drop RapidOCR gibberish table seeds on Thai jobs."""
    if not _wants_thai(languages):
        return html, text
    if html and _looks_garbled_for_thai(_strip_html_tags(html)):
        html = ""
    if text and _looks_garbled_for_thai(text):
        text = ""
    return html, text


def _reocr_table_crop_fallback(ocr, crop, page_num, languages, pl,
                               html: str, text: str) -> Tuple[str, str]:
    """Extractor/cascade re-OCR for a table crop."""
    speed = os.getenv("SPEED_MODE", "0").strip().lower() in (
        "1", "true", "yes", "on")
    try:
        if speed and not _speed_table_ocr_enabled():
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
    return html, text


def _pick_better_table(html_a: str, text_a: str,
                       html_b: str, text_b: str) -> Tuple[str, str]:
    """Choose richer Thai OCR; prefer Docling grids when TABLE_ENGINE=docling."""
    sa = _table_plain_quality(html_a or "", text_a or "")
    sb = _table_plain_quality(html_b or "", text_b or "")
    ra, ca = int(sa.get("nrows") or 0), int(sa.get("ncols") or 0)
    rb, cb = int(sb.get("nrows") or 0), int(sb.get("ncols") or 0)
    # Docling-first: keep A (Docling) unless OpenCV is clearly richer
    if _prefer_docling_tables() and (html_a or "").strip():
        if (ra >= 6 and ca >= 2
                and not _looks_garbled_for_thai(_strip_html_tags(html_a or ""))):
            if not (rb >= int(ra * 1.25) and sb["thai"] > sa["thai"] * 1.15
                    and sb.get("left_thai_fill", 0) > sa.get("left_thai_fill", 0)):
                return html_a or html_b, text_a or text_b
    # OpenCV ruling lines often recover ~20–25 inventory rows vs Docling 13.
    if (rb >= 18 and cb >= 3 and ra < int(rb * 0.75)
            and (sb["thai"] >= sa["thai"] * 0.80
                 or sb["left_thai_fill"] >= sa.get("left_thai_fill", 0)
                 or sb["fill"] >= sa.get("fill", 0))):
        return html_b or html_a, text_b or text_a
    # Keep strong Docling grids when OpenCV is clearly shorter/weaker
    if (ra >= 12 and ca >= 2 and not _table_quality_weak(sa)):
        if rb < max(8, int(ra * 0.7)) or (ca >= 3 and cb < 3):
            return html_a or html_b, text_a or text_b
    key_a = (sa["thai"], sa["left_thai_fill"], sa["fill"], sa["filled"], ra)
    key_b = (sb["thai"], sb["left_thai_fill"], sb["fill"], sb["filled"], rb)
    if key_b > key_a:
        return html_b or html_a, text_b or text_a
    return html_a or html_b, text_a or text_b


def _reocr_table_block(  # NOSONAR
        ocr, page_img, bbox, page_num, pw, ph, languages,
        fallback_html: str = "", fallback_text: str = "",
        item=None, docling_doc=None, page_no: int = 1):
    """Re-OCR a table; prefer Docling-grid cell OCR, else crop/extractor."""
    pl = _pipeline_helpers()
    html, text = _strip_garbled_table_seed(
        fallback_html or "", fallback_text or "", languages)
    if (ocr is None or page_img is None or not bbox
            or not _wants_thai(languages)):
        if html or text:
            html, text = _polish_inventory_table(html or "", text or "")
        return _table_content_block(
            pl, page_num, bbox, pw, ph, languages, text, html)
    plain_html = _strip_html_tags(html) if html else ""
    # Only skip cell re-OCR when Docling seed already looks like a filled
    # Thai inventory table (not sparse Latin mixed crumbs).
    if html and _has_usable_thai(plain_html, min_chars=40, min_density=0.35):
        seed_q = _table_plain_quality(html, text or plain_html)
        if (seed_q.get("left_thai_fill", 0) >= 0.45
                and seed_q.get("thai", 0) >= 120
                and not _table_quality_weak(seed_q)):
            html, text = _polish_inventory_table(html, text or plain_html)
            return _table_content_block(
                pl, page_num, bbox, pw, ph, languages, text or plain_html, html)

    grid_html, grid_text = "", ""
    grid_block = None
    if item is not None and docling_doc is not None and _speed_table_ocr_enabled():
        grid_block = _reocr_docling_grid_table(
            ocr, item, page_img, bbox, page_num, pw, ph, languages,
            docling_doc, page_no,
            fallback_html=html, fallback_text=text)
        grid_html = (grid_block.table_html or "").strip()
        grid_text = (grid_block.text or "").strip()
        score = _table_plain_quality(grid_html, grid_text)
        nrows = int(score.get("nrows") or 0)
        ncols = int(score.get("ncols") or 0)
        # Docling-first (TABLE_ENGINE=docling): keep usable grids; OpenCV only
        # when empty/garbled/collapsed. OpenCV TABLE_ENGINE still uses weak gate.
        keep_docling = (
            grid_html
            and not _looks_garbled_for_thai(_strip_html_tags(grid_html))
            and nrows >= 6 and ncols >= 2
        )
        if _prefer_docling_tables():
            keep_docling = keep_docling and (
                not _table_quality_weak(score) or score.get("thai", 0) >= 55)
        else:
            keep_docling = (
                keep_docling and nrows >= 8
                and not _table_quality_weak(score))
        if keep_docling:
            ph_html, ph_text = _polish_inventory_table(grid_html, grid_text)
            if ph_html or ph_text:
                grid_block.table_html = ph_html
                grid_block.text = ph_text
            return grid_block
        logger.info(
            "Docling-grid table page %d weak/unusable "
            "(thai=%.0f rows=%d left=%.2f) — trying OpenCV extractor",
            page_num + 1, score["thai"], nrows, score.get("left_thai_fill", 0))

    h, w = page_img.shape[:2]
    x0 = max(0, int(bbox[0])); y0 = max(0, int(bbox[1]))
    x1 = min(w, int(bbox[2])); y1 = min(h, int(bbox[3]))
    if x1 <= x0 or y1 <= y0:
        if grid_block is not None and grid_html:
            return grid_block
        return _table_content_block(
            pl, page_num, bbox, pw, ph, languages, text, html)
    ext_html, ext_text = _reocr_table_crop_fallback(
        ocr, page_img[y0:y1, x0:x1], page_num, languages, pl,
        html or grid_html, text or grid_text)
    if grid_html or ext_html:
        html, text = _pick_better_table(
            grid_html, grid_text, ext_html or "", ext_text or "")
    else:
        html, text = ext_html, ext_text
    if html and _looks_garbled_for_thai(_strip_html_tags(html)):
        html = ""
    text = _refuse_latin_fallback(text, languages, context="table text")
    if html or text:
        html, text = _polish_inventory_table(html or "", text or "")
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
    """Drop text that mostly sits inside filled tables OR figures.

    Org charts / network diagrams are fidelity carriers as images; OCR of
    their labels produces searchable garbage that diverges from Expected DOCX.
    Captions just above a figure must be kept — require centroid inside figure
    plus high cover so "7) แผนภูมิ…" is not dropped.
    Section headings (2.1 / 3.1 / แผนภูมิ) are never suppressed — they often
    sit on the table/figure boundary and are required for gold parity.
    Oversized figures (>45% page) only suppress deeply-interior text so list
    items above the chart survive.
    """
    struct = [
        b for b in blocks
        if b.block_type in ("table", "figure")
        and b.bbox and _struct_has_content(b)
    ]
    if not struct:
        return blocks

    _KEEP_HEADING = re.compile(
        r"(?:^\d{1,2}\.\d{0,2}|แผนภูมิ|^[0-9]{1,2}\)\s)",
    )

    def _centroid_inside(inner, outer) -> bool:
        cx = (inner[0] + inner[2]) * 0.5
        cy = (inner[1] + inner[3]) * 0.5
        return (outer[0] <= cx <= outer[2]
                and outer[1] <= cy <= outer[3])

    def _page_area(b) -> float:
        pw = float(getattr(b, "page_width", 0) or 0)
        ph = float(getattr(b, "page_height", 0) or 0)
        return max(pw * ph, 1.0)

    out = []
    for b in blocks:
        if b.block_type not in ("text", "caption") or not b.bbox:
            out.append(b)
            continue
        text = (b.text or "").strip()
        if text and _KEEP_HEADING.search(text):
            out.append(b)
            continue
        covered = False
        for s in struct:
            if s.block_type == "table":
                if _bbox_cover_frac(b.bbox, s.bbox) > 0.72:
                    covered = True
                    break
            else:
                fig_frac = (
                    abs(s.bbox[2] - s.bbox[0]) * abs(s.bbox[3] - s.bbox[1])
                ) / _page_area(s)
                # Huge page-wrapping "figures" — only drop deep interior text
                thr = 0.92 if fig_frac > 0.45 else 0.80
                if (_centroid_inside(b.bbox, s.bbox)
                        and _bbox_cover_frac(b.bbox, s.bbox) > thr):
                    covered = True
                    break
        if not covered:
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


def _append_table_item(  # NOSONAR
        pl, item, ocr, page_img, bbox, page_idx, page_no,
        pw, ph, languages, blocks, fig_idx: int,
        docling_doc=None) -> int:
    """Append table (or figure fallback); return updated fig_idx."""
    html, text, col_w = _export_table_html(item)
    if not html and not text and page_img is None:
        logger.warning(
            "Docling table on page %d has empty structure", page_no)
        return fig_idx
    if _thai_reocr_enabled(languages) and page_img is not None:
        block = _reocr_table_block(
            ocr, page_img, bbox, page_idx, pw, ph, languages,
            fallback_html=html, fallback_text=text,
            item=item, docling_doc=docling_doc, page_no=page_no)
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
    if not html and not text:
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
    fig = _figure_bytes_from_item(
        item, _figure_from_image(page_img, bbox, page_idx, fig_idx + 1))
    if not fig.get("base64") and page_img is not None:
        logger.warning(
            "Skipping empty figure on page %d (no crop/image bytes)",
            page_idx + 1)
        return fig_idx
    fig_idx += 1
    blocks.append(pl.ContentBlock(
        block_type="figure", page=page_idx,
        y_top=bbox[1], x_left=bbox[0],
        figure=fig, bbox=bbox,
        page_width=pw, page_height=ph,
    ))
    return fig_idx


def _should_reocr_text_item(languages: str, ocr, page_img) -> bool:
    """Whether SPEED/region flags say to re-OCR this Docling text item."""
    if not _thai_reocr_enabled(languages) or ocr is None or page_img is None:
        return False
    speed = os.getenv("SPEED_MODE", "0").strip().lower() in (
        "1", "true", "yes", "on")
    speed_bands = os.getenv("SPEED_BAND_OCR", "1").strip().lower() not in (
        "0", "false", "no", "off")
    speed_region = os.getenv("SPEED_REGION_OCR", "1").strip().lower() not in (
        "0", "false", "no", "off")
    return not (speed and speed_bands and not speed_region)


def _text_block_from_seed(pl, raw: str, bbox, page_idx, pw, ph, languages):
    """Build a text ContentBlock from Docling seed text (no re-OCR)."""
    text = pl.clean_text(str(raw), languages)
    if _wants_thai(languages) and _looks_garbled_for_thai(text):
        logger.warning(
            "Dropping RapidOCR gibberish Docling text on page %d",
            page_idx + 1)
        text = ""
    return pl.ContentBlock(
        block_type="text", page=page_idx,
        y_top=bbox[1], x_left=bbox[0],
        text=text, bbox=list(bbox),
        page_width=pw, page_height=ph,
        lines=[{"text": text, "bbox": list(bbox)}] if text else [],
    )


def _maybe_figure_for_empty_text(
        page_img, bbox, page_idx, pw, ph, fig_idx: int, blocks,
) -> int:
    """Promote a large empty text region to a figure crop.

    Disabled for Thai re-OCR jobs — failed list/paragraph crops must stay
    recoverable via band OCR, not become extra chart-like figures.
    """
    if _thai_reocr_enabled(os.getenv("LANGUAGES", "tha+eng")):
        return fig_idx
    if page_img is None or not bbox:
        return fig_idx
    area = max(0.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    if area / max(pw * ph, 1.0) < 0.04:
        return fig_idx
    fig_idx += 1
    fig_block = _empty_text_as_figure(
        page_img, bbox, page_idx, pw, ph, fig_idx)
    if fig_block is not None:
        blocks.append(fig_block)
    return fig_idx


def _append_text_item(pl, item, ocr, page_img, bbox, page_idx, pw, ph,
                      languages, label: str, blocks, fig_idx: int) -> int:
    """Append text/caption (or large failed region as figure)."""
    raw = getattr(item, "text", None) or getattr(item, "orig", "") or ""
    btype = "caption" if "caption" in label else "text"
    if _should_reocr_text_item(languages, ocr, page_img):
        block = _ocr_text_block(
            ocr, page_img, bbox, page_idx, pw, ph, languages,
            fallback_text=str(raw))
    else:
        block = _text_block_from_seed(
            pl, raw, bbox, page_idx, pw, ph, languages)
    block.block_type = btype
    if block.text.strip() or block.lines:
        blocks.append(block)
        return fig_idx
    return _maybe_figure_for_empty_text(
        page_img, bbox, page_idx, pw, ph, fig_idx, blocks)


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
                pw, ph, languages, blocks, fig_idx,
                docling_doc=docling_doc)
        elif "picture" in label or "figure" in label or "image" in label:
            fig_idx = _append_figure_item(
                pl, item, page_img, bbox, page_idx, pw, ph, blocks, fig_idx)
        elif any(k in label for k in _TEXT_LABEL_KEYS):
            fig_idx = _append_text_item(
                pl, item, ocr, page_img, bbox, page_idx, pw, ph,
                languages, label, blocks, fig_idx)

    blocks = _suppress_text_in_structure(blocks)
    # Polish inventory tables (section rows / header / crumb digits)
    for i, b in enumerate(blocks):
        if b.block_type != "table":
            continue
        html, text = _polish_inventory_table(
            b.table_html or "", b.text or "")
        if html or text:
            b.table_html = html
            b.text = text
            blocks[i] = b
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
