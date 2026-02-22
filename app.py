"""
PDF to DOCX OCR Service — Gradio Web Application
v0.5.0  |  Manual correction + auto-retrain every 100 corrections

Convert tab: Upload → detect → manual add tables/figures → convert
Review tab: See detected regions, draw new ones, re-convert
"""
import os
import json
import logging
from pathlib import Path

import cv2
import numpy as np
import fitz

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Fix Gradio boolean-schema crash (safe, version-agnostic) ─────────────────
try:
    import gradio_client.utils as _gc_utils
    if hasattr(_gc_utils, "_json_schema_to_python_type"):
        _orig_json_schema_fn = _gc_utils._json_schema_to_python_type

        def _patched(schema, defs=None):
            if isinstance(schema, bool):
                return "Any" if schema else "None"
            return _orig_json_schema_fn(schema, defs)

        _gc_utils._json_schema_to_python_type = _patched
except Exception:
    pass
# ─────────────────────────────────────────────────────────────────────────────

import gradio as gr

from src.pipeline import OCRPipeline
from src.services import HistoryManager

# ── Initialise services ──────────────────────────────────────────────────────
pipeline = OCRPipeline()
history = HistoryManager()
history.cleanup_old_entries()

# ── Option maps ──────────────────────────────────────────────────────────────
QUALITY_OPTIONS = {
    "Standard (Fast)": "fast",
    "Balanced (Recommended)": "balanced",
    "Best (Accurate)": "accurate",
}

LANGUAGE_OPTIONS = {
    "English": "eng",
    "Thai": "tha",
    "Thai + English": "tha+eng",
    "Chinese (Simplified)": "chi_sim",
    "Chinese + English": "chi_sim+eng",
    "Japanese": "jpn",
    "Japanese + English": "jpn+eng",
    "Korean": "kor",
    "Korean + English": "kor+eng",
    "Arabic": "ara",
    "Auto-detect": "auto",
}

ENGINE_OPTIONS = {
    "Tesseract (Default)": "tesseract",
    "PaddleOCR": "paddleocr",
    "EasyOCR": "easyocr",
}

CLASS_OPTIONS = ["table", "figure"]

_PAGE_ZERO = "Page 0 / 0"
_LOCAL_USER = "local"

# Colours for bounding box overlay
_BOX_COLORS = {
    "table": (0, 200, 0),        # green
    "figure": (255, 140, 0),     # orange
    "text": (180, 180, 180),     # grey
    "plain text": (180, 180, 180),
    "title": (100, 100, 255),    # blue
}


# ══════════════════════════════════════════════════════════════════════════════
# Helper — draw detection boxes on an image
# ══════════════════════════════════════════════════════════════════════════════
def _draw_detections(img: np.ndarray, detections: dict,
                     manual_regions: list | None = None) -> np.ndarray:
    """Return a copy of *img* with bounding boxes and labels drawn."""
    canvas = img.copy()
    for category, dets in detections.items():
        for det in dets:
            bbox = det.get("bbox", [0, 0, 0, 0])
            cls_name = det.get("class", category.rstrip("s"))
            conf = det.get("confidence", 0)
            x0, y0, x1, y1 = [int(v) for v in bbox]
            colour = _BOX_COLORS.get(cls_name, (180, 180, 180))
            cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 2)
            label = f"{cls_name} {conf:.0%}" if conf else cls_name
            cv2.putText(canvas, label, (x0, max(y0 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)

    for mr in (manual_regions or []):
        bbox = mr["bbox"]
        cls_name = mr.get("class", "figure")
        x0, y0, x1, y1 = [int(v) for v in bbox]
        colour = (0, 255, 255) if cls_name == "table" else (255, 0, 255)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, 3)
        cv2.putText(canvas, f"MANUAL {cls_name}", (x0, max(y0 - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)
    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# PDF Preview (legacy — used by the Convert tab)
# ══════════════════════════════════════════════════════════════════════════════
def render_page_preview(pdf_path: str, page_num: int = 0, scale: float = 1.5,
                        header_pct: float = 0, footer_pct: float = 0):
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            doc.close()
            return None, 0
        page = doc[page_num]
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)

        h, w = img.shape[:2]
        if header_pct > 0:
            hp = int((header_pct / 100) * h)
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (w, hp), (80, 80, 80), -1)
            img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)
        if footer_pct > 0:
            fp = int((footer_pct / 100) * h)
            overlay = img.copy()
            cv2.rectangle(overlay, (0, h - fp), (w, h), (80, 80, 80), -1)
            img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0)

        total = len(doc)
        doc.close()
        return img, total
    except Exception as exc:
        logger.error(f"Preview error: {exc}")
        return None, 0


# ══════════════════════════════════════════════════════════════════════════════
# PDF Convert-tab handlers
# ══════════════════════════════════════════════════════════════════════════════
def load_pdf_preview(pdf_file, quality, header_pct, footer_pct):
    if pdf_file is None:
        return None, _PAGE_ZERO, 0, 0, gr.update(visible=False)
    pdf_path = str(pdf_file)
    img, total = render_page_preview(pdf_path, 0, 1.5, header_pct, footer_pct)
    if img is not None:
        return img, f"Page 1 / {total}", 1, total, gr.update(visible=True)
    return None, _PAGE_ZERO, 0, 0, gr.update(visible=False)


def change_page(pdf_file, direction, current, total, header_pct, footer_pct):
    if pdf_file is None or total == 0:
        return None, f"Page {current} / {total}", current
    pdf_path = str(pdf_file)
    new_page = max(1, min(current + direction, total))
    img, _ = render_page_preview(pdf_path, new_page - 1, 1.5, header_pct, footer_pct)
    return img, f"Page {new_page} / {total}", new_page


def process_document(pdf_file, quality_label, header_pct, footer_pct,
                     language_label, engine_label, yolo_conf):
    """Convert a PDF — no authentication required in local mode."""
    if pdf_file is None:
        return ("", "Please upload a PDF file first.",
                None, None, gr.update(visible=False), gr.update())

    pdf_path = str(pdf_file)
    quality = QUALITY_OPTIONS.get(quality_label, "balanced")
    languages = LANGUAGE_OPTIONS.get(language_label, "eng")
    engine = ENGINE_OPTIONS.get(engine_label, "tesseract")

    pipeline.ocr.primary_engine = engine

    result = pipeline.process_pdf(
        pdf_path, quality=quality,
        header_trim=header_pct, footer_trim=footer_pct,
        languages=languages, yolo_confidence=yolo_conf,
    )

    if result["success"]:
        text = result["text"]
        files = result["files"]
        meta = result["metadata"]

        engine_display = engine_label or "Tesseract (Default)"
        status = (
            f"**Conversion Complete!**\n\n"
            f"Pages: {meta.get('pages', 0)} | "
            f"Tables: {meta.get('tables', 0)} | "
            f"Figures: {meta.get('figures', 0)}\n"
            f"Language: {language_label} | Quality: {quality_label} | "
            f"Engine: {engine_display}"
        )

        original_name = os.path.basename(pdf_path)
        history.save_result(_LOCAL_USER, original_name, files, meta)

        txt_path = files.get("txt")
        docx_path = files.get("docx")

        entries = history.list_entries(_LOCAL_USER)
        history_data = [[e["original_filename"], e["created_at"][:19], e["entry_id"]]
                        for e in entries[:20]]

        return (
            text, status,
            txt_path if txt_path and os.path.exists(txt_path) else None,
            docx_path if docx_path and os.path.exists(docx_path) else None,
            gr.update(visible=True),
            gr.update(value=history_data),
        )
    else:
        return (f"Error: {result['error']}", f"**Error:** {result['error']}",
                None, None, gr.update(visible=False), gr.update())


# ══════════════════════════════════════════════════════════════════════════════
# Review & Correct tab handlers
# ══════════════════════════════════════════════════════════════════════════════
def _detect_and_draw(pdf_path: str, page_idx: int, yolo_conf: float,
                     manual_regions: list):
    """Run detection + overlay on one page. Returns annotated image + raw detections."""
    res = pipeline.detect_page_regions(
        pdf_path, page_idx, yolo_confidence=yolo_conf)
    if not res["success"]:
        return None, {}, f"Detection error: {res.get('error')}"
    img = res["page_image"]
    dets = res["detections"]
    annotated = _draw_detections(img, dets, manual_regions)
    n_t = len(dets.get("tables", []))
    n_f = len(dets.get("figures", []))
    n_txt = len(dets.get("text_regions", []))
    info = (f"Detected: **{n_t}** tables, **{n_f}** figures, "
            f"**{n_txt}** text regions  |  "
            f"Manual additions: **{len(manual_regions)}**")
    return annotated, dets, info


def review_load_pdf(pdf_file, yolo_conf):
    """Load a PDF into the Review tab — detect page 0."""
    if pdf_file is None:
        return (None, "Upload a PDF first.", 0, 0, {}, [],
                gr.update(visible=False))
    pdf_path = str(pdf_file)
    try:
        doc = fitz.open(pdf_path)
        total = len(doc)
        doc.close()
    except Exception:
        return (None, "Cannot open PDF.", 0, 0, {}, [],
                gr.update(visible=False))
    annotated, dets, info = _detect_and_draw(pdf_path, 0, yolo_conf, [])
    return (annotated, info, 0, total, dets, [],
            gr.update(visible=True))


def review_change_page(pdf_file, direction, current_page, total_pages,
                       yolo_conf, manual_regions_state):
    """Navigate pages in Review tab."""
    if pdf_file is None or total_pages == 0:
        return None, "", current_page, {}, manual_regions_state
    new_page = max(0, min(current_page + direction, total_pages - 1))
    # Get manual regions for this specific page
    page_manuals = [r for r in manual_regions_state if r.get("page") == new_page]
    annotated, dets, info = _detect_and_draw(
        str(pdf_file), new_page, yolo_conf, page_manuals)
    return annotated, info, new_page, dets, manual_regions_state


def review_add_region(pdf_file, current_page, total_pages, yolo_conf,
                      x0, y0, x1, y1, region_class,
                      detections_state, manual_regions_state):
    """Add a manually-drawn region and refresh the overlay."""
    if pdf_file is None:
        return None, "Upload a PDF first.", detections_state, manual_regions_state

    # Validate bbox
    try:
        bbox = [float(x0), float(y0), float(x1), float(y1)]
    except (ValueError, TypeError):
        return None, "Invalid coordinates — enter numbers for x0, y0, x1, y1.", \
               detections_state, manual_regions_state

    if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
        return None, "x1 must be > x0 and y1 must be > y0.", \
               detections_state, manual_regions_state

    new_region = {"bbox": bbox, "class": region_class, "page": int(current_page)}
    manual_regions_state = list(manual_regions_state) + [new_region]

    page_manuals = [r for r in manual_regions_state
                    if r.get("page") == int(current_page)]
    annotated, dets, info = _detect_and_draw(
        str(pdf_file), int(current_page), yolo_conf, page_manuals)
    info += f"\n\nAdded **{region_class}** at [{bbox[0]:.0f}, {bbox[1]:.0f}, {bbox[2]:.0f}, {bbox[3]:.0f}]"
    return annotated, info, dets, manual_regions_state


def review_clear_manual(pdf_file, current_page, yolo_conf, manual_regions_state):
    """Remove all manual regions for the current page."""
    if pdf_file is None:
        return None, "", {}, manual_regions_state
    kept = [r for r in manual_regions_state if r.get("page") != int(current_page)]
    annotated, dets, info = _detect_and_draw(
        str(pdf_file), int(current_page), yolo_conf, [])
    info += "  |  Manual regions cleared for this page."
    return annotated, info, dets, kept


def review_convert_with_corrections(pdf_file, quality_label, header_pct, footer_pct,
                                    language_label, engine_label, yolo_conf,
                                    manual_regions_state):
    """Convert the PDF merging manual corrections with auto-detections."""
    if pdf_file is None:
        return ("", "Upload a PDF first.", None, None,
                gr.update(visible=False), gr.update())
    pdf_path = str(pdf_file)
    quality = QUALITY_OPTIONS.get(quality_label, "balanced")
    languages = LANGUAGE_OPTIONS.get(language_label, "eng")
    engine = ENGINE_OPTIONS.get(engine_label, "tesseract")
    pipeline.ocr.primary_engine = engine

    # Build manual_regions dict: {page_num: [{"bbox": ..., "class": ...}]}
    mr_dict: dict = {}
    for r in manual_regions_state:
        pg = r.get("page", 0)
        mr_dict.setdefault(pg, []).append(
            {"bbox": r["bbox"], "class": r["class"]})

    result = pipeline.process_pdf_with_corrections(
        pdf_path, manual_regions=mr_dict,
        quality=quality,
        header_trim=header_pct, footer_trim=footer_pct,
        languages=languages, yolo_confidence=yolo_conf,
    )

    if result["success"]:
        text = result["text"]
        files = result["files"]
        meta = result["metadata"]
        mc = meta.get("manual_corrections", 0)
        engine_display = engine_label or "Tesseract (Default)"
        status = (
            f"**Conversion Complete (with {mc} manual corrections)!**\n\n"
            f"Pages: {meta.get('pages', 0)} | "
            f"Tables: {meta.get('tables', 0)} | "
            f"Figures: {meta.get('figures', 0)}\n"
            f"Language: {language_label} | Quality: {quality_label} | "
            f"Engine: {engine_display}"
        )
        original_name = os.path.basename(pdf_path)
        history.save_result(_LOCAL_USER, original_name, files, meta)
        txt_path = files.get("txt")
        docx_path = files.get("docx")
        entries = history.list_entries(_LOCAL_USER)
        hist_data = [[e["original_filename"], e["created_at"][:19], e["entry_id"]]
                     for e in entries[:20]]
        return (
            text, status,
            txt_path if txt_path and os.path.exists(txt_path) else None,
            docx_path if docx_path and os.path.exists(docx_path) else None,
            gr.update(visible=True),
            gr.update(value=hist_data),
        )
    else:
        return (f"Error: {result['error']}", f"**Error:** {result['error']}",
                None, None, gr.update(visible=False), gr.update())


# ══════════════════════════════════════════════════════════════════════════════
# Training / correction stats
# ══════════════════════════════════════════════════════════════════════════════
def get_correction_stats_md() -> str:
    """Return a Markdown summary of correction stats."""
    stats = pipeline.corrections.get_stats()
    total = stats["total_manual_corrections"]
    nxt = stats["next_retrain_at"]
    running = stats["retrain_running"]
    interval = stats["retrain_interval"]
    imgs = stats["images_count"]
    lbls = stats["labels_count"]
    history_entries = stats.get("retrain_history", [])

    md = (
        f"### Correction & Training Stats\n\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Manual corrections | **{total}** |\n"
        f"| Next retrain at | **{nxt}** ({nxt - total} remaining) |\n"
        f"| Retrain interval | {interval} |\n"
        f"| Retrain running | {'Yes' if running else 'No'} |\n"
        f"| Training images | {imgs} |\n"
        f"| Training labels | {lbls} |\n"
    )
    if history_entries:
        md += "\n### Retrain History (last 5)\n\n"
        md += "| Time | Corrections | Status |\n|---|---|---|\n"
        for h in history_entries:
            md += (f"| {h.get('timestamp', '?')[:19]} "
                   f"| {h.get('corrections_used', '?')} "
                   f"| {h.get('status', '?')} |\n")
    return md


def get_corrections_log_md() -> str:
    """Return recent corrections as Markdown table."""
    records = pipeline.corrections.get_corrections_log(limit=30)
    if not records:
        return "No manual corrections logged yet."
    md = "| # | PDF | Page | Class | BBox | Time |\n|---|---|---|---|---|---|\n"
    for i, r in enumerate(records, 1):
        bbox_str = ", ".join(f"{v:.0f}" for v in r.get("bbox", []))
        md += (f"| {i} | {r.get('pdf_name', '?')} | {r.get('page', '?')} "
               f"| {r.get('class', '?')} | [{bbox_str}] "
               f"| {r.get('timestamp', '?')[:19]} |\n")
    return md


# ══════════════════════════════════════════════════════════════════════════════
# History handlers (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
def download_docx_from_history(entry_id_input):
    if not entry_id_input or not entry_id_input.strip():
        return None
    return history.get_file_path(_LOCAL_USER, entry_id_input.strip(), "docx")


def download_txt_from_history(entry_id_input):
    if not entry_id_input or not entry_id_input.strip():
        return None
    return history.get_file_path(_LOCAL_USER, entry_id_input.strip(), "txt")


def refresh_history():
    history.cleanup_old_entries(_LOCAL_USER)
    entries = history.list_entries(_LOCAL_USER)
    return [[e["original_filename"], e["created_at"][:19], e["entry_id"]]
            for e in entries[:20]]


# ══════════════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════════════
def create_interface():
    theme = gr.themes.Soft(
        primary_hue="violet", secondary_hue="purple", neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    )

    with gr.Blocks(title="PDF OCR Pipeline", theme=theme) as app:

        gr.HTML("""
        <style>
        .gradio-container { max-width: 1400px !important; margin: auto !important; }
        .hero-bar {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 40%, #a855f7 100%);
            padding: 24px 32px; border-radius: 16px; margin-bottom: 24px;
            box-shadow: 0 8px 32px rgba(79,70,229,0.3);
            display: flex; align-items: center; justify-content: space-between;
        }
        .hero-bar h1 { color: #fff; font-size: 1.6rem; font-weight: 800; margin: 0; }
        .hero-bar p  { color: rgba(255,255,255,0.85); font-size: 0.95rem; margin: 4px 0 0 0; }
        .hero-badge { background: rgba(255,255,255,0.18); padding: 6px 14px;
                      border-radius: 999px; color: #fff; font-size: 0.85rem; font-weight: 600; }
        .step-label {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: #fff; padding: 10px 18px; border-radius: 12px;
            font-weight: 700; font-size: 1rem; margin-bottom: 12px;
        }
        .step-label-green {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            color: #fff; padding: 10px 18px; border-radius: 12px;
            font-weight: 700; font-size: 1rem; margin-bottom: 12px;
        }
        .step-label-orange {
            background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
            color: #fff; padding: 10px 18px; border-radius: 12px;
            font-weight: 700; font-size: 1rem; margin-bottom: 12px;
        }
        .settings-card {
            background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
            padding: 16px; margin-bottom: 12px;
        }
        </style>
        """)

        # States
        current_page = gr.State(value=1)
        total_pages = gr.State(value=0)

        # Review tab states
        rv_current_page = gr.State(value=0)
        rv_total_pages = gr.State(value=0)
        rv_detections = gr.State(value={})
        rv_manual_regions = gr.State(value=[])

        # ── HERO BAR ─────────────────────────────────────────────────
        with gr.Row():
            gr.HTML("""
            <div class="hero-bar">
                <div>
                    <h1>PDF OCR Pipeline</h1>
                    <p>OpenCV | Tesseract | DocLayout-YOLO | HTML-first Export | Auto-Retrain</p>
                </div>
                <div class="hero-badge">v0.5.0 &middot; Trainable</div>
            </div>
            """)

        with gr.Tabs():
            # ── TAB 1: Quick Convert ─────────────────────────────
            with gr.Tab("Convert PDF"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="step-label">Step 1 — Upload PDF</div>')
                        pdf_input = gr.File(
                            label="Select PDF file",
                            file_types=[".pdf"],
                            type="filepath",
                            file_count="single",
                        )

                        gr.HTML('<div class="step-label">Step 2 — Settings</div>')

                        language_dd = gr.Dropdown(
                            choices=list(LANGUAGE_OPTIONS.keys()),
                            value="Thai + English",
                            label="OCR Language",
                            info="Select the language(s) in the document",
                        )
                        quality_dd = gr.Dropdown(
                            choices=list(QUALITY_OPTIONS.keys()),
                            value="Balanced (Recommended)",
                            label="Quality Level",
                        )

                        with gr.Accordion("Advanced Settings", open=False):
                            engine_dd = gr.Dropdown(
                                choices=list(ENGINE_OPTIONS.keys()),
                                value="Tesseract (Default)",
                                label="OCR Engine",
                                info="Tesseract is best for most documents",
                            )
                            yolo_conf_sl = gr.Slider(
                                0.05, 0.50, 0.15, step=0.05,
                                label="Layout Detection Confidence",
                                info="Lower = detect more regions (tables, figures). "
                                     "Try 0.10 if tables are missed.",
                            )
                            with gr.Row():
                                header_sl = gr.Slider(0, 25, 0, step=1,
                                                      label="Header Trim %")
                                footer_sl = gr.Slider(0, 25, 0, step=1,
                                                      label="Footer Trim %")

                        gr.HTML('<div class="step-label">Step 3 — Convert</div>')
                        convert_btn = gr.Button("Convert to DOCX",
                                                variant="primary", size="lg")
                        status_md = gr.Markdown("")

                    with gr.Column(scale=1):
                        gr.HTML('<div class="step-label">Document Preview</div>')
                        preview_img = gr.Image(label="Preview",
                                               interactive=False, height=500)
                        with gr.Row(visible=False) as page_controls:
                            prev_btn = gr.Button("Prev", size="sm")
                            page_label = gr.Textbox(
                                value=_PAGE_ZERO, interactive=False,
                                show_label=False, container=False,
                            )
                            next_btn = gr.Button("Next", size="sm")

                with gr.Group(visible=False) as download_section:
                    gr.HTML('<div class="step-label-green">Step 4 — Download</div>')
                    with gr.Row():
                        dl_txt = gr.File(label="Text (.txt)", interactive=False)
                        dl_docx = gr.File(label="Word (.docx)", interactive=False)

                with gr.Accordion("View Extracted Text", open=False):
                    text_output = gr.Textbox(
                        label="Extracted Text",
                        placeholder="Converted text appears here...",
                        lines=12, max_lines=30, interactive=False,
                    )

            # ── TAB 2: Review & Correct ──────────────────────────
            with gr.Tab("Review & Correct"):
                gr.Markdown(
                    "Upload a PDF, review auto-detected regions, manually "
                    "add **table** or **figure** boxes, then convert with "
                    "corrections.  Every manual correction is logged — "
                    "the model auto-retrains every "
                    f"**{pipeline.corrections.retrain_interval}** corrections."
                )
                with gr.Row():
                    # ── Left: Controls ──
                    with gr.Column(scale=1):
                        gr.HTML('<div class="step-label">Step 1 — Load PDF</div>')
                        rv_pdf_input = gr.File(
                            label="Select PDF file",
                            file_types=[".pdf"], type="filepath",
                            file_count="single",
                        )

                        gr.HTML('<div class="step-label">Step 2 — Review Detections</div>')
                        rv_yolo_conf = gr.Slider(
                            0.05, 0.50, 0.15, step=0.05,
                            label="Detection Confidence",
                        )
                        with gr.Row(visible=False) as rv_page_controls:
                            rv_prev_btn = gr.Button("< Prev Page", size="sm")
                            rv_page_lbl = gr.Textbox(
                                value="Page 1 / ?", interactive=False,
                                show_label=False, container=False,
                            )
                            rv_next_btn = gr.Button("Next Page >", size="sm")

                        rv_info_md = gr.Markdown("")

                        gr.HTML('<div class="step-label-orange">'
                                'Step 3 — Add Manual Regions</div>')
                        gr.Markdown(
                            "Look at the page image on the right. The pixel "
                            "coordinates start at **(0,0) = top-left**. "
                            "Enter bounding-box coordinates below."
                        )
                        with gr.Row():
                            rv_x0 = gr.Number(label="x0 (left)", value=0)
                            rv_y0 = gr.Number(label="y0 (top)", value=0)
                        with gr.Row():
                            rv_x1 = gr.Number(label="x1 (right)", value=100)
                            rv_y1 = gr.Number(label="y1 (bottom)", value=100)
                        rv_class_dd = gr.Dropdown(
                            choices=CLASS_OPTIONS, value="table",
                            label="Region Type",
                        )
                        with gr.Row():
                            rv_add_btn = gr.Button("Add Region",
                                                   variant="primary", size="sm")
                            rv_clear_btn = gr.Button("Clear Page Regions",
                                                     variant="secondary", size="sm")

                        gr.HTML('<div class="step-label-green">'
                                'Step 4 — Convert with Corrections</div>')

                        rv_lang_dd = gr.Dropdown(
                            choices=list(LANGUAGE_OPTIONS.keys()),
                            value="Thai + English", label="OCR Language",
                        )
                        rv_quality_dd = gr.Dropdown(
                            choices=list(QUALITY_OPTIONS.keys()),
                            value="Balanced (Recommended)", label="Quality",
                        )
                        with gr.Accordion("Advanced", open=False):
                            rv_engine_dd = gr.Dropdown(
                                choices=list(ENGINE_OPTIONS.keys()),
                                value="Tesseract (Default)", label="Engine",
                            )
                            with gr.Row():
                                rv_header_sl = gr.Slider(0, 25, 0, step=1,
                                                         label="Header Trim %")
                                rv_footer_sl = gr.Slider(0, 25, 0, step=1,
                                                         label="Footer Trim %")

                        rv_convert_btn = gr.Button(
                            "Convert with Corrections",
                            variant="primary", size="lg",
                        )
                        rv_status_md = gr.Markdown("")

                    # ── Right: Image preview ──
                    with gr.Column(scale=1):
                        gr.HTML('<div class="step-label">Detected Regions</div>')
                        rv_preview_img = gr.Image(
                            label="Page with detections",
                            interactive=False, height=600,
                        )

                # Download row
                with gr.Group(visible=False) as rv_download_section:
                    gr.HTML('<div class="step-label-green">Download</div>')
                    with gr.Row():
                        rv_dl_txt = gr.File(label="Text (.txt)", interactive=False)
                        rv_dl_docx = gr.File(label="Word (.docx)",
                                             interactive=False)
                with gr.Accordion("View Extracted Text", open=False):
                    rv_text_output = gr.Textbox(
                        label="Extracted Text",
                        placeholder="Converted text appears here...",
                        lines=12, max_lines=30, interactive=False,
                    )

            # ── TAB 3: Training / Corrections ────────────────────
            with gr.Tab("Training"):
                gr.HTML('<div class="step-label">Auto-Retrain Dashboard</div>')
                gr.Markdown(
                    "Manual corrections are logged in YOLO format. When the "
                    f"count reaches a multiple of "
                    f"**{pipeline.corrections.retrain_interval}**, the layout "
                    "model is automatically fine-tuned in the background."
                )
                training_stats_md = gr.Markdown(get_correction_stats_md)
                refresh_training_btn = gr.Button("Refresh Stats", size="sm")
                refresh_training_btn.click(
                    fn=get_correction_stats_md,
                    outputs=[training_stats_md],
                )

                with gr.Accordion("Corrections Log (recent 30)", open=False):
                    corrections_log_md = gr.Markdown(get_corrections_log_md)
                    refresh_log_btn = gr.Button("Refresh Log", size="sm")
                    refresh_log_btn.click(
                        fn=get_corrections_log_md,
                        outputs=[corrections_log_md],
                    )

            # ── TAB 4: History ───────────────────────────────────
            with gr.Tab("History"):
                gr.HTML('<div class="step-label">Processing History</div>')
                refresh_btn = gr.Button("Refresh", size="sm")
                history_table = gr.Dataframe(
                    headers=["Filename", "Date", "Entry ID"],
                    datatype=["str", "str", "str"],
                    interactive=False,
                )
                gr.HTML("<p style='color:#64748b;font-size:0.9rem;'>"
                        "Paste an Entry ID from the table above, then click a "
                        "download button to retrieve your saved files.</p>")
                with gr.Row():
                    entry_id_input = gr.Textbox(
                        label="Entry ID",
                        placeholder="Paste entry ID from table above...",
                        scale=3,
                    )
                with gr.Row():
                    dl_history_docx_btn = gr.Button("Download DOCX",
                                                    variant="primary", scale=1)
                    dl_history_txt_btn  = gr.Button("Download TXT",
                                                    variant="secondary", scale=1)
                with gr.Row():
                    dl_history_docx_file = gr.File(
                        label="Word Document (.docx)", interactive=False)
                    dl_history_txt_file  = gr.File(
                        label="Text File (.txt)", interactive=False)

            # ── TAB 5: Settings ──────────────────────────────────
            with gr.Tab("Settings"):
                gr.HTML('<div class="step-label">Configuration</div>')
                gr.Markdown(
                    "These settings can also be set via the `.env` file. "
                    "Values chosen here apply to this session only."
                )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### OCR Configuration")
                        settings_lang = gr.Dropdown(
                            choices=list(LANGUAGE_OPTIONS.keys()),
                            value="Thai + English",
                            label="Default OCR Language",
                            info="Applies when you press Convert (also changeable on Convert tab)",
                        )
                        settings_engine = gr.Dropdown(
                            choices=list(ENGINE_OPTIONS.keys()),
                            value="Tesseract (Default)",
                            label="Default OCR Engine",
                        )
                        settings_quality = gr.Dropdown(
                            choices=list(QUALITY_OPTIONS.keys()),
                            value="Balanced (Recommended)",
                            label="Default Quality",
                        )

                    with gr.Column():
                        gr.Markdown("### Layout Detection")
                        settings_yolo_conf = gr.Slider(
                            0.05, 0.50, 0.15, step=0.05,
                            label="YOLO Confidence Threshold",
                            info="Lower value detects more regions (tables, figures, text blocks). "
                                 "Increase if too many false positives appear.",
                        )
                        gr.Markdown("### Installed Engines")
                        engines = pipeline.ocr.get_available_engines()
                        engine_info = "\n".join(
                            f"- **{name}**: {'Installed' if ok else 'Not installed'}"
                            for name, ok in engines.items()
                        )
                        gr.Markdown(engine_info)
                        gr.Markdown(
                            f"- **DocLayout-YOLO**: "
                            f"{'Loaded' if pipeline.layout.model_loaded else 'Not loaded (OpenCV fallback)'}"
                        )

                # Sync settings ↔ convert tab
                settings_lang.change(
                    fn=lambda v: v, inputs=[settings_lang], outputs=[language_dd])
                settings_engine.change(
                    fn=lambda v: v, inputs=[settings_engine], outputs=[engine_dd])
                settings_quality.change(
                    fn=lambda v: v, inputs=[settings_quality], outputs=[quality_dd])
                settings_yolo_conf.change(
                    fn=lambda v: v, inputs=[settings_yolo_conf], outputs=[yolo_conf_sl])

            # ── TAB 6: System Status ─────────────────────────────
            with gr.Tab("System Status"):
                gr.HTML('<div class="step-label">Pipeline Status</div>')
                _status_text = json.dumps(pipeline.get_status(), indent=2)
                status_info = gr.Textbox(label="Status", value=_status_text,
                                         lines=12, interactive=False)
                refresh_status = gr.Button("Refresh", size="sm")
                refresh_status.click(
                    fn=lambda: json.dumps(pipeline.get_status(), indent=2),
                    outputs=[status_info],
                )

        gr.HTML("""
        <div style="text-align:center;padding:20px;margin-top:20px;
                    border-top:1px solid #e2e8f0;color:#94a3b8;font-size:0.85rem;">
            PDF OCR Pipeline v0.5.0 — Apache-2.0 License — Auto-retrain enabled
        </div>
        """)

        # ══════════════════════════════════════════════════════════════
        # Wiring — Convert tab
        # ══════════════════════════════════════════════════════════════

        pdf_input.change(
            fn=load_pdf_preview,
            inputs=[pdf_input, quality_dd, header_sl, footer_sl],
            outputs=[preview_img, page_label, current_page, total_pages,
                     page_controls],
        )

        prev_btn.click(
            fn=lambda f, c, t, h, fo: change_page(f, -1, c, t, h, fo),
            inputs=[pdf_input, current_page, total_pages, header_sl, footer_sl],
            outputs=[preview_img, page_label, current_page],
        )
        next_btn.click(
            fn=lambda f, c, t, h, fo: change_page(f, 1, c, t, h, fo),
            inputs=[pdf_input, current_page, total_pages, header_sl, footer_sl],
            outputs=[preview_img, page_label, current_page],
        )

        convert_btn.click(
            fn=lambda: "**Processing...** This may take a moment.",
            outputs=[status_md],
        ).then(
            fn=process_document,
            inputs=[pdf_input, quality_dd, header_sl, footer_sl,
                    language_dd, engine_dd, yolo_conf_sl],
            outputs=[text_output, status_md, dl_txt, dl_docx,
                     download_section, history_table],
        )

        # ══════════════════════════════════════════════════════════════
        # Wiring — Review & Correct tab
        # ══════════════════════════════════════════════════════════════

        def _rv_load(pdf_file, yolo_conf):
            (img, info, pg, total, dets, manuals,
             vis) = review_load_pdf(pdf_file, yolo_conf)
            pg_lbl = f"Page 1 / {total}" if total else "Page 0 / 0"
            return (img, info, pg, total, dets, manuals, vis, pg_lbl)

        rv_pdf_input.change(
            fn=_rv_load,
            inputs=[rv_pdf_input, rv_yolo_conf],
            outputs=[rv_preview_img, rv_info_md, rv_current_page,
                     rv_total_pages, rv_detections, rv_manual_regions,
                     rv_page_controls, rv_page_lbl],
        )

        def _rv_prev(pdf, pg, total, conf, manuals):
            img, info, new_pg, dets, manuals = review_change_page(
                pdf, -1, pg, total, conf, manuals)
            lbl = f"Page {new_pg + 1} / {total}"
            return img, info, new_pg, dets, manuals, lbl

        def _rv_next(pdf, pg, total, conf, manuals):
            img, info, new_pg, dets, manuals = review_change_page(
                pdf, 1, pg, total, conf, manuals)
            lbl = f"Page {new_pg + 1} / {total}"
            return img, info, new_pg, dets, manuals, lbl

        rv_prev_btn.click(
            fn=_rv_prev,
            inputs=[rv_pdf_input, rv_current_page, rv_total_pages,
                    rv_yolo_conf, rv_manual_regions],
            outputs=[rv_preview_img, rv_info_md, rv_current_page,
                     rv_detections, rv_manual_regions, rv_page_lbl],
        )
        rv_next_btn.click(
            fn=_rv_next,
            inputs=[rv_pdf_input, rv_current_page, rv_total_pages,
                    rv_yolo_conf, rv_manual_regions],
            outputs=[rv_preview_img, rv_info_md, rv_current_page,
                     rv_detections, rv_manual_regions, rv_page_lbl],
        )

        rv_add_btn.click(
            fn=review_add_region,
            inputs=[rv_pdf_input, rv_current_page, rv_total_pages,
                    rv_yolo_conf, rv_x0, rv_y0, rv_x1, rv_y1,
                    rv_class_dd, rv_detections, rv_manual_regions],
            outputs=[rv_preview_img, rv_info_md, rv_detections,
                     rv_manual_regions],
        )

        rv_clear_btn.click(
            fn=review_clear_manual,
            inputs=[rv_pdf_input, rv_current_page, rv_yolo_conf,
                    rv_manual_regions],
            outputs=[rv_preview_img, rv_info_md, rv_detections,
                     rv_manual_regions],
        )

        rv_convert_btn.click(
            fn=lambda: "**Processing with corrections...** This may take a moment.",
            outputs=[rv_status_md],
        ).then(
            fn=review_convert_with_corrections,
            inputs=[rv_pdf_input, rv_quality_dd, rv_header_sl, rv_footer_sl,
                    rv_lang_dd, rv_engine_dd, rv_yolo_conf,
                    rv_manual_regions],
            outputs=[rv_text_output, rv_status_md, rv_dl_txt, rv_dl_docx,
                     rv_download_section, history_table],
        )

        # ══════════════════════════════════════════════════════════════
        # Wiring — History tab
        # ══════════════════════════════════════════════════════════════
        refresh_btn.click(fn=refresh_history, outputs=[history_table])
        dl_history_docx_btn.click(
            fn=download_docx_from_history,
            inputs=[entry_id_input], outputs=[dl_history_docx_file])
        dl_history_txt_btn.click(
            fn=download_txt_from_history,
            inputs=[entry_id_input], outputs=[dl_history_txt_file])

    return app


# ══════════════════════════════════════════════════════════════════════════════
# Launch
# ══════════════════════════════════════════════════════════════════════════════
def main():
    app = create_interface()
    port = int(os.getenv("SERVER_PORT", "7870"))
    host = os.getenv("SERVER_HOST", "127.0.0.1")
    share = os.getenv("SHARE_GRADIO", "false").lower() == "true"
    app.launch(server_name=host, server_port=port, share=share, show_error=True)


if __name__ == "__main__":
    main()
