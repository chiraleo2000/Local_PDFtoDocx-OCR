"""
PDF to DOCX OCR Service — Gradio Web Application
v0.1.0-dev
"""
import os
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

import gradio as gr

# ── Fix Gradio 4.44.x bug: boolean JSON schemas crash API info parser ─────────
import gradio_client.utils as _gc_utils
_orig_json_schema_fn = _gc_utils._json_schema_to_python_type

def _patched_json_schema_to_python_type(schema, defs=None):
    if isinstance(schema, bool):
        return "Any" if schema else "None"
    return _orig_json_schema_fn(schema, defs)

_gc_utils._json_schema_to_python_type = _patched_json_schema_to_python_type
# ───────────────────────────────────────────────────────────────────────────────

from src.pipeline import OCRPipeline
from src.services import AuthManager, HistoryManager

# ── Initialise services ──────────────────────────────────────────────────────
pipeline = OCRPipeline()
auth = AuthManager()
history = HistoryManager()
history.cleanup_old_entries()

QUALITY_OPTIONS = {
    "Standard (Fast)": "fast",
    "Balanced (Recommended)": "balanced",
    "Best (Accurate)": "accurate",
}

PAGE_ZERO_LABEL = "Page 0 / 0"


# ══════════════════════════════════════════════════════════════════════════════
# PDF Preview
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
# Auth handlers
# ══════════════════════════════════════════════════════════════════════════════
def handle_login(username, password, session_state):
    result = auth.login(username, password)
    if result["success"]:
        return (
            result["token"], result["username"],
            gr.update(visible=False), gr.update(visible=True),
            f"Welcome, **{result['username']}**!",
            gr.update(value=f"User: {result['username']}", visible=True),
        )
    return ("", "", gr.update(visible=True), gr.update(visible=False),
            f"Error: {result['error']}", gr.update(visible=False))


def handle_register(username, password, session_state):
    result = auth.register(username, password)
    if result["success"]:
        return f"{result['message']} — You can now log in."
    return f"Error: {result['error']}"


def handle_logout(session_token):
    auth.logout(session_token)
    return ("", "", gr.update(visible=True), gr.update(visible=False),
            "", gr.update(visible=False))


# ══════════════════════════════════════════════════════════════════════════════
# PDF handlers
# ══════════════════════════════════════════════════════════════════════════════
def load_pdf_preview(pdf_file, quality, header_pct, footer_pct):
    if pdf_file is None:
        return None, PAGE_ZERO_LABEL, 0, 0, gr.update(visible=False)
    pdf_path = pdf_file
    img, total = render_page_preview(pdf_path, 0, 1.5, header_pct, footer_pct)
    if img is not None:
        return img, f"Page 1 / {total}", 1, total, gr.update(visible=True)
    return None, PAGE_ZERO_LABEL, 0, 0, gr.update(visible=False)


def change_page(pdf_file, direction, current, total, header_pct, footer_pct):
    if pdf_file is None or total == 0:
        return None, f"Page {current} / {total}", current
    pdf_path = pdf_file
    new_page = max(1, min(current + direction, total))
    img, _ = render_page_preview(pdf_path, new_page - 1, 1.5, header_pct, footer_pct)
    return img, f"Page {new_page} / {total}", new_page


def process_document(pdf_file, quality_label, header_pct, footer_pct,
                     session_token, username_state):
    if pdf_file is None:
        return ("", "Please upload a PDF file first.",
                None, None, gr.update(visible=False), gr.update())

    pdf_path = pdf_file
    quality = QUALITY_OPTIONS.get(quality_label, "balanced")

    result = pipeline.process_pdf(
        pdf_path, quality=quality,
        header_trim=header_pct, footer_trim=footer_pct,
    )

    if result["success"]:
        text = result["text"]
        files = result["files"]
        meta = result["metadata"]

        status = (
            f"**Conversion Complete!**\n\n"
            f"Pages: {meta.get('pages', 0)} | "
            f"Tables: {meta.get('tables', 0)} | "
            f"Figures: {meta.get('figures', 0)}\n"
            f"Quality: {quality_label}"
        )

        original_name = os.path.basename(pdf_path)
        username = auth.validate_session(session_token) or username_state or "anonymous"
        history.save_result(username, original_name, files, meta)

        txt_path = files.get("txt")
        docx_path = files.get("docx")

        entries = history.list_entries(username)
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


def download_from_history(username_state, session_token, entry_id_input):
    username = auth.validate_session(session_token) or username_state or "anonymous"
    if not entry_id_input:
        return None
    return history.get_file_path(username, entry_id_input.strip(), "docx")


def refresh_history(username_state, session_token):
    username = auth.validate_session(session_token) or username_state or "anonymous"
    history.cleanup_old_entries(username)
    entries = history.list_entries(username)
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
        .auth-card { max-width: 400px; margin: 60px auto; padding: 36px;
                     border: 1px solid #e2e8f0; border-radius: 20px;
                     background: #fff; box-shadow: 0 16px 48px rgba(0,0,0,0.08); }
        .auth-card h2 { text-align: center; margin-bottom: 20px; }
        </style>
        """)

        # State
        session_token = gr.State(value="")
        username_state = gr.State(value="")
        current_page = gr.State(value=1)
        total_pages = gr.State(value=0)

        # ── LOGIN ────────────────────────────────────────────────────
        with gr.Column(visible=True) as login_section:
            gr.HTML("""
            <div style="text-align:center; padding:30px 0 10px 0;">
                <h1 style="font-size:2rem; background:linear-gradient(135deg,#4f46e5,#a855f7);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; font-weight:800;">
                    PDF OCR Pipeline
                </h1>
                <p style="color:#64748b; font-size:1rem;">
                    Open-Source | GPU-Accelerated | Multi-Language
                </p>
            </div>
            """)
            with gr.Column(elem_classes=["auth-card"]):
                gr.HTML("<h2>Sign In</h2>")
                gr.HTML("<p style='text-align:center;color:#64748b;font-size:0.9rem;'>"
                        "Default: guest / guest123</p>")
                login_user = gr.Textbox(label="Username", placeholder="Enter username")
                login_pass = gr.Textbox(label="Password", placeholder="Enter password",
                                        type="password")
                login_btn = gr.Button("Sign In", variant="primary", size="lg")
                login_msg = gr.Markdown("")
                gr.HTML("<hr style='margin:20px 0;border:none;border-top:1px solid #e2e8f0;'>")
                gr.HTML("<p style='text-align:center;color:#64748b;'>Create Account</p>")
                reg_user = gr.Textbox(label="New Username", placeholder="Choose username")
                reg_pass = gr.Textbox(label="New Password", placeholder="Min 4 chars",
                                      type="password")
                reg_btn = gr.Button("Register", variant="secondary")
                reg_msg = gr.Markdown("")

        # ── MAIN ─────────────────────────────────────────────────────
        with gr.Column(visible=False) as main_section:
            with gr.Row():
                gr.HTML("""
                <div class="hero-bar">
                    <div>
                        <h1>PDF OCR Pipeline</h1>
                        <p>OpenCV | Tesseract | PaddleOCR | DocLayout-YOLO | DOCX Export</p>
                    </div>
                    <div class="hero-badge">v0.1.0-dev</div>
                </div>
                """)
            user_badge = gr.Markdown("", visible=False)
            logout_btn = gr.Button("Sign Out", size="sm", variant="secondary")

            with gr.Tabs():
                # ── TAB: Convert ─────────────────────────────────────
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
                            quality_dd = gr.Dropdown(
                                choices=list(QUALITY_OPTIONS.keys()),
                                value="Balanced (Recommended)",
                                label="Quality Level",
                            )
                            with gr.Accordion("Header/Footer Trim", open=False):
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
                                    value=PAGE_ZERO_LABEL, interactive=False,
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

                # ── TAB: History ─────────────────────────────────────
                with gr.Tab("History"):
                    gr.HTML('<div class="step-label">Processing History</div>')
                    refresh_btn = gr.Button("Refresh", size="sm")
                    history_table = gr.Dataframe(
                        headers=["Filename", "Date", "Entry ID"],
                        datatype=["str", "str", "str"],
                        interactive=False,
                    )
                    with gr.Row():
                        entry_id_input = gr.Textbox(label="Entry ID",
                                                    placeholder="Paste entry ID...")
                        dl_history_btn = gr.Button("Download DOCX", variant="primary")
                    dl_history_file = gr.File(label="Downloaded file", interactive=False)

                # ── TAB: System ──────────────────────────────────────
                with gr.Tab("System Status"):
                    gr.HTML('<div class="step-label">Pipeline Status</div>')
                    import json as _json
                    _status_text = _json.dumps(pipeline.get_status(), indent=2)
                    status_info = gr.Textbox(label="Status", value=_status_text,
                                             lines=10, interactive=False)
                    refresh_status = gr.Button("Refresh", size="sm")
                    refresh_status.click(
                        fn=lambda: _json.dumps(pipeline.get_status(), indent=2),
                        outputs=[status_info],
                    )

            gr.HTML("""
            <div style="text-align:center;padding:20px;margin-top:20px;
                        border-top:1px solid #e2e8f0;color:#94a3b8;font-size:0.85rem;">
                PDF OCR Pipeline v0.1.0-dev — Apache-2.0 License
            </div>
            """)

        # ── Event wiring ─────────────────────────────────────────────
        login_btn.click(
            fn=handle_login,
            inputs=[login_user, login_pass, session_token],
            outputs=[session_token, username_state, login_section, main_section,
                     login_msg, user_badge],
        )
        reg_btn.click(fn=handle_register,
                      inputs=[reg_user, reg_pass, session_token],
                      outputs=[reg_msg])
        logout_btn.click(
            fn=handle_logout, inputs=[session_token],
            outputs=[session_token, username_state, login_section, main_section,
                     login_msg, user_badge],
        )

        pdf_input.change(
            fn=load_pdf_preview,
            inputs=[pdf_input, quality_dd, header_sl, footer_sl],
            outputs=[preview_img, page_label, current_page, total_pages, page_controls],
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
                    session_token, username_state],
            outputs=[text_output, status_md, dl_txt, dl_docx,
                     download_section, history_table],
        )

        refresh_btn.click(
            fn=refresh_history,
            inputs=[username_state, session_token],
            outputs=[history_table],
        )
        dl_history_btn.click(
            fn=download_from_history,
            inputs=[username_state, session_token, entry_id_input],
            outputs=[dl_history_file],
        )

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
