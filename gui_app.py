"""
PDF to DOCX OCR — Desktop GUI Application  (v3.0)
Modern dark-themed Tkinter GUI (no Docker required).
First-run: checks/installs dependencies → launches main OCR UI.

Usage:
    python gui_app.py
    py -3 gui_app.py
"""
import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
import logging
import shutil
import subprocess
import importlib
import time

# ── Ensure project root is on sys.path ────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# .env loading (best-effort)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=False)
except ImportError:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("GUI")

_INSTALLED_MARKER = os.path.join(_PROJECT_ROOT, ".installed_ok")


# ══════════════════════════════════════════════════════════════════════════════
# Color palette — Dark theme with vibrant accents
# ══════════════════════════════════════════════════════════════════════════════
class Theme:
    """Centralized colour constants for the entire UI."""
    # Base surfaces
    BG_DARK       = "#1a1b2e"    # deep navy
    BG_MID        = "#232440"    # card / panel
    BG_LIGHT      = "#2d2f52"    # inputs / hover
    BG_SURFACE    = "#353764"    # elevated card

    # Accent colours
    ACCENT        = "#6c63ff"    # primary purple
    ACCENT_HOVER  = "#857dff"
    ACCENT_GLOW   = "#7b73ff"
    SUCCESS       = "#00d68f"    # green
    WARNING       = "#ffaa00"    # amber
    ERROR         = "#ff3d71"    # red/pink
    INFO          = "#00b4d8"    # cyan

    # Text
    TEXT          = "#eaeaff"    # primary text
    TEXT_DIM      = "#9899c2"    # secondary
    TEXT_MUTED    = "#6b6d99"    # placeholders

    # Borders
    BORDER        = "#3d3f6e"
    BORDER_FOCUS  = "#6c63ff"

    # Special
    CANVAS_BG     = "#111225"
    LOG_BG        = "#12132a"
    SCROLLBAR     = "#444677"

    # Gradient simulation
    HEADER_LEFT   = "#6c63ff"
    HEADER_RIGHT  = "#00b4d8"


# ══════════════════════════════════════════════════════════════════════════════
# Option maps
# ══════════════════════════════════════════════════════════════════════════════
QUALITY_OPTIONS = {
    "⚡ Standard (Fast)": "fast",
    "⚖ Balanced (Recommended)": "balanced",
    "🎯 Best (Accurate)": "accurate",
}

LANGUAGE_OPTIONS = {
    "🇬🇧 English": "eng",
    "🇹🇭 Thai": "tha",
    "🇹🇭 Thai + 🇬🇧 English": "tha+eng",
    "🇨🇳 Chinese (Simplified)": "chi_sim",
    "🇨🇳 Chinese + 🇬🇧 English": "chi_sim+eng",
    "🇯🇵 Japanese": "jpn",
    "🇯🇵 Japanese + 🇬🇧 English": "jpn+eng",
    "🇰🇷 Korean": "kor",
    "🇰🇷 Korean + 🇬🇧 English": "kor+eng",
    "🇸🇦 Arabic": "ara",
    "🔍 Auto-detect": "auto",
}

ENGINE_OPTIONS = {
    "Tesseract (Thai+English)": "tesseract",
    "EasyOCR (Thai+English)": "easyocr",
    "Thai-TrOCR (Line-level)": "thai_trocr",
    "PaddleOCR (Multilingual)": "paddleocr",
    "Typhoon OCR 3B (GPU LLM)": "typhoon",
}

PAGE_SIZE_OPTIONS = ["A4", "Letter", "Legal", "A3", "B5"]

MARGIN_OPTIONS = {
    "Normal (1\" all sides)": "Normal",
    "Narrow (0.5\" all sides)": "Narrow",
    "Moderate (0.75\" left/right)": "Moderate",
    "Wide (1.5\" left/right)": "Wide",
}


# ══════════════════════════════════════════════════════════════════════════════
# Dependency checker / installer
# ══════════════════════════════════════════════════════════════════════════════
REQUIRED_PACKAGES = [
    ("python-dotenv", "dotenv"),
    ("Pillow", "PIL"),
    ("numpy", "numpy"),
    ("opencv-python-headless", "cv2"),
    ("PyMuPDF", "fitz"),
    ("python-docx", "docx"),
    ("beautifulsoup4", "bs4"),
    ("lxml", "lxml"),
    ("htmldocx", "htmldocx"),
    ("requests", "requests"),
    ("easyocr", "easyocr"),
    ("paddleocr", "paddleocr"),
    ("pytesseract", "pytesseract"),
    ("onnxruntime", "onnxruntime"),
    ("transformers", "transformers"),
    ("doclayout-yolo", "doclayout_yolo"),
    ("huggingface_hub", "huggingface_hub"),
]


def _check_missing_packages():
    missing = []
    for pip_name, import_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
        except (ImportError, Exception):
            missing.append((pip_name, import_name))
    return missing


def _is_first_run():
    return not os.path.exists(_INSTALLED_MARKER)


def _mark_installed():
    with open(_INSTALLED_MARKER, "w") as f:
        f.write(f"installed {time.strftime('%Y-%m-%d %H:%M:%S')}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Rounded-rectangle helper for Canvas
# ══════════════════════════════════════════════════════════════════════════════
def _rounded_rect(canvas, x1, y1, x2, y2, radius=12, **kwargs):
    """Draw a rounded rectangle on a tk.Canvas."""
    pts = [
        x1 + radius, y1,
        x2 - radius, y1,
        x2, y1,
        x2, y1 + radius,
        x2, y2 - radius,
        x2, y2,
        x2 - radius, y2,
        x1 + radius, y2,
        x1, y2,
        x1, y2 - radius,
        x1, y1 + radius,
        x1, y1,
    ]
    return canvas.create_polygon(pts, smooth=True, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# First-Run Setup Wizard  (dark-themed)
# ══════════════════════════════════════════════════════════════════════════════
class SetupWizard(tk.Tk):
    """First-run installer dialog with dark theme."""

    def __init__(self):
        super().__init__()
        self.title("LocalOCR — First-Time Setup")
        self.geometry("700x600")
        self.resizable(False, False)
        self.configure(bg=Theme.BG_DARK)
        self._cancelled = False
        self._install_done = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # ── Gradient Header ──────────────────────────────────────────────
        hdr = tk.Canvas(self, height=80, bg=Theme.BG_DARK, highlightthickness=0)
        hdr.pack(fill="x")
        # Simulate gradient with overlapping rectangles
        for i in range(80):
            frac = i / 80
            r = int(108 * (1 - frac) + 0 * frac)
            g = int(99 * (1 - frac) + 180 * frac)
            b = int(255 * (1 - frac) + 216 * frac)
            colour = f"#{r:02x}{g:02x}{b:02x}"
            hdr.create_line(0, i, 700, i, fill=colour)
        hdr.create_text(24, 28, text="🚀 LocalOCR Setup", anchor="w",
                        fill="white", font=("Segoe UI", 20, "bold"))
        hdr.create_text(24, 56, text="First-time dependency installation",
                        anchor="w", fill="#d0d0ff", font=("Segoe UI", 11))

        # ── Install path ─────────────────────────────────────────────────
        path_frame = tk.Frame(self, bg=Theme.BG_MID, padx=16, pady=12)
        path_frame.pack(fill="x", padx=16, pady=(12, 4))

        tk.Label(path_frame, text="📂 Application Path",
                 bg=Theme.BG_MID, fg=Theme.ACCENT_GLOW,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        self._var_path = tk.StringVar(value=_PROJECT_ROOT)
        path_entry = tk.Entry(path_frame, textvariable=self._var_path,
                              state="readonly", font=("Consolas", 10),
                              bg=Theme.BG_LIGHT, fg=Theme.TEXT,
                              relief="flat", bd=0,
                              readonlybackground=Theme.BG_LIGHT,
                              insertbackground=Theme.TEXT)
        path_entry.pack(fill="x", pady=(4, 0), ipady=4)

        # ── Python info ──────────────────────────────────────────────────
        info_frame = tk.Frame(self, bg=Theme.BG_MID, padx=16, pady=10)
        info_frame.pack(fill="x", padx=16, pady=4)

        tk.Label(info_frame, text="🐍 Python Environment",
                 bg=Theme.BG_MID, fg=Theme.INFO,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        tk.Label(info_frame, text=f"  Executable: {sys.executable}",
                 bg=Theme.BG_MID, fg=Theme.TEXT_DIM,
                 font=("Consolas", 9)).pack(anchor="w")
        tk.Label(info_frame, text=f"  Version:    {sys.version.split()[0]}",
                 bg=Theme.BG_MID, fg=Theme.TEXT_DIM,
                 font=("Consolas", 9)).pack(anchor="w")

        # ── Log area ─────────────────────────────────────────────────────
        log_outer = tk.Frame(self, bg=Theme.BG_DARK)
        log_outer.pack(fill="both", expand=True, padx=16, pady=4)

        tk.Label(log_outer, text="📋 Installation Log",
                 bg=Theme.BG_DARK, fg=Theme.TEXT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 4))

        self._txt_log = scrolledtext.ScrolledText(
            log_outer, wrap="word", font=("Consolas", 9),
            height=12, state="disabled",
            bg=Theme.LOG_BG, fg=Theme.SUCCESS,
            insertbackground=Theme.SUCCESS,
            selectbackground=Theme.ACCENT,
            relief="flat", bd=0)
        self._txt_log.pack(fill="both", expand=True)

        # ── Progress bar ─────────────────────────────────────────────────
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Setup.Horizontal.TProgressbar",
                        troughcolor=Theme.BG_LIGHT,
                        background=Theme.ACCENT,
                        darkcolor=Theme.ACCENT,
                        lightcolor=Theme.ACCENT_GLOW,
                        bordercolor=Theme.BG_DARK,
                        thickness=8)
        self._progress = ttk.Progressbar(
            self, mode="indeterminate", length=300,
            style="Setup.Horizontal.TProgressbar")
        self._progress.pack(padx=16, pady=6, fill="x")

        # ── Buttons ──────────────────────────────────────────────────────
        btn_frame = tk.Frame(self, bg=Theme.BG_DARK, pady=10)
        btn_frame.pack(fill="x", padx=16)

        self._btn_cancel = tk.Button(
            btn_frame, text="  Cancel  ", command=self._on_close,
            bg=Theme.BG_LIGHT, fg=Theme.TEXT_DIM,
            activebackground=Theme.ERROR, activeforeground="white",
            font=("Segoe UI", 10), relief="flat", bd=0, cursor="hand2")
        self._btn_cancel.pack(side="left", padx=4)

        self._btn_install = tk.Button(
            btn_frame, text="  🚀 Install & Launch  ",
            command=self._start_install,
            bg=Theme.ACCENT, fg="white",
            activebackground=Theme.ACCENT_HOVER, activeforeground="white",
            font=("Segoe UI", 11, "bold"), relief="flat", bd=0,
            cursor="hand2", padx=16, pady=6)
        self._btn_install.pack(side="right", padx=4)

        self._btn_skip = tk.Button(
            btn_frame, text="  Skip (Launch Anyway)  ",
            command=self._skip,
            bg=Theme.BG_SURFACE, fg=Theme.TEXT_DIM,
            activebackground=Theme.WARNING, activeforeground="black",
            font=("Segoe UI", 10), relief="flat", bd=0, cursor="hand2")
        self._btn_skip.pack(side="right", padx=4)

    def _log(self, msg, colour=None):
        self._txt_log.configure(state="normal")
        if colour:
            tag = f"c_{colour}"
            self._txt_log.tag_configure(tag, foreground=colour)
            self._txt_log.insert("end", msg + "\n", tag)
        else:
            self._txt_log.insert("end", msg + "\n")
        self._txt_log.see("end")
        self._txt_log.configure(state="disabled")
        self.update_idletasks()

    def _start_install(self):
        self._btn_install.configure(state="disabled")
        self._btn_skip.configure(state="disabled")
        self._progress.start(10)
        threading.Thread(target=self._run_install, daemon=True).start()

    def _run_install(self):
        try:
            self.after(0, lambda: self._log(
                "🔍 Checking installed packages...", Theme.INFO))

            missing = _check_missing_packages()
            if not missing:
                self.after(0, lambda: self._log(
                    "✅ All packages already installed!", Theme.SUCCESS))
            else:
                names = [p for p, _ in missing]
                self.after(0, lambda: self._log(
                    f"📦 Missing packages ({len(names)}): {', '.join(names)}",
                    Theme.WARNING))
                self.after(0, lambda: self._log(""))

                torch_needed = False
                try:
                    importlib.import_module("torch")
                except (ImportError, Exception):
                    torch_needed = True

                if torch_needed:
                    self.after(0, lambda: self._log(
                        "⬇ Installing PyTorch (CPU)… this may take a while.",
                        Theme.INFO))
                    self._pip_install(
                        ["torch", "torchvision",
                         "--index-url",
                         "https://download.pytorch.org/whl/cpu"])

                req_file = os.path.join(_PROJECT_ROOT, "requirements.txt")
                if os.path.exists(req_file):
                    self.after(0, lambda: self._log(
                        "⬇ Installing from requirements.txt...", Theme.INFO))
                    self._pip_install(["-r", req_file])
                else:
                    for pip_name, _ in missing:
                        if pip_name in ("torch", "torchvision"):
                            continue
                        n = pip_name
                        self.after(0, lambda n=n: self._log(
                            f"  ⬇ Installing {n}...", Theme.TEXT_DIM))
                        self._pip_install([pip_name])

            self.after(0, lambda: self._log(""))
            self.after(0, lambda: self._log(
                "🔄 Verifying installation...", Theme.INFO))
            still_missing = _check_missing_packages()
            if still_missing:
                names = [p for p, _ in still_missing]
                self.after(0, lambda: self._log(
                    f"⚠ Still missing: {', '.join(names)}", Theme.WARNING))
            else:
                self.after(0, lambda: self._log(
                    "✅ All packages verified OK!", Theme.SUCCESS))

            _mark_installed()
            self._install_done = True
            self.after(0, self._on_install_done)

        except Exception as exc:
            err = str(exc)
            self.after(0, lambda: self._log(f"❌ ERROR: {err}", Theme.ERROR))
            self.after(0, lambda: self._progress.stop())
            self.after(0, lambda: self._btn_install.configure(state="normal"))
            self.after(0, lambda: self._btn_skip.configure(state="normal"))

    def _pip_install(self, args):
        cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + args
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self.after(0, lambda l=line: self._log(
                        f"  {l}", Theme.TEXT_MUTED))
            proc.wait()
            if proc.returncode != 0:
                self.after(0, lambda: self._log(
                    f"  ⚠ pip returned exit code {proc.returncode}",
                    Theme.WARNING))
        except Exception as exc:
            self.after(0, lambda: self._log(
                f"  ❌ pip error: {exc}", Theme.ERROR))

    def _on_install_done(self):
        self._progress.stop()
        self._log("")
        self._log("🎉 Setup complete! Launching LocalOCR...", Theme.SUCCESS)
        self.after(1200, self._launch_app)

    def _skip(self):
        _mark_installed()
        self._install_done = True
        self._launch_app()

    def _launch_app(self):
        self.destroy()
        app = OCRApp()
        app.mainloop()

    def _on_close(self):
        self._cancelled = True
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
# Lazy pipeline loader
# ══════════════════════════════════════════════════════════════════════════════
_pipeline = None
_pipeline_lock = threading.Lock()


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                from src.pipeline import OCRPipeline
                _pipeline = OCRPipeline()
    return _pipeline


# ══════════════════════════════════════════════════════════════════════════════
# PDF preview helper
# ══════════════════════════════════════════════════════════════════════════════
def _render_pdf_page_pil(pdf_path: str, page_num: int = 0,
                         max_width: int = 480, max_height: int = 640):
    import fitz
    from PIL import Image
    try:
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            doc.close()
            return None, 0
        page = doc[page_num]
        rect = page.rect
        scale_w = max_width / rect.width if rect.width else 1
        scale_h = max_height / rect.height if rect.height else 1
        scale = min(scale_w, scale_h, 2.0)
        mat = fitz.Matrix(scale, scale)
        pix = page.get_pixmap(matrix=mat)
        total = len(doc)
        doc.close()
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        return img, total
    except Exception as exc:
        logger.warning("Preview render failed: %s", exc)
        return None, 0


# ══════════════════════════════════════════════════════════════════════════════
# Main GUI Application  — Dark themed
# ══════════════════════════════════════════════════════════════════════════════
class OCRApp(tk.Tk):
    """Desktop GUI for the PDF-to-DOCX OCR Pipeline with modern dark theme."""

    APP_TITLE = "LocalOCR — PDF to DOCX Converter"
    VERSION   = "v0.3.1"
    WINDOW_SIZE = "1280x860"

    def __init__(self):
        super().__init__()
        self.title(self.APP_TITLE)
        self.geometry(self.WINDOW_SIZE)
        self.minsize(1000, 720)
        self.configure(bg=Theme.BG_DARK)

        self._configure_styles()

        # State
        self._pdf_path = None
        self._total_pages = 0
        self._current_page = 0
        self._last_result = None
        self._preview_photo = None  # prevent GC
        self._start_time = None

        self._build_menu()
        self._build_header()
        self._build_ui()
        self._build_status_bar()
        self._log("✨ Ready. Select a PDF to begin.", Theme.INFO)

        # Pre-load pipeline in background
        threading.Thread(target=self._warmup_pipeline, daemon=True).start()

    # ── Theme / Style Configuration ───────────────────────────────────────
    def _configure_styles(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        # Global defaults
        style.configure(".", background=Theme.BG_DARK,
                        foreground=Theme.TEXT, fieldbackground=Theme.BG_LIGHT,
                        bordercolor=Theme.BORDER, focuscolor=Theme.ACCENT,
                        font=("Segoe UI", 10))

        # Frames
        style.configure("TFrame", background=Theme.BG_DARK)
        style.configure("Card.TFrame", background=Theme.BG_MID)
        style.configure("Surface.TFrame", background=Theme.BG_SURFACE)

        # Labels
        style.configure("TLabel", background=Theme.BG_DARK,
                        foreground=Theme.TEXT)
        style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"),
                        foreground=Theme.ACCENT_GLOW, background=Theme.BG_DARK)
        style.configure("Subtitle.TLabel", font=("Segoe UI", 11),
                        foreground=Theme.TEXT_DIM, background=Theme.BG_DARK)
        style.configure("Status.TLabel", font=("Segoe UI", 10),
                        foreground=Theme.TEXT_DIM, background=Theme.BG_DARK)
        style.configure("Card.TLabel", background=Theme.BG_MID,
                        foreground=Theme.TEXT)
        style.configure("Accent.TLabel", background=Theme.BG_MID,
                        foreground=Theme.ACCENT_GLOW,
                        font=("Segoe UI", 10, "bold"))
        style.configure("Dim.TLabel", background=Theme.BG_MID,
                        foreground=Theme.TEXT_DIM)
        style.configure("File.TLabel", background=Theme.BG_DARK,
                        foreground=Theme.SUCCESS,
                        font=("Consolas", 10))

        # LabelFrames
        style.configure("TLabelframe", background=Theme.BG_MID,
                        foreground=Theme.ACCENT_GLOW,
                        bordercolor=Theme.BORDER)
        style.configure("TLabelframe.Label", background=Theme.BG_MID,
                        foreground=Theme.ACCENT_GLOW,
                        font=("Segoe UI", 10, "bold"))

        # Buttons
        style.configure("TButton", background=Theme.BG_SURFACE,
                        foreground=Theme.TEXT, padding=(12, 6),
                        font=("Segoe UI", 10))
        style.map("TButton",
                  background=[("active", Theme.ACCENT),
                              ("disabled", Theme.BG_LIGHT)],
                  foreground=[("active", "white"),
                              ("disabled", Theme.TEXT_MUTED)])

        style.configure("Accent.TButton", background=Theme.ACCENT,
                        foreground="white", padding=(18, 10),
                        font=("Segoe UI", 12, "bold"))
        style.map("Accent.TButton",
                  background=[("active", Theme.ACCENT_HOVER),
                              ("disabled", Theme.BG_LIGHT)],
                  foreground=[("disabled", Theme.TEXT_MUTED)])

        style.configure("Success.TButton", background=Theme.SUCCESS,
                        foreground="#1a1b2e", padding=(14, 7),
                        font=("Segoe UI", 10, "bold"))
        style.map("Success.TButton",
                  background=[("active", "#00e89c"),
                              ("disabled", Theme.BG_LIGHT)])

        style.configure("Danger.TButton", background=Theme.ERROR,
                        foreground="white", padding=(12, 6),
                        font=("Segoe UI", 10))
        style.map("Danger.TButton",
                  background=[("active", "#ff5e8a"),
                              ("disabled", Theme.BG_LIGHT)])

        # Combobox
        style.configure("TCombobox",
                        fieldbackground=Theme.BG_LIGHT,
                        background=Theme.BG_SURFACE,
                        foreground=Theme.TEXT,
                        arrowcolor=Theme.ACCENT,
                        selectbackground=Theme.ACCENT,
                        selectforeground="white")
        style.map("TCombobox",
                  fieldbackground=[("readonly", Theme.BG_LIGHT)],
                  foreground=[("readonly", Theme.TEXT)])
        # Dropdown list colours
        self.option_add("*TCombobox*Listbox.background", Theme.BG_LIGHT)
        self.option_add("*TCombobox*Listbox.foreground", Theme.TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", Theme.ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", "white")

        # Notebook (tabs)
        style.configure("TNotebook", background=Theme.BG_DARK,
                        bordercolor=Theme.BORDER)
        style.configure("TNotebook.Tab", background=Theme.BG_MID,
                        foreground=Theme.TEXT_DIM,
                        padding=(16, 8),
                        font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", Theme.ACCENT)],
                  foreground=[("selected", "white")],
                  expand=[("selected", [1, 1, 1, 0])])

        # Treeview
        style.configure("Treeview",
                        background=Theme.BG_MID,
                        foreground=Theme.TEXT,
                        fieldbackground=Theme.BG_MID,
                        rowheight=28,
                        font=("Segoe UI", 10),
                        bordercolor=Theme.BORDER)
        style.configure("Treeview.Heading",
                        background=Theme.BG_SURFACE,
                        foreground=Theme.ACCENT_GLOW,
                        font=("Segoe UI", 10, "bold"))
        style.map("Treeview",
                  background=[("selected", Theme.ACCENT)],
                  foreground=[("selected", "white")])

        # Scale
        style.configure("TScale", background=Theme.BG_MID,
                        troughcolor=Theme.BG_LIGHT,
                        bordercolor=Theme.BORDER)

        # Progressbar
        style.configure("Custom.Horizontal.TProgressbar",
                        troughcolor=Theme.BG_LIGHT,
                        background=Theme.ACCENT,
                        lightcolor=Theme.ACCENT_GLOW,
                        darkcolor=Theme.ACCENT,
                        bordercolor=Theme.BG_DARK,
                        thickness=6)

        # PanedWindow
        style.configure("TPanedwindow", background=Theme.BG_DARK)
        style.configure("Sash", sashthickness=6,
                        background=Theme.BORDER)

        # Scrollbar
        style.configure("Vertical.TScrollbar",
                        background=Theme.SCROLLBAR,
                        troughcolor=Theme.BG_DARK,
                        bordercolor=Theme.BG_DARK,
                        arrowcolor=Theme.TEXT_DIM)

    # ── Menu ──────────────────────────────────────────────────────────────
    def _build_menu(self):
        menubar = tk.Menu(self, bg=Theme.BG_MID, fg=Theme.TEXT,
                          activebackground=Theme.ACCENT,
                          activeforeground="white",
                          relief="flat", bd=0)
        self.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0,
                            bg=Theme.BG_MID, fg=Theme.TEXT,
                            activebackground=Theme.ACCENT,
                            activeforeground="white")
        file_menu.add_command(label="  📂  Open PDF…", command=self._browse_pdf,
                              accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="  ❌  Exit", command=self.destroy)
        menubar.add_cascade(label="  File  ", menu=file_menu)

        help_menu = tk.Menu(menubar, tearoff=0,
                            bg=Theme.BG_MID, fg=Theme.TEXT,
                            activebackground=Theme.ACCENT,
                            activeforeground="white")
        help_menu.add_command(label="  ℹ  About", command=self._show_about)
        menubar.add_cascade(label="  Help  ", menu=help_menu)

        self.bind_all("<Control-o>", lambda e: self._browse_pdf())

    def _show_about(self):
        messagebox.showinfo(
            "About LocalOCR",
            f"LocalOCR — PDF to DOCX Converter\n"
            f"{self.VERSION}\n\n"
            "Thai-optimised OCR pipeline\n"
            "EasyOCR + PaddleOCR + YOLO layout detection\n\n"
            "Apache-2.0 License\n"
            "github.com/chiraleo2000/Local_PDFtoDocx-OCR"
        )

    # ── Header Banner ─────────────────────────────────────────────────────
    def _build_header(self):
        hdr = tk.Canvas(self, height=64, bg=Theme.BG_DARK,
                        highlightthickness=0)
        hdr.pack(fill="x")
        # Gradient bar
        for i in range(64):
            frac = i / 64
            r = int(108 * (1 - frac) + 26 * frac)
            g = int(99 * (1 - frac) + 36 * frac)
            b = int(255 * (1 - frac) + 64 * frac)
            colour = f"#{r:02x}{g:02x}{b:02x}"
            hdr.create_line(0, i, 1400, i, fill=colour)

        hdr.create_text(20, 22, text="📄 LocalOCR", anchor="w",
                        fill="white", font=("Segoe UI", 18, "bold"))
        hdr.create_text(20, 46, text="PDF to DOCX • OCR • Layout Detection",
                        anchor="w", fill="#c0c0ff", font=("Segoe UI", 10))
        hdr.create_text(1260, 32, text=self.VERSION, anchor="e",
                        fill="#b0b0ff", font=("Segoe UI", 11, "bold"))

    # ── Main UI ───────────────────────────────────────────────────────────
    def _build_ui(self):
        # Toolbar row
        toolbar = tk.Frame(self, bg=Theme.BG_MID, padx=10, pady=8)
        toolbar.pack(fill="x", padx=0)

        self._btn_browse = ttk.Button(toolbar, text="📂 Open PDF…",
                                      command=self._browse_pdf)
        self._btn_browse.pack(side="left", padx=(4, 12))

        self._lbl_file = tk.Label(toolbar, text="No file selected",
                                  bg=Theme.BG_MID, fg=Theme.TEXT_DIM,
                                  font=("Consolas", 10), anchor="w")
        self._lbl_file.pack(side="left", fill="x", expand=True, padx=4)

        # Main paned area
        paned = ttk.PanedWindow(self, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=6, pady=(4, 2))

        left_frame = ttk.Frame(paned, style="TFrame")
        right_frame = ttk.Frame(paned, style="TFrame")
        paned.add(left_frame, weight=2)
        paned.add(right_frame, weight=3)

        self._build_left_panel(left_frame)
        self._build_right_panel(right_frame)

    def _build_left_panel(self, parent):
        """Settings + PDF preview."""
        # ── Settings Card ────────────────────────────────────────────────
        settings_lf = ttk.LabelFrame(parent, text="⚙ Settings", padding=10)
        settings_lf.pack(fill="x", padx=4, pady=4)

        row = 0

        def _add_combo(label_text, values, default_idx=0):
            nonlocal row
            ttk.Label(settings_lf, text=label_text,
                      style="Card.TLabel").grid(
                row=row, column=0, sticky="w", padx=(4, 8), pady=3)
            cb = ttk.Combobox(settings_lf, values=values, state="readonly",
                              width=28)
            cb.current(default_idx)
            cb.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
            row += 1
            return cb

        self._cb_quality = _add_combo(
            "Quality:", list(QUALITY_OPTIONS.keys()), 1)
        self._cb_language = _add_combo(
            "Language:", list(LANGUAGE_OPTIONS.keys()), 2)
        self._cb_engine = _add_combo(
            "OCR Engine:", list(ENGINE_OPTIONS.keys()), 1)
        self._cb_page_size = _add_combo(
            "Page Size:", PAGE_SIZE_OPTIONS, 0)
        self._cb_margin = _add_combo(
            "Margins:", list(MARGIN_OPTIONS.keys()), 0)

        # YOLO confidence
        ttk.Label(settings_lf, text="Detection Conf:",
                  style="Card.TLabel").grid(
            row=row, column=0, sticky="w", padx=(4, 8), pady=3)
        yolo_frame = tk.Frame(settings_lf, bg=Theme.BG_MID)
        yolo_frame.grid(row=row, column=1, sticky="ew", padx=4, pady=3)
        self._var_yolo = tk.DoubleVar(value=0.30)
        ttk.Scale(yolo_frame, from_=0.1, to=0.9,
                  orient="horizontal",
                  variable=self._var_yolo).pack(
            side="left", fill="x", expand=True)
        self._lbl_yolo_val = ttk.Label(yolo_frame, text="0.30", width=5,
                                       style="Accent.TLabel")
        self._lbl_yolo_val.pack(side="right")
        self._var_yolo.trace_add("write", lambda *_:
                                 self._lbl_yolo_val.configure(
                                     text=f"{self._var_yolo.get():.2f}"))
        row += 1

        # Header / Footer trim
        ttk.Label(settings_lf, text="Header Trim %:",
                  style="Card.TLabel").grid(
            row=row, column=0, sticky="w", padx=(4, 8), pady=3)
        self._var_header = tk.DoubleVar(value=0)
        ttk.Scale(settings_lf, from_=0, to=25, orient="horizontal",
                  variable=self._var_header).grid(
            row=row, column=1, sticky="ew", padx=4, pady=3)
        row += 1

        ttk.Label(settings_lf, text="Footer Trim %:",
                  style="Card.TLabel").grid(
            row=row, column=0, sticky="w", padx=(4, 8), pady=3)
        self._var_footer = tk.DoubleVar(value=0)
        ttk.Scale(settings_lf, from_=0, to=25, orient="horizontal",
                  variable=self._var_footer).grid(
            row=row, column=1, sticky="ew", padx=4, pady=3)
        row += 1

        settings_lf.columnconfigure(1, weight=1)

        # ── Convert Button ───────────────────────────────────────────────
        btn_frame = tk.Frame(parent, bg=Theme.BG_DARK, pady=4)
        btn_frame.pack(fill="x", padx=4)
        self._btn_convert = ttk.Button(
            btn_frame, text="  🔄  Convert PDF  ",
            style="Accent.TButton", command=self._start_convert)
        self._btn_convert.pack(fill="x", pady=4, ipady=2)

        # ── PDF Preview ──────────────────────────────────────────────────
        preview_lf = ttk.LabelFrame(parent, text="👁 PDF Preview", padding=4)
        preview_lf.pack(fill="both", expand=True, padx=4, pady=4)

        nav_frame = tk.Frame(preview_lf, bg=Theme.BG_MID)
        nav_frame.pack(fill="x")
        self._btn_prev_page = ttk.Button(nav_frame, text="◀ Prev",
                                         command=self._prev_page, width=8)
        self._btn_prev_page.pack(side="left", padx=2, pady=2)
        self._lbl_page = tk.Label(nav_frame, text="Page 0 / 0",
                                  bg=Theme.BG_MID, fg=Theme.TEXT,
                                  font=("Segoe UI", 10, "bold"))
        self._lbl_page.pack(side="left", expand=True)
        self._btn_next_page = ttk.Button(nav_frame, text="Next ▶",
                                         command=self._next_page, width=8)
        self._btn_next_page.pack(side="right", padx=2, pady=2)

        self._canvas = tk.Canvas(preview_lf, bg=Theme.CANVAS_BG,
                                 highlightthickness=0)
        self._canvas.pack(fill="both", expand=True, pady=4)
        self._canvas.bind("<Configure>", lambda e: self._update_preview())

    def _build_right_panel(self, parent):
        """Output text + results + download buttons."""
        nb = ttk.Notebook(parent)
        nb.pack(fill="both", expand=True, padx=4, pady=4)

        # ── Tab 1: OCR Text ──────────────────────────────────────────────
        text_tab = tk.Frame(nb, bg=Theme.BG_DARK, padx=4, pady=4)
        nb.add(text_tab, text="  📝 OCR Output  ")

        self._txt_output = scrolledtext.ScrolledText(
            text_tab, wrap="word", font=("Consolas", 10), state="disabled",
            bg=Theme.LOG_BG, fg=Theme.TEXT,
            insertbackground=Theme.TEXT,
            selectbackground=Theme.ACCENT,
            selectforeground="white",
            relief="flat", bd=0)
        self._txt_output.pack(fill="both", expand=True)

        # ── Tab 2: Conversion Info ───────────────────────────────────────
        info_tab = tk.Frame(nb, bg=Theme.BG_MID, padx=8, pady=8)
        nb.add(info_tab, text="  📊 Results  ")

        self._info_tree = ttk.Treeview(
            info_tab, columns=("value",), show="tree headings",
            height=12, selectmode="none")
        self._info_tree.heading("#0", text="Property")
        self._info_tree.heading("value", text="Value")
        self._info_tree.column("#0", width=200, stretch=False)
        self._info_tree.column("value", width=320, stretch=True)
        self._info_tree.pack(fill="both", expand=True)

        # ── Actions Row ──────────────────────────────────────────────────
        action_frame = tk.Frame(parent, bg=Theme.BG_MID, padx=8, pady=8)
        action_frame.pack(fill="x", padx=4, pady=(0, 2))

        self._btn_save_docx = ttk.Button(
            action_frame, text="💾 Save DOCX",
            style="Success.TButton",
            command=self._save_docx, state="disabled")
        self._btn_save_docx.pack(side="left", padx=4)

        self._btn_save_txt = ttk.Button(
            action_frame, text="📄 Save TXT",
            command=self._save_txt, state="disabled")
        self._btn_save_txt.pack(side="left", padx=4)

        self._btn_save_html = ttk.Button(
            action_frame, text="🌐 Save HTML",
            command=self._save_html, state="disabled")
        self._btn_save_html.pack(side="left", padx=4)

        self._btn_open_folder = ttk.Button(
            action_frame, text="📁 Open Folder",
            command=self._open_output_folder, state="disabled")
        self._btn_open_folder.pack(side="left", padx=4)

        # ── Log Panel ────────────────────────────────────────────────────
        log_frame = tk.Frame(parent, bg=Theme.BG_DARK, padx=4, pady=2)
        log_frame.pack(fill="x", padx=4, pady=(0, 4))

        log_header = tk.Frame(log_frame, bg=Theme.BG_DARK)
        log_header.pack(fill="x")
        tk.Label(log_header, text="📋 Log", bg=Theme.BG_DARK,
                 fg=Theme.ACCENT_GLOW,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")

        self._txt_log = scrolledtext.ScrolledText(
            log_frame, wrap="word", font=("Consolas", 9), height=7,
            state="disabled",
            bg=Theme.LOG_BG, fg=Theme.TEXT_DIM,
            insertbackground=Theme.TEXT,
            selectbackground=Theme.ACCENT,
            relief="flat", bd=0)
        self._txt_log.pack(fill="x")

        # Configure log colour tags
        self._txt_log.tag_configure("info", foreground=Theme.INFO)
        self._txt_log.tag_configure("success", foreground=Theme.SUCCESS)
        self._txt_log.tag_configure("warning", foreground=Theme.WARNING)
        self._txt_log.tag_configure("error", foreground=Theme.ERROR)
        self._txt_log.tag_configure("dim", foreground=Theme.TEXT_MUTED)

    # ── Status Bar ────────────────────────────────────────────────────────
    def _build_status_bar(self):
        bar = tk.Frame(self, bg=Theme.BG_MID, padx=10, pady=5)
        bar.pack(fill="x", side="bottom")

        self._progress = ttk.Progressbar(
            bar, mode="indeterminate", length=180,
            style="Custom.Horizontal.TProgressbar")
        self._progress.pack(side="left", padx=(0, 10))

        self._lbl_status = tk.Label(
            bar, text="Idle", bg=Theme.BG_MID, fg=Theme.TEXT_DIM,
            font=("Segoe UI", 10), anchor="w")
        self._lbl_status.pack(side="left", fill="x", expand=True)

        tk.Label(bar, text="LocalOCR v0.3.1",
                 bg=Theme.BG_MID, fg=Theme.TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(side="right")

    # ══════════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════════
    def _log(self, msg: str, colour: str = None):
        self._txt_log.configure(state="normal")
        tag_map = {
            Theme.INFO: "info", Theme.SUCCESS: "success",
            Theme.WARNING: "warning", Theme.ERROR: "error",
            Theme.TEXT_MUTED: "dim",
        }
        tag = tag_map.get(colour)
        if tag:
            self._txt_log.insert("end", msg + "\n", tag)
        elif colour:
            custom_tag = f"c_{colour}"
            self._txt_log.tag_configure(custom_tag, foreground=colour)
            self._txt_log.insert("end", msg + "\n", custom_tag)
        else:
            self._txt_log.insert("end", msg + "\n")
        self._txt_log.see("end")
        self._txt_log.configure(state="disabled")
        self._lbl_status.configure(text=msg[:140])

    def _set_output_text(self, text: str):
        self._txt_output.configure(state="normal")
        self._txt_output.delete("1.0", "end")
        self._txt_output.insert("1.0", text)
        self._txt_output.configure(state="disabled")

    def _set_busy(self, busy: bool):
        state = "disabled" if busy else "normal"
        self._btn_convert.configure(state=state)
        self._btn_browse.configure(state=state)
        if busy:
            self._progress.start(12)
            self._start_time = time.time()
        else:
            self._progress.stop()

    def _update_info_tree(self, meta: dict):
        self._info_tree.delete(*self._info_tree.get_children())
        elapsed = ""
        if self._start_time:
            elapsed = f"{time.time() - self._start_time:.1f}s"

        rows = [
            ("📄 Pages", meta.get("pages", "?")),
            ("📊 Tables Detected", meta.get("tables", "?")),
            ("🖼 Figures Detected", meta.get("figures", "?")),
            ("⚡ Quality", meta.get("quality", "?")),
            ("🌍 Languages", meta.get("languages", "?")),
        ]
        if elapsed:
            rows.append(("⏱ Processing Time", elapsed))

        engines = meta.get("engines", {})
        if engines:
            active = [k for k, v in engines.items() if v]
            rows.append(("🔧 Active OCR Engines",
                         ", ".join(active) if active else "none"))

        for label, val in rows:
            self._info_tree.insert("", "end", text=label, values=(val,))

        # Output file sizes
        if self._last_result:
            files = self._last_result.get("files", {})
            for fmt, path in files.items():
                if path and os.path.exists(str(path)):
                    size_kb = os.path.getsize(str(path)) / 1024
                    self._info_tree.insert(
                        "", "end", text=f"💾 Output {fmt.upper()}",
                        values=(f"{size_kb:.1f} KB",))

    # ══════════════════════════════════════════════════════════════════════
    # Pipeline warm-up  — staged with progress updates
    # ══════════════════════════════════════════════════════════════════════
    def _warmup_pipeline(self):
        steps = [
            ("📦 Importing core modules (fitz, cv2, numpy)...",
             lambda: (__import__("fitz"), __import__("cv2"), __import__("numpy"))),
            ("📦 Loading YOLO layout detector...",
             lambda: __import__("doclayout_yolo")),
            ("📦 Loading OCR engines (EasyOCR, PaddleOCR)...",
             lambda: (__import__("easyocr"), __import__("paddleocr"))),
            ("📦 Loading transformers & ONNX runtime...",
             lambda: (__import__("transformers"), __import__("onnxruntime"))),
            ("🔧 Initializing OCR pipeline...",
             lambda: _get_pipeline()),
        ]
        total = len(steps)
        try:
            self.after(0, lambda: self._set_progress_determinate(total))
            for i, (msg, fn) in enumerate(steps):
                self.after(0, lambda m=msg: self._log(m, Theme.TEXT_MUTED))
                self.after(0, lambda v=i: self._set_progress_value(v))
                t0 = time.time()
                try:
                    fn()
                except Exception:
                    pass  # optional modules may fail
                dt = time.time() - t0
                done_msg = f"   ✓ done ({dt:.1f}s)"
                self.after(0, lambda m=done_msg: self._log(m, Theme.TEXT_MUTED))
            self.after(0, lambda: self._set_progress_value(total))
            self.after(0, lambda: self._log(
                "✅ OCR pipeline loaded successfully.", Theme.SUCCESS))
        except Exception as exc:
            err_msg = str(exc)
            self.after(0, lambda: self._log(
                f"⚠ Pipeline load warning: {err_msg}", Theme.WARNING))
        finally:
            self.after(0, self._reset_progress)

    def _set_progress_determinate(self, maximum):
        self._progress.stop()
        self._progress.configure(mode="determinate", maximum=maximum, value=0)

    def _set_progress_value(self, value):
        self._progress.configure(value=value)

    def _reset_progress(self):
        self._progress.configure(mode="indeterminate", value=0)

    # ══════════════════════════════════════════════════════════════════════
    # File browsing
    # ══════════════════════════════════════════════════════════════════════
    def _browse_pdf(self):
        path = filedialog.askopenfilename(
            title="Select PDF file",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")],
        )
        if not path:
            return
        self._pdf_path = path
        name = os.path.basename(path)
        self._lbl_file.configure(text=f"📂 {name}", fg=Theme.SUCCESS)
        self._current_page = 0
        self._log(f"📂 Opened: {name}", Theme.INFO)
        self._update_preview()

    # ══════════════════════════════════════════════════════════════════════
    # Preview
    # ══════════════════════════════════════════════════════════════════════
    def _update_preview(self):
        if not self._pdf_path:
            return
        try:
            from PIL import ImageTk
        except ImportError:
            return

        cw = self._canvas.winfo_width() or 420
        ch = self._canvas.winfo_height() or 560
        pil_img, total = _render_pdf_page_pil(
            self._pdf_path, self._current_page,
            max_width=cw, max_height=ch)
        self._total_pages = total
        self._lbl_page.configure(
            text=f"Page {self._current_page + 1} / {total}")
        if pil_img:
            self._preview_photo = ImageTk.PhotoImage(pil_img)
            self._canvas.delete("all")
            self._canvas.create_image(
                cw // 2, ch // 2,
                image=self._preview_photo, anchor="center")

    def _prev_page(self):
        if self._current_page > 0:
            self._current_page -= 1
            self._update_preview()

    def _next_page(self):
        if self._current_page < self._total_pages - 1:
            self._current_page += 1
            self._update_preview()

    # ══════════════════════════════════════════════════════════════════════
    # Conversion
    # ══════════════════════════════════════════════════════════════════════
    def _start_convert(self):
        if not self._pdf_path:
            messagebox.showwarning("No PDF",
                                   "Please open a PDF file first.")
            return
        self._set_busy(True)
        name = os.path.basename(self._pdf_path)
        self._log(f"🔄 Converting: {name}...", Theme.INFO)
        threading.Thread(target=self._run_convert, daemon=True).start()

    def _run_convert(self):
        try:
            self.after(0, lambda: self._log(
                "  Step 1/4: Loading pipeline...", Theme.TEXT_MUTED))
            pipeline = _get_pipeline()

            quality = QUALITY_OPTIONS.get(
                self._cb_quality.get(), "balanced")
            languages = LANGUAGE_OPTIONS.get(
                self._cb_language.get(), "tha+eng")
            engine = ENGINE_OPTIONS.get(
                self._cb_engine.get(), "easyocr")
            page_size = self._cb_page_size.get() or "A4"
            margin_preset = MARGIN_OPTIONS.get(
                self._cb_margin.get(), "Normal")
            yolo_conf = self._var_yolo.get()
            header_pct = self._var_header.get()
            footer_pct = self._var_footer.get()

            pipeline.ocr.primary_engine = engine

            self.after(0, lambda: self._log(
                "  Step 2/4: Rendering & detecting layout...", Theme.TEXT_MUTED))

            # Count pages first for progress
            import fitz as _fitz
            _doc = _fitz.open(self._pdf_path)
            n_pages = len(_doc)
            _doc.close()
            self.after(0, lambda: self._set_progress_determinate(n_pages + 2))
            self.after(0, lambda: self._set_progress_value(1))

            # Hook into pipeline's logger to capture per-page progress
            import logging as _logging
            _pipe_logger = _logging.getLogger("Pipeline")
            _orig_level = _pipe_logger.level

            class _PageHandler(_logging.Handler):
                def __init__(self, app):
                    super().__init__()
                    self.app = app
                    self._page_count = 0
                def emit(self, record):
                    msg = record.getMessage()
                    if "Processing page" in msg:
                        self._page_count += 1
                        pc = self._page_count
                        self.app.after(0, lambda m=msg: self.app._log(
                            f"  📄 {m}", Theme.TEXT_MUTED))
                        self.app.after(0, lambda v=pc: self.app._set_progress_value(v + 1))

            _handler = _PageHandler(self)
            _pipe_logger.addHandler(_handler)

            try:
                result = pipeline.process_pdf(
                    self._pdf_path,
                    quality=quality,
                    header_trim=header_pct,
                    footer_trim=footer_pct,
                    languages=languages,
                    yolo_confidence=yolo_conf,
                    page_size=page_size,
                    margin_preset=margin_preset,
                )
            finally:
                _pipe_logger.removeHandler(_handler)

            self.after(0, lambda: self._log(
                "  Step 3/4: Exporting documents...", Theme.TEXT_MUTED))
            self.after(0, lambda: self._set_progress_value(n_pages + 1))
            time.sleep(0.1)  # let UI update

            self.after(0, lambda: self._log(
                "  Step 4/4: Finalising output files...", Theme.TEXT_MUTED))
            self.after(0, lambda: self._set_progress_value(n_pages + 2))

            self._last_result = result
            self.after(0, lambda: self._on_convert_done(result))
            self.after(0, self._reset_progress)

        except Exception as exc:
            err_msg = str(exc)
            self.after(0, lambda: self._on_convert_error(err_msg))
            self.after(0, self._reset_progress)

    def _on_convert_done(self, result: dict):
        self._set_busy(False)
        if result.get("success"):
            text = result.get("text", "")
            meta = result.get("metadata", {})
            self._set_output_text(text)
            self._update_info_tree(meta)

            pages = meta.get("pages", 0)
            tables = meta.get("tables", 0)
            figures = meta.get("figures", 0)
            elapsed = ""
            if self._start_time:
                elapsed = f" in {time.time() - self._start_time:.1f}s"
            self._log(
                f"✅ Conversion complete{elapsed}! "
                f"Pages: {pages}, Tables: {tables}, Figures: {figures}",
                Theme.SUCCESS)

            self._btn_save_docx.configure(state="normal")
            self._btn_save_txt.configure(state="normal")
            self._btn_save_html.configure(state="normal")
            self._btn_open_folder.configure(state="normal")
        else:
            err = result.get("error", "Unknown error")
            self._log(f"❌ Conversion failed: {err}", Theme.ERROR)
            messagebox.showerror("Conversion Error", err)

    def _on_convert_error(self, error_msg: str):
        self._set_busy(False)
        self._log(f"❌ Error: {error_msg}", Theme.ERROR)
        messagebox.showerror("Error", error_msg)

    # ══════════════════════════════════════════════════════════════════════
    # Save / Export
    # ══════════════════════════════════════════════════════════════════════
    def _save_file(self, fmt: str, ext: str, label: str):
        if not self._last_result:
            return
        src = self._last_result.get("files", {}).get(fmt)
        if not src or not os.path.exists(str(src)):
            messagebox.showwarning(f"No {label}",
                                   f"{label} file not available.")
            return
        default_name = (Path(self._pdf_path).stem + f".{ext}"
                        if self._pdf_path else f"output.{ext}")
        dest = filedialog.asksaveasfilename(
            title=f"Save {label}",
            defaultextension=f".{ext}",
            initialfile=default_name,
            filetypes=[(f"{label} File", f"*.{ext}"),
                       ("All Files", "*.*")],
        )
        if dest:
            shutil.copy2(str(src), dest)
            self._log(f"💾 {label} saved → {dest}", Theme.SUCCESS)

    def _save_docx(self):
        self._save_file("docx", "docx", "DOCX")

    def _save_txt(self):
        self._save_file("txt", "txt", "TXT")

    def _save_html(self):
        self._save_file("html", "html", "HTML")

    def _open_output_folder(self):
        if not self._last_result:
            return
        files = self._last_result.get("files", {})
        for fmt in ("docx", "txt", "html"):
            path = files.get(fmt)
            if path and os.path.exists(str(path)):
                folder = os.path.dirname(str(path))
                if sys.platform == "win32":
                    os.startfile(folder)
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", folder])
                else:
                    subprocess.Popen(["xdg-open", folder])
                return
        messagebox.showinfo("Info", "No output files found.")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if _is_first_run():
        wizard = SetupWizard()
        wizard.mainloop()
    else:
        app = OCRApp()
        app.mainloop()


if __name__ == "__main__":
    main()
