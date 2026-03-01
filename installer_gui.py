"""
LocalOCR Installer v0.3.2
=========================
Proper GUI installer for Windows / Linux / macOS.

Modes (radio selection):
  - Fresh Install  — cleans folder, downloads, creates venv, installs deps
  - Update/Repair  — re-downloads source, reinstalls deps in existing venv
  - Uninstall       — removes everything + desktop shortcut

Other buttons: Cancel (stops running operation), Close, Launch App.

Build:  py -3 build_installer.py
"""
import os
import sys
import subprocess
import shutil
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import urllib.request
import zipfile
import tempfile
import re
import glob

APP_NAME    = "LocalOCR"
APP_VERSION = "0.3.2"
REPO_URL    = "https://github.com/chiraleo2000/Local_PDFtoDocx-OCR/archive/refs/tags/v0.3.2.zip"
REPO_CLONE  = "https://github.com/chiraleo2000/Local_PDFtoDocx-OCR.git"
YOLO_MODEL_REPO   = "juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501"
YOLO_MODEL_FILE   = "doclayout_yolo_docstructbench_imgsz1280_2501.pt"
YOLO_MODEL_DIRECT = (
    "https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501"
    "/resolve/main/doclayout_yolo_docstructbench_imgsz1280_2501.pt"
)
MARKER      = ".installed_ok"
VENV_DIR    = "venv"
MIN_PY      = (3, 10)

if os.name == "nt":
    _HOME = os.environ.get("USERPROFILE", "C:\\")
else:
    _HOME = os.path.expanduser("~")
DEFAULT_DIR = os.path.join(_HOME, APP_NAME)

# ── Colours ───────────────────────────────────────────────────────────
BG        = "#1a1b2e"
BG_MID    = "#232440"
BG_LIGHT  = "#2d2f52"
ACCENT    = "#6c63ff"
ACCENT_H  = "#857dff"
GREEN     = "#00d68f"
YELLOW    = "#ffaa00"
RED       = "#ff3d71"
CYAN      = "#00b4d8"
TXT       = "#eaeaff"
TXT_DIM   = "#9899c2"
LOG_BG    = "#12132a"
DANGER    = "#c0392b"
DANGER_H  = "#e74c3c"
ORANGE    = "#f39c12"


# ══════════════════════════════════════════════════════════════════════
#  FIND REAL PYTHON 3.10+
# ══════════════════════════════════════════════════════════════════════
def _find_system_python():
    """Return (exe_path, 'x.y.z') or (None, None)."""
    def _try(args):
        try:
            kw = {}
            if os.name == "nt":
                kw["creationflags"] = subprocess.CREATE_NO_WINDOW
            r = subprocess.run(args + ["--version"], capture_output=True,
                               text=True, timeout=10, **kw)
            if r.returncode != 0:
                return None
            m = re.search(r"Python (\d+)\.(\d+)\.(\d+)", r.stdout + r.stderr)
            if not m:
                return None
            maj, mi, mic = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if (maj, mi) < MIN_PY:
                return None
            r2 = subprocess.run(args + ["-c", "import sys; print(sys.executable)"],
                                capture_output=True, text=True, timeout=10, **kw)
            if r2.returncode != 0:
                return None
            full = r2.stdout.strip()
            if os.name == "nt" and "WindowsApps" in full:
                return None
            return (full, f"{maj}.{mi}.{mic}")
        except Exception:
            return None

    # py -3 launcher (Windows)
    if os.name == "nt":
        r = _try(["py", "-3"])
        if r: return r

    # Our own interpreter (dev mode)
    if not getattr(sys, "frozen", False):
        r = _try([sys.executable])
        if r: return r

    # Common Windows paths
    if os.name == "nt":
        for base in [os.environ.get("LOCALAPPDATA", ""),
                     os.environ.get("APPDATA", ""),
                     "C:\\Python312", "C:\\Python311", "C:\\Python310",
                     "B:\\Python312", "B:\\Python311", "B:\\Python310"]:
            if not base: continue
            for sub in ["", "Programs\\Python\\Python312",
                        "Programs\\Python\\Python311",
                        "Programs\\Python\\Python310"]:
                d = os.path.join(base, sub) if sub else base
                exe = os.path.join(d, "python.exe")
                if os.path.isfile(exe):
                    r = _try([exe])
                    if r: return r

    # PATH search
    for name in ["python3", "python"]:
        p = shutil.which(name)
        if p:
            r = _try([p])
            if r: return r

    return (None, None)


def _venv_python(dest):
    if os.name == "nt":
        return os.path.join(dest, VENV_DIR, "Scripts", "python.exe")
    return os.path.join(dest, VENV_DIR, "bin", "python")


def _venv_pip(dest):
    if os.name == "nt":
        return os.path.join(dest, VENV_DIR, "Scripts", "pip.exe")
    return os.path.join(dest, VENV_DIR, "bin", "pip")


def _installed(path):
    return path and os.path.isfile(os.path.join(path, MARKER))


def _dir_has_content(path):
    """True if path exists and contains any files/dirs."""
    try:
        return os.path.isdir(path) and len(os.listdir(path)) > 0
    except Exception:
        return False


# ══════════════════════════════════════════════════════════════════════
class InstallerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} Installer v{APP_VERSION}")
        self.geometry("760x720")
        self.minsize(660, 620)
        self.configure(bg=BG)
        self._cancelled = False
        self._running   = False
        self._proc      = None
        self._sys_py    = None
        self._sys_pyver = None
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        self.after(100, self._detect_python)

    # ── UI ────────────────────────────────────────────────────────
    def _build_ui(self):
        # header
        c = tk.Canvas(self, height=68, bg=BG, highlightthickness=0)
        c.pack(fill="x")
        for i in range(68):
            t = i / 68
            r = int(108*(1-t)+26*t); g = int(99*(1-t)+27*t); b = int(255*(1-t)+46*t)
            c.create_line(0, i, 760, i, fill=f"#{r:02x}{g:02x}{b:02x}")
        c.create_text(20, 22, text=f"{APP_NAME} Setup v{APP_VERSION}",
                      anchor="w", fill="white", font=("Segoe UI", 17, "bold"))
        c.create_text(20, 48, text="PDF to DOCX OCR Converter",
                      anchor="w", fill="#d0d0ff", font=("Segoe UI", 10))

        # ── path ──────────────────────────────────────────────────
        pf = tk.Frame(self, bg=BG_MID, padx=14, pady=8)
        pf.pack(fill="x", padx=14, pady=(8, 2))
        tk.Label(pf, text="Install Location:", bg=BG_MID, fg=CYAN,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        row = tk.Frame(pf, bg=BG_MID); row.pack(fill="x", pady=(3, 0))
        self._var_path = tk.StringVar(value=DEFAULT_DIR)
        self._ent = tk.Entry(row, textvariable=self._var_path,
                             font=("Consolas", 10), bg=BG_LIGHT, fg=TXT,
                             relief="flat", insertbackground=TXT)
        self._ent.pack(side="left", fill="x", expand=True, ipady=3)
        self._btn_browse = tk.Button(row, text=" Browse ", command=self._browse,
                                     bg=BG_LIGHT, fg=TXT_DIM, relief="flat",
                                     font=("Segoe UI", 9), cursor="hand2",
                                     activebackground=ACCENT, activeforeground="white")
        self._btn_browse.pack(side="right", padx=(6, 0))

        # ── python info ───────────────────────────────────────────
        self._py_frame = tk.Frame(self, bg=BG_MID, padx=14, pady=3)
        self._py_frame.pack(fill="x", padx=14, pady=2)
        self._py_label = tk.Label(self._py_frame, text="Detecting Python...",
                                  bg=BG_MID, fg=YELLOW, font=("Consolas", 9))
        self._py_label.pack(anchor="w")

        # ── MODE SELECTION (radio buttons) ────────────────────────
        mf = tk.Frame(self, bg=BG_MID, padx=14, pady=8)
        mf.pack(fill="x", padx=14, pady=(4, 2))
        tk.Label(mf, text="Action:", bg=BG_MID, fg=CYAN,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))

        self._var_mode = tk.StringVar(value="install")
        modes = [
            ("install",   "Fresh Install",
             "Clean target folder, download source, create venv, install all packages"),
            ("update",    "Update / Repair",
             "Re-download source and update packages in existing venv"),
            ("uninstall", "Uninstall",
             "Remove all files and desktop shortcut"),
        ]
        for val, label, desc in modes:
            rf = tk.Frame(mf, bg=BG_MID)
            rf.pack(fill="x", pady=1)
            rb = tk.Radiobutton(rf, text=f"  {label}", variable=self._var_mode,
                                value=val, bg=BG_MID, fg=TXT,
                                selectcolor=BG_LIGHT, activebackground=BG_MID,
                                activeforeground=TXT,
                                font=("Segoe UI", 10), anchor="w",
                                cursor="hand2", indicatoron=True)
            rb.pack(side="left")
            tk.Label(rf, text=f"  — {desc}", bg=BG_MID, fg=TXT_DIM,
                     font=("Segoe UI", 8)).pack(side="left")

        # ── log ───────────────────────────────────────────────────
        lf = tk.Frame(self, bg=BG); lf.pack(fill="both", expand=True, padx=14, pady=4)
        tk.Label(lf, text="Log:", bg=BG, fg=TXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 3))
        self._log_w = scrolledtext.ScrolledText(
            lf, wrap="word", font=("Consolas", 9), height=13, state="disabled",
            bg=LOG_BG, fg=GREEN, insertbackground=GREEN,
            selectbackground=ACCENT, relief="flat", bd=0)
        self._log_w.pack(fill="both", expand=True)
        for t, clr in [("info",CYAN),("warn",YELLOW),("err",RED),
                        ("dim",TXT_DIM),("ok",GREEN)]:
            self._log_w.tag_configure(t, foreground=clr)

        # ── progress ──────────────────────────────────────────────
        sty = ttk.Style(self); sty.theme_use("clam")
        sty.configure("I.Horizontal.TProgressbar", troughcolor=BG_LIGHT,
                      background=ACCENT, darkcolor=ACCENT, lightcolor=ACCENT_H,
                      bordercolor=BG, thickness=8)
        self._prog = ttk.Progressbar(self, mode="determinate", maximum=100,
                                     style="I.Horizontal.TProgressbar")
        self._prog.pack(fill="x", padx=14, pady=6)

        # ── buttons ───────────────────────────────────────────────
        bf = tk.Frame(self, bg=BG, pady=8); bf.pack(fill="x", padx=14)
        self._bf = bf

        self._btn_cancel = tk.Button(bf, text="  Cancel  ", command=self._cancel,
            bg=BG_LIGHT, fg=TXT_DIM, activebackground=RED, activeforeground="white",
            font=("Segoe UI", 10), relief="flat", cursor="hand2")

        self._btn_close = tk.Button(bf, text="  Close  ", command=self._close,
            bg=BG_LIGHT, fg=TXT_DIM, activebackground=RED, activeforeground="white",
            font=("Segoe UI", 10), relief="flat", cursor="hand2")

        self._btn_go = tk.Button(bf, text="  Run  ", command=self._go,
            bg=ACCENT, fg="white", activebackground=ACCENT_H, activeforeground="white",
            font=("Segoe UI", 12, "bold"), relief="flat", cursor="hand2",
            padx=24, pady=3)

        self._btn_launch = tk.Button(bf, text="  Launch App  ",
            command=self._launch, bg=GREEN, fg="white",
            activebackground="#00e89d", activeforeground="white",
            font=("Segoe UI", 12, "bold"), relief="flat", cursor="hand2",
            padx=20, pady=3)

        # default layout
        self._btn_cancel.pack(side="left")
        self._btn_go.pack(side="right")

    # ── Python detection ──────────────────────────────────────────
    def _detect_python(self):
        def _w():
            py, ver = _find_system_python()
            self._sys_py = py; self._sys_pyver = ver
            if py:
                self.after(0, lambda: self._py_label.configure(
                    text=f"Python {ver}  |  {py}", fg=GREEN))
            else:
                self.after(0, lambda: self._py_label.configure(
                    text="ERROR: Python 3.10+ not found — install from python.org",
                    fg=RED))
                self.after(0, lambda: self._btn_go.configure(state="disabled"))
        threading.Thread(target=_w, daemon=True).start()

    # ── button layouts ────────────────────────────────────────────
    def _layout_idle(self):
        for w in self._bf.winfo_children(): w.pack_forget()
        self._btn_cancel.configure(bg=BG_LIGHT, fg=TXT_DIM)
        self._btn_cancel.pack(side="left")
        dest = self._var_path.get().strip()
        if _installed(dest):
            self._btn_launch.pack(side="right", padx=(6, 0))
        self._btn_go.configure(state="normal")
        self._btn_go.pack(side="right")

    def _layout_running(self):
        for w in self._bf.winfo_children(): w.pack_forget()
        self._btn_cancel.configure(bg=RED, fg="white")
        self._btn_cancel.pack(side="left")

    def _layout_done(self):
        for w in self._bf.winfo_children(): w.pack_forget()
        self._btn_close.pack(side="left")
        dest = self._var_path.get().strip()
        if _installed(dest):
            self._btn_launch.pack(side="right", padx=(6, 0))
        self._btn_go.configure(state="normal")
        self._btn_go.pack(side="right")

    # ── helpers ───────────────────────────────────────────────────
    def _browse(self):
        d = filedialog.askdirectory(title="Choose install folder",
                                    initialdir=self._var_path.get())
        if d: self._var_path.set(d)

    def _log(self, msg, tag=None):
        def _w():
            self._log_w.configure(state="normal")
            self._log_w.insert("end", msg + "\n", (tag,) if tag else ())
            self._log_w.see("end")
            self._log_w.configure(state="disabled")
        self.after(0, _w)

    def _clear_log(self):
        def _w():
            self._log_w.configure(state="normal")
            self._log_w.delete("1.0", "end")
            self._log_w.configure(state="disabled")
        self.after(0, _w)

    def _set_prog(self, v):
        self.after(0, lambda: self._prog.configure(value=v))

    def _lock(self):
        self._running = True; self._cancelled = False
        self._ent.configure(state="disabled")
        self._btn_browse.configure(state="disabled")
        self.after(0, self._layout_running)

    def _unlock(self):
        self._running = False
        self._ent.configure(state="normal")
        self._btn_browse.configure(state="normal")
        self.after(0, self._layout_done)

    def _cancel(self):
        if self._running:
            self._cancelled = True
            self._kill_proc()
            self._log("[CANCELLED] Stopped by user.", "warn")
            self.after(300, self._unlock)
        else:
            self._close()

    def _close(self):
        self._cancelled = True; self._kill_proc()
        self.destroy()

    def _kill_proc(self):
        p = self._proc
        if p and p.poll() is None:
            try: p.terminate(); p.wait(timeout=3)
            except Exception:
                try: p.kill()
                except Exception: pass
        self._proc = None

    def _run(self, cmd):
        """Run cmd, stream to log. Returns True on success."""
        if self._cancelled: return False
        try:
            kw = {}
            if os.name == "nt":
                kw["creationflags"] = subprocess.CREATE_NO_WINDOW
            p = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 bufsize=1, **kw)
            self._proc = p
            for line in p.stdout:
                if self._cancelled:
                    self._kill_proc(); return False
                line = line.rstrip()
                if line:
                    self._log(f"  {line}", "dim")
            p.wait()
            self._proc = None
            if p.returncode != 0:
                self._log(f"  (exit code {p.returncode})", "warn")
                return False
            return True
        except Exception as exc:
            self._proc = None
            self._log(f"  cmd error: {exc}", "err")
            return False

    def _check(self, cmd):
        """Run cmd silently, return True if exit 0."""
        try:
            kw = {}
            if os.name == "nt":
                kw["creationflags"] = subprocess.CREATE_NO_WINDOW
            r = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=30, **kw)
            return r.returncode == 0
        except Exception:
            return False

    # ══════════════════════════════════════════════════════════════
    #  DISPATCH (Run button)
    # ══════════════════════════════════════════════════════════════
    def _go(self):
        mode = self._var_mode.get()
        dest = self._var_path.get().strip()
        if not dest:
            messagebox.showerror("Error", "Choose an install path.")
            return
        if not self._sys_py:
            messagebox.showerror("Error",
                "Python 3.10+ not found.\nInstall from python.org first.")
            return

        if mode == "install":
            # Warn if folder has content
            if _dir_has_content(dest):
                ok = messagebox.askyesno("Folder Not Empty",
                    f"The folder already contains files:\n{dest}\n\n"
                    "ALL existing files will be DELETED before installation.\n\n"
                    "Continue?", icon="warning")
                if not ok:
                    return
            self._clear_log(); self._lock(); self._set_prog(0)
            threading.Thread(target=self._do_install, daemon=True).start()

        elif mode == "update":
            if not _installed(dest):
                messagebox.showwarning("Not Installed",
                    f"No installation found at:\n{dest}\n\n"
                    "Select 'Fresh Install' instead.")
                return
            self._clear_log(); self._lock(); self._set_prog(0)
            threading.Thread(target=self._do_update, daemon=True).start()

        elif mode == "uninstall":
            if not _installed(dest) and not _dir_has_content(dest):
                messagebox.showinfo("Nothing to Remove",
                    f"No installation at:\n{dest}")
                return
            ok = messagebox.askyesno("Confirm Uninstall",
                f"Remove ALL files from:\n{dest}\n\n"
                "This cannot be undone.", icon="warning")
            if not ok:
                return
            self._clear_log(); self._lock(); self._set_prog(0)
            threading.Thread(target=self._do_uninstall, daemon=True).start()

    # ══════════════════════════════════════════════════════════════
    #  FRESH INSTALL
    # ══════════════════════════════════════════════════════════════
    def _do_install(self):
        dest = self._var_path.get().strip()
        sys_py = self._sys_py
        vpy = _venv_python(dest)
        vpip = _venv_pip(dest)
        try:
            # 1 — clean target
            self._log("[1/9] Preparing install directory...", "info")
            self._set_prog(2)
            if _dir_has_content(dest):
                self._log(f"  Cleaning {dest} ...", "dim")
                for item in os.listdir(dest):
                    path = os.path.join(dest, item)
                    try:
                        if os.path.isdir(path):
                            shutil.rmtree(path)
                        else:
                            os.remove(path)
                    except Exception as e:
                        self._log(f"  Could not remove {item}: {e}", "warn")
                self._log("  Folder cleaned", "ok")
            os.makedirs(dest, exist_ok=True)
            self._log(f"  {dest}", "dim")
            if self._cancelled: return

            # 2 — download
            self._log("[2/9] Downloading source...", "info")
            self._set_prog(5)
            if not self._download(dest):
                if not self._cancelled:
                    self._log("ERROR: Download failed.", "err")
                self.after(0, self._unlock); return
            self._set_prog(20)
            if self._cancelled: return

            # 3 — venv
            self._log("[3/9] Creating virtual environment...", "info")
            self._set_prog(22)
            venv_path = os.path.join(dest, VENV_DIR)
            if os.path.isfile(vpy):
                self._log("  venv already exists", "dim")
            else:
                self._log(f"  {sys_py} -m venv {VENV_DIR}", "dim")
                ok = self._run([sys_py, "-m", "venv", venv_path])
                if not ok or not os.path.isfile(vpy):
                    self._log("ERROR: venv creation failed!", "err")
                    self.after(0, self._unlock); return
                self._log("  [OK] venv created", "ok")
            self._set_prog(28)
            if self._cancelled: return

            # 4 — pip
            self._log("[4/9] Upgrading pip...", "info")
            self._run([vpy, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
            self._set_prog(32)
            if self._cancelled: return

            # 5 — PyTorch
            self._log("[5/9] Installing PyTorch...", "info")
            if self._check([vpy, "-c", "import torch"]):
                self._log("  Already in venv", "dim")
            else:
                self._log("  Downloading PyTorch CPU (~400 MB)...", "info")
                self._run([vpip, "install",
                           "torch", "torchvision",
                           "--index-url", "https://download.pytorch.org/whl/cpu"])
            self._set_prog(50)
            if self._cancelled: return

            # 6 — core
            self._log("[6/9] Core packages...", "info")
            self._run([vpip, "install", "--quiet",
                       "PyMuPDF", "opencv-python-headless", "Pillow",
                       "numpy", "python-dotenv", "python-docx",
                       "beautifulsoup4", "lxml", "htmldocx", "requests"])
            self._set_prog(62)
            if self._cancelled: return

            # 7 — OCR
            self._log("[7/9] OCR engines...", "info")
            self._run([vpip, "install", "--quiet",
                       "easyocr", "paddleocr", "paddlepaddle"])
            self._set_prog(75)
            if self._cancelled: return

            # 8 — requirements.txt
            self._log("[8/10] Remaining packages...", "info")
            req = os.path.join(dest, "requirements.txt")
            if os.path.isfile(req):
                self._run([vpip, "install", "--quiet", "-r", req])
            else:
                self._run([vpip, "install", "--quiet",
                           "transformers", "onnxruntime",
                           "doclayout-yolo", "huggingface_hub"])
            self._set_prog(82)
            if self._cancelled: return

            # 9 — DocLayout-YOLO model
            self._log("[9/10] Downloading DocLayout-YOLO model (~30 MB)...", "info")
            model_dir = os.path.join(dest, "models", "DocLayout-YOLO-DocStructBench")
            model_pt  = os.path.join(model_dir, YOLO_MODEL_FILE)
            if os.path.isfile(model_pt):
                self._log(f"  [OK] Model already present: {model_pt}", "ok")
            else:
                if not self._download_yolo_model(dest, vpy, model_dir, model_pt):
                    self._log("  [WARN] Model download failed — app will error on first run.", "warn")
                    self._log("  Re-run installer (Update) or run from the app's Help menu.", "warn")
            self._set_prog(92)
            if self._cancelled: return

            # 10 — shortcuts
            self._log("[10/10] Creating shortcuts...", "info")
            self._write_marker(dest, vpy)
            self._write_launcher(dest, vpy)
            self._write_desktop_shortcut(dest, vpy)
            self._write_uninstaller(dest)
            self._set_prog(96)
            if self._cancelled: return

            # verify
            self._log("", "dim")
            self._log("Verifying install...", "info")
            for mod in ["fitz", "cv2", "PIL", "torch", "easyocr", "docx"]:
                if self._check([vpy, "-c", f"import {mod}"]):
                    self._log(f"  [OK] {mod}", "ok")
                else:
                    self._log(f"  [WARN] {mod} not found in venv", "warn")
            # Verify YOLO model
            if os.path.isfile(model_pt):
                self._log(f"  [OK] YOLO model: {model_pt}", "ok")
            else:
                self._log("  [WARN] YOLO model missing — run Update to retry.", "warn")

            self._set_prog(100)
            self._log("", "dim")
            self._log("=== Installation complete! ===", "ok")
            self._log(f"Path: {dest}", "ok")
            self._log(f"Venv: {vpy}", "dim")
            self._log("Click 'Launch App' to start LocalOCR.", "info")

        except Exception as exc:
            if not self._cancelled:
                self._log(f"ERROR: {exc}", "err")
        finally:
            if self._cancelled: self._log("Cancelled.", "warn")
            self.after(0, self._unlock)

    # ══════════════════════════════════════════════════════════════
    #  UPDATE / REPAIR
    # ══════════════════════════════════════════════════════════════
    def _do_update(self):
        dest = self._var_path.get().strip()
        sys_py = self._sys_py
        vpy = _venv_python(dest)
        vpip = _venv_pip(dest)
        try:
            # 1 — update source
            self._log("[1/4] Updating source...", "info")
            self._set_prog(10)
            git_dir = os.path.join(dest, ".git")
            if os.path.isdir(git_dir) and shutil.which("git"):
                self._log("  git pull...", "dim")
                if not self._run(["git", "-C", dest, "pull", "--rebase"]):
                    self._log("  Falling back to ZIP re-download...", "warn")
                    self._download(dest)
            else:
                self._download(dest)
            self._set_prog(30)
            if self._cancelled: return

            # 2 — ensure venv
            self._log("[2/4] Checking venv...", "info")
            if not os.path.isfile(vpy):
                self._log("  Venv missing — recreating...", "warn")
                venv_path = os.path.join(dest, VENV_DIR)
                self._run([sys_py, "-m", "venv", venv_path])
            if os.path.isfile(vpy):
                self._log("  [OK] venv present", "ok")
            else:
                self._log("  ERROR: Could not create venv!", "err")
                self.after(0, self._unlock); return
            self._set_prog(40)
            if self._cancelled: return

            # 3 — reinstall packages
            self._log("[3/4] Updating packages...", "info")
            req = os.path.join(dest, "requirements.txt")
            if os.path.isfile(req):
                self._run([vpip, "install", "--upgrade", "--quiet", "-r", req])
            # Also reinstall core packages
            self._run([vpip, "install", "--upgrade", "--quiet",
                       "PyMuPDF", "opencv-python-headless", "easyocr",
                       "python-docx", "htmldocx"])
            self._set_prog(80)
            if self._cancelled: return

            # 4 — YOLO model
            self._log("[4/5] Checking DocLayout-YOLO model...", "info")
            model_dir = os.path.join(dest, "models", "DocLayout-YOLO-DocStructBench")
            model_pt  = os.path.join(model_dir, YOLO_MODEL_FILE)
            if os.path.isfile(model_pt):
                self._log(f"  [OK] Model already present: {model_pt}", "ok")
            else:
                self._log("  Model missing — downloading (~30 MB)...", "warn")
                if not self._download_yolo_model(dest, vpy, model_dir, model_pt):
                    self._log("  [WARN] Model download failed — re-run Update to retry.", "warn")
                else:
                    self._log(f"  [OK] Model downloaded: {model_pt}", "ok")
            self._set_prog(90)
            if self._cancelled: return

            # 5 — refresh marker
            self._log("[5/5] Finalizing...", "info")
            self._write_marker(dest, vpy)
            self._write_launcher(dest, vpy)
            self._set_prog(100)
            self._log("", "dim")
            self._log("=== Update complete! ===", "ok")

        except Exception as exc:
            if not self._cancelled:
                self._log(f"ERROR: {exc}", "err")
        finally:
            if self._cancelled: self._log("Cancelled.", "warn")
            self.after(0, self._unlock)

    # ══════════════════════════════════════════════════════════════
    #  UNINSTALL
    # ══════════════════════════════════════════════════════════════
    def _do_uninstall(self):
        dest = self._var_path.get().strip()
        try:
            self._log("[1/3] Removing desktop shortcut...", "info")
            self._set_prog(20)
            self._remove_shortcut()

            self._log("[2/3] Removing all files...", "info")
            self._set_prog(50)
            if os.path.isdir(dest):
                shutil.rmtree(dest, ignore_errors=True)
                self._log(f"  Removed {dest}", "ok")
            else:
                self._log(f"  Not found: {dest}", "warn")

            self._log("[3/3] Verifying...", "info")
            self._set_prog(90)
            if not os.path.exists(dest):
                self._log("  Fully removed", "ok")
            else:
                self._log("  Some files may remain (in use?)", "warn")

            self._set_prog(100)
            self._log("", "dim")
            self._log("=== Uninstall complete. ===", "ok")
        except Exception as exc:
            self._log(f"ERROR: {exc}", "err")
        finally:
            self.after(0, self._unlock)

    # ══════════════════════════════════════════════════════════════
    #  LAUNCH — actually open the GUI app visible on screen
    # ══════════════════════════════════════════════════════════════
    def _launch(self):
        dest = self._var_path.get().strip()
        gui  = os.path.join(dest, "gui_app.py")
        if not os.path.isfile(gui):
            self._log(f"gui_app.py not found in {dest}", "err")
            messagebox.showerror("Error", f"gui_app.py not found in:\n{dest}")
            return

        # Use venv Python; prefer pythonw on Windows for no-console
        vpy = _venv_python(dest)
        exe = vpy
        if os.name == "nt":
            pyw = vpy.replace("python.exe", "pythonw.exe")
            if os.path.isfile(pyw):
                exe = pyw
        if not os.path.isfile(exe):
            exe = self._sys_py or "python"

        # Ensure .installed_ok marker exists (gui_app.py checks this)
        marker = os.path.join(dest, MARKER)
        if not os.path.isfile(marker):
            self._write_marker(dest, exe)

        self._log(f"Launching: {exe} gui_app.py", "ok")
        self._log(f"  Working dir: {dest}", "dim")
        try:
            # DETACHED_PROCESS only — do NOT use CREATE_NO_WINDOW
            # CREATE_NO_WINDOW suppresses the tkinter GUI window!
            if os.name == "nt":
                subprocess.Popen(
                    [exe, gui],
                    cwd=dest,
                    creationflags=subprocess.DETACHED_PROCESS,
                    close_fds=True)
            else:
                subprocess.Popen(
                    [exe, gui],
                    cwd=dest,
                    start_new_session=True,
                    close_fds=True)
            self._log("App launched successfully!", "ok")
        except Exception as exc:
            err = str(exc)
            self._log(f"Launch error: {err}", "err")
            messagebox.showerror("Launch Failed", err)

    # ══════════════════════════════════════════════════════════════    #  YOLO MODEL DOWNLOAD
    # ════════════════════════════════════════════════════════════
    def _download_yolo_model(self, dest, vpy, model_dir, model_pt):
        """Download the DocLayout-YOLO model. Returns True on success."""
        os.makedirs(model_dir, exist_ok=True)
        # Write temp helper script
        script = f'''
import os, sys, shutil, urllib.request
model_dir = r"{model_dir}"
model_pt  = r"{model_pt}"
model_file = "{YOLO_MODEL_FILE}"
hf_repo    = "{YOLO_MODEL_REPO}"
direct_url = "{YOLO_MODEL_DIRECT}"
os.makedirs(model_dir, exist_ok=True)
if os.path.isfile(model_pt):
    print("Model already present:", model_pt); sys.exit(0)
# Try huggingface_hub first
try:
    from huggingface_hub import hf_hub_download
    print("Downloading via huggingface_hub...")
    cached = hf_hub_download(hf_repo, model_file)
    shutil.copy2(cached, model_pt)
    print("Downloaded:", model_pt); sys.exit(0)
except Exception as e:
    print("huggingface_hub failed:", e)
# Direct URL fallback
try:
    print("Trying direct URL download...")
    def _progress(block, block_size, total):
        if total > 0:
            pct = min(100, block * block_size * 100 // total)
            print(f"  {{pct}}%", flush=True) if pct % 10 == 0 else None
    urllib.request.urlretrieve(direct_url, model_pt, reporthook=_progress)
    print("Downloaded:", model_pt); sys.exit(0)
except Exception as e:
    print("Direct download failed:", e); sys.exit(1)
'''
        tmp_script = os.path.join(tempfile.gettempdir(), "_localocr_dl_model.py")
        try:
            with open(tmp_script, "w", encoding="utf-8") as f:
                f.write(script)
            ok = self._run([vpy, tmp_script])
            return ok and os.path.isfile(model_pt)
        except Exception as exc:
            self._log(f"  Model download error: {exc}", "err")
            return False
        finally:
            try: os.remove(tmp_script)
            except OSError: pass

    # ════════════════════════════════════════════════════════════    #  DOWNLOAD
    # ══════════════════════════════════════════════════════════════
    def _download(self, dest):
        # git
        try:
            if shutil.which("git"):
                self._log("  git clone...", "dim")
                tmp = dest + "__tmp"
                if os.path.exists(tmp):
                    shutil.rmtree(tmp, ignore_errors=True)
                ok = self._run(["git", "clone", "--depth", "1",
                                "--branch", f"v{APP_VERSION}",
                                REPO_CLONE, tmp])
                if ok and os.path.isdir(tmp):
                    self._copy_tree(tmp, dest)
                    shutil.rmtree(tmp, ignore_errors=True)
                    self._log("  [OK] Cloned", "ok")
                    return True
        except Exception as exc:
            self._log(f"  git failed: {exc}", "warn")
        if self._cancelled: return False
        # zip
        try:
            self._log("  Downloading ZIP...", "dim")
            zp = os.path.join(tempfile.gettempdir(), "localocr.zip")
            urllib.request.urlretrieve(REPO_URL, zp)
            self._log("  Extracting...", "dim")
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(tempfile.gettempdir())
                top = zf.namelist()[0].split("/")[0]
            ext = os.path.join(tempfile.gettempdir(), top)
            self._copy_tree(ext, dest)
            shutil.rmtree(ext, ignore_errors=True)
            try: os.remove(zp)
            except OSError: pass
            self._log("  [OK] Downloaded", "ok")
            return True
        except Exception as exc:
            self._log(f"  ZIP failed: {exc}", "err")
            return False

    @staticmethod
    def _copy_tree(src_dir, dst_dir):
        for item in os.listdir(src_dir):
            s = os.path.join(src_dir, item)
            d = os.path.join(dst_dir, item)
            if os.path.isdir(s):
                if os.path.exists(d):
                    shutil.rmtree(d, ignore_errors=True)
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)

    # ══════════════════════════════════════════════════════════════
    #  FILE HELPERS
    # ══════════════════════════════════════════════════════════════
    def _write_marker(self, dest, vpy):
        with open(os.path.join(dest, MARKER), "w", encoding="utf-8") as f:
            f.write(f"version={APP_VERSION}\n")
            f.write(f"installed={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"python={vpy}\n")
            f.write(f"venv={os.path.join(dest, VENV_DIR)}\n")
        self._log("  .installed_ok written", "dim")

    def _write_launcher(self, dest, vpy):
        if os.name == "nt":
            bat = os.path.join(dest, f"{APP_NAME}.bat")
            with open(bat, "w", encoding="utf-8") as f:
                f.write('@echo off\n')
                f.write(f'title {APP_NAME}\n')
                f.write(f'cd /d "{dest}"\n')
                f.write(f'"{vpy}" gui_app.py\n')
            self._log("  LocalOCR.bat created", "dim")
        else:
            sh = os.path.join(dest, "localocr.sh")
            with open(sh, "w", encoding="utf-8") as f:
                f.write('#!/usr/bin/env bash\n')
                f.write(f'cd "{dest}"\n')
                f.write(f'"{vpy}" gui_app.py\n')
            os.chmod(sh, 0o755)
            self._log("  localocr.sh created", "dim")

    def _write_desktop_shortcut(self, dest, vpy):
        if os.name != "nt":
            try:
                desk = os.path.join(os.path.expanduser("~"), "Desktop")
                if os.path.isdir(desk):
                    df = os.path.join(desk, f"{APP_NAME}.desktop")
                    with open(df, "w") as f:
                        f.write("[Desktop Entry]\nType=Application\n")
                        f.write(f"Name={APP_NAME}\n")
                        f.write(f'Exec="{vpy}" "{dest}/gui_app.py"\n')
                        f.write(f"Path={dest}\nTerminal=false\n")
                    os.chmod(df, 0o755)
                    self._log(f"  Desktop shortcut: {df}", "ok")
            except Exception as exc:
                self._log(f"  Desktop shortcut failed: {exc}", "warn")
            return

        try:
            desktop = os.path.join(_HOME, "Desktop")
            if not os.path.isdir(desktop):
                self._log("  Desktop folder not found", "warn")
                return
            lnk = os.path.join(desktop, f"{APP_NAME}.lnk")
            gui = os.path.join(dest, "gui_app.py")
            pyw = vpy.replace("python.exe", "pythonw.exe")
            if not os.path.isfile(pyw):
                pyw = vpy
            vbs = (
                'Set ws = CreateObject("WScript.Shell")\n'
                f'Set sc = ws.CreateShortcut("{lnk}")\n'
                f'sc.TargetPath = "{pyw}"\n'
                f'sc.Arguments = """{gui}"""\n'
                f'sc.WorkingDirectory = "{dest}"\n'
                f'sc.Description = "{APP_NAME} - PDF to DOCX OCR"\n'
                'sc.Save\n'
            )
            vbs_f = os.path.join(tempfile.gettempdir(), "localocr_lnk.vbs")
            with open(vbs_f, "w", encoding="utf-8") as f:
                f.write(vbs)
            r = subprocess.run(["cscript", "//Nologo", vbs_f],
                               capture_output=True, text=True, timeout=15,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            try: os.remove(vbs_f)
            except OSError: pass
            if os.path.isfile(lnk):
                self._log(f"  [OK] Desktop shortcut: {lnk}", "ok")
            else:
                self._log(f"  Shortcut failed (cscript exit {r.returncode})", "warn")
                if r.stderr.strip():
                    self._log(f"  {r.stderr.strip()}", "warn")
        except Exception as exc:
            self._log(f"  Shortcut failed: {exc}", "warn")

    def _write_uninstaller(self, dest):
        if os.name != "nt":
            return
        p = os.path.join(dest, "uninstall.bat")
        with open(p, "w", encoding="utf-8") as f:
            f.write('@echo off\n')
            f.write('title Uninstall LocalOCR\n')
            f.write('echo.\n')
            f.write('echo  === Uninstall LocalOCR ===\n')
            f.write(f'echo  Dir: {dest}\n')
            f.write('echo.\n')
            f.write('set /p c="Remove? (Y/N): "\n')
            f.write('if /i not "%c%"=="Y" (echo Cancelled. & pause & exit /b)\n')
            f.write('echo Removing shortcut...\n')
            f.write(f'del "{_HOME}\\Desktop\\LocalOCR.lnk" 2>nul\n')
            f.write(f'cd /d "{_HOME}"\n')
            f.write(f'rmdir /s /q "{dest}"\n')
            f.write('echo Done! & pause\n')
        self._log("  uninstall.bat created", "dim")

    def _remove_shortcut(self):
        if os.name == "nt":
            lnk = os.path.join(_HOME, "Desktop", f"{APP_NAME}.lnk")
        else:
            lnk = os.path.join(os.path.expanduser("~"), "Desktop",
                               f"{APP_NAME}.desktop")
        try:
            if os.path.isfile(lnk):
                os.remove(lnk)
                self._log(f"  Removed {lnk}", "ok")
            else:
                self._log("  No shortcut found", "dim")
        except Exception as exc:
            self._log(f"  {exc}", "warn")


def main():
    app = InstallerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
