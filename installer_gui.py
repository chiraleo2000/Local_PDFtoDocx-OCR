"""
LocalOCR Installer v0.3.0
=========================
Proper GUI installer: finds real Python, creates venv, installs into venv.

Flow:
  1. Find real system Python 3.10+
  2. Download source (git clone or ZIP)
  3. Create venv inside install dir
  4. pip install everything into venv
  5. Create launcher, desktop shortcut, uninstaller
  6. Verify packages inside the venv

Buttons: Install | Update | Uninstall | Launch | Cancel | Close

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

APP_NAME    = "LocalOCR"
APP_VERSION = "0.3.0"
REPO_URL    = "https://github.com/chiraleo2000/Local_PDFtoDocx-OCR/archive/refs/tags/v0.3.0.zip"
REPO_CLONE  = "https://github.com/chiraleo2000/Local_PDFtoDocx-OCR.git"
MARKER      = ".installed_ok"
VENV_DIR    = "venv"                    # relative to install dir
MIN_PY      = (3, 10)

if os.name == "nt":
    DEFAULT_DIR = os.path.join(os.environ.get("USERPROFILE", "C:\\"), APP_NAME)
else:
    DEFAULT_DIR = os.path.join(os.path.expanduser("~"), APP_NAME)

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
ORANGE_H  = "#f1c40f"


# ══════════════════════════════════════════════════════════════════════
#  FIND REAL PYTHON — the critical fix
# ══════════════════════════════════════════════════════════════════════
def _find_system_python():
    """
    Find a real Python 3.10+ executable on this system.
    Returns (path, version_string) or (None, None).

    Strategy on Windows:
      1. Try 'py -3' (Python Launcher — most reliable on Windows)
      2. Search common install locations
      3. Try 'python3', 'python' from PATH (but verify not MS Store alias)
    """
    def _check(exe_args):
        """Run python --version, return (full_path, '3.x.y') or None."""
        try:
            r = subprocess.run(
                exe_args + ["--version"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if r.returncode != 0:
                return None
            m = re.search(r"Python (\d+)\.(\d+)\.(\d+)", r.stdout + r.stderr)
            if not m:
                return None
            major, minor, micro = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if (major, minor) < MIN_PY:
                return None
            # Get the actual full path
            r2 = subprocess.run(
                exe_args + ["-c", "import sys; print(sys.executable)"],
                capture_output=True, text=True, timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if r2.returncode == 0 and r2.stdout.strip():
                full = r2.stdout.strip()
                # On Windows, reject MS Store WindowsApps stubs
                if os.name == "nt" and "WindowsApps" in full:
                    return None
                return (full, f"{major}.{minor}.{micro}")
            return None
        except Exception:
            return None

    # 1. Python Launcher (Windows)
    if os.name == "nt":
        result = _check(["py", "-3"])
        if result:
            return result

    # 2. Not frozen? Use our own interpreter
    if not getattr(sys, "frozen", False):
        exe = sys.executable
        if exe and os.path.isfile(exe):
            result = _check([exe])
            if result:
                return result

    # 3. Common Windows install paths
    if os.name == "nt":
        for base in [os.environ.get("LOCALAPPDATA", ""),
                     os.environ.get("APPDATA", ""),
                     "C:\\Python312", "C:\\Python311", "C:\\Python310",
                     "B:\\Python312", "B:\\Python311", "B:\\Python310"]:
            if not base:
                continue
            for sub in ["", "Programs\\Python\\Python312",
                        "Programs\\Python\\Python311",
                        "Programs\\Python\\Python310"]:
                d = os.path.join(base, sub) if sub else base
                exe = os.path.join(d, "python.exe")
                if os.path.isfile(exe):
                    result = _check([exe])
                    if result:
                        return result

    # 4. PATH search
    for name in (["python3", "python"] if os.name != "nt"
                 else ["python3", "python"]):
        p = shutil.which(name)
        if p:
            result = _check([p])
            if result:
                return result

    return (None, None)


def _venv_python(dest):
    """Return the python exe inside the venv."""
    if os.name == "nt":
        return os.path.join(dest, VENV_DIR, "Scripts", "python.exe")
    return os.path.join(dest, VENV_DIR, "bin", "python")


def _venv_pip(dest):
    """Return the pip exe inside the venv."""
    if os.name == "nt":
        return os.path.join(dest, VENV_DIR, "Scripts", "pip.exe")
    return os.path.join(dest, VENV_DIR, "bin", "pip")


def _installed(path):
    return os.path.isfile(os.path.join(path, MARKER))


# ══════════════════════════════════════════════════════════════════════
class InstallerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} Installer v{APP_VERSION}")
        self.geometry("740x680")
        self.minsize(640, 580)
        self.configure(bg=BG)
        self._cancelled = False
        self._running   = False
        self._proc      = None
        self._sys_py    = None          # system Python path
        self._sys_pyver = None          # system Python version
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._close)
        # Detect Python in background so UI appears instantly
        self.after(100, self._detect_python)
        self.after(200, self._refresh_buttons)

    # ── build UI ──────────────────────────────────────────────────
    def _build_ui(self):
        # gradient header
        c = tk.Canvas(self, height=72, bg=BG, highlightthickness=0)
        c.pack(fill="x")
        for i in range(72):
            t = i / 72
            r = int(108*(1-t)+26*t); g = int(99*(1-t)+27*t); b = int(255*(1-t)+46*t)
            c.create_line(0, i, 740, i, fill=f"#{r:02x}{g:02x}{b:02x}")
        c.create_text(20, 24, text=f"{APP_NAME} Setup v{APP_VERSION}",
                      anchor="w", fill="white", font=("Segoe UI", 18, "bold"))
        c.create_text(20, 52, text="PDF to DOCX OCR Converter",
                      anchor="w", fill="#d0d0ff", font=("Segoe UI", 10))

        # path
        pf = tk.Frame(self, bg=BG_MID, padx=16, pady=10)
        pf.pack(fill="x", padx=16, pady=(10, 2))
        tk.Label(pf, text="Install Location:", bg=BG_MID, fg=CYAN,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        row = tk.Frame(pf, bg=BG_MID); row.pack(fill="x", pady=(4, 0))
        self._var_path = tk.StringVar(value=DEFAULT_DIR)
        self._var_path.trace_add("write", lambda *_: self.after(120, self._refresh_buttons))
        self._ent = tk.Entry(row, textvariable=self._var_path,
                             font=("Consolas", 10), bg=BG_LIGHT, fg=TXT,
                             relief="flat", insertbackground=TXT)
        self._ent.pack(side="left", fill="x", expand=True, ipady=4)
        self._btn_browse = tk.Button(row, text=" Browse ", command=self._browse,
                                     bg=BG_LIGHT, fg=TXT_DIM, relief="flat",
                                     font=("Segoe UI", 9), cursor="hand2",
                                     activebackground=ACCENT, activeforeground="white")
        self._btn_browse.pack(side="right", padx=(6, 0))

        # status
        self._lbl = tk.Label(self, text="", bg=BG, fg=TXT_DIM,
                             font=("Segoe UI", 9), anchor="w")
        self._lbl.pack(fill="x", padx=20, pady=(2, 0))

        # python info
        self._py_frame = tk.Frame(self, bg=BG_MID, padx=16, pady=4)
        self._py_frame.pack(fill="x", padx=16, pady=2)
        self._py_label = tk.Label(self._py_frame,
                                  text="Detecting Python...",
                                  bg=BG_MID, fg=YELLOW,
                                  font=("Consolas", 9))
        self._py_label.pack(anchor="w")

        # log
        lf = tk.Frame(self, bg=BG); lf.pack(fill="both", expand=True, padx=16, pady=4)
        tk.Label(lf, text="Log:", bg=BG, fg=TXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self._log_w = scrolledtext.ScrolledText(
            lf, wrap="word", font=("Consolas", 9), height=14, state="disabled",
            bg=LOG_BG, fg=GREEN, insertbackground=GREEN,
            selectbackground=ACCENT, relief="flat", bd=0)
        self._log_w.pack(fill="both", expand=True)
        for t, clr in [("info",CYAN),("warn",YELLOW),("err",RED),
                        ("dim",TXT_DIM),("ok",GREEN)]:
            self._log_w.tag_configure(t, foreground=clr)

        # progress
        sty = ttk.Style(self); sty.theme_use("clam")
        sty.configure("I.Horizontal.TProgressbar", troughcolor=BG_LIGHT,
                      background=ACCENT, darkcolor=ACCENT, lightcolor=ACCENT_H,
                      bordercolor=BG, thickness=8)
        self._prog = ttk.Progressbar(self, mode="determinate", maximum=100,
                                     style="I.Horizontal.TProgressbar")
        self._prog.pack(fill="x", padx=16, pady=6)

        # button bar
        bf = tk.Frame(self, bg=BG, pady=10); bf.pack(fill="x", padx=16)
        self._bf = bf
        self._btn_cancel = tk.Button(bf, text="  Cancel  ", command=self._cancel,
            bg=BG_LIGHT, fg=TXT_DIM, activebackground=RED, activeforeground="white",
            font=("Segoe UI", 10), relief="flat", cursor="hand2")
        self._btn_close = tk.Button(bf, text="  Close  ", command=self._close,
            bg=BG_LIGHT, fg=TXT_DIM, activebackground=RED, activeforeground="white",
            font=("Segoe UI", 10), relief="flat", cursor="hand2")
        self._btn_uninstall = tk.Button(bf, text="  Uninstall  ",
            command=self._do_uninstall_start, bg=DANGER, fg="white",
            activebackground=DANGER_H, activeforeground="white",
            font=("Segoe UI", 10, "bold"), relief="flat", cursor="hand2", padx=10)
        self._btn_update = tk.Button(bf, text="  Update  ",
            command=self._do_update_start, bg=ORANGE, fg="white",
            activebackground=ORANGE_H, activeforeground="white",
            font=("Segoe UI", 11, "bold"), relief="flat", cursor="hand2", padx=14)
        self._btn_install = tk.Button(bf, text="  Install  ",
            command=self._do_install_start, bg=ACCENT, fg="white",
            activebackground=ACCENT_H, activeforeground="white",
            font=("Segoe UI", 12, "bold"), relief="flat", cursor="hand2", padx=20, pady=3)
        self._btn_launch = tk.Button(bf, text="  Launch App  ",
            command=self._launch, bg=GREEN, fg="white",
            activebackground="#00e89d", activeforeground="white",
            font=("Segoe UI", 12, "bold"), relief="flat", cursor="hand2", padx=20, pady=3)

    # ── Python detection ──────────────────────────────────────────
    def _detect_python(self):
        def _work():
            py, ver = _find_system_python()
            self._sys_py = py
            self._sys_pyver = ver
            if py:
                self.after(0, lambda: self._py_label.configure(
                    text=f"Python {ver}  |  {py}", fg=GREEN))
            else:
                self.after(0, lambda: self._py_label.configure(
                    text="ERROR: Python 3.10+ not found! Install Python first.",
                    fg=RED))
                self.after(0, lambda: self._btn_install.configure(state="disabled"))
        threading.Thread(target=_work, daemon=True).start()

    # ── button layout ─────────────────────────────────────────────
    def _refresh_buttons(self):
        if self._running:
            return
        for w in self._bf.winfo_children():
            w.pack_forget()
        dest = self._var_path.get().strip()
        installed = _installed(dest) if dest else False
        if installed:
            self._lbl.configure(text=f"Status: Installed at {dest}", fg=GREEN)
            self._btn_close.pack(side="left")
            self._btn_uninstall.pack(side="right", padx=(6, 0))
            self._btn_update.pack(side="right", padx=(6, 0))
            self._btn_launch.pack(side="right", padx=(6, 0))
        else:
            self._lbl.configure(text="Status: Not installed", fg=TXT_DIM)
            self._btn_cancel.pack(side="left")
            self._btn_install.pack(side="right")

    def _show_running_buttons(self):
        for w in self._bf.winfo_children():
            w.pack_forget()
        self._btn_cancel.configure(bg=RED, fg="white", text="  Cancel  ")
        self._btn_cancel.pack(side="left")

    def _show_done_buttons(self):
        for w in self._bf.winfo_children():
            w.pack_forget()
        self._btn_close.pack(side="left")
        dest = self._var_path.get().strip()
        if _installed(dest):
            self._btn_uninstall.pack(side="right", padx=(6, 0))
            self._btn_update.pack(side="right", padx=(6, 0))
            self._btn_launch.pack(side="right", padx=(6, 0))
        else:
            self._btn_install.pack(side="right")

    # ── helpers ───────────────────────────────────────────────────
    def _browse(self):
        d = filedialog.askdirectory(title="Choose install folder",
                                    initialdir=self._var_path.get())
        if d:
            self._var_path.set(d)

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
        self._running = True
        self._cancelled = False
        self._ent.configure(state="disabled")
        self._btn_browse.configure(state="disabled")
        self.after(0, self._show_running_buttons)

    def _unlock(self):
        self._running = False
        self._ent.configure(state="normal")
        self._btn_browse.configure(state="normal")
        self.after(0, self._show_done_buttons)

    def _cancel(self):
        if self._running:
            self._cancelled = True
            self._kill_proc()
            self._log("[CANCELLED] Operation cancelled by user.", "warn")
            self.after(300, self._unlock)
        else:
            self._close()

    def _close(self):
        self._cancelled = True
        self._kill_proc()
        self.destroy()

    def _kill_proc(self):
        p = self._proc
        if p and p.poll() is None:
            try:
                p.terminate(); p.wait(timeout=3)
            except Exception:
                try: p.kill()
                except Exception: pass
        self._proc = None

    # ── run subprocess ────────────────────────────────────────────
    def _run(self, cmd):
        """Run cmd, stream output to log. Returns True on success."""
        if self._cancelled:
            return False
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
                    self._kill_proc()
                    return False
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

    # ══════════════════════════════════════════════════════════════
    #  INSTALL
    # ══════════════════════════════════════════════════════════════
    def _do_install_start(self):
        if not self._sys_py:
            messagebox.showerror("Error",
                "Python 3.10+ not found on this system.\n"
                "Install Python from python.org first.")
            return
        self._clear_log()
        self._lock()
        self._set_prog(0)
        threading.Thread(target=self._do_install, daemon=True).start()

    def _do_install(self):
        dest = self._var_path.get().strip()
        if not dest:
            self._log("ERROR: Choose a path first.", "err")
            self.after(0, self._unlock); return
        sys_py = self._sys_py
        venv_py = _venv_python(dest)
        venv_pip = _venv_pip(dest)

        try:
            # ── 1. Create directory ───────────────────────────────
            self._log(f"[1/9] Creating install directory...", "info")
            self._set_prog(3)
            os.makedirs(dest, exist_ok=True)
            self._log(f"  {dest}", "dim")
            if self._cancelled: return

            # ── 2. Download source ────────────────────────────────
            self._log("[2/9] Downloading source...", "info")
            self._set_prog(5)
            if not self._download(dest):
                if not self._cancelled:
                    self._log("ERROR: Could not download source.", "err")
                self.after(0, self._unlock); return
            self._set_prog(20)
            if self._cancelled: return

            # ── 3. Create venv ────────────────────────────────────
            self._log("[3/9] Creating virtual environment...", "info")
            self._set_prog(22)
            venv_path = os.path.join(dest, VENV_DIR)
            if os.path.isfile(venv_py):
                self._log(f"  venv already exists at {venv_path}", "dim")
            else:
                self._log(f"  {sys_py} -m venv {venv_path}", "dim")
                ok = self._run([sys_py, "-m", "venv", venv_path])
                if not ok or not os.path.isfile(venv_py):
                    self._log("ERROR: Failed to create venv!", "err")
                    self._log(f"  Expected: {venv_py}", "err")
                    self.after(0, self._unlock); return
                self._log("  [OK] venv created", "ok")
            self._set_prog(28)
            if self._cancelled: return

            # ── 4. Upgrade pip in venv ────────────────────────────
            self._log("[4/9] Upgrading pip in venv...", "info")
            self._run([venv_py, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])
            self._set_prog(32)
            if self._cancelled: return

            # ── 5. Install PyTorch in venv ────────────────────────
            self._log("[5/9] Installing PyTorch...", "info")
            # Check if already in venv
            r = subprocess.run(
                [venv_py, "-c", "import torch; print(torch.__version__)"],
                capture_output=True, text=True, timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
            if r.returncode == 0:
                self._log(f"  Already installed: torch {r.stdout.strip()}", "dim")
            else:
                self._log("  Installing PyTorch CPU (~400 MB)...", "info")
                self._run([venv_pip, "install",
                           "torch", "torchvision",
                           "--index-url", "https://download.pytorch.org/whl/cpu"])
            self._set_prog(50)
            if self._cancelled: return

            # ── 6. Core packages ──────────────────────────────────
            self._log("[6/9] Installing core packages...", "info")
            self._run([venv_pip, "install", "--quiet",
                       "PyMuPDF", "opencv-python-headless", "Pillow",
                       "numpy", "python-dotenv", "python-docx",
                       "beautifulsoup4", "lxml", "htmldocx", "requests"])
            self._set_prog(60)
            if self._cancelled: return

            # ── 7. OCR engines ────────────────────────────────────
            self._log("[7/9] Installing OCR engines...", "info")
            self._run([venv_pip, "install", "--quiet",
                       "easyocr", "paddleocr", "paddlepaddle"])
            self._set_prog(75)
            if self._cancelled: return

            # ── 8. AI/ML + requirements.txt ───────────────────────
            self._log("[8/9] Installing AI/ML packages...", "info")
            req = os.path.join(dest, "requirements.txt")
            if os.path.isfile(req):
                self._run([venv_pip, "install", "--quiet", "-r", req])
            else:
                self._run([venv_pip, "install", "--quiet",
                           "transformers", "onnxruntime",
                           "doclayout-yolo", "huggingface_hub"])
            self._set_prog(85)
            if self._cancelled: return

            # ── 9. Shortcuts & launchers ──────────────────────────
            self._log("[9/9] Creating shortcuts & launchers...", "info")
            self._write_marker(dest, venv_py)
            self._write_launcher(dest, venv_py)
            self._write_desktop_shortcut(dest, venv_py)
            self._write_uninstaller(dest)
            self._set_prog(95)
            if self._cancelled: return

            # ── Verify inside venv ────────────────────────────────
            self._log("", "dim")
            self._log("Verifying packages in venv...", "info")
            for mod in ["fitz", "cv2", "PIL", "torch", "easyocr", "docx"]:
                r = subprocess.run(
                    [venv_py, "-c", f"import {mod}; print('{mod} OK')"],
                    capture_output=True, text=True, timeout=30,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                if r.returncode == 0:
                    self._log(f"  [OK] {mod}", "ok")
                else:
                    self._log(f"  [WARN] {mod} not found in venv", "warn")

            self._set_prog(100)
            self._log("", "dim")
            self._log("=== Installation complete! ===", "ok")
            self._log(f"Installed to: {dest}", "ok")
            self._log(f"Venv Python: {venv_py}", "ok")
            self._log("Click 'Launch App' to start, or 'Close' to exit.", "info")

        except Exception as exc:
            if not self._cancelled:
                self._log(f"ERROR: {exc}", "err")
        finally:
            if self._cancelled:
                self._log("Cancelled.", "warn")
            self.after(0, self._unlock)

    # ══════════════════════════════════════════════════════════════
    #  UPDATE
    # ══════════════════════════════════════════════════════════════
    def _do_update_start(self):
        self._clear_log()
        self._lock()
        self._set_prog(0)
        threading.Thread(target=self._do_update, daemon=True).start()

    def _do_update(self):
        dest = self._var_path.get().strip()
        venv_py = _venv_python(dest)
        venv_pip = _venv_pip(dest)
        # Fall back to system py if no venv
        if not os.path.isfile(venv_pip):
            venv_pip = None
        try:
            self._log("[1/3] Updating source...", "info")
            self._set_prog(10)
            git_dir = os.path.join(dest, ".git")
            if os.path.isdir(git_dir) and shutil.which("git"):
                self._log("  git pull...", "dim")
                if not self._run(["git", "-C", dest, "pull", "--rebase"]):
                    self._download(dest)
            else:
                self._download(dest)
            self._set_prog(40)
            if self._cancelled: return

            self._log("[2/3] Updating packages...", "info")
            if venv_pip:
                req = os.path.join(dest, "requirements.txt")
                if os.path.isfile(req):
                    self._run([venv_pip, "install", "--upgrade", "--quiet", "-r", req])
                self._log("  Packages updated in venv", "ok")
            else:
                self._log("  No venv found — skipping package update", "warn")
            self._set_prog(80)
            if self._cancelled: return

            self._log("[3/3] Finalizing...", "info")
            self._write_marker(dest, venv_py if os.path.isfile(venv_py) else self._sys_py or "python")
            self._set_prog(100)
            self._log("", "dim")
            self._log("=== Update complete! ===", "ok")

        except Exception as exc:
            if not self._cancelled:
                self._log(f"ERROR: {exc}", "err")
        finally:
            if self._cancelled:
                self._log("Cancelled.", "warn")
            self.after(0, self._unlock)

    # ══════════════════════════════════════════════════════════════
    #  UNINSTALL
    # ══════════════════════════════════════════════════════════════
    def _do_uninstall_start(self):
        dest = self._var_path.get().strip()
        if not messagebox.askyesno("Confirm",
                f"Remove LocalOCR from:\n{dest}\n\nThis deletes ALL files including the venv.",
                icon="warning"):
            return
        self._clear_log()
        self._lock()
        self._set_prog(0)
        threading.Thread(target=self._do_uninstall, daemon=True).start()

    def _do_uninstall(self):
        dest = self._var_path.get().strip()
        try:
            self._log("[1/3] Removing desktop shortcut...", "info")
            self._set_prog(20)
            self._remove_shortcut()

            self._log("[2/3] Removing install directory...", "info")
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
    #  LAUNCH — uses venv Python
    # ══════════════════════════════════════════════════════════════
    def _launch(self):
        dest = self._var_path.get().strip()
        gui  = os.path.join(dest, "gui_app.py")
        # Prefer venv pythonw, then venv python, then system
        vpy = _venv_python(dest)
        if os.name == "nt":
            vpyw = vpy.replace("python.exe", "pythonw.exe")
            if os.path.isfile(vpyw):
                vpy = vpyw
        if not os.path.isfile(vpy):
            vpy = self._sys_py or "python"
        if not os.path.isfile(gui):
            self._log(f"gui_app.py not found in {dest}", "err")
            return
        self._log(f"Launching with: {vpy}", "ok")
        try:
            kw = {}
            if os.name == "nt":
                kw["creationflags"] = (subprocess.CREATE_NO_WINDOW
                                       | subprocess.DETACHED_PROCESS)
            else:
                kw["start_new_session"] = True
            subprocess.Popen([vpy, gui], cwd=dest, **kw)
            self._log("App launched!", "ok")
        except Exception as exc:
            self._log(f"Launch error: {exc}", "err")

    # ══════════════════════════════════════════════════════════════
    #  DOWNLOAD SOURCE
    # ══════════════════════════════════════════════════════════════
    def _download(self, dest):
        # git clone
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
        if self._cancelled:
            return False
        # zip fallback
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
    #  FILE HELPERS — all use venv Python path
    # ══════════════════════════════════════════════════════════════
    def _write_marker(self, dest, py):
        with open(os.path.join(dest, MARKER), "w", encoding="utf-8") as f:
            f.write(f"version={APP_VERSION}\n")
            f.write(f"installed={time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"python={py}\n")
            f.write(f"venv={os.path.join(dest, VENV_DIR)}\n")
        self._log("  .installed_ok written", "dim")

    def _write_launcher(self, dest, venv_py):
        if os.name == "nt":
            bat = os.path.join(dest, f"{APP_NAME}.bat")
            with open(bat, "w", encoding="utf-8") as f:
                f.write('@echo off\n')
                f.write(f'title {APP_NAME}\n')
                f.write(f'cd /d "{dest}"\n')
                f.write(f'"{venv_py}" gui_app.py\n')
            self._log("  LocalOCR.bat created", "dim")
        else:
            sh = os.path.join(dest, "localocr.sh")
            with open(sh, "w", encoding="utf-8") as f:
                f.write('#!/usr/bin/env bash\n')
                f.write(f'cd "{dest}"\n')
                f.write(f'"{venv_py}" gui_app.py\n')
            os.chmod(sh, 0o755)
            self._log("  localocr.sh created", "dim")

    def _write_desktop_shortcut(self, dest, venv_py):
        if os.name != "nt":
            try:
                desk = os.path.join(os.path.expanduser("~"), "Desktop")
                if os.path.isdir(desk):
                    df = os.path.join(desk, f"{APP_NAME}.desktop")
                    with open(df, "w") as f:
                        f.write("[Desktop Entry]\nType=Application\n")
                        f.write(f"Name={APP_NAME}\n")
                        f.write(f'Exec="{venv_py}" "{dest}/gui_app.py"\n')
                        f.write(f"Path={dest}\nTerminal=false\n")
                    os.chmod(df, 0o755)
                    self._log(f"  Desktop shortcut: {df}", "ok")
            except Exception as exc:
                self._log(f"  Desktop shortcut failed: {exc}", "warn")
            return

        # Windows — VBScript for shortcut creation
        try:
            desktop = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")
            if not os.path.isdir(desktop):
                self._log("  Desktop folder not found", "warn")
                return
            lnk = os.path.join(desktop, f"{APP_NAME}.lnk")
            gui = os.path.join(dest, "gui_app.py")
            # Use pythonw from venv for windowless launch
            pyw = venv_py.replace("python.exe", "pythonw.exe")
            if not os.path.isfile(pyw):
                pyw = venv_py
            vbs = (
                'Set ws = CreateObject("WScript.Shell")\n'
                f'Set sc = ws.CreateShortcut("{lnk}")\n'
                f'sc.TargetPath = "{pyw}"\n'
                f'sc.Arguments = """{gui}"""\n'
                f'sc.WorkingDirectory = "{dest}"\n'
                f'sc.Description = "{APP_NAME} - PDF to DOCX OCR"\n'
                'sc.Save\n'
            )
            vbs_file = os.path.join(tempfile.gettempdir(), "localocr_lnk.vbs")
            with open(vbs_file, "w", encoding="utf-8") as f:
                f.write(vbs)
            r = subprocess.run(["cscript", "//Nologo", vbs_file],
                               capture_output=True, text=True, timeout=15,
                               creationflags=subprocess.CREATE_NO_WINDOW)
            try: os.remove(vbs_file)
            except OSError: pass
            if os.path.isfile(lnk):
                self._log(f"  [OK] Desktop shortcut: {lnk}", "ok")
            else:
                self._log(f"  Shortcut may have failed (exit {r.returncode})", "warn")
                if r.stderr.strip():
                    self._log(f"  {r.stderr.strip()}", "warn")
        except Exception as exc:
            self._log(f"  Shortcut failed: {exc}", "warn")

    def _write_uninstaller(self, dest):
        if os.name != "nt":
            return
        home = os.environ.get("USERPROFILE", "")
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
            f.write(f'del "{home}\\Desktop\\LocalOCR.lnk" 2>nul\n')
            f.write(f'cd /d "{home}"\n')
            f.write(f'rmdir /s /q "{dest}"\n')
            f.write('echo Done! & pause\n')
        self._log("  uninstall.bat created", "dim")

    def _remove_shortcut(self):
        if os.name == "nt":
            lnk = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop",
                               f"{APP_NAME}.lnk")
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
