"""
LocalOCR Installer — Windows EXE wrapper
Builds a small .exe that:
  1. Asks for install path
  2. Clones/downloads the repo
  3. Installs Python dependencies
  4. Creates desktop shortcut
  5. Launches the app

Build:  py -3 build_installer.py
Output: dist/LocalOCR-Setup.exe
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

APP_NAME = "LocalOCR"
APP_VERSION = "0.3.0"
REPO_URL = "https://github.com/chiraleo2000/Local_PDFtoDocx-OCR/archive/refs/tags/v0.3.0.zip"
REPO_CLONE = "https://github.com/chiraleo2000/Local_PDFtoDocx-OCR.git"
DEFAULT_INSTALL = os.path.join(os.environ.get("USERPROFILE", "C:\\"), "LocalOCR")

# ── Color palette ─────────────────────────────────────────────────────────
BG       = "#1a1b2e"
BG_MID   = "#232440"
BG_LIGHT = "#2d2f52"
ACCENT   = "#6c63ff"
ACCENT_H = "#857dff"
SUCCESS  = "#00d68f"
WARNING  = "#ffaa00"
ERROR    = "#ff3d71"
INFO     = "#00b4d8"
TEXT     = "#eaeaff"
TEXT_DIM = "#9899c2"
LOG_BG   = "#12132a"


class InstallerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_NAME} Installer v{APP_VERSION}")
        self.geometry("720x620")
        self.resizable(False, False)
        self.configure(bg=BG)
        self._cancelled = False
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        # Header gradient
        hdr = tk.Canvas(self, height=72, bg=BG, highlightthickness=0)
        hdr.pack(fill="x")
        for i in range(72):
            f = i / 72
            r = int(108 * (1 - f) + 0 * f)
            g = int(99 * (1 - f) + 180 * f)
            b = int(255 * (1 - f) + 216 * f)
            hdr.create_line(0, i, 720, i, fill=f"#{r:02x}{g:02x}{b:02x}")
        hdr.create_text(20, 24, text=f"LocalOCR Setup v{APP_VERSION}",
                        anchor="w", fill="white", font=("Segoe UI", 18, "bold"))
        hdr.create_text(20, 52, text="PDF to DOCX OCR Converter — Installer",
                        anchor="w", fill="#d0d0ff", font=("Segoe UI", 10))

        # Install path
        pf = tk.Frame(self, bg=BG_MID, padx=16, pady=12)
        pf.pack(fill="x", padx=16, pady=(12, 4))
        tk.Label(pf, text="Install Location:", bg=BG_MID, fg=INFO,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w")
        row = tk.Frame(pf, bg=BG_MID)
        row.pack(fill="x", pady=(4, 0))
        self._var_path = tk.StringVar(value=DEFAULT_INSTALL)
        self._ent_path = tk.Entry(row, textvariable=self._var_path,
                                  font=("Consolas", 10), bg=BG_LIGHT,
                                  fg=TEXT, relief="flat", insertbackground=TEXT)
        self._ent_path.pack(side="left", fill="x", expand=True, ipady=4)
        tk.Button(row, text="Browse...", command=self._browse_path,
                  bg=BG_LIGHT, fg=TEXT_DIM, relief="flat",
                  font=("Segoe UI", 9), cursor="hand2").pack(side="right", padx=(6, 0))

        # Python info
        inf = tk.Frame(self, bg=BG_MID, padx=16, pady=8)
        inf.pack(fill="x", padx=16, pady=4)
        py_exe = sys.executable
        py_ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        tk.Label(inf, text=f"Python {py_ver}  |  {py_exe}",
                 bg=BG_MID, fg=TEXT_DIM, font=("Consolas", 9)).pack(anchor="w")

        # Log
        lf = tk.Frame(self, bg=BG)
        lf.pack(fill="both", expand=True, padx=16, pady=4)
        tk.Label(lf, text="Installation Log:", bg=BG, fg=TEXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        self._log_widget = scrolledtext.ScrolledText(
            lf, wrap="word", font=("Consolas", 9), height=14,
            state="disabled", bg=LOG_BG, fg=SUCCESS,
            insertbackground=SUCCESS, selectbackground=ACCENT,
            relief="flat", bd=0)
        self._log_widget.pack(fill="both", expand=True)
        self._log_widget.tag_configure("info", foreground=INFO)
        self._log_widget.tag_configure("warn", foreground=WARNING)
        self._log_widget.tag_configure("err", foreground=ERROR)
        self._log_widget.tag_configure("dim", foreground=TEXT_DIM)
        self._log_widget.tag_configure("ok", foreground=SUCCESS)

        # Progress
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("I.Horizontal.TProgressbar",
                        troughcolor=BG_LIGHT, background=ACCENT,
                        darkcolor=ACCENT, lightcolor=ACCENT_H,
                        bordercolor=BG, thickness=8)
        self._progress = ttk.Progressbar(self, mode="determinate",
                                         maximum=100,
                                         style="I.Horizontal.TProgressbar")
        self._progress.pack(fill="x", padx=16, pady=6)

        # Buttons
        bf = tk.Frame(self, bg=BG, pady=10)
        bf.pack(fill="x", padx=16)
        self._btn_cancel = tk.Button(bf, text="  Cancel  ",
                                     command=self._on_close,
                                     bg=BG_LIGHT, fg=TEXT_DIM,
                                     activebackground=ERROR,
                                     activeforeground="white",
                                     font=("Segoe UI", 10), relief="flat",
                                     cursor="hand2")
        self._btn_cancel.pack(side="left")
        self._btn_install = tk.Button(bf, text="  Install  ",
                                      command=self._start_install,
                                      bg=ACCENT, fg="white",
                                      activebackground=ACCENT_H,
                                      activeforeground="white",
                                      font=("Segoe UI", 12, "bold"),
                                      relief="flat", cursor="hand2",
                                      padx=24, pady=4)
        self._btn_install.pack(side="right")

    def _browse_path(self):
        d = filedialog.askdirectory(title="Choose install folder",
                                    initialdir=self._var_path.get())
        if d:
            self._var_path.set(d)

    def _log(self, msg, tag=None):
        self._log_widget.configure(state="normal")
        if tag:
            self._log_widget.insert("end", msg + "\n", tag)
        else:
            self._log_widget.insert("end", msg + "\n")
        self._log_widget.see("end")
        self._log_widget.configure(state="disabled")
        self.update_idletasks()

    def _set_progress(self, val):
        self._progress.configure(value=val)
        self.update_idletasks()

    def _start_install(self):
        self._btn_install.configure(state="disabled")
        threading.Thread(target=self._run_install, daemon=True).start()

    def _run_install(self):
        dest = self._var_path.get().strip()
        if not dest:
            self.after(0, lambda: messagebox.showerror("Error", "Choose a path"))
            self.after(0, lambda: self._btn_install.configure(state="normal"))
            return

        try:
            # Step 1: Create directory (5%)
            self.after(0, lambda: self._log("[1/8] Creating install directory...", "info"))
            self.after(0, lambda: self._set_progress(5))
            os.makedirs(dest, exist_ok=True)
            self.after(0, lambda: self._log(f"  -> {dest}", "dim"))

            # Step 2: Download source (25%)
            self.after(0, lambda: self._log("[2/8] Downloading LocalOCR source...", "info"))
            self.after(0, lambda: self._set_progress(10))

            # Try git clone first
            cloned = False
            try:
                if shutil.which("git"):
                    self.after(0, lambda: self._log("  Using git clone...", "dim"))
                    subprocess.check_call(
                        ["git", "clone", "--depth", "1", "--branch", f"v{APP_VERSION}",
                         REPO_CLONE, dest + "_tmp"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=120)
                    # Move contents
                    tmp = dest + "_tmp"
                    for item in os.listdir(tmp):
                        s = os.path.join(tmp, item)
                        d2 = os.path.join(dest, item)
                        if os.path.isdir(s):
                            if os.path.exists(d2):
                                shutil.rmtree(d2)
                            shutil.copytree(s, d2)
                        else:
                            shutil.copy2(s, d2)
                    shutil.rmtree(tmp, ignore_errors=True)
                    cloned = True
                    self.after(0, lambda: self._log("  [OK] Source cloned", "ok"))
            except Exception as exc:
                self.after(0, lambda: self._log(f"  git clone failed: {exc}", "warn"))

            if not cloned:
                # Fall back to ZIP download
                self.after(0, lambda: self._log("  Downloading ZIP...", "dim"))
                zip_path = os.path.join(tempfile.gettempdir(), "localocr.zip")
                urllib.request.urlretrieve(REPO_URL, zip_path)
                self.after(0, lambda: self._set_progress(18))
                self.after(0, lambda: self._log("  Extracting...", "dim"))
                with zipfile.ZipFile(zip_path, "r") as zf:
                    zf.extractall(tempfile.gettempdir())
                    top = zf.namelist()[0].split("/")[0]
                extracted = os.path.join(tempfile.gettempdir(), top)
                for item in os.listdir(extracted):
                    s = os.path.join(extracted, item)
                    d2 = os.path.join(dest, item)
                    if os.path.isdir(s):
                        if os.path.exists(d2):
                            shutil.rmtree(d2)
                        shutil.copytree(s, d2)
                    else:
                        shutil.copy2(s, d2)
                shutil.rmtree(extracted, ignore_errors=True)
                os.remove(zip_path)
                self.after(0, lambda: self._log("  [OK] Source downloaded", "ok"))

            self.after(0, lambda: self._set_progress(25))

            # Step 3: Upgrade pip (30%)
            self.after(0, lambda: self._log("[3/8] Upgrading pip...", "info"))
            self._run_cmd([sys.executable, "-m", "pip", "install",
                           "--upgrade", "pip", "--quiet"])
            self.after(0, lambda: self._set_progress(30))

            # Step 4: Install PyTorch (50%)
            self.after(0, lambda: self._log("[4/8] Checking PyTorch...", "info"))
            try:
                __import__("torch")
                self.after(0, lambda: self._log("  Already installed, skipping", "dim"))
            except ImportError:
                self.after(0, lambda: self._log(
                    "  Installing PyTorch CPU (~400MB)...", "info"))
                self._run_cmd([sys.executable, "-m", "pip", "install",
                               "torch", "torchvision",
                               "--index-url",
                               "https://download.pytorch.org/whl/cpu"])
            self.after(0, lambda: self._set_progress(50))

            # Step 5: Install core deps (60%)
            self.after(0, lambda: self._log("[5/8] Installing core packages...", "info"))
            self._run_cmd([sys.executable, "-m", "pip", "install", "--quiet",
                           "PyMuPDF", "opencv-python-headless", "Pillow",
                           "numpy", "python-dotenv", "python-docx",
                           "beautifulsoup4", "lxml", "htmldocx", "requests"])
            self.after(0, lambda: self._set_progress(60))

            # Step 6: Install OCR engines (75%)
            self.after(0, lambda: self._log("[6/8] Installing OCR engines...", "info"))
            self._run_cmd([sys.executable, "-m", "pip", "install", "--quiet",
                           "easyocr", "paddleocr"])
            self.after(0, lambda: self._set_progress(75))

            # Step 7: Install AI/ML + remaining (85%)
            self.after(0, lambda: self._log("[7/8] Installing AI/ML packages...", "info"))
            req = os.path.join(dest, "requirements.txt")
            if os.path.exists(req):
                self._run_cmd([sys.executable, "-m", "pip", "install",
                               "--quiet", "-r", req])
            else:
                self._run_cmd([sys.executable, "-m", "pip", "install", "--quiet",
                               "transformers", "onnxruntime",
                               "doclayout-yolo", "huggingface_hub"])
            self.after(0, lambda: self._set_progress(85))

            # Step 8: Create shortcuts (95%)
            self.after(0, lambda: self._log("[8/8] Creating shortcuts...", "info"))

            # .installed_ok marker
            with open(os.path.join(dest, ".installed_ok"), "w") as f:
                f.write(f"installed {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

            # Launcher bat
            bat = os.path.join(dest, "LocalOCR.bat")
            with open(bat, "w") as f:
                f.write(f'@echo off\ntitle LocalOCR\ncd /d "{dest}"\npy -3 gui_app.py\n')

            # Desktop shortcut
            try:
                desktop = os.path.join(os.environ.get("USERPROFILE", ""), "Desktop")
                lnk = os.path.join(desktop, "LocalOCR.lnk")
                # Find pythonw
                pydir = os.path.dirname(sys.executable)
                pythonw = os.path.join(pydir, "pythonw.exe")
                if not os.path.exists(pythonw):
                    pythonw = sys.executable
                gui_py = os.path.join(dest, "gui_app.py")
                ps = (f'$ws=New-Object -ComObject WScript.Shell;'
                      f'$s=$ws.CreateShortcut("{lnk}");'
                      f'$s.TargetPath="{pythonw}";'
                      f'$s.Arguments=\'"{gui_py}"\';'
                      f'$s.WorkingDirectory="{dest}";'
                      f'$s.Description="LocalOCR - PDF to DOCX";'
                      f'$s.Save()')
                subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                               capture_output=True, timeout=10)
                self.after(0, lambda: self._log("  Desktop shortcut created", "ok"))
            except Exception:
                self.after(0, lambda: self._log("  Shortcut creation skipped", "warn"))

            # Uninstaller
            uninst = os.path.join(dest, "uninstall.bat")
            with open(uninst, "w") as f:
                f.write(f'@echo off\ntitle Uninstall LocalOCR\n')
                f.write(f'echo Removing LocalOCR...\n')
                f.write(f'rmdir /s /q "{dest}"\n')
                home = os.environ.get("USERPROFILE", "")
                f.write(f'del "{home}\\Desktop\\LocalOCR.lnk" 2>nul\n')
                f.write(f'echo Done.\npause\n')

            self.after(0, lambda: self._set_progress(95))

            # Verify
            self.after(0, lambda: self._log("", "dim"))
            self.after(0, lambda: self._log("Verifying key packages...", "info"))
            for mod_name in ["fitz", "cv2", "PIL", "torch", "easyocr"]:
                try:
                    __import__(mod_name)
                    self.after(0, lambda m=mod_name: self._log(f"  [OK] {m}", "ok"))
                except Exception:
                    self.after(0, lambda m=mod_name: self._log(f"  [WARN] {m} missing", "warn"))

            self.after(0, lambda: self._set_progress(100))
            self.after(0, lambda: self._log("", "dim"))
            self.after(0, lambda: self._log(
                "Installation complete! Launching LocalOCR...", "ok"))

            # Launch
            time.sleep(1)
            gui = os.path.join(dest, "gui_app.py")
            if os.path.exists(gui):
                subprocess.Popen([sys.executable, gui], cwd=dest)
                self.after(2000, self.destroy)
            else:
                self.after(0, lambda: self._log(
                    "gui_app.py not found — launch manually", "warn"))
                self.after(0, lambda: self._btn_install.configure(state="normal"))

        except Exception as exc:
            err = str(exc)
            self.after(0, lambda: self._log(f"ERROR: {err}", "err"))
            self.after(0, lambda: self._btn_install.configure(state="normal"))

    def _run_cmd(self, cmd):
        """Run a subprocess, streaming output to log."""
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in proc.stdout:
                line = line.rstrip()
                if line:
                    self.after(0, lambda l=line: self._log(f"  {l}", "dim"))
            proc.wait()
            if proc.returncode != 0:
                self.after(0, lambda: self._log(
                    f"  (exit code {proc.returncode})", "warn"))
        except Exception as exc:
            self.after(0, lambda: self._log(f"  cmd error: {exc}", "err"))

    def _on_close(self):
        self._cancelled = True
        self.destroy()


def main():
    app = InstallerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
