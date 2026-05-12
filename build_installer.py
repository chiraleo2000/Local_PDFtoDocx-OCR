"""
Build LocalOCR distributable packages.

Usage:
    py -3 build_installer.py            # Build both: Setup installer + app bundle
    py -3 build_installer.py --setup    # Build only LocalOCR-Setup.exe
    py -3 build_installer.py --app      # Build only LocalOCR app (onedir)

Outputs:
    dist/LocalOCR-Setup.exe             Single-file GUI installer  (~12 MB)
    dist/LocalOCR/LocalOCR.exe          One-dir app bundle with models alongside
"""
import os
import sys
import time
import subprocess
import shutil
import argparse
import importlib.util
from pathlib import Path

ROOT  = Path(__file__).resolve().parent
DIST  = ROOT / "dist"
BUILD = ROOT / "build"


# ── helpers ───────────────────────────────────────────────────────────────────

def _clean_spec(name: str) -> None:
    for f in ROOT.glob(f"{name}.spec"):
        f.unlink()

def _clean_build(subdir: str) -> None:
    d = BUILD / subdir
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)

def _require_modules(modules: list[str]) -> None:
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        print("[ERROR] Missing required build/runtime modules:", ", ".join(missing))
        print("        Install requirements with: py -3 -m pip install -r requirements.txt")
        sys.exit(1)

def _require_supported_python() -> None:
    if sys.version_info >= (3, 13):
        print("[ERROR] LocalOCR full OCR deployment requires Python 3.12.")
        print("        PaddlePaddle/PaddleOCR wheels are not available for Python 3.13+.")
        print("        Recreate .venv with: uv python install 3.12; uv venv --python 3.12 .venv")
        sys.exit(1)


# ══════════════════════════════════════════════════════════════════════════════
# 1 — LocalOCR-Setup.exe  (single-file installer)
# ══════════════════════════════════════════════════════════════════════════════
def build_setup_exe() -> None:
    """Build the GUI installer (LocalOCR-Setup.exe) — single file, ~12 MB."""
    _clean_spec("LocalOCR-Setup")
    _clean_build("setup")

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name",    "LocalOCR-Setup",
        "--onefile",
        "--windowed",
        "--noconfirm",
        "--clean",
        f"--distpath={DIST}",
        f"--workpath={BUILD / 'setup'}",
        str(ROOT / "installer_gui.py"),
    ]
    print("\n[BUILD] Building LocalOCR-Setup.exe …")
    subprocess.check_call(cmd)

    exe = DIST / "LocalOCR-Setup.exe"
    if not exe.exists():
        print("[ERROR] LocalOCR-Setup.exe not found after build")
        sys.exit(1)
    size_mb = exe.stat().st_size / (1024 * 1024)
    print(f"[OK]   LocalOCR-Setup.exe  ({size_mb:.1f} MB)  →  {exe}")


# ══════════════════════════════════════════════════════════════════════════════
# 2 — LocalOCR  (one-dir app bundle — models sit alongside the .exe)
# ══════════════════════════════════════════════════════════════════════════════
def build_app_onedir() -> None:
    """Build the main app as a one-dir bundle and copy models/ alongside it."""
    _require_supported_python()
    _require_modules([
        "doclayout_yolo", "fitz", "PIL", "numpy",
        "huggingface_hub", "lap", "shapely",
        "easyocr", "paddleocr", "onnxruntime",
        "transformers", "pytesseract", "Cython", "skimage",
    ])

    _clean_spec("LocalOCR")
    _clean_build("app")

    app_dist = DIST / "LocalOCR"

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name",    "LocalOCR",
        "--onedir",
        "--windowed",
        "--noconfirm",
        "--clean",
        f"--distpath={DIST}",
        f"--workpath={BUILD / 'app'}",

        # ── Collect the src/ package and every submodule ──────────────────────
        "--collect-submodules", "src",

        # ── Explicit hidden imports for dynamic (lazy) loading in gui_app.py ──
        "--hidden-import", "src.pipeline",
        "--hidden-import", "src.services",
        "--hidden-import", "src.ocr_engine",
        "--hidden-import", "src.layout_detector",
        "--hidden-import", "src.preprocessor",
        "--hidden-import", "src.exporter",
        "--hidden-import", "src.correction_store",

        # ── OCR/runtime modules loaded lazily at runtime ─────────────────────
        "--hidden-import", "easyocr",
        "--hidden-import", "paddleocr",
        "--hidden-import", "onnxruntime",
        "--hidden-import", "transformers",
        "--hidden-import", "pytesseract",
        "--hidden-import", "doclayout_yolo",
        "--hidden-import", "lap",
        "--hidden-import", "shapely",
        "--hidden-import", "skimage.morphology",

        # ── Tkinter image support ─────────────────────────────────────────────
        "--hidden-import", "PIL._tkinter_finder",

        # ── Collect heavy ML packages (all submodules + package data) ─────────
        "--collect-all",  "doclayout_yolo",
        "--collect-all",  "easyocr",
        "--collect-all",  "paddleocr",
        "--collect-all",  "onnxruntime",
        "--collect-all",  "transformers",
        "--collect-all",  "Cython",
        "--collect-all",  "skimage",

        str(ROOT / "gui_app.py"),
    ]
    print("\n[BUILD] Building LocalOCR app (onedir) …")
    subprocess.check_call(cmd)

    # Locate exe (Windows vs Linux/macOS)
    exe = app_dist / "LocalOCR.exe"
    if not exe.exists():
        exe = app_dist / "LocalOCR"
    if not exe.exists():
        print("[ERROR] LocalOCR executable not found after build")
        sys.exit(1)

    # ── Copy models/ directory next to the exe ────────────────────────────────
    src_models = ROOT / "models"
    dst_models = app_dist / "models"
    if src_models.is_dir():
        print(f"\n[COPY] models/ → {dst_models} …")
        if dst_models.exists():
            shutil.rmtree(dst_models)
        shutil.copytree(str(src_models), str(dst_models))
        print("[OK]   models/ copied alongside app")
    else:
        print(f"[WARN] models/ not found at {src_models}")
        print("       Place models/ in the same folder as LocalOCR.exe before running.")

    # ── Write .installed_ok so the SetupWizard is skipped on first launch ─────
    marker = app_dist / ".installed_ok"
    marker.write_text(
        f"version=bundled\ninstalled={time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        encoding="utf-8",
    )
    print("[OK]   .installed_ok marker written")

    # ── Copy optional env files ───────────────────────────────────────────────
    for fname in (".env.example", ".env"):
        if (ROOT / fname).exists():
            shutil.copy2(str(ROOT / fname), str(app_dist / fname))

    total_mb = (
        sum(fi.stat().st_size for fi in app_dist.rglob("*") if fi.is_file())
        / (1024 * 1024)
    )
    print(f"\n[OK]   LocalOCR bundle: {app_dist}  ({total_mb:.1f} MB total)")
    print(f"       Run: {exe}")


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    """Parse arguments and run selected build targets."""
    parser = argparse.ArgumentParser(
        description="Build LocalOCR distributable packages.")
    parser.add_argument(
        "--setup", action="store_true",
        help="Build only the LocalOCR-Setup.exe installer")
    parser.add_argument(
        "--app", action="store_true",
        help="Build only the LocalOCR app (onedir bundle)")
    args = parser.parse_args()

    build_both  = not args.setup and not args.app
    build_setup = build_both or args.setup
    build_app   = build_both or args.app

    os.chdir(ROOT)
    DIST.mkdir(exist_ok=True)
    BUILD.mkdir(exist_ok=True)

    if build_setup:
        build_setup_exe()
    if build_app:
        build_app_onedir()

    print(f"\n[DONE] Build complete.  Output folder: {DIST}")


if __name__ == "__main__":
    main()
