"""
Build LocalOCR Desktop App — PyInstaller EXE builder
Creates a standalone .exe for Windows with all dependencies bundled.

Usage:
    py -3 build_exe.py          # Build the exe
    py -3 build_exe.py --clean  # Clean previous build then rebuild
"""
import os
import sys
import shutil
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DIST_DIR = PROJECT_ROOT / "dist"
BUILD_DIR = PROJECT_ROOT / "build"
SPEC_FILE = PROJECT_ROOT / "LocalOCR.spec"
APP_NAME = "LocalOCR"
ENTRY_POINT = PROJECT_ROOT / "gui_app.py"


def ensure_pyinstaller():
    """Install PyInstaller if not already present."""
    try:
        import PyInstaller  # noqa: F401
        print("[OK] PyInstaller is installed.")
    except ImportError:
        print("[INFO] Installing PyInstaller…")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("[OK] PyInstaller installed.")


def clean():
    """Remove previous build artifacts."""
    for d in (DIST_DIR, BUILD_DIR):
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"[CLEAN] Removed {d}")
    if SPEC_FILE.exists():
        SPEC_FILE.unlink()
        print(f"[CLEAN] Removed {SPEC_FILE}")


def build():
    """Run PyInstaller to create the exe."""
    models_dir = PROJECT_ROOT / "models"
    src_dir = PROJECT_ROOT / "src"
    correction_data_dir = PROJECT_ROOT / "correction_data"
    env_example = PROJECT_ROOT / ".env.example"

    # Build data includes
    datas = [
        f"--add-data={src_dir};src",
    ]
    if models_dir.exists():
        datas.append(f"--add-data={models_dir};models")
    if correction_data_dir.exists():
        datas.append(f"--add-data={correction_data_dir};correction_data")
    if env_example.exists():
        datas.append(f"--add-data={env_example};.")

    # Hidden imports that PyInstaller may miss
    hidden_imports = [
        "--hidden-import=src",
        "--hidden-import=src.pipeline",
        "--hidden-import=src.ocr_engine",
        "--hidden-import=src.layout_detector",
        "--hidden-import=src.exporter",
        "--hidden-import=src.preprocessor",
        "--hidden-import=src.correction_store",
        "--hidden-import=src.services",
        "--hidden-import=PIL",
        "--hidden-import=PIL.ImageTk",
        "--hidden-import=fitz",
        "--hidden-import=cv2",
        "--hidden-import=numpy",
        "--hidden-import=dotenv",
        "--hidden-import=docx",
        "--hidden-import=bs4",
        "--hidden-import=lxml",
        "--hidden-import=htmldocx",
        "--hidden-import=easyocr",
        "--hidden-import=torch",
        "--hidden-import=torchvision",
    ]

    # Optional hidden imports (only add if installed)
    optional_modules = [
        "paddleocr", "paddlepaddle", "paddle",
        "pytesseract", "onnxruntime", "transformers",
        "doclayout_yolo",
    ]
    for mod in optional_modules:
        try:
            __import__(mod)
            hidden_imports.append(f"--hidden-import={mod}")
        except ImportError:
            pass

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--onedir",                   # one-directory mode (faster startup)
        "--windowed",                 # no console window
        "--noconfirm",
        "--clean",
        f"--distpath={DIST_DIR}",
        f"--workpath={BUILD_DIR}",
        *datas,
        *hidden_imports,
        "--collect-all=easyocr",
        "--collect-all=docx",
        str(ENTRY_POINT),
    ]

    print("\n[BUILD] Running PyInstaller…")
    print(f"  Command: {' '.join(cmd[:8])} …")
    print()
    subprocess.check_call(cmd)

    exe_path = DIST_DIR / APP_NAME / f"{APP_NAME}.exe"
    if exe_path.exists():
        print(f"\n{'='*60}")
        print(f"  BUILD SUCCESSFUL!")
        print(f"  EXE: {exe_path}")
        print(f"  Folder: {DIST_DIR / APP_NAME}")
        print(f"{'='*60}")
    else:
        print(f"\n[WARNING] Expected exe not found at {exe_path}")
        print(f"  Check the dist/ directory for output.")


def main():
    parser = argparse.ArgumentParser(description="Build LocalOCR desktop exe")
    parser.add_argument("--clean", action="store_true",
                        help="Clean build artifacts before building")
    parser.add_argument("--clean-only", action="store_true",
                        help="Only clean, do not build")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)

    if args.clean or args.clean_only:
        clean()
    if args.clean_only:
        return

    ensure_pyinstaller()
    build()


if __name__ == "__main__":
    main()
