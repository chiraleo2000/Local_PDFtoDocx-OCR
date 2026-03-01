"""
Build the lightweight LocalOCR Setup .exe
Only bundles the installer GUI (tkinter + stdlib) — no torch/easyocr.

Usage:  py -3 build_installer.py
Output: dist/LocalOCR-Setup.exe  (single file, ~12 MB)
"""
import os, sys, subprocess, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
BUILD = ROOT / "build"

def main():
    os.chdir(ROOT)
    # Clean
    for d in [DIST, BUILD]:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
    for f in ROOT.glob("*.spec"):
        f.unlink()

    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", "LocalOCR-Setup",
        "--onefile",
        "--windowed",
        "--noconfirm",
        "--clean",
        f"--distpath={DIST}",
        f"--workpath={BUILD}",
        str(ROOT / "installer_gui.py"),
    ]
    print("[BUILD] Building LocalOCR-Setup.exe …")
    subprocess.check_call(cmd)

    exe = DIST / "LocalOCR-Setup.exe"
    if exe.exists():
        size_mb = exe.stat().st_size / (1024 * 1024)
        print(f"\n[OK] Built: {exe}  ({size_mb:.1f} MB)")
    else:
        print("[ERROR] Build failed — exe not found")
        sys.exit(1)

if __name__ == "__main__":
    main()
