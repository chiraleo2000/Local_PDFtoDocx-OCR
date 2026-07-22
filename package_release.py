"""
Package LocalOCR v0.5.5 release artifacts (4 files).

Outputs under dist/release/:
  1. LocalOCR-v0.5.5-windows-setup.exe   — GUI installer
  2. LocalOCR-v0.5.5-windows-x64.zip     — Windows source + install.bat
  3. LocalOCR-v0.5.5-linux-x64.zip       — Linux source + install.sh
  4. LocalOCR-v0.5.5-macos.zip           — macOS source + install.sh

If dist/LocalOCR/ (PyInstaller onedir) exists, it is also zipped as the
Windows portable bundle inside the windows-x64 zip when --prefer-bundle
is set; otherwise the source installer package is used.
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

VERSION = "0.5.5"
ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
OUT = DIST / "release"

# Files/dirs included in source platform packages
INCLUDE = [
    "app.py",
    "gui_app.py",
    "installer_gui.py",
    "install.bat",
    "install.sh",
    "requirements.txt",
    "pyproject.toml",
    "README.md",
    "LICENSE",
    ".env.example",
    "Dockerfile",
    "docker-compose.yml",
    "src",
    "models",
    "correction_data",
]

EXCLUDE_DIR_NAMES = {
    "__pycache__", ".pytest_cache", ".git", ".venv", ".venv312",
    "node_modules", "e2e_output", "dist", "build",
}


def _should_skip(path: Path) -> bool:
    return any(part in EXCLUDE_DIR_NAMES for part in path.parts)


def _add_tree(zf: zipfile.ZipFile, src: Path, arc_prefix: str) -> None:
    if src.is_file():
        zf.write(src, f"{arc_prefix}/{src.name}")
        return
    for p in src.rglob("*"):
        if not p.is_file() or _should_skip(p.relative_to(src)):
            continue
        rel = p.relative_to(src)
        zf.write(p, f"{arc_prefix}/{rel.as_posix()}")


def _write_source_zip(dest: Path, platform_label: str) -> None:
    root_name = f"LocalOCR-v{VERSION}-{platform_label}"
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        readme = (
            f"LocalOCR v{VERSION} ({platform_label})\n\n"
            "Install:\n"
            "  Windows:  run install.bat\n"
            "  Linux/macOS:  chmod +x install.sh && ./install.sh\n"
            "  Or: pip install -r requirements.txt && python gui_app.py\n"
        )
        zf.writestr(f"{root_name}/INSTALL.txt", readme)
        for name in INCLUDE:
            src = ROOT / name
            if not src.exists():
                continue
            if src.is_dir():
                _add_tree(zf, src, f"{root_name}/{name}")
            else:
                zf.write(src, f"{root_name}/{name}")
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"[OK] {dest.name}  ({size_mb:.1f} MB)")


def _zip_dir(src_dir: Path, dest: Path, arc_root: str) -> None:
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in src_dir.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(src_dir)
            if _should_skip(rel):
                continue
            zf.write(p, f"{arc_root}/{rel.as_posix()}")
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"[OK] {dest.name}  ({size_mb:.1f} MB)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prefer-bundle",
        action="store_true",
        help="Prefer dist/LocalOCR onedir for the Windows zip when present",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    # 1) Windows Setup.exe
    setup_src = DIST / "LocalOCR-Setup.exe"
    setup_dst = OUT / f"LocalOCR-v{VERSION}-windows-setup.exe"
    if not setup_src.exists():
        raise SystemExit(
            f"[ERROR] Missing {setup_src}. Run: py -3 build_installer.py --setup"
        )
    shutil.copy2(setup_src, setup_dst)
    print(f"[OK] {setup_dst.name}  "
          f"({setup_dst.stat().st_size / (1024 * 1024):.1f} MB)")

    # 2) Windows zip
    win_zip = OUT / f"LocalOCR-v{VERSION}-windows-x64.zip"
    bundle = DIST / "LocalOCR"
    if args.prefer_bundle and (bundle / "LocalOCR.exe").exists():
        _zip_dir(bundle, win_zip, f"LocalOCR-v{VERSION}-windows-x64")
    else:
        _write_source_zip(win_zip, "windows-x64")

    # 3) Linux zip
    _write_source_zip(OUT / f"LocalOCR-v{VERSION}-linux-x64.zip", "linux-x64")

    # 4) macOS zip
    _write_source_zip(OUT / f"LocalOCR-v{VERSION}-macos.zip", "macos")

    print(f"\n[DONE] 4 release files in {OUT}")
    for p in sorted(OUT.iterdir()):
        if p.is_file() and VERSION in p.name:
            print(f"  - {p.name}")


if __name__ == "__main__":
    main()
