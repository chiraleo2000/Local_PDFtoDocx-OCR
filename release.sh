#!/usr/bin/env bash
# ============================================================
#  LocalOCR Release v0.5.5 — Linux / macOS
#  1. Run tests   2. Docker deploy   3. Git commit + tag + push
#  (Windows .exe is built by release.bat on a Windows machine)
# ============================================================
set -euo pipefail
VERSION="0.5.5"
cd "$(dirname "$0")"

echo
echo "===== [1/3] Running tests ====="
python3 -m pytest tests/test_absolute_export.py -q

echo
echo "===== [2/3] Docker deploy (CPU) ====="
if command -v docker >/dev/null 2>&1; then
    docker compose up --build -d localocr
    echo "[OK] LocalOCR running at http://localhost:7870"
    echo "     GPU variant: docker compose --profile gpu up --build -d localocr-gpu"
else
    echo "[WARN] docker not found — skipping deploy."
fi

echo
echo "===== [3/3] Git commit + tag + push ====="
git add -A
git commit -m "Release v${VERSION} - absolute-position layout-faithful DOCX export (text frames, positioned real Word tables, OpenCV figure placement)" \
    || echo "[INFO] Nothing new to commit."
git tag -a "v${VERSION}" -m "LocalOCR v${VERSION}" 2>/dev/null \
    || echo "[INFO] Tag v${VERSION} already exists."
git push origin HEAD --tags

echo
echo "===== Release v${VERSION} complete ====="
