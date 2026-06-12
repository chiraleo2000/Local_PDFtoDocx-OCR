@echo off
REM ============================================================
REM  LocalOCR Release v0.5.0 — Windows
REM  1. Run tests   2. Build LocalOCR-Setup.exe + app bundle
REM  3. Docker deploy   4. Git commit + tag + push
REM ============================================================
setlocal
set VERSION=0.5.0
cd /d "%~dp0"

echo.
echo ===== [1/4] Running tests =====
py -3 -m pytest tests/test_absolute_export.py -q
if errorlevel 1 (
    echo [ERROR] Tests failed - aborting release.
    exit /b 1
)

echo.
echo ===== [2/4] Building LocalOCR-Setup.exe + app bundle =====
py -3 build_installer.py
if errorlevel 1 (
    echo [WARN] Installer build failed - continuing without exe.
) else (
    echo [OK] dist\LocalOCR-Setup.exe and dist\LocalOCR\LocalOCR.exe built.
)

echo.
echo ===== [3/4] Docker deploy (CPU) =====
docker compose up --build -d localocr
if errorlevel 1 (
    echo [WARN] Docker deploy failed - is Docker Desktop running?
) else (
    echo [OK] LocalOCR running at http://localhost:7870
)

echo.
echo ===== [4/4] Git commit + tag + push =====
git add -A
git commit -m "Release v%VERSION% - absolute-position layout-faithful DOCX export (text frames, positioned real Word tables, OpenCV figure placement)"
git tag -a v%VERSION% -m "LocalOCR v%VERSION%"
git push origin HEAD --tags
if errorlevel 1 (
    echo [ERROR] Push failed - check credentials/remote, then run: git push origin HEAD --tags
    exit /b 1
)

echo.
echo ===== Release v%VERSION% complete =====
endlocal
