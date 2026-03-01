@echo off
REM ══════════════════════════════════════════════════════════════════════════
REM  LocalOCR Installer for Windows  v0.3.2
REM  Detailed progress reporting during installation.
REM ══════════════════════════════════════════════════════════════════════════
setlocal enabledelayedexpansion
title LocalOCR Installer v0.3.2
color 0B

echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║                                                              ║
echo  ║      LocalOCR — PDF to DOCX Converter                       ║
echo  ║      Installer v0.3.2                                        ║
echo  ║                                                              ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

set "SRC_DIR=%~dp0"
set "SRC_DIR=%SRC_DIR:~0,-1%"
set "STEP=0"
set "TOTAL_STEPS=9"

REM ── Helper: print step ─────────────────────────────────────────────────
:step1
set /a STEP+=1
echo.
echo  ────────────────────────────────────────────────────────────
echo   [%STEP%/%TOTAL_STEPS%] Checking Python installation...
echo  ────────────────────────────────────────────────────────────

where py >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   ERROR: Python launcher (py) not found.
    echo   Please install Python 3.10+ from https://www.python.org/downloads/
    echo   Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('py -3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"') do set PYVER=%%i
echo   [OK] Python %PYVER% found
for /f "tokens=*" %%i in ('py -3 -c "import sys; print(sys.executable)"') do set PYEXE=%%i
echo   [OK] Location: %PYEXE%

REM ── Step 2: Check tkinter ──────────────────────────────────────────────
set /a STEP+=1
echo.
echo  ────────────────────────────────────────────────────────────
echo   [%STEP%/%TOTAL_STEPS%] Checking tkinter (GUI toolkit)...
echo  ────────────────────────────────────────────────────────────

py -3 -c "import tkinter; print('  [OK] tkinter available')" 2>nul
if %errorlevel% neq 0 (
    echo   [WARN] tkinter not available - GUI may not work.
    echo   Reinstall Python with "tcl/tk" option enabled.
)

REM ── Step 3: Choose install directory ───────────────────────────────────
set /a STEP+=1
echo.
echo  ────────────────────────────────────────────────────────────
echo   [%STEP%/%TOTAL_STEPS%] Choose installation directory
echo  ────────────────────────────────────────────────────────────

set "DEFAULT_INSTALL=%USERPROFILE%\LocalOCR"
set /p "INSTALL_DIR=  Install to [%DEFAULT_INSTALL%]: "
if "%INSTALL_DIR%"=="" set "INSTALL_DIR=%DEFAULT_INSTALL%"
echo.
echo   Target: %INSTALL_DIR%

REM ── Step 4: Copy application files ────────────────────────────────────
set /a STEP+=1
echo.
echo  ────────────────────────────────────────────────────────────
echo   [%STEP%/%TOTAL_STEPS%] Copying application files...
echo  ────────────────────────────────────────────────────────────

if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

echo   Copying src\...
xcopy "%SRC_DIR%\src" "%INSTALL_DIR%\src\" /E /I /Y /Q >nul 2>&1
echo   [OK] src\

echo   Copying models\...
xcopy "%SRC_DIR%\models" "%INSTALL_DIR%\models\" /E /I /Y /Q >nul 2>&1
echo   [OK] models\

echo   Copying correction_data\...
xcopy "%SRC_DIR%\correction_data" "%INSTALL_DIR%\correction_data\" /E /I /Y /Q >nul 2>&1
echo   [OK] correction_data\

echo   Copying tests\...
xcopy "%SRC_DIR%\tests" "%INSTALL_DIR%\tests\" /E /I /Y /Q >nul 2>&1
echo   [OK] tests\

echo   Copying application files...
copy /Y "%SRC_DIR%\gui_app.py" "%INSTALL_DIR%\" >nul 2>&1
copy /Y "%SRC_DIR%\app.py" "%INSTALL_DIR%\" >nul 2>&1
copy /Y "%SRC_DIR%\requirements.txt" "%INSTALL_DIR%\" >nul 2>&1
if exist "%SRC_DIR%\.env" copy /Y "%SRC_DIR%\.env" "%INSTALL_DIR%\" >nul 2>&1
if exist "%SRC_DIR%\.env.example" copy /Y "%SRC_DIR%\.env.example" "%INSTALL_DIR%\" >nul 2>&1

echo installed > "%INSTALL_DIR%\.installed_ok"
echo   [OK] All files copied

REM ── Step 5: Upgrade pip ───────────────────────────────────────────────
set /a STEP+=1
echo.
echo  ────────────────────────────────────────────────────────────
echo   [%STEP%/%TOTAL_STEPS%] Upgrading pip...
echo  ────────────────────────────────────────────────────────────

py -3 -m pip install --upgrade pip >nul 2>&1
echo   [OK] pip is up to date

REM ── Step 6: Install Python dependencies ───────────────────────────────
set /a STEP+=1
echo.
echo  ════════════════════════════════════════════════════════════
echo   [%STEP%/%TOTAL_STEPS%] Installing Python dependencies
echo        (This is the longest step - please be patient)
echo  ════════════════════════════════════════════════════════════

REM 6a: PyTorch
py -3 -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   [6a] Installing PyTorch (CPU)...
    echo        This downloads ~400MB, may take several minutes...
    py -3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    if %errorlevel% equ 0 (
        echo   [OK] PyTorch installed
    ) else (
        echo   [WARN] PyTorch install had issues
    )
) else (
    echo   [6a] PyTorch already installed - skipping
)

REM 6b: Core packages
echo.
echo   [6b] Installing core packages (PyMuPDF, OpenCV, Pillow, numpy)...
py -3 -m pip install --quiet PyMuPDF opencv-python-headless Pillow numpy python-dotenv
echo   [OK] Core packages done

REM 6c: Document packages
echo.
echo   [6c] Installing document packages (python-docx, beautifulsoup4, lxml)...
py -3 -m pip install --quiet python-docx beautifulsoup4 lxml htmldocx requests
echo   [OK] Document packages done

REM 6d: OCR engines
echo.
echo   [6d] Installing OCR engines (EasyOCR, PaddleOCR)...
py -3 -m pip install --quiet easyocr paddleocr
echo   [OK] OCR engines done

REM 6e: AI / ML packages
echo.
echo   [6e] Installing AI/ML packages (transformers, onnxruntime, YOLO, dill)...
py -3 -m pip install --quiet transformers onnxruntime doclayout-yolo huggingface_hub dill
echo   [OK] AI/ML packages done

REM 6f: Remaining from requirements.txt
echo.
echo   [6f] Installing any remaining packages from requirements.txt...
py -3 -m pip install --quiet -r "%INSTALL_DIR%\requirements.txt"
if %errorlevel% neq 0 (
    echo   [WARN] Some packages may have failed. App may still work.
) else (
    echo   [OK] All requirements installed
)

REM ── Step 7: Verify installation ───────────────────────────────────────
set /a STEP+=1
echo.
echo  ────────────────────────────────────────────────────────────
echo   [%STEP%/%TOTAL_STEPS%] Verifying installation...
echo  ────────────────────────────────────────────────────────────

py -3 -c "import fitz; print('   [OK] PyMuPDF')"
py -3 -c "import cv2; print('   [OK] OpenCV')"
py -3 -c "import numpy; print('   [OK] numpy')"
py -3 -c "import PIL; print('   [OK] Pillow')"
py -3 -c "import docx; print('   [OK] python-docx')"
py -3 -c "import torch; print('   [OK] PyTorch')"
py -3 -c "import easyocr; print('   [OK] EasyOCR')"
py -3 -c "import doclayout_yolo; print('   [OK] DocLayout-YOLO')"
py -3 -c "import transformers; print('   [OK] Transformers')"

REM ── Step 8: Download DocLayout-YOLO model weights ──────────────────────
set /a STEP+=1
echo.
echo  ════════════════════════════════════════════════════════════
echo   [%STEP%/%TOTAL_STEPS%] Downloading DocLayout-YOLO model (~30 MB)...
echo  ════════════════════════════════════════════════════════════

set "MODEL_DIR=%INSTALL_DIR%\models\DocLayout-YOLO-DocStructBench"
set "MODEL_FILE=doclayout_yolo_docstructbench_imgsz1280_2501.pt"
set "MODEL_PT=%MODEL_DIR%\%MODEL_FILE%"

if exist "%MODEL_PT%" (
    echo   [OK] Model already present: %MODEL_PT%
) else (
    echo   Writing download helper...
    set "DL_SCRIPT=%TEMP%\_localocr_dl_model.py"
    (
        echo import os, sys, shutil, urllib.request
        echo model_dir = r"%MODEL_DIR%"
        echo model_pt  = r"%MODEL_PT%"
        echo hf_repo   = "juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501"
        echo hf_file   = "%MODEL_FILE%"
        echo direct    = "https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501/resolve/main/%MODEL_FILE%"
        echo os.makedirs(model_dir, exist_ok=True^)
        echo if os.path.isfile(model_pt^):
        echo     print("Already present:", model_pt^); sys.exit(0^)
        echo try:
        echo     from huggingface_hub import hf_hub_download
        echo     print("Downloading via huggingface_hub..."^)
        echo     cached = hf_hub_download(hf_repo, hf_file^)
        echo     shutil.copy2(cached, model_pt^)
        echo     print("Done:", model_pt^); sys.exit(0^)
        echo except Exception as e:
        echo     print("huggingface_hub failed:", e^)
        echo try:
        echo     print("Trying direct URL..."^)
        echo     urllib.request.urlretrieve(direct, model_pt^)
        echo     print("Done:", model_pt^); sys.exit(0^)
        echo except Exception as e:
        echo     print("Direct download failed:", e^); sys.exit(1^)
    ) > "%DL_SCRIPT%"
    py -3 "%DL_SCRIPT%"
    if %errorlevel% equ 0 (
        echo   [OK] Model downloaded: %MODEL_PT%
    ) else (
        echo   [WARN] Model download failed. The app will auto-retry on first launch.
        echo   Or download manually from:
        echo   https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench-imgsz1280-2501
    )
    if exist "%DL_SCRIPT%" del "%DL_SCRIPT%" >nul 2>&1
)

REM ── Step 9: Create shortcuts & launchers ──────────────────────────────
set /a STEP+=1
echo.
echo  ────────────────────────────────────────────────────────────
echo   [%STEP%/%TOTAL_STEPS%] Creating shortcuts and launchers...
echo  ────────────────────────────────────────────────────────────

REM Create launcher batch
echo @echo off> "%INSTALL_DIR%\LocalOCR.bat"
echo title LocalOCR>> "%INSTALL_DIR%\LocalOCR.bat"
echo cd /d "%INSTALL_DIR%">> "%INSTALL_DIR%\LocalOCR.bat"
echo py -3 gui_app.py>> "%INSTALL_DIR%\LocalOCR.bat"
echo   [OK] Launcher: %INSTALL_DIR%\LocalOCR.bat

REM Desktop shortcut
for /f "tokens=*" %%i in ('py -3 -c "import sys, os; print(os.path.join(os.path.dirname(sys.executable), 'pythonw.exe'))"') do set PYTHONW=%%i
if not exist "%PYTHONW%" (
    for /f "tokens=*" %%i in ('py -3 -c "import sys; print(sys.executable)"') do set PYTHONW=%%i
)
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%USERPROFILE%\Desktop\LocalOCR.lnk'); $s.TargetPath = '%PYTHONW%'; $s.Arguments = '\"%INSTALL_DIR%\gui_app.py\"'; $s.WorkingDirectory = '%INSTALL_DIR%'; $s.Description = 'LocalOCR - PDF to DOCX Converter'; $s.Save()"
if %errorlevel% equ 0 (
    echo   [OK] Desktop shortcut: LocalOCR
) else (
    echo   [WARN] Could not create desktop shortcut
)

REM Uninstaller
(
echo @echo off
echo title LocalOCR Uninstaller
echo echo Removing LocalOCR from %INSTALL_DIR% ...
echo rmdir /s /q "%INSTALL_DIR%"
echo del "%USERPROFILE%\Desktop\LocalOCR.lnk" 2^>nul
echo echo LocalOCR has been removed.
echo pause
) > "%INSTALL_DIR%\uninstall.bat"
echo   [OK] Uninstaller created

REM ── Done ───────────────────────────────────────────────────────────────
echo.
echo  ╔══════════════════════════════════════════════════════════════╗
echo  ║                                                              ║
echo  ║   INSTALLATION COMPLETE!                                     ║
echo  ║                                                              ║
echo  ║   App:        %INSTALL_DIR%
echo  ║   Launcher:   %INSTALL_DIR%\LocalOCR.bat
echo  ║   Shortcut:   Desktop\LocalOCR                               ║
echo  ║   Uninstall:  %INSTALL_DIR%\uninstall.bat
echo  ║                                                              ║
echo  ╚══════════════════════════════════════════════════════════════╝
echo.

echo  Starting LocalOCR...
cd /d "%INSTALL_DIR%"
start "" py -3 gui_app.py

echo  App launched! You can close this window.
timeout /t 5
exit /b 0
