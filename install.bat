@echo off
REM ══════════════════════════════════════════════════════════════════════════
REM  LocalOCR Installer for Windows
REM  Installs dependencies, sets up the app, creates desktop shortcut,
REM  and launches the GUI.
REM ══════════════════════════════════════════════════════════════════════════
setlocal enabledelayedexpansion
title LocalOCR Installer

echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║       LocalOCR — PDF to DOCX Converter              ║
echo  ║       Installer v1.0                                 ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

REM ── Detect source directory (where this script lives) ──────────────────
set "SRC_DIR=%~dp0"
set "SRC_DIR=%SRC_DIR:~0,-1%"

REM ── Check for Python ───────────────────────────────────────────────────
echo [1/6] Checking Python...
where py >nul 2>&1
if %errorlevel% neq 0 (
    echo  ERROR: Python launcher (py) not found.
    echo  Please install Python 3.10+ from https://www.python.org/downloads/
    echo  Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)
for /f "tokens=*" %%i in ('py -3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYVER=%%i
echo  Found Python %PYVER%

REM ── Ask for installation directory ─────────────────────────────────────
echo.
echo [2/6] Choose installation directory
set "DEFAULT_INSTALL=%USERPROFILE%\LocalOCR"
set /p "INSTALL_DIR=  Install to [%DEFAULT_INSTALL%]: "
if "%INSTALL_DIR%"=="" set "INSTALL_DIR=%DEFAULT_INSTALL%"

echo  Installing to: %INSTALL_DIR%
echo.

REM ── Create install directory & copy files ──────────────────────────────
echo [3/6] Copying application files...
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"

REM Copy all needed files
xcopy "%SRC_DIR%\src" "%INSTALL_DIR%\src\" /E /I /Y /Q >nul 2>&1
xcopy "%SRC_DIR%\models" "%INSTALL_DIR%\models\" /E /I /Y /Q >nul 2>&1
xcopy "%SRC_DIR%\correction_data" "%INSTALL_DIR%\correction_data\" /E /I /Y /Q >nul 2>&1
xcopy "%SRC_DIR%\tests" "%INSTALL_DIR%\tests\" /E /I /Y /Q >nul 2>&1

copy /Y "%SRC_DIR%\gui_app.py" "%INSTALL_DIR%\" >nul 2>&1
copy /Y "%SRC_DIR%\app.py" "%INSTALL_DIR%\" >nul 2>&1
copy /Y "%SRC_DIR%\requirements.txt" "%INSTALL_DIR%\" >nul 2>&1
if exist "%SRC_DIR%\.env" copy /Y "%SRC_DIR%\.env" "%INSTALL_DIR%\" >nul 2>&1
if exist "%SRC_DIR%\.env.example" copy /Y "%SRC_DIR%\.env.example" "%INSTALL_DIR%\" >nul 2>&1

REM Write marker file so app knows it's installed
echo installed> "%INSTALL_DIR%\.installed"
echo  Files copied.

REM ── Install Python dependencies ────────────────────────────────────────
echo.
echo [4/6] Installing Python dependencies (this may take several minutes)...
echo  This will install: PyMuPDF, OpenCV, EasyOCR, PaddleOCR, YOLO, etc.
echo.

py -3 -m pip install --upgrade pip >nul 2>&1

REM Install torch CPU (smaller) unless user already has GPU torch
py -3 -c "import torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo  Installing PyTorch (CPU)...
    py -3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
) else (
    echo  PyTorch already installed, skipping.
)

echo  Installing remaining dependencies...
py -3 -m pip install -r "%INSTALL_DIR%\requirements.txt"
if %errorlevel% neq 0 (
    echo.
    echo  WARNING: Some dependencies may have failed to install.
    echo  The app may still work with reduced functionality.
    echo.
)

REM ── Create Desktop Shortcut ────────────────────────────────────────────
echo.
echo [5/6] Creating desktop shortcut...

REM Find pythonw.exe for no-console launch
for /f "tokens=*" %%i in ('py -3 -c "import sys, os; print(os.path.join(os.path.dirname(sys.executable), 'pythonw.exe'))"') do set PYTHONW=%%i
if not exist "%PYTHONW%" (
    for /f "tokens=*" %%i in ('py -3 -c "import sys; print(sys.executable)"') do set PYTHONW=%%i
)

REM Create shortcut via PowerShell
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%USERPROFILE%\Desktop\LocalOCR.lnk'); $s.TargetPath = '%PYTHONW%'; $s.Arguments = '\"%INSTALL_DIR%\gui_app.py\"'; $s.WorkingDirectory = '%INSTALL_DIR%'; $s.Description = 'LocalOCR - PDF to DOCX Converter'; $s.Save()"

if %errorlevel% equ 0 (
    echo  Desktop shortcut created: %USERPROFILE%\Desktop\LocalOCR.lnk
) else (
    echo  WARNING: Could not create desktop shortcut.
)

REM ── Create launcher batch file ─────────────────────────────────────────
echo @echo off> "%INSTALL_DIR%\LocalOCR.bat"
echo title LocalOCR>> "%INSTALL_DIR%\LocalOCR.bat"
echo cd /d "%INSTALL_DIR%">> "%INSTALL_DIR%\LocalOCR.bat"
echo py -3 gui_app.py>> "%INSTALL_DIR%\LocalOCR.bat"
echo  Launcher created: %INSTALL_DIR%\LocalOCR.bat

REM ── Create uninstaller ─────────────────────────────────────────────────
(
echo @echo off
echo title LocalOCR Uninstaller
echo echo Removing LocalOCR from %INSTALL_DIR% ...
echo rmdir /s /q "%INSTALL_DIR%"
echo del "%USERPROFILE%\Desktop\LocalOCR.lnk" 2^>nul
echo echo LocalOCR has been removed.
echo pause
) > "%INSTALL_DIR%\uninstall.bat"
echo  Uninstaller created.

REM ── Launch the app ─────────────────────────────────────────────────────
echo.
echo [6/6] Launching LocalOCR...
echo.
echo  ╔══════════════════════════════════════════════════════╗
echo  ║  Installation complete!                              ║
echo  ║  App installed to: %INSTALL_DIR%
echo  ║  Desktop shortcut: LocalOCR                          ║
echo  ║  To uninstall: run uninstall.bat in install folder   ║
echo  ╚══════════════════════════════════════════════════════╝
echo.

cd /d "%INSTALL_DIR%"
start "" py -3 gui_app.py

echo  App is starting... You can close this window.
timeout /t 5
exit /b 0
