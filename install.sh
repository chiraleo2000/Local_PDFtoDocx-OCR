#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
#  LocalOCR Installer for Linux / macOS
#  Installs dependencies, sets up the app, creates desktop launcher, and
#  launches the GUI.
# ══════════════════════════════════════════════════════════════════════════
set -e

echo ""
echo " ╔══════════════════════════════════════════════════════╗"
echo " ║       LocalOCR — PDF to DOCX Converter              ║"
echo " ║       Installer v1.0                                 ║"
echo " ╚══════════════════════════════════════════════════════╝"
echo ""

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Check Python ─────────────────────────────────────────────────────────
echo "[1/6] Checking Python..."
PYTHON=""
for cmd in python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || true)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -ge 3 ] && [ "$minor" -ge 10 ] 2>/dev/null; then
            PYTHON="$cmd"
            echo "  Found $cmd ($ver)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "  ERROR: Python 3.10+ not found."
    echo "  Install with: sudo apt install python3 python3-pip python3-tk"
    echo "  or: brew install python3 python-tk"
    exit 1
fi

# ── Ensure tkinter is available ──────────────────────────────────────────
if ! "$PYTHON" -c "import tkinter" &>/dev/null; then
    echo "  WARNING: tkinter not found. Installing..."
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y python3-tk
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y python3-tkinter
    elif command -v brew &>/dev/null; then
        brew install python-tk
    else
        echo "  ERROR: Cannot install tkinter automatically."
        echo "  Please install python3-tk for your distribution."
        exit 1
    fi
fi

# ── Ask for installation directory ───────────────────────────────────────
echo ""
echo "[2/6] Choose installation directory"
DEFAULT_INSTALL="$HOME/LocalOCR"
read -p "  Install to [$DEFAULT_INSTALL]: " INSTALL_DIR
INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_INSTALL}"
echo "  Installing to: $INSTALL_DIR"
echo ""

# ── Copy files ───────────────────────────────────────────────────────────
echo "[3/6] Copying application files..."
mkdir -p "$INSTALL_DIR"

cp -r "$SRC_DIR/src" "$INSTALL_DIR/"
[ -d "$SRC_DIR/models" ] && cp -r "$SRC_DIR/models" "$INSTALL_DIR/"
[ -d "$SRC_DIR/correction_data" ] && cp -r "$SRC_DIR/correction_data" "$INSTALL_DIR/"
[ -d "$SRC_DIR/tests" ] && cp -r "$SRC_DIR/tests" "$INSTALL_DIR/"

cp -f "$SRC_DIR/gui_app.py" "$INSTALL_DIR/"
cp -f "$SRC_DIR/app.py" "$INSTALL_DIR/"
cp -f "$SRC_DIR/requirements.txt" "$INSTALL_DIR/"
[ -f "$SRC_DIR/.env" ] && cp -f "$SRC_DIR/.env" "$INSTALL_DIR/"
[ -f "$SRC_DIR/.env.example" ] && cp -f "$SRC_DIR/.env.example" "$INSTALL_DIR/"

touch "$INSTALL_DIR/.installed"
echo "  Files copied."

# ── Install dependencies ────────────────────────────────────────────────
echo ""
echo "[4/6] Installing Python dependencies (this may take several minutes)..."
"$PYTHON" -m pip install --upgrade pip 2>/dev/null || true

# Install torch CPU if not present
if ! "$PYTHON" -c "import torch" &>/dev/null; then
    echo "  Installing PyTorch (CPU)..."
    "$PYTHON" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else
    echo "  PyTorch already installed, skipping."
fi

echo "  Installing remaining dependencies..."
"$PYTHON" -m pip install -r "$INSTALL_DIR/requirements.txt" || {
    echo ""
    echo "  WARNING: Some dependencies may have failed."
    echo "  The app may still work with reduced functionality."
}

# ── Create desktop launcher ─────────────────────────────────────────────
echo ""
echo "[5/6] Creating desktop launcher..."

# Create shell launcher
cat > "$INSTALL_DIR/localocr.sh" << LAUNCHER
#!/usr/bin/env bash
cd "$INSTALL_DIR"
$PYTHON gui_app.py "\$@"
LAUNCHER
chmod +x "$INSTALL_DIR/localocr.sh"

# Linux .desktop file
if [ "$(uname)" = "Linux" ]; then
    DESKTOP_DIR="$HOME/Desktop"
    APPS_DIR="$HOME/.local/share/applications"
    mkdir -p "$APPS_DIR"

    cat > "$APPS_DIR/localocr.desktop" << DESKTOP
[Desktop Entry]
Type=Application
Name=LocalOCR
Comment=PDF to DOCX OCR Converter
Exec=$INSTALL_DIR/localocr.sh
Path=$INSTALL_DIR
Terminal=false
Categories=Office;Utility;
DESKTOP
    chmod +x "$APPS_DIR/localocr.desktop"

    if [ -d "$DESKTOP_DIR" ]; then
        cp "$APPS_DIR/localocr.desktop" "$DESKTOP_DIR/"
        chmod +x "$DESKTOP_DIR/localocr.desktop"
        echo "  Desktop shortcut created."
    fi
    echo "  Application menu entry created."
fi

# macOS: create simple alias
if [ "$(uname)" = "Darwin" ]; then
    ln -sf "$INSTALL_DIR/localocr.sh" "/usr/local/bin/localocr" 2>/dev/null || true
    echo "  Command 'localocr' available (or run localocr.sh directly)."
fi

# Create uninstaller
cat > "$INSTALL_DIR/uninstall.sh" << UNINSTALL
#!/usr/bin/env bash
echo "Removing LocalOCR from $INSTALL_DIR ..."
rm -rf "$INSTALL_DIR"
rm -f "$HOME/Desktop/localocr.desktop" 2>/dev/null
rm -f "$HOME/.local/share/applications/localocr.desktop" 2>/dev/null
rm -f "/usr/local/bin/localocr" 2>/dev/null
echo "LocalOCR has been removed."
UNINSTALL
chmod +x "$INSTALL_DIR/uninstall.sh"

# ── Launch ──────────────────────────────────────────────────────────────
echo ""
echo "[6/6] Launching LocalOCR..."
echo ""
echo " ╔══════════════════════════════════════════════════════╗"
echo " ║  Installation complete!                              ║"
echo " ║  App installed to: $INSTALL_DIR"
echo " ║  Run: $INSTALL_DIR/localocr.sh                      "
echo " ║  Uninstall: $INSTALL_DIR/uninstall.sh               "
echo " ╚══════════════════════════════════════════════════════╝"
echo ""

cd "$INSTALL_DIR"
"$PYTHON" gui_app.py &
echo "  App is starting... You can close this terminal."
