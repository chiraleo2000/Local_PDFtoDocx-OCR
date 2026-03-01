#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════════
#  LocalOCR — Installer for Linux & macOS  (v0.3.1)
#
#  Usage:
#    bash install.sh                # Interactive install
#    bash install.sh --no-venv      # Use system Python (not recommended)
#    bash install.sh --gpu          # Install GPU-accelerated PyTorch
#    bash install.sh --update       # Re-run over an existing install
#    bash install.sh --uninstall    # Remove LocalOCR
# ══════════════════════════════════════════════════════════════════════════════
set -euo pipefail

# ── Colour helpers ────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
ok()      { echo -e "${GREEN}[ OK ]${RESET}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
err()     { echo -e "${RED}[ERR ]${RESET}  $*" >&2; }
die()     { err "$*"; exit 1; }
header()  { echo -e "\n${BOLD}${CYAN}══  $*  ${RESET}"; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}╔══════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${CYAN}║   LocalOCR — PDF to DOCX Converter                  ║${RESET}"
echo -e "${BOLD}${CYAN}║   Installer v0.3.1  |  Linux & macOS                ║${RESET}"
echo -e "${BOLD}${CYAN}╚══════════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Parse flags ──────────────────────────────────────────────────────────────
USE_VENV=true
USE_GPU=false
MODE="install"   # install | update | uninstall

for arg in "$@"; do
    case "$arg" in
        --no-venv)    USE_VENV=false ;;
        --gpu)        USE_GPU=true ;;
        --update)     MODE=update ;;
        --uninstall)  MODE=uninstall ;;
        -h|--help)
            echo "Usage: bash install.sh [--no-venv] [--gpu] [--update] [--uninstall]"
            exit 0 ;;
        *) warn "Unknown flag: $arg" ;;
    esac
done

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
OS="$(uname -s)"   # Linux | Darwin

# ══════════════════════════════════════════════════════════════════════════════
# UNINSTALL
# ══════════════════════════════════════════════════════════════════════════════
if [ "$MODE" = "uninstall" ]; then
    header "Uninstalling LocalOCR"
    DEFAULT_INSTALL="$HOME/LocalOCR"
    read -r -p "  Remove directory [$DEFAULT_INSTALL]: " INSTALL_DIR
    INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_INSTALL}"
    [ ! -d "$INSTALL_DIR" ] && die "Directory not found: $INSTALL_DIR"

    rm -rf "$INSTALL_DIR"
    rm -f "$HOME/.local/share/applications/localocr.desktop" 2>/dev/null || true
    rm -f "$HOME/.local/bin/localocr"                        2>/dev/null || true
    rm -f "/usr/local/bin/localocr"                          2>/dev/null || true
    rm -rf "$HOME/Applications/LocalOCR.app"                 2>/dev/null || true
    # XDG Desktop shortcut
    XDG_DESK=""
    command -v xdg-user-dir &>/dev/null && \
        XDG_DESK="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
    [ -z "$XDG_DESK" ] && [ -d "$HOME/Desktop" ] && XDG_DESK="$HOME/Desktop"
    [ -n "$XDG_DESK" ] && rm -f "$XDG_DESK/localocr.desktop" 2>/dev/null || true

    ok "LocalOCR removed from $INSTALL_DIR"
    exit 0
fi

# ══════════════════════════════════════════════════════════════════════════════
# 1/6  PYTHON
# ══════════════════════════════════════════════════════════════════════════════
header "1/6  Checking Python"

PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        ver=$("$cmd" -c \
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" \
            2>/dev/null || true)
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "${major:-0}" -ge 3 ] && [ "${minor:-0}" -ge 10 ] 2>/dev/null; then
            PYTHON="$cmd"
            ok "Using $cmd ($ver)"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    err "Python 3.10+ not found."
    if [ "$OS" = "Darwin" ]; then
        if command -v brew &>/dev/null; then
            info "Installing Python via Homebrew..."
            brew install python@3.12
            PYTHON="python3.12"
        else
            die "Install Homebrew first: https://brew.sh  then: brew install python@3.12"
        fi
    elif command -v apt-get &>/dev/null; then
        info "Installing Python via apt..."
        sudo apt-get update -qq
        sudo apt-get install -y python3 python3-pip python3-venv python3-tk
        PYTHON="python3"
    elif command -v dnf &>/dev/null; then
        info "Installing Python via dnf..."
        sudo dnf install -y python3 python3-pip python3-tkinter
        PYTHON="python3"
    elif command -v pacman &>/dev/null; then
        info "Installing Python via pacman..."
        sudo pacman -S --noconfirm python python-pip tk
        PYTHON="python3"
    elif command -v zypper &>/dev/null; then
        info "Installing Python via zypper (openSUSE)..."
        sudo zypper install -y python312 python312-pip python312-tk
        PYTHON="python3"
    else
        die "Please install Python 3.10+ manually: https://www.python.org/downloads"
    fi
fi

# ── Tkinter availability ──────────────────────────────────────────────────────
if ! "$PYTHON" -c "import tkinter" &>/dev/null 2>&1; then
    warn "tkinter not found — attempting to install..."
    if [ "$OS" = "Darwin" ]; then
        if command -v brew &>/dev/null; then
            PYVER=$("$PYTHON" -c \
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
            brew install "python-tk@${PYVER}" 2>/dev/null \
                || brew install python-tk \
                || die "Install tkinter manually: brew install python-tk"
        else
            die "Install: brew install python-tk  (or install Homebrew first)"
        fi
    elif command -v apt-get &>/dev/null; then
        sudo apt-get install -y python3-tk
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y python3-tkinter
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm tk
    elif command -v zypper &>/dev/null; then
        sudo zypper install -y python3-tk
    else
        die "Please install python3-tk for your distribution then re-run."
    fi
fi
ok "tkinter available"

# ══════════════════════════════════════════════════════════════════════════════
# 2/6  INSTALLATION DIRECTORY
# ══════════════════════════════════════════════════════════════════════════════
header "2/6  Installation Directory"

DEFAULT_INSTALL="$HOME/LocalOCR"
if [ "$MODE" = "update" ]; then
    INSTALL_DIR="${LOCALOCR_DIR:-$DEFAULT_INSTALL}"
    info "Update mode — target: $INSTALL_DIR"
else
    read -r -p "  Install to [$DEFAULT_INSTALL]: " INSTALL_DIR
    INSTALL_DIR="${INSTALL_DIR:-$DEFAULT_INSTALL}"
fi
info "Target: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"

# ══════════════════════════════════════════════════════════════════════════════
# 3/6  COPY FILES
# ══════════════════════════════════════════════════════════════════════════════
header "3/6  Copying Application Files"

# Prefer rsync (preserves metadata + handles deletions cleanly)
if command -v rsync &>/dev/null; then
    rsync -a --delete "$SRC_DIR/src/" "$INSTALL_DIR/src/"
else
    mkdir -p "$INSTALL_DIR/src"
    cp -r "$SRC_DIR/src/." "$INSTALL_DIR/src/"
fi

for item in gui_app.py app.py requirements.txt; do
    [ -f "$SRC_DIR/$item" ] && cp -f "$SRC_DIR/$item" "$INSTALL_DIR/"
done
for dir in models correction_data tests; do
    [ -d "$SRC_DIR/$dir" ] && cp -r "$SRC_DIR/$dir" "$INSTALL_DIR/"
done
for dotfile in .env .env.example; do
    [ -f "$SRC_DIR/$dotfile" ] && cp -f "$SRC_DIR/$dotfile" "$INSTALL_DIR/"
done

date "+installed %Y-%m-%d %H:%M:%S" > "$INSTALL_DIR/.installed_ok"
ok "Files copied."

# ══════════════════════════════════════════════════════════════════════════════
# 4/6  VIRTUAL ENVIRONMENT + DEPENDENCIES
# ══════════════════════════════════════════════════════════════════════════════
header "4/6  Installing Dependencies"

if $USE_VENV; then
    VENV_DIR="$INSTALL_DIR/.venv"
    if [ ! -d "$VENV_DIR" ]; then
        info "Creating virtual environment..."
        "$PYTHON" -m venv "$VENV_DIR" \
            || die "venv creation failed. Install python3-venv for your distro."
    else
        info "Reusing existing venv at $VENV_DIR"
    fi
    APP_PYTHON="$VENV_DIR/bin/python"
    APP_PIP="$VENV_DIR/bin/pip"
else
    APP_PYTHON="$PYTHON"
    APP_PIP="$PYTHON -m pip"
fi

$APP_PIP install --upgrade pip --quiet

# ── PyTorch (CPU or GPU) ──────────────────────────────────────────────────────
if $APP_PYTHON -c "import torch" &>/dev/null 2>&1; then
    ok "PyTorch already installed — skipping."
else
    if $USE_GPU; then
        info "Installing PyTorch with CUDA (GPU) support..."
        $APP_PIP install torch torchvision \
            --index-url https://download.pytorch.org/whl/cu121 --quiet \
            || warn "GPU PyTorch failed — trying CPU fallback..."
        # Fallback to CPU if GPU wheels failed
        if ! $APP_PYTHON -c "import torch" &>/dev/null 2>&1; then
            info "Falling back to CPU PyTorch..."
            $APP_PIP install torch torchvision \
                --index-url https://download.pytorch.org/whl/cpu --quiet
        fi
    else
        info "Installing PyTorch (CPU)..."
        $APP_PIP install torch torchvision \
            --index-url https://download.pytorch.org/whl/cpu --quiet
    fi
fi

# ── App requirements ──────────────────────────────────────────────────────────
info "Installing application requirements..."
$APP_PIP install -r "$INSTALL_DIR/requirements.txt" --quiet 2>&1 | grep -v "^$" | tail -5 || {
    warn "Some dependencies may have failed — the app will start with reduced functionality."
}
ok "Dependencies installed."

# ══════════════════════════════════════════════════════════════════════════════
# 5/6  LAUNCHERS & DESKTOP INTEGRATION
# ══════════════════════════════════════════════════════════════════════════════
header "5/6  Creating Launchers"

# ── Shell launcher script ─────────────────────────────────────────────────────
LAUNCHER="$INSTALL_DIR/localocr.sh"
cat > "$LAUNCHER" << LAUNCHER_SCRIPT
#!/usr/bin/env bash
# LocalOCR launcher — generated by installer v0.3.1
cd "$INSTALL_DIR"
exec "$APP_PYTHON" gui_app.py "\$@"
LAUNCHER_SCRIPT
chmod +x "$LAUNCHER"
ok "Launcher: $LAUNCHER"

# ── Linux desktop integration ─────────────────────────────────────────────────
if [ "$OS" = "Linux" ]; then
    APPS_DIR="$HOME/.local/share/applications"
    mkdir -p "$APPS_DIR"

    # Icon: use bundled file if present, else a system theme icon name
    ICON_PATH="utilities-text-editor"
    for candidate in "$INSTALL_DIR/icon.png" "$INSTALL_DIR/src/icon.png"; do
        [ -f "$candidate" ] && { ICON_PATH="$candidate"; break; }
    done

    cat > "$APPS_DIR/localocr.desktop" << DESKTOP_FILE
[Desktop Entry]
Type=Application
Version=0.3.1
Name=LocalOCR
GenericName=PDF OCR Converter
Comment=Convert scanned PDFs to editable DOCX with AI-powered OCR
Exec=$LAUNCHER %f
Path=$INSTALL_DIR
Icon=$ICON_PATH
Terminal=false
MimeType=application/pdf;
Categories=Office;Utility;Graphics;
Keywords=OCR;PDF;Thai;Scan;DOCX;
StartupNotify=true
DESKTOP_FILE
    chmod +x "$APPS_DIR/localocr.desktop"
    update-desktop-database "$APPS_DIR" 2>/dev/null || true
    ok "App menu entry: $APPS_DIR/localocr.desktop"

    # Desktop shortcut via XDG
    XDG_DESK=""
    command -v xdg-user-dir &>/dev/null && \
        XDG_DESK="$(xdg-user-dir DESKTOP 2>/dev/null || true)"
    [ -z "$XDG_DESK" ] && [ -d "$HOME/Desktop" ] && XDG_DESK="$HOME/Desktop"
    if [ -n "$XDG_DESK" ] && [ -d "$XDG_DESK" ]; then
        cp -f "$APPS_DIR/localocr.desktop" "$XDG_DESK/localocr.desktop"
        chmod +x "$XDG_DESK/localocr.desktop"
        ok "Desktop shortcut: $XDG_DESK/localocr.desktop"
    fi

    # ~/.local/bin symlink for terminal use
    LOCAL_BIN="$HOME/.local/bin"
    mkdir -p "$LOCAL_BIN"
    ln -sf "$LAUNCHER" "$LOCAL_BIN/localocr"
    ok "Terminal command: localocr  (ensure ~/.local/bin is in your PATH)"
fi

# ── macOS .app bundle ─────────────────────────────────────────────────────────
if [ "$OS" = "Darwin" ]; then
    APPS_MACOS="$HOME/Applications"
    APP_BUNDLE="$APPS_MACOS/LocalOCR.app"
    CONTENTS="$APP_BUNDLE/Contents"
    mkdir -p "$CONTENTS/MacOS" "$CONTENTS/Resources"

    # Info.plist
    cat > "$CONTENTS/Info.plist" << PLIST_EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key>                <string>LocalOCR</string>
    <key>CFBundleDisplayName</key>         <string>LocalOCR</string>
    <key>CFBundleIdentifier</key>          <string>com.localocr.app</string>
    <key>CFBundleVersion</key>             <string>0.3.1</string>
    <key>CFBundleShortVersionString</key>  <string>0.3.1</string>
    <key>CFBundlePackageType</key>         <string>APPL</string>
    <key>CFBundleExecutable</key>          <string>LocalOCR</string>
    <key>NSHighResolutionCapable</key>     <true/>
    <key>NSHumanReadableCopyright</key>    <string>LocalOCR © 2026</string>
    <key>LSMinimumSystemVersion</key>      <string>11.0</string>
    <key>CFBundleDocumentTypes</key>
    <array>
        <dict>
            <key>CFBundleTypeName</key>        <string>PDF Document</string>
            <key>CFBundleTypeExtensions</key>
            <array><string>pdf</string></array>
            <key>CFBundleTypeRole</key>        <string>Viewer</string>
        </dict>
    </array>
</dict>
</plist>
PLIST_EOF

    # Executable wrapper inside the .app
    cat > "$CONTENTS/MacOS/LocalOCR" << MACOS_EXE
#!/usr/bin/env bash
cd "$INSTALL_DIR"
exec "$APP_PYTHON" gui_app.py "\$@"
MACOS_EXE
    chmod +x "$CONTENTS/MacOS/LocalOCR"

    # Copy icon if one exists
    for candidate in "$INSTALL_DIR/icon.icns" "$SRC_DIR/icon.icns"; do
        [ -f "$candidate" ] && cp "$candidate" "$CONTENTS/Resources/AppIcon.icns" && break
    done

    # Remove Gatekeeper quarantine flag
    xattr -dr com.apple.quarantine "$APP_BUNDLE" 2>/dev/null || true

    ok ".app bundle: $APP_BUNDLE"

    # Symlink for terminal use
    if mkdir -p /usr/local/bin 2>/dev/null || [ -d /usr/local/bin ]; then
        ln -sf "$LAUNCHER" "/usr/local/bin/localocr" 2>/dev/null \
            && ok "Terminal command: localocr" \
            || warn "Could not write /usr/local/bin/localocr — run $LAUNCHER directly"
    fi

    # Open ~/Applications in Finder so user sees the .app
    open "$APPS_MACOS" 2>/dev/null || true
fi

# ── Uninstaller ───────────────────────────────────────────────────────────────
UNINSTALL_SCRIPT="$INSTALL_DIR/uninstall.sh"
cat > "$UNINSTALL_SCRIPT" << UNINSTALL_EOF
#!/usr/bin/env bash
echo "Removing LocalOCR from $INSTALL_DIR ..."
rm -rf  "$INSTALL_DIR"
rm -f   "\$HOME/.local/share/applications/localocr.desktop"
rm -f   "\$HOME/.local/bin/localocr"
rm -f   "/usr/local/bin/localocr"
rm -rf  "\$HOME/Applications/LocalOCR.app"
XDG_DESK=""
command -v xdg-user-dir &>/dev/null && XDG_DESK="\$(xdg-user-dir DESKTOP 2>/dev/null || true)"
[ -z "\$XDG_DESK" ] && [ -d "\$HOME/Desktop" ] && XDG_DESK="\$HOME/Desktop"
[ -n "\$XDG_DESK" ] && rm -f "\$XDG_DESK/localocr.desktop"
echo "Done. LocalOCR has been removed."
UNINSTALL_EOF
chmod +x "$UNINSTALL_SCRIPT"

# ══════════════════════════════════════════════════════════════════════════════
# 6/6  SUMMARY & LAUNCH
# ══════════════════════════════════════════════════════════════════════════════
header "6/6  Complete"
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}${GREEN}║  LocalOCR v0.3.1 installed successfully!                 ║${RESET}"
echo -e "${BOLD}${GREEN}║                                                          ║${RESET}"
printf  "${BOLD}${GREEN}║  Location : %-43s║${RESET}\n" "$INSTALL_DIR"
printf  "${BOLD}${GREEN}║  Run      : %-43s║${RESET}\n" "bash $LAUNCHER"
if [ "$OS" = "Linux" ]; then
printf  "${BOLD}${GREEN}║  Shortcut : localocr%-37s║${RESET}\n" ""
elif [ "$OS" = "Darwin" ]; then
printf  "${BOLD}${GREEN}║  App      : ~/Applications/LocalOCR.app%-18s║${RESET}\n" ""
printf  "${BOLD}${GREEN}║  Terminal : localocr%-37s║${RESET}\n" ""
fi
printf  "${BOLD}${GREEN}║  Uninstall: bash %-41s║${RESET}\n" "$INSTALL_DIR/uninstall.sh"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════════╝${RESET}"
echo ""

info "Starting LocalOCR..."
cd "$INSTALL_DIR"
nohup "$APP_PYTHON" gui_app.py > /tmp/localocr.log 2>&1 &
echo -e "  PID $! — logs at /tmp/localocr.log"
echo ""
