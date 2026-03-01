"""
Create a Windows Desktop Shortcut for LocalOCR.

Usage:
    py -3 create_shortcut.py            # Create shortcut for gui_app.py (dev mode)
    py -3 create_shortcut.py --exe      # Create shortcut for built exe
"""
import os
import sys
import argparse
from pathlib import Path


def create_shortcut(target: str, shortcut_name: str = "LocalOCR",
                    working_dir: str = "", icon_path: str = "",
                    arguments: str = ""):
    """Create a .lnk shortcut on the Windows Desktop using COM."""
    try:
        # Use PowerShell to create the shortcut (works without extra packages)
        desktop = Path(os.path.expanduser("~")) / "Desktop"
        if not desktop.exists():
            # Fallback via shell:desktop
            desktop = Path(os.environ.get("USERPROFILE", "~")) / "Desktop"
        shortcut_path = desktop / f"{shortcut_name}.lnk"

        ps_script = f'''
$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut("{shortcut_path}")
$Shortcut.TargetPath = "{target}"
$Shortcut.Arguments = "{arguments}"
$Shortcut.WorkingDirectory = "{working_dir}"
'''
        if icon_path:
            ps_script += f'$Shortcut.IconLocation = "{icon_path}"\n'

        ps_script += '''$Shortcut.Description = "LocalOCR — PDF to DOCX Converter"
$Shortcut.Save()
'''
        # Execute via powershell
        import subprocess
        result = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass",
             "-Command", ps_script],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            print(f"[OK] Desktop shortcut created: {shortcut_path}")
            return True
        else:
            print(f"[ERROR] PowerShell error: {result.stderr.strip()}")
            return False

    except Exception as exc:
        print(f"[ERROR] Failed to create shortcut: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Create LocalOCR desktop shortcut")
    parser.add_argument("--exe", action="store_true",
                        help="Create shortcut for the built .exe "
                             "(default: shortcut for gui_app.py)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent

    if args.exe:
        # Point to the built exe
        exe_path = project_root / "dist" / "LocalOCR" / "LocalOCR.exe"
        if not exe_path.exists():
            print(f"[ERROR] EXE not found at: {exe_path}")
            print("  Run 'py -3 build_exe.py' first to build the exe.")
            sys.exit(1)
        create_shortcut(
            target=str(exe_path),
            shortcut_name="LocalOCR",
            working_dir=str(exe_path.parent),
        )
    else:
        # Point to python running gui_app.py (dev mode)
        python_exe = sys.executable
        gui_script = project_root / "gui_app.py"
        if not gui_script.exists():
            print(f"[ERROR] gui_app.py not found at: {gui_script}")
            sys.exit(1)

        # Use pythonw.exe (no console) if available
        pythonw = Path(python_exe).parent / "pythonw.exe"
        if pythonw.exists():
            python_exe = str(pythonw)

        create_shortcut(
            target=python_exe,
            shortcut_name="LocalOCR",
            working_dir=str(project_root),
            arguments=str(gui_script),
        )


if __name__ == "__main__":
    main()
