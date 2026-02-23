"""Native file dialogs via zenity (Linux)."""

import subprocess
import shutil


def _has_zenity() -> bool:
    return shutil.which("zenity") is not None


def _has_kdialog() -> bool:
    return shutil.which("kdialog") is not None


def open_file_dialog(title: str = "Open", directory: str = "",
                     filter_str: str = "") -> str | None:
    """Show a native open-file dialog. Returns path or None."""
    if _has_zenity():
        cmd = ["zenity", "--file-selection", f"--title={title}"]
        if directory:
            cmd.append(f"--filename={directory}/")
        if filter_str:
            for flt in filter_str.split(";;"):
                cmd.extend(["--file-filter", flt])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    elif _has_kdialog():
        cmd = ["kdialog", "--getopenfilename", directory or ".", filter_str or "*"]
        cmd.extend(["--title", title])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return None


def save_file_dialog(title: str = "Save", directory: str = "",
                     filter_str: str = "") -> str | None:
    """Show a native save-file dialog. Returns path or None."""
    if _has_zenity():
        cmd = ["zenity", "--file-selection", "--save", "--confirm-overwrite",
               f"--title={title}"]
        if directory:
            cmd.append(f"--filename={directory}/")
        if filter_str:
            for flt in filter_str.split(";;"):
                cmd.extend(["--file-filter", flt])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    elif _has_kdialog():
        cmd = ["kdialog", "--getsavefilename", directory or ".", filter_str or "*"]
        cmd.extend(["--title", title])
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
    return None
