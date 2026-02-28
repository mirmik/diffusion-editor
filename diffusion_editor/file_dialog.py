"""Native file dialogs via tkinter."""

import tkinter as tk
from tkinter import filedialog
from tcbase import log


def _ensure_root():
    """Get hidden tkinter root window."""
    try:
        root = tk._default_root
        if root is None:
            root = tk.Tk()
            root.withdraw()
        return root
    except Exception as e:
        log.warn(f"[FileDialog] tkinter root init fallback: {e}")
        root = tk.Tk()
        root.withdraw()
        return root


def _parse_filters(filter_str: str):
    """Parse 'Label | *.ext1 *.ext2;;Label2 | *.ext3' into tkinter filetypes."""
    if not filter_str:
        return [("All files", "*.*")]
    result = []
    for part in filter_str.split(";;"):
        if "|" in part:
            label, exts = part.split("|", 1)
            result.append((label.strip(), exts.strip()))
        else:
            result.append((part.strip(), "*.*"))
    return result


def open_file_dialog(title: str = "Open", directory: str = "",
                     filter_str: str = "") -> str | None:
    _ensure_root()
    path = filedialog.askopenfilename(
        title=title,
        initialdir=directory or None,
        filetypes=_parse_filters(filter_str),
    )
    return path if path else None


def save_file_dialog(title: str = "Save", directory: str = "",
                     filter_str: str = "") -> str | None:
    _ensure_root()
    path = filedialog.asksaveasfilename(
        title=title,
        initialdir=directory or None,
        filetypes=_parse_filters(filter_str),
    )
    return path if path else None


def open_directory_dialog(title: str = "Select Directory",
                          directory: str = "") -> str | None:
    _ensure_root()
    path = filedialog.askdirectory(
        title=title,
        initialdir=directory or None,
        mustexist=True,
    )
    return path if path else None
