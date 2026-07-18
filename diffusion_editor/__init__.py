"""Diffusion Editor package bootstrap."""

from __future__ import annotations

import os as _os
from pathlib import Path as _Path


def _restore_saved_termin_sdk() -> None:
    """Expose the installer-selected SDK before submodules import tcbase."""

    if _os.environ.get("TERMIN_SDK"):
        return
    state_file = _Path(__file__).resolve().parents[1] / ".termin-sdk"
    if not state_file.is_file():
        return
    saved = state_file.read_text(encoding="utf-8").strip()
    if not saved or "\n" in saved or "\r" in saved:
        return
    root = _Path(saved).expanduser()
    if not root.is_absolute():
        root = state_file.parent / root
    if (root / "lib").is_dir():
        _os.environ["TERMIN_SDK"] = str(root.resolve())


_restore_saved_termin_sdk()

del _os, _Path
