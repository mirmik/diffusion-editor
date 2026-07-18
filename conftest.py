"""Restore the installer-selected Termin SDK before pytest imports tests."""

from __future__ import annotations

import os

from diffusion_editor.sdk_runtime import resolve_sdk


os.environ.setdefault("TERMIN_SDK", str(resolve_sdk()))
