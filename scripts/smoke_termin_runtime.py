#!/usr/bin/env python3
"""Render one tcgui frame and reject shader diagnostics from the child host."""

from __future__ import annotations

import gc
import os
from pathlib import Path
import subprocess
import sys


_SHADER_FAILURE_MARKERS = (
    "shader is unavailable",
    "missing fragment_source",
    "missing vertex_source",
    "missing shader artifact",
    "failed to create solid shader",
    "failed to create texture shader",
    "skipping batch",
    "skipping text draw",
)


def _render_one_frame() -> int:
    from tcbase import log
    from tcgui.widgets.label import Label
    from tcgui.widgets.panel import Panel
    from tcgui.widgets.ui import UI
    from tcgui.widgets.units import pct
    from termin.display import SDLBackendWindow
    from tgfx import Tgfx2Context, configure_default_shader_runtime

    log.set_level(log.Level.INFO)
    if not configure_default_shader_runtime("diffusion-editor-smoke"):
        raise RuntimeError(
            "Termin shader runtime is unavailable; check TERMIN_SDK/bin/termin_shaderc and slangc"
        )

    window = SDLBackendWindow("Diffusion Editor runtime smoke", 320, 200)
    graphics = Tgfx2Context.from_window(window.device_ptr(), window.context_ptr())
    root = Panel()
    root.preferred_width = pct(100)
    root.preferred_height = pct(100)
    label = Label()
    label.text = "Diffusion Editor SDK smoke"
    root.add_child(label)
    ui = UI(graphics=graphics)
    ui.root = root

    try:
        width, height = window.framebuffer_size()
        texture = ui.render_compose(
            width, height, background_color=(0.12, 0.12, 0.14, 1.0)
        )
        if texture is None:
            raise RuntimeError("tcgui produced no texture for a non-empty UI")
        window.present(texture)
        print(f"Termin OpenGL render smoke OK: {width}x{height}")
        texture = None
        return 0
    finally:
        ui.root = None
        ui._renderer.close()
        ui = None
        label = None
        root = None
        graphics = None
        from tcbase.profiler import Profiler
        from tgfx import font as tgfx_font

        Profiler._instance = None
        tgfx_font._default_font_atlas = None
        gc.collect()
        window.close()


def _run_checked_child() -> int:
    sdk = os.environ.get("TERMIN_SDK")
    if not sdk:
        from diffusion_editor.sdk_runtime import resolve_sdk

        sdk = str(resolve_sdk())
        os.environ["TERMIN_SDK"] = sdk
    if not (Path(sdk) / "python-runtime-manifest.json").is_file():
        raise RuntimeError(f"TERMIN_SDK is incomplete: {sdk}")

    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--child"],
        env=os.environ.copy(),
        capture_output=True,
        text=True,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        return result.returncode

    combined = f"{result.stdout}\n{result.stderr}".lower()
    failures = [marker for marker in _SHADER_FAILURE_MARKERS if marker in combined]
    if failures:
        print(
            "ERROR: shader failure diagnostics detected: " + ", ".join(failures),
            file=sys.stderr,
        )
        return 1
    return 0


def main() -> int:
    if sys.argv[1:] == ["--child"]:
        return _render_one_frame()
    if sys.argv[1:]:
        raise SystemExit(f"usage: {Path(sys.argv[0]).name}")
    return _run_checked_child()


if __name__ == "__main__":
    raise SystemExit(main())
