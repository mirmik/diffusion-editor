#!/usr/bin/env python3
"""Exercise tcgui and diffusion-editor GPU composition on the Termin SDK."""

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


def _render_frames(frame_count: int, project_path: Path | None = None) -> int:
    import numpy as np

    from tcbase import log
    from tcgui.widgets.label import Label
    from tcgui.widgets.panel import Panel
    from tcgui.widgets.ui import UI
    from tcgui.widgets.units import pct
    from termin.display import SDLBackendWindow
    from tgfx import Tgfx2Context, configure_default_shader_runtime

    from diffusion_editor.canvas.gpu_compositor import GPUCompositor
    from diffusion_editor.document.layer_stack import LayerStack

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
    compositor = None

    try:
        # This is the path that caught the API drift missed by the original
        # one-frame UI-only smoke: TcShader resource layout, texture upload,
        # symbolic binding, offscreen passes, and immediate triangles.
        stack = LayerStack(tile_size=16)
        if project_path is None:
            pixels = np.zeros((32, 32, 4), dtype=np.uint8)
            pixels[..., 0] = 180
            pixels[..., 3] = 255
            stack.init_from_image(pixels)
        else:
            stack.load_project(str(project_path))
        compositor = GPUCompositor(stack, graphics=graphics)
        compositor.composite()
        if graphics.context.in_frame:
            raise RuntimeError("GPUCompositor stranded an open RenderContext2 frame")

        width, height = window.framebuffer_size()
        texture = None
        for _ in range(frame_count):
            texture = ui.render_compose(
                width, height, background_color=(0.12, 0.12, 0.14, 1.0)
            )
            if texture is None:
                raise RuntimeError("tcgui produced no texture for a non-empty UI")
            if graphics.context.in_frame:
                raise RuntimeError("tcgui stranded an open RenderContext2 frame")
        window.present(texture)
        project_note = f", project={project_path}" if project_path else ""
        print(
            f"Termin render smoke OK: {width}x{height}, frames={frame_count}"
            f"{project_note}"
        )
        texture = None
        return 0
    finally:
        if compositor is not None:
            compositor.dispose()
            compositor = None
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


def _run_checked_child(frame_count: int, project_path: Path | None) -> int:
    sdk = os.environ.get("TERMIN_SDK")
    if not sdk:
        from diffusion_editor.sdk_runtime import resolve_sdk

        sdk = str(resolve_sdk())
        os.environ["TERMIN_SDK"] = sdk
    if not (Path(sdk) / "python-runtime-manifest.json").is_file():
        raise RuntimeError(f"TERMIN_SDK is incomplete: {sdk}")

    child_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--frames",
        str(frame_count),
    ]
    if project_path is not None:
        child_command.extend(("--project", str(project_path)))
    result = subprocess.run(
        child_command,
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--project", type=Path)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be positive")
    if args.project is not None and not args.project.is_file():
        parser.error(f"project does not exist: {args.project}")
    if args.child:
        return _render_frames(args.frames, args.project)
    return _run_checked_child(args.frames, args.project)


if __name__ == "__main__":
    raise SystemExit(main())
