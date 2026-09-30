#!/usr/bin/env python3
"""Native SVG/Undo/composite smoke; --gpu checks a real Vulkan compositor."""
from __future__ import annotations

import argparse
from functools import partial
import os
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image
from termin.gui_native import OffscreenGuiComposition, DynamicTextureLease
from termin.graphics import configure_default_shader_runtime

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.native_root import NativeEditorRoot, bundled_native_font_path
from diffusion_editor.document.svg_layer import svg_size
from diffusion_editor.sdk_runtime import resolve_sdk


class Settings:
    def __init__(self, directory):
        self.directory = directory

    def get(self, key, default=None):
        if key == "recovery_dir":
            return str(self.directory / "recovery")
        if key == "mcp_server_enabled":
            return False
        return default

    def set(self, key, value):
        pass


class IdleEngine:
    model_info = {}

    def poll_event(self):
        return None

    def shutdown(self):
        pass


DEMO = '''<svg xmlns="http://www.w3.org/2000/svg" width="640" height="400" viewBox="0 0 640 400">
<rect width="640" height="400" fill="#d9e8ec"/>
<path id="building" d="M80 70 H340 L430 150 V290 H80 Z" fill="#9ac0d1" stroke="#263e51" stroke-width="4"/>
<path id="route" d="M30 340 L160 340 L160 210 L350 210" fill="none" stroke="#dc8034" stroke-width="5" stroke-dasharray="10 6"/>
<text x="80" y="45" font-size="24">SVG / Редактируемый план</text>
</svg>'''
PROBE = '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32"><rect id="probe" width="32" height="32" fill="red"/></svg>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svg", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix="diffusion-svg-smoke-"))
    output.mkdir(parents=True, exist_ok=True)
    source = args.svg.read_text(encoding="utf-8-sig") if args.svg else DEMO
    width, height = svg_size(source, (640, 400))
    sdk = resolve_sdk()
    options = {}
    if args.gpu:
        # App shaders are currently not shipped in SDK artifacts (Kanboard #2694).
        # Use explicit composition options so this smoke owns its temporary cache.
        os.environ["TERMIN_SHADER_DEV_COMPILE"] = "1"
        os.environ["TERMIN_SDK_SHADER_CACHE_ROOT"] = str(output / "shaders")
        if not configure_default_shader_runtime("svg-smoke"):
            raise RuntimeError("GPU smoke requires the SDK shader compiler and Slang")
        options = dict(
            application_graphics_domain=True,
            shader_compiler_path=os.environ["TERMIN_SHADERC"],
            slang_compiler_path=os.environ["TERMIN_SLANGC"],
            shader_cache_root=str(output / "shaders/cache"),
            shader_artifact_root=str(output / "shaders/artifacts"),
            enable_shader_dev_compile=True,
        )
    engine = IdleEngine()
    app = EditorApplication(settings=Settings(output), engines=EngineSet(*([engine] * 5)))
    composition = OffscreenGuiComposition(
        width=1280, height=900, backend="vulkan", sdk_root=str(sdk),
        font_path=str(bundled_native_font_path(sdk)),
        continuous_rendering=False, **options)
    with NativeEditorRoot(app, composition,
            texture_lease_factory=partial(DynamicTextureLease, composition)) as root:
        bridge = root.canvas.controller.composite_bridge
        if args.gpu and not bridge.using_gpu:
            raise RuntimeError("GPU smoke unexpectedly fell back to CPU composition")
        app.layer_stack.init_from_image(np.full((height, width, 4), 255, dtype=np.uint8))
        app.document.svg.add(source, args.svg.stem if args.svg else "SVG demo")
        root.tick()

        def check_parity():
            actual = root.canvas.controller.get_composite()
            expected = app.layer_stack.composite()
            difference = np.abs(actual.astype(np.int16) - expected.astype(np.int16))
            if difference.max() > 1:
                raise AssertionError(f"CPU/native composite mismatch: {difference.max()}")

        check_parity()
        identity = app.document.svg.add(PROBE, "Smoke probe")
        app.document.svg.update_element("probe", {"fill": "blue"}, layer_id=identity)
        root.tick()
        check_parity()
        assert tuple(root.canvas.controller.get_composite()[5, 5]) == (0, 0, 255, 255)
        app.document.undo()
        root.tick()
        assert tuple(root.canvas.controller.get_composite()[5, 5]) == (255, 0, 0, 255)
        app.document.undo()
        root.tick()
        check_parity()
        root.canvas.fit_in_view()
        root.tick()
        app.layer_stack.save_project(str(output / "svg-demo.deproj"))
        pixels = composition.read_frame_rgba_float()
        Image.fromarray(np.rint(np.clip(pixels, 0, 1) * 255).astype(np.uint8)).save(output / "editor.png")
        print(f"SVG smoke passed ({'GPU' if args.gpu else 'CPU'}): {output}")


if __name__ == "__main__":
    main()
