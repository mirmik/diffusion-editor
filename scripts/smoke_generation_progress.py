#!/usr/bin/env python3
"""Render generation progress through the actual worker/engine/native UI path.

Defaults to the isolated fake worker. --real uses official Qwen Turbo on CUDA.
Neither mode opens or modifies the user's document.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image
from termin.gui_native import OffscreenGuiComposition, DynamicTextureLease
from termin.graphics import (configure_default_shader_runtime, get_shader_artifact_root,
                             get_shader_cache_root, get_shader_compiler_path,
                             get_shader_dev_compile_enabled)

from diffusion_editor.app.application import EditorApplication, EngineSet
from diffusion_editor.app.native_root import NativeEditorRoot, bundled_native_font_path
from diffusion_editor.app.generation_panels import GenerationAction, GenerationIntent, GenerationPhase
from diffusion_editor.document.tool import TextToImageTool
from diffusion_editor.engines.text_to_image_engine import TextToImageEngine
from diffusion_editor.workers.ml_process import MlProcessClient


class MemorySettings:
    def __init__(self, output):
        self.values = {"recovery_dir": str(output / "recovery"), "mcp_server_enabled": False}

    def get(self, key, default=None):
        return self.values.get(key, default)

    def set(self, key, value):
        self.values[key] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--cfg", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("/tmp/diffusion-editor-progress-smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not configure_default_shader_runtime("generation-progress-smoke"):
        raise RuntimeError("Shader runtime is unavailable")
    client = MlProcessClient(backend="real" if args.real else "fake",
                             **({} if args.real else {"python": sys.executable}))
    engines = replace(EngineSet.create_default(), text_to_image=TextToImageEngine(client))
    app = EditorApplication(settings=MemorySettings(args.output), engines=engines)
    app.layer_stack.init_from_image(np.zeros((384,384,4), dtype=np.uint8))
    layer = app.layer_stack.active_layer
    layer.tool = TextToImageTool("qwen-image-2.1-turbo")
    layer.tool.set_parameter("prompt", "A red ceramic teapot on a plain white background.")
    layer.tool.set_parameter("seed", 42)
    layer.tool.set_parameter("true_cfg_scale", args.cfg)
    composition = OffscreenGuiComposition(width=1440, height=1000,
        continuous_rendering=False, font_path=str(bundled_native_font_path()),
        application_graphics_domain=True, shader_artifact_root=get_shader_artifact_root(),
        shader_cache_root=get_shader_cache_root(), shader_compiler_path=get_shader_compiler_path(),
        slang_compiler_path=os.environ.get("TERMIN_SLANGC", ""),
        enable_shader_dev_compile=get_shader_dev_compile_enabled())
    events = []
    captured = set()
    with NativeEditorRoot(app, composition,
                          texture_lease_factory=lambda: DynamicTextureLease(composition)) as root:
        root.view.inspector_tabs.selected_index = 1
        root.generation_panels_coordinator.handle_intent(GenerationIntent(GenerationAction.RUN))
        deadline = time.monotonic() + 300
        previous = None
        while time.monotonic() < deadline:
            root.tick()
            state = root.generation_panels_coordinator.state.text_to_image
            progress = state.progress
            if progress != previous:
                previous = progress
                if progress is not None:
                    events.append(progress.to_dict())
                    print(progress.text, flush=True)
            if progress is not None and progress.total is not None:
                bar = root.generation_panels.text_to_image_progress
                assert bar.widget.visible and abs(bar.value-progress.fraction) < 1e-6
                if "sampling" not in captured and progress.current > 0:
                    composition.request_repaint()
                    composition.render_frame()
                    pixels = composition.read_frame_rgba_float()
                    # The framebuffer readback is linear; PNG is displayed in sRGB.
                    rgb = np.where(pixels[:,:,:3] <= .0031308, pixels[:,:,:3]*12.92,
                                   1.055*np.maximum(pixels[:,:,:3],0)**(1/2.4)-.055)
                    pixels[:,:,:3] = rgb
                    Image.fromarray(np.rint(np.clip(pixels,0,1)*255).astype(np.uint8)).save(
                        args.output / "sampling-ui.png")
                    captured.add("sampling")
            if state.phase == GenerationPhase.ERROR:
                raise RuntimeError(state.message)
            if state.phase == GenerationPhase.RESULT:
                assert not root.generation_panels.text_to_image_progress.widget.visible
                Image.fromarray(layer.image).save(args.output / "result.png")
                break
            time.sleep(.01)
        else:
            raise RuntimeError("Generation progress smoke timed out")
    assert events and "sampling" in captured
    (args.output / "progress.json").write_text(json.dumps(events, indent=2))
    print(f"Verified {'real GPU' if args.real else 'fake-worker'} progress/UI: {args.output}")


if __name__ == "__main__":
    main()
