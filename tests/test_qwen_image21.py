from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from diffusion_editor.document.layer import Layer
from diffusion_editor.document.tool import TextToImageTool
from diffusion_editor.generation.image_edit_profiles import image_edit_profile
from diffusion_editor.generation.text_to_image_profiles import text_to_image_profile
from diffusion_editor.generation.patch_resolver import resolve_source_patch
from diffusion_editor.generation.reference_resolver import layer_reference_image
from diffusion_editor.workers.ml_backend import RealMlBackend
from diffusion_editor.workers.qwen_image21_process import QwenImage21Pipeline

PROFILE = "qwen-image-2.1"


def test_new_profile_keeps_independent_settings_and_load_keys():
    profile = text_to_image_profile(PROFILE)
    tool = TextToImageTool()
    tool.set_profile(PROFILE)
    tool.set_parameter("prompt", "Transparent teapot")
    tool.set_parameter("seed", 123)
    tool.set_profile("qwen-image-2512")
    assert tool.parameters["prompt"] == ""
    tool.set_profile(PROFILE)
    assert tool.parameters["prompt"] == "Transparent teapot"
    assert tool.parameters["seed"] == 123
    assert profile.inference_size(33, 65) == (64, 96)
    assert profile.default_lora_adapters == ()
    assert profile.defaults()["true_cfg_scale"] == 1
    keys = profile.load_values({})
    assert {"runtime_python", "runtime_root", "model", "vae_checkpoint",
            "text_encoder_checkpoint", "memory_mode"} <= keys.keys()
    assert "prompt" not in keys and "output_mode" not in keys
    assert image_edit_profile(PROFILE).supports_alpha


def test_source_patch_and_reference_keep_alpha_only_when_requested():
    pixels = np.zeros((10, 12, 4), dtype=np.uint8)
    pixels[:, :, :3] = [20, 40, 60]
    pixels[2:8, 3:9, 3] = 128
    layer = Layer("source", 12, 10, pixels)
    layer.patch_rect = (1, 1, 10, 9)
    patch = resolve_source_patch(layer, pixels, preserve_alpha=True)
    assert patch.image.mode == "RGBA"
    assert patch.image.getextrema()[3] == (0, 128)
    assert resolve_source_patch(layer, pixels).image.mode == "RGB"
    reference = layer_reference_image(layer, preserve_alpha=True)
    assert reference.image.mode == "RGBA"
    assert reference.image.getextrema()[3] == (0, 128)
    assert layer_reference_image(layer).image.mode == "RGB"


def _fake_runtime(tmp_path, body):
    p = text_to_image_profile(PROFILE).defaults()
    p.update(runtime_python=sys.executable, runtime_root=str(tmp_path), device="cpu")
    for key in ("model", "vae_checkpoint", "text_encoder_checkpoint"):
        file = tmp_path / (key + ".safetensors")
        file.write_bytes(b"test")
        p[key] = str(file)
    runner = tmp_path / "runner.py"
    runner.write_text(body)
    return p, runner


def test_real_child_ipc_preserves_rgba_and_closes(tmp_path):
    p, runner = _fake_runtime(tmp_path, '''
import json, sys
from PIL import Image
for line in sys.stdin:
    r = json.loads(line)
    if r['operation'] == 'generate':
        source = Image.open(r['images'][0])
        reference = Image.open(r['images'][1])
        assert source.mode == reference.mode == 'RGBA'
        assert reference.getpixel((0, 0))[3] == 64
        assert r['seed'] == 123 and r['parameters']['prompt'] == 'edit'
        source.save(r['output'])
    print(json.dumps({'ok': True, 'result': {'provider': 'test'}}), flush=True)
''')
    pipe = QwenImage21Pipeline(p, runner_path=runner, timeout=10)
    process = pipe._process
    directory = pipe._directory
    p['prompt'] = 'edit'
    image = Image.new("RGBA", (32, 32), (40, 80, 120, 64))
    try:
        result = pipe.generate(p, 123, 32, 32, image=image, reference_image=image)
        assert result.mode == "RGBA" and result.getpixel((0, 0)) == (40, 80, 120, 64)
    finally:
        pipe.close()
    assert process.poll() is not None
    assert not directory.exists()
    pipe.close()


@pytest.mark.parametrize("body,message", [
    ("import json\nprint(json.dumps({'ok':False,'error':'bad checkpoint'}),flush=True)\n", "bad checkpoint"),
    ("pass\n", "exited unexpectedly"),
    ("import time\ntime.sleep(10)\n", "timed out"),
])
def test_child_errors_are_reported_and_reaped(tmp_path, monkeypatch, body, message):
    import diffusion_editor.workers.qwen_image21_process as module
    p, runner = _fake_runtime(tmp_path, body)
    original = module.subprocess.Popen
    children = []

    def spawn(*args, **kwargs):
        child = original(*args, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError, match=message):
        QwenImage21Pipeline(p, runner_path=runner, timeout=0.5)
    assert children and children[0].poll() is not None


def test_backend_routes_new_model_sizes_alpha_and_lifecycle(tmp_path, monkeypatch):
    import diffusion_editor.workers.qwen_image21_process as module
    p, _ = _fake_runtime(tmp_path, "")
    events = []

    class FakePipeline:
        runtime_info = {"provider": "test", "comfy_revision": "test-revision"}
        scheduler = None

        def __init__(self, parameters, adapters):
            self.parameters = parameters
            events.append("load")

        def generate(self, parameters, seed, width, height, **kwargs):
            events.append((width, height, seed))
            if kwargs:
                assert kwargs["image"].mode == "RGBA"
                assert kwargs["reference_image"].getpixel((0, 0))[3] == 64
            return Image.new("RGBA", (width, height), (40, 80, 120, 64))

        def close(self):
            events.append("close")

    monkeypatch.setattr(module, "QwenImage21Pipeline", FakePipeline)
    monkeypatch.setattr(RealMlBackend, "_release_accelerator_memory", staticmethod(lambda: None))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    backend = RealMlBackend()
    p.update(seed=123)
    data = dict(profile_id=PROFILE, parameters=p, lora_adapters=[])
    backend.load_text_to_image(data)
    result, seed, provenance = backend.text_to_image(dict(data, width=33, height=65))
    assert result.size == (33, 65) and result.mode == "RGBA" and seed == 123
    assert provenance["runtime"]["provider_runtime"]["comfy_revision"] == "test-revision"
    backend.load_image_edit(data)
    assert events[:4] == ["load", (64, 96, 123), "close", "load"]
    source = Image.new("RGBA", (35, 47), (40, 80, 120, 64))
    result, seed, provenance = backend.image_edit(data, source, source.copy())
    assert result.size == source.size and result.getextrema()[3] == (64, 64)
    backend.unload_image_edit()
    assert events[-1] == "close"
    assert backend._instruct_pipe is None and backend._text_to_image_pipe is None


def test_frozen_result_application_retains_transparency_after_profile_switch():
    from diffusion_editor.document.layer_stack import LayerStack
    from diffusion_editor.generation.job_context import (
        ApplyFrozenGeneratedResultCommand, FrozenPasteContext,
    )
    stack = LayerStack()
    layer = Layer("generated", 4, 4, np.zeros((4, 4, 4), dtype=np.uint8))
    layer.tool = TextToImageTool(model_profile_id=PROFILE)
    stack.insert_layer(layer)
    paste = FrozenPasteContext.capture_layer(layer, preserve_alpha=True)
    # Application uses the captured policy, not the currently selected model.
    layer.tool.set_profile("qwen-image-2512")
    result = Image.new("RGBA", (4, 4), (30, 60, 90, 64))
    ApplyFrozenGeneratedResultCommand(layer, result, paste, "Qwen result").apply(stack)
    assert np.all(layer.image[:, :, 3] == 64)
