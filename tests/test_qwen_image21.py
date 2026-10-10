from __future__ import annotations

import io
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
TURBO8 = "qwen-image-2.1-turbo8"
TURBO = "qwen-image-2.1-turbo"


@pytest.mark.parametrize("get_profile", [text_to_image_profile, image_edit_profile])
def test_turbo8_profile_has_fixed_sampling_and_enabled_adapter(get_profile):
    profile = get_profile(TURBO8)
    assert profile.supports_alpha
    p = profile.normalize({"steps": 40, "true_cfg_scale": 4})
    assert p["steps"] == 8 and p["true_cfg_scale"] == 1
    adapter, = profile.normalize_lora_adapters(None)
    assert adapter.stable_id == "turbo8" and adapter.enabled and adapter.weight == 1
    assert adapter.source.endswith(".safetensors")
    assert get_profile(PROFILE).defaults()["steps"] == 40


def test_turbo8_settings_and_adapters_survive_profile_switch():
    tool = TextToImageTool()
    tool.set_profile(TURBO8)
    tool.set_parameter("prompt", "transparent vase")
    assert tool.lora_adapters[0].stable_id == "turbo8"
    tool.set_profile(PROFILE)
    assert tool.parameters["steps"] == 40 and not tool.lora_adapters
    tool.set_profile(TURBO8)
    assert tool.parameters["steps"] == 8
    assert tool.parameters["prompt"] == "transparent vase"
    assert tool.lora_adapters[0].enabled


def test_turbo8_sampling_repatches_each_resolution_without_mutating_base(monkeypatch):
    from diffusion_editor.workers.qwen_image21_runner import Runtime

    calls = []

    class ModelSamplingFlux:
        def patch(self, model, **kwargs):
            calls.append((model, kwargs))
            return (object(),)

    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_model_advanced",
                        SimpleNamespace(ModelSamplingFlux=ModelSamplingFlux))
    runtime = Runtime.__new__(Runtime)
    runtime.model = object()
    runtime.sampling_mode = "turbo8"
    a = runtime.sampling_model(512, 768)
    b = runtime.sampling_model(2048, 1024)
    assert a is not b and a is not runtime.model
    assert calls == [(runtime.model, dict(max_shift=0.6935, base_shift=0.5,
                                         width=512, height=768)),
                     (runtime.model, dict(max_shift=0.6935, base_shift=0.5,
                                         width=2048, height=1024))]
    runtime.sampling_mode = "base"
    assert runtime.sampling_model(512, 512) is runtime.model
    assert len(calls) == 2


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


@pytest.mark.parametrize("adapters", [[], [{"stable_id": "turbo8", "source": "x",
                                         "enabled": False, "weight": 1}],
                                    [{"stable_id": "turbo8", "source": "x",
                                      "enabled": True, "weight": 0.5}]])
def test_turbo8_refuses_missing_or_disabled_distillation_adapter(tmp_path, adapters):
    p, runner = _fake_runtime(tmp_path, "raise AssertionError('must not start')")
    p["sampling_mode"] = "turbo8"
    with pytest.raises(ValueError, match="Turbo8 requires its enabled LoRA"):
        QwenImage21Pipeline(p, adapters, runner_path=runner)


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


def test_child_progress_reaches_terminal_before_response(tmp_path, monkeypatch):
    p, runner = _fake_runtime(tmp_path, r'''
import json, os, pathlib, sys, time
request = json.loads(sys.stdin.readline())
os.write(2, '\rШаг 1/2'.encode('utf-8'))
ack = pathlib.Path(request['parameters']['runtime_root']) / 'progress-seen'
deadline = time.monotonic() + 3
while not ack.exists() and time.monotonic() < deadline:
    time.sleep(0.01)
assert ack.exists(), 'progress was buffered until the response'
os.write(2, '\rШаг 2/2\n'.encode('utf-8'))
print(json.dumps({'ok': True, 'result': {}}), flush=True)
sys.stdin.read()
''')

    class Terminal(io.StringIO):
        def flush(self):
            if '\rШаг 1/2' in self.getvalue():
                (tmp_path / 'progress-seen').touch()

    terminal = Terminal()
    monkeypatch.setattr(sys, 'stderr', terminal)
    pipe = QwenImage21Pipeline(p, runner_path=runner, timeout=5)
    try:
        pipe._stop_process()
        expected = '\rШаг 1/2\rШаг 2/2\n'
        assert terminal.getvalue() == expected
        assert (pipe._directory / 'runtime.log').read_bytes() == expected.encode('utf-8')
        assert not pipe._stderr_thread.is_alive()
    finally:
        pipe.close()


def test_child_error_keeps_diagnostic_tail_with_closed_terminal(tmp_path, monkeypatch):
    p, runner = _fake_runtime(tmp_path, r'''
import json, os, sys
sys.stdin.readline()
os.write(2, b'diagnostic detail without a newline')
print(json.dumps({'ok': False, 'error': 'bad checkpoint'}), flush=True)
sys.stdin.read()
''')
    terminal = io.StringIO()
    terminal.close()
    monkeypatch.setattr(sys, 'stderr', terminal)
    with pytest.raises(RuntimeError, match='bad checkpoint\ndiagnostic detail without a newline'):
        QwenImage21Pipeline(p, runner_path=runner, timeout=5)


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


@pytest.mark.parametrize("profile_id", [PROFILE, TURBO8, TURBO])
def test_backend_routes_new_model_sizes_alpha_and_lifecycle(tmp_path, monkeypatch, profile_id):
    import diffusion_editor.workers.qwen_image21_process as module
    p, _ = _fake_runtime(tmp_path, "")
    events = []

    class FakePipeline:
        runtime_info = {"provider": "test", "comfy_revision": "test-revision"}
        scheduler = None

        def __init__(self, parameters, adapters):
            self.parameters = parameters
            assert parameters["sampling_mode"] == ("turbo" if profile_id == TURBO else "turbo8" if profile_id == TURBO8 else "base")
            events.append("load")

        def generate(self, parameters, seed, width, height, **kwargs):
            events.append((width, height, seed))
            if "image" in kwargs:
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
    data = dict(profile_id=profile_id, parameters=p)
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


@pytest.mark.parametrize("get_profile", [text_to_image_profile, image_edit_profile])
def test_official_turbo_uses_its_own_checkpoint_without_lora(get_profile):
    profile = get_profile(TURBO)
    p = profile.normalize({"steps": 40, "true_cfg_scale": 4})
    assert (p["steps"], p["true_cfg_scale"]) == (8, 1)
    assert p["model"].endswith("qwen_image_2.1_turbo_bf16.safetensors")
    assert profile.model_id == "Qwen/Qwen-Image-2.1-Turbo"
    assert profile.default_lora_adapters == ()
    assert get_profile(TURBO8).default_lora_adapters[0].stable_id == "turbo8"


def test_reference_only_padding_preserves_shape_and_transparency():
    from diffusion_editor.workers.qwen_image21_runner import prepare_reference
    source = Image.new("RGBA", (80, 40), (10, 20, 30, 128))
    result = prepare_reference(source, (64, 64), reference_only=True)
    assert result.size == (64, 64)
    assert result.getbbox() == (0, 16, 64, 48)
    assert result.getpixel((32, 32))[3] == 128
    assert source.size == (80, 40)
    assert prepare_reference(source, (64, 64)).getbbox() == (0, 0, 64, 64)


def test_official_turbo_passes_exact_sigmas_to_sampler(monkeypatch):
    torch = SimpleNamespace(zeros=np.zeros, zeros_like=np.zeros_like,
                            tensor=lambda value, dtype: np.array(value, dtype=dtype),
                            float32=np.float32)
    monkeypatch.setitem(sys.modules, "torch", torch)
    from diffusion_editor.workers.qwen_image21_runner import Runtime, TURBO_SIGMAS
    calls = []
    sample = SimpleNamespace(
        prepare_noise=lambda latent, seed: torch.zeros_like(latent),
        sample=lambda *args, **kwargs: calls.append((args, kwargs)) or args[8])
    comfy = SimpleNamespace(sample=sample, utils=SimpleNamespace(PROGRESS_BAR_ENABLED=True))
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.sample", sample)
    monkeypatch.setitem(sys.modules, "comfy.utils", comfy.utils)
    monkeypatch.setitem(sys.modules, "nodes", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "latent_preview", SimpleNamespace(prepare_callback=lambda *a: None))
    runtime = Runtime.__new__(Runtime)
    runtime.sampling_mode = "turbo"
    runtime.sample(object(), 42, 8, 1, [], [], {"samples": torch.zeros((1, 64, 4, 4))})
    args, kwargs = calls[0]
    assert args[2:6] == (8, 1.0, "euler", "simple")
    assert kwargs["sigmas"].tolist() == pytest.approx(
        [1, .978453, .95418, .926626, .89508, .845148, .704534, .414568, 0])
    assert len(TURBO_SIGMAS) == 9
    assert kwargs["seed"] == 42


def test_text_to_image_engine_transports_reference_without_target_pixels():
    import threading
    from diffusion_editor.engines.text_to_image_engine import TextToImageEngine
    from diffusion_editor.generation.types import TextToImageRequest
    reference = Image.new("RGBA", (8, 5), (10, 20, 30, 64))
    calls = []

    class Client:
        def request(self, operation, data, cancel, *, images):
            calls.append((operation, data, images))
            return {"image": Image.new("RGBA", (33, 65)), "seed": 42}

    engine = TextToImageEngine(client=Client())
    request = TextToImageRequest(TURBO, text_to_image_profile(TURBO).defaults(),
                                 33, 65, reference_image=reference)
    result = engine._run_inference(request, threading.Event())
    assert result.image.size == (33, 65)
    operation, data, images = calls[0]
    assert operation == "text_to_image"
    assert images == {"reference_0": reference}
    assert data["reference_count"] == 1
    assert "image" not in data and "image" not in images


def test_reference_only_child_ipc_has_exactly_one_image(tmp_path):
    p, runner = _fake_runtime(tmp_path, '''
import json, sys
from PIL import Image
for line in sys.stdin:
    r = json.loads(line)
    if r['operation'] == 'generate':
        assert r['reference_only'] is True
        assert len(r['images']) == 1
        im = Image.open(r['images'][0])
        assert im.getpixel((0, 0)) == (10, 20, 30, 64)
        im.save(r['output'])
    print(json.dumps({'ok': True, 'result': {'provider': 'test'}}), flush=True)
''')
    pipe = QwenImage21Pipeline(p, runner_path=runner)
    try:
        image = pipe.generate(p, 42, 32, 32,
                              reference_image=Image.new("RGBA", (32, 32), (10, 20, 30, 64)))
        assert image.getpixel((0, 0)) == (10, 20, 30, 64)
    finally:
        pipe.close()
