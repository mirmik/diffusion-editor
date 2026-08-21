from __future__ import annotations

from types import SimpleNamespace
import sys

from PIL import Image
import pytest

from diffusion_editor.generation.image_edit_profiles import (
    LEGACY_INSTRUCT_PROFILE_ID,
    QWEN_IMAGE_EDIT_PROFILE_ID,
    SENSENOVA_U15_PROFILE_ID,
    image_edit_profile,
)
from diffusion_editor.workers.ml_backend import RealMlBackend


class _FakeVae:
    def enable_tiling(self) -> None:
        pass


class _FakeScheduler:
    config = {}


class _FakePipeline:
    def __init__(self) -> None:
        self.vae = _FakeVae()
        self.scheduler = _FakeScheduler()

    def to(self, _device: str) -> None:
        pass


def _legacy_parameters() -> dict[str, object]:
    parameters = image_edit_profile(
        LEGACY_INSTRUCT_PROFILE_ID).defaults()
    parameters.update({
        "model": "fake/instruct-model",
        "device": "cpu",
        "dtype": "float32",
        "local_files_only": True,
    })
    return parameters


def _fake_diffusers(loader):
    class PipelineFactory:
        @classmethod
        def from_pretrained(cls, *_args, **_kwargs):
            return loader()

    class SchedulerFactory:
        @classmethod
        def from_config(cls, _config):
            return _FakeScheduler()

    return SimpleNamespace(
        EulerAncestralDiscreteScheduler=SchedulerFactory,
        Flux2KleinPipeline=PipelineFactory,
        QwenImageEditPlusPipeline=PipelineFactory,
        StableDiffusionInstructPix2PixPipeline=PipelineFactory,
    )


def test_image_edit_load_releases_old_pipeline_before_constructing_new(
        monkeypatch):
    backend = RealMlBackend()
    backend._instruct_pipe = object()
    backend._image_edit_profile_id = "old-profile"
    events: list[str] = []

    monkeypatch.setattr(
        backend, "_release_accelerator_memory",
        lambda: events.append("release"))
    monkeypatch.setattr(backend, "_device", lambda _requested: "cpu")
    monkeypatch.setattr(
        backend, "_configured_dtype",
        lambda _name, _device: "float32")

    def load_new():
        events.append("load")
        assert backend._instruct_pipe is None
        assert backend._image_edit_profile_id is None
        return _FakePipeline()

    monkeypatch.setitem(sys.modules, "diffusers", _fake_diffusers(load_new))

    result = backend.load_image_edit({
        "profile_id": LEGACY_INSTRUCT_PROFILE_ID,
        "parameters": _legacy_parameters(),
    })

    assert events == ["release", "load"]
    assert result["loaded"] is True
    assert backend._instruct_pipe is not None
    assert backend._image_edit_profile_id == LEGACY_INSTRUCT_PROFILE_ID


def test_failed_image_edit_reload_leaves_backend_unloaded(monkeypatch):
    backend = RealMlBackend()
    backend._instruct_pipe = object()
    backend._instruct_identity = object()
    backend._instruct_warnings = ("old warning",)
    backend._instruct_device = "cuda"
    backend._instruct_dtype = "float16"
    backend._image_edit_profile_id = "old-profile"

    monkeypatch.setattr(
        backend, "_release_accelerator_memory", lambda: None)
    monkeypatch.setattr(backend, "_device", lambda _requested: "cpu")
    monkeypatch.setattr(
        backend, "_configured_dtype",
        lambda _name, _device: "float32")

    def fail_load():
        assert backend._instruct_pipe is None
        raise RuntimeError("new model failed")

    monkeypatch.setitem(sys.modules, "diffusers", _fake_diffusers(fail_load))

    with pytest.raises(RuntimeError, match="new model failed"):
        backend.load_image_edit({
            "profile_id": LEGACY_INSTRUCT_PROFILE_ID,
            "parameters": _legacy_parameters(),
        })

    assert backend._instruct_pipe is None
    assert backend._instruct_identity is None
    assert backend._instruct_warnings == ()
    assert backend._instruct_device is None
    assert backend._instruct_dtype is None
    assert backend._image_edit_profile_id is None


def test_qwen_image_edit_passes_second_image_as_ordered_list(monkeypatch):
    captured: dict[str, object] = {}

    class FakeGenerator:
        def __init__(self, *, device):
            captured["generator_device"] = device

        def manual_seed(self, seed):
            captured["seed"] = seed
            return self

    class FakeEditPipeline(_FakePipeline):
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", (8, 6), "blue")])

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(Generator=FakeGenerator))
    backend = RealMlBackend()
    backend._instruct_pipe = FakeEditPipeline()
    backend._image_edit_profile_id = QWEN_IMAGE_EDIT_PROFILE_ID
    backend._instruct_device = "cpu"
    backend._instruct_dtype = "float32"
    parameters = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID).defaults()
    parameters.update({"prompt": "combine", "seed": 123})
    first = Image.new("RGBA", (8, 6), "red")
    second = Image.new("RGBA", (5, 7), "green")

    result, seed, _provenance = backend.image_edit(
        {
            "profile_id": QWEN_IMAGE_EDIT_PROFILE_ID,
            "parameters": parameters,
        },
        first,
        second,
    )

    assert seed == 123
    assert result.size == (8, 6)
    assert isinstance(captured["image"], list)
    assert [image.size for image in captured["image"]] == [(8, 6), (5, 7)]
    assert all(image.mode == "RGB" for image in captured["image"])


def test_sensenova_provider_loads_gguf_and_uses_standalone_edit_adapter(
        monkeypatch, tmp_path):
    from diffusion_editor.workers import sensenova_image_edit

    backend = RealMlBackend()
    model_dir = tmp_path / "config"
    model_dir.mkdir()
    checkpoint = tmp_path / "model.gguf"
    checkpoint.write_bytes(b"fake-sensenova-weights")
    captured: dict[str, object] = {}

    class FakeSenseNovaPipeline:
        scheduler = None
        runtime_info = {"vram_mode": "full"}

        def __init__(self, **kwargs):
            captured.update(kwargs)

        def edit(self, image, parameters, seed):
            captured["parameters"] = parameters
            captured["seed"] = seed
            return image.convert("RGB")

    monkeypatch.setattr(
        sensenova_image_edit,
        "SenseNovaImageEditPipeline",
        FakeSenseNovaPipeline,
    )
    monkeypatch.setattr(
        backend, "_release_accelerator_memory", lambda: None)
    monkeypatch.setattr(backend, "_device", lambda _requested: "cuda")
    monkeypatch.setattr(
        backend, "_configured_dtype",
        lambda _name, _device: "bfloat16")
    monkeypatch.setitem(sys.modules, "diffusers", _fake_diffusers(
        lambda: pytest.fail("Diffusers pipeline must not load for SenseNova")))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    parameters = image_edit_profile(SENSENOVA_U15_PROFILE_ID).defaults()
    parameters.update({
        "model": str(model_dir),
        "gguf_checkpoint": str(checkpoint),
        "prompt": "repaint the car",
        "seed": 42015,
    })

    loaded = backend.load_image_edit({
        "profile_id": SENSENOVA_U15_PROFILE_ID,
        "parameters": parameters,
    })
    result, seed, provenance = backend.image_edit(
        {
            "profile_id": SENSENOVA_U15_PROFILE_ID,
            "parameters": parameters,
        },
        Image.new("RGB", (8, 6), "red"),
    )

    assert loaded["component_mode"] == "gguf"
    assert captured["model_path"] == str(model_dir)
    assert captured["gguf_checkpoint"] == str(checkpoint)
    assert captured["seed"] == 42015
    assert result.size == (8, 6)
    assert seed == 42015
    assert provenance["request"]["parameters"][
        "model_profile_id"] == SENSENOVA_U15_PROFILE_ID
    assert provenance["model"]["content_hash"].startswith("sha256:")
    assert provenance["model"]["config_identity"][
        "local_override"] == str(model_dir)
