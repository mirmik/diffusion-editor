from __future__ import annotations

from types import SimpleNamespace
import sys

from PIL import Image
import pytest

from diffusion_editor.generation.image_edit_profiles import (
    FLUX2_KLEIN_PROFILE_ID,
    LEGACY_INSTRUCT_PROFILE_ID,
    QWEN_IMAGE_EDIT_PROFILE_ID,
    QWEN_IMAGE_EDIT_RAPID_AIO_V23_PROFILE_ID,
    QWEN_TEXT_ENCODER_HERETIC_BF16_ID,
    QWEN_TEXT_ENCODER_UPSTREAM_ID,
    SENSENOVA_U15_PROFILE_ID,
    image_edit_profile,
)
from diffusion_editor.generation.text_to_image_profiles import (
    QWEN_IMAGE_2512_PROFILE_ID,
    text_to_image_profile,
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


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("group", "group"),
        ("model", "model"),
        ("sequential", "sequential"),
        ("none", "to:cuda"),
    ],
)
def test_text_to_image_memory_modes(monkeypatch, mode, expected):
    events: list[object] = []

    class FakePipeline:
        def enable_group_offload(self, **kwargs):
            events.append(("group", kwargs))

        def enable_model_cpu_offload(self):
            events.append("model")

        def enable_sequential_cpu_offload(self):
            events.append("sequential")

        def to(self, device):
            events.append(f"to:{device}")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(device=lambda value: f"device:{value}"),
    )

    effective = RealMlBackend._configure_text_to_image_memory(
        FakePipeline(), mode, "cuda")

    assert effective == mode
    if mode == "group":
        event, kwargs = events[0]
        assert event == expected
        assert kwargs == {
            "onload_device": "device:cuda",
            "offload_device": "device:cpu",
            "offload_type": "block_level",
            "num_blocks_per_group": 1,
            "use_stream": True,
            "low_cpu_mem_usage": True,
        }
    else:
        assert events == [expected]


def test_text_to_image_cpu_ignores_accelerator_offload_mode():
    events: list[str] = []
    pipe = SimpleNamespace(to=lambda device: events.append(device))

    effective = RealMlBackend._configure_text_to_image_memory(
        pipe, "group", "cpu")

    assert effective == "none"
    assert events == ["cpu"]


def test_qwen_lora_loader_normalizes_prefixed_kohya_checkpoint(
    monkeypatch,
    tmp_path,
):
    source = tmp_path / "prefixed-qwen.safetensors"
    source.touch()
    raw_state = {
        "transformer.transformer_blocks.0.attn.to_q.alpha": object(),
        (
            "transformer.transformer_blocks.0.attn.to_q."
            "lora_down.weight"
        ): object(),
        (
            "transformer.transformer_blocks.0.attn.to_q."
            "lora_up.weight"
        ): object(),
    }
    captured: dict[str, object] = {}

    class Header:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def keys(self):
            return raw_state.keys()

    class FakePipe:
        def load_lora_weights(self, state_dict, *, adapter_name):
            captured["state_dict"] = state_dict
            captured["adapter_name"] = adapter_name

    monkeypatch.setitem(
        sys.modules,
        "safetensors",
        SimpleNamespace(
            SafetensorError=RuntimeError,
            safe_open=lambda *_args, **_kwargs: Header(),
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "safetensors.torch",
        SimpleNamespace(load_file=lambda _path: raw_state),
    )

    RealMlBackend._load_qwen_lora_weights(
        FakePipe(), str(source), "test_adapter")

    assert captured["adapter_name"] == "test_adapter"
    assert set(captured["state_dict"]) == {
        "transformer_blocks.0.attn.to_q.alpha",
        "transformer_blocks.0.attn.to_q.lora_down.weight",
        "transformer_blocks.0.attn.to_q.lora_up.weight",
    }


def test_text_to_image_load_applies_scaled_fp8_component_offload(monkeypatch):
    captured: dict[str, object] = {}

    class FakeQwenPipeline(_FakePipeline):
        transformer = object()

        def enable_model_cpu_offload(self):
            captured["model_offload"] = True

        def load_lora_weights(self, path, *, adapter_name):
            captured["loaded_lora"] = (path, adapter_name)

        def set_adapters(self, names, *, adapter_weights):
            captured["active_lora"] = (names, adapter_weights)

    class PipelineFactory:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured["model"] = model
            captured["load_kwargs"] = kwargs
            return FakeQwenPipeline()

    backend = RealMlBackend()
    monkeypatch.setattr(
        backend, "_release_accelerator_memory", lambda: None)
    monkeypatch.setattr(backend, "_device", lambda _requested: "cuda")
    monkeypatch.setattr(
        backend, "_configured_dtype",
        lambda _name, _device: "bfloat16")
    monkeypatch.setattr(
        backend, "_cast_adapter_parameters",
        lambda _model, name, dtype: captured.update(
            cast_lora=(name, dtype)))
    monkeypatch.setattr(
        backend,
        "_load_qwen_transformer_component",
        lambda model, **kwargs: (
            captured.update(transformer_model=model, transformer_kwargs=kwargs)
            or ("fp8-transformer", "local-scaled-fp8")
        ),
    )
    monkeypatch.setattr(
        backend,
        "_load_qwen_text_encoder_component",
        lambda model, **kwargs: (
            captured.update(encoder_model=model, encoder_kwargs=kwargs)
            or ("fp8-text-encoder", "local-scaled-fp8")
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(device=lambda value: f"device:{value}"),
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(QwenImagePipeline=PipelineFactory),
    )
    parameters = text_to_image_profile(
        QWEN_IMAGE_2512_PROFILE_ID).defaults()
    parameters.update({
        "model": "fake/qwen-image",
        "local_files_only": True,
        "transformer_checkpoint": "/models/qwen-image-2512-fp8.safetensors",
        "text_encoder_checkpoint": "/models/qwen-vl-fp8.safetensors",
    })

    loaded = backend.load_text_to_image({
        "profile_id": QWEN_IMAGE_2512_PROFILE_ID,
        "parameters": parameters,
        "lora_adapters": [{
            "stable_id": "lightning",
            "label": "Lightning 4-step",
            "source": "/models/qwen-image-lightning.safetensors",
            "weight": 1.0,
            "enabled": True,
        }],
    })

    assert loaded["loaded"] is True
    assert loaded["offload_mode"] == "model"
    assert loaded["component_mode"] == "local-scaled-fp8"
    assert captured["model"] == "fake/qwen-image"
    assert captured["transformer_model"] == "fake/qwen-image"
    assert captured["transformer_kwargs"] == {
        "transformer_path": "/models/qwen-image-2512-fp8.safetensors",
        "compute_dtype": "bfloat16",
        "revision": None,
        "local_files_only": True,
    }
    assert captured["encoder_kwargs"] == {
        "text_encoder_source": "/models/qwen-vl-fp8.safetensors",
        "compute_dtype": "bfloat16",
        "revision": None,
        "local_files_only": True,
    }
    assert captured["load_kwargs"]["transformer"] == "fp8-transformer"
    assert captured["load_kwargs"]["text_encoder"] == "fp8-text-encoder"
    assert captured["loaded_lora"] == (
        "/models/qwen-image-lightning.safetensors",
        "text_to_image_0_lightning",
    )
    assert captured["cast_lora"] == (
        "text_to_image_0_lightning", "bfloat16")
    assert captured["active_lora"] == (
        ["text_to_image_0_lightning"], [1.0])
    assert loaded["active_lora_adapters"] == [
        "text_to_image_0_lightning"]
    assert captured["model_offload"] is True


def test_text_to_image_combines_fp8_transformer_with_heretic_encoder(
        monkeypatch, tmp_path):
    from diffusion_editor.generation import image_edit_profiles

    captured: dict[str, object] = {}
    heretic = tmp_path / "heretic"
    heretic.mkdir()

    class FakeQwenPipeline(_FakePipeline):
        transformer = object()

    class PipelineFactory:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured.update(model=model, load_kwargs=kwargs)
            return FakeQwenPipeline()

    monkeypatch.setattr(
        image_edit_profiles, "_QWEN_HERETIC_TEXT_ENCODER", str(heretic))
    backend = RealMlBackend()
    monkeypatch.setattr(
        backend, "_release_accelerator_memory", lambda: None)
    monkeypatch.setattr(backend, "_device", lambda _requested: "cpu")
    monkeypatch.setattr(
        backend, "_configured_dtype", lambda _name, _device: "bfloat16")
    monkeypatch.setattr(
        backend, "_cached_pipeline_directory",
        lambda _model, _revision: "/cache/qwen-image-2512")
    monkeypatch.setattr(
        backend, "_load_qwen_transformer_component",
        lambda _model, **_kwargs: (
            "fp8-transformer", "local-scaled-fp8"))
    monkeypatch.setattr(
        backend, "_load_qwen_text_encoder_component",
        lambda _model, **kwargs: (
            captured.update(encoder_kwargs=kwargs)
            or ("heretic-encoder", "text-encoder-bfloat16")))
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        SimpleNamespace(QwenImagePipeline=PipelineFactory),
    )
    parameters = text_to_image_profile(
        QWEN_IMAGE_2512_PROFILE_ID).defaults()
    parameters.update({
        "model": "fake/qwen-image",
        "device": "cpu",
        "local_files_only": True,
        "offload_mode": "none",
        "transformer_checkpoint": "/models/qwen-image-fp8.safetensors",
        "text_encoder_variant": QWEN_TEXT_ENCODER_HERETIC_BF16_ID,
    })

    loaded = backend.load_text_to_image({
        "profile_id": QWEN_IMAGE_2512_PROFILE_ID,
        "parameters": parameters,
        "lora_adapters": [],
    })

    assert loaded["component_mode"] == (
        "local-scaled-fp8+text-encoder-bfloat16")
    assert captured["model"] == "/cache/qwen-image-2512"
    assert captured["load_kwargs"]["text_encoder"] == "heretic-encoder"
    assert captured["encoder_kwargs"]["text_encoder_source"] == str(heretic)
    assert loaded["model_identity"]["text_encoder"] == {
        "source": str(heretic),
        "variant": QWEN_TEXT_ENCODER_HERETIC_BF16_ID,
    }


def test_text_to_image_omits_inactive_negative_prompt(monkeypatch):
    captured: dict[str, object] = {}

    class FakeGenerator:
        def __init__(self, *, device):
            captured["generator_device"] = device

        def manual_seed(self, seed):
            captured["seed"] = seed
            return self

    class FakeQwenPipeline:
        def __call__(self, **kwargs):
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                images=[Image.new("RGB", (16, 16), "red")])

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(Generator=FakeGenerator))
    profile = text_to_image_profile(QWEN_IMAGE_2512_PROFILE_ID)
    adapters = tuple(
        adapter.to_dict() for adapter in profile.default_lora_adapters)
    backend = RealMlBackend()
    backend._text_to_image_pipe = FakeQwenPipeline()
    backend._text_to_image_profile_id = QWEN_IMAGE_2512_PROFILE_ID
    backend._text_to_image_lora_adapters = adapters
    backend._text_to_image_device = "cpu"
    backend._text_to_image_dtype = "bfloat16"
    backend._text_to_image_offload_mode = "none"
    backend._text_to_image_component_mode = "local-scaled-fp8"
    parameters = profile.defaults()
    parameters.update({
        "prompt": "teapot",
        "negative_prompt": "watermark",
        "true_cfg_scale": 1.0,
        "seed": 42,
    })

    backend.text_to_image({
        "profile_id": QWEN_IMAGE_2512_PROFILE_ID,
        "parameters": parameters,
        "lora_adapters": adapters,
        "width": 16,
        "height": 16,
    })

    assert "negative_prompt" not in captured["kwargs"]
    assert captured["kwargs"]["true_cfg_scale"] == 1.0


@pytest.mark.parametrize(
    "profile_id",
    [
        QWEN_IMAGE_EDIT_PROFILE_ID,
        QWEN_IMAGE_EDIT_RAPID_AIO_V23_PROFILE_ID,
        FLUX2_KLEIN_PROFILE_ID,
    ],
)
def test_diffusers_image_edit_passes_second_image_as_ordered_list(
        monkeypatch, profile_id):
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
    backend._image_edit_profile_id = profile_id
    backend._instruct_device = "cpu"
    backend._instruct_dtype = "float32"
    backend._image_edit_lora_adapters = tuple(
        adapter.to_dict()
        for adapter in image_edit_profile(
            profile_id).default_lora_adapters
    )
    parameters = image_edit_profile(profile_id).defaults()
    parameters.update({"prompt": "combine", "seed": 123})
    first = Image.new("RGBA", (8, 6), "red")
    second = Image.new("RGBA", (5, 7), "green")

    result, seed, _provenance = backend.image_edit(
        {
            "profile_id": profile_id,
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
    if profile_id in {
        QWEN_IMAGE_EDIT_PROFILE_ID,
        QWEN_IMAGE_EDIT_RAPID_AIO_V23_PROFILE_ID,
    }:
        assert "negative_prompt" not in captured


def test_qwen_multiple_angles_loads_lightning_and_angle_adapters(monkeypatch):
    captured: dict[str, object] = {"loaded": [], "cast": []}

    class FakeQwenPipeline(_FakePipeline):
        def __init__(self):
            super().__init__()
            self.transformer = object()

        def load_lora_weights(self, path, *, adapter_name):
            captured["loaded"].append((path, adapter_name))

        def set_adapters(self, names, *, adapter_weights):
            captured["active"] = (names, adapter_weights)

    backend = RealMlBackend()
    monkeypatch.setattr(
        backend, "_release_accelerator_memory", lambda: None)
    monkeypatch.setattr(backend, "_device", lambda _requested: "cpu")
    monkeypatch.setattr(
        backend, "_configured_dtype",
        lambda _name, _device: "float32")
    monkeypatch.setattr(
        backend,
        "_cast_adapter_parameters",
        lambda _model, name, _dtype: captured["cast"].append(name),
    )
    monkeypatch.setitem(
        sys.modules,
        "diffusers",
        _fake_diffusers(FakeQwenPipeline),
    )
    parameters = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID).defaults()
    parameters.update({
        "model": "fake/qwen-image-edit",
        "device": "cpu",
        "dtype": "float32",
        "local_files_only": True,
        "cpu_offload": False,
        "transformer_checkpoint": "",
        "text_encoder_checkpoint": "",
        "text_encoder_variant": QWEN_TEXT_ENCODER_UPSTREAM_ID,
    })
    adapters = (
        {
            "stable_id": "lightning",
            "label": "Lightning",
            "source": "/models/lightning.safetensors",
            "weight": 1.0,
            "enabled": True,
        },
        {
            "stable_id": "multiple-angles",
            "label": "Multiple Angles",
            "source": "/models/multiple-angles.safetensors",
            "weight": 0.9,
            "enabled": True,
        },
    )

    loaded = backend.load_image_edit({
        "profile_id": QWEN_IMAGE_EDIT_PROFILE_ID,
        "parameters": parameters,
        "lora_adapters": adapters,
    })

    assert loaded["profile_id"] == QWEN_IMAGE_EDIT_PROFILE_ID
    assert loaded["active_lora_adapters"] == [
        "image_edit_0_lightning", "image_edit_1_multiple_angles",
    ]
    assert loaded["active_lora_weights"] == [1.0, 0.9]
    assert captured["loaded"] == [
        ("/models/lightning.safetensors", "image_edit_0_lightning"),
        (
            "/models/multiple-angles.safetensors",
            "image_edit_1_multiple_angles",
        ),
    ]
    assert captured["cast"] == [
        "image_edit_0_lightning", "image_edit_1_multiple_angles",
    ]
    assert captured["active"] == (
        ["image_edit_0_lightning", "image_edit_1_multiple_angles"],
        [1.0, 0.9],
    )


def test_qwen_rapid_aio_loads_sharded_fp8_without_default_lora(
        monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    transformer_dir = tmp_path / "rapid-v23"
    transformer_dir.mkdir()

    class FakeQwenPipeline(_FakePipeline):
        transformer = object()

    class PipelineFactory:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            captured.update(model=model, load_kwargs=kwargs)
            return FakeQwenPipeline()

    backend = RealMlBackend()
    monkeypatch.setattr(
        backend, "_release_accelerator_memory", lambda: None)
    monkeypatch.setattr(backend, "_device", lambda _requested: "cpu")
    monkeypatch.setattr(
        backend, "_configured_dtype",
        lambda _name, _device: "bfloat16")
    monkeypatch.setattr(
        backend,
        "_cached_pipeline_directory",
        lambda _model, _revision: "/cache/qwen-edit-2511",
    )
    monkeypatch.setattr(
        backend,
        "_load_qwen_transformer_component",
        lambda model, **kwargs: (
            captured.update(transformer_model=model, transformer_kwargs=kwargs)
            or ("rapid-transformer", "local-diffusers-fp8-layerwise")
        ),
    )
    monkeypatch.setattr(
        backend,
        "_load_qwen_text_encoder_component",
        lambda model, **kwargs: (
            captured.update(encoder_model=model, encoder_kwargs=kwargs)
            or ("fp8-text-encoder", "local-scaled-fp8")
        ),
    )
    fake_diffusers = _fake_diffusers(FakeQwenPipeline)
    fake_diffusers.QwenImageEditPlusPipeline = PipelineFactory
    monkeypatch.setitem(sys.modules, "diffusers", fake_diffusers)
    profile = image_edit_profile(
        QWEN_IMAGE_EDIT_RAPID_AIO_V23_PROFILE_ID)
    parameters = profile.defaults()
    parameters.update({
        "model": "Qwen/Qwen-Image-Edit-2511",
        "device": "cpu",
        "dtype": "bfloat16",
        "cpu_offload": False,
        "local_files_only": True,
        "transformer_checkpoint": str(transformer_dir),
        "text_encoder_checkpoint": "/models/qwen-vl-fp8.safetensors",
    })

    loaded = backend.load_image_edit({
        "profile_id": profile.stable_id,
        "parameters": parameters,
        "lora_adapters": [],
    })

    assert loaded["component_mode"] == "local-diffusers-fp8-layerwise"
    assert loaded["active_lora_adapters"] == []
    assert loaded["model_identity"]["repository"] == profile.model_id
    assert captured["model"] == "/cache/qwen-edit-2511"
    assert captured["load_kwargs"]["transformer"] == "rapid-transformer"
    assert captured["transformer_kwargs"]["compute_dtype"] == "bfloat16"


def test_qwen_rapid_aio_rejects_missing_transformer(monkeypatch):
    monkeypatch.setitem(
        sys.modules, "diffusers", _fake_diffusers(_FakePipeline))
    profile = image_edit_profile(
        QWEN_IMAGE_EDIT_RAPID_AIO_V23_PROFILE_ID)
    parameters = profile.defaults()
    parameters.update({
        "transformer_checkpoint": "",
        "text_encoder_checkpoint": "",
    })

    with pytest.raises(ValueError, match="requires its Transformer"):
        RealMlBackend().load_image_edit({
            "profile_id": profile.stable_id,
            "parameters": parameters,
            "lora_adapters": [],
        })


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

        def edit(self, image, parameters, seed, *, reference_image=None):
            captured["parameters"] = parameters
            captured["seed"] = seed
            captured["image"] = image
            captured["reference_image"] = reference_image
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
        Image.new("RGB", (5, 7), "blue"),
    )

    assert loaded["component_mode"] == "gguf"
    assert captured["model_path"] == str(model_dir)
    assert captured["gguf_checkpoint"] == str(checkpoint)
    assert captured["seed"] == 42015
    assert captured["image"].size == (8, 6)
    assert captured["reference_image"].size == (5, 7)
    assert result.size == (8, 6)
    assert seed == 42015
    assert provenance["request"]["parameters"][
        "model_profile_id"] == SENSENOVA_U15_PROFILE_ID
    assert provenance["model"]["content_hash"].startswith("sha256:")
    assert provenance["model"]["config_identity"][
        "local_override"] == str(model_dir)
