from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from diffusion_editor.document.layer import Layer
from diffusion_editor.document.layer_stack import LayerStack
from diffusion_editor.document.tool import (
    DiffusionTool,
    InstructTool,
    LamaTool,
)
from diffusion_editor.generation.provenance import (
    FrozenJsonObject,
    GenerationProvenance,
    ModelIdentity,
    ModelIdentityPolicy,
    ModelIdentityPolicyError,
    ModelIdentityStatus,
    RequestProvenance,
    enforce_model_identity_policy,
    floating_model_identity,
    resolve_local_model_identity,
)
from diffusion_editor.workers.ml_backend import RealMlBackend
from diffusion_editor.workers.ml_worker import _Backend


def _assert_numpy_rng_state_equal(before, after) -> None:
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def _provenance(
        *,
        operation: str = "diffusion",
        extensions: dict | None = None,
) -> GenerationProvenance:
    return GenerationProvenance(
        operation=operation,
        model=ModelIdentity(
            provider="local",
            repository="model.safetensors",
            revision=None,
            content_hash=f"sha256:{'a' * 64}",
            local_override="/models/model.safetensors",
            status=ModelIdentityStatus.CONFIRMED_IMMUTABLE,
            extensions=FrozenJsonObject.capture({
                "artifact_size": 123,
            }),
        ),
        request=RequestProvenance.capture(
            operation,
            {
                "prompt": "a lighthouse",
                "input_image_hash": f"image-sha256:{'b' * 64}",
            },
        ),
        seed=17,
        width=8,
        height=6,
        runtime=FrozenJsonObject.capture({
            "pipeline": "FakePipeline",
            "device": "cpu",
            "dtype": "float32",
        }),
        extensions=FrozenJsonObject.capture(extensions),
    )


def test_latent_noise_is_seeded_and_does_not_touch_global_numpy_rng():
    image = Image.new("RGB", (12, 10), (20, 30, 40))
    mask = Image.new("L", image.size, 255)

    np.random.seed(918273)
    before = np.random.get_state()
    first = RealMlBackend._prepare_masked_content(
        image,
        mask,
        "latent_noise",
        seed=123,
    )
    after = np.random.get_state()
    same = RealMlBackend._prepare_masked_content(
        image,
        mask,
        "latent_noise",
        seed=123,
    )
    different = RealMlBackend._prepare_masked_content(
        image,
        mask,
        "latent_noise",
        seed=124,
    )

    _assert_numpy_rng_state_equal(before, after)
    np.testing.assert_array_equal(np.asarray(first), np.asarray(same))
    assert not np.array_equal(np.asarray(first), np.asarray(different))


def test_diffusion_uses_only_a_local_torch_generator_and_seeded_preprocess(
        monkeypatch,
):
    class FakeGenerator:
        def __init__(self, *, device: str):
            assert device == "cpu"
            self.seed = None

        def manual_seed(self, seed: int):
            self.seed = seed
            return self

    # Deliberately provides no torch.manual_seed, torch.seed, or torch.randint:
    # the backend must use only the request-owned Generator.
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(Generator=FakeGenerator),
    )

    class FakePipeline:
        scheduler = SimpleNamespace()

        def __init__(self):
            self.calls: list[tuple[bytes, int]] = []

        def __call__(self, **kwargs):
            generated = kwargs["image"].copy()
            self.calls.append((
                generated.tobytes(),
                kwargs["generator"].seed,
            ))
            return SimpleNamespace(images=[generated])

    backend = RealMlBackend()
    pipe = FakePipeline()
    backend._diffusion_pipe = pipe
    backend._diffusion_mode = "inpaint"
    backend._diffusion_device = "cpu"
    backend._diffusion_dtype = "float32"
    monkeypatch.setattr(
        backend,
        "_ensure_diffusion_mode",
        lambda _mode: None,
    )

    base_data = {
        "prompt": "test",
        "negative_prompt": "",
        "strength": 0.5,
        "steps": 2,
        "guidance_scale": 1.0,
        "mode": "inpaint",
        "masked_content": "latent_noise",
        "ip_adapter_scale": 0.6,
        "width": 8,
        "height": 8,
    }
    images = {
        "image": Image.new("RGB", (8, 8), "red"),
        "mask": Image.new("L", (8, 8), 255),
        "ip_adapter": None,
    }

    first = backend.diffusion({**base_data, "seed": 101}, images)
    second = backend.diffusion({**base_data, "seed": 101}, images)
    third = backend.diffusion({**base_data, "seed": 102}, images)

    assert first[1] == second[1] == 101
    assert third[1] == 102
    assert pipe.calls[0] == pipe.calls[1]
    assert pipe.calls[0] != pipe.calls[2]
    assert first[2] == second[2]
    assert first[2]["request"]["parameters"]["seed"] == 101


def test_fake_worker_response_matches_versioned_provenance_golden(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    model_path = tmp_path / "fake-model.safetensors"
    model_bytes = b"fake-model-artifact"
    model_path.write_bytes(model_bytes)
    progress: list[str] = []
    backend = _Backend("fake")

    loaded = backend.execute(
        "load_diffusion",
        {
            "output_dir": str(output_dir),
            "model_path": str(model_path),
            "prediction_type": None,
        },
        progress.append,
    )
    request = {
        "output_dir": str(output_dir),
        "prompt": "golden prompt",
        "negative_prompt": "noise",
        "strength": 0.4,
        "steps": 3,
        "guidance_scale": 2.5,
        "seed": 77,
        "mode": "txt2img",
        "masked_content": "original",
        "ip_adapter_scale": 0.6,
        "width": 8,
        "height": 6,
    }
    generated = backend.execute(
        "diffusion",
        request,
        progress.append,
    )

    digest = hashlib.sha256(model_bytes).hexdigest()
    expected_identity = {
        "provider": "local",
        "repository": model_path.name,
        "revision": None,
        "content_hash": f"sha256:{digest}",
        "local_override": str(model_path),
        "status": "confirmed_immutable",
        "warning": None,
    }
    assert loaded["model_info"]["model_identity"] == expected_identity
    assert generated["provenance"] == {
        "schema_version": 1,
        "operation": "diffusion",
        "model": expected_identity,
        "request": {
            "schema_version": 1,
            "kind": "diffusion",
            "parameters": {
                key: request[key]
                for key in (
                    "prompt",
                    "negative_prompt",
                    "strength",
                    "steps",
                    "guidance_scale",
                    "seed",
                    "mode",
                    "masked_content",
                    "ip_adapter_scale",
                    "width",
                    "height",
                )
            },
        },
        "seed": 77,
        "width": 8,
        "height": 6,
        "runtime": {
            "backend": "fake",
            "pipeline": "FakeDiffusionPipeline",
            "scheduler": "FakeScheduler",
            "device": "cpu",
            "dtype": "float32",
        },
        "warnings": [],
    }


def test_provenance_dto_preserves_future_versions_and_unknown_fields():
    raw = {
        "schema_version": 9,
        "operation": "future-generation",
        "model": {
            "provider": "registry",
            "repository": "org/model",
            "revision": "immutable-revision",
            "content_hash": None,
            "local_override": None,
            "status": "future-attested",
            "warning": None,
            "signature": {"algorithm": "future-v2"},
        },
        "request": {
            "schema_version": 4,
            "kind": "future-generation",
            "parameters": {"seed": 1},
            "request_extension": [1, 2, 3],
        },
        "seed": 1,
        "width": 16,
        "height": 8,
        "runtime": {"runtime_extension": True},
        "warnings": [],
        "top_level_extension": {"nested": "kept"},
    }

    decoded = GenerationProvenance.from_dict(raw)

    assert decoded.to_dict() == raw
    assert not decoded.model.is_confirmed_immutable
    assert enforce_model_identity_policy(
        decoded.model,
        ModelIdentityPolicy.WARN,
    ) == (
        "Model identity is future-attested; "
        "exact artifact is not confirmed",
    )


def test_model_identity_hash_verification_and_floating_policy(tmp_path):
    model_path = tmp_path / "model.bin"
    model_path.write_bytes(b"verified weights")
    expected = "sha256:" + hashlib.sha256(
        b"verified weights"
    ).hexdigest()

    identity, warnings = resolve_local_model_identity(
        model_path,
        expected_content_hash=expected,
        policy=ModelIdentityPolicy.REQUIRE_IMMUTABLE,
    )
    assert identity.status == ModelIdentityStatus.CONFIRMED_IMMUTABLE
    assert identity.content_hash == expected
    assert warnings == ()

    mismatch, warnings = resolve_local_model_identity(
        model_path,
        expected_content_hash=f"sha256:{'0' * 64}",
    )
    assert mismatch.status == ModelIdentityStatus.HASH_MISMATCH
    assert mismatch.content_hash == expected
    assert "hash mismatch" in warnings[0].lower()
    with pytest.raises(ModelIdentityPolicyError, match="hash mismatch"):
        resolve_local_model_identity(
            model_path,
            expected_content_hash=f"sha256:{'0' * 64}",
            policy=ModelIdentityPolicy.REQUIRE_IMMUTABLE,
        )

    missing, warnings = resolve_local_model_identity(
        tmp_path / "missing.bin",
    )
    assert missing.status == ModelIdentityStatus.UNKNOWN
    assert missing.content_hash is None
    assert "does not exist" in warnings[0]

    floating = floating_model_identity(
        "huggingface",
        "org/model",
        revision="main",
    )
    assert enforce_model_identity_policy(
        floating,
        ModelIdentityPolicy.WARN,
    ) == (floating.warning,)
    with pytest.raises(ModelIdentityPolicyError, match="floating"):
        enforce_model_identity_policy(
            floating,
            ModelIdentityPolicy.REQUIRE_IMMUTABLE,
        )

    pinned_revision = ModelIdentity(
        provider="registry",
        repository="org/model",
        revision="immutable-revision-id",
        content_hash=None,
        local_override=None,
        status=ModelIdentityStatus.CONFIRMED_IMMUTABLE,
    )
    assert enforce_model_identity_policy(
        pinned_revision,
        ModelIdentityPolicy.REQUIRE_IMMUTABLE,
    ) == ()


def test_generation_provenance_and_identity_survive_project_roundtrip():
    stack = LayerStack()
    stack.on_changed = lambda: None
    stack.init_from_image(np.zeros((6, 8, 4), dtype=np.uint8))

    diffusion = DiffusionTool(
        source_patch=None,
        patch_x=0,
        patch_y=0,
        patch_w=8,
        patch_h=6,
        prompt="a lighthouse",
        negative_prompt="",
        strength=0.5,
        guidance_scale=7.0,
        steps=10,
        seed=17,
        model_path="/models/model.safetensors",
        mode="txt2img",
    )
    provenance = _provenance(extensions={
        "future_top_level": {"kept": True},
    })
    diffusion.model_identity = provenance.model
    diffusion.model_identity_policy = ModelIdentityPolicy.REQUIRE_IMMUTABLE
    diffusion.generation_provenance = provenance

    tools = [
        ("Diffusion", diffusion),
        (
            "Instruct",
            InstructTool(
                source_patch=None,
                patch_x=0,
                patch_y=0,
                patch_w=8,
                patch_h=6,
            ),
        ),
        (
            "LaMa",
            LamaTool(
                source_patch=None,
                patch_x=0,
                patch_y=0,
                patch_w=8,
                patch_h=6,
            ),
        ),
    ]
    for name, tool in tools:
        layer = Layer(name, 8, 6)
        layer.tool = tool
        if tool is not diffusion:
            tool.generation_provenance = _provenance(
                operation=tool.tool_type,
            )
        stack.insert_layer(layer)

    restored = LayerStack()
    restored.on_changed = lambda: None
    restored.load_state(stack.serialize_state())
    by_name = {layer.name: layer for layer in restored.all_layers()}

    restored_diffusion = by_name["Diffusion"].tool
    assert restored_diffusion.model_identity.to_dict() == (
        provenance.model.to_dict()
    )
    assert (
        restored_diffusion.model_identity_policy
        == ModelIdentityPolicy.REQUIRE_IMMUTABLE
    )
    assert restored_diffusion.generation_provenance.to_dict() == (
        provenance.to_dict()
    )
    assert (
        by_name["Instruct"].tool.generation_provenance.operation
        == "instruct"
    )
    assert by_name["LaMa"].tool.generation_provenance.operation == "lama"
