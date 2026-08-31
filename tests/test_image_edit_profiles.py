from __future__ import annotations

import io
import zipfile

from PIL import Image

from diffusion_editor.document.tool import InstructTool
from diffusion_editor.document.tool_serialization import (
    load_tool,
    save_tool_assets,
    serialize_tool,
)
from diffusion_editor.generation.image_edit_profiles import (
    DEFAULT_IMAGE_EDIT_PROFILE_ID,
    FLUX2_KLEIN_PROFILE_ID,
    LEGACY_INSTRUCT_PROFILE_ID,
    QWEN_IMAGE_EDIT_PROFILE_ID,
    QWEN_IMAGE_EDIT_RAPID_AIO_V23_PROFILE_ID,
    QWEN_TEXT_ENCODER_CUSTOM_ID,
    QWEN_TEXT_ENCODER_HERETIC_BF16_ID,
    QWEN_TEXT_ENCODER_HUIHUI_BF16_ID,
    QWEN_TEXT_ENCODER_STANDARD_FP8_ID,
    QWEN_TEXT_ENCODER_UPSTREAM_ID,
    SENSENOVA_U15_PROFILE_ID,
    all_image_edit_parameters,
    image_edit_profile,
    image_edit_profiles,
    normalize_lora_adapters,
    normalize_profile_store,
    qwen_multiple_angles_lora_adapter,
    resolve_qwen_text_encoder_source,
)
from diffusion_editor.generation.provenance import capture_tool_state


def _tool(profile_id=DEFAULT_IMAGE_EDIT_PROFILE_ID):
    return InstructTool(
        source_patch=None,
        patch_x=1,
        patch_y=2,
        patch_w=32,
        patch_h=24,
        model_profile_id=profile_id,
    )


def test_builtin_profiles_are_stable_and_declarative():
    profiles = image_edit_profiles()
    assert [profile.stable_id for profile in profiles] == [
        QWEN_IMAGE_EDIT_PROFILE_ID,
        QWEN_IMAGE_EDIT_RAPID_AIO_V23_PROFILE_ID,
        FLUX2_KLEIN_PROFILE_ID,
        SENSENOVA_U15_PROFILE_ID,
        LEGACY_INSTRUCT_PROFILE_ID,
    ]
    assert profiles[0].primary
    assert image_edit_profile(FLUX2_KLEIN_PROFILE_ID).fast
    for profile in profiles:
        ids = [parameter.stable_id for parameter in profile.parameters]
        assert len(ids) == len(set(ids))
        assert {"prompt", "steps", "seed", "model", "dtype", "device"} <= set(ids)
        assert set(profile.normalize({})) == set(ids)
    assert [profile.max_input_images for profile in profiles] == [
        2, 2, 2, 2, 1,
    ]


def test_qwen_prefers_installed_scaled_fp8_components():
    qwen = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
    defaults = qwen.defaults()

    assert bool(defaults["transformer_checkpoint"]) == bool(
        defaults["text_encoder_checkpoint"])
    if defaults["transformer_checkpoint"]:
        assert defaults["transformer_checkpoint"].endswith(
            "qwen_image_edit_2511_fp8mixed.safetensors")
        assert defaults["text_encoder_checkpoint"].endswith(
            "qwen_2.5_vl_7b_fp8_scaled.safetensors")
    assert defaults["cpu_offload"] is bool(
        defaults["transformer_checkpoint"])
    assert defaults["text_encoder_variant"] == (
        QWEN_TEXT_ENCODER_STANDARD_FP8_ID
        if defaults["text_encoder_checkpoint"]
        else QWEN_TEXT_ENCODER_UPSTREAM_ID
    )
    assert [choice.value for choice in qwen.parameter(
        "text_encoder_variant").choices] == [
        QWEN_TEXT_ENCODER_UPSTREAM_ID,
        QWEN_TEXT_ENCODER_STANDARD_FP8_ID,
        QWEN_TEXT_ENCODER_HERETIC_BF16_ID,
        QWEN_TEXT_ENCODER_HUIHUI_BF16_ID,
        QWEN_TEXT_ENCODER_CUSTOM_ID,
    ]
    assert image_edit_profile(
        FLUX2_KLEIN_PROFILE_ID).defaults()["cpu_offload"] is False


def test_qwen_encoder_selector_resolves_independently(monkeypatch, tmp_path):
    from diffusion_editor.generation import image_edit_profiles

    heretic = tmp_path / "heretic"
    heretic.mkdir()
    huihui = tmp_path / "huihui"
    huihui.mkdir()
    monkeypatch.setattr(
        image_edit_profiles, "_QWEN_HERETIC_TEXT_ENCODER", str(heretic))
    monkeypatch.setattr(
        image_edit_profiles, "_QWEN_HUIHUI_TEXT_ENCODER", str(huihui))

    assert resolve_qwen_text_encoder_source({
        "text_encoder_variant": QWEN_TEXT_ENCODER_UPSTREAM_ID,
    }) == ""
    assert resolve_qwen_text_encoder_source({
        "text_encoder_variant": QWEN_TEXT_ENCODER_HERETIC_BF16_ID,
    }) == str(heretic)
    assert resolve_qwen_text_encoder_source({
        "text_encoder_variant": QWEN_TEXT_ENCODER_HUIHUI_BF16_ID,
    }) == str(huihui)
    assert resolve_qwen_text_encoder_source({
        "text_encoder_variant": QWEN_TEXT_ENCODER_CUSTOM_ID,
        "text_encoder_checkpoint": "org/custom-encoder",
    }) == "org/custom-encoder"


def test_qwen_rapid_aio_v23_is_an_independent_four_step_profile():
    profile = image_edit_profile(
        QWEN_IMAGE_EDIT_RAPID_AIO_V23_PROFILE_ID)
    defaults = profile.defaults()

    assert profile.fast
    assert not profile.primary
    assert profile.model_id == (
        "prithivMLmods/Qwen-Image-Edit-Rapid-AIO-V23")
    assert defaults["model"] == "Qwen/Qwen-Image-Edit-2511"
    assert defaults["steps"] == 4
    assert defaults["true_cfg_scale"] == 1.0
    assert defaults["guidance_scale"] == 1.0
    assert profile.default_lora_adapters == ()
    if defaults["transformer_checkpoint"]:
        assert defaults["transformer_checkpoint"].endswith(
            "qwen-image-edit-rapid-aio-v23")
        assert defaults["text_encoder_checkpoint"].endswith(
            "qwen_2.5_vl_7b_fp8_scaled.safetensors")
        assert defaults["cpu_offload"] is True


def test_qwen_multiple_angles_is_an_explicit_optional_adapter():
    profile = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
    adapter = qwen_multiple_angles_lora_adapter()

    assert [item.stable_id for item in profile.default_lora_adapters] == [
        "lightning",
    ]
    assert adapter.stable_id == "multiple-angles"
    assert adapter.source.endswith(
        "qwen-image-edit-2511-multiple-angles-lora.safetensors")
    assert adapter.weight == 1.0
    assert not {
        "lora_path", "lora_scale", "angle_lora_path", "angle_lora_scale",
    } & {parameter.stable_id for parameter in profile.parameters}


def test_lora_stack_normalizes_ten_ordered_unique_adapters():
    adapters = normalize_lora_adapters([
        {
            "stable_id": f"adapter-{index}",
            "label": f"Adapter {index}",
            "source": f"/models/{index}.safetensors",
            "weight": index / 10,
            "enabled": index % 2 == 0,
        }
        for index in range(10)
    ])

    assert len(adapters) == 10
    assert [adapter.stable_id for adapter in adapters] == [
        f"adapter-{index}" for index in range(10)
    ]
    assert adapters[7].source == "/models/7.safetensors"
    assert adapters[7].weight == 0.7
    assert adapters[7].enabled is False


def test_sensenova_prefers_installed_config_and_gguf():
    profile = image_edit_profile(SENSENOVA_U15_PROFILE_ID)
    defaults = profile.defaults()

    assert profile.provider == "sensenova_u1.it2i"
    assert defaults["model"]
    assert defaults["gguf_checkpoint"]
    assert defaults["gguf_checkpoint"].endswith(
        "SenseNova-U1.5-8B-MoT-Preview-Q8.gguf")
    assert defaults["steps"] == 8
    assert defaults["cfg_scale"] == 4.0
    assert defaults["img_cfg_scale"] == 1.0
    assert defaults["target_megapixels"] == 1.0


def test_pre_fp8_qwen_store_adopts_new_component_offload_default():
    old_qwen = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID).defaults()
    old_qwen.pop("transformer_checkpoint")
    old_qwen.pop("text_encoder_checkpoint")
    old_qwen["cpu_offload"] = False

    migrated = normalize_profile_store({
        QWEN_IMAGE_EDIT_PROFILE_ID: old_qwen,
    })[QWEN_IMAGE_EDIT_PROFILE_ID]

    assert migrated["cpu_offload"] is bool(
        migrated["transformer_checkpoint"])


def test_schema_union_contains_every_profile_parameter_once():
    union = all_image_edit_parameters()
    union_ids = [parameter.stable_id for parameter in union]
    assert len(union_ids) == len(set(union_ids))
    assert {
        parameter.stable_id
        for profile in image_edit_profiles()
        for parameter in profile.parameters
    } == set(union_ids)


def test_switching_profiles_keeps_independent_parameter_values():
    tool = _tool()
    tool.set_parameter("prompt", "precise qwen edit")
    tool.set_parameter("true_cfg_scale", 1.25)
    tool.set_profile(FLUX2_KLEIN_PROFILE_ID)
    tool.set_parameter("prompt", "fast flux edit")
    tool.set_parameter("guidance_scale", 1.5)
    tool.set_profile(QWEN_IMAGE_EDIT_PROFILE_ID)

    assert tool.instruction == "precise qwen edit"
    assert tool.parameters["true_cfg_scale"] == 1.25
    tool.set_profile(FLUX2_KLEIN_PROFILE_ID)
    assert tool.instruction == "fast flux edit"
    assert tool.guidance_scale == 1.5


def test_image_edit_profiles_roundtrip_with_legacy_fields():
    tool = _tool(FLUX2_KLEIN_PROFILE_ID)
    tool.set_parameter("prompt", "replace the sky")
    tool.set_parameter("steps", 7)
    data = serialize_tool(tool, "edit")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w"):
        pass
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue()), "r") as archive:
        restored = load_tool(data, archive).tool

    assert isinstance(restored, InstructTool)
    assert restored.model_profile_id == FLUX2_KLEIN_PROFILE_ID
    assert restored.parameters["prompt"] == "replace the sky"
    assert restored.parameters["steps"] == 7
    restored.set_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
    assert restored.parameters == image_edit_profile(
        QWEN_IMAGE_EDIT_PROFILE_ID).defaults()


def test_flat_qwen_lora_fields_migrate_to_persistent_adapter_stack():
    old_parameters = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID).defaults()
    old_parameters.update({
        "lora_path": "/models/old-lightning.safetensors",
        "lora_scale": 0.75,
        "angle_lora_path": "/models/old-angles.safetensors",
        "angle_lora_scale": 0.85,
    })
    tool = InstructTool(
        source_patch=None,
        patch_x=0,
        patch_y=0,
        patch_w=8,
        patch_h=8,
        model_profile_id=QWEN_IMAGE_EDIT_PROFILE_ID,
        profile_parameters={
            QWEN_IMAGE_EDIT_PROFILE_ID: old_parameters,
        },
    )

    assert [adapter.source for adapter in tool.lora_adapters] == [
        "/models/old-lightning.safetensors",
        "/models/old-angles.safetensors",
    ]
    assert [adapter.weight for adapter in tool.lora_adapters] == [0.75, 0.85]
    assert "lora_path" not in tool.parameters
    data = serialize_tool(tool, "edit")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w"):
        pass
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue()), "r") as archive:
        restored = load_tool(data, archive).tool
    assert [adapter.to_dict() for adapter in restored.lora_adapters] == [
        adapter.to_dict() for adapter in tool.lora_adapters]


def test_explicit_empty_lora_stack_survives_roundtrip():
    tool = _tool(QWEN_IMAGE_EDIT_PROFILE_ID)
    tool.set_lora_adapters([])
    data = serialize_tool(tool, "edit")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w"):
        pass
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue()), "r") as archive:
        restored = load_tool(data, archive).tool

    assert restored.lora_adapters == ()


def test_image_edit_external_reference_roundtrip():
    tool = _tool(QWEN_IMAGE_EDIT_PROFILE_ID)
    tool.reference_image = Image.new("RGB", (7, 5), (12, 34, 56))
    tool.reference_image_name_hint = "portrait.png"
    data = serialize_tool(tool, "edit")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        save_tool_assets(tool, archive, "edit")
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue()), "r") as archive:
        restored = load_tool(data, archive).tool

    assert restored.reference_image_name_hint == "portrait.png"
    assert restored.reference_image.size == (7, 5)
    assert restored.reference_image.getpixel((0, 0)) == (12, 34, 56)


def test_legacy_instruct_document_migrates_without_changing_values():
    data = {
        "tool_type": "instruct",
        "patch_x": 0,
        "patch_y": 0,
        "patch_w": 8,
        "patch_h": 8,
        "instruction": "make it blue",
        "image_guidance_scale": 2.2,
        "guidance_scale": 9.0,
        "steps": 33,
        "seed": 44,
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w"):
        pass
    with zipfile.ZipFile(io.BytesIO(buffer.getvalue()), "r") as archive:
        tool = load_tool(data, archive).tool

    assert isinstance(tool, InstructTool)
    assert tool.model_profile_id == LEGACY_INSTRUCT_PROFILE_ID
    assert tool.instruction == "make it blue"
    assert tool.image_guidance_scale == 2.2
    assert tool.guidance_scale == 9.0
    assert tool.steps == 33
    assert tool.seed == 44


def test_tool_state_fingerprint_captures_profile_and_every_parameter():
    tool = _tool(QWEN_IMAGE_EDIT_PROFILE_ID)
    before = capture_tool_state(tool).fingerprint
    tool.set_parameter("negative_prompt", "no geometry drift")
    assert capture_tool_state(tool).fingerprint != before
    before_switch = capture_tool_state(tool).fingerprint
    tool.set_profile(FLUX2_KLEIN_PROFILE_ID)
    assert capture_tool_state(tool).fingerprint != before_switch


def test_tool_state_fingerprint_captures_lora_stack():
    tool = _tool(QWEN_IMAGE_EDIT_PROFILE_ID)
    before = capture_tool_state(tool).fingerprint
    adapters = [adapter.to_dict() for adapter in tool.lora_adapters]
    adapters[0]["weight"] = 0.55
    tool.set_lora_adapters(adapters)

    assert capture_tool_state(tool).fingerprint != before
