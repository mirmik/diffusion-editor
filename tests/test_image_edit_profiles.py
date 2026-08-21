from __future__ import annotations

import io
import zipfile

from diffusion_editor.document.tool import InstructTool
from diffusion_editor.document.tool_serialization import load_tool, serialize_tool
from diffusion_editor.generation.image_edit_profiles import (
    DEFAULT_IMAGE_EDIT_PROFILE_ID,
    FLUX2_KLEIN_PROFILE_ID,
    LEGACY_INSTRUCT_PROFILE_ID,
    QWEN_IMAGE_EDIT_PROFILE_ID,
    SENSENOVA_U15_PROFILE_ID,
    all_image_edit_parameters,
    image_edit_profile,
    image_edit_profiles,
    normalize_profile_store,
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
        FLUX2_KLEIN_PROFILE_ID,
        SENSENOVA_U15_PROFILE_ID,
        LEGACY_INSTRUCT_PROFILE_ID,
    ]
    assert profiles[0].primary
    assert profiles[1].fast
    for profile in profiles:
        ids = [parameter.stable_id for parameter in profile.parameters]
        assert len(ids) == len(set(ids))
        assert {"prompt", "steps", "seed", "model", "dtype", "device"} <= set(ids)
        assert set(profile.normalize({})) == set(ids)


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
    assert image_edit_profile(
        FLUX2_KLEIN_PROFILE_ID).defaults()["cpu_offload"] is False


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
