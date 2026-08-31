from diffusion_editor.document.tool import TextToImageTool
from diffusion_editor.generation.text_to_image_profiles import (
    DEFAULT_QWEN_IMAGE_OFFLOAD_MODE,
    QWEN_IMAGE_2512_PROFILE_ID,
    QWEN_IMAGE_PROFILE_ID,
    TEXT_TO_IMAGE_PROFILE_SCHEMA_VERSION,
    normalize_text_to_image_profile_store,
    text_to_image_profile,
    text_to_image_profiles,
)
from diffusion_editor.generation.image_edit_profiles import (
    QWEN_TEXT_ENCODER_HERETIC_BF16_ID,
    QWEN_TEXT_ENCODER_HUIHUI_BF16_ID,
    QWEN_TEXT_ENCODER_STANDARD_FP8_ID,
    QWEN_TEXT_ENCODER_UPSTREAM_ID,
)


def test_qwen_image_profile_exposes_txt2img_defaults():
    profile = text_to_image_profile(QWEN_IMAGE_2512_PROFILE_ID)
    values = profile.defaults()

    assert profile.provider == "diffusers.qwen_image"
    assert values["model"] == "Qwen/Qwen-Image-2512"
    lightning = profile.default_lora_adapters[0]
    assert lightning.stable_id == "lightning"
    assert lightning.label == "Lightning 4-step"
    assert values["steps"] == (4 if lightning.enabled else 50)
    assert values["true_cfg_scale"] == (1.0 if lightning.enabled else 4.0)
    if lightning.enabled:
        assert lightning.source.endswith(
            "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors")
    assert values["max_sequence_length"] == 512
    assert values["offload_mode"] == DEFAULT_QWEN_IMAGE_OFFLOAD_MODE
    assert values["text_encoder_variant"] == (
        QWEN_TEXT_ENCODER_STANDARD_FP8_ID
        if values["text_encoder_checkpoint"]
        else QWEN_TEXT_ENCODER_UPSTREAM_ID
    )
    assert QWEN_TEXT_ENCODER_HERETIC_BF16_ID in {
        choice.value
        for choice in profile.parameter("text_encoder_variant").choices
    }
    assert QWEN_TEXT_ENCODER_HUIHUI_BF16_ID in {
        choice.value
        for choice in profile.parameter("text_encoder_variant").choices
    }
    if values["transformer_checkpoint"]:
        assert values["transformer_checkpoint"].endswith(
            "qwen_image_2512_fp8_e4m3fn_scaled_comfyui.safetensors")
    if values["text_encoder_checkpoint"]:
        assert values["text_encoder_checkpoint"].endswith(
            "qwen_2.5_vl_7b_fp8_scaled.safetensors")
    assert "cpu_offload" not in values
    assert profile.inference_size(23, 11) == (32, 16)
    assert profile.inference_size(1024, 768) == (1024, 768)


def test_original_qwen_image_is_an_independent_txt2img_profile():
    profile = text_to_image_profile(QWEN_IMAGE_PROFILE_ID)
    values = profile.defaults()

    assert [item.stable_id for item in text_to_image_profiles()] == [
        QWEN_IMAGE_2512_PROFILE_ID,
        QWEN_IMAGE_PROFILE_ID,
    ]
    assert profile.title == "Qwen Image (August 2025)"
    assert profile.provider == "diffusers.qwen_image"
    assert profile.model_id == "Qwen/Qwen-Image"
    assert values["model"] == "Qwen/Qwen-Image"
    assert values["steps"] == 25
    assert values["true_cfg_scale"] == 3.0
    assert profile.default_lora_adapters == ()
    if values["transformer_checkpoint"]:
        assert values["transformer_checkpoint"].endswith(
            "qwen_image_fp8mixed.safetensors")
    if values["text_encoder_checkpoint"]:
        assert values["text_encoder_checkpoint"].endswith(
            "qwen_2.5_vl_7b_fp8_scaled.safetensors")


def test_qwen_image_profiles_keep_parameters_and_loras_independent():
    tool = TextToImageTool()
    tool.set_profile(QWEN_IMAGE_PROFILE_ID)
    tool.set_parameter("prompt", "original base model")
    tool.set_parameter("steps", 31)
    tool.set_lora_adapters([{
        "stable_id": "base-only",
        "label": "Base-only adapter",
        "source": "/models/base-only.safetensors",
        "weight": 0.7,
        "enabled": True,
    }])

    tool.set_profile(QWEN_IMAGE_2512_PROFILE_ID)
    assert tool.parameters["prompt"] == ""
    assert tool.parameters["steps"] != 31
    assert all(adapter.stable_id != "base-only"
               for adapter in tool.lora_adapters)

    tool.set_profile(QWEN_IMAGE_PROFILE_ID)
    assert tool.parameters["prompt"] == "original base model"
    assert tool.parameters["steps"] == 31
    assert [adapter.stable_id for adapter in tool.lora_adapters] == [
        "base-only"]


def test_tool_normalizes_profile_parameters_without_patch_geometry():
    tool = TextToImageTool(profile_parameters={
        QWEN_IMAGE_2512_PROFILE_ID: {
            "prompt": "river valley",
            "steps": "24",
            "seed": "42",
        },
    })

    assert tool.parameters["prompt"] == "river valley"
    assert tool.parameters["steps"] == 24
    assert tool.parameters["seed"] == 42
    assert not hasattr(tool, "patch_w")


def test_legacy_cpu_offload_migrates_to_explicit_vram_strategy():
    profile = text_to_image_profile(QWEN_IMAGE_2512_PROFILE_ID)
    legacy = profile.defaults()
    legacy.pop("offload_mode")
    legacy["cpu_offload"] = True

    migrated = normalize_text_to_image_profile_store({
        QWEN_IMAGE_2512_PROFILE_ID: legacy,
    })[QWEN_IMAGE_2512_PROFILE_ID]

    assert migrated["offload_mode"] == "model"
    assert "cpu_offload" not in migrated

    legacy["cpu_offload"] = False
    migrated = normalize_text_to_image_profile_store({
        QWEN_IMAGE_2512_PROFILE_ID: legacy,
    })[QWEN_IMAGE_2512_PROFILE_ID]
    assert migrated["offload_mode"] == "none"


def test_pre_fp8_profile_adopts_checkpoint_pair_and_component_offload():
    profile = text_to_image_profile(QWEN_IMAGE_2512_PROFILE_ID)
    old_values = profile.defaults()
    old_values.pop("transformer_checkpoint")
    old_values.pop("text_encoder_checkpoint")
    old_values["offload_mode"] = "group"

    migrated = normalize_text_to_image_profile_store({
        QWEN_IMAGE_2512_PROFILE_ID: old_values,
    })[QWEN_IMAGE_2512_PROFILE_ID]

    assert migrated["transformer_checkpoint"] == profile.parameter(
        "transformer_checkpoint").default
    assert migrated["text_encoder_checkpoint"] == profile.parameter(
        "text_encoder_checkpoint").default
    assert migrated["offload_mode"] == "model"


def test_pre_lightning_tool_adopts_acceleration_defaults_once():
    profile = text_to_image_profile(QWEN_IMAGE_2512_PROFILE_ID)
    old_values = profile.defaults()
    old_values.update({"steps": 50, "true_cfg_scale": 4.0})
    tool = TextToImageTool(
        profile_parameters={QWEN_IMAGE_2512_PROFILE_ID: old_values},
        profile_lora_adapters={QWEN_IMAGE_2512_PROFILE_ID: []},
        profile_schema_version=1,
    )

    assert tool.profile_schema_version == TEXT_TO_IMAGE_PROFILE_SCHEMA_VERSION
    if profile.default_lora_adapters[0].enabled:
        assert tool.parameters["steps"] == 4
        assert tool.parameters["true_cfg_scale"] == 1.0
        assert [adapter.stable_id for adapter in tool.lora_adapters] == [
            "lightning"]
    else:
        assert tool.parameters["steps"] == 50
        assert tool.parameters["true_cfg_scale"] == 4.0


def test_current_tool_keeps_explicit_empty_lora_stack():
    tool = TextToImageTool(
        profile_lora_adapters={QWEN_IMAGE_2512_PROFILE_ID: []},
        profile_schema_version=TEXT_TO_IMAGE_PROFILE_SCHEMA_VERSION,
    )

    assert tool.lora_adapters == ()


def test_schema_v2_does_not_reapply_lightning_migration():
    profile = text_to_image_profile(QWEN_IMAGE_2512_PROFILE_ID)
    values = profile.defaults()
    values.update({"steps": 50, "true_cfg_scale": 4.0})
    tool = TextToImageTool(
        profile_parameters={QWEN_IMAGE_2512_PROFILE_ID: values},
        profile_lora_adapters={QWEN_IMAGE_2512_PROFILE_ID: []},
        profile_schema_version=2,
    )

    assert tool.parameters["steps"] == 50
    assert tool.parameters["true_cfg_scale"] == 4.0
    assert tool.lora_adapters == ()
