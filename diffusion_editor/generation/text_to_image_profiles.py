"""Provider-neutral profiles for image generation without source pixels."""

from __future__ import annotations

from dataclasses import dataclass, replace
import os
from pathlib import Path
from typing import Any, Iterable

from .image_edit_profiles import (
    ImageEditLoraAdapter as TextToImageLoraAdapter,
    ImageEditParameter as TextToImageParameter,
    ParameterChoice,
    ParameterKind,
    infer_qwen_text_encoder_variant,
    normalize_lora_adapters,
    qwen_text_encoder_parameters,
)


QWEN_IMAGE_PROFILE_ID = "qwen-image"
QWEN_IMAGE_2512_PROFILE_ID = "qwen-image-2512"
DEFAULT_TEXT_TO_IMAGE_PROFILE_ID = QWEN_IMAGE_2512_PROFILE_ID
DEFAULT_QWEN_IMAGE_OFFLOAD_MODE = "model"
TEXT_TO_IMAGE_PROFILE_SCHEMA_VERSION = 4


def _choice(value: str, label: str | None = None) -> ParameterChoice:
    return ParameterChoice(value=value, label=label or value)


@dataclass(frozen=True)
class TextToImageProfile:
    stable_id: str
    title: str
    provider: str
    model_id: str
    description: str
    parameters: tuple[TextToImageParameter, ...]
    default_lora_adapters: tuple[TextToImageLoraAdapter, ...] = ()
    dimension_multiple: int = 1

    def parameter(self, stable_id: str) -> TextToImageParameter:
        for parameter in self.parameters:
            if parameter.stable_id == stable_id:
                return parameter
        raise KeyError(stable_id)

    def defaults(self) -> dict[str, Any]:
        return {
            parameter.stable_id: parameter.default
            for parameter in self.parameters
        }

    def normalize(self, values: dict[str, Any] | None) -> dict[str, Any]:
        source = values or {}
        return {
            parameter.stable_id: parameter.normalize(
                source.get(parameter.stable_id, parameter.default)
            )
            for parameter in self.parameters
        }

    def load_values(self, values: dict[str, Any] | None) -> dict[str, Any]:
        normalized = self.normalize(values)
        return {
            parameter.stable_id: normalized[parameter.stable_id]
            for parameter in self.parameters
            if parameter.load_time
        }

    def normalize_lora_adapters(
        self,
        values: Iterable[TextToImageLoraAdapter | dict[str, Any]] | None,
    ) -> tuple[TextToImageLoraAdapter, ...]:
        source = self.default_lora_adapters if values is None else values
        return normalize_lora_adapters(source)

    def inference_size(self, width: int, height: int) -> tuple[int, int]:
        """Return the smallest provider-valid size covering the target."""

        if width < 1 or height < 1:
            raise ValueError("text-to-image dimensions must be positive")
        multiple = max(1, int(self.dimension_multiple))
        return (
            max(multiple, ((int(width) + multiple - 1) // multiple) * multiple),
            max(multiple, ((int(height) + multiple - 1) // multiple) * multiple),
        )


_QWEN_IMAGE_MODEL = os.environ.get(
    "DIFFUSION_EDITOR_QWEN_IMAGE_MODEL",
    "Qwen/Qwen-Image-2512",
)
_QWEN_IMAGE_BASE_MODEL = os.environ.get(
    "DIFFUSION_EDITOR_QWEN_IMAGE_BASE_MODEL",
    "Qwen/Qwen-Image",
)
_DEFAULT_QWEN_IMAGE_TRANSFORMER = Path(
    "~/soft/ComfyUI/models/diffusion_models/"
    "qwen_image_2512_fp8_e4m3fn_scaled_comfyui.safetensors"
).expanduser()
_DEFAULT_QWEN_IMAGE_BASE_TRANSFORMER = Path(
    "~/soft/ComfyUI/models/diffusion_models/"
    "qwen_image_fp8mixed.safetensors"
).expanduser()
_DEFAULT_QWEN_TEXT_ENCODER = Path(
    "~/soft/ComfyUI/models/text_encoders/"
    "qwen_2.5_vl_7b_fp8_scaled.safetensors"
).expanduser()
_DEFAULT_QWEN_IMAGE_LORA = Path(
    "~/soft/ComfyUI/models/loras/"
    "Qwen-Image-2512-Lightning-4steps-V1.0-bf16.safetensors"
).expanduser()
_QWEN_IMAGE_TRANSFORMER = os.environ.get(
    "DIFFUSION_EDITOR_QWEN_IMAGE_TRANSFORMER",
    str(_DEFAULT_QWEN_IMAGE_TRANSFORMER)
    if _DEFAULT_QWEN_IMAGE_TRANSFORMER.is_file() else "",
)
_QWEN_IMAGE_BASE_TRANSFORMER = os.environ.get(
    "DIFFUSION_EDITOR_QWEN_IMAGE_BASE_TRANSFORMER",
    str(_DEFAULT_QWEN_IMAGE_BASE_TRANSFORMER)
    if _DEFAULT_QWEN_IMAGE_BASE_TRANSFORMER.is_file() else "",
)
_QWEN_IMAGE_TEXT_ENCODER = os.environ.get(
    "DIFFUSION_EDITOR_QWEN_IMAGE_TEXT_ENCODER",
    str(_DEFAULT_QWEN_TEXT_ENCODER)
    if _DEFAULT_QWEN_TEXT_ENCODER.is_file() else "",
)
_QWEN_IMAGE_LORA = os.environ.get(
    "DIFFUSION_EDITOR_QWEN_IMAGE_LORA",
    str(_DEFAULT_QWEN_IMAGE_LORA)
    if _DEFAULT_QWEN_IMAGE_LORA.is_file() else "",
)


_PROFILES = (
    TextToImageProfile(
        stable_id=QWEN_IMAGE_2512_PROFILE_ID,
        title="Qwen Image 2512",
        provider="diffusers.qwen_image",
        model_id=_QWEN_IMAGE_MODEL,
        description=(
            "Text-to-image generation. Output dimensions always follow the "
            "pixel size of the target layer."
        ),
        dimension_multiple=16,
        parameters=(
            TextToImageParameter(
                "prompt", "Prompt", ParameterKind.TEXT, "", "Conditioning",
                placeholder="Describe the image to generate",
            ),
            TextToImageParameter(
                "negative_prompt", "Negative prompt", ParameterKind.TEXT,
                "", "Conditioning",
            ),
            TextToImageParameter(
                "true_cfg_scale", "True CFG scale", ParameterKind.FLOAT,
                1.0 if _QWEN_IMAGE_LORA else 4.0,
                "Sampling", 0.0, 20.0, 0.1, 2,
            ),
            TextToImageParameter(
                "steps", "Steps", ParameterKind.INTEGER,
                4 if _QWEN_IMAGE_LORA else 50,
                "Sampling", 1, 100, 1,
            ),
            TextToImageParameter(
                "seed", "Seed", ParameterKind.INTEGER, -1, "Sampling",
                -1, 2**32 - 1, 1,
            ),
            TextToImageParameter(
                "max_sequence_length", "Max sequence length",
                ParameterKind.INTEGER, 512, "Conditioning", 1, 1024, 1,
            ),
            TextToImageParameter(
                "model", "Model / local directory", ParameterKind.STRING,
                _QWEN_IMAGE_MODEL, "Model",
                placeholder="Hugging Face ID or local directory",
                load_time=True,
            ),
            TextToImageParameter(
                "revision", "Revision", ParameterKind.STRING, "", "Model",
                placeholder="branch, tag or commit (empty = default)",
                load_time=True,
            ),
            TextToImageParameter(
                "local_files_only", "Local files only",
                ParameterKind.BOOLEAN, False, "Model", load_time=True,
            ),
            TextToImageParameter(
                "transformer_checkpoint", "Scaled FP8 transformer",
                ParameterKind.STRING, _QWEN_IMAGE_TRANSFORMER, "Model",
                placeholder="Comfy-style scaled FP8 safetensors",
                load_time=True,
            ),
        ) + qwen_text_encoder_parameters(
            custom_source_default=_QWEN_IMAGE_TEXT_ENCODER,
        ) + (
            TextToImageParameter(
                "dtype", "Dtype", ParameterKind.CHOICE,
                "bfloat16", "Runtime",
                choices=(
                    _choice("bfloat16"), _choice("float16"),
                    _choice("float32"),
                ),
                load_time=True,
            ),
            TextToImageParameter(
                "device", "Device", ParameterKind.CHOICE,
                "cuda", "Runtime",
                choices=(_choice("cuda"), _choice("cpu")),
                load_time=True,
            ),
            TextToImageParameter(
                "offload_mode", "VRAM strategy", ParameterKind.CHOICE,
                DEFAULT_QWEN_IMAGE_OFFLOAD_MODE, "Runtime",
                choices=(
                    _choice("model", "Component CPU offload (FP8 default)"),
                    _choice("group", "Group CPU offload (BF16 fallback)"),
                    _choice("sequential", "Sequential CPU offload (slowest)"),
                    _choice("none", "Resident on device (most VRAM)"),
                ),
                load_time=True,
            ),
            TextToImageParameter(
                "vae_tiling", "VAE tiling", ParameterKind.BOOLEAN,
                False, "Runtime", load_time=True,
            ),
            TextToImageParameter(
                "attention_kwargs", "Attention kwargs (JSON)",
                ParameterKind.STRING, "", "Runtime", placeholder="{}",
            ),
        ),
        default_lora_adapters=(
            TextToImageLoraAdapter(
                "lightning",
                "Lightning 4-step",
                _QWEN_IMAGE_LORA,
                1.0,
                bool(_QWEN_IMAGE_LORA),
            ),
        ),
    ),
)


def _with_defaults(
        parameters: tuple[TextToImageParameter, ...],
        **defaults: Any) -> tuple[TextToImageParameter, ...]:
    """Clone a shared provider contract with profile-specific defaults."""

    return tuple(
        replace(parameter, default=defaults[parameter.stable_id])
        if parameter.stable_id in defaults else parameter
        for parameter in parameters
    )


_PROFILES += (
    TextToImageProfile(
        stable_id=QWEN_IMAGE_PROFILE_ID,
        title="Qwen Image (August 2025)",
        provider="diffusers.qwen_image",
        model_id=_QWEN_IMAGE_BASE_MODEL,
        description=(
            "Original Qwen Image text-to-image model from August 2025. "
            "Output dimensions always follow the pixel size of the target "
            "layer."
        ),
        dimension_multiple=16,
        parameters=_with_defaults(
            _PROFILES[0].parameters,
            model=_QWEN_IMAGE_BASE_MODEL,
            transformer_checkpoint=_QWEN_IMAGE_BASE_TRANSFORMER,
            steps=25,
            true_cfg_scale=3.0,
        ),
        default_lora_adapters=(),
    ),
)

_PROFILE_BY_ID = {profile.stable_id: profile for profile in _PROFILES}


def text_to_image_profiles() -> tuple[TextToImageProfile, ...]:
    return _PROFILES


def text_to_image_profile(stable_id: str) -> TextToImageProfile:
    try:
        return _PROFILE_BY_ID[stable_id]
    except KeyError as exc:
        raise ValueError(f"Unknown text-to-image profile: {stable_id}") from exc


def all_text_to_image_parameters() -> tuple[TextToImageParameter, ...]:
    result: list[TextToImageParameter] = []
    seen: set[str] = set()
    for profile in _PROFILES:
        for parameter in profile.parameters:
            if parameter.stable_id not in seen:
                result.append(parameter)
                seen.add(parameter.stable_id)
    return tuple(result)


def normalize_text_to_image_profile_store(
    values: dict[str, dict[str, Any]] | None,
    *,
    adopt_lightning_defaults: bool = False,
) -> dict[str, dict[str, Any]]:
    source = values or {}
    result: dict[str, dict[str, Any]] = {}
    for profile in _PROFILES:
        stored = source.get(profile.stable_id)
        if isinstance(stored, dict):
            stored = dict(stored)
            if "text_encoder_variant" not in stored:
                stored["text_encoder_variant"] = (
                    infer_qwen_text_encoder_variant(stored)
                )
            if (
                    adopt_lightning_defaults
                    and profile.stable_id == QWEN_IMAGE_2512_PROFILE_ID
                    and _QWEN_IMAGE_LORA):
                # Preserve explicit sampling choices while replacing only the
                # former unaccelerated defaults in pre-Lightning documents.
                if stored.get("steps") == 50:
                    stored["steps"] = profile.parameter("steps").default
                if stored.get("true_cfg_scale") == 4.0:
                    stored["true_cfg_scale"] = profile.parameter(
                        "true_cfg_scale").default
            if (
                    profile.stable_id == QWEN_IMAGE_2512_PROFILE_ID
                    and "transformer_checkpoint" not in stored
                    and "text_encoder_checkpoint" not in stored):
                # Profiles saved before the scaled-FP8 runtime existed must
                # adopt both checkpoints as a pair.  The short-lived BF16
                # default used group offload, which is unnecessarily slow for
                # the smaller FP8 components.
                stored["transformer_checkpoint"] = profile.parameter(
                    "transformer_checkpoint").default
                stored["text_encoder_checkpoint"] = profile.parameter(
                    "text_encoder_checkpoint").default
                if stored.get("offload_mode") == "group":
                    stored["offload_mode"] = DEFAULT_QWEN_IMAGE_OFFLOAD_MODE
            if (
                    profile.stable_id == QWEN_IMAGE_2512_PROFILE_ID
                    and "offload_mode" not in stored
                    and "cpu_offload" in stored):
                # Replace the former boolean with the equivalent explicit
                # mode.  The scaled-FP8 default makes component-level offload
                # safe on a 32 GiB GPU.
                stored["offload_mode"] = (
                    DEFAULT_QWEN_IMAGE_OFFLOAD_MODE
                    if bool(stored["cpu_offload"]) else "none"
                )
        result[profile.stable_id] = profile.normalize(stored)
    return result


def normalize_text_to_image_lora_store(
    values: dict[str, list[dict[str, Any]]] | None,
    *,
    adopt_lightning_defaults: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    source = values if isinstance(values, dict) else {}
    result: dict[str, list[dict[str, Any]]] = {}
    for profile in _PROFILES:
        stored = source.get(profile.stable_id)
        if (
                adopt_lightning_defaults
                and profile.stable_id == QWEN_IMAGE_2512_PROFILE_ID
                and isinstance(stored, (list, tuple))
                and not stored):
            # Old Text to Image tools serialized an explicit empty adapter
            # list, so normal defaulting alone cannot adopt Lightning.
            stored = None
        adapters = profile.normalize_lora_adapters(
            stored if isinstance(stored, (list, tuple)) else None)
        result[profile.stable_id] = [
            adapter.to_dict() for adapter in adapters]
    return result
