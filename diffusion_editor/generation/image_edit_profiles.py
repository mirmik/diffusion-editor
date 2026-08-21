"""Provider-neutral profiles and parameter schema for image editing models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
from typing import Any, Iterable


LEGACY_INSTRUCT_PROFILE_ID = "instruct-pix2pix"
QWEN_IMAGE_EDIT_PROFILE_ID = "qwen-image-edit-2511"
FLUX2_KLEIN_PROFILE_ID = "flux2-klein-4b"
DEFAULT_IMAGE_EDIT_PROFILE_ID = QWEN_IMAGE_EDIT_PROFILE_ID


class ParameterKind(str, Enum):
    TEXT = "text"
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    CHOICE = "choice"


@dataclass(frozen=True)
class ParameterChoice:
    value: str
    label: str


@dataclass(frozen=True)
class ImageEditParameter:
    """One visible parameter accepted by a profile/provider contract."""

    stable_id: str
    label: str
    kind: ParameterKind
    default: Any
    group: str
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    decimals: int = 0
    choices: tuple[ParameterChoice, ...] = ()
    placeholder: str = ""
    description: str = ""
    load_time: bool = False

    def normalize(self, value: Any) -> Any:
        if self.kind == ParameterKind.BOOLEAN:
            return bool(value)
        if self.kind == ParameterKind.INTEGER:
            result = int(value)
            if self.minimum is not None:
                result = max(int(self.minimum), result)
            if self.maximum is not None:
                result = min(int(self.maximum), result)
            return result
        if self.kind == ParameterKind.FLOAT:
            result = float(value)
            if self.minimum is not None:
                result = max(float(self.minimum), result)
            if self.maximum is not None:
                result = min(float(self.maximum), result)
            return result
        if self.kind == ParameterKind.CHOICE:
            result = str(value)
            allowed = {choice.value for choice in self.choices}
            return result if result in allowed else str(self.default)
        return str(value)


@dataclass(frozen=True)
class ImageEditProfile:
    stable_id: str
    title: str
    provider: str
    model_id: str
    description: str
    parameters: tuple[ImageEditParameter, ...]
    primary: bool = False
    fast: bool = False

    def parameter(self, stable_id: str) -> ImageEditParameter:
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


def _choice(value: str, label: str | None = None) -> ParameterChoice:
    return ParameterChoice(value=value, label=label or value)


def _runtime_parameters(
    default_model: str,
    *,
    cpu_offload: bool = False,
) -> tuple[ImageEditParameter, ...]:
    return (
        ImageEditParameter(
            "model", "Model / local directory", ParameterKind.STRING,
            default_model, "Model", placeholder="Hugging Face ID or directory",
            load_time=True,
        ),
        ImageEditParameter(
            "revision", "Revision", ParameterKind.STRING, "", "Model",
            placeholder="branch, tag or commit (empty = default)",
            load_time=True,
        ),
        ImageEditParameter(
            "local_files_only", "Local files only", ParameterKind.BOOLEAN,
            False, "Model",
            load_time=True,
        ),
        ImageEditParameter(
            "dtype", "Dtype", ParameterKind.CHOICE, "bfloat16", "Runtime",
            choices=(
                _choice("bfloat16"), _choice("float16"),
                _choice("float32"),
            ),
            load_time=True,
        ),
        ImageEditParameter(
            "device", "Device", ParameterKind.CHOICE, "cuda", "Runtime",
            choices=(_choice("cuda"), _choice("cpu")),
            load_time=True,
        ),
        ImageEditParameter(
            "cpu_offload", "Component CPU offload", ParameterKind.BOOLEAN,
            cpu_offload, "Runtime",
            load_time=True,
        ),
        ImageEditParameter(
            "vae_tiling", "VAE tiling", ParameterKind.BOOLEAN,
            False, "Runtime",
            load_time=True,
        ),
        ImageEditParameter(
            "attention_kwargs", "Attention kwargs (JSON)",
            ParameterKind.STRING, "", "Runtime", placeholder="{}",
        ),
    )


_QWEN_MODEL = os.environ.get(
    "DIFFUSION_EDITOR_QWEN_IMAGE_EDIT_MODEL",
    "Qwen/Qwen-Image-Edit-2511",
)
_FLUX_MODEL = os.environ.get(
    "DIFFUSION_EDITOR_FLUX2_KLEIN_MODEL",
    "black-forest-labs/FLUX.2-klein-4B",
)
_DEFAULT_QWEN_LORA = Path(
    "~/soft/ComfyUI/models/loras/"
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
).expanduser()
_DEFAULT_QWEN_TRANSFORMER = Path(
    "~/soft/ComfyUI/models/diffusion_models/"
    "qwen_image_edit_2511_fp8mixed.safetensors"
).expanduser()
_DEFAULT_QWEN_TEXT_ENCODER = Path(
    "~/soft/ComfyUI/models/text_encoders/"
    "qwen_2.5_vl_7b_fp8_scaled.safetensors"
).expanduser()
_QWEN_LORA = os.environ.get(
    "DIFFUSION_EDITOR_QWEN_IMAGE_EDIT_LORA",
    str(_DEFAULT_QWEN_LORA) if _DEFAULT_QWEN_LORA.is_file() else "",
)
_QWEN_TRANSFORMER = os.environ.get(
    "DIFFUSION_EDITOR_QWEN_IMAGE_EDIT_TRANSFORMER",
    str(_DEFAULT_QWEN_TRANSFORMER)
    if _DEFAULT_QWEN_TRANSFORMER.is_file() else "",
)
_QWEN_TEXT_ENCODER = os.environ.get(
    "DIFFUSION_EDITOR_QWEN_IMAGE_EDIT_TEXT_ENCODER",
    str(_DEFAULT_QWEN_TEXT_ENCODER)
    if _DEFAULT_QWEN_TEXT_ENCODER.is_file() else "",
)
_QWEN_LOCAL_FP8 = bool(_QWEN_TRANSFORMER and _QWEN_TEXT_ENCODER)


_PROFILES = (
    ImageEditProfile(
        stable_id=QWEN_IMAGE_EDIT_PROFILE_ID,
        title="Qwen Image Edit 2511",
        provider="diffusers.qwen_image_edit_plus",
        model_id=_QWEN_MODEL,
        description="Primary profile; best preservation of source geometry.",
        primary=True,
        parameters=(
            ImageEditParameter(
                "prompt", "Prompt", ParameterKind.TEXT, "", "Conditioning",
                placeholder="Describe the requested edit",
            ),
            ImageEditParameter(
                "negative_prompt", "Negative prompt", ParameterKind.TEXT,
                "", "Conditioning",
            ),
            ImageEditParameter(
                "true_cfg_scale", "True CFG scale", ParameterKind.FLOAT,
                1.0, "Sampling", 0.0, 20.0, 0.1, 2,
            ),
            ImageEditParameter(
                "guidance_scale", "Guidance scale", ParameterKind.FLOAT,
                1.0, "Sampling", 0.0, 20.0, 0.1, 2,
            ),
            ImageEditParameter(
                "steps", "Steps", ParameterKind.INTEGER,
                4 if _QWEN_LORA else 40, "Sampling",
                1, 100, 1,
            ),
            ImageEditParameter(
                "seed", "Seed", ParameterKind.INTEGER, -1, "Sampling",
                -1, 2**32 - 1, 1,
            ),
            ImageEditParameter(
                "width", "Width (0 = source)", ParameterKind.INTEGER,
                0, "Output", 0, 4096, 1,
            ),
            ImageEditParameter(
                "height", "Height (0 = source)", ParameterKind.INTEGER,
                0, "Output", 0, 4096, 1,
            ),
            ImageEditParameter(
                "sigmas", "Sigmas (comma-separated)", ParameterKind.STRING,
                "", "Sampling",
            ),
            ImageEditParameter(
                "max_sequence_length", "Max sequence length",
                ParameterKind.INTEGER, 512, "Conditioning", 1, 1024, 1,
            ),
            ImageEditParameter(
                "lora_path", "LoRA path", ParameterKind.STRING,
                _QWEN_LORA, "Model",
                load_time=True,
            ),
            ImageEditParameter(
                "lora_scale", "LoRA scale", ParameterKind.FLOAT,
                1.0, "Model", -4.0, 4.0, 0.05, 2,
                load_time=True,
            ),
            ImageEditParameter(
                "transformer_checkpoint", "Transformer checkpoint",
                ParameterKind.STRING, _QWEN_TRANSFORMER, "Model",
                placeholder="Empty = transformer from Model",
                description="Standalone scaled-FP8 safetensors override.",
                load_time=True,
            ),
            ImageEditParameter(
                "text_encoder_checkpoint", "Text encoder checkpoint",
                ParameterKind.STRING, _QWEN_TEXT_ENCODER, "Model",
                placeholder="Empty = text encoder from Model",
                description="Standalone scaled-FP8 safetensors override.",
                load_time=True,
            ),
        ) + _runtime_parameters(
            _QWEN_MODEL, cpu_offload=_QWEN_LOCAL_FP8),
    ),
    ImageEditProfile(
        stable_id=FLUX2_KLEIN_PROFILE_ID,
        title="FLUX.2 Klein 4B",
        provider="diffusers.flux2",
        model_id=_FLUX_MODEL,
        description="Fast profile; more creative and less locality-preserving.",
        fast=True,
        parameters=(
            ImageEditParameter(
                "prompt", "Prompt", ParameterKind.TEXT, "", "Conditioning",
                placeholder="Describe the requested edit",
            ),
            ImageEditParameter(
                "guidance_scale", "Guidance / CFG", ParameterKind.FLOAT,
                1.0, "Sampling", 0.0, 20.0, 0.1, 2,
            ),
            ImageEditParameter(
                "steps", "Steps", ParameterKind.INTEGER, 4, "Sampling",
                1, 100, 1,
            ),
            ImageEditParameter(
                "seed", "Seed", ParameterKind.INTEGER, -1, "Sampling",
                -1, 2**32 - 1, 1,
            ),
            ImageEditParameter(
                "width", "Width (0 = source)", ParameterKind.INTEGER,
                0, "Output", 0, 4096, 1,
            ),
            ImageEditParameter(
                "height", "Height (0 = source)", ParameterKind.INTEGER,
                0, "Output", 0, 4096, 1,
            ),
            ImageEditParameter(
                "sigmas", "Sigmas (comma-separated)", ParameterKind.STRING,
                "", "Sampling",
            ),
            ImageEditParameter(
                "max_sequence_length", "Max sequence length",
                ParameterKind.INTEGER, 512, "Conditioning", 1, 1024, 1,
            ),
            ImageEditParameter(
                "text_encoder_out_layers",
                "Text encoder output layers", ParameterKind.STRING,
                "9,18,27", "Conditioning",
            ),
        ) + _runtime_parameters(_FLUX_MODEL),
    ),
    ImageEditProfile(
        stable_id=LEGACY_INSTRUCT_PROFILE_ID,
        title="InstructPix2Pix (legacy)",
        provider="diffusers.instruct_pix2pix",
        model_id="timbrooks/instruct-pix2pix",
        description="Existing compatibility profile; retained unchanged.",
        parameters=(
            ImageEditParameter(
                "prompt", "Instruction", ParameterKind.TEXT, "",
                "Conditioning", placeholder="Describe the requested edit",
            ),
            ImageEditParameter(
                "image_guidance_scale", "Image guidance",
                ParameterKind.FLOAT, 1.5, "Sampling", 1.0, 3.0, 0.1, 1,
            ),
            ImageEditParameter(
                "guidance_scale", "CFG scale", ParameterKind.FLOAT,
                7.0, "Sampling", 1.0, 20.0, 0.1, 1,
            ),
            ImageEditParameter(
                "steps", "Steps", ParameterKind.INTEGER, 20, "Sampling",
                1, 100, 1,
            ),
            ImageEditParameter(
                "seed", "Seed", ParameterKind.INTEGER, -1, "Sampling",
                -1, 2**32 - 1, 1,
            ),
        ) + _runtime_parameters("timbrooks/instruct-pix2pix"),
    ),
)

_PROFILE_BY_ID = {profile.stable_id: profile for profile in _PROFILES}


def image_edit_profiles() -> tuple[ImageEditProfile, ...]:
    return _PROFILES


def image_edit_profile(stable_id: str) -> ImageEditProfile:
    try:
        return _PROFILE_BY_ID[stable_id]
    except KeyError as exc:
        raise ValueError(f"Unknown image edit profile: {stable_id}") from exc


def all_image_edit_parameters() -> tuple[ImageEditParameter, ...]:
    """Return the schema union in deterministic first-seen order."""
    result: list[ImageEditParameter] = []
    seen: set[str] = set()
    for profile in _PROFILES:
        for parameter in profile.parameters:
            if parameter.stable_id not in seen:
                result.append(parameter)
                seen.add(parameter.stable_id)
    return tuple(result)


def normalize_profile_store(
    values: dict[str, dict[str, Any]] | None,
) -> dict[str, dict[str, Any]]:
    source = values or {}
    result: dict[str, dict[str, Any]] = {}
    for profile in _PROFILES:
        stored = source.get(profile.stable_id)
        if isinstance(stored, dict):
            stored = dict(stored)
            if (
                profile.stable_id == QWEN_IMAGE_EDIT_PROFILE_ID
                and "transformer_checkpoint" not in stored
                and "text_encoder_checkpoint" not in stored
            ):
                # Profiles saved before standalone FP8 support contain the old
                # resident-BF16 offload default.  Adopt the new component
                # defaults once; documents saved afterwards retain an explicit
                # user choice for every visible parameter.
                stored["cpu_offload"] = profile.parameter(
                    "cpu_offload").default
        result[profile.stable_id] = profile.normalize(stored)
    return result


def parse_float_list(value: Any) -> list[float] | None:
    text = str(value).strip()
    if not text:
        return None
    return [float(item.strip()) for item in text.split(",") if item.strip()]


def parse_int_tuple(value: Any) -> tuple[int, ...]:
    return tuple(
        int(item.strip())
        for item in str(value).split(",")
        if item.strip()
    )


def profile_parameter_ids(profile_id: str) -> tuple[str, ...]:
    return tuple(
        parameter.stable_id
        for parameter in image_edit_profile(profile_id).parameters
    )


def iter_profile_groups(profile_id: str) -> Iterable[str]:
    seen: set[str] = set()
    for parameter in image_edit_profile(profile_id).parameters:
        if parameter.group not in seen:
            seen.add(parameter.group)
            yield parameter.group
