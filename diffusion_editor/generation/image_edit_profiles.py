"""Provider-neutral profiles and parameter schema for image editing models."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable


LEGACY_INSTRUCT_PROFILE_ID = "instruct-pix2pix"
QWEN_IMAGE_EDIT_PROFILE_ID = "qwen-image-edit-2511"
FLUX2_KLEIN_PROFILE_ID = "flux2-klein-4b"
SENSENOVA_U15_PROFILE_ID = "sensenova-u1.5-8b-mot-preview"
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
class ImageEditLoraAdapter:
    """One ordered LoRA adapter in an image-edit load configuration."""

    stable_id: str
    label: str
    source: str
    weight: float = 1.0
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "stable_id": self.stable_id,
            "label": self.label,
            "source": self.source,
            "weight": self.weight,
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class ImageEditProfile:
    stable_id: str
    title: str
    provider: str
    model_id: str
    description: str
    parameters: tuple[ImageEditParameter, ...]
    default_lora_adapters: tuple[ImageEditLoraAdapter, ...] = ()
    primary: bool = False
    fast: bool = False
    max_input_images: int = 1

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

    def normalize_lora_adapters(
        self,
        values: Iterable[ImageEditLoraAdapter | dict[str, Any]] | None,
    ) -> tuple[ImageEditLoraAdapter, ...]:
        source = self.default_lora_adapters if values is None else values
        return normalize_lora_adapters(source)


def _choice(value: str, label: str | None = None) -> ParameterChoice:
    return ParameterChoice(value=value, label=label or value)


def normalize_lora_adapters(
    values: Iterable[ImageEditLoraAdapter | dict[str, Any]] | None,
) -> tuple[ImageEditLoraAdapter, ...]:
    if values is None:
        return ()
    result: list[ImageEditLoraAdapter] = []
    used_ids: set[str] = set()
    for index, raw in enumerate(values):
        if isinstance(raw, ImageEditLoraAdapter):
            data = raw.to_dict()
        elif isinstance(raw, dict):
            data = raw
        else:
            continue
        source = str(data.get("source", "")).strip()
        fallback_id = f"lora-{index + 1}"
        stable_id = re.sub(
            r"[^a-zA-Z0-9_-]+", "-",
            str(data.get("stable_id", fallback_id)).strip(),
        ).strip("-") or fallback_id
        candidate = stable_id
        suffix = 2
        while candidate in used_ids:
            candidate = f"{stable_id}-{suffix}"
            suffix += 1
        stable_id = candidate
        used_ids.add(stable_id)
        label = str(data.get("label", ""))
        if not label.strip():
            label = Path(source).stem if source else f"LoRA {index + 1}"
        try:
            weight = float(data.get("weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        if not math.isfinite(weight):
            weight = 1.0
        weight = max(-4.0, min(weight, 4.0))
        result.append(ImageEditLoraAdapter(
            stable_id=stable_id,
            label=label,
            source=source,
            weight=weight,
            enabled=bool(data.get("enabled", True)),
        ))
    return tuple(result)


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
_DEFAULT_SENSENOVA_MODEL = Path(
    "~/soft/ComfyUI/models/sensenova/"
    "SenseNova-U1.5-8B-MoT-Preview"
).expanduser()
_DEFAULT_SENSENOVA_GGUF = Path(
    "~/soft/ComfyUI/models/gguf/"
    "SenseNova-U1.5-8B-MoT-Preview-Q8.gguf"
).expanduser()
_SENSENOVA_MODEL = os.environ.get(
    "DIFFUSION_EDITOR_SENSENOVA_MODEL",
    str(_DEFAULT_SENSENOVA_MODEL)
    if _DEFAULT_SENSENOVA_MODEL.is_dir()
    else "sensenova/SenseNova-U1.5-8B-MoT-Preview",
)
_SENSENOVA_GGUF = os.environ.get(
    "DIFFUSION_EDITOR_SENSENOVA_GGUF",
    str(_DEFAULT_SENSENOVA_GGUF)
    if _DEFAULT_SENSENOVA_GGUF.is_file() else "",
)
_DEFAULT_QWEN_LORA = Path(
    "~/soft/ComfyUI/models/loras/"
    "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
).expanduser()
_DEFAULT_QWEN_MULTIPLE_ANGLES_LORA = Path(
    "~/soft/ComfyUI/models/loras/"
    "qwen-image-edit-2511-multiple-angles-lora.safetensors"
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
_QWEN_MULTIPLE_ANGLES_LORA = os.environ.get(
    "DIFFUSION_EDITOR_QWEN_MULTIPLE_ANGLES_LORA",
    str(_DEFAULT_QWEN_MULTIPLE_ANGLES_LORA)
    if _DEFAULT_QWEN_MULTIPLE_ANGLES_LORA.is_file() else "",
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


def qwen_multiple_angles_lora_adapter(
    *,
    weight: float = 1.0,
    enabled: bool = True,
) -> ImageEditLoraAdapter:
    """Build the optional Multiple Angles adapter for the primary Qwen profile."""

    return ImageEditLoraAdapter(
        "multiple-angles",
        "Multiple Angles",
        _QWEN_MULTIPLE_ANGLES_LORA,
        weight,
        enabled and bool(_QWEN_MULTIPLE_ANGLES_LORA),
    )


def _qwen_parameters(
    *,
    prompt_placeholder: str = "Describe the requested edit",
) -> tuple[ImageEditParameter, ...]:
    parameters = (
        ImageEditParameter(
            "prompt", "Prompt", ParameterKind.TEXT, "", "Conditioning",
            placeholder=prompt_placeholder,
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
            4 if _QWEN_LORA else 40, "Sampling", 1, 100, 1,
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
    )
    return parameters + (
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
        _QWEN_MODEL, cpu_offload=_QWEN_LOCAL_FP8)


_PROFILES = (
    ImageEditProfile(
        stable_id=QWEN_IMAGE_EDIT_PROFILE_ID,
        title="Qwen Image Edit 2511",
        provider="diffusers.qwen_image_edit_plus",
        model_id=_QWEN_MODEL,
        description="Primary profile; best preservation of source geometry.",
        primary=True,
        max_input_images=2,
        parameters=_qwen_parameters(),
        default_lora_adapters=(
            ImageEditLoraAdapter(
                "lightning",
                "Lightning 4-step",
                _QWEN_LORA,
                1.0,
                bool(_QWEN_LORA),
            ),
        ),
    ),
    ImageEditProfile(
        stable_id=FLUX2_KLEIN_PROFILE_ID,
        title="FLUX.2 Klein 4B",
        provider="diffusers.flux2",
        model_id=_FLUX_MODEL,
        description="Fast profile; more creative and less locality-preserving.",
        fast=True,
        max_input_images=2,
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
        stable_id=SENSENOVA_U15_PROFILE_ID,
        title="SenseNova U1.5 8B MoT (Q8)",
        provider="sensenova_u1.it2i",
        model_id=_SENSENOVA_MODEL,
        description=(
            "Unified multimodal GGUF editor with strong instruction "
            "following and material rendering."
        ),
        max_input_images=2,
        parameters=(
            ImageEditParameter(
                "prompt", "Prompt", ParameterKind.TEXT, "", "Conditioning",
                placeholder="Describe the requested edit",
            ),
            ImageEditParameter(
                "cfg_scale", "CFG scale", ParameterKind.FLOAT,
                4.0, "Sampling", 0.0, 20.0, 0.1, 2,
            ),
            ImageEditParameter(
                "img_cfg_scale", "Image CFG scale", ParameterKind.FLOAT,
                1.0, "Sampling", 0.0, 20.0, 0.1, 2,
            ),
            ImageEditParameter(
                "cfg_norm", "CFG normalization", ParameterKind.CHOICE,
                "none", "Sampling",
                choices=(
                    _choice("none"), _choice("global"), _choice("channel"),
                ),
            ),
            ImageEditParameter(
                "timestep_shift", "Timestep shift", ParameterKind.FLOAT,
                3.0, "Sampling", 0.0, 20.0, 0.1, 2,
            ),
            ImageEditParameter(
                "cfg_interval_start", "CFG interval start",
                ParameterKind.FLOAT, 0.0, "Sampling", 0.0, 1.0, 0.05, 2,
            ),
            ImageEditParameter(
                "cfg_interval_end", "CFG interval end",
                ParameterKind.FLOAT, 1.0, "Sampling", 0.0, 1.0, 0.05, 2,
            ),
            ImageEditParameter(
                "steps", "Steps", ParameterKind.INTEGER,
                8, "Sampling", 1, 100, 1,
            ),
            ImageEditParameter(
                "seed", "Seed", ParameterKind.INTEGER, -1, "Sampling",
                -1, 2**32 - 1, 1,
            ),
            ImageEditParameter(
                "width", "Width (0 = auto)", ParameterKind.INTEGER,
                0, "Output", 0, 8192, 32,
            ),
            ImageEditParameter(
                "height", "Height (0 = auto)", ParameterKind.INTEGER,
                0, "Output", 0, 8192, 32,
            ),
            ImageEditParameter(
                "target_megapixels", "Auto-size megapixels",
                ParameterKind.FLOAT, 1.0, "Output", 0.25, 32.0, 0.25, 2,
            ),
            ImageEditParameter(
                "model", "Config / tokenizer directory",
                ParameterKind.STRING, _SENSENOVA_MODEL, "Model",
                placeholder="Hugging Face ID or local directory",
                load_time=True,
            ),
            ImageEditParameter(
                "gguf_checkpoint", "GGUF checkpoint",
                ParameterKind.STRING, _SENSENOVA_GGUF, "Model",
                placeholder="Absolute path to SenseNova GGUF weights",
                load_time=True,
            ),
            ImageEditParameter(
                "dtype", "Compute dtype", ParameterKind.CHOICE,
                "bfloat16", "Runtime",
                choices=(
                    _choice("bfloat16"), _choice("float16"),
                    _choice("float32"),
                ),
                load_time=True,
            ),
            ImageEditParameter(
                "device", "Device", ParameterKind.CHOICE,
                "cuda", "Runtime",
                choices=(_choice("cuda"), _choice("cpu")),
                load_time=True,
            ),
            ImageEditParameter(
                "attention_backend", "Attention backend",
                ParameterKind.CHOICE, "auto", "Runtime",
                choices=(
                    _choice("auto"), _choice("sdpa"), _choice("flash"),
                ),
                load_time=True,
            ),
            ImageEditParameter(
                "vram_mode", "VRAM mode", ParameterKind.CHOICE,
                "full", "Runtime",
                choices=tuple(_choice(value) for value in (
                    "full", "fast", "balanced", "low")),
                load_time=True,
            ),
            ImageEditParameter(
                "fast_vram_fraction", "Fast VRAM fraction",
                ParameterKind.FLOAT, 0.9, "Runtime", 0.1, 1.0, 0.05, 2,
                load_time=True,
            ),
            ImageEditParameter(
                "fast_vram_headroom_gib", "Fast VRAM headroom (GiB)",
                ParameterKind.FLOAT, 2.0, "Runtime", 0.0, 64.0, 0.5, 1,
                load_time=True,
            ),
            ImageEditParameter(
                "fast_activation_reserve_gib",
                "Fast activation reserve (GiB)",
                ParameterKind.FLOAT, 4.0, "Runtime", 0.0, 64.0, 0.5, 1,
                load_time=True,
            ),
            ImageEditParameter(
                "fast_vram_budget_gib", "Fast VRAM budget (GiB, 0 = auto)",
                ParameterKind.FLOAT, 0.0, "Runtime", 0.0, 128.0, 0.5, 1,
                load_time=True,
            ),
        ),
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


def normalize_profile_lora_store(
    values: dict[str, list[dict[str, Any]]] | None,
    *,
    legacy_profile_parameters: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Normalize per-profile adapter stacks and migrate flat Qwen fields."""

    source = values if isinstance(values, dict) else {}
    legacy = (
        legacy_profile_parameters
        if isinstance(legacy_profile_parameters, dict) else {}
    )
    result: dict[str, list[dict[str, Any]]] = {}
    for profile in _PROFILES:
        stored = source.get(profile.stable_id)
        if isinstance(stored, (list, tuple)):
            adapters = profile.normalize_lora_adapters(stored)
        else:
            old_parameters = legacy.get(profile.stable_id)
            migrated: list[dict[str, Any]] | None = None
            if isinstance(old_parameters, dict) and (
                    "lora_path" in old_parameters
                    or "angle_lora_path" in old_parameters):
                migrated = []
                if "lora_path" in old_parameters:
                    path = str(old_parameters.get("lora_path", ""))
                    migrated.append({
                        "stable_id": "lightning",
                        "label": "Lightning 4-step",
                        "source": path,
                        "weight": old_parameters.get("lora_scale", 1.0),
                        "enabled": bool(path.strip()),
                    })
                if "angle_lora_path" in old_parameters:
                    path = str(old_parameters.get("angle_lora_path", ""))
                    migrated.append({
                        "stable_id": "multiple-angles",
                        "label": "Multiple Angles",
                        "source": path,
                        "weight": old_parameters.get(
                            "angle_lora_scale", 1.0),
                        "enabled": bool(path.strip()),
                    })
            adapters = profile.normalize_lora_adapters(migrated)
        result[profile.stable_id] = [
            adapter.to_dict() for adapter in adapters]
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
