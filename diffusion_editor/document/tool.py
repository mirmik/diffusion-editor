"""Tool — attached auto-drawer with persistent settings for a Layer.

Tools hold AI-generation config (prompt, steps, seed, etc.).
Mask and Patch are owned by Layer, not by Tool — tools access them via
the layer reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from uuid import uuid4

from PIL import Image

from ..generation.provenance import (
    GenerationProvenance,
    ModelIdentity,
    ModelIdentityPolicy,
)
from ..generation.image_edit_profiles import (
    LEGACY_INSTRUCT_PROFILE_ID,
    image_edit_profile,
    normalize_profile_lora_store,
    normalize_profile_store,
)
from ..generation.text_to_image_profiles import (
    DEFAULT_TEXT_TO_IMAGE_PROFILE_ID,
    TEXT_TO_IMAGE_PROFILE_SCHEMA_VERSION,
    normalize_text_to_image_lora_store,
    normalize_text_to_image_profile_store,
    text_to_image_profile,
)


class Tool:
    """Abstract auto-drawer attached to a Layer."""

    tool_type: str = ""
    generation_provenance: GenerationProvenance | None = None


class DiffusionTool(Tool):
    tool_type = "diffusion"

    def __init__(self,
                 source_patch: Image.Image | None,
                 patch_x: int, patch_y: int, patch_w: int, patch_h: int,
                 prompt: str, negative_prompt: str,
                 strength: float, guidance_scale: float, steps: int,
                 seed: int,
                 model_path: str = "", prediction_type: str = "",
                 mode: str = "inpaint"):
        self.mode = mode
        self.source_patch = source_patch
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.seed = seed
        self.model_path = model_path
        self.prediction_type = prediction_type
        self.ip_adapter_layer_id: str | None = None
        self.ip_adapter_layer_name_hint: str = ""
        self.ip_adapter_scale: float = 0.6
        self.masked_content: str = "original"
        self.resize_to_model_resolution: bool = False
        self.model_identity: ModelIdentity | None = None
        self.model_identity_policy = ModelIdentityPolicy.WARN
        self.generation_provenance = None


@dataclass(frozen=True)
class TextToImageReference:
    stable_id: str = field(default_factory=lambda: uuid4().hex)
    layer_id: str | None = None
    layer_name_hint: str = ""
    image: Image.Image | None = None
    image_name_hint: str = ""

    def copy(self):
        return replace(self, image=self.image.copy() if self.image is not None else None)

    @property
    def label(self) -> str:
        if self.layer_id is not None:
            return f"Layer: {self.layer_name_hint or self.layer_id}"
        return f"External: {self.image_name_hint or 'image'}"


def _first_reference_property(name, default):
    """Compatibility access to the first entry for older integrations."""
    def get(tool):
        return getattr(tool.references[0], name) if tool.references else default

    def set_value(tool, value):
        refs = list(tool.references)
        first = refs[0] if refs else TextToImageReference()
        first = replace(first, **{name: value})
        if first.layer_id is not None or first.image is not None:
            tool.references = (first, *refs[1:])
        elif refs:
            tool.references = tuple(refs[1:])
    return property(get, set_value)


class TextToImageTool(Tool):
    """Persistent settings for generation that has no source image."""

    tool_type = "text_to_image"
    reference_layer_id = _first_reference_property("layer_id", None)
    reference_layer_name_hint = _first_reference_property("layer_name_hint", "")
    reference_image = _first_reference_property("image", None)
    reference_image_name_hint = _first_reference_property("image_name_hint", "")

    def __init__(
            self,
            model_profile_id: str = DEFAULT_TEXT_TO_IMAGE_PROFILE_ID,
            profile_parameters: dict[str, dict] | None = None,
            profile_lora_adapters: dict[str, list[dict]] | None = None,
            *,
            profile_schema_version: int = TEXT_TO_IMAGE_PROFILE_SCHEMA_VERSION,
            reference_layer_id: str | None = None,
            reference_layer_name_hint: str = "",
            reference_image: Image.Image | None = None,
            reference_image_name_hint: str = "",
            references: tuple[TextToImageReference, ...] | None = None):
        if references is None:
            references = ((TextToImageReference(
                layer_id=reference_layer_id, layer_name_hint=reference_layer_name_hint,
                image=reference_image, image_name_hint=reference_image_name_hint),)
                if reference_layer_id is not None or reference_image is not None else ())
        self.set_references(references)
        # Schema v2 introduced Lightning defaults.  Later migrations must not
        # re-apply that one-time policy to already-v2 documents.
        adopt_lightning_defaults = int(profile_schema_version) < 2
        self.profile_parameters = normalize_text_to_image_profile_store(
            profile_parameters,
            adopt_lightning_defaults=adopt_lightning_defaults)
        self.profile_lora_adapters = normalize_text_to_image_lora_store(
            profile_lora_adapters,
            adopt_lightning_defaults=adopt_lightning_defaults)
        self.profile_schema_version = TEXT_TO_IMAGE_PROFILE_SCHEMA_VERSION
        self.model_profile_id = model_profile_id
        text_to_image_profile(model_profile_id)
        self.generation_provenance = None

    @staticmethod
    def validate_references(references):
        references = tuple(references)
        if len(references) > 10:
            raise ValueError("Text to Image supports at most 10 references")
        ids = set()
        for ref in references:
            if not isinstance(ref, TextToImageReference):
                raise ValueError("Invalid Text to Image reference")
            if not isinstance(ref.stable_id, str) or not ref.stable_id or ref.stable_id in ids:
                raise ValueError("Reference IDs must be non-empty and unique")
            ids.add(ref.stable_id)
            if (ref.layer_id is None) == (ref.image is None):
                raise ValueError("A reference needs exactly one layer or image")
            if ref.layer_id is not None and (not isinstance(ref.layer_id, str) or not ref.layer_id):
                raise ValueError("Reference layer ID must be a non-empty string")
            if not isinstance(ref.layer_name_hint, str) or not isinstance(ref.image_name_hint, str):
                raise ValueError("Reference names must be strings")
        return references

    def set_references(self, references) -> None:
        self.references = tuple(ref.copy() for ref in self.validate_references(references))

    @property
    def parameters(self) -> dict:
        return self.profile_parameters[self.model_profile_id]

    @property
    def lora_adapters(self):
        profile = text_to_image_profile(self.model_profile_id)
        return profile.normalize_lora_adapters(
            self.profile_lora_adapters[self.model_profile_id])

    def set_profile(self, profile_id: str) -> None:
        text_to_image_profile(profile_id)
        self.model_profile_id = profile_id

    def set_parameter(self, stable_id: str, value) -> None:
        parameter = text_to_image_profile(
            self.model_profile_id).parameter(stable_id)
        self.parameters[stable_id] = parameter.normalize(value)

    def set_lora_adapters(self, values) -> None:
        profile = text_to_image_profile(self.model_profile_id)
        self.profile_lora_adapters[self.model_profile_id] = [
            adapter.to_dict()
            for adapter in profile.normalize_lora_adapters(values)
        ]


class LamaTool(Tool):
    tool_type = "lama"

    def __init__(self,
                 source_patch: Image.Image | None,
                 patch_x: int, patch_y: int, patch_w: int, patch_h: int):
        self.source_patch = source_patch
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.generation_provenance = None


class InstructTool(Tool):
    tool_type = "instruct"

    def __init__(self,
                 source_patch: Image.Image | None,
                 patch_x: int, patch_y: int, patch_w: int, patch_h: int,
                 instruction: str = "",
                 image_guidance_scale: float = 1.5,
                 guidance_scale: float = 7.0,
                 steps: int = 20,
                 seed: int = -1,
                 model_profile_id: str = LEGACY_INSTRUCT_PROFILE_ID,
                 profile_parameters: dict[str, dict] | None = None,
                 profile_lora_adapters: dict[str, list[dict]] | None = None,
                 reference_layer_id: str | None = None,
                 reference_layer_name_hint: str = "",
                 reference_image: Image.Image | None = None,
                 reference_image_name_hint: str = ""):
        self.source_patch = source_patch
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.patch_w = patch_w
        self.patch_h = patch_h
        # Keep parameters for every profile so switching models is lossless.
        self.profile_parameters = normalize_profile_store(profile_parameters)
        self.profile_lora_adapters = normalize_profile_lora_store(
            profile_lora_adapters,
            legacy_profile_parameters=profile_parameters,
        )
        self.model_profile_id = model_profile_id
        self.reference_layer_id = reference_layer_id
        self.reference_layer_name_hint = reference_layer_name_hint
        self.reference_image = (
            reference_image.copy() if reference_image is not None else None)
        self.reference_image_name_hint = reference_image_name_hint
        image_edit_profile(model_profile_id)  # fail early on corrupt state
        if profile_parameters is None:
            legacy = self.profile_parameters[LEGACY_INSTRUCT_PROFILE_ID]
            legacy.update({
                "prompt": instruction,
                "image_guidance_scale": image_guidance_scale,
                "guidance_scale": guidance_scale,
                "steps": steps,
                "seed": seed,
            })
        self.generation_provenance = None

    @property
    def parameters(self) -> dict:
        return self.profile_parameters[self.model_profile_id]

    @property
    def lora_adapters(self):
        profile = image_edit_profile(self.model_profile_id)
        return profile.normalize_lora_adapters(
            self.profile_lora_adapters[self.model_profile_id])

    def set_profile(self, profile_id: str) -> None:
        image_edit_profile(profile_id)
        self.model_profile_id = profile_id

    def set_parameter(self, stable_id: str, value) -> None:
        parameter = image_edit_profile(self.model_profile_id).parameter(stable_id)
        self.parameters[stable_id] = parameter.normalize(value)

    def set_lora_adapters(self, values) -> None:
        profile = image_edit_profile(self.model_profile_id)
        self.profile_lora_adapters[self.model_profile_id] = [
            adapter.to_dict()
            for adapter in profile.normalize_lora_adapters(values)
        ]

    # Compatibility surface for old document commands and integrations.
    @property
    def instruction(self) -> str:
        return str(self.parameters.get("prompt", ""))

    @instruction.setter
    def instruction(self, value: str) -> None:
        self.parameters["prompt"] = str(value)

    @property
    def image_guidance_scale(self) -> float:
        return float(self.parameters.get("image_guidance_scale", 1.5))

    @image_guidance_scale.setter
    def image_guidance_scale(self, value: float) -> None:
        self.parameters["image_guidance_scale"] = float(value)

    @property
    def guidance_scale(self) -> float:
        return float(self.parameters.get("guidance_scale", 1.0))

    @guidance_scale.setter
    def guidance_scale(self, value: float) -> None:
        self.parameters["guidance_scale"] = float(value)

    @property
    def steps(self) -> int:
        return int(self.parameters.get("steps", 4))

    @steps.setter
    def steps(self, value: int) -> None:
        self.parameters["steps"] = int(value)

    @property
    def seed(self) -> int:
        return int(self.parameters.get("seed", -1))

    @seed.setter
    def seed(self, value: int) -> None:
        self.parameters["seed"] = int(value)


# New code may use the domain name while old project files and integrations keep
# the stable ``instruct`` tool type and class.
ImageEditTool = InstructTool
