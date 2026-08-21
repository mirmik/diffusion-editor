"""Tool — attached auto-drawer with persistent settings for a Layer.

Tools hold AI-generation config (prompt, steps, seed, etc.).
Mask and Patch are owned by Layer, not by Tool — tools access them via
the layer reference.
"""

from __future__ import annotations

from PIL import Image

from ..generation.provenance import (
    GenerationProvenance,
    ModelIdentity,
    ModelIdentityPolicy,
)
from ..generation.image_edit_profiles import (
    LEGACY_INSTRUCT_PROFILE_ID,
    image_edit_profile,
    normalize_profile_store,
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
                 profile_parameters: dict[str, dict] | None = None):
        self.source_patch = source_patch
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.patch_w = patch_w
        self.patch_h = patch_h
        # Keep parameters for every profile so switching models is lossless.
        self.profile_parameters = normalize_profile_store(profile_parameters)
        self.model_profile_id = model_profile_id
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

    def set_profile(self, profile_id: str) -> None:
        image_edit_profile(profile_id)
        self.model_profile_id = profile_id

    def set_parameter(self, stable_id: str, value) -> None:
        parameter = image_edit_profile(self.model_profile_id).parameter(stable_id)
        self.parameters[stable_id] = parameter.normalize(value)

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
