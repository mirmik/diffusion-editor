"""Tool — attached auto-drawer with persistent settings for a Layer.

Tools hold AI-generation config (prompt, steps, seed, etc.).
Mask and Patch are owned by Layer, not by Tool — tools access them via
the layer reference.
"""

from __future__ import annotations

from PIL import Image


class Tool:
    """Abstract auto-drawer attached to a Layer."""

    tool_type: str = ""


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


class InstructTool(Tool):
    tool_type = "instruct"

    def __init__(self,
                 source_patch: Image.Image | None,
                 patch_x: int, patch_y: int, patch_w: int, patch_h: int,
                 instruction: str = "",
                 image_guidance_scale: float = 1.5,
                 guidance_scale: float = 7.0,
                 steps: int = 20,
                 seed: int = -1):
        self.source_patch = source_patch
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.instruction = instruction
        self.image_guidance_scale = image_guidance_scale
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.seed = seed
