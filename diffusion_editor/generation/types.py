"""Typed data passed between editor workflows and generation engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from PIL import Image

Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class GenerationError:
    message: str
    log_message: str | None = None


@dataclass(frozen=True)
class PatchSource:
    image: Image.Image
    canvas_rect: Rect
    source: Literal["patch", "mask_bbox", "existing"]

    @property
    def width(self) -> int:
        return self.canvas_rect[2] - self.canvas_rect[0]

    @property
    def height(self) -> int:
        return self.canvas_rect[3] - self.canvas_rect[1]


@dataclass(frozen=True)
class ReferenceImage:
    image: Image.Image
    layer_id: str
    layer_name: str
    local_rect: Rect
    source: Literal["patch", "alpha_bbox", "full_layer"]


@dataclass(frozen=True)
class ReferenceResolveResult:
    reference: ReferenceImage | None = None
    error: GenerationError | None = None


@dataclass(frozen=True)
class DiffusionRequest:
    image: Image.Image | None
    mask_image: Image.Image | None
    prompt: str
    negative_prompt: str
    strength: float
    steps: int
    guidance_scale: float
    seed: int
    mode: str
    masked_content: str
    ip_adapter_image: Image.Image | None
    ip_adapter_scale: float
    width: int
    height: int


@dataclass(frozen=True)
class DiffusionRequestBuildResult:
    request: DiffusionRequest | None = None
    error: GenerationError | None = None


@dataclass(frozen=True)
class EnginePollEvent:
    task_type: Literal["load", "load_ip_adapter", "inference", "segmentation"]
    result: object | None = None
    error: str | None = None
    meta: object | None = None


@dataclass(frozen=True)
class DiffusionInferenceResult:
    image: Image.Image
    seed: int


@dataclass(frozen=True)
class InstructRequest:
    image: Image.Image
    instruction: str
    guidance_scale: float
    image_guidance_scale: float
    steps: int
    seed: int


@dataclass(frozen=True)
class InstructInferenceResult:
    image: Image.Image
    seed: int


@dataclass(frozen=True)
class LamaRequest:
    image: Image.Image
    mask_image: Image.Image


@dataclass(frozen=True)
class LamaResult:
    image: Image.Image


@dataclass(frozen=True)
class SegmentationRequest:
    image: np.ndarray
    invert: bool = True


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
