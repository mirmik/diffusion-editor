"""Typed data for Grounding DINO and SAM 2.1 workflows."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GROUNDING_MODELS = [
    ("Tiny (fast, ~340 MB)", "IDEA-Research/grounding-dino-tiny"),
    ("Base (balanced, ~690 MB)", "IDEA-Research/grounding-dino-base"),
]

SAM2_MODELS = [
    ("Tiny (fast, ~82 MB)", "facebook/sam2.1-hiera-tiny"),
    ("Small (balanced, ~184 MB)", "facebook/sam2.1-hiera-small"),
    ("Base+ (accurate, ~322 MB)", "facebook/sam2.1-hiera-base-plus"),
]


@dataclass(frozen=True)
class GroundingParams:
    prompt: str
    model_id: str
    box_threshold: float
    text_threshold: float
    use_gpu: bool
    sam2_model_id: str | None
    sam2_mask_channel: int
    mask_threshold: float
    max_hole_area: int
    max_sprinkle_area: int
    multimask: bool
    non_overlap: bool


@dataclass(frozen=True)
class GroundingRequest:
    image: np.ndarray
    params: GroundingParams


@dataclass(frozen=True)
class GroundingDetection:
    label: str
    x0: int
    y0: int
    x1: int
    y1: int
    score: float
    mask: np.ndarray | None = None


@dataclass(frozen=True)
class GroundingResult:
    detections: tuple[GroundingDetection, ...]


@dataclass(frozen=True)
class GroundingEngineEvent:
    status: str | None = None
    result: GroundingResult | None = None
    error: str | None = None
