"""Typed data passed between editor workflows and generation engines."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
from PIL import Image

from .provenance import GenerationProvenance

Rect = tuple[int, int, int, int]


class ReconstructionStage(str, Enum):
    SOURCE_IMAGE = "source_image"
    SPARSE_OCCUPANCY = "sparse_occupancy"
    LR_SHAPE_FLOW = "lr_shape_flow"
    LR_SHAPE_LATENT = "lr_shape_latent"
    HR_COORDINATES = "hr_coordinates"
    HR_SHAPE_FLOW = "hr_shape_flow"
    HR_SHAPE_LATENT = "hr_shape_latent"
    TEXTURE_FLOW = "texture_flow"
    TEXTURE_LATENT = "texture_latent"
    FINAL_MESH = "final_mesh"


RECONSTRUCTION_STAGES = tuple(ReconstructionStage)

# Only stages that produce a distinct, human-readable artifact belong in the
# editor's preview list. Flow integration and the texture latent remain part of
# the worker protocol, but exposing them as previews would imply an artifact
# that does not exist.
RECONSTRUCTION_PREVIEW_STAGES = (
    ReconstructionStage.SOURCE_IMAGE,
    ReconstructionStage.SPARSE_OCCUPANCY,
    ReconstructionStage.LR_SHAPE_LATENT,
    ReconstructionStage.HR_COORDINATES,
    ReconstructionStage.HR_SHAPE_LATENT,
    ReconstructionStage.FINAL_MESH,
)

RECONSTRUCTION_STAGE_LABELS = {
    ReconstructionStage.SOURCE_IMAGE: "Source image",
    ReconstructionStage.SPARSE_OCCUPANCY: "Sparse occupancy",
    ReconstructionStage.LR_SHAPE_FLOW: "LR shape flow",
    ReconstructionStage.LR_SHAPE_LATENT: "LR shape latent",
    ReconstructionStage.HR_COORDINATES: "LR -> HR coordinates",
    ReconstructionStage.HR_SHAPE_FLOW: "HR shape flow",
    ReconstructionStage.HR_SHAPE_LATENT: "HR shape latent",
    ReconstructionStage.TEXTURE_FLOW: "Texture flow",
    ReconstructionStage.TEXTURE_LATENT: "Texture latent",
    ReconstructionStage.FINAL_MESH: "Final mesh",
}


class ReconstructionStageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    READY = "ready"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass(frozen=True)
class ReconstructionParameters:
    seed: int = 42
    steps: int = 12
    resolution: int = 1024
    manual_fov_degrees: float = 0.0
    decimation_target: int = 200_000
    texture_size: int = 2048
    low_vram: bool = True

    def __post_init__(self) -> None:
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("reconstruction seed must be in [0, 2147483647]")
        if not 1 <= self.steps <= 50:
            raise ValueError("reconstruction steps must be in [1, 50]")
        if self.resolution not in (1024, 1280, 1536):
            raise ValueError("reconstruction resolution must be 1024, 1280, or 1536")
        if not 0.0 <= self.manual_fov_degrees <= 120.0:
            raise ValueError("manual camera FOV must be in [0, 120] degrees")
        if not 50_000 <= self.decimation_target <= 1_000_000:
            raise ValueError("mesh face target must be in [50000, 1000000]")
        if self.texture_size not in (1024, 2048, 4096):
            raise ValueError("texture size must be 1024, 2048, or 4096")

    def to_dict(self) -> dict:
        return {
            "seed": self.seed,
            "steps": self.steps,
            "resolution": self.resolution,
            "manual_fov_degrees": self.manual_fov_degrees,
            "decimation_target": self.decimation_target,
            "texture_size": self.texture_size,
            "low_vram": self.low_vram,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "ReconstructionParameters":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            seed=int(payload.get("seed", 42)),
            steps=int(payload.get("steps", 12)),
            resolution=int(payload.get("resolution", 1024)),
            manual_fov_degrees=float(payload.get("manual_fov_degrees", 0.0)),
            decimation_target=int(payload.get("decimation_target", 200_000)),
            texture_size=int(payload.get("texture_size", 2048)),
            low_vram=bool(payload.get("low_vram", True)),
        )


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
    task_type: Literal[
        "load", "load_ip_adapter", "inference", "segmentation", "reconstruction"
    ]
    result: object | None = None
    error: str | None = None
    meta: object | None = None
    job_id: str | None = None


@dataclass(frozen=True)
class DiffusionInferenceResult:
    image: Image.Image
    seed: int
    provenance: GenerationProvenance | None = None


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
    provenance: GenerationProvenance | None = None


@dataclass(frozen=True)
class LamaRequest:
    image: Image.Image
    mask_image: Image.Image


@dataclass(frozen=True)
class LamaResult:
    image: Image.Image
    provenance: GenerationProvenance | None = None


@dataclass(frozen=True)
class ReconstructionRequest:
    image: Image.Image
    parameters: ReconstructionParameters = ReconstructionParameters()
    target_stage: ReconstructionStage = ReconstructionStage.FINAL_MESH


@dataclass(frozen=True)
class ReconstructionStageArtifact:
    stage: ReconstructionStage
    path: str
    preview_kind: Literal["image", "points", "mesh", "latent"]


@dataclass(frozen=True)
class ReconstructionStageEvent:
    stage: ReconstructionStage
    status: ReconstructionStageStatus
    progress: int = 0
    total: int = 0
    artifact: ReconstructionStageArtifact | None = None


@dataclass(frozen=True)
class ReconstructionResult:
    glb_path: str
    source_path: str
    completed_stage: ReconstructionStage = ReconstructionStage.FINAL_MESH
    artifacts: tuple[ReconstructionStageArtifact, ...] = ()


@dataclass(frozen=True)
class SegmentationRequest:
    image: np.ndarray
    invert: bool = True


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    provenance: GenerationProvenance | None = None
