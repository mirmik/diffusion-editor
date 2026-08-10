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
    NORMAL_MAP = "normal_map"
    POINT_CLOUD = "point_cloud"
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
    ReconstructionStage.POINT_CLOUD,
    ReconstructionStage.SPARSE_OCCUPANCY,
    ReconstructionStage.LR_SHAPE_LATENT,
    ReconstructionStage.HR_COORDINATES,
    ReconstructionStage.HR_SHAPE_LATENT,
    ReconstructionStage.FINAL_MESH,
)

RECONSTRUCTION_STAGE_LABELS = {
    ReconstructionStage.SOURCE_IMAGE: "Source image",
    ReconstructionStage.NORMAL_MAP: "Normal map",
    ReconstructionStage.POINT_CLOUD: "Point cloud",
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


class ReconstructionRunKind(str, Enum):
    BASE = "base"
    MASKED_REFINE = "masked_refine"
    MASKED_TEXTURE_REFINE = "masked_texture_refine"


class ReconstructionBackend(str, Enum):
    PIXAL3D = "pixal3d"
    TRELLIS2 = "trellis2"
    SPAR3D = "spar3d"
    HI3DGEN = "hi3dgen"


RECONSTRUCTION_BACKEND_LABELS = {
    ReconstructionBackend.PIXAL3D: "Pixal3D",
    ReconstructionBackend.TRELLIS2: "TRELLIS.2",
    ReconstructionBackend.SPAR3D: "SPAR3D",
    ReconstructionBackend.HI3DGEN: "Hi3DGen",
}


RECONSTRUCTION_BACKEND_STAGES = {
    ReconstructionBackend.PIXAL3D: tuple(
        stage for stage in RECONSTRUCTION_STAGES
        if stage not in {
            ReconstructionStage.NORMAL_MAP,
            ReconstructionStage.POINT_CLOUD,
        }
    ),
    ReconstructionBackend.TRELLIS2: tuple(
        stage for stage in RECONSTRUCTION_STAGES
        if stage not in {
            ReconstructionStage.NORMAL_MAP,
            ReconstructionStage.POINT_CLOUD,
        }
    ),
    ReconstructionBackend.SPAR3D: (
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.POINT_CLOUD,
        ReconstructionStage.FINAL_MESH,
    ),
    ReconstructionBackend.HI3DGEN: (
        ReconstructionStage.SOURCE_IMAGE,
        ReconstructionStage.NORMAL_MAP,
        ReconstructionStage.SPARSE_OCCUPANCY,
        ReconstructionStage.HR_SHAPE_FLOW,
        ReconstructionStage.HR_SHAPE_LATENT,
        ReconstructionStage.FINAL_MESH,
    ),
}


@dataclass(frozen=True)
class ReconstructionParameters:
    backend: ReconstructionBackend = ReconstructionBackend.PIXAL3D
    seed: int = 42
    steps: int = 12
    resolution: int = 1024
    lr_conditioning_resolution: int = 512
    manual_fov_degrees: float = 0.0
    decimation_target: int = 200_000
    texture_size: int = 2048
    low_vram: bool = True
    spar3d_guidance_scale: float = 3.0
    hi3dgen_slat_steps: int = 6
    hi3dgen_guidance_scale: float = 3.0
    hi3dgen_normal_resolution: int = 768

    def __post_init__(self) -> None:
        if not isinstance(self.backend, ReconstructionBackend):
            object.__setattr__(
                self, "backend", ReconstructionBackend(str(self.backend))
            )
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("reconstruction seed must be in [0, 2147483647]")
        if not 1 <= self.steps <= 50:
            raise ValueError("reconstruction steps must be in [1, 50]")
        if self.resolution not in (1024, 1280, 1536):
            raise ValueError("reconstruction resolution must be 1024, 1280, or 1536")
        if self.lr_conditioning_resolution not in (512, 1024):
            raise ValueError("LR conditioning resolution must be 512 or 1024")
        if not 0.0 <= self.manual_fov_degrees <= 120.0:
            raise ValueError("manual camera FOV must be in [0, 120] degrees")
        if not 50_000 <= self.decimation_target <= 1_000_000:
            raise ValueError("mesh face target must be in [50000, 1000000]")
        if self.texture_size not in (1024, 2048, 4096):
            raise ValueError("texture size must be 1024, 2048, or 4096")
        if not 0.0 <= self.spar3d_guidance_scale <= 20.0:
            raise ValueError("SPAR3D guidance scale must be in [0, 20]")
        if not 1 <= self.hi3dgen_slat_steps <= 50:
            raise ValueError("Hi3DGen structured-latent steps must be in [1, 50]")
        if not 0.0 <= self.hi3dgen_guidance_scale <= 20.0:
            raise ValueError("Hi3DGen guidance scale must be in [0, 20]")
        if self.hi3dgen_normal_resolution not in (512, 768, 1024):
            raise ValueError("Hi3DGen normal resolution must be 512, 768, or 1024")

    def to_dict(self) -> dict:
        return {
            "backend": self.backend.value,
            "seed": self.seed,
            "steps": self.steps,
            "resolution": self.resolution,
            "lr_conditioning_resolution": self.lr_conditioning_resolution,
            "manual_fov_degrees": self.manual_fov_degrees,
            "decimation_target": self.decimation_target,
            "texture_size": self.texture_size,
            "low_vram": self.low_vram,
            "spar3d_guidance_scale": self.spar3d_guidance_scale,
            "hi3dgen_slat_steps": self.hi3dgen_slat_steps,
            "hi3dgen_guidance_scale": self.hi3dgen_guidance_scale,
            "hi3dgen_normal_resolution": self.hi3dgen_normal_resolution,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "ReconstructionParameters":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            backend=ReconstructionBackend(
                payload.get("backend", ReconstructionBackend.PIXAL3D.value)
            ),
            seed=int(payload.get("seed", 42)),
            steps=int(payload.get("steps", 12)),
            resolution=int(payload.get("resolution", 1024)),
            lr_conditioning_resolution=int(
                payload.get("lr_conditioning_resolution", 512)
            ),
            manual_fov_degrees=float(payload.get("manual_fov_degrees", 0.0)),
            decimation_target=int(payload.get("decimation_target", 200_000)),
            texture_size=int(payload.get("texture_size", 2048)),
            low_vram=bool(payload.get("low_vram", True)),
            spar3d_guidance_scale=float(
                payload.get("spar3d_guidance_scale", 3.0)
            ),
            hi3dgen_slat_steps=int(payload.get("hi3dgen_slat_steps", 6)),
            hi3dgen_guidance_scale=float(
                payload.get("hi3dgen_guidance_scale", 3.0)
            ),
            hi3dgen_normal_resolution=int(
                payload.get("hi3dgen_normal_resolution", 768)
            ),
        )


@dataclass(frozen=True)
class ReconstructionRefineParameters:
    strength: float = 0.35
    steps: int = 8
    seed: int = 123
    rescale_t: float = 3.0
    guidance_strength: float = 7.5
    resize_detail_to_1024: bool = True

    def __post_init__(self) -> None:
        if not 0.0 < self.strength <= 1.0:
            raise ValueError("refine strength must be in (0, 1]")
        if not 1 <= self.steps <= 50:
            raise ValueError("refine steps must be in [1, 50]")
        if not 0 <= self.seed <= 2_147_483_647:
            raise ValueError("refine seed must be in [0, 2147483647]")
        if self.rescale_t <= 0.0:
            raise ValueError("refine rescale_t must be positive")
        if self.guidance_strength < 0.0:
            raise ValueError("refine guidance strength must be non-negative")

    def to_dict(self) -> dict:
        return {
            "strength": self.strength,
            "steps": self.steps,
            "seed": self.seed,
            "rescale_t": self.rescale_t,
            "guidance_strength": self.guidance_strength,
            "resize_detail_to_1024": self.resize_detail_to_1024,
        }

    @classmethod
    def from_dict(cls, payload: object) -> "ReconstructionRefineParameters":
        if not isinstance(payload, dict):
            return cls()
        return cls(
            strength=float(payload.get("strength", 0.35)),
            steps=int(payload.get("steps", 8)),
            seed=int(payload.get("seed", 123)),
            rescale_t=float(payload.get("rescale_t", 3.0)),
            guidance_strength=float(payload.get("guidance_strength", 7.5)),
            resize_detail_to_1024=bool(
                payload.get("resize_detail_to_1024", True)
            ),
        )


@dataclass(frozen=True)
class ReconstructionRun:
    run_id: str
    kind: ReconstructionRunKind
    glb_path: str
    source_path: str
    conditioning_path: str | None = None
    checkpoint_path: str | None = None
    texture_checkpoint_path: str | None = None
    parent_run_id: str | None = None
    vertex_count: int = 0
    triangle_count: int = 0
    mesh_count: int = 0
    backend: ReconstructionBackend = ReconstructionBackend.PIXAL3D
    stage_statuses: tuple[ReconstructionStageStatus, ...] = ()
    stage_progress: tuple[tuple[int, int], ...] = ()
    stage_artifacts: tuple["ReconstructionStageArtifact", ...] = ()


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
class ReconstructionRefineRequest:
    conditioning_image: Image.Image
    mask_image: Image.Image
    base_checkpoint_path: str
    parameters: ReconstructionRefineParameters = ReconstructionRefineParameters()
    generation_parameters: ReconstructionParameters = ReconstructionParameters()


@dataclass(frozen=True)
class ReconstructionTextureRefineRequest:
    conditioning_image: Image.Image
    mask_image: Image.Image
    shape_checkpoint_path: str
    texture_checkpoint_path: str
    parameters: ReconstructionRefineParameters = ReconstructionRefineParameters()
    generation_parameters: ReconstructionParameters = ReconstructionParameters()


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
    kind: ReconstructionRunKind = ReconstructionRunKind.BASE
    conditioning_path: str | None = None
    checkpoint_path: str | None = None
    texture_checkpoint_path: str | None = None
    backend: ReconstructionBackend = ReconstructionBackend.PIXAL3D


@dataclass(frozen=True)
class SegmentationRequest:
    image: np.ndarray
    invert: bool = True


@dataclass(frozen=True)
class SegmentationResult:
    mask: np.ndarray
    provenance: GenerationProvenance | None = None
