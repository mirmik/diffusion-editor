"""Document node for one image-to-3D reconstruction."""

from __future__ import annotations

from enum import Enum
import os

from ..generation.types import (
    RECONSTRUCTION_BACKEND_STAGES,
    RECONSTRUCTION_STAGES,
    ReconstructionParameters,
    ReconstructionRefineParameters,
    ReconstructionRun,
    ReconstructionRunKind,
    ReconstructionStage,
    ReconstructionStageArtifact,
    ReconstructionStageEvent,
    ReconstructionStageStatus,
)
from .layer import Layer


class ReconstructionStatus(str, Enum):
    EMPTY = "empty"
    GENERATING = "generating"
    READY = "ready"
    FAILED = "failed"
    MISSING = "artifact missing"


class ReconstructionLayer(Layer):
    """Non-raster tree node owning one reconstruction result."""

    node_type = "reconstruction"
    contributes_to_composite = False
    accepts_pixel_edits = False

    def __init__(self, name: str, *, layer_id: str | None = None) -> None:
        # LayerStack currently owns a homogeneous Layer tree. A transparent
        # 1x1 backing keeps its structural invariants without allocating a
        # canvas-sized raster for this non-pixel node.
        super().__init__(name, 1, 1, layer_id=layer_id)
        self.reconstruction_status = ReconstructionStatus.EMPTY
        self.glb_path: str | None = None
        self.vertex_count = 0
        self.triangle_count = 0
        self.mesh_count = 0
        self.generation_parameters = ReconstructionParameters()
        self.refine_parameters = ReconstructionRefineParameters()
        self.runs: tuple[ReconstructionRun, ...] = ()
        self.active_run_id: str | None = None
        self.resume_checkpoint_path: str | None = None
        self.resume_stage: ReconstructionStage | None = None
        self.resume_source_sha256: str | None = None
        self.resume_parameters: ReconstructionParameters | None = None
        self._initialize_stage_state()

    @property
    def active_run(self) -> ReconstructionRun | None:
        return next(
            (run for run in self.runs if run.run_id == self.active_run_id),
            None,
        )

    @property
    def base_run(self) -> ReconstructionRun | None:
        return next(
            (run for run in reversed(self.runs)
             if run.kind is ReconstructionRunKind.BASE),
            None,
        )

    def _initialize_stage_state(self) -> None:
        """Initialize transient staged-generation state.

        Base layer deserialization intentionally constructs subclasses with
        ``__new__``. Keep this state in one initializer so loaded projects gain
        fields introduced after their manifest was written.
        """
        self.target_stage = ReconstructionStage.FINAL_MESH
        self.selected_preview_stage = ReconstructionStage.SOURCE_IMAGE
        self.preview_stage_pinned = False
        self.stage_statuses = {
            stage: ReconstructionStageStatus.PENDING
            for stage in RECONSTRUCTION_STAGES
        }
        self.stage_progress = {
            stage: (0, 0) for stage in RECONSTRUCTION_STAGES
        }
        self.stage_artifacts: dict[
            ReconstructionStage, ReconstructionStageArtifact
        ] = {}

    def begin_staged_generation(
        self, resume_from: ReconstructionStage | None = None
    ) -> None:
        supported = RECONSTRUCTION_BACKEND_STAGES[
            self.generation_parameters.backend
        ]
        if self.target_stage not in supported:
            self.target_stage = ReconstructionStage.FINAL_MESH
        target_index = supported.index(self.target_stage)
        for stage in RECONSTRUCTION_STAGES:
            if (
                resume_from is not None
                and stage in supported
                and supported.index(stage) <= supported.index(resume_from)
            ):
                # Keep accepted previews and progress for the prefix which the
                # worker is going to restore from its checkpoint.
                continue
            self.stage_statuses[stage] = (
                ReconstructionStageStatus.PENDING
                if stage in supported and supported.index(stage) <= target_index
                else ReconstructionStageStatus.SKIPPED
            )
            self.stage_progress[stage] = (0, 0)
        if resume_from is None:
            self.stage_artifacts.clear()

    def apply_stage_event(self, event: ReconstructionStageEvent) -> None:
        self.stage_statuses[event.stage] = event.status
        self.stage_progress[event.stage] = (event.progress, event.total)
        if event.artifact is not None:
            self.stage_artifacts[event.stage] = event.artifact
            if not self.preview_stage_pinned:
                self.selected_preview_stage = event.stage

    def to_dict(self, path: str) -> dict:
        payload = super().to_dict(path)
        payload["type"] = "reconstruction"
        payload["reconstruction"] = {
            "status": self.reconstruction_status.value,
            "glb_path": self.glb_path,
            "vertex_count": self.vertex_count,
            "triangle_count": self.triangle_count,
            "mesh_count": self.mesh_count,
            "generation_parameters": self.generation_parameters.to_dict(),
            "refine_parameters": self.refine_parameters.to_dict(),
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict, archive, tile_size: int = 256):
        layer = cls._from_dict_base(payload, archive, tile_size=tile_size)
        state = payload.get("reconstruction", {})
        path = state.get("glb_path")
        status = state.get("status", ReconstructionStatus.EMPTY.value)
        if path and not os.path.isfile(path):
            status = ReconstructionStatus.MISSING.value
        try:
            layer.reconstruction_status = ReconstructionStatus(status)
        except ValueError:
            layer.reconstruction_status = ReconstructionStatus.EMPTY
        layer.glb_path = str(path) if path else None
        layer.vertex_count = int(state.get("vertex_count", 0))
        layer.triangle_count = int(state.get("triangle_count", 0))
        layer.mesh_count = int(state.get("mesh_count", 0))
        layer.runs = ()
        layer.active_run_id = None
        layer.resume_checkpoint_path = None
        layer.resume_stage = None
        layer.resume_source_sha256 = None
        layer.resume_parameters = None
        try:
            layer.generation_parameters = ReconstructionParameters.from_dict(
                state.get("generation_parameters")
            )
        except (TypeError, ValueError):
            layer.generation_parameters = ReconstructionParameters()
        try:
            layer.refine_parameters = ReconstructionRefineParameters.from_dict(
                state.get("refine_parameters")
            )
        except (TypeError, ValueError):
            layer.refine_parameters = ReconstructionRefineParameters()
        layer._initialize_stage_state()
        if layer.glb_path and os.path.isfile(layer.glb_path):
            restored_run = ReconstructionRun(
                run_id="restored-base",
                kind=ReconstructionRunKind.BASE,
                glb_path=layer.glb_path,
                source_path="",
                vertex_count=layer.vertex_count,
                triangle_count=layer.triangle_count,
                mesh_count=layer.mesh_count,
                stage_statuses=tuple(
                    ReconstructionStageStatus.READY
                    if stage is ReconstructionStage.FINAL_MESH
                    else ReconstructionStageStatus.PENDING
                    for stage in RECONSTRUCTION_STAGES
                ),
                stage_progress=tuple(
                    (0, 0) for _stage in RECONSTRUCTION_STAGES
                ),
                stage_artifacts=(ReconstructionStageArtifact(
                    ReconstructionStage.FINAL_MESH,
                    layer.glb_path,
                    "mesh",
                ),),
            )
            layer.runs = (restored_run,)
            layer.active_run_id = restored_run.run_id
            final_stage = ReconstructionStage.FINAL_MESH
            layer.stage_statuses[final_stage] = ReconstructionStageStatus.READY
            layer.stage_artifacts[final_stage] = ReconstructionStageArtifact(
                final_stage,
                layer.glb_path,
                "mesh",
            )
            layer.selected_preview_stage = final_stage
        return layer
