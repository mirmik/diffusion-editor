"""Document node for one image-to-3D reconstruction."""

from __future__ import annotations

from enum import Enum
import os

from ..generation.types import (
    RECONSTRUCTION_STAGES,
    ReconstructionParameters,
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
        self._initialize_stage_state()

    def _initialize_stage_state(self) -> None:
        """Initialize transient staged-generation state.

        Base layer deserialization intentionally constructs subclasses with
        ``__new__``. Keep this state in one initializer so loaded projects gain
        fields introduced after their manifest was written.
        """
        self.target_stage = ReconstructionStage.FINAL_MESH
        self.selected_preview_stage = ReconstructionStage.SOURCE_IMAGE
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

    def begin_staged_generation(self) -> None:
        target_index = RECONSTRUCTION_STAGES.index(self.target_stage)
        for index, stage in enumerate(RECONSTRUCTION_STAGES):
            self.stage_statuses[stage] = (
                ReconstructionStageStatus.PENDING
                if index <= target_index
                else ReconstructionStageStatus.SKIPPED
            )
            self.stage_progress[stage] = (0, 0)
        self.stage_artifacts.clear()

    def apply_stage_event(self, event: ReconstructionStageEvent) -> None:
        self.stage_statuses[event.stage] = event.status
        self.stage_progress[event.stage] = (event.progress, event.total)
        if event.artifact is not None:
            self.stage_artifacts[event.stage] = event.artifact
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
        try:
            layer.generation_parameters = ReconstructionParameters.from_dict(
                state.get("generation_parameters")
            )
        except (TypeError, ValueError):
            layer.generation_parameters = ReconstructionParameters()
        layer._initialize_stage_state()
        if layer.glb_path and os.path.isfile(layer.glb_path):
            final_stage = ReconstructionStage.FINAL_MESH
            layer.stage_statuses[final_stage] = ReconstructionStageStatus.READY
            layer.stage_artifacts[final_stage] = ReconstructionStageArtifact(
                final_stage,
                layer.glb_path,
                "mesh",
            )
            layer.selected_preview_stage = final_stage
        return layer
