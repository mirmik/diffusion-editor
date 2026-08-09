"""Document node for one image-to-3D reconstruction."""

from __future__ import annotations

from enum import Enum
import os

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

    def to_dict(self, path: str) -> dict:
        payload = super().to_dict(path)
        payload["type"] = "reconstruction"
        payload["reconstruction"] = {
            "status": self.reconstruction_status.value,
            "glb_path": self.glb_path,
            "vertex_count": self.vertex_count,
            "triangle_count": self.triangle_count,
            "mesh_count": self.mesh_count,
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
        return layer
