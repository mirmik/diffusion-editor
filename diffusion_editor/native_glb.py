"""Minimal facade over the installed Termin native cgltf binding."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_native_module = None


@dataclass(frozen=True)
class NativeMeshInfo:
    name: str
    primitive_count: int
    vertex_count: int
    index_count: int
    skinned: bool


@dataclass(frozen=True)
class NativeBaseColorTexture:
    """Encoded base-color image used by a whole static mesh."""

    payload: bytes
    mime_type: str
    factor: tuple[float, float, float, float]


def _load_native_module():
    global _native_module
    if _native_module is not None:
        return _native_module

    from termin.glb import _glb_native as module

    info = module.backend_info()
    if info.get("name") != "cgltf":
        raise RuntimeError(f"Unexpected Termin GLB backend: {info!r}")
    _native_module = module
    return module


class NativeGLBDocument:
    """Small static-mesh facade over Termin's native cgltf extension."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        module = _load_native_module()
        self._document = module.NativeDocument(str(self.path))
        self.meshes = tuple(
            NativeMeshInfo(
                name=entry["name"],
                primitive_count=int(entry["primitive_count"]),
                vertex_count=int(entry["vertex_count"]),
                index_count=int(entry["index_count"]),
                skinned=bool(entry["skinned"]),
            )
            for entry in self._document.meshes
        )

    def build_mesh(
        self,
        mesh_index: int,
        mesh_uuid: str,
        *,
        name: str = "",
        convert_to_z_up: bool = True,
    ):
        from tmesh import tc_mesh_get

        self._document.build_mesh(
            int(mesh_index),
            str(mesh_uuid),
            str(name),
            bool(convert_to_z_up),
            False,
        )
        mesh = tc_mesh_get(mesh_uuid)
        if mesh is None or not mesh.is_valid:
            raise RuntimeError(
                f"Native GLB build published no tmesh for UUID {mesh_uuid!r}"
            )
        return mesh

    def base_color_texture(self, mesh_index: int) -> NativeBaseColorTexture | None:
        """Return a texture that can be applied while drawing the whole mesh.

        ``draw_tc_mesh`` currently draws every primitive in one call.  A texture
        is therefore safe to bind only when the mesh has one primitive.  GLBs
        without a compatible material deliberately retain the viewport's solid
        fallback.
        """

        info = self.meshes[int(mesh_index)]
        if info.primitive_count != 1:
            return None
        primitive = self._document.primitive_info(int(mesh_index), 0)
        if not primitive["has_material"]:
            return None
        material = self._document.material_info(int(primitive["material_index"]))
        view = material["base_color_texture"]
        if (
            not view.get("present", False)
            or int(view.get("texcoord", -1)) != 0
            or bool(view.get("has_transform", False))
        ):
            return None
        texture = self._document.texture_info(int(view["texture_index"]))
        if not texture["has_image"]:
            return None
        image_index = int(texture["image_index"])
        image = self._document.image_info(image_index)
        factor = tuple(float(value) for value in material["base_color_factor"])
        return NativeBaseColorTexture(
            payload=bytes(self._document.image_payload(image_index)),
            mime_type=str(image["mime_type"]),
            factor=(factor[0], factor[1], factor[2], factor[3]),
        )


def inspect_glb_stats(path: str | Path) -> tuple[int, int, int]:
    """Read mesh statistics without replacing the model shown in a viewport."""
    document = NativeGLBDocument(path)
    vertices = sum(mesh.vertex_count for mesh in document.meshes)
    triangles = sum(mesh.index_count for mesh in document.meshes) // 3
    return vertices, triangles, len(document.meshes)
