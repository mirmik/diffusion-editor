"""Minimal cgltf binding loaded directly from the selected Termin SDK.

The current high-level ``termin-glb`` distribution declares the complete
editor asset stack, including legacy tcgui.  Diffusion Editor only needs the
native ``NativeDocument -> tmesh`` boundary, so this module loads that single
extension from the trusted SDK wheelhouse until Termin provides a split wheel.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import tempfile
import zipfile

from .sdk_runtime import resolve_sdk


_native_module = None
_extraction_root: tempfile.TemporaryDirectory[str] | None = None


@dataclass(frozen=True)
class NativeMeshInfo:
    name: str
    vertex_count: int
    index_count: int
    skinned: bool


def _load_native_module():
    global _native_module, _extraction_root
    if _native_module is not None:
        return _native_module

    wheelhouse = resolve_sdk() / "wheels"
    wheels = sorted(wheelhouse.glob("termin_glb-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(
            "Expected exactly one termin_glb wheel in the selected SDK, got "
            f"{len(wheels)} in {wheelhouse}"
        )
    with zipfile.ZipFile(wheels[0]) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.startswith("termin/glb/_glb_native")
            and (name.endswith(".so") or name.endswith(".pyd"))
        ]
        if len(members) != 1:
            raise RuntimeError(
                f"termin_glb wheel contains {len(members)} native bindings"
            )
        _extraction_root = tempfile.TemporaryDirectory(
            prefix="diffusion-editor-glb-binding-"
        )
        extracted = Path(archive.extract(members[0], _extraction_root.name))

    spec = importlib.util.spec_from_file_location("_glb_native", extracted)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load cgltf binding from {extracted}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
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
