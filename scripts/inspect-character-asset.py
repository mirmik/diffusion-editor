#!/usr/bin/env python3
"""Inspect rig, bounds, and material dependencies of character assets.

Run through Blender, for example::

    blender --background --factory-startup --disable-autoexec \
      --python scripts/inspect-character-asset.py -- April.fbx snow.blend
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import bpy


def _arguments() -> list[Path]:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    if not values:
        raise SystemExit("pass one or more FBX paths after --")
    return [Path(value).resolve() for value in values]


def _world_bounds(objects: list[bpy.types.Object]) -> tuple[list[float], list[float]]:
    minimum = [float("inf")] * 3
    maximum = [float("-inf")] * 3
    for obj in objects:
        for corner in obj.bound_box:
            world = obj.matrix_world @ __import__("mathutils").Vector(corner)
            for axis in range(3):
                minimum[axis] = min(minimum[axis], float(world[axis]))
                maximum[axis] = max(maximum[axis], float(world[axis]))
    return minimum, maximum


def _image_paths(materials: set[bpy.types.Material]) -> list[dict]:
    images = {}
    for material in materials:
        if not material.use_nodes or material.node_tree is None:
            continue
        for node in material.node_tree.nodes:
            if node.type != "TEX_IMAGE" or node.image is None:
                continue
            raw = bpy.path.abspath(node.image.filepath)
            path = Path(raw).resolve() if raw else None
            key = str(path) if path else node.image.name
            images[key] = {
                "name": node.image.name,
                "path": str(path) if path else None,
                "exists": bool(path and path.exists()),
            }
    return list(images.values())


def main() -> int:
    records = []
    for asset in _arguments():
        if asset.suffix.lower() == ".blend":
            result = bpy.ops.wm.open_mainfile(
                filepath=str(asset), load_ui=False, use_scripts=False
            )
            if "FINISHED" not in result:
                raise RuntimeError(f"failed to open {asset}: {result}")
        elif asset.suffix.lower() == ".fbx":
            bpy.ops.wm.read_factory_settings(use_empty=True)
            result = bpy.ops.import_scene.fbx(filepath=str(asset), use_anim=False)
            if "FINISHED" not in result:
                raise RuntimeError(f"failed to import {asset}: {result}")
        else:
            raise ValueError(f"unsupported character asset: {asset}")
        meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
        armatures = [
            obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"
        ]
        minimum, maximum = _world_bounds(meshes)
        materials = {
            slot.material
            for obj in meshes
            for slot in obj.material_slots
            if slot.material is not None
        }
        records.append(
            {
                "asset": str(asset),
                "size_bytes": asset.stat().st_size,
                "meshes": [
                    {
                        "name": obj.name,
                        "vertices": len(obj.data.vertices),
                        "armature_modifiers": [
                            modifier.object.name
                            for modifier in obj.modifiers
                            if modifier.type == "ARMATURE"
                            and modifier.object is not None
                        ],
                        "vertex_groups": [group.name for group in obj.vertex_groups],
                        "location_world": [float(value) for value in obj.location],
                        "materials": [
                            slot.material.name
                            for slot in obj.material_slots
                            if slot.material is not None
                        ],
                    }
                    for obj in meshes
                ],
                "armatures": [
                    {
                        "name": obj.name,
                        "bones": [bone.name for bone in obj.data.bones],
                    }
                    for obj in armatures
                ],
                "bounds_world": {"minimum": minimum, "maximum": maximum},
                "dimensions_world": [
                    maximum[index] - minimum[index] for index in range(3)
                ],
                "images": _image_paths(materials),
            }
        )
    print("CHARACTER_ASSET_INSPECTION_BEGIN")
    print(json.dumps(records, indent=2))
    print("CHARACTER_ASSET_INSPECTION_END")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
