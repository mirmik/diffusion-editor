#!/usr/bin/env python3
"""Render four cheap inventory views of an FBX or the currently open blend.

Run through Blender::

    blender --background --factory-startup --disable-autoexec \
      --python scripts/render-character-inventory-preview.py -- \
      --asset character.fbx --output-dir /tmp/preview

    blender --background --disable-autoexec character.blend \
      --python scripts/render-character-inventory-preview.py -- \
      --output-dir /tmp/preview
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def _arguments() -> argparse.Namespace:
    values = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", type=Path, help="FBX to import.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resolution", type=int, default=256)
    args = parser.parse_args(values)
    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    return args


def _load_asset(path: Path | None) -> str:
    if path is None:
        if not bpy.data.filepath:
            raise RuntimeError("open a blend file or pass --asset")
        return str(Path(bpy.data.filepath).resolve())
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    if path.suffix.lower() == ".fbx":
        result = bpy.ops.import_scene.fbx(filepath=str(path), use_anim=False)
    elif path.suffix.lower() == ".dae":
        result = bpy.ops.wm.collada_import(filepath=str(path))
    else:
        raise ValueError(f"unsupported imported asset: {path}")
    if "FINISHED" not in result:
        raise RuntimeError(f"failed to import {path}: {result}")
    return str(path)


def _candidate_meshes(imported: bool) -> list[bpy.types.Object]:
    meshes = []
    for obj in bpy.context.scene.objects:
        if obj.type != "MESH" or obj.hide_render:
            continue
        upper = obj.name.upper()
        if upper.startswith(("WGT-", "HLP-")):
            continue
        if not imported and not obj.visible_get():
            continue
        meshes.append(obj)
    if not meshes:
        raise RuntimeError("asset contains no render-visible mesh objects")
    selected = set(meshes)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            obj.hide_render = obj not in selected
    return meshes


def _bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    minimum = Vector((math.inf, math.inf, math.inf))
    maximum = Vector((-math.inf, -math.inf, -math.inf))
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for original in objects:
        evaluated = original.evaluated_get(depsgraph)
        for corner in evaluated.bound_box:
            point = evaluated.matrix_world @ Vector(corner)
            for axis in range(3):
                minimum[axis] = min(minimum[axis], point[axis])
                maximum[axis] = max(maximum[axis], point[axis])
    if not all(math.isfinite(value) for value in (*minimum, *maximum)):
        raise RuntimeError("could not calculate finite character bounds")
    return minimum, maximum


def _look_at(obj: bpy.types.Object, target: Vector) -> None:
    obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()


def _configure(resolution: int) -> bpy.types.Object:
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_WORKBENCH"
    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.film_transparent = True
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.display.shading.show_specular_highlight = True
    camera_data = bpy.data.cameras.new("InventoryPreviewCamera")
    camera_data.type = "PERSP"
    camera_data.lens = 55.0
    camera_data.sensor_width = 36.0
    camera_data.clip_start = 0.001
    camera_data.clip_end = 100000.0
    camera = bpy.data.objects.new("InventoryPreviewCamera", camera_data)
    bpy.context.scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


def main() -> int:
    args = _arguments()
    source = _load_asset(args.asset)
    imported = args.asset is not None
    objects = _candidate_meshes(imported)
    minimum, maximum = _bounds(objects)
    center = (minimum + maximum) * 0.5
    span = maximum - minimum
    camera = _configure(args.resolution)
    half_fov = float(camera.data.angle_y) * 0.5
    fit_half_extent = max(
        float(span[0]) * 0.5,
        float(span[1]) * 0.5,
        float(span[2]) * 0.5,
        0.001,
    )
    distance = (
        fit_half_extent / max(math.tan(half_fov), 0.05)
    ) * 1.18
    elevation = math.radians(6.0)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rendered = []
    for azimuth_degrees in (0, 90, 180, 270):
        azimuth = math.radians(azimuth_degrees)
        horizontal = math.cos(elevation) * distance
        camera.location = center + Vector(
            (
                math.sin(azimuth) * horizontal,
                -math.cos(azimuth) * horizontal,
                math.sin(elevation) * distance,
            )
        )
        _look_at(camera, center)
        bpy.context.view_layer.update()
        output = args.output_dir / f"az{azimuth_degrees:03d}.png"
        bpy.context.scene.render.filepath = str(output)
        bpy.ops.render.render(write_still=True)
        rendered.append(output.name)
    manifest = {
        "source": source,
        "blender_version": bpy.app.version_string,
        "mesh_count": len(objects),
        "mesh_names": [obj.name for obj in objects],
        "armatures": [
            {"name": obj.name, "bones": len(obj.data.bones)}
            for obj in bpy.context.scene.objects
            if obj.type == "ARMATURE"
        ],
        "bounds": {"minimum": list(minimum), "maximum": list(maximum)},
        "rendered": rendered,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print("CHARACTER_INVENTORY_PREVIEW=" + json.dumps(manifest))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
