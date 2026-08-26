#!/usr/bin/env python3
"""Render a Pixal3D GLB at the nine nominal Multiple Angles cameras.

Run through Blender, for example::

    blender --background --python scripts/render-pixal3d-multiple-angles.py -- \
        model.glb output-dir --source-image front.png
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


AZIMUTHS = (
    (0, "front view"),
    (315, "front-left quarter view"),
    (45, "front-right quarter view"),
)
ELEVATIONS = (
    ("low", -30, "low-angle shot"),
    ("eye", 0, "eye-level shot"),
    ("elevated", 30, "elevated shot"),
)


def _arguments() -> argparse.Namespace:
    raw = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--source-image", type=Path)
    parser.add_argument("--width", type=int)
    parser.add_argument("--height", type=int)
    parser.add_argument("--margin", type=float, default=1.10)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--background", default="0.62,0.61,0.61")
    return parser.parse_args(raw)


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        for block in list(collection):
            if block.users == 0:
                collection.remove(block)


def _world_bounds(objects: list[bpy.types.Object]) -> tuple[Vector, Vector]:
    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in objects
        for corner in obj.bound_box
    ]
    if not corners:
        raise RuntimeError("the imported GLB contains no mesh bounds")
    minimum = Vector(tuple(min(point[axis] for point in corners) for axis in range(3)))
    maximum = Vector(tuple(max(point[axis] for point in corners) for axis in range(3)))
    return minimum, maximum


def _bounds_corners(minimum: Vector, maximum: Vector) -> list[Vector]:
    return [
        Vector((x, y, z))
        for x in (minimum.x, maximum.x)
        for y in (minimum.y, maximum.y)
        for z in (minimum.z, maximum.z)
    ]


def _camera_direction(azimuth_degrees: float, elevation_degrees: float) -> Vector:
    """Return subject-to-camera direction in Pixal3D's exported Z-up frame."""
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    horizontal = math.cos(elevation)
    # Pixal3D's semantic front is viewed from +Y after glTF import.
    return Vector((
        -math.sin(azimuth) * horizontal,
        math.cos(azimuth) * horizontal,
        math.sin(elevation),
    )).normalized()


def _view_basis(direction: Vector) -> tuple[Vector, Vector]:
    forward = -direction
    right = forward.cross(Vector((0.0, 0.0, 1.0))).normalized()
    up = right.cross(forward).normalized()
    return right, up


def _shared_ortho_scale(
    corners: list[Vector],
    center: Vector,
    aspect: float,
    margin: float,
) -> float:
    required = 0.0
    for _elevation_name, elevation, _descriptor in ELEVATIONS:
        for azimuth, _azimuth_descriptor in AZIMUTHS:
            right, up = _view_basis(_camera_direction(azimuth, elevation))
            horizontal = [((point - center).dot(right)) for point in corners]
            vertical = [((point - center).dot(up)) for point in corners]
            horizontal_extent = max(horizontal) - min(horizontal)
            vertical_extent = max(vertical) - min(vertical)
            required = max(required, vertical_extent, horizontal_extent / aspect)
    return required * margin


def _add_area_light(
    name: str,
    location: tuple[float, float, float],
    energy: float,
    size: float,
    target: Vector,
) -> None:
    data = bpy.data.lights.new(name=name, type="AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = target + Vector(location)
    light.rotation_euler = (target - light.location).to_track_quat("-Z", "Y").to_euler()


def main() -> int:
    args = _arguments()
    if not args.model.is_file():
        raise SystemExit(f"missing GLB: {args.model}")
    if args.source_image is not None and not args.source_image.is_file():
        raise SystemExit(f"missing source image: {args.source_image}")
    if args.source_image is not None:
        image = bpy.data.images.load(str(args.source_image.resolve()), check_existing=False)
        width, height = int(image.size[0]), int(image.size[1])
        bpy.data.images.remove(image)
    else:
        width = args.width or 1024
        height = args.height or 1024
    if args.width is not None:
        width = args.width
    if args.height is not None:
        height = args.height
    if width <= 0 or height <= 0:
        raise SystemExit("render dimensions must be positive")

    background = tuple(float(value) for value in args.background.split(","))
    if len(background) != 3:
        raise SystemExit("--background must be R,G,B")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    _clear_scene()
    bpy.ops.import_scene.gltf(filepath=str(args.model.resolve()))
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    minimum, maximum = _world_bounds(meshes)
    center = (minimum + maximum) * 0.5
    corners = _bounds_corners(minimum, maximum)
    diagonal = (maximum - minimum).length
    aspect = width / height
    ortho_scale = _shared_ortho_scale(corners, center, aspect, args.margin)

    camera_data = bpy.data.cameras.new("Multiple Angles Camera")
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = ortho_scale
    camera_data.lens = 70.0
    camera = bpy.data.objects.new("Multiple Angles Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    bpy.context.scene.camera = camera

    light_scale = max(diagonal, 1.0)
    _add_area_light(
        "Key", (-1.4 * light_scale, 1.5 * light_scale, 1.8 * light_scale),
        1100.0, 1.6 * light_scale, center,
    )
    _add_area_light(
        "Fill", (1.6 * light_scale, 0.8 * light_scale, 0.5 * light_scale),
        700.0, 1.8 * light_scale, center,
    )
    _add_area_light(
        "Rim", (0.0, -1.8 * light_scale, 1.3 * light_scale),
        900.0, 1.2 * light_scale, center,
    )

    scene = bpy.context.scene
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    if hasattr(scene, "eevee"):
        scene.eevee.taa_render_samples = args.samples
    scene.render.resolution_x = width
    scene.render.resolution_y = height
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.use_file_extension = True
    scene.render.use_overwrite = True
    scene.render.use_persistent_data = True
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.world.use_nodes = True
    world_background = scene.world.node_tree.nodes.get("Background")
    world_background.inputs["Color"].default_value = (*background, 1.0)
    world_background.inputs["Strength"].default_value = 0.7

    jobs = []
    camera_distance = max(diagonal * 3.0, 3.0)
    for elevation_name, elevation_degrees, elevation_descriptor in ELEVATIONS:
        for azimuth_degrees, azimuth_descriptor in AZIMUTHS:
            direction = _camera_direction(azimuth_degrees, elevation_degrees)
            camera.location = center + direction * camera_distance
            camera.rotation_euler = (
                center - camera.location
            ).to_track_quat("-Z", "Y").to_euler()
            output = args.output_dir / (
                f"mv-{elevation_name}-{azimuth_degrees:03d}.png"
            )
            scene.render.filepath = str(output.resolve())
            print(
                f"Rendering {output.name}: azimuth={azimuth_degrees}, "
                f"elevation={elevation_degrees}",
                flush=True,
            )
            bpy.ops.render.render(write_still=True)
            jobs.append({
                "elevation": elevation_name,
                "elevation_degrees": elevation_degrees,
                "azimuth_degrees": azimuth_degrees,
                "azimuth_descriptor": azimuth_descriptor,
                "elevation_descriptor": elevation_descriptor,
                "output": str(output.resolve()),
                "camera_location": list(camera.location),
                "camera_rotation_euler": list(camera.rotation_euler),
            })

    manifest = {
        "schema": "diffusion-editor.pixal3d-multiple-angles-renders",
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "source_image": (
            str(args.source_image.resolve()) if args.source_image is not None else None
        ),
        "output_size": [width, height],
        "coordinate_convention": (
            "Blender Z-up after glTF import; Pixal3D semantic front viewed from +Y; "
            "positive azimuth moves the camera toward -X, matching the "
            "Multiple Angles 045/315 image convention"
        ),
        "projection": "orthographic",
        "ortho_scale": ortho_scale,
        "margin": args.margin,
        "mesh_statistics": {
            "objects": len(meshes),
            "vertices": sum(len(obj.data.vertices) for obj in meshes),
            "faces": sum(len(obj.data.polygons) for obj in meshes),
        },
        "bounds": {"minimum": list(minimum), "maximum": list(maximum)},
        "jobs": jobs,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
