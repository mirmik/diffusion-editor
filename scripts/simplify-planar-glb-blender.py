#!/usr/bin/env python3
"""Remove redundant planar geometry from a GLB without voxel remeshing.

Run through Blender, for example::

    blender -b --python scripts/simplify-planar-glb-blender.py -- \
        input.glb output.glb --angle-degrees 0.5

The dissolve modifier preserves boundaries by default. The result is
triangulated before GLB export so the reported face count matches the file.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

import bpy


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--angle-degrees", type=float, default=0.5)
    parser.add_argument(
        "--dissolve-boundaries",
        action="store_true",
        help="Also dissolve vertices on open mesh boundaries",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def _join_imported_meshes() -> bpy.types.Object:
    meshes = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not meshes:
        raise RuntimeError("GLB contained no mesh objects")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in meshes:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def _counts(obj: bpy.types.Object) -> tuple[int, int, int]:
    mesh = obj.data
    return len(mesh.vertices), len(mesh.edges), len(mesh.polygons)


def main() -> int:
    args = _arguments()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not 0.0 <= args.angle_degrees <= 180.0:
        raise ValueError("angle-degrees must be in [0, 180]")

    started = time.monotonic()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(args.input.resolve()))
    obj = _join_imported_meshes()
    source_vertices, source_edges, source_faces = _counts(obj)
    print(
        f"[source] vertices={source_vertices:,} edges={source_edges:,} "
        f"faces={source_faces:,}",
        flush=True,
    )

    dissolve = obj.modifiers.new(name="Conservative planar dissolve", type="DECIMATE")
    dissolve.decimate_type = "DISSOLVE"
    dissolve.angle_limit = math.radians(args.angle_degrees)
    dissolve.use_dissolve_boundaries = args.dissolve_boundaries
    dissolve.delimit = {"MATERIAL", "SEAM", "SHARP", "UV"}
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=dissolve.name)
    dissolved_vertices, dissolved_edges, dissolved_polygons = _counts(obj)
    print(
        f"[dissolved] vertices={dissolved_vertices:,} "
        f"edges={dissolved_edges:,} polygons={dissolved_polygons:,}",
        flush=True,
    )

    triangulate = obj.modifiers.new(name="Triangulate for GLB", type="TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=triangulate.name)
    mesh_was_invalid = obj.data.validate(clean_customdata=True)
    obj.data.update()
    final_vertices, final_edges, final_faces = _counts(obj)
    print(
        f"[final] vertices={final_vertices:,} edges={final_edges:,} "
        f"faces={final_faces:,}",
        flush=True,
    )

    for polygon in obj.data.polygons:
        polygon.use_smooth = True
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=str(args.output.resolve()),
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_normals=True,
        export_materials="NONE",
    )

    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "algorithm": "Blender limited planar dissolve; no voxel remesh",
        "angle_degrees": args.angle_degrees,
        "dissolve_boundaries": args.dissolve_boundaries,
        "source_vertices": source_vertices,
        "source_edges": source_edges,
        "source_faces": source_faces,
        "dissolved_vertices": dissolved_vertices,
        "dissolved_edges": dissolved_edges,
        "dissolved_polygons": dissolved_polygons,
        "final_vertices": final_vertices,
        "final_edges": final_edges,
        "final_faces": final_faces,
        "mesh_validation_changed_data": mesh_was_invalid,
        "face_reduction_ratio": 1.0 - final_faces / source_faces,
        "elapsed_seconds": time.monotonic() - started,
        "coordinates": "glTF 2.0 right-handed Y-up",
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
