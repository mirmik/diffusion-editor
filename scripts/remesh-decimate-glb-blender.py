#!/usr/bin/env python3
"""Fuse a GLB with Blender's voxel remesher, then decimate it.

Run through Blender, for example::

    blender -b --python scripts/remesh-decimate-glb-blender.py -- \
        input.glb output.glb --voxel-size 0.00075 --target-faces 1000000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import bpy
import bmesh


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.00075)
    parser.add_argument("--target-faces", type=int, default=1_000_000)
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--weld-distance", type=float, default=1e-7)
    parser.add_argument(
        "--remove-disconnected-threshold",
        type=float,
        help=(
            "Discard remeshed components smaller than this ratio of the "
            "largest component's polygon count"
        ),
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def _mesh_counts(obj) -> tuple[int, int]:
    return len(obj.data.vertices), len(obj.data.polygons)


def _topology_counts(obj) -> tuple[int, int]:
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    boundary = sum(edge.is_boundary for edge in mesh.edges)
    non_manifold = sum(not edge.is_manifold for edge in mesh.edges)
    mesh.free()
    return boundary, non_manifold


def _fill_boundary_loops(obj, weld_distance: float) -> dict[str, int]:
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    boundary_before = sum(edge.is_boundary for edge in mesh.edges)
    non_manifold_before = sum(not edge.is_manifold for edge in mesh.edges)
    vertices_before_weld = len(mesh.verts)
    bmesh.ops.remove_doubles(
        mesh,
        verts=list(mesh.verts),
        dist=weld_distance,
    )
    bmesh.ops.dissolve_degenerate(
        mesh,
        edges=list(mesh.edges),
        dist=weld_distance,
    )
    boundary = [edge for edge in mesh.edges if edge.is_boundary]
    boundary_after_weld = len(boundary)
    vertices_after_weld = len(mesh.verts)
    non_manifold_after_weld = sum(not edge.is_manifold for edge in mesh.edges)
    result = bmesh.ops.holes_fill(mesh, edges=boundary, sides=0)
    created = list(result.get("faces", ()))
    bmesh.ops.recalc_face_normals(mesh, faces=list(mesh.faces))
    mesh.normal_update()
    boundary_after = sum(edge.is_boundary for edge in mesh.edges)
    non_manifold_after = sum(not edge.is_manifold for edge in mesh.edges)
    mesh.to_mesh(obj.data)
    mesh.free()
    obj.data.update()
    return {
        "boundary_edges_before_weld": boundary_before,
        "non_manifold_edges_before_fill": non_manifold_before,
        "vertices_welded": vertices_before_weld - vertices_after_weld,
        "boundary_edges_after_weld": boundary_after_weld,
        "non_manifold_edges_after_weld": non_manifold_after_weld,
        "hole_cap_faces_created": len(created),
        "boundary_edges_after_fill": boundary_after,
        "non_manifold_edges_after_fill": non_manifold_after,
    }


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


def main() -> int:
    args = _arguments()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.voxel_size <= 0:
        raise ValueError("voxel-size must be positive")
    if args.target_faces <= 0:
        raise ValueError("target-faces must be positive")
    if args.weld_distance < 0:
        raise ValueError("weld-distance must be non-negative")
    if (
        args.remove_disconnected_threshold is not None
        and not 0 <= args.remove_disconnected_threshold <= 1
    ):
        raise ValueError("remove-disconnected-threshold must be in [0, 1]")

    started = time.monotonic()
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    bpy.ops.import_scene.gltf(filepath=str(args.input.resolve()))
    obj = _join_imported_meshes()
    source_vertices, source_faces = _mesh_counts(obj)
    print(
        f"[source] vertices={source_vertices:,} faces={source_faces:,}",
        flush=True,
    )
    repair = {}
    if args.fill_holes:
        repair = _fill_boundary_loops(obj, args.weld_distance)
        print(f"[repair] {repair}", flush=True)

    remesh = obj.modifiers.new(name="Voxel stitch", type="REMESH")
    remesh.mode = "VOXEL"
    remesh.voxel_size = args.voxel_size
    remesh.adaptivity = 0.0
    remesh.use_remove_disconnected = (
        args.remove_disconnected_threshold is not None
    )
    if args.remove_disconnected_threshold is not None:
        remesh.threshold = args.remove_disconnected_threshold
    remesh.use_smooth_shade = True
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=remesh.name)
    remesh_vertices, remesh_polygons = _mesh_counts(obj)
    print(
        f"[remesh] vertices={remesh_vertices:,} "
        f"polygons={remesh_polygons:,}",
        flush=True,
    )

    triangulate = obj.modifiers.new(name="Triangulate remesh", type="TRIANGULATE")
    bpy.ops.object.modifier_apply(modifier=triangulate.name)
    _, remesh_faces = _mesh_counts(obj)
    if remesh_faces > args.target_faces:
        decimate = obj.modifiers.new(name="Quadric decimation", type="DECIMATE")
        decimate.decimate_type = "COLLAPSE"
        decimate.ratio = args.target_faces / remesh_faces
        decimate.use_collapse_triangulate = False
        bpy.ops.object.modifier_apply(modifier=decimate.name)
    mesh_was_invalid = obj.data.validate(clean_customdata=True)
    obj.data.update()
    final_vertices, final_faces = _mesh_counts(obj)
    boundary_edges, non_manifold_edges = _topology_counts(obj)
    print(
        f"[final] vertices={final_vertices:,} faces={final_faces:,} "
        f"boundary_edges={boundary_edges:,} "
        f"non_manifold_edges={non_manifold_edges:,}",
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
        "algorithm": "Blender voxel remesh followed by collapse decimation",
        "voxel_size": args.voxel_size,
        "target_faces": args.target_faces,
        "fill_holes": args.fill_holes,
        "weld_distance": args.weld_distance,
        "remove_disconnected_threshold": args.remove_disconnected_threshold,
        "source_vertices": source_vertices,
        "source_faces": source_faces,
        "repair": repair,
        "remesh_vertices": remesh_vertices,
        "remesh_polygons": remesh_polygons,
        "remesh_triangles": remesh_faces,
        "final_vertices": final_vertices,
        "final_faces": final_faces,
        "mesh_validation_changed_data": mesh_was_invalid,
        "boundary_edges": boundary_edges,
        "non_manifold_edges": non_manifold_edges,
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
