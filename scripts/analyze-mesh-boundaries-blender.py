#!/usr/bin/env python3
"""Report boundary components and non-manifold edges of a GLB mesh."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys
import time

import bmesh
import bpy


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--top", type=int, default=30)
    return parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])


def _join_imported_meshes() -> bpy.types.Object:
    objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not objects:
        raise RuntimeError("GLB contained no mesh objects")
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    if len(objects) > 1:
        bpy.ops.object.join()
    obj = bpy.context.view_layer.objects.active
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    return obj


def main() -> int:
    args = _arguments()
    started = time.monotonic()
    bpy.ops.import_scene.gltf(filepath=str(args.input.resolve()))
    obj = _join_imported_meshes()
    mesh = bmesh.new()
    mesh.from_mesh(obj.data)
    mesh.verts.ensure_lookup_table()
    mesh.edges.ensure_lookup_table()

    boundary_edges = [edge for edge in mesh.edges if edge.is_boundary]
    visited = bytearray(len(mesh.edges))
    components = []
    for seed in boundary_edges:
        if visited[seed.index]:
            continue
        visited[seed.index] = 1
        queue = deque([seed])
        edge_count = 0
        perimeter = 0.0
        vertex_ids: set[int] = set()
        while queue:
            edge = queue.popleft()
            edge_count += 1
            perimeter += (edge.verts[0].co - edge.verts[1].co).length
            for vertex in edge.verts:
                vertex_ids.add(vertex.index)
                for neighbor in vertex.link_edges:
                    if neighbor.is_boundary and not visited[neighbor.index]:
                        visited[neighbor.index] = 1
                        queue.append(neighbor)

        vertices = [mesh.verts[index] for index in vertex_ids]
        boundary_degrees = [
            sum(edge.is_boundary for edge in vertex.link_edges)
            for vertex in vertices
        ]
        minimum = [min(vertex.co[axis] for vertex in vertices) for axis in range(3)]
        maximum = [max(vertex.co[axis] for vertex in vertices) for axis in range(3)]
        center = [(low + high) * 0.5 for low, high in zip(minimum, maximum)]
        components.append({
            "edges": edge_count,
            "vertices": len(vertices),
            "perimeter": perimeter,
            "closed_loop": all(degree == 2 for degree in boundary_degrees),
            "branch_vertices": sum(degree != 2 for degree in boundary_degrees),
            "bounds_min": minimum,
            "bounds_max": maximum,
            "center": center,
        })

    components.sort(key=lambda item: (item["edges"], item["perimeter"]), reverse=True)
    multi_face_edges = [
        edge
        for edge in mesh.edges
        if not edge.is_manifold and not edge.is_boundary and not edge.is_wire
    ]
    report = {
        "input": str(args.input.resolve()),
        "coordinates_after_blender_import": "right-handed Z-up",
        "vertices": len(mesh.verts),
        "edges": len(mesh.edges),
        "faces": len(mesh.faces),
        "boundary_edges": len(boundary_edges),
        "boundary_components": len(components),
        "closed_boundary_loops": sum(
            component["closed_loop"] for component in components
        ),
        "branched_boundary_components": sum(
            not component["closed_loop"] for component in components
        ),
        "multi_face_nonmanifold_edges": len(multi_face_edges),
        "largest_boundary_components": components[: args.top],
        "elapsed_seconds": time.monotonic() - started,
    }
    mesh.free()
    encoded = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n", encoding="utf-8")
    print(encoded, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
