#!/usr/bin/env python3
"""Split non-manifold topology and fill only small boundary loops in a PLY."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-perimeter", type=float, default=0.001)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument(
        "--fill-mode",
        choices=("metric", "trivial"),
        default="metric",
        help="metric triangulates the boundary; trivial adds a centroid vertex",
    )
    parser.add_argument(
        "--fix-exact-degeneracies",
        action="store_true",
        help="repair only zero-area/infinite-aspect triangles after filling",
    )
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.max_perimeter <= 0:
        raise ValueError("max-perimeter must be positive")

    from meshlib import mrmeshpy as meshlib

    started = time.monotonic()
    mesh = meshlib.loadMesh(args.input)
    loaded_vertices = mesh.topology.numValidVerts()
    loaded_faces = mesh.topology.numValidFaces()
    initial_holes = mesh.topology.findHoleRepresentiveEdges()
    duplicated = meshlib.duplicateMultiHoleVertices(mesh)
    topology = mesh.topology
    holes = list(topology.findHoleRepresentiveEdges())
    perimeters = np.asarray(
        [meshlib.holePerimeter(topology, mesh.points, edge) for edge in holes],
        dtype=np.float64,
    )

    if perimeters.size:
        perimeter_summary = {
            "min": float(perimeters.min()),
            "p50": float(np.quantile(perimeters, 0.50)),
            "p90": float(np.quantile(perimeters, 0.90)),
            "p95": float(np.quantile(perimeters, 0.95)),
            "p99": float(np.quantile(perimeters, 0.99)),
            "max": float(perimeters.max()),
        }
    else:
        perimeter_summary = {}

    selected = meshlib.std_vector_Id_EdgeTag()
    for edge, perimeter in zip(holes, perimeters):
        if perimeter <= args.max_perimeter:
            selected.append(edge)
    print(
        f"[holes] initial={len(initial_holes):,} normalized={len(holes):,} "
        f"selected={len(selected):,} max_perimeter={args.max_perimeter}",
        flush=True,
    )

    if not args.analyze_only:
        if args.fill_mode == "trivial":
            for edge in selected:
                meshlib.fillHoleTrivially(mesh, edge)
        else:
            params = meshlib.FillHoleParams()
            params.metric = meshlib.getUniversalMetric(mesh)
            meshlib.fillHoles(mesh, selected, params)
    degenerate_region = meshlib.findDegenerateFaces(meshlib.MeshPart(mesh), 1e12)
    degenerate_faces_before_fix = degenerate_region.count()
    if args.fix_exact_degeneracies and degenerate_faces_before_fix:
        fix_params = meshlib.FixMeshDegeneraciesParams()
        fix_params.criticalTriAspectRatio = 1e12
        fix_params.tinyEdgeLength = 0.0
        fix_params.maxDeviation = 0.001
        fix_params.region = degenerate_region
        fix_params.mode = meshlib.FixMeshDegeneraciesParams.Mode.RemeshPatch
        fix_params.mimicPatch = True
        meshlib.fixMeshDegeneracies(mesh, fix_params)
    degenerate_faces_after_fix = meshlib.findDegenerateFaces(
        meshlib.MeshPart(mesh), 1e12
    ).count()
    remaining_holes = mesh.topology.findHoleRepresentiveEdges()
    final_vertices = mesh.topology.numValidVerts()
    final_faces = mesh.topology.numValidFaces()
    print(
        f"[filled] vertices={final_vertices:,} faces={final_faces:,} "
        f"remaining_holes={len(remaining_holes):,}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    meshlib.saveMesh(mesh, args.output)
    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "algorithm": "MeshLib topology normalization and selective fillHoles",
        "analyze_only": args.analyze_only,
        "fill_mode": args.fill_mode,
        "fix_exact_degeneracies": args.fix_exact_degeneracies,
        "max_perimeter": args.max_perimeter,
        "loaded_vertices": loaded_vertices,
        "loaded_faces": loaded_faces,
        "initial_holes": len(initial_holes),
        "duplicated_multi_hole_vertices": duplicated,
        "normalized_holes": len(holes),
        "perimeter_summary": perimeter_summary,
        "selected_holes": len(selected),
        "remaining_holes": len(remaining_holes),
        "degenerate_faces_before_fix": degenerate_faces_before_fix,
        "degenerate_faces_after_fix": degenerate_faces_after_fix,
        "final_vertices": final_vertices,
        "final_faces": final_faces,
        "elapsed_seconds": time.monotonic() - started,
        "coordinates": "unchanged from input PLY",
    }
    report_path = args.report or args.output.with_suffix(".json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
