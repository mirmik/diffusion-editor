#!/usr/bin/env python3
"""Reduce an over-tessellated GLB with fast quadric edge collapse.

This is a geometry-preserving simplification pass, not a voxel remesh. It
first removes vertices that are not referenced by any triangle, then collapses
low-error edges until the requested triangle count is reached.
"""

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
    parser.add_argument("--target-faces", type=int, required=True)
    parser.add_argument(
        "--aggressiveness",
        type=float,
        default=5.0,
        help="0 favors quality, 10 favors speed; default: 5",
    )
    parser.add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.target_faces <= 0:
        raise ValueError("target-faces must be positive")
    if not 0.0 <= args.aggressiveness <= 10.0:
        raise ValueError("aggressiveness must be in [0, 10]")

    import fast_simplification
    import trimesh

    started = time.monotonic()
    mesh = trimesh.load(args.input, force="mesh", process=False)
    source_vertices = len(mesh.vertices)
    source_faces = len(mesh.faces)
    source_bounds = np.asarray(mesh.bounds).tolist()
    if args.target_faces >= source_faces:
        raise ValueError(
            f"target-faces ({args.target_faces}) must be below input face "
            f"count ({source_faces})"
        )

    mesh.remove_unreferenced_vertices()
    compact_vertices = len(mesh.vertices)
    compact_faces = len(mesh.faces)
    print(
        f"[compact] vertices={source_vertices:,} -> {compact_vertices:,}; "
        f"faces={compact_faces:,}",
        flush=True,
    )

    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float64)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    simplified_vertices, simplified_faces = fast_simplification.simplify(
        vertices,
        faces,
        target_count=args.target_faces,
        agg=args.aggressiveness,
        verbose=True,
    )
    result = trimesh.Trimesh(
        vertices=simplified_vertices,
        faces=simplified_faces,
        process=False,
    )
    result.remove_unreferenced_vertices()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.export(args.output)

    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "algorithm": "fast-simplification quadric edge collapse",
        "aggressiveness": args.aggressiveness,
        "target_faces": args.target_faces,
        "source_vertices_in_file": source_vertices,
        "source_used_vertices": compact_vertices,
        "source_unreferenced_vertices_removed": (
            source_vertices - compact_vertices
        ),
        "source_faces": source_faces,
        "output_vertices": int(len(result.vertices)),
        "output_faces": int(len(result.faces)),
        "source_bounds": source_bounds,
        "output_bounds": np.asarray(result.bounds).tolist(),
        "face_reduction_ratio": 1.0 - len(result.faces) / source_faces,
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
