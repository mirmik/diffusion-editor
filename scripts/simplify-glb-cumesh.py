#!/usr/bin/env python3
"""Simplify triangle geometry with TRELLIS.2's GPU CuMesh implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cumesh
import numpy as np
import torch
import trimesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--target-faces", type=int, required=True)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.target_faces <= 0:
        parser.error("--target-faces must be positive")

    started = time.perf_counter()
    loaded = trimesh.load(args.input, force="scene", process=False)
    geometries = tuple(loaded.dump())
    if not geometries:
        raise RuntimeError(f"No triangle geometry in {args.input}")
    source = trimesh.util.concatenate(geometries)
    source_vertices = np.ascontiguousarray(source.vertices, dtype=np.float32)
    source_faces = np.ascontiguousarray(source.faces, dtype=np.int32)
    if args.target_faces >= len(source_faces):
        raise ValueError("target must be smaller than the source face count")
    load_seconds = time.perf_counter() - started

    vertices = torch.from_numpy(source_vertices).cuda()
    faces = torch.from_numpy(source_faces).cuda()
    torch.cuda.synchronize()

    simplify_started = time.perf_counter()
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    mesh.simplify(args.target_faces, verbose=args.verbose)
    out_vertices, out_faces = mesh.read()
    torch.cuda.synchronize()
    simplify_seconds = time.perf_counter() - simplify_started

    result = trimesh.Trimesh(
        vertices=out_vertices.detach().cpu().numpy(),
        faces=out_faces.detach().cpu().numpy(),
        process=False,
    )
    result.remove_unreferenced_vertices()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.export(args.output)

    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "algorithm": "TRELLIS.2 CuMesh GPU simplify",
        "target_faces": args.target_faces,
        "source_vertices": int(len(source_vertices)),
        "source_faces": int(len(source_faces)),
        "output_vertices": int(len(result.vertices)),
        "output_faces": int(len(result.faces)),
        "timings_seconds": {
            "load": load_seconds,
            "simplify": simplify_seconds,
            "total": time.perf_counter() - started,
        },
        "coordinates": "unchanged; standard glTF right-handed Y-up",
    }
    args.output.with_suffix(args.output.suffix + ".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
