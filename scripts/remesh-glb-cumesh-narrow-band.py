#!/usr/bin/env python3
"""Rebuild a triangle mesh with TRELLIS.2's CuMesh narrow-band DC path.

This intentionally mirrors the geometry portion of
``o_voxel.postprocess.to_glb(remesh=True)`` without requiring an O-Voxel
attribute volume.  The output is an untextured, standard Y-up GLB suitable for
topology and silhouette comparisons.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cumesh
import numpy as np
import torch
import trimesh


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="scene", process=False)
    geometries = tuple(loaded.dump())
    if not geometries:
        raise RuntimeError(f"No triangle geometry in {path}")
    mesh = trimesh.util.concatenate(geometries)
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise RuntimeError(f"No triangle faces in {path}")
    return mesh


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--resolution", type=int, default=768)
    parser.add_argument("--band", type=float, default=1.0)
    parser.add_argument("--project-back", type=float, default=0.0)
    parser.add_argument("--target-faces", type=int, default=4_000_000)
    parser.add_argument("--initial-fill-perimeter", type=float, default=3e-2)
    parser.add_argument("--skip-initial-fill", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.resolution < 32 or args.resolution % 32 != 0:
        parser.error("--resolution must be at least 32 and divisible by 32")

    started = time.perf_counter()
    source = load_mesh(args.input)
    load_seconds = time.perf_counter() - started
    source_vertices = np.asarray(source.vertices, dtype=np.float32)
    source_faces = np.asarray(source.faces, dtype=np.int32)
    bounds = np.asarray(source.bounds, dtype=np.float32)

    vertices = torch.from_numpy(source_vertices).cuda()
    faces = torch.from_numpy(source_faces).cuda()
    torch.cuda.synchronize()

    clean_started = time.perf_counter()
    mesh = cumesh.CuMesh()
    mesh.init(vertices, faces)
    if not args.skip_initial_fill:
        mesh.fill_holes(max_hole_perimeter=args.initial_fill_perimeter)
    vertices, faces = mesh.read()
    torch.cuda.synchronize()
    clean_seconds = time.perf_counter() - clean_started

    center = torch.from_numpy(bounds.mean(axis=0)).to(device=vertices.device)
    extent = float((bounds[1] - bounds[0]).max())
    scale = (args.resolution + 3 * args.band) / args.resolution * extent

    bvh_started = time.perf_counter()
    bvh = cumesh.cuBVH(vertices, faces)
    torch.cuda.synchronize()
    bvh_seconds = time.perf_counter() - bvh_started

    remesh_started = time.perf_counter()
    out_vertices, out_faces = cumesh.remeshing.remesh_narrow_band_dc(
        vertices,
        faces,
        center=center,
        scale=scale,
        resolution=args.resolution,
        band=args.band,
        project_back=args.project_back,
        verbose=args.verbose,
        bvh=bvh,
    )
    torch.cuda.synchronize()
    remesh_seconds = time.perf_counter() - remesh_started
    remesh_vertices = int(out_vertices.shape[0])
    remesh_faces = int(out_faces.shape[0])

    simplify_started = time.perf_counter()
    if args.target_faces > 0 and remesh_faces > args.target_faces:
        mesh.init(out_vertices, out_faces)
        mesh.simplify(args.target_faces, verbose=args.verbose)
        out_vertices, out_faces = mesh.read()
        torch.cuda.synchronize()
    simplify_seconds = time.perf_counter() - simplify_started

    output_vertices = out_vertices.detach().cpu().numpy()
    output_faces = out_faces.detach().cpu().numpy()
    result = trimesh.Trimesh(
        vertices=output_vertices,
        faces=output_faces,
        process=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.export(args.output)

    report = {
        "input": str(args.input.resolve()),
        "output": str(args.output.resolve()),
        "resolution": args.resolution,
        "band": args.band,
        "project_back": args.project_back,
        "target_faces": args.target_faces,
        "initial_fill_perimeter": None if args.skip_initial_fill else args.initial_fill_perimeter,
        "source_vertices": int(len(source_vertices)),
        "source_faces": int(len(source_faces)),
        "remesh_vertices": remesh_vertices,
        "remesh_faces": remesh_faces,
        "output_vertices": int(len(output_vertices)),
        "output_faces": int(len(output_faces)),
        "bounds": bounds.tolist(),
        "center": center.detach().cpu().tolist(),
        "scale": scale,
        "timings_seconds": {
            "load": load_seconds,
            "initial_clean": clean_seconds,
            "bvh": bvh_seconds,
            "remesh": remesh_seconds,
            "simplify": simplify_seconds,
            "total": time.perf_counter() - started,
        },
    }
    report_path = args.output.with_suffix(args.output.suffix + ".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
