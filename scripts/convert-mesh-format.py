#!/usr/bin/env python3
"""Convert triangle geometry between mesh formats without changing coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--remove-duplicate-faces", action="store_true")
    parser.add_argument(
        "--remove-isolated-double-faces",
        action="store_true",
        help="remove both faces from coincident two-face groups",
    )
    args = parser.parse_args()

    loaded = trimesh.load(args.input, force="scene", process=False)
    geometries = tuple(loaded.dump())
    if not geometries:
        raise RuntimeError(f"No triangle geometry in {args.input}")
    mesh = trimesh.util.concatenate(geometries)
    source_faces = len(mesh.faces)
    if args.remove_isolated_double_faces:
        groups = trimesh.grouping.group_rows(
            np.sort(mesh.faces, axis=1), require_count=2
        )
        keep = np.ones(len(mesh.faces), dtype=bool)
        if len(groups):
            keep[np.asarray(groups).reshape(-1)] = False
        mesh.update_faces(keep)
        mesh.remove_unreferenced_vertices()
    elif args.remove_duplicate_faces:
        mesh.update_faces(mesh.unique_faces())
        mesh.remove_unreferenced_vertices()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(args.output)
    print(
        f"{args.input} -> {args.output}: "
        f"{len(mesh.vertices):,} vertices, {len(mesh.faces):,} faces; "
        f"removed duplicate faces={source_faces - len(mesh.faces):,}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
