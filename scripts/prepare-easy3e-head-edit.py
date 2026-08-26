#!/usr/bin/env python3
"""Prepare a reproducible Easy3E local head-edit case.

The condition remains a whole-object RGBA image, while an ellipsoid in the
source model's coordinate system limits the editable 3D region to the head.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
import trimesh


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--alpha-mask", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--edit-name", default="head")
    parser.add_argument("--head-center-from-top", type=float, default=0.055)
    parser.add_argument("--head-radius-up", type=float, default=0.100)
    parser.add_argument("--head-radius-side", type=float, default=0.095)
    parser.add_argument("--head-radius-depth", type=float, default=0.110)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    model_dir = case_dir / "model"
    cond_dir = case_dir / "cond"
    mask_dir = case_dir / "mask"
    for directory in (model_dir, cond_dir, mask_dir):
        directory.mkdir(parents=True, exist_ok=True)

    model_link = model_dir / args.model.name
    if model_link.is_symlink() or model_link.exists():
        model_link.unlink()
    model_link.symlink_to(args.model.resolve())

    target = Image.open(args.target).convert("RGB")
    alpha = Image.open(args.alpha_mask).convert("L")
    if alpha.size != target.size:
        alpha = alpha.resize(target.size, Image.Resampling.NEAREST)
    rgba = target.copy()
    rgba.putalpha(alpha)
    cond_path = cond_dir / f"{args.edit_name}.png"
    rgba.save(cond_path)

    scene = trimesh.load(args.model, force="scene")
    bounds = np.asarray(scene.bounds, dtype=np.float64)
    extents = bounds[1] - bounds[0]
    up_axis = int(np.argmax(extents))
    remaining = [axis for axis in range(3) if axis != up_axis]
    depth_axis = min(remaining, key=lambda axis: extents[axis])
    side_axis = next(axis for axis in remaining if axis != depth_axis)
    height = float(extents[up_axis])

    center = (bounds[0] + bounds[1]) * 0.5
    center[up_axis] = bounds[1, up_axis] - args.head_center_from_top * height
    radii = np.empty(3, dtype=np.float64)
    radii[up_axis] = args.head_radius_up * height
    radii[side_axis] = args.head_radius_side * height
    radii[depth_axis] = args.head_radius_depth * height

    mask = trimesh.creation.icosphere(subdivisions=4, radius=1.0)
    mask.vertices = mask.vertices * radii + center
    mask_path = mask_dir / f"{args.edit_name}.glb"
    mask.export(mask_path)

    manifest = {
        "schema": "diffusion-editor.easy3e-head-edit-case",
        "schema_version": 1,
        "model": str(args.model.resolve()),
        "model_link": str(model_link),
        "target": str(args.target.resolve()),
        "target_rgba": str(cond_path),
        "alpha_mask": str(args.alpha_mask.resolve()),
        "mask_mesh": str(mask_path),
        "bounds": bounds.tolist(),
        "axes": {"up": up_axis, "side": side_axis, "depth": depth_axis},
        "head_ellipsoid": {"center": center.tolist(), "radii": radii.tolist()},
        "easy3e": {
            "condition_contract": "whole-object RGBA",
            "editable_region": "solid 3D ellipsoid around the head",
        },
    }
    with (case_dir / "case-manifest.json").open("w", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
