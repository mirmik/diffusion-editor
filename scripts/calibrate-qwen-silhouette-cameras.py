#!/usr/bin/env python3
"""Calibrate bounded per-view cameras against Qwen body silhouettes."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import trimesh


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument(
        "--fixed-mesh",
        type=Path,
        help="Calibrate against this already-shaped mesh without renormalizing it.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--fov", type=float, default=35.0)
    parser.add_argument("--mask-threshold", type=float, default=18.0)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--distance-min", type=float, default=3.5)
    parser.add_argument("--distance-max", type=float, default=5.5)
    parser.add_argument("--distance-step", type=float, default=0.1)
    parser.add_argument("--azimuth-radius", type=float, default=20.0)
    parser.add_argument("--azimuth-step", type=float, default=5.0)
    parser.add_argument("--elevation-radius", type=float, default=15.0)
    parser.add_argument("--elevation-step", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def _values(low: float, high: float, step: float) -> list[float]:
    count = int(round((high - low) / step))
    return [low + index * step for index in range(count + 1)]


def per_view_iou_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, predicted.ndim))
    intersection = torch.sum(predicted * target, dim=dimensions)
    union = (
        torch.sum(predicted, dim=dimensions)
        + torch.sum(target, dim=dimensions)
        - intersection
    )
    return 1.0 - (intersection + 1.0e-6) / (union + 1.0e-6)


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    scripts_dir = Path(__file__).parent
    base = _load_module(
        scripts_dir / "fit-qwen-silhouette-mesh.py", "silhouette_fit",
    )
    macro = _load_module(
        scripts_dir / "fit-qwen-makehuman-macros.py", "macro_fit",
    )
    records, targets = base.load_views(
        args.image_dir, args.image_size, args.mask_threshold, args.mask_dir,
    )
    targets = targets.to(device)
    if args.fixed_mesh is None:
        vertices_np, faces_np, normalization = base.load_template(
            args.template, 0, None, "body", "y", 0,
        )
        target_arrays = macro.load_macro_targets(
            args.targets_dir, len(vertices_np), normalization["scale"],
        )
        target_tensors = {
            key: torch.from_numpy(value).to(device)
            for key, value in target_arrays.items()
        }
        vertices = macro.shaped_vertices(
            torch.from_numpy(vertices_np).to(device),
            target_tensors,
            torch.full((4,), 0.5, device=device),
        )
    else:
        mesh = trimesh.load(args.fixed_mesh, process=False, force="mesh")
        vertices = torch.from_numpy(
            np.asarray(mesh.vertices, dtype=np.float32),
        ).to(device)
        faces_np = np.asarray(mesh.faces, dtype=np.int64)
    faces = torch.from_numpy(faces_np).to(device)
    render_args = SimpleNamespace(
        image_size=args.image_size,
        distance=4.0,
        fov=args.fov,
        hard_forward_silhouette=True,
    )
    distance_values = _values(
        args.distance_min, args.distance_max, args.distance_step,
    )
    azimuth_offsets = _values(
        -args.azimuth_radius, args.azimuth_radius, args.azimuth_step,
    )
    elevation_offsets = _values(
        -args.elevation_radius, args.elevation_radius, args.elevation_step,
    )
    calibrated_views = []
    for view_index, record in enumerate(records):
        candidates = [
            {
                **record,
                "azimuth_degrees": record["azimuth_degrees"] + azimuth_offset,
                "elevation_degrees": (
                    record["elevation_degrees"] + elevation_offset
                ),
                "distance": distance,
            }
            for azimuth_offset, elevation_offset, distance in itertools.product(
                azimuth_offsets, elevation_offsets, distance_values,
            )
        ]
        best_loss = float("inf")
        best_candidate = None
        for start in range(0, len(candidates), args.batch_size):
            batch = candidates[start:start + args.batch_size]
            indices = torch.arange(len(batch))
            with torch.no_grad():
                rendered = base.render_views(
                    vertices, faces, batch, indices, render_args,
                )
                target = targets[view_index:view_index + 1].expand_as(rendered)
                losses = per_view_iou_loss(rendered, target)
                batch_loss, batch_index = torch.min(losses, dim=0)
            if float(batch_loss) < best_loss:
                best_loss = float(batch_loss)
                best_candidate = batch[int(batch_index)]
        assert best_candidate is not None
        calibrated = {
            "name": record["name"],
            "nominal_azimuth_degrees": record["azimuth_degrees"],
            "nominal_elevation_degrees": record["elevation_degrees"],
            "calibrated_azimuth_degrees": best_candidate["azimuth_degrees"],
            "calibrated_elevation_degrees": best_candidate["elevation_degrees"],
            "calibrated_distance": best_candidate["distance"],
            "silhouette_loss": best_loss,
            "iou": 1.0 - best_loss,
        }
        calibrated_views.append(calibrated)
        print(json.dumps(calibrated), flush=True)
    calibrated_records = []
    for record, calibrated in zip(records, calibrated_views):
        calibrated_records.append({
            **record,
            "azimuth_degrees": calibrated["calibrated_azimuth_degrees"],
            "elevation_degrees": calibrated["calibrated_elevation_degrees"],
            "distance": calibrated["calibrated_distance"],
        })
    with torch.no_grad():
        rendered = base.render_views(
            vertices,
            faces,
            calibrated_records,
            torch.arange(len(calibrated_records)),
            render_args,
        )
        full_loss = float(torch.mean(per_view_iou_loss(rendered, targets)))
    report = {
        "schema": "diffusion-editor.qwen-silhouette-camera-calibration",
        "schema_version": 1,
        "image_dir": str(args.image_dir.resolve()),
        "template": str(args.template.resolve()),
        "targets_dir": str(args.targets_dir.resolve()),
        "fixed_mesh": (
            None if args.fixed_mesh is None else str(args.fixed_mesh.resolve())
        ),
        "mask_dir": (
            None if args.mask_dir is None else str(args.mask_dir.resolve())
        ),
        "search": {
            "distance": [args.distance_min, args.distance_max, args.distance_step],
            "azimuth_offset": [
                -args.azimuth_radius, args.azimuth_radius, args.azimuth_step,
            ],
            "elevation_offset": [
                -args.elevation_radius, args.elevation_radius,
                args.elevation_step,
            ],
        },
        "mean_silhouette_loss": full_loss,
        "mean_iou": 1.0 - full_loss,
        "views": calibrated_views,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    comparison = args.output.with_name(args.output.stem + "-comparison.png")
    base._comparison_sheet(
        targets.cpu().numpy(),
        rendered.cpu().numpy(),
        calibrated_records,
        comparison,
    )
    print(json.dumps({
        "mean_iou": report["mean_iou"],
        "output": str(args.output),
        "comparison": str(comparison),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
