#!/usr/bin/env python3
"""Search native MakeHuman macro parameters using the true rasterized IoU."""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import torch


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
    parser.add_argument("--camera-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--grid-steps", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--fov", type=float, default=35.0)
    parser.add_argument("--mask-threshold", type=float, default=18.0)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.grid_steps < 2:
        raise SystemExit("--grid-steps must be at least 2")
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
    records, target_masks = base.load_views(
        args.image_dir, args.image_size, args.mask_threshold, args.mask_dir,
    )
    camera_report = json.loads(args.camera_report.read_text(encoding="utf-8"))
    calibrated = {view["name"]: view for view in camera_report["views"]}
    for record in records:
        view = calibrated[record["name"]]
        record["azimuth_degrees"] = view["calibrated_azimuth_degrees"]
        record["elevation_degrees"] = view["calibrated_elevation_degrees"]
        record["distance"] = view["calibrated_distance"]

    vertices_np, faces_np, normalization = base.load_template(
        args.template, 0, None, "body", "y", 0,
    )
    target_arrays = macro.load_macro_targets(
        args.targets_dir, len(vertices_np), normalization["scale"],
    )
    vertices = torch.from_numpy(vertices_np).to(device)
    faces = torch.from_numpy(faces_np).to(device)
    target_masks = target_masks.to(device)
    target_tensors = {
        key: torch.from_numpy(value).to(device)
        for key, value in target_arrays.items()
    }
    render_args = SimpleNamespace(
        image_size=args.image_size,
        distance=4.0,
        fov=args.fov,
        hard_forward_silhouette=True,
    )
    indices = torch.arange(len(records))
    values = torch.linspace(0.0, 1.0, args.grid_steps, device=device)
    best_loss = float("inf")
    best_parameters = None
    best_vertices = None
    candidates = list(itertools.product(range(args.grid_steps), repeat=4))
    with torch.no_grad():
        for candidate_index, candidate in enumerate(candidates, start=1):
            parameters = values[torch.tensor(candidate, device=device)]
            shaped = macro.shaped_vertices(vertices, target_tensors, parameters)
            rendered = base.render_views(
                shaped, faces, records, indices, render_args,
            )
            loss = float(base.silhouette_loss(
                rendered, target_masks, "iou",
            ))
            if loss < best_loss:
                best_loss = loss
                best_parameters = parameters.clone()
                best_vertices = shaped.clone()
                print(json.dumps({
                    "candidate": candidate_index,
                    "iou": 1.0 - best_loss,
                    "parameters": {
                        name: float(value)
                        for name, value in zip(
                            macro.PARAMETER_NAMES, best_parameters,
                        )
                    },
                }), flush=True)

    assert best_parameters is not None and best_vertices is not None
    with torch.no_grad():
        best_render = base.render_views(
            best_vertices, faces, records, indices, render_args,
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base._comparison_sheet(
        target_masks.cpu().numpy(),
        best_render.cpu().numpy(),
        records,
        args.output_dir / "best.png",
    )
    base._export_mesh(
        args.output_dir / "fitted.ply",
        best_vertices.cpu().numpy(),
        faces_np,
    )
    parameter_values = {
        name: float(value)
        for name, value in zip(macro.PARAMETER_NAMES, best_parameters.cpu())
    }
    report = {
        "schema": "diffusion-editor.qwen-makehuman-macro-grid-search",
        "schema_version": 1,
        "grid_steps": args.grid_steps,
        "candidate_count": len(candidates),
        "best_iou": 1.0 - best_loss,
        "best_parameters": parameter_values,
        "camera_report": str(args.camera_report.resolve()),
        "mask_dir": (
            None if args.mask_dir is None else str(args.mask_dir.resolve())
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
