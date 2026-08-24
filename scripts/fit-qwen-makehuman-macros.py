#!/usr/bin/env python3
"""Fit native MakeHuman macro parameters to nine Qwen body silhouettes."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import torch


MUSCLE_LABELS = ("minmuscle", "averagemuscle", "maxmuscle")
WEIGHT_LABELS = ("minweight", "averageweight", "maxweight")
HEIGHT_LABELS = ("minheight", "maxheight")
PROPORTION_LABELS = ("uncommonproportions", "idealproportions")
PARAMETER_NAMES = ("muscle", "weight", "height", "body_proportions")


def _base_module():
    path = Path(__file__).with_name("fit-qwen-silhouette-mesh.py")
    spec = importlib.util.spec_from_file_location("qwen_silhouette_fit", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load fitting helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--camera-report",
        type=Path,
        help="Use per-view calibrated azimuth, elevation, and distance.",
    )
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--parameter-prior-weight", type=float, default=0.001)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--distance", type=float, default=4.0)
    parser.add_argument("--fov", type=float, default=35.0)
    parser.add_argument("--mask-threshold", type=float, default=18.0)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--hard-forward-silhouette", action="store_true")
    parser.add_argument("--seed", type=int, default=20260823)
    return parser.parse_args()


def load_target(path: Path, vertex_count: int, scale: float) -> np.ndarray:
    result = np.zeros((vertex_count, 3), dtype=np.float32)
    if not path.is_file():
        raise SystemExit(f"missing MakeHuman target: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields or fields[0].startswith("#"):
            continue
        if len(fields) < 4:
            raise SystemExit(f"malformed target row in {path}: {line!r}")
        index = int(fields[0])
        if index < vertex_count:
            result[index] = np.asarray(fields[1:4], dtype=np.float32) * scale
    return result


def load_macro_targets(
    root: Path,
    vertex_count: int,
    scale: float,
) -> dict[str, np.ndarray]:
    universal = np.zeros((3, 3, vertex_count, 3), dtype=np.float32)
    height = np.zeros((3, 3, 2, vertex_count, 3), dtype=np.float32)
    proportions = np.zeros((3, 3, 2, vertex_count, 3), dtype=np.float32)
    for muscle_index, muscle in enumerate(MUSCLE_LABELS):
        for weight_index, weight in enumerate(WEIGHT_LABELS):
            prefix = f"male-young-{muscle}-{weight}"
            universal[muscle_index, weight_index] = load_target(
                root / f"universal-{prefix}.target", vertex_count, scale,
            )
            for extreme_index, extreme in enumerate(HEIGHT_LABELS):
                height[muscle_index, weight_index, extreme_index] = load_target(
                    root / "height" / f"{prefix}-{extreme}.target",
                    vertex_count,
                    scale,
                )
            for extreme_index, extreme in enumerate(PROPORTION_LABELS):
                proportions[
                    muscle_index, weight_index, extreme_index
                ] = load_target(
                    root / "proportions" / f"{prefix}-{extreme}.target",
                    vertex_count,
                    scale,
                )
    return {
        "race": load_target(
            root / "caucasian-male-young.target", vertex_count, scale,
        ),
        "universal": universal,
        "height": height,
        "proportions": proportions,
    }


def triangular_weights(value: torch.Tensor) -> torch.Tensor:
    low = torch.clamp(1.0 - 2.0 * value, min=0.0)
    high = torch.clamp(2.0 * value - 1.0, min=0.0)
    middle = 1.0 - low - high
    return torch.stack((low, middle, high))


def extreme_weights(value: torch.Tensor) -> torch.Tensor:
    low = torch.clamp(1.0 - 2.0 * value, min=0.0)
    high = torch.clamp(2.0 * value - 1.0, min=0.0)
    return torch.stack((low, high))


def shaped_vertices(
    base_vertices: torch.Tensor,
    targets: dict[str, torch.Tensor],
    parameters: torch.Tensor,
) -> torch.Tensor:
    muscle, weight, height, proportions = parameters
    muscle_weights = triangular_weights(muscle)
    weight_weights = triangular_weights(weight)
    shape = base_vertices + targets["race"]
    shape = shape + torch.einsum(
        "i,j,ijvc->vc",
        muscle_weights,
        weight_weights,
        targets["universal"],
    )
    shape = shape + torch.einsum(
        "i,j,k,ijkvc->vc",
        muscle_weights,
        weight_weights,
        extreme_weights(height),
        targets["height"],
    )
    shape = shape + torch.einsum(
        "i,j,k,ijkvc->vc",
        muscle_weights,
        weight_weights,
        extreme_weights(proportions),
        targets["proportions"],
    )
    return shape


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = _base_module()
    records, target_masks = base.load_views(
        args.image_dir, args.image_size, args.mask_threshold, args.mask_dir,
    )
    if args.camera_report is not None:
        camera_report = json.loads(args.camera_report.read_text(encoding="utf-8"))
        calibrated = {view["name"]: view for view in camera_report["views"]}
        for record in records:
            view = calibrated.get(record["name"])
            if view is None:
                raise SystemExit(
                    f"camera report has no view for {record['name']}"
                )
            record["azimuth_degrees"] = view["calibrated_azimuth_degrees"]
            record["elevation_degrees"] = view["calibrated_elevation_degrees"]
            record["distance"] = view["calibrated_distance"]
    vertices_np, faces_np, normalization = base.load_template(
        args.template, 0, None, "body", "y", 0,
    )
    macro_np = load_macro_targets(
        args.targets_dir, len(vertices_np), normalization["scale"],
    )
    vertices = torch.from_numpy(vertices_np).to(device)
    faces = torch.from_numpy(faces_np).to(device)
    target_masks = target_masks.to(device)
    macro_targets = {
        name: torch.from_numpy(value).to(device)
        for name, value in macro_np.items()
    }
    logits = torch.zeros(len(PARAMETER_NAMES), device=device, requires_grad=True)
    optimizer = torch.optim.Adam((logits,), lr=args.learning_rate)
    all_indices = torch.arange(len(records))
    history = []
    started = time.monotonic()

    def current_parameters() -> torch.Tensor:
        return torch.sigmoid(logits)

    def current_vertices() -> torch.Tensor:
        return shaped_vertices(
            vertices, macro_targets, current_parameters(),
        )

    with torch.no_grad():
        initial_vertices = current_vertices()
        initial = base.render_views(
            initial_vertices, faces, records, all_indices, args,
        )
        initial_loss = float(base.silhouette_loss(
            initial, target_masks, "iou",
        ))
    best_loss = initial_loss
    best_iteration = 0
    best_vertices = initial_vertices.detach().clone()
    best_parameters = current_parameters().detach().clone()

    for iteration in range(args.iterations):
        parameters = current_parameters()
        shaped = shaped_vertices(vertices, macro_targets, parameters)
        predicted = base.render_views(
            shaped, faces, records, all_indices, args,
        )
        silhouette = base.silhouette_loss(predicted, target_masks, "iou")
        prior = torch.mean((parameters - 0.5) ** 2)
        loss = silhouette + args.parameter_prior_weight * prior
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        should_evaluate = (
            iteration == 0
            or (iteration + 1) % args.evaluation_interval == 0
            or iteration + 1 == args.iterations
        )
        if should_evaluate:
            with torch.no_grad():
                evaluated_vertices = current_vertices()
                evaluated = base.render_views(
                    evaluated_vertices, faces, records, all_indices, args,
                )
                full_loss = float(base.silhouette_loss(
                    evaluated, target_masks, "iou",
                ))
                full_mse = float(torch.mean((evaluated - target_masks) ** 2))
                values = current_parameters().detach()
            entry = {
                "iteration": iteration + 1,
                "total": float(loss.detach()),
                "silhouette_loss": float(silhouette.detach()),
                "parameter_prior": float(prior.detach()),
                "full_silhouette_loss": full_loss,
                "full_silhouette_mse": full_mse,
                "parameters": {
                    name: float(value)
                    for name, value in zip(PARAMETER_NAMES, values)
                },
            }
            history.append(entry)
            if full_loss < best_loss:
                best_loss = full_loss
                best_iteration = iteration + 1
                best_vertices = evaluated_vertices.detach().clone()
                best_parameters = values.clone()
            print(json.dumps(entry), flush=True)

    with torch.no_grad():
        final = base.render_views(
            best_vertices, faces, records, all_indices, args,
        ).cpu().numpy()
    targets_np = target_masks.cpu().numpy()
    initial_np = initial.cpu().numpy()
    base._comparison_sheet(
        targets_np, initial_np, records, args.output_dir / "initial.png",
    )
    base._comparison_sheet(
        targets_np, final, records, args.output_dir / "final.png",
    )
    base._export_mesh(
        args.output_dir / "template.ply", vertices_np, faces_np,
    )
    base._export_mesh(
        args.output_dir / "fitted.ply", best_vertices.cpu().numpy(), faces_np,
    )
    parameter_values = {
        name: float(value)
        for name, value in zip(PARAMETER_NAMES, best_parameters.cpu())
    }
    mhm_lines = [
        "version v1.2.0",
        "name qwen_body_fit",
        "modifier macrodetails/Gender 1.000000",
        "modifier macrodetails/Age 0.500000",
        "modifier macrodetails/African 0.000000",
        "modifier macrodetails/Asian 0.000000",
        "modifier macrodetails/Caucasian 1.000000",
        f"modifier macrodetails-universal/Muscle {parameter_values['muscle']:.6f}",
        f"modifier macrodetails-universal/Weight {parameter_values['weight']:.6f}",
        f"modifier macrodetails-height/Height {parameter_values['height']:.6f}",
        (
            "modifier macrodetails-proportions/BodyProportions "
            f"{parameter_values['body_proportions']:.6f}"
        ),
        "skeleton default.mhskel",
    ]
    (args.output_dir / "fitted.mhm").write_text(
        "\n".join(mhm_lines) + "\n", encoding="utf-8",
    )
    report = {
        "schema": "diffusion-editor.qwen-makehuman-macro-fit",
        "schema_version": 1,
        "algorithm": "native MakeHuman macro targets plus differentiable rendering",
        "image_dir": str(args.image_dir.resolve()),
        "template": str(args.template.resolve()),
        "targets_dir": str(args.targets_dir.resolve()),
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "fixed_macros": {
            "gender": 1.0,
            "age": 0.5,
            "african": 0.0,
            "asian": 0.0,
            "caucasian": 1.0,
        },
        "best_parameters": parameter_values,
        "best_iteration": best_iteration,
        "initial_silhouette_loss": initial_loss,
        "final_silhouette_loss": float(base.silhouette_loss(
            torch.from_numpy(final).to(device), target_masks, "iou",
        )),
        "initial_iou": 1.0 - initial_loss,
        "final_iou": 1.0 - float(base.silhouette_loss(
            torch.from_numpy(final).to(device), target_masks, "iou",
        )),
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        key: report[key]
        for key in (
            "best_iteration", "initial_iou", "final_iou",
            "best_parameters", "elapsed_seconds",
        )
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
