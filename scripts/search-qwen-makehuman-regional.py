#!/usr/bin/env python3
"""Coordinate-search native MakeHuman macro and regional body targets."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


MACRO_NAMES = ("muscle", "weight", "height", "body_proportions")
REGIONAL_SPECS = {
    "head_width": ("head/head-scale-horiz",),
    "head_height": ("head/head-scale-vert",),
    "head_depth": ("head/head-scale-depth",),
    "neck_width": ("neck/neck-scale-horiz",),
    "torso_width": ("torso/torso-scale-horiz",),
    "torso_depth": ("torso/torso-scale-depth",),
    "torso_height": ("torso/torso-scale-vert",),
    "torso_vshape": ("torso/torso-vshape",),
    "hip_width": ("hip/hip-scale-horiz",),
    "upperarm_thickness": (
        "armslegs/l-upperarm-scale-horiz",
        "armslegs/l-upperarm-scale-depth",
        "armslegs/r-upperarm-scale-horiz",
        "armslegs/r-upperarm-scale-depth",
    ),
    "upperarm_length": (
        "armslegs/l-upperarm-scale-vert",
        "armslegs/r-upperarm-scale-vert",
    ),
    "lowerarm_thickness": (
        "armslegs/l-lowerarm-scale-horiz",
        "armslegs/l-lowerarm-scale-depth",
        "armslegs/r-lowerarm-scale-horiz",
        "armslegs/r-lowerarm-scale-depth",
    ),
    "lowerarm_length": (
        "armslegs/l-lowerarm-scale-vert",
        "armslegs/r-lowerarm-scale-vert",
    ),
    "upperleg_thickness": (
        "armslegs/l-upperleg-scale-horiz",
        "armslegs/l-upperleg-scale-depth",
        "armslegs/r-upperleg-scale-horiz",
        "armslegs/r-upperleg-scale-depth",
    ),
    "lowerleg_thickness": (
        "armslegs/l-lowerleg-scale-horiz",
        "armslegs/l-lowerleg-scale-depth",
        "armslegs/r-lowerleg-scale-horiz",
        "armslegs/r-lowerleg-scale-depth",
    ),
    "upperlegs_height": ("armslegs/upperlegs-height",),
    "lowerlegs_height": ("armslegs/lowerlegs-height",),
    "hand_scale": ("armslegs/l-hand-scale", "armslegs/r-hand-scale"),
    "foot_scale": ("armslegs/l-foot-scale", "armslegs/r-foot-scale"),
}


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
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument(
        "--initial-report",
        type=Path,
        help="Continue from parameters in an earlier regional-search report.",
    )
    parser.add_argument(
        "--initial-step",
        type=float,
        default=1.0,
        help="Coordinate-search step for the first pass; halves each pass.",
    )
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--fov", type=float, default=35.0)
    parser.add_argument("--mask-threshold", type=float, default=18.0)
    parser.add_argument("--mask-dir", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser.parse_args()


def load_regional_targets(
    root: Path,
    macro,
    vertex_count: int,
    scale: float,
) -> tuple[torch.Tensor, list[list[str]]]:
    arrays = []
    modifier_names = []
    for paths in REGIONAL_SPECS.values():
        negative = np.zeros((vertex_count, 3), dtype=np.float32)
        positive = np.zeros((vertex_count, 3), dtype=np.float32)
        mhm_names = []
        for relative in paths:
            negative += macro.load_target(
                root / f"{relative}-decr.target", vertex_count, scale,
            )
            positive += macro.load_target(
                root / f"{relative}-incr.target", vertex_count, scale,
            )
            mhm_names.append(relative)
        arrays.append(np.stack((negative, positive)))
        modifier_names.append(mhm_names)
    return torch.from_numpy(np.stack(arrays)), modifier_names


def apply_regional(
    vertices: torch.Tensor,
    targets: torch.Tensor,
    values: torch.Tensor,
) -> torch.Tensor:
    negative_weights = torch.clamp(-values, min=0.0)
    positive_weights = torch.clamp(values, min=0.0)
    return vertices + torch.einsum(
        "ik,ikvc->vc",
        torch.stack((negative_weights, positive_weights), dim=1),
        targets,
    )


def main() -> int:
    args = parse_args()
    if args.passes < 1:
        raise SystemExit("--passes must be positive")
    if not 0.0 < args.initial_step <= 1.0:
        raise SystemExit("--initial-step must be in (0, 1]")
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
    vertex_count = len(vertices_np)
    macro_arrays = macro.load_macro_targets(
        args.targets_dir / "macrodetails", vertex_count, normalization["scale"],
    )
    regional, modifier_names = load_regional_targets(
        args.targets_dir, macro, vertex_count, normalization["scale"],
    )
    vertices = torch.from_numpy(vertices_np).to(device)
    faces = torch.from_numpy(faces_np).to(device)
    targets = targets.to(device)
    macro_targets = {
        key: torch.from_numpy(value).to(device)
        for key, value in macro_arrays.items()
    }
    regional = regional.to(device)
    render_args = SimpleNamespace(
        image_size=args.image_size,
        distance=4.0,
        fov=args.fov,
        hard_forward_silhouette=True,
    )
    indices = torch.arange(len(records))
    parameter_names = list(MACRO_NAMES) + list(REGIONAL_SPECS)
    parameters = torch.zeros(len(parameter_names), device=device)
    parameters[:4] = torch.tensor((0.5, 0.5, 0.5, 0.5), device=device)
    if args.initial_report is not None:
        initial_report = json.loads(
            args.initial_report.read_text(encoding="utf-8")
        )
        initial_values = initial_report.get("parameters")
        if not isinstance(initial_values, dict):
            raise SystemExit("initial report has no parameter mapping")
        missing = [name for name in parameter_names if name not in initial_values]
        if missing:
            raise SystemExit(
                "initial report is missing parameters: " + ", ".join(missing)
            )
        parameters[:] = torch.tensor(
            [float(initial_values[name]) for name in parameter_names],
            device=device,
        )

    def shape(values: torch.Tensor) -> torch.Tensor:
        macro_shape = macro.shaped_vertices(
            vertices, macro_targets, values[:4],
        )
        return apply_regional(macro_shape, regional, values[4:])

    def evaluate(values: torch.Tensor):
        shaped = shape(values)
        rendered = base.render_views(
            shaped, faces, records, indices, render_args,
        )
        loss = base.silhouette_loss(rendered, targets, "iou")
        return float(loss), shaped, rendered

    with torch.no_grad():
        best_loss, best_vertices, best_render = evaluate(parameters)
        initial_iou = 1.0 - best_loss
        print(json.dumps({"initial_iou": initial_iou}), flush=True)
        for pass_index in range(args.passes):
            step = args.initial_step / (2 ** pass_index)
            for parameter_index, parameter_name in enumerate(parameter_names):
                center = float(parameters[parameter_index])
                if parameter_index < 4:
                    candidates = np.clip(
                        [center - step, center - step / 2, center,
                         center + step / 2, center + step],
                        0.0,
                        1.0,
                    )
                else:
                    candidates = np.clip(
                        [center - step, center - step / 2, center,
                         center + step / 2, center + step],
                        -1.0,
                        1.0,
                    )
                for candidate in sorted(set(float(value) for value in candidates)):
                    trial = parameters.clone()
                    trial[parameter_index] = candidate
                    loss, shaped, rendered = evaluate(trial)
                    if loss < best_loss:
                        best_loss = loss
                        parameters = trial
                        best_vertices = shaped
                        best_render = rendered
                print(json.dumps({
                    "pass": pass_index + 1,
                    "parameter": parameter_name,
                    "value": float(parameters[parameter_index]),
                    "best_iou": 1.0 - best_loss,
                }), flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base._comparison_sheet(
        targets.cpu().numpy(), best_render.cpu().numpy(), records,
        args.output_dir / "best.png",
    )
    base._export_mesh(
        args.output_dir / "fitted.ply",
        best_vertices.cpu().numpy(),
        faces_np,
    )
    values = {
        name: float(value)
        for name, value in zip(parameter_names, parameters.cpu())
    }
    mhm_lines = [
        "version v1.2.0",
        "name qwen_body_regional_fit",
        "modifier macrodetails/Gender 1.000000",
        "modifier macrodetails/Age 0.500000",
        "modifier macrodetails/African 0.000000",
        "modifier macrodetails/Asian 0.000000",
        "modifier macrodetails/Caucasian 1.000000",
        f"modifier macrodetails-universal/Muscle {values['muscle']:.6f}",
        f"modifier macrodetails-universal/Weight {values['weight']:.6f}",
        f"modifier macrodetails-height/Height {values['height']:.6f}",
        (
            "modifier macrodetails-proportions/BodyProportions "
            f"{values['body_proportions']:.6f}"
        ),
    ]
    for value, names in zip(parameters[4:].cpu(), modifier_names):
        for name in names:
            mhm_lines.append(f"modifier {name} {float(value):.6f}")
    mhm_lines.append("skeleton default.mhskel")
    (args.output_dir / "fitted.mhm").write_text(
        "\n".join(mhm_lines) + "\n", encoding="utf-8",
    )
    report = {
        "schema": "diffusion-editor.qwen-makehuman-regional-search",
        "schema_version": 1,
        "algorithm": "coordinate search over exact MakeHuman targets and hard IoU",
        "passes": args.passes,
        "initial_step": args.initial_step,
        "initial_report": (
            None
            if args.initial_report is None
            else str(args.initial_report.resolve())
        ),
        "initial_iou": initial_iou,
        "best_iou": 1.0 - best_loss,
        "parameters": values,
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
