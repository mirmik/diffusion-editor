#!/usr/bin/env python3
"""Jointly fit shared MakeHuman shape and bounded per-view arm poses."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import torch
from pytorch3d.structures import Meshes


MACRO_NAMES = ("muscle", "weight", "height", "body_proportions")
POSE_BONES = (
    "clavicle.L",
    "clavicle.R",
    "upperarm01.L",
    "upperarm01.R",
    "lowerarm01.L",
    "lowerarm01.R",
)
POSE_LIMIT_DEGREES = (25.0, 25.0, 55.0, 55.0, 80.0, 80.0)


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
    parser.add_argument("--rig", type=Path, required=True)
    parser.add_argument("--skin-weights", type=Path, required=True)
    parser.add_argument("--camera-report", type=Path, required=True)
    parser.add_argument("--initial-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--evaluation-interval", type=int, default=10)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--fov", type=float, default=35.0)
    parser.add_argument("--mask-threshold", type=float, default=18.0)
    parser.add_argument(
        "--mask-dir",
        type=Path,
        help="Precomputed foreground masks named exactly like the nine views.",
    )
    parser.add_argument("--shape-learning-rate", type=float, default=0.015)
    parser.add_argument("--pose-learning-rate", type=float, default=0.03)
    parser.add_argument("--shape-prior-weight", type=float, default=0.0005)
    parser.add_argument("--pose-prior-weight", type=float, default=0.001)
    parser.add_argument("--coordinate-refine-passes", type=int, default=0)
    parser.add_argument("--shape-coordinate-step", type=float, default=0.125)
    parser.add_argument("--pose-coordinate-step-degrees", type=float, default=8.0)
    parser.add_argument("--camera-azimuth-step-degrees", type=float, default=4.0)
    parser.add_argument("--camera-elevation-step-degrees", type=float, default=4.0)
    parser.add_argument("--camera-distance-step", type=float, default=0.1)
    parser.add_argument("--camera-azimuth-radius-degrees", type=float, default=12.0)
    parser.add_argument("--camera-elevation-radius-degrees", type=float, default=12.0)
    parser.add_argument("--camera-distance-radius", type=float, default=0.4)
    parser.add_argument("--camera-prior-weight", type=float, default=0.001)
    parser.add_argument("--refine-camera", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--hard-forward-silhouette", action="store_true")
    return parser.parse_args()


def _inverse_sigmoid(values: torch.Tensor) -> torch.Tensor:
    values = torch.clamp(values, 1.0e-4, 1.0 - 1.0e-4)
    return torch.log(values / (1.0 - values))


def _inverse_tanh(values: torch.Tensor) -> torch.Tensor:
    values = torch.clamp(values, -1.0 + 1.0e-4, 1.0 - 1.0e-4)
    return 0.5 * torch.log((1.0 + values) / (1.0 - values))


def _per_view_iou(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dimensions = tuple(range(1, predicted.ndim))
    intersection = torch.sum(predicted * target, dim=dimensions)
    union = (
        torch.sum(predicted, dim=dimensions)
        + torch.sum(target, dim=dimensions)
        - intersection
    )
    return (intersection + 1.0e-6) / (union + 1.0e-6)


def _shaped_bone_heads(
    vertices: torch.Tensor,
    head_indices: list[torch.Tensor],
) -> torch.Tensor:
    return torch.stack([
        vertices[indices].mean(dim=0) for indices in head_indices
    ])


def _render_per_view(
    posed_vertices: torch.Tensor,
    faces: torch.Tensor,
    records: list[dict],
    render_args: SimpleNamespace,
    base,
) -> torch.Tensor:
    count = len(records)
    indices = torch.arange(count, device=posed_vertices.device)
    meshes = Meshes(
        verts=[posed_vertices[index] for index in range(count)],
        faces=[faces for _ in range(count)],
    )
    renderer = base.make_renderer(
        records,
        indices,
        render_args.image_size,
        render_args.distance,
        render_args.fov,
        posed_vertices.device,
    )
    if not render_args.hard_forward_silhouette:
        return renderer(meshes)[..., 3]
    fragments = renderer.rasterizer(meshes)
    soft = renderer.shader(fragments, meshes)[..., 3]
    hard = (fragments.pix_to_face[..., 0] >= 0).to(soft.dtype)
    return soft + (hard - soft).detach()


def _write_mhm(
    path: Path,
    values: dict[str, float],
    regional_names: list[list[str]],
    regional_parameter_names: list[str],
) -> None:
    lines = [
        "version v1.2.0",
        "name qwen_body_joint_shape_pose_fit",
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
    for parameter_name, modifier_group in zip(
        regional_parameter_names, regional_names
    ):
        for modifier_name in modifier_group:
            lines.append(f"modifier {modifier_name} {values[parameter_name]:.6f}")
    lines.append("skeleton default.mhskel")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.iterations < 1 or args.evaluation_interval < 1:
        raise SystemExit("iterations and evaluation interval must be positive")
    if args.coordinate_refine_passes < 0:
        raise SystemExit("coordinate refine passes cannot be negative")
    if args.shape_coordinate_step <= 0 or args.pose_coordinate_step_degrees <= 0:
        raise SystemExit("coordinate refinement steps must be positive")
    camera_steps = (
        args.camera_azimuth_step_degrees,
        args.camera_elevation_step_degrees,
        args.camera_distance_step,
    )
    camera_radii = (
        args.camera_azimuth_radius_degrees,
        args.camera_elevation_radius_degrees,
        args.camera_distance_radius,
    )
    if any(value <= 0 for value in camera_steps + camera_radii):
        raise SystemExit("camera refinement steps and radii must be positive")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")

    scripts_dir = Path(__file__).parent
    base = _load_module(
        scripts_dir / "fit-qwen-silhouette-mesh.py", "silhouette_fit_joint",
    )
    macro = _load_module(
        scripts_dir / "fit-qwen-makehuman-macros.py", "macro_fit_joint",
    )
    regional_module = _load_module(
        scripts_dir / "search-qwen-makehuman-regional.py", "regional_fit_joint",
    )

    records, target_masks = base.load_views(
        args.image_dir, args.image_size, args.mask_threshold, args.mask_dir,
    )
    camera_report = json.loads(args.camera_report.read_text(encoding="utf-8"))
    calibrated = {view["name"]: view for view in camera_report["views"]}
    for record in records:
        view = calibrated.get(record["name"])
        if view is None:
            raise SystemExit(f"camera report has no view for {record['name']}")
        record["azimuth_degrees"] = view["calibrated_azimuth_degrees"]
        record["elevation_degrees"] = view["calibrated_elevation_degrees"]
        record["distance"] = view["calibrated_distance"]
    body_vertices_np, faces_np, normalization = base.load_template(
        args.template, 0, None, "body", "y", 0,
    )
    raw_vertices_np = base._obj_vertices(args.template, "y")
    full_vertices_np = (
        raw_vertices_np - normalization["center"][None, :]
    ) * normalization["scale"]
    body_vertex_count = len(body_vertices_np)
    if not np.allclose(
        body_vertices_np, full_vertices_np[:body_vertex_count], atol=1.0e-6
    ):
        raise SystemExit(
            "HM08 body vertices are not the leading full-mesh vertices"
        )
    full_vertex_count = len(full_vertices_np)
    macro_arrays = macro.load_macro_targets(
        args.targets_dir / "macrodetails",
        full_vertex_count,
        normalization["scale"],
    )
    regional_targets, modifier_names = regional_module.load_regional_targets(
        args.targets_dir,
        macro,
        full_vertex_count,
        normalization["scale"],
    )
    articulation = base.load_articulation(
        args.template,
        args.rig,
        args.skin_weights,
        body_vertex_count,
        normalization,
        list(POSE_BONES),
    )

    full_base_vertices = torch.from_numpy(full_vertices_np).to(device)
    faces = torch.from_numpy(faces_np).to(device)
    targets = target_masks.to(device)
    macro_targets = {
        name: torch.from_numpy(value).to(device)
        for name, value in macro_arrays.items()
    }
    regional_targets = regional_targets.to(device)
    skin_weights = torch.from_numpy(articulation["weights"]).to(device)
    head_indices = [
        torch.tensor(indices, device=device, dtype=torch.long)
        for indices in articulation["bone_head_vertex_indices"]
    ]
    pose_bone_indices = articulation["pose_bone_indices"]
    pose_limits = torch.deg2rad(torch.tensor(
        POSE_LIMIT_DEGREES, device=device, dtype=full_base_vertices.dtype,
    ))[None, :, None]
    zero_scale = torch.zeros((), device=device)
    zero_translation = torch.zeros(3, device=device)

    regional_names = list(regional_module.REGIONAL_SPECS)
    parameter_names = list(MACRO_NAMES) + regional_names
    initial_report = json.loads(args.initial_report.read_text(encoding="utf-8"))
    initial_values = initial_report.get("parameters")
    if not isinstance(initial_values, dict):
        raise SystemExit("initial report has no parameter mapping")
    missing = [name for name in parameter_names if name not in initial_values]
    if missing:
        raise SystemExit("initial report is missing: " + ", ".join(missing))
    initial_macro = torch.tensor(
        [float(initial_values[name]) for name in MACRO_NAMES], device=device,
    )
    initial_regional = torch.tensor(
        [float(initial_values[name]) for name in regional_names], device=device,
    )
    macro_logits = _inverse_sigmoid(initial_macro).detach().requires_grad_(True)
    regional_logits = _inverse_tanh(initial_regional).detach().requires_grad_(True)
    initial_pose = torch.zeros(
        (len(records), len(POSE_BONES), 3), device=device,
    )
    pose_report = initial_report.get("per_view_pose_axis_angle_degrees")
    if isinstance(pose_report, dict):
        for view_index, record in enumerate(records):
            view_pose = pose_report.get(record["name"])
            if not isinstance(view_pose, dict):
                raise SystemExit(
                    f"initial joint report has no pose for {record['name']}"
                )
            for bone_index, bone in enumerate(POSE_BONES):
                components = view_pose.get(bone)
                if not isinstance(components, list) or len(components) != 3:
                    raise SystemExit(
                        f"initial joint report has no 3D pose for {bone} "
                        f"in {record['name']}"
                    )
                initial_pose[view_index, bone_index] = torch.deg2rad(
                    torch.tensor(components, device=device)
                )
    camera_state = initial_report.get("per_view_camera")
    if isinstance(camera_state, dict):
        for record in records:
            view_camera = camera_state.get(record["name"])
            if not isinstance(view_camera, dict):
                raise SystemExit(
                    f"initial joint report has no camera for {record['name']}"
                )
            for key in ("azimuth_degrees", "elevation_degrees", "distance"):
                record[key] = float(view_camera[key])
    initial_cameras = [{
        "azimuth_degrees": float(record["azimuth_degrees"]),
        "elevation_degrees": float(record["elevation_degrees"]),
        "distance": float(record["distance"]),
    } for record in records]
    pose_logits = _inverse_tanh(
        initial_pose / pose_limits,
    ).detach().requires_grad_(True)

    def parameter_values() -> tuple[torch.Tensor, torch.Tensor]:
        return torch.sigmoid(macro_logits), torch.tanh(regional_logits)

    def rest_shape() -> tuple[torch.Tensor, torch.Tensor]:
        macros, regional = parameter_values()
        shaped = macro.shaped_vertices(
            full_base_vertices, macro_targets, macros,
        )
        full_shape = regional_module.apply_regional(
            shaped, regional_targets, regional,
        )
        return full_shape, full_shape[:body_vertex_count]

    def poses() -> torch.Tensor:
        return torch.tanh(pose_logits) * pose_limits

    def posed_body_shapes(
        full_rest: torch.Tensor,
        body_rest: torch.Tensor,
        pose_values: torch.Tensor,
    ) -> torch.Tensor:
        bone_heads = _shaped_bone_heads(full_rest, head_indices)
        return torch.stack([
            base.articulate_vertices(
                body_rest,
                bone_heads,
                articulation["parents"],
                skin_weights,
                pose_bone_indices,
                pose_values[index],
                zero_scale,
                zero_translation,
            )
            for index in range(len(records))
        ])

    optimization_render_args = SimpleNamespace(
        image_size=args.image_size,
        distance=4.0,
        fov=args.fov,
        hard_forward_silhouette=False,
    )
    evaluation_render_args = SimpleNamespace(
        image_size=args.image_size,
        distance=4.0,
        fov=args.fov,
        hard_forward_silhouette=args.hard_forward_silhouette,
    )
    hard_render_args = SimpleNamespace(
        image_size=args.image_size,
        distance=4.0,
        fov=args.fov,
        hard_forward_silhouette=True,
    )

    def evaluate(render_args: SimpleNamespace):
        macros, regional = parameter_values()
        full_rest, body_rest = rest_shape()
        pose_values = poses()
        posed = posed_body_shapes(full_rest, body_rest, pose_values)
        rendered = _render_per_view(posed, faces, records, render_args, base)
        ious = _per_view_iou(rendered, targets)
        return macros, regional, body_rest, pose_values, posed, rendered, ious

    optimizer = torch.optim.Adam((
        {"params": (macro_logits, regional_logits), "lr": args.shape_learning_rate},
        {"params": (pose_logits,), "lr": args.pose_learning_rate},
    ))
    neutral_macro = torch.full_like(initial_macro, 0.5)
    neutral_regional = torch.zeros_like(initial_regional)
    started = time.monotonic()
    history = []

    with torch.no_grad():
        initial = evaluate(evaluation_render_args)
        initial_iou = float(initial[-1].mean())
    best_iou = initial_iou
    best_iteration = 0
    best_macro = initial[0].detach().clone()
    best_regional = initial[1].detach().clone()
    best_pose = initial[3].detach().clone()

    for iteration in range(1, args.iterations + 1):
        macros, regional, _rest, pose_values, _posed, _rendered, ious = evaluate(
            optimization_render_args
        )
        silhouette_loss = 1.0 - ious.mean()
        shape_prior = (
            torch.mean((macros - neutral_macro) ** 2)
            + torch.mean((regional - neutral_regional) ** 2)
        )
        normalized_pose = pose_values / pose_limits
        pose_prior = torch.mean(normalized_pose ** 2)
        loss = (
            silhouette_loss
            + args.shape_prior_weight * shape_prior
            + args.pose_prior_weight * pose_prior
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            (macro_logits, regional_logits, pose_logits), 1.0,
        )
        optimizer.step()

        if (
            iteration == 1
            or iteration % args.evaluation_interval == 0
            or iteration == args.iterations
        ):
            with torch.no_grad():
                evaluated = evaluate(evaluation_render_args)
                mean_iou = float(evaluated[-1].mean())
                entry = {
                    "iteration": iteration,
                    "mean_iou": mean_iou,
                    "optimization_iou": float(ious.detach().mean()),
                    "total_loss": float(loss.detach()),
                    "shape_prior": float(shape_prior.detach()),
                    "pose_prior": float(pose_prior.detach()),
                }
                history.append(entry)
                if mean_iou > best_iou:
                    best_iou = mean_iou
                    best_iteration = iteration
                    best_macro = evaluated[0].detach().clone()
                    best_regional = evaluated[1].detach().clone()
                    best_pose = evaluated[3].detach().clone()
            print(json.dumps(entry), flush=True)

    coordinate_history = []
    if args.coordinate_refine_passes:
        def install_values(
            macro_values: torch.Tensor,
            regional_values: torch.Tensor,
            pose_values: torch.Tensor,
        ) -> None:
            macro_logits.copy_(_inverse_sigmoid(macro_values))
            regional_logits.copy_(_inverse_tanh(regional_values))
            pose_logits.copy_(_inverse_tanh(pose_values / pose_limits))

        def coordinate_score(
            cached_ious: torch.Tensor | None = None,
            view_index: int | None = None,
        ):
            macro_values, regional_values = parameter_values()
            full_rest, body_rest = rest_shape()
            pose_values = poses()
            posed = posed_body_shapes(full_rest, body_rest, pose_values)
            if view_index is None:
                rendered = _render_per_view(
                    posed, faces, records, hard_render_args, base,
                )
                ious = _per_view_iou(rendered, targets)
            else:
                rendered = _render_per_view(
                    posed[view_index:view_index + 1],
                    faces,
                    records[view_index:view_index + 1],
                    hard_render_args,
                    base,
                )
                view_iou = _per_view_iou(
                    rendered, targets[view_index:view_index + 1],
                )
                ious = cached_ious.clone()
                ious[view_index] = view_iou[0]
            shape_prior = (
                torch.mean((macro_values - neutral_macro) ** 2)
                + torch.mean((regional_values - neutral_regional) ** 2)
            )
            pose_prior = torch.mean((pose_values / pose_limits) ** 2)
            camera_prior = 0.0
            if args.refine_camera:
                for record, center in zip(records, initial_cameras):
                    camera_prior += (
                        (
                            (record["azimuth_degrees"] - center["azimuth_degrees"])
                            / args.camera_azimuth_radius_degrees
                        ) ** 2
                        + (
                            (
                                record["elevation_degrees"]
                                - center["elevation_degrees"]
                            ) / args.camera_elevation_radius_degrees
                        ) ** 2
                        + (
                            (record["distance"] - center["distance"])
                            / args.camera_distance_radius
                        ) ** 2
                    )
                camera_prior /= len(records) * 3
            mean_iou = ious.mean()
            score = (
                mean_iou
                - args.shape_prior_weight * shape_prior
                - args.pose_prior_weight * pose_prior
                - args.camera_prior_weight * camera_prior
            )
            return float(score), float(mean_iou), ious

        with torch.no_grad():
            coordinate_macro = best_macro.clone()
            coordinate_regional = best_regional.clone()
            coordinate_pose = best_pose.clone()
            install_values(
                coordinate_macro, coordinate_regional, coordinate_pose,
            )
            current_score, current_iou, current_ious = coordinate_score()

            for pass_index in range(args.coordinate_refine_passes):
                shape_step = args.shape_coordinate_step / (2 ** pass_index)
                pose_step = math.radians(
                    args.pose_coordinate_step_degrees / (2 ** pass_index)
                )
                accepted_shape = 0
                accepted_pose = 0
                accepted_camera = 0

                for parameter_index in range(len(parameter_names)):
                    is_macro = parameter_index < len(MACRO_NAMES)
                    values = coordinate_macro if is_macro else coordinate_regional
                    value_index = (
                        parameter_index
                        if is_macro
                        else parameter_index - len(MACRO_NAMES)
                    )
                    lower, upper = (1.0e-4, 1.0 - 1.0e-4) if is_macro else (
                        -1.0 + 1.0e-4, 1.0 - 1.0e-4,
                    )
                    original = float(values[value_index])
                    winning_value = original
                    winning_score = current_score
                    winning_iou = current_iou
                    winning_ious = current_ious
                    for direction in (-1.0, 1.0):
                        candidate = min(
                            upper, max(lower, original + direction * shape_step),
                        )
                        if abs(candidate - original) < 1.0e-8:
                            continue
                        values[value_index] = candidate
                        install_values(
                            coordinate_macro, coordinate_regional, coordinate_pose,
                        )
                        score, mean_iou, candidate_ious = coordinate_score()
                        if score > winning_score + 1.0e-9:
                            winning_value = candidate
                            winning_score = score
                            winning_iou = mean_iou
                            winning_ious = candidate_ious
                    values[value_index] = winning_value
                    install_values(
                        coordinate_macro, coordinate_regional, coordinate_pose,
                    )
                    if winning_value != original:
                        accepted_shape += 1
                        current_score = winning_score
                        current_iou = winning_iou
                        current_ious = winning_ious

                print(json.dumps({
                    "coordinate_pass": pass_index + 1,
                    "stage": "shape",
                    "mean_iou": current_iou,
                    "accepted": accepted_shape,
                    "step": shape_step,
                }), flush=True)

                for view_index, record in enumerate(records):
                    view_accepted = 0
                    for bone_index in range(len(POSE_BONES)):
                        component_limit = float(pose_limits[0, bone_index, 0])
                        for axis_index in range(3):
                            original = float(
                                coordinate_pose[view_index, bone_index, axis_index]
                            )
                            winning_value = original
                            winning_score = current_score
                            winning_iou = current_iou
                            winning_ious = current_ious
                            for direction in (-1.0, 1.0):
                                candidate = min(
                                    component_limit,
                                    max(-component_limit, original + direction * pose_step),
                                )
                                if abs(candidate - original) < 1.0e-8:
                                    continue
                                coordinate_pose[
                                    view_index, bone_index, axis_index
                                ] = candidate
                                install_values(
                                    coordinate_macro,
                                    coordinate_regional,
                                    coordinate_pose,
                                )
                                score, mean_iou, candidate_ious = coordinate_score(
                                    current_ious, view_index,
                                )
                                if score > winning_score + 1.0e-9:
                                    winning_value = candidate
                                    winning_score = score
                                    winning_iou = mean_iou
                                    winning_ious = candidate_ious
                            coordinate_pose[
                                view_index, bone_index, axis_index
                            ] = winning_value
                            install_values(
                                coordinate_macro,
                                coordinate_regional,
                                coordinate_pose,
                            )
                            if winning_value != original:
                                accepted_pose += 1
                                view_accepted += 1
                                current_score = winning_score
                                current_iou = winning_iou
                                current_ious = winning_ious
                    print(json.dumps({
                        "coordinate_pass": pass_index + 1,
                        "stage": "pose",
                        "view": record["name"],
                        "mean_iou": current_iou,
                        "accepted": view_accepted,
                        "step_degrees": math.degrees(pose_step),
                    }), flush=True)

                if args.refine_camera:
                    camera_specs = (
                        (
                            "azimuth_degrees",
                            args.camera_azimuth_step_degrees / (2 ** pass_index),
                            args.camera_azimuth_radius_degrees,
                        ),
                        (
                            "elevation_degrees",
                            args.camera_elevation_step_degrees / (2 ** pass_index),
                            args.camera_elevation_radius_degrees,
                        ),
                        (
                            "distance",
                            args.camera_distance_step / (2 ** pass_index),
                            args.camera_distance_radius,
                        ),
                    )
                    for view_index, record in enumerate(records):
                        view_accepted = 0
                        for key, step, radius in camera_specs:
                            center = initial_cameras[view_index][key]
                            lower = center - radius
                            upper = center + radius
                            original = float(record[key])
                            winning_value = original
                            winning_score = current_score
                            winning_iou = current_iou
                            winning_ious = current_ious
                            for direction in (-1.0, 1.0):
                                candidate = min(
                                    upper, max(lower, original + direction * step),
                                )
                                if abs(candidate - original) < 1.0e-8:
                                    continue
                                record[key] = candidate
                                score, mean_iou, candidate_ious = coordinate_score(
                                    current_ious, view_index,
                                )
                                if score > winning_score + 1.0e-9:
                                    winning_value = candidate
                                    winning_score = score
                                    winning_iou = mean_iou
                                    winning_ious = candidate_ious
                            record[key] = winning_value
                            if winning_value != original:
                                accepted_camera += 1
                                view_accepted += 1
                                current_score = winning_score
                                current_iou = winning_iou
                                current_ious = winning_ious
                        print(json.dumps({
                            "coordinate_pass": pass_index + 1,
                            "stage": "camera",
                            "view": record["name"],
                            "mean_iou": current_iou,
                            "accepted": view_accepted,
                            "camera": {
                                key: float(record[key])
                                for key in (
                                    "azimuth_degrees",
                                    "elevation_degrees",
                                    "distance",
                                )
                            },
                        }), flush=True)

                coordinate_entry = {
                    "pass": pass_index + 1,
                    "mean_iou": current_iou,
                    "score": current_score,
                    "shape_step": shape_step,
                    "pose_step_degrees": math.degrees(pose_step),
                    "accepted_shape_coordinates": accepted_shape,
                    "accepted_pose_coordinates": accepted_pose,
                    "accepted_camera_coordinates": accepted_camera,
                }
                coordinate_history.append(coordinate_entry)
                print(json.dumps(coordinate_entry), flush=True)

            best_macro = coordinate_macro.clone()
            best_regional = coordinate_regional.clone()
            best_pose = coordinate_pose.clone()
            best_iou = current_iou
            best_iteration = f"coordinate-pass-{args.coordinate_refine_passes}"

    with torch.no_grad():
        macro_logits.copy_(_inverse_sigmoid(best_macro))
        regional_logits.copy_(_inverse_tanh(best_regional))
        pose_logits.copy_(torch.atanh(torch.clamp(
            best_pose / pose_limits, -1.0 + 1.0e-4, 1.0 - 1.0e-4,
        )))
        final = evaluate(evaluation_render_args)
    final_macro, final_regional, final_rest = final[:3]
    final_pose, final_posed, final_render, final_ious = final[3:]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base._comparison_sheet(
        targets.cpu().numpy(),
        final_render.cpu().numpy(),
        records,
        args.output_dir / "best.png",
    )
    base._export_mesh(
        args.output_dir / "fitted-rest.ply", final_rest.cpu().numpy(), faces_np,
    )
    for record, vertices in zip(records, final_posed):
        base._export_mesh(
            args.output_dir / f"posed-{Path(record['name']).stem}.ply",
            vertices.cpu().numpy(),
            faces_np,
        )

    values = {
        name: float(value)
        for name, value in zip(
            parameter_names,
            torch.cat((final_macro, final_regional)).cpu(),
        )
    }
    _write_mhm(
        args.output_dir / "fitted.mhm",
        values,
        modifier_names,
        regional_names,
    )
    pose_degrees = torch.rad2deg(final_pose).cpu()
    per_view_pose = {
        record["name"]: {
            bone: [float(component) for component in vector]
            for bone, vector in zip(POSE_BONES, pose_degrees[index])
        }
        for index, record in enumerate(records)
    }
    per_view_camera = {
        record["name"]: {
            "azimuth_degrees": float(record["azimuth_degrees"]),
            "elevation_degrees": float(record["elevation_degrees"]),
            "distance": float(record["distance"]),
        }
        for record in records
    }
    camera_output = {
        "schema": "diffusion-editor.qwen-silhouette-camera-calibration",
        "schema_version": 2,
        "source_camera_report": str(args.camera_report.resolve()),
        "source_fit_report": str(args.initial_report.resolve()),
        "mean_iou": float(final_ious.mean()),
        "views": [{
            "name": record["name"],
            "initial_azimuth_degrees": initial_camera["azimuth_degrees"],
            "initial_elevation_degrees": initial_camera["elevation_degrees"],
            "initial_distance": initial_camera["distance"],
            "calibrated_azimuth_degrees": float(record["azimuth_degrees"]),
            "calibrated_elevation_degrees": float(record["elevation_degrees"]),
            "calibrated_distance": float(record["distance"]),
            "iou": float(iou),
        } for record, initial_camera, iou in zip(
            records, initial_cameras, final_ious.cpu(),
        )],
    }
    (args.output_dir / "cameras.json").write_text(
        json.dumps(camera_output, indent=2) + "\n", encoding="utf-8",
    )
    report = {
        "schema": "diffusion-editor.qwen-makehuman-joint-shape-pose-fit",
        "schema_version": 2,
        "algorithm": (
            "shared MakeHuman shape plus bounded per-view LBS arm pose "
            "and alternating bounded camera refinement"
        ),
        "image_dir": str(args.image_dir.resolve()),
        "camera_report": str(args.camera_report.resolve()),
        "initial_report": str(args.initial_report.resolve()),
        "mask_dir": (
            None if args.mask_dir is None else str(args.mask_dir.resolve())
        ),
        "rig": str(args.rig.resolve()),
        "skin_weights": str(args.skin_weights.resolve()),
        "pose_bones": list(POSE_BONES),
        "pose_limit_degrees": dict(zip(POSE_BONES, POSE_LIMIT_DEGREES)),
        "initial_iou": initial_iou,
        "best_iou": float(final_ious.mean()),
        "best_iteration": best_iteration,
        "per_view_iou": {
            record["name"]: float(iou)
            for record, iou in zip(records, final_ious.cpu())
        },
        "parameters": values,
        "per_view_pose_axis_angle_degrees": per_view_pose,
        "per_view_camera": per_view_camera,
        "optimization": {
            key: value
            for key, value in vars(args).items()
            if not isinstance(value, Path)
        },
        "elapsed_seconds": time.monotonic() - started,
        "history": history,
        "coordinate_history": coordinate_history,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8",
    )
    print(json.dumps({
        "initial_iou": report["initial_iou"],
        "best_iou": report["best_iou"],
        "best_iteration": report["best_iteration"],
        "elapsed_seconds": report["elapsed_seconds"],
        "output_dir": str(args.output_dir),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
