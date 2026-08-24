#!/usr/bin/env python3
"""Refine nine independent poses of one frozen MakeHuman body and camera rig."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as functional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for module_path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shared_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--iterations-per-view", type=int, default=350)
    parser.add_argument("--evaluation-interval", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--pose-delta-prior-weight", type=float, default=0.15)
    parser.add_argument("--soft-l1-pixels", type=float, default=7.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def _array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        values = np.ascontiguousarray(array)
        digest.update(str(values.dtype).encode())
        digest.update(str(values.shape).encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    if args.iterations_per_view < 1 or args.evaluation_interval < 1:
        raise SystemExit("iterations and evaluation interval must be positive")
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("confidence must be in [0, 1]")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shared_module = _load_module(
        SCRIPTS_DIR / "fit-makehuman-dwpose-joints.py", "shared_joint_fit")
    base = _load_module(
        SCRIPTS_DIR / "fit-qwen-silhouette-mesh.py", "per_view_pose_base")
    macro = _load_module(
        SCRIPTS_DIR / "fit-qwen-makehuman-macros.py", "per_view_pose_macro")
    regional_module = _load_module(
        SCRIPTS_DIR / "search-qwen-makehuman-regional.py", "per_view_pose_regional")

    shared_report = json.loads(args.shared_report.read_text(encoding="utf-8"))
    inputs = shared_report["inputs"]
    camera_report_path = Path(inputs["camera_report"])
    keypoint_path = Path(inputs["keypoints"])
    template_path = Path(inputs["template"])
    rig_path = Path(inputs["rig"])
    weights_path = Path(inputs["skin_weights"])
    targets_dir = Path(inputs["targets_dir"])
    camera_report = json.loads(camera_report_path.read_text(encoding="utf-8"))
    cameras = camera_report["cameras"]
    camera_digest_before = shared_module._camera_digest(cameras)
    keypoint_report = json.loads(keypoint_path.read_text(encoding="utf-8"))
    cache_views = {view["name"]: view for view in keypoint_report["views"]}
    view_names = [camera["name"] for camera in cameras]
    joint_names = list(shared_module.JOINT_TO_BONE)

    observations = []
    observation_weights = []
    sizes = []
    image_paths = []
    for camera in cameras:
        view = cache_views[camera["name"]]
        width, height = int(view["width"]), int(view["height"])
        sizes.append((width, height))
        image_paths.append(Path(view["path"]))
        points = []
        scores = []
        for name in joint_names:
            point = view["points"][name]
            points.append((
                (float(point["x"]) - width * 0.5) / height,
                (height * 0.5 - float(point["y"])) / height,
            ))
            score = float(point["score"])
            scores.append(score if score >= args.confidence else 0.0)
        observations.append(points)
        observation_weights.append(scores)
    observations_np = np.asarray(observations, dtype=np.float32)
    observation_weights_np = np.asarray(observation_weights, dtype=np.float32)

    body_vertices_np, faces_np, normalization = base.load_template(
        template_path, 0, None, "body", "y", 0)
    raw_vertices_np = base._obj_vertices(template_path, "y")
    full_vertices_np = (
        raw_vertices_np - normalization["center"][None, :]
    ) * normalization["scale"]
    body_vertex_count = len(body_vertices_np)
    macro_arrays = macro.load_macro_targets(
        targets_dir / "macrodetails", len(full_vertices_np), normalization["scale"])
    regional_targets_np, _modifier_names = regional_module.load_regional_targets(
        targets_dir, macro, len(full_vertices_np), normalization["scale"])
    articulation = base.load_articulation(
        template_path, rig_path, weights_path, body_vertex_count,
        normalization, list(shared_module.POSE_BONES))

    dtype = torch.float32
    full_base = torch.from_numpy(full_vertices_np).to(device)
    macro_targets = {
        name: torch.from_numpy(values).to(device)
        for name, values in macro_arrays.items()
    }
    regional_targets = regional_targets_np.to(device)
    shape_values = shared_report["fitted_parameters"]
    regional_names = list(regional_module.REGIONAL_SPECS)
    macro_values = torch.tensor(
        [float(shape_values[name]) for name in shared_module.MACRO_NAMES],
        device=device, dtype=dtype)
    regional_values = torch.tensor(
        [float(shape_values[name]) for name in regional_names],
        device=device, dtype=dtype)
    with torch.no_grad():
        full_rest = macro.shaped_vertices(full_base, macro_targets, macro_values)
        full_rest = regional_module.apply_regional(
            full_rest, regional_targets, regional_values)
        body_rest = full_rest[:body_vertex_count]
    head_indices = [
        torch.tensor(indices, device=device, dtype=torch.long)
        for indices in articulation["bone_head_vertex_indices"]
    ]
    with torch.no_grad():
        bone_heads = shared_module._shaped_bone_heads(full_rest, head_indices)
    shape_digest_before = _array_digest(
        full_rest.cpu().numpy(), bone_heads.cpu().numpy())
    bone_names = articulation["bone_names"]
    joint_bone_indices = [
        bone_names.index(shared_module.JOINT_TO_BONE[name])
        for name in joint_names
    ]
    identity_skin = torch.eye(len(bone_names), device=device, dtype=dtype)
    skin_weights = torch.from_numpy(articulation["weights"]).to(device)
    zero_scale = torch.zeros((), device=device)
    zero_translation = torch.zeros(3, device=device)

    shared_pose = torch.deg2rad(torch.tensor([
        shared_report["shared_pose_axis_angle_degrees"][bone]
        for bone in shared_module.POSE_BONES
    ], device=device, dtype=dtype))
    # Limits describe deviations from the common pose, not absolute rotations.
    delta_limit_degrees = torch.tensor((
        12.0, 12.0,
        25.0, 25.0,
        35.0, 35.0,
        10.0, 10.0,
        20.0, 20.0,
        30.0, 30.0,
    ), device=device, dtype=dtype)[:, None]
    delta_limits = torch.deg2rad(delta_limit_degrees)

    similarity = shared_report["global_body_similarity"]
    world_rotation = torch.tensor(
        similarity["rotation"], device=device, dtype=dtype)
    world_scale = torch.tensor(similarity["scale"], device=device, dtype=dtype)
    world_translation = torch.tensor(
        similarity["translation"], device=device, dtype=dtype)
    camera_rotations = torch.tensor([
        camera["world_to_camera_rotation"] for camera in cameras
    ], device=device, dtype=dtype)
    camera_scales = torch.tensor([
        camera["scale"] for camera in cameras
    ], device=device, dtype=dtype)
    camera_translations = torch.tensor([
        camera["translation_normalized"] for camera in cameras
    ], device=device, dtype=dtype)
    camera_reference = (
        camera_rotations.clone(), camera_scales.clone(), camera_translations.clone())
    target = torch.from_numpy(observations_np).to(device)
    target_weights = torch.from_numpy(observation_weights_np).to(device)
    image_heights = torch.tensor(
        [height for _width, height in sizes], device=device, dtype=dtype)

    def posed_joints(pose: torch.Tensor) -> torch.Tensor:
        all_heads = base.articulate_vertices(
            bone_heads, bone_heads, articulation["parents"], identity_skin,
            articulation["pose_bone_indices"], pose,
            zero_scale, zero_translation)
        local = all_heads[joint_bone_indices]
        return world_scale * (local @ world_rotation.T) + world_translation

    def project(view_index: int, world_joints: torch.Tensor) -> torch.Tensor:
        return (
            camera_scales[view_index]
            * (world_joints @ camera_rotations[view_index, :2].T)
            + camera_translations[view_index]
        )

    with torch.no_grad():
        shared_world_joints = posed_joints(shared_pose)
        shared_normalized = torch.stack([
            project(index, shared_world_joints) for index in range(len(cameras))])

    per_view_poses = []
    per_view_world_joints = []
    histories = {}
    pose_delta_summary = {}
    started = time.monotonic()
    for view_index, view_name in enumerate(view_names):
        delta_logits = torch.zeros(
            (len(shared_module.POSE_BONES), 3), device=device,
            requires_grad=True)
        optimizer = torch.optim.Adam((delta_logits,), lr=args.learning_rate)

        def evaluate():
            delta = torch.tanh(delta_logits) * delta_limits
            pose = shared_pose + delta
            world_joints = posed_joints(pose)
            normalized = project(view_index, world_joints)
            residual_pixels = (
                normalized - target[view_index]) * image_heights[view_index]
            robust = functional.smooth_l1_loss(
                residual_pixels, torch.zeros_like(residual_pixels),
                beta=args.soft_l1_pixels, reduction="none").sum(dim=-1)
            scores = target_weights[view_index]
            reprojection = torch.sum(robust * scores) / torch.sum(scores)
            squared = torch.sum(residual_pixels ** 2, dim=-1)
            weighted_rms = torch.sqrt(
                torch.sum(squared * scores) / torch.sum(scores))
            prior = torch.mean((delta / delta_limits) ** 2)
            loss = reprojection + args.pose_delta_prior_weight * prior
            return loss, reprojection, weighted_rms, prior, pose, delta, world_joints

        with torch.no_grad():
            initial = evaluate()
            best_loss = float(initial[0])
            best_iteration = 0
            best_logits = delta_logits.detach().clone()
        history = []
        for iteration in range(1, args.iterations_per_view + 1):
            current = evaluate()
            optimizer.zero_grad()
            current[0].backward()
            torch.nn.utils.clip_grad_norm_((delta_logits,), 2.0)
            optimizer.step()
            if (
                iteration == 1
                or iteration % args.evaluation_interval == 0
                or iteration == args.iterations_per_view
            ):
                with torch.no_grad():
                    checked = evaluate()
                    entry = {
                        "iteration": iteration,
                        "weighted_rms_pixels": float(checked[2]),
                        "reprojection_loss": float(checked[1]),
                        "pose_delta_prior": float(checked[3]),
                        "total_loss": float(checked[0]),
                    }
                    history.append(entry)
                    if float(checked[0]) < best_loss:
                        best_loss = float(checked[0])
                        best_iteration = iteration
                        best_logits = delta_logits.detach().clone()
        with torch.no_grad():
            delta_logits.copy_(best_logits)
            fitted = evaluate()
            per_view_poses.append(fitted[4].cpu().numpy())
            per_view_world_joints.append(fitted[6].cpu().numpy())
            delta_degrees = torch.rad2deg(fitted[5]).cpu().numpy()
        histories[view_name] = {
            "best_iteration": best_iteration,
            "history": history,
        }
        pose_delta_summary[view_name] = {
            "rms_degrees": float(np.sqrt(np.mean(delta_degrees ** 2))),
            "max_component_degrees": float(np.max(np.abs(delta_degrees))),
            "per_bone_axis_angle_degrees": {
                bone: values.tolist()
                for bone, values in zip(shared_module.POSE_BONES, delta_degrees)
            },
        }
        print(json.dumps({
            "view": view_name,
            "shared_rms_pixels": float(initial[2]),
            "refined_rms_pixels": float(fitted[2]),
            "pose_delta_rms_degrees": pose_delta_summary[view_name]["rms_degrees"],
            "best_iteration": best_iteration,
        }), flush=True)

    per_view_poses_np = np.stack(per_view_poses)
    per_view_world_joints_np = np.stack(per_view_world_joints)
    with torch.no_grad():
        refined_normalized = torch.stack([
            project(index, torch.from_numpy(per_view_world_joints_np[index]).to(device))
            for index in range(len(cameras))
        ]).cpu().numpy()
    shared_normalized_np = shared_normalized.cpu().numpy()
    observed_pixels = shared_module._pixels(observations_np, sizes)
    shared_pixels = shared_module._pixels(shared_normalized_np, sizes)
    refined_pixels = shared_module._pixels(refined_normalized, sizes)
    shared_overall, shared_per_view = shared_module._metrics(
        shared_pixels, observed_pixels, observation_weights_np, view_names)
    refined_overall, refined_per_view = shared_module._metrics(
        refined_pixels, observed_pixels, observation_weights_np, view_names)
    shared_grid = shared_module._draw_overlays(
        args.output_dir, "shared", image_paths, view_names, joint_names,
        observed_pixels, shared_pixels, observation_weights_np, shared_per_view)
    refined_grid = shared_module._draw_overlays(
        args.output_dir, "per-view", image_paths, view_names, joint_names,
        observed_pixels, refined_pixels, observation_weights_np, refined_per_view)

    faces = np.asarray(faces_np)
    pose_artifacts = {}
    with torch.no_grad():
        for view_index, view_name in enumerate(view_names):
            pose = torch.from_numpy(per_view_poses_np[view_index]).to(device)
            local_mesh = base.articulate_vertices(
                body_rest, bone_heads, articulation["parents"], skin_weights,
                articulation["pose_bone_indices"], pose,
                zero_scale, zero_translation)
            world_mesh = (
                world_scale * (local_mesh @ world_rotation.T) + world_translation
            ).cpu().numpy()
            mesh_path = args.output_dir / f"{view_name}-posed-makehuman.ply"
            joint_path = args.output_dir / f"{view_name}-joints.ply"
            base._export_mesh(mesh_path, world_mesh, faces)
            shared_module._write_joint_ply(
                joint_path, joint_names, per_view_world_joints_np[view_index])
            pose_artifacts[view_name] = {
                "posed_mesh": str(mesh_path),
                "joints_ply": str(joint_path),
            }

    camera_digest_after = shared_module._camera_digest(cameras)
    cameras_frozen = (
        camera_digest_before == camera_digest_after
        and torch.equal(camera_rotations, camera_reference[0])
        and torch.equal(camera_scales, camera_reference[1])
        and torch.equal(camera_translations, camera_reference[2])
    )
    shape_digest_after = _array_digest(
        full_rest.cpu().numpy(), bone_heads.cpu().numpy())
    shape_frozen = shape_digest_before == shape_digest_after
    if not cameras_frozen or not shape_frozen:
        raise RuntimeError("a frozen camera or body-shape tensor changed")

    per_view_table = {}
    for view_name in view_names:
        before = shared_per_view[view_name]["weighted_rms_pixels"]
        after = refined_per_view[view_name]["weighted_rms_pixels"]
        per_view_table[view_name] = {
            "shared_weighted_rms_pixels": before,
            "refined_weighted_rms_pixels": after,
            "improvement_pixels": before - after,
            "improvement_percent": 100.0 * (before - after) / before,
            **pose_delta_summary[view_name],
        }
    pose_degrees = np.rad2deg(per_view_poses_np)
    report = {
        "schema": "diffusion-editor.makehuman-dwpose-per-view-pose-refinement",
        "schema_version": 1,
        "shared_report": str(args.shared_report.resolve()),
        "hypothesis": (
            "one frozen body shape and segment lengths; one independently "
            "refined major-bone pose per generated view"
        ),
        "frozen_state": {
            "cameras": cameras_frozen,
            "camera_sha256_before": camera_digest_before,
            "camera_sha256_after": camera_digest_after,
            "shape": shape_frozen,
            "shape_sha256_before": shape_digest_before,
            "shape_sha256_after": shape_digest_after,
            "global_body_similarity": shared_report["global_body_similarity"],
            "fitted_parameters": shape_values,
            "limb_segment_lengths_in_makehuman_space": shared_report[
                "limb_segment_lengths_in_makehuman_space"]["fitted"],
        },
        "shared_pose_axis_angle_degrees": shared_report[
            "shared_pose_axis_angle_degrees"],
        "per_view_pose_axis_angle_degrees": {
            view_name: {
                bone: values.tolist()
                for bone, values in zip(shared_module.POSE_BONES, pose_degrees[index])
            }
            for index, view_name in enumerate(view_names)
        },
        "per_view_comparison": per_view_table,
        "shared_reprojection": {
            "overall": shared_overall,
            "per_view": shared_per_view,
        },
        "refined_reprojection": {
            "overall": refined_overall,
            "per_view": refined_per_view,
        },
        "optimization": {
            "iterations_per_view": args.iterations_per_view,
            "learning_rate": args.learning_rate,
            "pose_delta_prior_weight": args.pose_delta_prior_weight,
            "soft_l1_pixels": args.soft_l1_pixels,
            "elapsed_seconds": time.monotonic() - started,
            "per_view": histories,
        },
        "per_view_world_joints": {
            view_name: [
                {"name": name, "xyz": point.tolist()}
                for name, point in zip(joint_names, per_view_world_joints_np[index])
            ]
            for index, view_name in enumerate(view_names)
        },
        "artifacts": {
            "shared_reprojection_grid": str(shared_grid),
            "per_view_reprojection_grid": str(refined_grid),
            "per_view": pose_artifacts,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "shared": shared_overall,
        "per_view_refined": refined_overall,
        "cameras_frozen": cameras_frozen,
        "shape_frozen": shape_frozen,
        "report": str(report_path),
        "reprojection_grid": str(refined_grid),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
