#!/usr/bin/env python3
"""Refine small per-view limb lengths after independent pose fitting."""

from __future__ import annotations

import argparse
import importlib.util
import json
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

LENGTH_SEGMENTS = (
    "upperarm",
    "lowerarm",
    "upperleg",
    "lowerleg",
)
SEGMENT_BONES = {
    "upperarm": (("upperarm01.L", "lowerarm01.L"),
                 ("upperarm01.R", "lowerarm01.R")),
    "lowerarm": (("lowerarm01.L", "wrist.L"),
                 ("lowerarm01.R", "wrist.R")),
    "upperleg": (("upperleg01.L", "lowerleg01.L"),
                 ("upperleg01.R", "lowerleg01.R")),
    "lowerleg": (("lowerleg01.L", "foot.L"),
                 ("lowerleg01.R", "foot.R")),
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
    parser.add_argument("pose_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--iterations-per-view", type=int, default=400)
    parser.add_argument("--evaluation-interval", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--max-length-fraction", type=float, default=0.15)
    parser.add_argument("--length-prior-weight", type=float, default=0.20)
    parser.add_argument("--pose-correction-prior-weight", type=float, default=0.15)
    parser.add_argument("--soft-l1-pixels", type=float, default=7.0)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def _joint_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    weights: np.ndarray,
    view_names: list[str],
    joint_names: list[str],
) -> tuple[dict[str, dict], dict[str, dict]]:
    distances = np.linalg.norm(predicted - observed, axis=-1)
    per_view_joint = {
        view_name: {
            joint: float(distances[view_index, joint_index])
            for joint_index, joint in enumerate(joint_names)
            if weights[view_index, joint_index] > 0.0
        }
        for view_index, view_name in enumerate(view_names)
    }
    per_joint = {}
    for joint_index, joint in enumerate(joint_names):
        valid = weights[:, joint_index] > 0.0
        values = distances[valid, joint_index]
        scores = weights[valid, joint_index]
        per_joint[joint] = {
            "weighted_rms_pixels": float(np.sqrt(
                np.sum(scores * values ** 2) / np.sum(scores))),
            "median_pixels": float(np.median(values)),
            "max_pixels": float(np.max(values)),
        }
    return per_view_joint, per_joint


def _length_influences(
    bone_names: list[str],
    parents: list[int],
    bone_heads: torch.Tensor,
) -> torch.Tensor:
    """Bone-head displacements produced by a +100% segment-length change."""

    bone_index = {name: index for index, name in enumerate(bone_names)}
    result = torch.zeros(
        (len(LENGTH_SEGMENTS), len(bone_names), 3),
        device=bone_heads.device, dtype=bone_heads.dtype)
    for segment_index, segment in enumerate(LENGTH_SEGMENTS):
        for start_name, end_name in SEGMENT_BONES[segment]:
            start = bone_index[start_name]
            end = bone_index[end_name]
            reverse_path = []
            current = end
            while current >= 0:
                reverse_path.append(current)
                if current == start:
                    break
                current = parents[current]
            if reverse_path[-1] != start:
                raise RuntimeError(
                    f"{end_name} is not a descendant of {start_name}")
            path = list(reversed(reverse_path))
            distances = [0.0]
            for first, second in zip(path, path[1:]):
                distances.append(distances[-1] + float(torch.linalg.vector_norm(
                    bone_heads[second] - bone_heads[first])))
            total = distances[-1]
            if total <= 1.0e-8:
                raise RuntimeError(f"zero-length rig segment {segment}")
            path_alpha = {
                index: distance / total
                for index, distance in zip(path, distances)
            }
            path_set = set(path)
            segment_vector = bone_heads[end] - bone_heads[start]
            for index in range(len(bone_names)):
                ancestor = index
                while ancestor >= 0 and ancestor not in path_set:
                    ancestor = parents[ancestor]
                if ancestor in path_alpha:
                    result[segment_index, index] += (
                        path_alpha[ancestor] * segment_vector)
    return result


def main() -> int:
    args = parse_args()
    if args.iterations_per_view < 1 or args.evaluation_interval < 1:
        raise SystemExit("iterations and evaluation interval must be positive")
    if not 0.0 < args.max_length_fraction < 0.5:
        raise SystemExit("max length fraction must be in (0, 0.5)")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    shared_module = _load_module(
        SCRIPTS_DIR / "fit-makehuman-dwpose-joints.py", "length_shared_fit")
    base = _load_module(
        SCRIPTS_DIR / "fit-qwen-silhouette-mesh.py", "length_fit_base")
    macro = _load_module(
        SCRIPTS_DIR / "fit-qwen-makehuman-macros.py", "length_fit_macro")
    regional_module = _load_module(
        SCRIPTS_DIR / "search-qwen-makehuman-regional.py", "length_fit_regional")

    pose_report = json.loads(args.pose_report.read_text(encoding="utf-8"))
    shared_report_path = Path(pose_report["shared_report"])
    shared_report = json.loads(shared_report_path.read_text(encoding="utf-8"))
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
    regional_targets, _modifier_names = regional_module.load_regional_targets(
        targets_dir, macro, len(full_vertices_np), normalization["scale"])
    regional_targets = regional_targets.to(device)
    articulation = base.load_articulation(
        template_path, rig_path, weights_path, body_vertex_count,
        normalization, list(shared_module.POSE_BONES))

    dtype = torch.float32
    full_base = torch.from_numpy(full_vertices_np).to(device)
    macro_targets = {
        name: torch.from_numpy(values).to(device)
        for name, values in macro_arrays.items()
    }
    shape_values = pose_report["frozen_state"]["fitted_parameters"]
    regional_names = list(regional_module.REGIONAL_SPECS)
    macro_values = torch.tensor([
        float(shape_values[name]) for name in shared_module.MACRO_NAMES
    ], device=device, dtype=dtype)
    common_regional_values = torch.tensor([
        float(shape_values[name]) for name in regional_names
    ], device=device, dtype=dtype)
    with torch.no_grad():
        macro_shape = macro.shaped_vertices(full_base, macro_targets, macro_values)
        common_full_rest = regional_module.apply_regional(
            macro_shape, regional_targets, common_regional_values)
        common_body_rest = common_full_rest[:body_vertex_count]
    head_indices = [
        torch.tensor(indices, device=device, dtype=torch.long)
        for indices in articulation["bone_head_vertex_indices"]
    ]
    with torch.no_grad():
        common_bone_heads = shared_module._shaped_bone_heads(
            common_full_rest, head_indices)

    bone_names = articulation["bone_names"]
    joint_bone_indices = [
        bone_names.index(shared_module.JOINT_TO_BONE[name])
        for name in joint_names
    ]
    identity_skin = torch.eye(len(bone_names), device=device, dtype=dtype)
    skin_weights = torch.from_numpy(articulation["weights"]).to(device)
    length_influences = _length_influences(
        bone_names, articulation["parents"], common_bone_heads)
    zero_scale = torch.zeros((), device=device)
    zero_translation = torch.zeros(3, device=device)

    baseline_poses = torch.deg2rad(torch.tensor([
        [pose_report["per_view_pose_axis_angle_degrees"][view][bone]
         for bone in shared_module.POSE_BONES]
        for view in view_names
    ], device=device, dtype=dtype))
    pose_correction_limit_degrees = torch.tensor((
        8.0, 8.0,
        12.0, 12.0,
        18.0, 18.0,
        6.0, 6.0,
        10.0, 10.0,
        15.0, 15.0,
    ), device=device, dtype=dtype)[:, None]
    pose_correction_limits = torch.deg2rad(pose_correction_limit_degrees)

    similarity = pose_report["frozen_state"]["global_body_similarity"]
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

    def posed_joints(
        length_delta: torch.Tensor,
        pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        head_displacement = torch.einsum(
            "s,sbc->bc", length_delta, length_influences)
        bone_heads = common_bone_heads + head_displacement
        all_heads = base.articulate_vertices(
            bone_heads, bone_heads, articulation["parents"], identity_skin,
            articulation["pose_bone_indices"], pose,
            zero_scale, zero_translation)
        local = all_heads[joint_bone_indices]
        world = world_scale * (local @ world_rotation.T) + world_translation
        return local, world, bone_heads, head_displacement

    def project(view_index: int, world_joints: torch.Tensor) -> torch.Tensor:
        return (
            camera_scales[view_index]
            * (world_joints @ camera_rotations[view_index, :2].T)
            + camera_translations[view_index]
        )

    with torch.no_grad():
        baseline_local = []
        baseline_world = []
        baseline_normalized = []
        zero_lengths = torch.zeros(len(LENGTH_SEGMENTS), device=device)
        for index in range(len(cameras)):
            local, world, _heads, _displacement = posed_joints(
                zero_lengths, baseline_poses[index])
            baseline_local.append(local)
            baseline_world.append(world)
            baseline_normalized.append(project(index, world))
        baseline_local = torch.stack(baseline_local)
        baseline_world = torch.stack(baseline_world)
        baseline_normalized = torch.stack(baseline_normalized)

    fitted_poses = []
    fitted_length_deltas = []
    fitted_local_joints = []
    fitted_world_joints = []
    histories = {}
    started = time.monotonic()
    for view_index, view_name in enumerate(view_names):
        length_logits = torch.zeros(
            len(LENGTH_SEGMENTS), device=device, requires_grad=True)
        pose_logits = torch.zeros(
            (len(shared_module.POSE_BONES), 3), device=device,
            requires_grad=True)
        optimizer = torch.optim.Adam(
            (length_logits, pose_logits), lr=args.learning_rate)

        def evaluate():
            length_delta = (
                torch.tanh(length_logits) * args.max_length_fraction)
            pose_correction = torch.tanh(pose_logits) * pose_correction_limits
            pose = baseline_poses[view_index] + pose_correction
            local, world, bone_heads, head_displacement = posed_joints(
                length_delta, pose)
            normalized = project(view_index, world)
            residual_pixels = (
                normalized - target[view_index]) * image_heights[view_index]
            robust = functional.smooth_l1_loss(
                residual_pixels, torch.zeros_like(residual_pixels),
                beta=args.soft_l1_pixels, reduction="none").sum(dim=-1)
            scores = target_weights[view_index]
            reprojection = torch.sum(robust * scores) / torch.sum(scores)
            squared = torch.sum(residual_pixels ** 2, dim=-1)
            rms = torch.sqrt(torch.sum(squared * scores) / torch.sum(scores))
            length_prior = torch.mean(
                (length_delta / args.max_length_fraction) ** 2)
            pose_prior = torch.mean(
                (pose_correction / pose_correction_limits) ** 2)
            loss = (
                reprojection
                + args.length_prior_weight * length_prior
                + args.pose_correction_prior_weight * pose_prior
            )
            return {
                "loss": loss,
                "reprojection": reprojection,
                "rms": rms,
                "length_prior": length_prior,
                "pose_prior": pose_prior,
                "length_delta": length_delta,
                "bone_heads": bone_heads,
                "head_displacement": head_displacement,
                "pose": pose,
                "pose_correction": pose_correction,
                "local": local,
                "world": world,
            }

        with torch.no_grad():
            initial = evaluate()
            best_loss = float(initial["loss"])
            best_iteration = 0
            best_length_logits = length_logits.detach().clone()
            best_pose_logits = pose_logits.detach().clone()
        history = []
        for iteration in range(1, args.iterations_per_view + 1):
            current = evaluate()
            optimizer.zero_grad()
            current["loss"].backward()
            torch.nn.utils.clip_grad_norm_((length_logits, pose_logits), 2.0)
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
                        "weighted_rms_pixels": float(checked["rms"]),
                        "reprojection_loss": float(checked["reprojection"]),
                        "length_prior": float(checked["length_prior"]),
                        "pose_correction_prior": float(checked["pose_prior"]),
                        "total_loss": float(checked["loss"]),
                    }
                    history.append(entry)
                    if float(checked["loss"]) < best_loss:
                        best_loss = float(checked["loss"])
                        best_iteration = iteration
                        best_length_logits = length_logits.detach().clone()
                        best_pose_logits = pose_logits.detach().clone()
        with torch.no_grad():
            length_logits.copy_(best_length_logits)
            pose_logits.copy_(best_pose_logits)
            fitted = evaluate()
            fitted_poses.append(fitted["pose"].cpu().numpy())
            fitted_length_deltas.append(
                fitted["length_delta"].cpu().numpy())
            fitted_local_joints.append(fitted["local"].cpu().numpy())
            fitted_world_joints.append(fitted["world"].cpu().numpy())
        histories[view_name] = {
            "best_iteration": best_iteration,
            "history": history,
        }
        print(json.dumps({
            "view": view_name,
            "rotations_only_rms_pixels": float(initial["rms"]),
            "length_refined_rms_pixels": float(fitted["rms"]),
            "length_fraction_delta": {
                name: float(value)
                for name, value in zip(
                    LENGTH_SEGMENTS, fitted["length_delta"].cpu())
            },
            "best_iteration": best_iteration,
        }), flush=True)

    fitted_poses_np = np.stack(fitted_poses)
    fitted_length_deltas_np = np.stack(fitted_length_deltas)
    fitted_local_joints_np = np.stack(fitted_local_joints)
    fitted_world_joints_np = np.stack(fitted_world_joints)
    with torch.no_grad():
        refined_normalized = torch.stack([
            project(index, torch.from_numpy(fitted_world_joints_np[index]).to(device))
            for index in range(len(cameras))
        ]).cpu().numpy()
    baseline_normalized_np = baseline_normalized.cpu().numpy()
    observed_pixels = shared_module._pixels(observations_np, sizes)
    baseline_pixels = shared_module._pixels(baseline_normalized_np, sizes)
    refined_pixels = shared_module._pixels(refined_normalized, sizes)
    baseline_overall, baseline_per_view = shared_module._metrics(
        baseline_pixels, observed_pixels, observation_weights_np, view_names)
    refined_overall, refined_per_view = shared_module._metrics(
        refined_pixels, observed_pixels, observation_weights_np, view_names)
    baseline_per_view_joint, baseline_per_joint = _joint_metrics(
        baseline_pixels, observed_pixels, observation_weights_np,
        view_names, joint_names)
    refined_per_view_joint, refined_per_joint = _joint_metrics(
        refined_pixels, observed_pixels, observation_weights_np,
        view_names, joint_names)
    baseline_grid = shared_module._draw_overlays(
        args.output_dir, "rotations-only", image_paths, view_names, joint_names,
        observed_pixels, baseline_pixels, observation_weights_np,
        baseline_per_view)
    refined_grid = shared_module._draw_overlays(
        args.output_dir, "length-refined", image_paths, view_names, joint_names,
        observed_pixels, refined_pixels, observation_weights_np,
        refined_per_view)

    pose_artifacts = {}
    faces = np.asarray(faces_np)
    per_view_lengths = {}
    with torch.no_grad():
        for view_index, view_name in enumerate(view_names):
            length_delta = torch.from_numpy(
                fitted_length_deltas_np[view_index]).to(device)
            head_displacement = torch.einsum(
                "s,sbc->bc", length_delta, length_influences)
            bone_heads = common_bone_heads + head_displacement
            body_rest = common_body_rest + skin_weights @ head_displacement
            pose = torch.from_numpy(fitted_poses_np[view_index]).to(device)
            local_mesh = base.articulate_vertices(
                body_rest, bone_heads, articulation["parents"], skin_weights,
                articulation["pose_bone_indices"], pose,
                zero_scale, zero_translation)
            world_mesh = (
                world_scale * (local_mesh @ world_rotation.T) + world_translation
            ).cpu().numpy()
            mesh_path = args.output_dir / f"{view_name}-length-refined-makehuman.ply"
            joint_path = args.output_dir / f"{view_name}-length-refined-joints.ply"
            base._export_mesh(mesh_path, world_mesh, faces)
            shared_module._write_joint_ply(
                joint_path, joint_names, fitted_world_joints_np[view_index])
            pose_artifacts[view_name] = {
                "posed_mesh": str(mesh_path),
                "joints_ply": str(joint_path),
            }
            baseline_lengths = shared_module._segment_lengths(
                joint_names, baseline_local[view_index].cpu().numpy())
            refined_lengths = shared_module._segment_lengths(
                joint_names, fitted_local_joints_np[view_index])
            per_view_lengths[view_name] = {
                "symmetric_fraction_delta": {
                    name: float(value)
                    for name, value in zip(
                        LENGTH_SEGMENTS, fitted_length_deltas_np[view_index])
                },
                "symmetric_scale_factor": {
                    name: float(1.0 + value)
                    for name, value in zip(
                        LENGTH_SEGMENTS, fitted_length_deltas_np[view_index])
                },
                "segment_lengths_makehuman_space": {
                    segment: {
                        "rotations_only": baseline_lengths[segment],
                        "refined": refined_lengths[segment],
                        "change_percent": 100.0 * (
                            refined_lengths[segment] / baseline_lengths[segment] - 1.0),
                    }
                    for segment in baseline_lengths
                },
            }

    camera_digest_after = shared_module._camera_digest(cameras)
    cameras_frozen = (
        camera_digest_before == camera_digest_after
        and torch.equal(camera_rotations, camera_reference[0])
        and torch.equal(camera_scales, camera_reference[1])
        and torch.equal(camera_translations, camera_reference[2])
    )
    if not cameras_frozen:
        raise RuntimeError("frozen cameras changed during length refinement")
    pose_degrees = np.rad2deg(fitted_poses_np)
    per_view_comparison = {}
    for view_name in view_names:
        before = baseline_per_view[view_name]["weighted_rms_pixels"]
        after = refined_per_view[view_name]["weighted_rms_pixels"]
        per_view_comparison[view_name] = {
            "rotations_only_weighted_rms_pixels": before,
            "length_refined_weighted_rms_pixels": after,
            "improvement_pixels": before - after,
            "improvement_percent": 100.0 * (before - after) / before,
        }
    report = {
        "schema": "diffusion-editor.makehuman-dwpose-per-view-length-refinement",
        "schema_version": 1,
        "pose_report": str(args.pose_report.resolve()),
        "hypothesis": (
            "frozen cameras and global body; symmetric per-view arm and leg "
            "length corrections plus small pose re-optimization"
        ),
        "frozen_state": {
            "cameras": cameras_frozen,
            "camera_sha256_before": camera_digest_before,
            "camera_sha256_after": camera_digest_after,
            "global_body_similarity": similarity,
            "common_shape_parameters": shape_values,
            "makehuman_shape_parameters_frozen": True,
            "only_variable_shape_parameters": [],
            "variable_rig_segment_scales": list(LENGTH_SEGMENTS),
        },
        "per_view_lengths": per_view_lengths,
        "per_view_pose_axis_angle_degrees": {
            view_name: {
                bone: values.tolist()
                for bone, values in zip(shared_module.POSE_BONES, pose_degrees[index])
            }
            for index, view_name in enumerate(view_names)
        },
        "per_view_comparison": per_view_comparison,
        "rotations_only_reprojection": {
            "overall": baseline_overall,
            "per_view": baseline_per_view,
            "per_joint": baseline_per_joint,
            "per_view_joint_pixels": baseline_per_view_joint,
        },
        "length_refined_reprojection": {
            "overall": refined_overall,
            "per_view": refined_per_view,
            "per_joint": refined_per_joint,
            "per_view_joint_pixels": refined_per_view_joint,
        },
        "optimization": {
            "iterations_per_view": args.iterations_per_view,
            "learning_rate": args.learning_rate,
            "max_length_fraction": args.max_length_fraction,
            "length_prior_weight": args.length_prior_weight,
            "pose_correction_prior_weight": args.pose_correction_prior_weight,
            "soft_l1_pixels": args.soft_l1_pixels,
            "elapsed_seconds": time.monotonic() - started,
            "per_view": histories,
        },
        "artifacts": {
            "rotations_only_grid": str(baseline_grid),
            "length_refined_grid": str(refined_grid),
            "per_view": pose_artifacts,
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "rotations_only": baseline_overall,
        "length_refined": refined_overall,
        "cameras_frozen": cameras_frozen,
        "report": str(report_path),
        "reprojection_grid": str(refined_grid),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
