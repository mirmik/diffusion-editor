#!/usr/bin/env python3
"""Fit one MakeHuman rig and limb lengths to DWPose tracks in frozen cameras."""

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
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as functional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for module_path in (PROJECT_ROOT, SCRIPTS_DIR):
    if str(module_path) not in sys.path:
        sys.path.insert(0, str(module_path))

from diffusion_editor.generation.joint_fitting import estimate_similarity


MACRO_NAMES = ("muscle", "weight", "height", "body_proportions")
OPTIMIZED_MACROS = ("height", "body_proportions")
OPTIMIZED_REGIONAL = (
    "torso_width",
    "torso_height",
    "hip_width",
    "upperarm_length",
    "lowerarm_length",
    "upperlegs_height",
    "lowerlegs_height",
)
POSE_BONES = (
    "clavicle.L", "clavicle.R",
    "upperarm01.L", "upperarm01.R",
    "lowerarm01.L", "lowerarm01.R",
    "pelvis.L", "pelvis.R",
    "upperleg01.L", "upperleg01.R",
    "lowerleg01.L", "lowerleg01.R",
    "foot.L", "foot.R",
)
POSE_LIMIT_DEGREES = (
    30.0, 30.0,
    75.0, 75.0,
    110.0, 110.0,
    20.0, 20.0,
    55.0, 55.0,
    100.0, 100.0,
    55.0, 55.0,
)
# DWPose landmark -> (bone whose transform carries the landmark, endpoint on
# that bone used as the MakeHuman landmark).  The toe bones stay in their rest
# pose and inherit the optimized rigid foot transform; no toe motorics are
# introduced merely to orient the foot.
JOINT_TO_BONE_ENDPOINT = {
    "left_shoulder": ("upperarm01.L", "head"),
    "right_shoulder": ("upperarm01.R", "head"),
    "left_elbow": ("lowerarm01.L", "head"),
    "right_elbow": ("lowerarm01.R", "head"),
    "left_wrist": ("wrist.L", "head"),
    "right_wrist": ("wrist.R", "head"),
    "left_hip": ("upperleg01.L", "head"),
    "right_hip": ("upperleg01.R", "head"),
    "left_knee": ("lowerleg01.L", "head"),
    "right_knee": ("lowerleg01.R", "head"),
    "left_ankle": ("foot.L", "head"),
    "right_ankle": ("foot.R", "head"),
    "left_big_toe": ("toe1-2.L", "tail"),
    "left_small_toe": ("toe5-3.L", "tail"),
    "right_big_toe": ("toe1-2.R", "tail"),
    "right_small_toe": ("toe5-3.R", "tail"),
}
BODY_JOINT_NAMES = tuple(JOINT_TO_BONE_ENDPOINT)[:12]
FOOT_JOINT_NAMES = tuple(JOINT_TO_BONE_ENDPOINT)[12:]
# Compatibility for the older per-view experiments, which intentionally use
# only the original body12 head landmarks.
JOINT_TO_BONE = {
    name: JOINT_TO_BONE_ENDPOINT[name][0]
    for name in BODY_JOINT_NAMES
}
SKELETON_EDGES = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_big_toe"),
    ("left_big_toe", "left_small_toe"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_big_toe"),
    ("right_big_toe", "right_small_toe"),
)
LIMB_SEGMENTS = {
    "left_upperarm": ("left_shoulder", "left_elbow"),
    "right_upperarm": ("right_shoulder", "right_elbow"),
    "left_lowerarm": ("left_elbow", "left_wrist"),
    "right_lowerarm": ("right_elbow", "right_wrist"),
    "left_upperleg": ("left_hip", "left_knee"),
    "right_upperleg": ("right_hip", "right_knee"),
    "left_lowerleg": ("left_knee", "left_ankle"),
    "right_lowerleg": ("right_knee", "right_ankle"),
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
    parser.add_argument("camera_report", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--keypoints", type=Path)
    parser.add_argument(
        "--template", type=Path,
        default=PROJECT_ROOT / "data/templates/makehuman-hm08-base.obj")
    parser.add_argument(
        "--rig", type=Path,
        default=PROJECT_ROOT / "data/templates/makehuman-default.mhskel")
    parser.add_argument(
        "--skin-weights", type=Path,
        default=PROJECT_ROOT / "data/templates/makehuman-default_weights.mhw")
    parser.add_argument("--targets-dir", type=Path, required=True)
    parser.add_argument("--initial-report", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--evaluation-interval", type=int, default=50)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--soft-l1-pixels", type=float, default=7.0)
    parser.add_argument("--shape-learning-rate", type=float, default=0.012)
    parser.add_argument("--pose-learning-rate", type=float, default=0.025)
    parser.add_argument("--alignment-learning-rate", type=float, default=0.008)
    parser.add_argument("--shape-prior-weight", type=float, default=0.06)
    parser.add_argument("--pose-prior-weight", type=float, default=0.025)
    parser.add_argument("--alignment-prior-weight", type=float, default=0.002)
    parser.add_argument("--factorized-3d-prior-weight", type=float, default=0.01)
    parser.add_argument(
        "--reflect-template-forward",
        action="store_true",
        help=(
            "Reflect the shaped MakeHuman rest mesh and rig through local Z "
            "before pose fitting, while keeping cameras and target joints in "
            "their existing world gauge."
        ),
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=20260824)
    return parser.parse_args()


def _inverse_sigmoid(values: torch.Tensor) -> torch.Tensor:
    values = torch.clamp(values, 1.0e-4, 1.0 - 1.0e-4)
    return torch.log(values / (1.0 - values))


def _shaped_bone_heads(
    vertices: torch.Tensor,
    head_indices: list[torch.Tensor],
) -> torch.Tensor:
    return torch.stack([vertices[indices].mean(dim=0) for indices in head_indices])


def _shaped_landmarks(
    vertices: torch.Tensor,
    vertex_indices: list[torch.Tensor],
) -> torch.Tensor:
    return torch.stack([vertices[indices].mean(dim=0) for indices in vertex_indices])


def _camera_digest(cameras: list[dict]) -> str:
    payload = [{
        "name": camera["name"],
        "scale": camera["scale"],
        "translation_normalized": camera["translation_normalized"],
        "world_to_camera_rotation": camera["world_to_camera_rotation"],
    } for camera in cameras]
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _metrics(
    predicted_pixels: np.ndarray,
    observed_pixels: np.ndarray,
    weights: np.ndarray,
    view_names: list[str],
) -> tuple[dict, dict[str, dict]]:
    distances = np.linalg.norm(predicted_pixels - observed_pixels, axis=-1)

    def summarize(values: np.ndarray, scores: np.ndarray) -> dict:
        valid = scores > 0.0
        weighted_rms = math.sqrt(
            float(np.sum(scores[valid] * values[valid] ** 2)
                  / np.sum(scores[valid])))
        return {
            "weighted_rms_pixels": weighted_rms,
            "rms_pixels": float(np.sqrt(np.mean(values[valid] ** 2))),
            "median_pixels": float(np.median(values[valid])),
            "max_pixels": float(np.max(values[valid])),
            "observations": int(np.sum(valid)),
        }

    per_view = {
        name: summarize(distances[index], weights[index])
        for index, name in enumerate(view_names)
    }
    return summarize(distances, weights), per_view


def _pixels(normalized: np.ndarray, sizes: list[tuple[int, int]]) -> np.ndarray:
    result = []
    for values, (width, height) in zip(normalized, sizes):
        result.append(np.stack((
            values[:, 0] * height + width * 0.5,
            height * 0.5 - values[:, 1] * height,
        ), axis=-1))
    return np.stack(result)


def _draw_overlays(
    output_dir: Path,
    label: str,
    paths: list[Path],
    view_names: list[str],
    joint_names: list[str],
    observed: np.ndarray,
    predicted: np.ndarray,
    weights: np.ndarray,
    per_view: dict[str, dict],
) -> Path:
    joint_index = {name: index for index, name in enumerate(joint_names)}
    overlays = []
    for view_index, (path, view_name) in enumerate(zip(paths, view_names)):
        image = Image.open(path).convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        for first, second in SKELETON_EDGES:
            a, b = joint_index[first], joint_index[second]
            if weights[view_index, a] <= 0.0 or weights[view_index, b] <= 0.0:
                continue
            draw.line((*observed[view_index, a], *observed[view_index, b]),
                      fill=(65, 255, 115, 190), width=3)
            draw.line((*predicted[view_index, a], *predicted[view_index, b]),
                      fill=(255, 50, 210, 220), width=3)
        for joint, score in enumerate(weights[view_index]):
            if score <= 0.0:
                continue
            actual = observed[view_index, joint]
            fitted = predicted[view_index, joint]
            draw.line((*actual, *fitted), fill=(255, 215, 40, 170), width=2)
            x, y = actual
            draw.ellipse((x - 5, y - 5, x + 5, y + 5),
                         fill=(65, 255, 115, 240), outline=(0, 30, 0, 255), width=2)
            x, y = fitted
            draw.line((x - 6, y, x + 6, y), fill=(255, 40, 210, 255), width=3)
            draw.line((x, y - 6, x, y + 6), fill=(255, 40, 210, 255), width=3)
        value = per_view[view_name]["weighted_rms_pixels"]
        draw.rectangle((8, 8, min(image.width - 8, 610), 40), fill=(0, 0, 0, 195))
        draw.text((14, 15), f"{label}  {view_name}  joint RMS {value:.1f}px",
                  fill=(255, 255, 255, 255))
        overlay = output_dir / f"{view_name}-{label}-joints.png"
        image.save(overlay)
        overlays.append(image.convert("RGB"))

    thumb_width = 360
    thumbs = [image.resize(
        (thumb_width, round(image.height * thumb_width / image.width)),
        Image.Resampling.LANCZOS) for image in overlays]
    cell_height = max(image.height for image in thumbs)
    sheet = Image.new("RGB", (thumb_width * 3, cell_height * 3), (25, 25, 28))
    for index, image in enumerate(thumbs):
        sheet.paste(image, ((index % 3) * thumb_width, (index // 3) * cell_height))
    target = output_dir / f"{label}-joint-reprojection-grid.png"
    sheet.save(target)
    return target


def _write_joint_ply(
    path: Path,
    names: list[str],
    points: np.ndarray,
) -> None:
    indices = {name: index for index, name in enumerate(names)}
    edges = [(indices[first], indices[second]) for first, second in SKELETON_EDGES]
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(points)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        f"element edge {len(edges)}", "property int vertex1", "property int vertex2",
        "end_header",
    ]
    for point in points:
        lines.append(f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} 255 55 210")
    lines.extend(f"{first} {second}" for first, second in edges)
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _segment_lengths(names: list[str], points: np.ndarray) -> dict[str, float]:
    indices = {name: index for index, name in enumerate(names)}
    return {
        segment: float(np.linalg.norm(
            points[indices[second]] - points[indices[first]]))
        for segment, (first, second) in LIMB_SEGMENTS.items()
    }


def _write_mhm(
    path: Path,
    values: dict[str, float],
    regional_names: list[str],
    modifier_names: list[list[str]],
) -> None:
    lines = [
        "version v1.2.0",
        "name qwen_dwpose_joint_fit",
        "modifier macrodetails/Gender 1.000000",
        "modifier macrodetails/Age 0.500000",
        "modifier macrodetails/African 0.000000",
        "modifier macrodetails/Asian 0.000000",
        "modifier macrodetails/Caucasian 1.000000",
        f"modifier macrodetails-universal/Muscle {values['muscle']:.6f}",
        f"modifier macrodetails-universal/Weight {values['weight']:.6f}",
        f"modifier macrodetails-height/Height {values['height']:.6f}",
        ("modifier macrodetails-proportions/BodyProportions "
         f"{values['body_proportions']:.6f}"),
    ]
    for parameter, modifiers in zip(regional_names, modifier_names):
        lines.extend(f"modifier {name} {values[parameter]:.6f}" for name in modifiers)
    lines.append("skeleton default.mhskel")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.iterations < 1 or args.evaluation_interval < 1:
        raise SystemExit("iterations and evaluation interval must be positive")
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("confidence must be in [0, 1]")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    base = _load_module(SCRIPTS_DIR / "fit-qwen-silhouette-mesh.py", "joint_fit_base")
    macro = _load_module(SCRIPTS_DIR / "fit-qwen-makehuman-macros.py", "joint_fit_macro")
    regional_module = _load_module(
        SCRIPTS_DIR / "search-qwen-makehuman-regional.py", "joint_fit_regional")

    camera_report = json.loads(args.camera_report.read_text(encoding="utf-8"))
    cameras = camera_report.get("cameras")
    if not isinstance(cameras, list) or len(cameras) < 3:
        raise SystemExit("camera report has no frozen camera list")
    camera_digest_before = _camera_digest(cameras)
    keypoint_path = args.keypoints or args.camera_report.parent / "dwpose-keypoints.json"
    keypoint_report = json.loads(keypoint_path.read_text(encoding="utf-8"))
    cache_views = {view["name"]: view for view in keypoint_report["views"]}
    joint_names = list(JOINT_TO_BONE_ENDPOINT)
    body_joint_indices = [joint_names.index(name) for name in BODY_JOINT_NAMES]
    foot_joint_indices = [joint_names.index(name) for name in FOOT_JOINT_NAMES]
    view_names = [camera["name"] for camera in cameras]

    observations = []
    weights = []
    sizes = []
    image_paths = []
    for camera in cameras:
        view = cache_views.get(camera["name"])
        if view is None:
            raise SystemExit(f"keypoint cache has no view {camera['name']}")
        width, height = int(view["width"]), int(view["height"])
        sizes.append((width, height))
        image_paths.append(Path(view["path"]))
        view_observations = []
        view_weights = []
        for name in joint_names:
            point = view["points"][name]
            view_observations.append((
                (float(point["x"]) - width * 0.5) / height,
                (height * 0.5 - float(point["y"])) / height,
            ))
            score = float(point["score"])
            view_weights.append(score if score >= args.confidence else 0.0)
        observations.append(view_observations)
        weights.append(view_weights)
    observations_np = np.asarray(observations, dtype=np.float32)
    weights_np = np.asarray(weights, dtype=np.float32)
    if np.any(np.sum(weights_np > 0.0, axis=0) < 3):
        raise SystemExit("one or more joint tracks have fewer than three observations")

    point_map = {point["name"]: point["xyz"] for point in camera_report["points"]}
    factorized_points_np = np.asarray(
        [point_map[name] for name in BODY_JOINT_NAMES], np.float32)
    camera_rotations_np = np.asarray(
        [camera["world_to_camera_rotation"] for camera in cameras], np.float32)
    camera_scales_np = np.asarray([camera["scale"] for camera in cameras], np.float32)
    camera_translations_np = np.asarray(
        [camera["translation_normalized"] for camera in cameras], np.float32)

    body_vertices_np, faces_np, normalization = base.load_template(
        args.template, 0, None, "body", "y", 0)
    raw_vertices_np = base._obj_vertices(args.template, "y")
    full_vertices_np = (
        raw_vertices_np - normalization["center"][None, :]
    ) * normalization["scale"]
    body_vertex_count = len(body_vertices_np)
    if not np.allclose(
            body_vertices_np, full_vertices_np[:body_vertex_count], atol=1.0e-6):
        raise SystemExit("HM08 body vertices are not the leading full-mesh vertices")

    macro_arrays = macro.load_macro_targets(
        args.targets_dir / "macrodetails", len(full_vertices_np), normalization["scale"])
    regional_targets_np, modifier_names = regional_module.load_regional_targets(
        args.targets_dir, macro, len(full_vertices_np), normalization["scale"])
    articulation = base.load_articulation(
        args.template, args.rig, args.skin_weights, body_vertex_count,
        normalization, list(POSE_BONES))
    rig_data = json.loads(args.rig.read_text(encoding="utf-8"))

    full_base = torch.from_numpy(full_vertices_np).to(device)
    faces = torch.from_numpy(faces_np).to(device)
    macro_targets = {
        name: torch.from_numpy(values).to(device)
        for name, values in macro_arrays.items()
    }
    regional_targets = regional_targets_np.to(device)
    skin_weights = torch.from_numpy(articulation["weights"]).to(device)
    head_indices = [
        torch.tensor(indices, device=device, dtype=torch.long)
        for indices in articulation["bone_head_vertex_indices"]
    ]
    bone_names = articulation["bone_names"]
    landmark_vertex_indices = []
    landmark_bone_indices = []
    for name in joint_names:
        bone, endpoint = JOINT_TO_BONE_ENDPOINT[name]
        rig_joint = rig_data["bones"][bone][endpoint]
        landmark_vertex_indices.append(torch.tensor(
            rig_data["joints"][rig_joint], device=device, dtype=torch.long))
        landmark_bone_indices.append(bone_names.index(bone))
    landmark_skin = torch.zeros(
        (len(joint_names), len(bone_names)),
        device=device, dtype=full_base.dtype)
    landmark_skin[
        torch.arange(len(joint_names), device=device),
        torch.tensor(landmark_bone_indices, device=device),
    ] = 1.0
    zero_scale = torch.zeros((), device=device)
    zero_translation = torch.zeros(3, device=device)

    initial_report = json.loads(args.initial_report.read_text(encoding="utf-8"))
    initial_values = initial_report.get("parameters")
    regional_names = list(regional_module.REGIONAL_SPECS)
    parameter_names = list(MACRO_NAMES) + regional_names
    if not isinstance(initial_values, dict):
        raise SystemExit("initial report has no parameter mapping")
    missing = [name for name in parameter_names if name not in initial_values]
    if missing:
        raise SystemExit("initial report is missing: " + ", ".join(missing))
    initial_macro = torch.tensor(
        [float(initial_values[name]) for name in MACRO_NAMES], device=device)
    initial_regional = torch.tensor(
        [float(initial_values[name]) for name in regional_names], device=device)
    macro_indices = torch.tensor(
        [MACRO_NAMES.index(name) for name in OPTIMIZED_MACROS],
        device=device, dtype=torch.long)
    regional_indices = torch.tensor(
        [regional_names.index(name) for name in OPTIMIZED_REGIONAL],
        device=device, dtype=torch.long)
    macro_logits = _inverse_sigmoid(initial_macro[macro_indices]).detach().requires_grad_(True)
    # Regional MakeHuman modifiers are piecewise-linear targets.  Optimize
    # their values directly: several useful initial values are exactly +/-1,
    # where a tanh parameterization would suppress the very gradient needed to
    # move them back into the range.  A differentiable range penalty below
    # prevents material extrapolation outside MakeHuman's conventional bounds.
    regional_values_parameter = (
        initial_regional[regional_indices].clone().detach().requires_grad_(True)
    )
    pose_limits = torch.deg2rad(torch.tensor(
        POSE_LIMIT_DEGREES, device=device, dtype=full_base.dtype))[:, None]
    pose_logits = torch.zeros(
        (len(POSE_BONES), 3), device=device, requires_grad=True)

    def parameter_values() -> tuple[torch.Tensor, torch.Tensor]:
        macros = initial_macro.scatter(0, macro_indices, torch.sigmoid(macro_logits))
        regional = initial_regional.scatter(
            0, regional_indices, regional_values_parameter)
        return macros, regional

    def rest_shape() -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
    ]:
        macros, regional = parameter_values()
        shaped = macro.shaped_vertices(full_base, macro_targets, macros)
        full_rest = regional_module.apply_regional(shaped, regional_targets, regional)
        if args.reflect_template_forward:
            full_rest = full_rest * full_rest.new_tensor((1.0, 1.0, -1.0))
        bone_heads = _shaped_bone_heads(full_rest, head_indices)
        landmarks = _shaped_landmarks(full_rest, landmark_vertex_indices)
        return full_rest, full_rest[:body_vertex_count], bone_heads, landmarks

    def local_joint_positions(
        bone_heads: torch.Tensor,
        rest_landmarks: torch.Tensor,
        pose: torch.Tensor,
    ) -> torch.Tensor:
        return base.articulate_vertices(
            rest_landmarks, bone_heads, articulation["parents"], landmark_skin,
            articulation["pose_bone_indices"], pose, zero_scale, zero_translation)

    with torch.no_grad():
        _initial_full, _initial_body, initial_heads, initial_landmarks = rest_shape()
        neutral_joints = local_joint_positions(
            initial_heads, initial_landmarks, torch.zeros_like(pose_logits)).cpu().numpy()
    alignment = estimate_similarity(
        neutral_joints[body_joint_indices], factorized_points_np,
        np.mean(weights_np[:, body_joint_indices], axis=0))
    initial_rotation = torch.tensor(
        alignment.rotation, device=device, dtype=full_base.dtype)
    initial_scale = torch.tensor(alignment.scale, device=device, dtype=full_base.dtype)
    initial_translation = torch.tensor(
        alignment.translation, device=device, dtype=full_base.dtype)
    global_rotation_delta = torch.zeros(3, device=device, requires_grad=True)
    global_log_scale_delta = torch.zeros((), device=device, requires_grad=True)
    global_translation_delta = torch.zeros(3, device=device, requires_grad=True)

    target = torch.from_numpy(observations_np).to(device)
    target_weights = torch.from_numpy(weights_np).to(device)
    factorized_points = torch.from_numpy(factorized_points_np).to(device)
    camera_rotations = torch.from_numpy(camera_rotations_np).to(device)
    camera_scales = torch.from_numpy(camera_scales_np).to(device)
    camera_translations = torch.from_numpy(camera_translations_np).to(device)
    camera_rotations_reference = camera_rotations.clone()
    camera_scales_reference = camera_scales.clone()
    camera_translations_reference = camera_translations.clone()
    image_heights = torch.tensor(
        [height for _width, height in sizes], device=device, dtype=full_base.dtype)
    factorized_extent = torch.linalg.vector_norm(
        factorized_points.max(dim=0).values - factorized_points.min(dim=0).values)

    def state():
        macros, regional = parameter_values()
        full_rest, body_rest, bone_heads, rest_landmarks = rest_shape()
        pose = torch.tanh(pose_logits) * pose_limits
        local_joints = local_joint_positions(bone_heads, rest_landmarks, pose)
        delta_rotation = base.axis_angle_to_matrix(global_rotation_delta[None])[0]
        world_rotation = delta_rotation @ initial_rotation
        world_scale = initial_scale * torch.exp(global_log_scale_delta)
        world_translation = initial_translation + global_translation_delta
        world_joints = world_scale * (local_joints @ world_rotation.T) + world_translation
        normalized = (
            camera_scales[:, None, None]
            * torch.einsum("vij,pj->vpi", camera_rotations[:, :2], world_joints)
            + camera_translations[:, None, :]
        )
        residual_pixels = (
            normalized - target) * image_heights[:, None, None]
        robust = functional.smooth_l1_loss(
            residual_pixels, torch.zeros_like(residual_pixels),
            beta=args.soft_l1_pixels, reduction="none").sum(dim=-1)
        reprojection_loss = torch.sum(robust * target_weights) / torch.sum(target_weights)
        squared_pixels = torch.sum(residual_pixels ** 2, dim=-1)
        weighted_rms = torch.sqrt(
            torch.sum(squared_pixels * target_weights) / torch.sum(target_weights))
        regional_range_penalty = torch.mean((
            functional.relu(torch.abs(regional[regional_indices]) - 1.0) / 0.05
        ) ** 2)
        shape_prior = (
            torch.mean(((macros[macro_indices] - initial_macro[macro_indices]) / 0.25) ** 2)
            + torch.mean(((regional[regional_indices] - initial_regional[regional_indices]) / 0.5) ** 2)
            + regional_range_penalty
        )
        pose_prior = torch.mean((pose / pose_limits) ** 2)
        alignment_prior = (
            torch.mean((global_rotation_delta / math.radians(20.0)) ** 2)
            + (global_log_scale_delta / 0.20) ** 2
            + torch.mean((global_translation_delta / 0.05) ** 2)
        )
        factorized_3d_prior = torch.mean(
            ((world_joints[body_joint_indices] - factorized_points)
             / factorized_extent) ** 2)
        loss = (
            reprojection_loss
            + args.shape_prior_weight * shape_prior
            + args.pose_prior_weight * pose_prior
            + args.alignment_prior_weight * alignment_prior
            + args.factorized_3d_prior_weight * factorized_3d_prior
        )
        return {
            "loss": loss,
            "reprojection_loss": reprojection_loss,
            "weighted_rms": weighted_rms,
            "shape_prior": shape_prior,
            "pose_prior": pose_prior,
            "alignment_prior": alignment_prior,
            "factorized_3d_prior": factorized_3d_prior,
            "macros": macros,
            "regional": regional,
            "pose": pose,
            "full_rest": full_rest,
            "body_rest": body_rest,
            "bone_heads": bone_heads,
            "rest_landmarks": rest_landmarks,
            "world_rotation": world_rotation,
            "world_scale": world_scale,
            "world_translation": world_translation,
            "local_joints": local_joints,
            "world_joints": world_joints,
            "normalized": normalized,
        }

    optimized_tensors = (
        macro_logits, regional_values_parameter, pose_logits,
        global_rotation_delta, global_log_scale_delta, global_translation_delta,
    )
    optimizer = torch.optim.Adam((
        {"params": (macro_logits, regional_values_parameter), "lr": args.shape_learning_rate},
        {"params": (pose_logits,), "lr": args.pose_learning_rate},
        {"params": (
            global_rotation_delta, global_log_scale_delta,
            global_translation_delta), "lr": args.alignment_learning_rate},
    ))

    with torch.no_grad():
        initial_state = state()
        initial_normalized = initial_state["normalized"].cpu().numpy()
        initial_local_joints = initial_state["local_joints"].cpu().numpy()
        initial_world_joints = initial_state["world_joints"].cpu().numpy()
        best_loss = float(initial_state["loss"])
        best_iteration = 0
        best_values = [value.detach().clone() for value in optimized_tensors]
    history = []
    started = time.monotonic()
    print(json.dumps({
        "iteration": 0,
        "weighted_rms_pixels": float(initial_state["weighted_rms"]),
        "total_loss": best_loss,
    }), flush=True)

    for iteration in range(1, args.iterations + 1):
        current = state()
        optimizer.zero_grad()
        current["loss"].backward()
        torch.nn.utils.clip_grad_norm_(optimized_tensors, 2.0)
        optimizer.step()
        if (
            iteration == 1
            or iteration % args.evaluation_interval == 0
            or iteration == args.iterations
        ):
            with torch.no_grad():
                evaluated = state()
                total = float(evaluated["loss"])
                entry = {
                    "iteration": iteration,
                    "weighted_rms_pixels": float(evaluated["weighted_rms"]),
                    "reprojection_loss": float(evaluated["reprojection_loss"]),
                    "total_loss": total,
                    "shape_prior": float(evaluated["shape_prior"]),
                    "pose_prior": float(evaluated["pose_prior"]),
                    "alignment_prior": float(evaluated["alignment_prior"]),
                    "factorized_3d_prior": float(evaluated["factorized_3d_prior"]),
                }
                history.append(entry)
                if total < best_loss:
                    best_loss = total
                    best_iteration = iteration
                    best_values = [value.detach().clone() for value in optimized_tensors]
            print(json.dumps(entry), flush=True)

    with torch.no_grad():
        for destination, value in zip(optimized_tensors, best_values):
            destination.copy_(value)
        fitted = state()
        pose_without_foot_rotation = fitted["pose"].clone()
        for bone in ("foot.L", "foot.R"):
            pose_without_foot_rotation[POSE_BONES.index(bone)] = 0.0
        local_without_foot_rotation = local_joint_positions(
            fitted["bone_heads"], fitted["rest_landmarks"],
            pose_without_foot_rotation)
        ankle_indices = [
            joint_names.index("left_ankle"),
            joint_names.index("right_ankle"),
        ]
        toe_indices = foot_joint_indices
        ankle_pivot_drift = torch.linalg.vector_norm(
            fitted["local_joints"][ankle_indices]
            - local_without_foot_rotation[ankle_indices], dim=1)
        toe_motion = torch.linalg.vector_norm(
            fitted["local_joints"][toe_indices]
            - local_without_foot_rotation[toe_indices], dim=1)
        if float(ankle_pivot_drift.max()) > 1.0e-6:
            raise RuntimeError("foot rotation moved its ankle pivot")
        final_normalized = fitted["normalized"].cpu().numpy()
        body_posed = base.articulate_vertices(
            fitted["body_rest"], fitted["bone_heads"], articulation["parents"],
            skin_weights, articulation["pose_bone_indices"], fitted["pose"],
            zero_scale, zero_translation)
        body_world = (
            fitted["world_scale"] * (body_posed @ fitted["world_rotation"].T)
            + fitted["world_translation"]
        )

    frozen_tensors_unchanged = (
        torch.equal(camera_rotations, camera_rotations_reference)
        and torch.equal(camera_scales, camera_scales_reference)
        and torch.equal(camera_translations, camera_translations_reference)
    )
    camera_digest_after = _camera_digest(cameras)
    cameras_frozen = frozen_tensors_unchanged and camera_digest_before == camera_digest_after
    if not cameras_frozen:
        raise RuntimeError("frozen camera data changed during optimization")

    observed_pixels = _pixels(observations_np, sizes)
    initial_pixels = _pixels(initial_normalized, sizes)
    final_pixels = _pixels(final_normalized, sizes)
    initial_overall, initial_per_view = _metrics(
        initial_pixels, observed_pixels, weights_np, view_names)
    final_overall, final_per_view = _metrics(
        final_pixels, observed_pixels, weights_np, view_names)
    initial_body_overall, initial_body_per_view = _metrics(
        initial_pixels[:, body_joint_indices],
        observed_pixels[:, body_joint_indices],
        weights_np[:, body_joint_indices], view_names)
    final_body_overall, final_body_per_view = _metrics(
        final_pixels[:, body_joint_indices],
        observed_pixels[:, body_joint_indices],
        weights_np[:, body_joint_indices], view_names)
    initial_foot_overall, initial_foot_per_view = _metrics(
        initial_pixels[:, foot_joint_indices],
        observed_pixels[:, foot_joint_indices],
        weights_np[:, foot_joint_indices], view_names)
    final_foot_overall, final_foot_per_view = _metrics(
        final_pixels[:, foot_joint_indices],
        observed_pixels[:, foot_joint_indices],
        weights_np[:, foot_joint_indices], view_names)
    initial_grid = _draw_overlays(
        args.output_dir, "initial", image_paths, view_names, joint_names,
        observed_pixels, initial_pixels, weights_np, initial_per_view)
    final_grid = _draw_overlays(
        args.output_dir, "fitted", image_paths, view_names, joint_names,
        observed_pixels, final_pixels, weights_np, final_per_view)

    posed_mesh = args.output_dir / "makehuman-posed-fitted.ply"
    export_faces = faces
    if args.reflect_template_forward:
        # Reflection reverses mesh handedness.  Restore outward winding for
        # consumers that render triangles rather than the diagnostic points.
        export_faces = faces[:, (0, 2, 1)]
    base._export_mesh(
        posed_mesh, body_world.cpu().numpy(), export_faces.cpu().numpy())
    joints_ply = args.output_dir / "makehuman-fitted-joints.ply"
    _write_joint_ply(joints_ply, joint_names, fitted["world_joints"].cpu().numpy())

    final_parameters = {
        name: float(value)
        for name, value in zip(
            parameter_names,
            torch.cat((fitted["macros"], fitted["regional"])).cpu(),
        )
    }
    mhm_path = args.output_dir / "makehuman-fitted-shape.mhm"
    _write_mhm(mhm_path, final_parameters, regional_names, modifier_names)
    pose_degrees = torch.rad2deg(fitted["pose"]).cpu().numpy()
    report = {
        "schema": "diffusion-editor.makehuman-dwpose-frozen-camera-joint-fit",
        "schema_version": 1,
        "inputs": {
            "camera_report": str(args.camera_report.resolve()),
            "keypoints": str(keypoint_path.resolve()),
            "initial_report": str(args.initial_report.resolve()),
            "template": str(args.template.resolve()),
            "rig": str(args.rig.resolve()),
            "skin_weights": str(args.skin_weights.resolve()),
            "targets_dir": str(args.targets_dir.resolve()),
        },
        "camera_policy": {
            "frozen": cameras_frozen,
            "optimized_camera_parameters": [],
            "camera_count": len(cameras),
            "sha256_before": camera_digest_before,
            "sha256_after": camera_digest_after,
            "projection": camera_report.get("projection"),
        },
        "template_forward_policy": {
            "source_forward_axis": "+Z",
            "reflected_before_pose_fit": args.reflect_template_forward,
            "world_joint_targets_unchanged": True,
        },
        "joint_to_makehuman_landmark": {
            name: {"bone": bone, "endpoint": endpoint}
            for name, (bone, endpoint) in JOINT_TO_BONE_ENDPOINT.items()
        },
        "joint_groups": {
            "body12": list(BODY_JOINT_NAMES),
            "rigid_foot_targets": list(FOOT_JOINT_NAMES),
        },
        "foot_target_confidence": {
            name: {
                "minimum": float(weights_np[:, joint_names.index(name)].min()),
                "mean": float(weights_np[:, joint_names.index(name)].mean()),
                "maximum": float(weights_np[:, joint_names.index(name)].max()),
            }
            for name in FOOT_JOINT_NAMES
        },
        "foot_rotation_validation": {
            "ankle_pivot_max_drift_makehuman_space": float(
                ankle_pivot_drift.max()),
            "toe_motion_without_vs_with_foot_rotation_makehuman_space": {
                name: float(value)
                for name, value in zip(FOOT_JOINT_NAMES, toe_motion)
            },
        },
        "optimized_shape_parameters": list(OPTIMIZED_MACROS + OPTIMIZED_REGIONAL),
        "initial_parameters": {name: float(initial_values[name]) for name in parameter_names},
        "fitted_parameters": final_parameters,
        "shared_pose_axis_angle_degrees": {
            bone: values.tolist() for bone, values in zip(POSE_BONES, pose_degrees)
        },
        "limb_segment_lengths_in_factorization_world": {
            "initial": _segment_lengths(
                joint_names, initial_world_joints),
            "fitted": _segment_lengths(
                joint_names, fitted["world_joints"].cpu().numpy()),
        },
        "limb_segment_lengths_in_makehuman_space": {
            "initial": _segment_lengths(
                joint_names, initial_local_joints),
            "fitted": _segment_lengths(
                joint_names, fitted["local_joints"].cpu().numpy()),
        },
        "global_body_similarity": {
            "scale": float(fitted["world_scale"]),
            "rotation": fitted["world_rotation"].cpu().numpy().tolist(),
            "translation": fitted["world_translation"].cpu().numpy().tolist(),
        },
        "optimization": {
            "iterations": args.iterations,
            "best_iteration": best_iteration,
            "elapsed_seconds": time.monotonic() - started,
            "confidence_threshold": args.confidence,
            "soft_l1_pixels": args.soft_l1_pixels,
            "weights": {
                "shape_prior": args.shape_prior_weight,
                "pose_prior": args.pose_prior_weight,
                "alignment_prior": args.alignment_prior_weight,
                "factorized_3d_prior": args.factorized_3d_prior_weight,
            },
            "history": history,
        },
        "initial_reprojection": {
            "overall": initial_overall,
            "per_view": initial_per_view,
            "body12": {
                "overall": initial_body_overall,
                "per_view": initial_body_per_view,
            },
            "rigid_foot_targets": {
                "overall": initial_foot_overall,
                "per_view": initial_foot_per_view,
            },
        },
        "fitted_reprojection": {
            "overall": final_overall,
            "per_view": final_per_view,
            "body12": {
                "overall": final_body_overall,
                "per_view": final_body_per_view,
            },
            "rigid_foot_targets": {
                "overall": final_foot_overall,
                "per_view": final_foot_per_view,
            },
        },
        "fitted_world_joints": [
            {"name": name, "xyz": point.tolist()}
            for name, point in zip(joint_names, fitted["world_joints"].cpu().numpy())
        ],
        "artifacts": {
            "initial_reprojection_grid": str(initial_grid),
            "fitted_reprojection_grid": str(final_grid),
            "posed_mesh": str(posed_mesh),
            "joints_ply": str(joints_ply),
            "makehuman_shape": str(mhm_path),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "initial": initial_overall,
        "fitted": final_overall,
        "best_iteration": best_iteration,
        "cameras_frozen": cameras_frozen,
        "report": str(report_path),
        "reprojection_grid": str(final_grid),
        "posed_mesh": str(posed_mesh),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
