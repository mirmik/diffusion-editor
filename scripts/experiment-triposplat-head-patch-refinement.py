#!/usr/bin/env python3
"""Crop a head from a full-body splat and refine its VAE latent from references."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
RENDERED_ROUNDTRIP_SCRIPT = ROOT / "scripts/experiment-triposplat-vae-rendered-features-roundtrip.py"
DEFAULT_EXPERIMENT = Path(
    "/home/mirmik/mnt-nvme/canonical-experiments/"
    "triposplat-process-vaan-4view-then-face5-warmup8-density262k-524k-"
    "render1024-steps33-seed42-run"
)
DEFAULT_OUTPUT = Path(
    "/home/mirmik/mnt-nvme/canonical-experiments/"
    "triposplat-vaan-fullbody-head-patch-refinement"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_EXPERIMENT / "06-final-gaussians/triposplat-524288.ply",
    )
    parser.add_argument("--reference", type=Path, default=Path("/tmp/triposplat-head-tight-source.png"))
    parser.add_argument("--reference-mask", type=Path, default=Path("/tmp/triposplat-head-tight-mask.png"))
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        help="DWPose head-crop manifest for multiview-reference-only conditioning",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-root", type=Path, default=Path("/tmp/TripoSplat-Training"))
    parser.add_argument("--triposplat-root", type=Path, default=Path("/home/mirmik/soft/TripoSplat"))
    parser.add_argument("--checkpoints", type=Path, default=Path("/home/mirmik/mnt-nvme/models/TripoSplat"))
    parser.add_argument("--deps", type=Path, default=Path("/tmp/triposplat-encoder-deps"))
    parser.add_argument("--crop-z-min", type=float, default=0.30)
    parser.add_argument("--crop-abs-x", type=float, default=0.13)
    parser.add_argument("--crop-abs-y", type=float, default=0.15)
    parser.add_argument("--normalized-extent", type=float, default=0.90)
    parser.add_argument("--input-points", type=int, default=16_384)
    parser.add_argument("--output-gaussians", type=int, default=262_144)
    parser.add_argument("--reference-weight", type=float, default=12.0)
    parser.add_argument(
        "--conditioning-mode",
        choices=(
            "self-plus-reference",
            "reference-only",
            "multiview-reference-only",
            "registered-multiview-reference-only",
        ),
        default="self-plus-reference",
        help=(
            "reference-only modes feed no RGB-derived features from crop renders; "
            "renders are then used only for geometric visibility and previews"
        ),
    )
    parser.add_argument("--reference-camera-radius-factor", type=float, default=1.45)
    parser.add_argument("--feature-render-size", type=int, default=1024)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_helpers():
    spec = importlib.util.spec_from_file_location("triposplat_rendered_roundtrip", RENDERED_ROUNDTRIP_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {RENDERED_ROUNDTRIP_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def crop_and_normalize(source: dict[str, torch.Tensor], args: argparse.Namespace):
    means = source["means"]
    mask = (
        (means[:, 2] > args.crop_z_min)
        & (means[:, 0].abs() < args.crop_abs_x)
        & (means[:, 1].abs() < args.crop_abs_y)
    )
    count = int(mask.sum().item())
    if count < args.input_points:
        raise ValueError(f"Head crop has only {count} Gaussians")
    crop = {key: value[mask] for key, value in source.items()}
    bounds_min = crop["means"].amin(dim=0)
    bounds_max = crop["means"].amax(dim=0)
    center = (bounds_min + bounds_max) * 0.5
    world_extent = float((bounds_max - bounds_min).max())
    local_scale = args.normalized_extent / world_extent
    crop["means"] = (crop["means"] - center) * local_scale
    crop["scales"] = crop["scales"] * local_scale
    transform = {
        "world_bounds_min": bounds_min.detach().cpu().tolist(),
        "world_bounds_max": bounds_max.detach().cpu().tolist(),
        "world_center": center.detach().cpu().tolist(),
        "world_extent": world_extent,
        "world_to_local_scale": local_scale,
        "local_to_world_scale": 1.0 / local_scale,
    }
    return crop, mask, transform


def camera_for_direction(
    means: torch.Tensor,
    size: int,
    direction: np.ndarray,
    radius_factor: float,
    fov_degrees: float = 40.0,
) -> dict:
    points = means.detach().float().cpu().numpy()
    bounds_min, bounds_max = points.min(0), points.max(0)
    center = (bounds_min + bounds_max) * 0.5
    extent = float((bounds_max - bounds_min).max())
    direction = np.asarray(direction, dtype=np.float32)
    direction /= np.linalg.norm(direction)
    focal = 0.5 * size / math.tan(math.radians(fov_degrees) * 0.5)
    K = np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]], dtype=np.float32)
    return {
        "azimuth": 0,
        "elevation": 0,
        "view": look_at(
            center + direction * extent * radius_factor,
            center,
            np.array([0, 0, 1], dtype=np.float32),
        ),
        "K": K,
    }


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = np.stack((right, down, forward), axis=0)
    view[:3, 3] = -view[:3, :3] @ eye
    return view


def prepare_reference(image_path: Path, mask_path: Path, size: int) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    white = Image.new("RGB", image.size, "white")
    white.paste(image, mask=mask)
    white = white.resize((size, size), Image.Resampling.LANCZOS)
    tensor = torch.from_numpy(np.array(white, dtype=np.float32) / 255).permute(2, 0, 1).cuda()
    return tensor, white


def prepare_reference_image(image_path: Path, size: int) -> tuple[torch.Tensor, Image.Image]:
    image = Image.open(image_path).convert("RGB").resize(
        (size, size), Image.Resampling.LANCZOS
    )
    tensor = torch.from_numpy(
        np.array(image, dtype=np.float32) / 255
    ).permute(2, 0, 1).cuda()
    return tensor, image


def load_multiview_references(manifest_path: Path) -> list[dict]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != 24:
        raise ValueError("Multiview reference manifest must contain exactly 24 jobs")
    result = []
    for job in jobs:
        path = Path(job["output"]).expanduser()
        if not path.is_file():
            path = manifest_path.parent / path.name
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append({
            "name": str(job["name"]),
            "path": path,
            "azimuth": float(job["azimuth_degrees"]),
            "elevation": float(job["elevation_degrees"]),
            "face_mean_confidence": float(job.get("face_mean_confidence", 1.0)),
            "crop_box": job.get("crop_box"),
            "source_size": job.get("source_size"),
            "full_camera": job.get("full_camera"),
        })
    return result


def direction_for_angles(azimuth: float, elevation: float) -> np.ndarray:
    azimuth_radians = math.radians(azimuth)
    elevation_radians = math.radians(elevation)
    return np.array([
        math.cos(elevation_radians) * math.cos(azimuth_radians),
        math.cos(elevation_radians) * math.sin(azimuth_radians),
        math.sin(elevation_radians),
    ], dtype=np.float32)


def camera_for_registered_crop(
    record: dict,
    transform: dict,
    output_size: int,
) -> dict:
    camera = record.get("full_camera")
    crop_box = record.get("crop_box")
    source_size = record.get("source_size")
    if not isinstance(camera, dict) or not isinstance(crop_box, list):
        raise ValueError(f"{record['name']} has no registered full camera/crop")
    if source_size != [1024, 1024]:
        raise ValueError(
            f"{record['name']} expected a prepared 1024² source, got {source_size}"
        )
    direction = np.asarray(camera["unit_view_direction"], dtype=np.float32)
    distance = float(camera["distance"])
    fov_degrees = float(camera["fov_degrees"])
    world_view = look_at(
        direction * distance,
        np.zeros(3, dtype=np.float32),
        np.array([0, 0, 1], dtype=np.float32),
    )

    # p_world = p_local / scale + head_center.  Multiplying camera-space
    # coordinates by scale leaves perspective projection unchanged, yielding
    # this equivalent rigid view in normalized head coordinates.
    head_center = np.asarray(transform["world_center"], dtype=np.float32)
    local_scale = float(transform["world_to_local_scale"])
    local_view = world_view.copy()
    local_view[:3, 3] = local_scale * (
        world_view[:3, :3] @ head_center + world_view[:3, 3]
    )

    left, top, right, bottom = map(float, crop_box)
    crop_side = max(1, int(round(max(right - left, bottom - top))))
    resize_scale = output_size / crop_side
    focal = 0.5 * 1024 / math.tan(math.radians(fov_degrees) * 0.5)
    K = np.array([
        [focal * resize_scale, 0, (512.0 - left) * resize_scale],
        [0, focal * resize_scale, (512.0 - top) * resize_scale],
        [0, 0, 1],
    ], dtype=np.float32)
    return {
        "azimuth": record["azimuth"],
        "elevation": record["elevation"],
        "view": local_view,
        "K": K,
        "registration": {
            "kind": "full-camera-plus-crop-intrinsics",
            "full_direction": direction.tolist(),
            "full_distance": distance,
            "full_fov_degrees": fov_degrees,
            "crop_box": crop_box,
            "crop_side_rounded": crop_side,
            "resize_scale": resize_scale,
            "principal_point": [float(K[0, 2]), float(K[1, 2])],
        },
    }


def fit_registered_camera_to_reference_foreground(
    camera: dict,
    source: dict[str, torch.Tensor],
    reference_path: Path,
    size: int,
) -> dict:
    image = np.asarray(
        Image.open(reference_path).convert("RGB").resize(
            (size, size), Image.Resampling.LANCZOS
        )
    )
    foreground = image.max(axis=2) > 8
    ys, xs = np.nonzero(foreground)
    if len(xs) < 100:
        raise RuntimeError(f"Cannot isolate reference foreground: {reference_path}")
    desired = np.array([
        np.percentile(xs, 0.2), np.percentile(ys, 0.2),
        np.percentile(xs, 99.8), np.percentile(ys, 99.8),
    ], dtype=np.float32)

    points = source["means"].detach().float().cpu().numpy()
    opacities = source["opacities"].detach().float().cpu().numpy().reshape(-1)
    view = camera["view"]
    camera_points = points @ view[:3, :3].T + view[:3, 3]
    valid = (camera_points[:, 2] > 1e-5) & (opacities > 0.05)
    normalized = camera_points[valid, :2] / camera_points[valid, 2:3]
    low = np.percentile(normalized, 0.5, axis=0)
    high = np.percentile(normalized, 99.5, axis=0)
    projected_extent = np.maximum(high - low, 1e-6)
    desired_extent = np.maximum(desired[2:] - desired[:2], 1.0)
    focal = float(np.min(desired_extent / projected_extent))
    normalized_center = (low + high) * 0.5
    desired_center = (desired[:2] + desired[2:]) * 0.5
    principal = desired_center - normalized_center * focal
    camera["K"] = np.array([
        [focal, 0, principal[0]],
        [0, focal, principal[1]],
        [0, 0, 1],
    ], dtype=np.float32)
    camera["registration"]["foreground_similarity_fit"] = {
        "reference_bounds": desired.tolist(),
        "projected_normalized_bounds": [*low.tolist(), *high.tolist()],
        "focal_pixels": focal,
        "principal_point": principal.tolist(),
    }
    return camera


@torch.inference_mode()
def encode_feature_maps(image: torch.Tensor, dinov3, flux2, rf, seed: int):
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]
    dino_tokens = dinov3(pixel_values=((image - mean) / std)[None].to(torch.bfloat16))
    dino_map = dino_tokens[:, 5:, :].transpose(1, 2).reshape(1, 1280, 64, 64).float()
    flux_image = F.interpolate(image[None], size=(512, 512), mode="bilinear", align_corners=False)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    flux_tokens = flux2.encode(
        flux_image.to(torch.bfloat16) * 2 - 1,
        deterministic=False,
        generator=generator,
    )
    return dino_map, rf.unpatchify_flux2(flux_tokens).float()


@torch.inference_mode()
def project_feature_maps(
    points: torch.Tensor,
    camera: dict,
    size: int,
    dino_map: torch.Tensor,
    flux_map: torch.Tensor,
    rf,
):
    grid = rf.project_grid(points, camera, size)[None, None]
    dino = F.grid_sample(dino_map, grid, mode="bilinear", align_corners=False)[0, :, 0, :].T
    flux = F.grid_sample(flux_map, grid, mode="bilinear", align_corners=False)[0, :, 0, :].T
    return dino, flux


@torch.inference_mode()
def front_visibility_weights(
    source: dict[str, torch.Tensor],
    points: torch.Tensor,
    camera: dict,
    size: int,
    rf,
    depth_tolerance: float = 0.04,
) -> torch.Tensor:
    """Soft z-buffer test for assigning a single-view reference to 3D points."""
    from gsplat import rasterization

    view = torch.from_numpy(camera["view"])[None].to(points.device)
    intrinsics = torch.from_numpy(camera["K"])[None].to(points.device)
    rendered, alpha, _ = rasterization(
        means=source["means"], quats=source["quats"], scales=source["scales"],
        opacities=source["opacities"], colors=source["colors"],
        viewmats=view, Ks=intrinsics, width=size, height=size,
        packed=True, render_mode="RGB+ED",
    )
    depth_map = rendered[0, ..., 3].float()[None, None]
    alpha_map = alpha[0, ..., 0].float()[None, None]
    grid = rf.project_grid(points, camera, size)[None, None]
    sampled_depth = F.grid_sample(
        depth_map, grid, mode="bilinear", align_corners=False,
    )[0, 0, 0]
    sampled_alpha = F.grid_sample(
        alpha_map, grid, mode="bilinear", align_corners=False,
    )[0, 0, 0]
    view_matrix = torch.from_numpy(camera["view"]).to(points.device)
    camera_points = points @ view_matrix[:3, :3].T + view_matrix[:3, 3]
    depth_delta = (camera_points[:, 2] - sampled_depth).abs()
    inside = (grid[0, 0].abs() <= 1).all(dim=-1)
    visibility = torch.exp(-torch.square(depth_delta / depth_tolerance))
    visibility *= (sampled_alpha > 0.05) & inside & (camera_points[:, 2] > 0)
    return visibility.clamp(0, 1)


@torch.inference_mode()
def extract_multiview_reference_features(
    source: dict[str, torch.Tensor],
    points: torch.Tensor,
    records: list[dict],
    dinov3,
    flux2,
    rf,
    args: argparse.Namespace,
    transform: dict,
):
    accumulated_dino = torch.zeros(
        (len(points), 1280), device="cuda", dtype=torch.float32
    )
    accumulated_flux = torch.zeros(
        (len(points), 32), device="cuda", dtype=torch.float32
    )
    accumulated_weight = torch.zeros(
        len(points), device="cuda", dtype=torch.float32
    )
    maximum_visibility = torch.zeros_like(accumulated_weight)
    per_view = []
    target_front = None
    target_front_camera = None

    for index, record in enumerate(records):
        print(
            f"Reference view {index + 1:02d}/{len(records)}: {record['name']} "
            f"az={record['azimuth']:g}, el={record['elevation']:+g}",
            flush=True,
        )
        image, preview = prepare_reference_image(
            record["path"], args.feature_render_size
        )
        if args.conditioning_mode == "registered-multiview-reference-only":
            camera = camera_for_registered_crop(
                record, transform, args.feature_render_size
            )
            camera = fit_registered_camera_to_reference_foreground(
                camera, source, record["path"], args.feature_render_size
            )
        else:
            camera = camera_for_direction(
                source["means"],
                args.feature_render_size,
                direction_for_angles(record["azimuth"], record["elevation"]),
                args.reference_camera_radius_factor,
            )
            camera["azimuth"] = record["azimuth"]
            camera["elevation"] = record["elevation"]
        dino_map, flux_map = encode_feature_maps(
            image, dinov3, flux2, rf, args.seed + 20_000 + index
        )
        view_dino, view_flux = project_feature_maps(
            points, camera, args.feature_render_size, dino_map, flux_map, rf
        )
        visibility = front_visibility_weights(
            source, points, camera, args.feature_render_size, rf
        )
        weight = visibility[:, None]
        accumulated_dino += view_dino * weight
        accumulated_flux += view_flux * weight
        accumulated_weight += visibility
        maximum_visibility = torch.maximum(maximum_visibility, visibility)
        per_view.append({
            "name": record["name"],
            "path": str(record["path"]),
            "azimuth": record["azimuth"],
            "elevation": record["elevation"],
            "face_mean_confidence": record["face_mean_confidence"],
            "camera_registration": camera.get("registration"),
            "visibility_mean": float(visibility.mean().item()),
            "points_above_0_5": int((visibility > 0.5).sum().item()),
            "points_above_0_1": int((visibility > 0.1).sum().item()),
        })
        if record["azimuth"] == 0 and record["elevation"] == 0:
            target_front = preview
            target_front_camera = camera
        del image, dino_map, flux_map, view_dino, view_flux, visibility

    divisor = accumulated_weight.clamp_min(1e-6)[:, None]
    combined_dino = accumulated_dino / divisor
    combined_flux = accumulated_flux / divisor
    unseen = accumulated_weight <= 1e-6
    combined_dino[unseen] = 0
    combined_flux[unseen] = 0
    if target_front is None or target_front_camera is None:
        raise RuntimeError("Multiview manifest has no eye-level azimuth-000 view")
    return (
        combined_dino,
        combined_flux,
        accumulated_weight,
        maximum_visibility,
        per_view,
        target_front,
        target_front_camera,
    )


def install_cpu_fps_bridge():
    import pytorch3d.ops

    native_fps = pytorch3d.ops.sample_farthest_points

    def bridge(values, lengths=None, K=50, random_start_point=False):
        sampled, indices = native_fps(
            values.cpu(),
            lengths=lengths.cpu() if lengths is not None else None,
            K=K,
            random_start_point=random_start_point,
        )
        return sampled.to(values.device), indices.to(values.device)

    pytorch3d.ops.sample_farthest_points = bridge


@torch.inference_mode()
def encode_latent(encoder, points: torch.Tensor, dino: torch.Tensor, flux: torch.Tensor):
    encoder_points = (points + 0.5).clamp(0, 1)[None]
    condition = {
        "points": encoder_points,
        "features": dino.half()[None],
        "points2": encoder_points,
        "features2": flux.half()[None],
    }
    with torch.autocast("cuda", dtype=torch.float16):
        latent, query_points = encoder(
            x=None, cond=condition, sample_posterior=False, return_fps=True,
        )
    return latent.half(), query_points


def save_local_gaussian(base, gaussian, root: Path, name: str):
    gaussian.save_ply(root / f"03-{name}-gaussians-local.ply")
    points, colors = base.gaussian_arrays(gaussian)
    base.write_point_ply(root / f"03-{name}-centers-local.ply", points, colors * 255)
    return points, colors


def write_gaussian_ply(path: Path, source: dict[str, torch.Tensor], rf) -> None:
    means = source["means"].detach().float().cpu().numpy()
    colors = source["colors"].detach().float().cpu().numpy()
    opacities = source["opacities"].detach().float().cpu().numpy()
    scales = source["scales"].detach().float().cpu().numpy()
    quats = source["quats"].detach().float().cpu().numpy()
    exported_means = means @ rf.EXPORT_TRANSFORM.T
    exported_rotations = rf.EXPORT_TRANSFORM[None] @ rf.quat_to_matrix(quats)
    exported_quats = rf.matrix_to_quat(exported_rotations)
    features_dc = (colors - 0.5) / rf.SH_C0
    opacity_logits = np.log(np.clip(opacities, 1e-7, 1 - 1e-7)) - np.log1p(
        -np.clip(opacities, 1e-7, 1 - 1e-7)
    )
    scale_logs = np.log(np.clip(scales, 1e-8, None))
    names = [
        "x", "y", "z", "nx", "ny", "nz",
        "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
        "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    dtype = np.dtype([(name, "<f4") for name in names])
    values = np.empty(len(means), dtype=dtype)
    packed = np.column_stack([
        exported_means, np.zeros_like(exported_means), features_dc,
        opacity_logits, scale_logs, exported_quats,
    ]).astype(np.float32)
    for index, name in enumerate(names):
        values[name] = packed[:, index]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(values)}\n"
        + "".join(f"property float {name}\n" for name in names)
        + "end_header\n"
    ).encode("ascii")
    path.write_bytes(header + values.tobytes())


def render_eye_orbit(rf, source: dict[str, torch.Tensor], size: int) -> list[Image.Image]:
    cameras = [camera for camera in rf.orbit_cameras(source["means"], size) if camera["elevation"] == 0]
    images = []
    for camera in cameras:
        tensor = rf.render_source(source, camera, size)
        images.append(Image.fromarray((tensor.permute(1, 2, 0) * 255).byte().cpu().numpy(), "RGB"))
    return images


def main() -> int:
    args = arguments()
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    for entry in (args.deps, args.training_root, args.triposplat_root):
        sys.path.insert(0, str(entry))
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    rf = load_helpers()
    base = rf.load_base_module()

    print("Loading and cropping full-body splat", flush=True)
    full = rf.load_source_gaussians(args.source, base)
    crop, crop_mask, transform = crop_and_normalize(full, args)
    crop_count = len(crop["means"])
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(crop_count, args.input_points, replace=False)
    points = crop["means"][indices]
    colors = crop["colors"][indices]
    base.write_point_ply(
        args.output / "00-head-crop-input-centers-local.ply",
        points.detach().cpu().numpy(), colors.detach().cpu().numpy() * 255,
    )

    from triposplat import load_decoder, load_dinov3, load_vae_encoder

    print("Loading DINOv3 and FLUX2", flush=True)
    dinov3 = load_dinov3(
        str(args.checkpoints / "clip_vision/dino_v3_vit_h.safetensors"),
        device="cuda", dtype=torch.bfloat16,
    )
    flux2 = load_vae_encoder(
        str(args.checkpoints / "vae/flux2-vae.safetensors"),
        device="cuda", dtype=torch.bfloat16,
    )

    multiview_records = []
    reference_view_stats = []
    reference_weight_sum = None
    if args.conditioning_mode == "self-plus-reference":
        self_cameras = rf.orbit_cameras(crop["means"], args.feature_render_size)
        self_dino, self_flux, source_previews = rf.extract_projected_features(
            crop, points, self_cameras, dinov3, flux2,
            args.feature_render_size, args.output, args.seed,
        )
    else:
        self_cameras = []
        self_dino = torch.zeros((args.input_points, 1280), device="cuda", dtype=torch.float32)
        self_flux = torch.zeros((args.input_points, 32), device="cuda", dtype=torch.float32)
        source_previews = []

    if args.conditioning_mode in (
        "multiview-reference-only",
        "registered-multiview-reference-only",
    ):
        if args.reference_manifest is None:
            raise ValueError(
                "--reference-manifest is required for multiview-reference-only"
            )
        multiview_records = load_multiview_references(args.reference_manifest)
        (
            combined_dino,
            combined_flux,
            reference_weight_sum,
            reference_visibility,
            reference_view_stats,
            reference_preview,
            reference_camera,
        ) = extract_multiview_reference_features(
            crop, points, multiview_records, dinov3, flux2, rf, args, transform
        )
        reference_dino = combined_dino
        reference_flux = combined_flux
        contact_path = args.reference_manifest.parent / "contact-head-crops.jpg"
        if contact_path.is_file():
            Image.open(contact_path).convert("RGB").save(
                args.output / "00-reference-head-crops.jpg", quality=94
            )
    else:
        reference_tensor, reference_preview = prepare_reference(
            args.reference, args.reference_mask, args.feature_render_size
        )
        reference_camera = camera_for_direction(
            crop["means"], args.feature_render_size,
            np.array([1, 0, 0], dtype=np.float32),
            args.reference_camera_radius_factor,
        )
        reference_dino_map, reference_flux_map = encode_feature_maps(
            reference_tensor, dinov3, flux2, rf, args.seed + 20_000
        )
        reference_dino, reference_flux = project_feature_maps(
            points, reference_camera, args.feature_render_size,
            reference_dino_map, reference_flux_map, rf,
        )
        reference_visibility = front_visibility_weights(
            crop, points, reference_camera, args.feature_render_size, rf,
        )
        reference_weight_sum = reference_visibility
        if args.conditioning_mode == "reference-only":
            reference_strength = reference_visibility[:, None]
            combined_dino = reference_dino * reference_strength
            combined_flux = reference_flux * reference_strength
        else:
            self_view_count = float(len(self_cameras))
            reference_strength = args.reference_weight * reference_visibility[:, None]
            combined_dino = (
                self_dino * self_view_count + reference_dino * reference_strength
            ) / (self_view_count + reference_strength)
            combined_flux = (
                self_flux * self_view_count + reference_flux * reference_strength
            ) / (self_view_count + reference_strength)
    torch.save({
        "points": points.cpu(), "source_indices": torch.from_numpy(indices),
        "self_dino": self_dino.half().cpu(), "self_flux2": self_flux.half().cpu(),
        "reference_dino": reference_dino.half().cpu(),
        "reference_flux2": reference_flux.half().cpu(),
        "reference_visibility": reference_visibility.half().cpu(),
        "reference_weight_sum": reference_weight_sum.half().cpu(),
        "combined_dino": combined_dino.half().cpu(),
        "combined_flux2": combined_flux.half().cpu(),
    }, args.output / "01-head-patch-features.pt")
    if source_previews:
        source_front = source_previews[8]
    else:
        source_front_tensor = rf.render_source(
            crop, reference_camera, args.feature_render_size
        )
        source_front = Image.fromarray(
            (source_front_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy(), "RGB"
        ).resize((512, 512), Image.Resampling.LANCZOS)
    target_front = reference_preview.resize((512, 512), Image.Resampling.LANCZOS)
    del dinov3, flux2
    if args.conditioning_mode not in (
        "multiview-reference-only",
        "registered-multiview-reference-only",
    ):
        del reference_dino_map, reference_flux_map, reference_tensor
    torch.cuda.empty_cache()

    from deg.models.gs_seqence_vae.gs_fixlen_vae import ElasticFixedlenEncoder
    from safetensors.torch import load_file

    install_cpu_fps_bridge()
    print("Loading 3D VAE encoder", flush=True)
    encoder = ElasticFixedlenEncoder(
        pcd_pe_mode="pcd_ape_v2", query_mode="fps", model_channels=1024,
        cond_channels=1280, cond_channels2=32, q_token_length=8192,
        latent_channels=16, num_blocks=16, num_heads=16, mlp_ratio=4,
        use_fp16=True, use_2_cross_block=True,
    ).eval().cuda()
    encoder.load_state_dict(load_file(
        str(args.checkpoints / "vae/triposplat_vae_encoder_fp16.safetensors")
    ), strict=True)
    baseline_name = (
        "zero-feature baseline"
        if args.conditioning_mode.endswith("reference-only")
        else "self-render baseline"
    )
    print(f"Encoding {baseline_name}", flush=True)
    self_latent, self_query_points = encode_latent(encoder, points, self_dino, self_flux)
    print("Encoding close-up-refined head patch", flush=True)
    refined_latent, refined_query_points = encode_latent(
        encoder, points, combined_dino, combined_flux
    )
    torch.save({
        "self_latent": self_latent.cpu(), "self_query_points": self_query_points.cpu(),
        "refined_latent": refined_latent.cpu(),
        "refined_query_points": refined_query_points.cpu(),
    }, args.output / "02-head-patch-latents.pt")
    del encoder, self_dino, self_flux, reference_dino, reference_flux, combined_dino, combined_flux
    torch.cuda.empty_cache()

    print("Loading Gaussian decoder", flush=True)
    decoder = load_decoder(
        str(args.checkpoints / "vae/triposplat_vae_decoder_fp16.safetensors"),
        device="cuda", dtype=torch.float16,
    )
    print("Decoding baseline and refined patches", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        self_gaussian = base.decode_detailed(
            decoder, self_latent.cuda(), args.output_gaussians, args.seed + 4000
        )
        refined_gaussian = base.decode_detailed(
            decoder, refined_latent.cuda(), args.output_gaussians, args.seed + 4000
        )
    self_points, _ = save_local_gaussian(base, self_gaussian, args.output, "self-roundtrip")
    refined_points, _ = save_local_gaussian(base, refined_gaussian, args.output, "refined")

    crop_views = render_eye_orbit(rf, crop, args.render_size)
    self_views = render_eye_orbit(rf, rf.gaussian_render_tensors(self_gaussian), args.render_size)
    refined_views = render_eye_orbit(rf, rf.gaussian_render_tensors(refined_gaussian), args.render_size)
    comparison_images = []
    comparison_labels = []
    baseline_short_label = (
        "zero" if args.conditioning_mode.endswith("reference-only") else "self"
    )
    for index, azimuth in enumerate(range(0, 360, 45)):
        comparison_images.extend([crop_views[index], self_views[index], refined_views[index]])
        comparison_labels.extend([
            f"crop {azimuth:03d}",
            f"{baseline_short_label} {azimuth:03d}",
            f"refined {azimuth:03d}",
        ])
    base.contact_sheet(comparison_images, comparison_labels).save(
        args.output / "04-crop-vs-self-vs-refined-orbit.jpg", quality=92
    )
    base.contact_sheet(
        [target_front, source_front, self_views[0], refined_views[0]],
        ["close-up target", "cropped full-body splat", baseline_name, "refined roundtrip"],
    ).save(args.output / "04-front-refinement-stages.jpg", quality=94)

    print("Building deliberately hard full-body merge", flush=True)
    refined_local = rf.gaussian_render_tensors(refined_gaussian)
    local_to_world_scale = float(transform["local_to_world_scale"])
    world_center = torch.tensor(
        transform["world_center"], device="cuda", dtype=torch.float32,
    )
    refined_world = {
        "means": refined_local["means"] * local_to_world_scale + world_center,
        "colors": refined_local["colors"],
        "opacities": refined_local["opacities"],
        "scales": refined_local["scales"] * local_to_world_scale,
        "quats": refined_local["quats"],
    }
    hard_merged = {
        key: torch.cat((full[key][~crop_mask], refined_world[key]), dim=0)
        for key in full
    }
    write_gaussian_ply(args.output / "05-hard-merged-fullbody-gaussians.ply", hard_merged, rf)
    base.write_point_ply(
        args.output / "05-hard-merged-fullbody-centers.ply",
        hard_merged["means"].detach().cpu().numpy(),
        hard_merged["colors"].detach().cpu().numpy() * 255,
    )
    original_body_views = render_eye_orbit(rf, full, args.render_size)
    merged_body_views = render_eye_orbit(rf, hard_merged, args.render_size)
    body_comparison_images = []
    body_comparison_labels = []
    for index, azimuth in enumerate(range(0, 360, 45)):
        body_comparison_images.extend([original_body_views[index], merged_body_views[index]])
        body_comparison_labels.extend([f"original {azimuth:03d}", f"hard merge {azimuth:03d}"])
    base.contact_sheet(body_comparison_images, body_comparison_labels).save(
        args.output / "05-original-vs-hard-merged-fullbody-orbit.jpg", quality=92
    )

    report = {
        "schema": "diffusion-editor.triposplat-head-patch-refinement.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source),
        "reference": (
            str(args.reference_manifest)
            if args.conditioning_mode in (
                "multiview-reference-only",
                "registered-multiview-reference-only",
            )
            else str(args.reference)
        ),
        "conditioning_mode": args.conditioning_mode,
        "crop": {
            "z_min": args.crop_z_min, "abs_x": args.crop_abs_x, "abs_y": args.crop_abs_y,
            "gaussian_count": crop_count, "transform": transform,
        },
        "input_points": args.input_points,
        "self_render_feature_views": len(self_cameras),
        "reference_feature_views": (
            len(multiview_records)
            if args.conditioning_mode in (
                "multiview-reference-only",
                "registered-multiview-reference-only",
            )
            else 1
        ),
        "reference_weight": args.reference_weight,
        "reference_visibility": {
            "mean": float(reference_visibility.mean().item()),
            "points_above_0_5": int((reference_visibility > 0.5).sum().item()),
            "points_above_0_1": int((reference_visibility > 0.1).sum().item()),
            "points_seen_by_any_view": int((reference_weight_sum > 1e-6).sum().item()),
            "weight_sum_mean": float(reference_weight_sum.mean().item()),
            "weight_sum_max": float(reference_weight_sum.max().item()),
        },
        "reference_views": reference_view_stats,
        "reference_camera_radius_factor": args.reference_camera_radius_factor,
        "output_gaussians_per_variant": args.output_gaussians,
        "hard_merged_gaussian_count": len(hard_merged["means"]),
        "self_latent": base.summary(self_latent.float().cpu().numpy()),
        "refined_latent": base.summary(refined_latent.float().cpu().numpy()),
        "self_output_bounds": {"min": self_points.min(0).tolist(), "max": self_points.max(0).tolist()},
        "refined_output_bounds": {
            "min": refined_points.min(0).tolist(), "max": refined_points.max(0).tolist(),
        },
        "self_output_opacity": base.summary(self_gaussian.get_opacity.detach().float().cpu().numpy()),
        "refined_output_opacity": base.summary(
            refined_gaussian.get_opacity.detach().float().cpu().numpy()
        ),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
