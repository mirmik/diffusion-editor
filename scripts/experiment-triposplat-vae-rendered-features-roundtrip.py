#!/usr/bin/env python3
"""Round-trip Gaussian centers with rendered DINOv3/FLUX2 point features."""

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
BASE_SCRIPT = ROOT / "scripts/experiment-triposplat-vae-centers-roundtrip.py"
DEFAULT_SOURCE = Path(
    "/home/mirmik/mnt-nvme/canonical-experiments/"
    "triposplat-process-head-tight-3view-velocity-mean-seed42/"
    "06-final-gaussians/triposplat-262144.ply"
)
DEFAULT_OUTPUT = Path(
    "/home/mirmik/mnt-nvme/canonical-experiments/"
    "triposplat-head-vae-rendered-features-roundtrip"
)
DEFAULT_TRAINING_ROOT = Path("/tmp/TripoSplat-Training")
DEFAULT_TRIPOSPLAT_ROOT = Path("/home/mirmik/soft/TripoSplat")
DEFAULT_CHECKPOINTS = Path("/home/mirmik/mnt-nvme/models/TripoSplat")
DEFAULT_DEPS = Path("/tmp/triposplat-encoder-deps")
SH_C0 = 0.28209479177387814
EXPORT_TRANSFORM = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-root", type=Path, default=DEFAULT_TRAINING_ROOT)
    parser.add_argument("--triposplat-root", type=Path, default=DEFAULT_TRIPOSPLAT_ROOT)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--deps", type=Path, default=DEFAULT_DEPS)
    parser.add_argument("--input-points", type=int, default=16_384)
    parser.add_argument("--output-gaussians", type=int, default=262_144)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--feature-render-size", type=int, default=1024)
    parser.add_argument("--render-size", type=int, default=512)
    return parser.parse_args()


def load_base_module():
    spec = importlib.util.spec_from_file_location("triposplat_vae_roundtrip_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helpers from {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1).reshape(-1, 3, 3)


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    # Stable branch-based conversion, returned as scalar-first WXYZ.
    result = np.empty((len(matrix), 4), dtype=np.float32)
    for index, value in enumerate(matrix):
        trace = float(np.trace(value))
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2.0
            result[index] = [
                0.25 * s,
                (value[2, 1] - value[1, 2]) / s,
                (value[0, 2] - value[2, 0]) / s,
                (value[1, 0] - value[0, 1]) / s,
            ]
        else:
            axis = int(np.argmax(np.diag(value)))
            if axis == 0:
                s = math.sqrt(max(1.0 + value[0, 0] - value[1, 1] - value[2, 2], 0)) * 2
                result[index] = [
                    (value[2, 1] - value[1, 2]) / s, 0.25 * s,
                    (value[0, 1] + value[1, 0]) / s,
                    (value[0, 2] + value[2, 0]) / s,
                ]
            elif axis == 1:
                s = math.sqrt(max(1.0 + value[1, 1] - value[0, 0] - value[2, 2], 0)) * 2
                result[index] = [
                    (value[0, 2] - value[2, 0]) / s,
                    (value[0, 1] + value[1, 0]) / s, 0.25 * s,
                    (value[1, 2] + value[2, 1]) / s,
                ]
            else:
                s = math.sqrt(max(1.0 + value[2, 2] - value[0, 0] - value[1, 1], 0)) * 2
                result[index] = [
                    (value[1, 0] - value[0, 1]) / s,
                    (value[0, 2] + value[2, 0]) / s,
                    (value[1, 2] + value[2, 1]) / s, 0.25 * s,
                ]
    return result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1e-12)


def load_source_gaussians(path: Path, base, device: str = "cuda") -> dict[str, torch.Tensor]:
    ply = base.read_binary_vertex_ply(path)
    exported_means = np.column_stack((ply["x"], ply["y"], ply["z"])).astype(np.float32)
    exported_quats = np.column_stack([ply[f"rot_{i}"] for i in range(4)]).astype(np.float32)
    # Gaussian.save_ply applies EXPORT_TRANSFORM.  Undo it so both this run and
    # the previous XYZ-only run use the decoder's native Z-up coordinates.
    means = exported_means @ EXPORT_TRANSFORM
    rotation_matrices = quat_to_matrix(exported_quats)
    native_rotation_matrices = EXPORT_TRANSFORM.T[None] @ rotation_matrices
    quats = matrix_to_quat(native_rotation_matrices)
    colors = np.clip(
        np.column_stack([ply[f"f_dc_{i}"] for i in range(3)]) * SH_C0 + 0.5,
        0.0,
        1.0,
    ).astype(np.float32)
    opacities = 1.0 / (1.0 + np.exp(-np.asarray(ply["opacity"], dtype=np.float32)))
    scales = np.exp(np.column_stack([ply[f"scale_{i}"] for i in range(3)])).astype(np.float32)
    return {
        "means": torch.from_numpy(means).to(device),
        "colors": torch.from_numpy(colors).to(device),
        "opacities": torch.from_numpy(opacities).to(device),
        "scales": torch.from_numpy(scales).to(device),
        "quats": torch.from_numpy(quats).to(device),
    }


def orbit_cameras(means: torch.Tensor, size: int) -> list[dict]:
    points = means.detach().float().cpu().numpy()
    bounds_min, bounds_max = points.min(0), points.max(0)
    center = (bounds_min + bounds_max) * 0.5
    radius = max(float((bounds_max - bounds_min).max()) * 2.0, 1.0)
    focal = 0.5 * size / math.tan(math.radians(40.0) * 0.5)
    K = np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]], dtype=np.float32)
    cameras = []
    for elevation in (-20, 0, 20):
        elevation_radians = math.radians(elevation)
        for azimuth in range(0, 360, 45):
            azimuth_radians = math.radians(azimuth)
            direction = np.array([
                math.cos(elevation_radians) * math.cos(azimuth_radians),
                math.cos(elevation_radians) * math.sin(azimuth_radians),
                math.sin(elevation_radians),
            ], dtype=np.float32)
            cameras.append({
                "azimuth": azimuth,
                "elevation": elevation,
                "view": look_at(center + direction * radius, center, np.array([0, 0, 1], dtype=np.float32)),
                "K": K,
            })
    return cameras


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


@torch.inference_mode()
def render_source(source: dict[str, torch.Tensor], camera: dict, size: int) -> torch.Tensor:
    from gsplat import rasterization

    view = torch.from_numpy(camera["view"])[None].to(source["means"].device)
    K = torch.from_numpy(camera["K"])[None].to(source["means"].device)
    rendered, alpha, _ = rasterization(
        means=source["means"], quats=source["quats"], scales=source["scales"],
        opacities=source["opacities"], colors=source["colors"],
        viewmats=view, Ks=K, width=size, height=size, packed=True,
    )
    return (rendered[0, ..., :3] + 1.0 - alpha[0]).clamp(0, 1).permute(2, 0, 1)


def project_grid(points: torch.Tensor, camera: dict, size: int) -> torch.Tensor:
    view = torch.from_numpy(camera["view"]).to(points.device)
    camera_points = points @ view[:3, :3].T + view[:3, 3]
    z = camera_points[:, 2].clamp_min(1e-6)
    K = camera["K"]
    pixel_x = float(K[0, 0]) * camera_points[:, 0] / z + float(K[0, 2])
    pixel_y = float(K[1, 1]) * camera_points[:, 1] / z + float(K[1, 2])
    return torch.stack((pixel_x / size * 2 - 1, pixel_y / size * 2 - 1), dim=-1)


def unpatchify_flux2(tokens: torch.Tensor) -> torch.Tensor:
    batch, length, channels = tokens.shape
    side = math.isqrt(length)
    if side * side != length or channels % 4:
        raise ValueError(f"Unexpected FLUX2 token shape: {tuple(tokens.shape)}")
    latents = tokens.transpose(1, 2).reshape(batch, channels, side, side)
    latents = latents.reshape(batch, channels // 4, 2, 2, side, side)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    return latents.reshape(batch, channels // 4, side * 2, side * 2)


@torch.inference_mode()
def extract_projected_features(
    source: dict[str, torch.Tensor],
    points: torch.Tensor,
    cameras: list[dict],
    dinov3,
    flux2,
    size: int,
    output: Path,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[Image.Image]]:
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]
    accumulated_dino = torch.zeros((len(points), 1280), device="cuda", dtype=torch.float32)
    accumulated_flux = torch.zeros((len(points), 32), device="cuda", dtype=torch.float32)
    preview_images = []
    preview_dir = output / "01-source-feature-renders"
    preview_dir.mkdir()

    for index, camera in enumerate(cameras):
        print(
            f"Feature view {index + 1:02d}/{len(cameras)}: "
            f"az={camera['azimuth']:03d}, el={camera['elevation']:+03d}",
            flush=True,
        )
        image = render_source(source, camera, size)
        preview = Image.fromarray((image.permute(1, 2, 0) * 255).byte().cpu().numpy(), "RGB")
        preview = preview.resize((512, 512), Image.Resampling.LANCZOS)
        preview.save(
            preview_dir / f"el-{camera['elevation']:+03d}-az-{camera['azimuth']:03d}.png"
        )
        preview_images.append(preview)

        dino_tokens = dinov3(pixel_values=((image - mean) / std)[None].to(torch.bfloat16))
        dino_map = dino_tokens[:, 5:, :].transpose(1, 2).reshape(1, 1280, 64, 64).float()

        flux_image = F.interpolate(image[None], size=(512, 512), mode="bilinear", align_corners=False)
        generator = torch.Generator(device="cuda").manual_seed(seed + 10_000 + index)
        flux_tokens = flux2.encode(
            flux_image.to(torch.bfloat16) * 2 - 1,
            deterministic=False,
            generator=generator,
        )
        flux_map = unpatchify_flux2(flux_tokens).float()

        grid = project_grid(points, camera, size)[None, None]
        accumulated_dino += F.grid_sample(
            dino_map, grid, mode="bilinear", align_corners=False,
        )[0, :, 0, :].T
        accumulated_flux += F.grid_sample(
            flux_map, grid, mode="bilinear", align_corners=False,
        )[0, :, 0, :].T

    divisor = float(len(cameras))
    return accumulated_dino / divisor, accumulated_flux / divisor, preview_images


def gaussian_render_tensors(gaussian) -> dict[str, torch.Tensor]:
    return {
        "means": gaussian.get_xyz.float(),
        "colors": torch.clamp(gaussian._features_dc[:, 0, :].float() * SH_C0 + 0.5, 0, 1),
        "opacities": gaussian.get_opacity[:, 0].float(),
        "scales": gaussian.get_scaling.float(),
        "quats": (gaussian._rotation + gaussian.rots_bias[None, :]).float(),
    }


def main() -> int:
    args = arguments()
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    for entry in (args.deps, args.training_root, args.triposplat_root):
        sys.path.insert(0, str(entry))
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)
    base = load_base_module()

    print("Loading source Gaussian PLY", flush=True)
    source = load_source_gaussians(args.source, base)
    all_points = source["means"].detach().float().cpu().numpy()
    all_colors = source["colors"].detach().float().cpu().numpy()
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(all_points), args.input_points, replace=False)
    points_np = all_points[indices]
    colors_np = all_colors[indices]
    points = torch.from_numpy(points_np).cuda()
    base.write_point_ply(args.output / "01-encoder-input-centers.ply", points_np, colors_np * 255)

    from triposplat import load_decoder, load_dinov3, load_vae_encoder

    print("Loading DINOv3 and FLUX2 image encoders", flush=True)
    dinov3 = load_dinov3(
        str(args.checkpoints / "clip_vision/dino_v3_vit_h.safetensors"),
        device="cuda", dtype=torch.bfloat16,
    )
    flux2 = load_vae_encoder(
        str(args.checkpoints / "vae/flux2-vae.safetensors"),
        device="cuda", dtype=torch.bfloat16,
    )
    cameras = orbit_cameras(source["means"], args.feature_render_size)
    dino_features, flux_features, source_previews = extract_projected_features(
        source, points, cameras, dinov3, flux2, args.feature_render_size, args.output, args.seed
    )
    torch.save({
        "source_indices": torch.from_numpy(indices),
        "points": torch.from_numpy(points_np),
        "dino": dino_features.half().cpu(),
        "flux2": flux_features.half().cpu(),
    }, args.output / "01-projected-point-features.pt")
    feature_stats = {
        "dino": base.summary(dino_features.cpu().numpy()),
        "flux2": base.summary(flux_features.cpu().numpy()),
    }
    dino_features_cpu = dino_features.half().cpu()
    flux_features_cpu = flux_features.half().cpu()
    del dinov3, flux2, source, dino_features, flux_features
    torch.cuda.empty_cache()

    from deg.models.gs_seqence_vae.gs_fixlen_vae import ElasticFixedlenEncoder
    import pytorch3d.ops
    from safetensors.torch import load_file

    native_fps = pytorch3d.ops.sample_farthest_points

    def cpu_fps_bridge(values, lengths=None, K=50, random_start_point=False):
        sampled, sampled_indices = native_fps(
            values.cpu(), lengths=lengths.cpu() if lengths is not None else None,
            K=K, random_start_point=random_start_point,
        )
        return sampled.to(values.device), sampled_indices.to(values.device)

    pytorch3d.ops.sample_farthest_points = cpu_fps_bridge
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
    encoder_points = (points + 0.5).clamp(0, 1)[None]
    condition = {
        "points": encoder_points,
        "features": dino_features_cpu.cuda()[None],
        "points2": encoder_points,
        "features2": flux_features_cpu.cuda()[None],
    }
    print("Encoding centers with rendered point features", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        latent, query_points = encoder(
            x=None, cond=condition, sample_posterior=False, return_fps=True,
        )
    latent = latent.half()
    torch.save({
        "latent": latent.cpu(), "query_points": query_points.cpu(),
        "source_indices": torch.from_numpy(indices),
    }, args.output / "02-encoded-latent.pt")
    latent_stats = base.summary(latent.float().cpu().numpy())
    del encoder, condition
    torch.cuda.empty_cache()

    print("Loading Gaussian decoder", flush=True)
    decoder = load_decoder(
        str(args.checkpoints / "vae/triposplat_vae_decoder_fp16.safetensors"),
        device="cuda", dtype=torch.float16,
    )
    print(f"Decoding {args.output_gaussians} Gaussians", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        gaussian = base.decode_detailed(
            decoder, latent.cuda(), args.output_gaussians, args.seed + 4000
        )
    gaussian.save_ply(args.output / "03-roundtrip-gaussians.ply")
    output_points, output_colors = base.gaussian_arrays(gaussian)
    base.write_point_ply(
        args.output / "03-roundtrip-centers.ply", output_points, output_colors * 255
    )

    print("Rendering comparison", flush=True)
    output_source = gaussian_render_tensors(gaussian)
    output_cameras = orbit_cameras(output_source["means"], args.render_size)
    output_eye_cameras = [camera for camera in output_cameras if camera["elevation"] == 0]
    output_previews = []
    output_dir = args.output / "04-roundtrip-renders"
    output_dir.mkdir()
    for camera in output_eye_cameras:
        image_tensor = render_source(output_source, camera, args.render_size)
        image = Image.fromarray((image_tensor.permute(1, 2, 0) * 255).byte().cpu().numpy(), "RGB")
        image.save(output_dir / f"azimuth-{camera['azimuth']:03d}.png")
        output_previews.append(image)
    source_eye_previews = source_previews[8:16]
    comparison_images = []
    comparison_labels = []
    for index, azimuth in enumerate(range(0, 360, 45)):
        comparison_images.extend([source_eye_previews[index], output_previews[index]])
        comparison_labels.extend([f"source {azimuth:03d}", f"roundtrip {azimuth:03d}"])
    base.contact_sheet(comparison_images, comparison_labels).save(
        args.output / "04-source-vs-roundtrip-contact-sheet.jpg", quality=92
    )
    base.contact_sheet(output_previews, [f"azimuth {a:03d}" for a in range(0, 360, 45)]).save(
        args.output / "04-roundtrip-orbit-contact-sheet.jpg", quality=92
    )

    report = {
        "schema": "diffusion-editor.triposplat-vae-rendered-features-roundtrip.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "XYZ plus point-projected DINOv3/FLUX2 features from 24 source-splat renders",
        "source": str(args.source),
        "feature_views": len(cameras),
        "feature_view_grid": "8 azimuths x elevations -20/0/+20 degrees",
        "input_center_count_total": len(all_points),
        "input_center_count_encoder": args.input_points,
        "output_gaussian_count": len(output_points),
        "seed": args.seed,
        "features": feature_stats,
        "latent": latent_stats,
        "input_bounds": {"min": points_np.min(0).tolist(), "max": points_np.max(0).tolist()},
        "output_bounds": {"min": output_points.min(0).tolist(), "max": output_points.max(0).tolist()},
        "output_opacity": base.summary(gaussian.get_opacity.detach().float().cpu().numpy()),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
