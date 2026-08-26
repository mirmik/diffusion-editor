#!/usr/bin/env python3
"""Round-trip TripoSplat Gaussian centers through the released 3D VAE.

This deliberately performs a geometry-only ablation.  The public encoder was
trained with DINOv3 and FLUX2 features attached to every point, but here both
feature tensors are zero.  Consequently the only input information is XYZ.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


DEFAULT_SOURCE = Path(
    "/home/mirmik/mnt-nvme/canonical-experiments/"
    "triposplat-process-head-tight-3view-velocity-mean-seed42/"
    "06-final-gaussians/points-262144.ply"
)
DEFAULT_OUTPUT = Path(
    "/home/mirmik/mnt-nvme/canonical-experiments/"
    "triposplat-head-vae-centers-zero-features-roundtrip"
)
DEFAULT_TRAINING_ROOT = Path("/tmp/TripoSplat-Training")
DEFAULT_TRIPOSPLAT_ROOT = Path("/home/mirmik/soft/TripoSplat")
DEFAULT_CHECKPOINTS = Path("/home/mirmik/mnt-nvme/models/TripoSplat")
DEFAULT_DEPS = Path("/tmp/triposplat-encoder-deps")
SH_C0 = 0.28209479177387814


def parse_args() -> argparse.Namespace:
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
    parser.add_argument("--render-size", type=int, default=512)
    return parser.parse_args()


def read_binary_vertex_ply(path: Path) -> dict[str, np.ndarray]:
    scalar_types = {
        "float": "<f4", "float32": "<f4", "double": "<f8",
        "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
        "ushort": "<u2", "uint16": "<u2", "int": "<i4", "int32": "<i4",
    }
    properties: list[tuple[str, str]] = []
    vertex_count = None
    with path.open("rb") as stream:
        if stream.readline().strip() != b"ply":
            raise ValueError(f"Not a PLY file: {path}")
        if stream.readline().strip() != b"format binary_little_endian 1.0":
            raise ValueError("Only binary_little_endian PLY is supported")
        in_vertices = False
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError("PLY header has no end_header")
            line = raw.decode("ascii").strip()
            if line.startswith("element "):
                _, name, count = line.split()
                in_vertices = name == "vertex"
                if in_vertices:
                    vertex_count = int(count)
            elif line.startswith("property ") and in_vertices:
                parts = line.split()
                if parts[1] == "list":
                    raise ValueError("List properties are not supported")
                properties.append((parts[2], scalar_types[parts[1]]))
            elif line == "end_header":
                break
        if vertex_count is None:
            raise ValueError("PLY has no vertex element")
        values = np.fromfile(stream, dtype=np.dtype(properties), count=vertex_count)
    return {name: np.asarray(values[name]) for name, _ in properties}


def write_point_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    xyz = np.ascontiguousarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.ascontiguousarray(np.clip(rgb, 0, 255), dtype=np.uint8).reshape(-1, 3)
    dtype = np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ])
    values = np.empty(len(xyz), dtype=dtype)
    values["x"], values["y"], values["z"] = xyz.T
    values["red"], values["green"], values["blue"] = rgb.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(values)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    path.write_bytes(header + values.tobytes())


def gaussian_arrays(gaussian) -> tuple[np.ndarray, np.ndarray]:
    points = gaussian.get_xyz.detach().float().cpu().numpy()
    colors = np.clip(
        gaussian._features_dc.detach().float().cpu().numpy()[:, 0, :] * SH_C0 + 0.5,
        0.0,
        1.0,
    )
    return points, colors


def decode_detailed(decoder, latent: torch.Tensor, count: int, seed: int):
    from model import OctreeProbabilityFixedlenDecoder
    from triposplat import _build_gaussians

    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state()
    try:
        torch.manual_seed(seed)
        anchors = max(1, count // decoder.gaussians_per_point)
        points = OctreeProbabilityFixedlenDecoder.sample(
            decoder.octree,
            latent,
            num_points=anchors,
            level=decoder._MAX_VOXEL_LEVEL,
            temperature=1.0,
            algo="systematic",
        )
        attributes = decoder.gs(x=points, cond=latent)
        return _build_gaussians(decoder.gs, points, attributes)[0]
    finally:
        torch.set_rng_state(cpu_state)
        torch.cuda.set_rng_state(cuda_state)


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
def render_orbit(gaussian, size: int) -> tuple[list[Image.Image], list[str]]:
    from gsplat import rasterization

    means = gaussian.get_xyz.float()
    colors = torch.clamp(gaussian._features_dc[:, 0, :].float() * SH_C0 + 0.5, 0, 1)
    opacities = gaussian.get_opacity[:, 0].float()
    scales = gaussian.get_scaling.float()
    quats = (gaussian._rotation + gaussian.rots_bias[None, :]).float()
    bounds_min, bounds_max = means.amin(dim=0), means.amax(dim=0)
    center = ((bounds_min + bounds_max) * 0.5).cpu().numpy()
    radius = max(float((bounds_max - bounds_min).max()) * 2.0, 1.0)
    directions = [
        np.array([math.cos(math.radians(a)), math.sin(math.radians(a)), 0.0], dtype=np.float32)
        for a in range(0, 360, 45)
    ]
    views = np.stack([
        look_at(center + direction * radius, center, np.array([0, 0, 1], dtype=np.float32))
        for direction in directions
    ])
    focal = 0.5 * size / math.tan(math.radians(40.0) * 0.5)
    K = np.array([[focal, 0, size / 2], [0, focal, size / 2], [0, 0, 1]], dtype=np.float32)
    rendered, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=torch.from_numpy(views).to(means.device),
        Ks=torch.from_numpy(np.repeat(K[None], len(views), axis=0)).to(means.device),
        width=size,
        height=size,
        packed=True,
    )
    rendered = rendered[..., :3] + 1.0 - alpha
    images = [
        Image.fromarray((frame * 255).clamp(0, 255).byte().cpu().numpy(), "RGB")
        for frame in rendered
    ]
    return images, [f"azimuth {a:03d}" for a in range(0, 360, 45)]


def contact_sheet(images: list[Image.Image], labels: list[str]) -> Image.Image:
    width, height = images[0].size
    columns = 4
    rows = math.ceil(len(images) / columns)
    label_height = 30
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "#202020")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index % columns * width
        y = index // columns * (height + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 8, y + height + 5), label, fill="white", font=font)
    return sheet


def summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float32)
    return {
        "shape": list(values.shape),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
    }


def main() -> int:
    args = parse_args()
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    for entry in (args.deps, args.training_root, args.triposplat_root):
        sys.path.insert(0, str(entry))

    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    source = read_binary_vertex_ply(args.source)
    all_points = np.column_stack((source["x"], source["y"], source["z"])).astype(np.float32)
    if {"red", "green", "blue"}.issubset(source):
        all_colors = np.column_stack((source["red"], source["green"], source["blue"]))
    else:
        all_colors = np.full((len(all_points), 3), 200, dtype=np.uint8)
    if len(all_points) < args.input_points:
        raise SystemExit(f"Need {args.input_points} points, source has {len(all_points)}")
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(all_points), args.input_points, replace=False)
    points = all_points[indices]
    colors = all_colors[indices]
    write_point_ply(args.output / "01-encoder-input-centers.ply", points, colors)

    from deg.models.gs_seqence_vae.gs_fixlen_vae import ElasticFixedlenEncoder
    import pytorch3d.ops
    from safetensors.torch import load_file
    from triposplat import load_decoder

    # The available PyTorch3D wheel has its FPS kernel for CPU only.  Keep the
    # released encoder unmodified and bridge just that one preprocessing op
    # back to CUDA.  FPS is non-learned and produces the same query-point data.
    native_sample_farthest_points = pytorch3d.ops.sample_farthest_points

    def sample_farthest_points_cpu_bridge(points, lengths=None, K=50, random_start_point=False):
        sampled, indices_out = native_sample_farthest_points(
            points.cpu(),
            lengths=lengths.cpu() if lengths is not None else None,
            K=K,
            random_start_point=random_start_point,
        )
        return sampled.to(points.device), indices_out.to(points.device)

    pytorch3d.ops.sample_farthest_points = sample_farthest_points_cpu_bridge

    print("Loading 3D VAE encoder", flush=True)
    encoder = ElasticFixedlenEncoder(
        pcd_pe_mode="pcd_ape_v2",
        query_mode="fps",
        model_channels=1024,
        cond_channels=1280,
        cond_channels2=32,
        q_token_length=8192,
        latent_channels=16,
        num_blocks=16,
        num_heads=16,
        mlp_ratio=4,
        use_fp16=True,
        use_2_cross_block=True,
    ).eval().cuda()
    encoder.load_state_dict(load_file(
        str(args.checkpoints / "vae/triposplat_vae_encoder_fp16.safetensors")
    ), strict=True)

    encoder_points = torch.from_numpy(points + 0.5).clamp(0, 1).unsqueeze(0).cuda()
    condition = {
        "points": encoder_points,
        "features": torch.zeros((1, args.input_points, 1280), device="cuda", dtype=torch.float16),
        "points2": encoder_points,
        "features2": torch.zeros((1, args.input_points, 32), device="cuda", dtype=torch.float16),
    }
    print("Encoding XYZ with zero DINO/FLUX features", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        latent, query_points = encoder(
            x=None,
            cond=condition,
            sample_posterior=False,
            return_fps=True,
        )
    latent = latent.half()
    torch.save({
        "latent": latent.cpu(),
        "query_points": query_points.cpu(),
        "source_indices": torch.from_numpy(indices),
    }, args.output / "02-encoded-latent.pt")
    latent_np = latent.float().cpu().numpy()
    del encoder, condition
    torch.cuda.empty_cache()

    print("Loading Gaussian decoder", flush=True)
    decoder = load_decoder(
        str(args.checkpoints / "vae/triposplat_vae_decoder_fp16.safetensors"),
        device="cuda",
        dtype=torch.float16,
    )
    print(f"Decoding {args.output_gaussians} Gaussians", flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        gaussian = decode_detailed(decoder, latent.cuda(), args.output_gaussians, args.seed + 4000)
    gaussian.save_ply(args.output / "03-roundtrip-gaussians.ply")
    output_points, output_colors = gaussian_arrays(gaussian)
    write_point_ply(
        args.output / "03-roundtrip-centers.ply",
        output_points,
        output_colors * 255,
    )

    print("Rendering result for inspection", flush=True)
    images, labels = render_orbit(gaussian, args.render_size)
    render_dir = args.output / "04-roundtrip-renders"
    render_dir.mkdir()
    for image, label in zip(images, labels):
        image.save(render_dir / f"{label.replace(' ', '-')}.png")
    contact_sheet(images, labels).save(args.output / "04-roundtrip-orbit-contact-sheet.jpg", quality=92)

    report = {
        "schema": "diffusion-editor.triposplat-vae-centers-roundtrip.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "XYZ-only; DINOv3 and FLUX2 point features are all zero",
        "source": str(args.source),
        "input_center_count_total": len(all_points),
        "input_center_count_encoder": args.input_points,
        "output_gaussian_count": len(output_points),
        "seed": args.seed,
        "input_bounds": {"min": points.min(0).tolist(), "max": points.max(0).tolist()},
        "output_bounds": {"min": output_points.min(0).tolist(), "max": output_points.max(0).tolist()},
        "latent": summary(latent_np),
        "output_opacity": summary(gaussian.get_opacity.detach().float().cpu().numpy()),
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
