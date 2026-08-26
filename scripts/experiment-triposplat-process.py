#!/usr/bin/env python3
"""Inspect TripoSplat as a staged image-to-3D process.

The upstream project intentionally exposes a compact inference API.  This
runner keeps that implementation untouched while recording the otherwise
ephemeral flow trajectory, jointly predicted camera token, octree anchors and
Gaussian-density variants.  It also renders selected trajectory states through
``gsplat`` when that optional package is available.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import html
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


_SH_C0 = 0.28209479177387814


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--triposplat-root", type=Path,
        default=Path("/home/mirmik/soft/TripoSplat"),
    )
    parser.add_argument(
        "--checkpoints", type=Path,
        default=Path("/home/mirmik/mnt-nvme/models/TripoSplat"),
    )
    parser.add_argument("--mask", type=Path)
    parser.add_argument(
        "--conditioning-image",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional view to encode into the same flow-conditioning "
            "sequence. May be repeated. This is an experimental, "
            "non-upstream multiview mode."
        ),
    )
    parser.add_argument(
        "--conditioning-mask",
        type=Path,
        action="append",
        default=[],
        help="Foreground mask matching --conditioning-image; may be repeated.",
    )
    parser.add_argument(
        "--conditioning-fusion",
        choices=(
            "token-concat", "velocity-mean", "velocity-weighted",
            "camera-alternating",
        ),
        default="token-concat",
        help=(
            "How additional views affect the shared latent. token-concat "
            "matches the original exploratory hack; velocity-mean runs one "
            "denoiser pass per view and averages only latent velocities; "
            "velocity-weighted uses explicitly supplied normalized weights; "
            "camera-alternating applies exactly one view per diffusion step "
            "and cycles through the views."
        ),
    )
    parser.add_argument(
        "--view-weight",
        type=float,
        action="append",
        default=[],
        help=(
            "Non-negative velocity weight for each input view. Repeat in "
            "input-view order; required by velocity-weighted."
        ),
    )
    parser.add_argument(
        "--alternating-view-order-seed",
        type=int,
        help=(
            "Shuffle the camera-alternating view cycle reproducibly with "
            "this seed. By default input-view order is retained."
        ),
    )
    parser.add_argument(
        "--alternating-view-index",
        type=int,
        action="append",
        default=[],
        help=(
            "Explicit camera-alternating cycle entry. Repeat to define an "
            "ordered schedule; indices may repeat. The schedule cycles after "
            "the primary-view warm-up."
        ),
    )
    parser.add_argument(
        "--fixed-camera-azimuth",
        type=float,
        action="append",
        default=[],
        help=(
            "Known camera azimuth in degrees for each input view. Repeat in "
            "input-view order. Zero points along +X and +90 along +Y."
        ),
    )
    parser.add_argument(
        "--fixed-camera-elevation",
        type=float,
        action="append",
        default=[],
        help=(
            "Known camera elevation in degrees. Omit for zero; pass once to "
            "reuse one elevation for every view, or repeat per input view."
        ),
    )
    parser.add_argument(
        "--fixed-camera-distance",
        type=float,
        default=2.0,
        help="Camera-to-object-center distance used by fixed camera tokens.",
    )
    parser.add_argument(
        "--fixed-camera-fov",
        type=float,
        default=40.0,
        help="Vertical field of view in degrees used by fixed camera tokens.",
    )
    parser.add_argument(
        "--fixed-camera-trajectory",
        choices=("flow-inpaint", "clean"),
        default="flow-inpaint",
        help=(
            "flow-inpaint follows the training noise schedule; clean exposes "
            "the exact target camera at every denoising step."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--initial-latent",
        type=Path,
        help=(
            "Start from a previously encoded clean TripoSPLAT latent instead "
            "of pure flow noise. Requires --img2img-strength. The file may "
            "contain either a tensor or a dictionary of tensors."
        ),
    )
    parser.add_argument(
        "--initial-latent-key",
        default="latent",
        help=(
            "Tensor key inside --initial-latent when it contains a "
            "dictionary (default: latent)."
        ),
    )
    parser.add_argument(
        "--img2img-strength",
        type=float,
        help=(
            "Rectified-flow noise time for --initial-latent. For example, "
            "0.5 initializes sampling as 0.5 * clean + 0.5 * noise and "
            "denoises from t=0.5 to t=0."
        ),
    )
    parser.add_argument(
        "--image-feature-seed",
        type=int,
        action="append",
        default=[],
        help=(
            "Seed for the stochastic image feature encoding of each input "
            "view. Repeat in input-view order. By default seeds are derived "
            "from --seed and the view index."
        ),
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument(
        "--primary-view-warmup-steps",
        type=int,
        default=0,
        help=(
            "In a per-view fusion mode, use only input view 0 for this many "
            "initial diffusion steps, then begin combining all views."
        ),
    )
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument(
        "--counts", type=int, nargs="+",
        default=(32768, 65536, 131072, 262144),
    )
    parser.add_argument(
        "--trajectory-steps", type=int, nargs="+",
        default=(0, 4, 8, 12, 16, 17, 18, 19, 20),
    )
    parser.add_argument("--trajectory-count", type=int, default=32768)
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace artifacts previously produced in the output directory.",
    )
    return parser.parse_args()


def _json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _tensor_summary(value: torch.Tensor) -> dict:
    data = value.detach().float()
    return {
        "shape": list(data.shape),
        "dtype": str(value.dtype),
        "min": float(data.min()),
        "max": float(data.max()),
        "mean": float(data.mean()),
        "std": float(data.std()),
        "l2": float(torch.linalg.vector_norm(data)),
    }


def _camera_summary(camera: torch.Tensor) -> dict:
    raw = camera.detach().float().reshape(-1).cpu().numpy()
    direction = raw[:3]
    norm = float(np.linalg.norm(direction))
    unit = direction / max(norm, 1e-8)
    inverse_distance = float(raw[3])
    scale = float(raw[4])
    distance = 1.0 / inverse_distance if inverse_distance > 1e-8 else None
    half_angle_tan = scale * inverse_distance
    fov = (
        math.degrees(2.0 * math.atan(half_angle_tan))
        if half_angle_tan > 0.0 else None
    )
    return {
        "raw": raw.tolist(),
        "direction_norm": norm,
        "unit_view_direction": unit.tolist(),
        "inverse_distance": inverse_distance,
        "distance": distance,
        "camera_scale": scale,
        "fov_degrees": fov,
    }


def _camera_summaries(cameras: torch.Tensor) -> list[dict]:
    return [
        _camera_summary(cameras[index:index + 1])
        for index in range(cameras.shape[0])
    ]


def _camera_target_values(
    azimuth_degrees: float,
    elevation_degrees: float,
    distance: float,
    fov_degrees: float,
) -> list[float]:
    azimuth = math.radians(azimuth_degrees)
    elevation = math.radians(elevation_degrees)
    horizontal = math.cos(elevation)
    direction = (
        horizontal * math.cos(azimuth),
        horizontal * math.sin(azimuth),
        math.sin(elevation),
    )
    return [
        *direction,
        1.0 / distance,
        distance * math.tan(math.radians(fov_degrees) / 2.0),
    ]


def _known_camera_state(
    camera_noise: torch.Tensor,
    camera_targets: torch.Tensor,
    timestep: float,
) -> torch.Tensor:
    """Return the analytically noised known camera at a flow timestep."""
    t = float(timestep)
    return camera_noise * t + camera_targets * (1.0 - t)


def _fixed_camera_state(
    camera_noise: torch.Tensor,
    camera_targets: torch.Tensor,
    timestep: float,
    trajectory: str,
) -> torch.Tensor:
    if trajectory == "clean":
        return camera_targets.clone()
    if trajectory == "flow-inpaint":
        return _known_camera_state(camera_noise, camera_targets, timestep)
    raise ValueError(f"unsupported fixed camera trajectory: {trajectory}")


def _velocity_agreement(velocities: list[torch.Tensor]) -> dict:
    flat = torch.stack(
        [velocity.detach().float().reshape(-1) for velocity in velocities],
        dim=0,
    )
    normalized = torch.nn.functional.normalize(flat, dim=1)
    cosine = normalized @ normalized.T
    centered = flat - flat.mean(dim=0, keepdim=True)
    return {
        "per_view_l2": torch.linalg.vector_norm(flat, dim=1).cpu().tolist(),
        "pairwise_cosine": cosine.cpu().tolist(),
        "mean_deviation_l2": float(
            torch.linalg.vector_norm(centered, dim=1).mean()
        ),
    }


def _cuda_memory() -> dict:
    return {
        "allocated_mib": torch.cuda.memory_allocated() / 2**20,
        "reserved_mib": torch.cuda.memory_reserved() / 2**20,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


@contextmanager
def _timed_stage(report: dict, name: str):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    print(f"[stage] {name} ...", flush=True)
    try:
        yield
    finally:
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        report["stages"][name] = {
            "seconds": elapsed,
            "cuda": _cuda_memory(),
        }
        print(f"[stage] {name}: {elapsed:.2f}s", flush=True)


def _source_with_optional_mask(image_path: Path, mask_path: Path | None) -> Image.Image:
    source = Image.open(image_path).convert("RGBA")
    if mask_path is not None:
        mask = Image.open(mask_path).convert("L")
        if mask.size != source.size:
            mask = mask.resize(source.size, Image.Resampling.LANCZOS)
        source.putalpha(mask)
    return source


def _write_point_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    xyz = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.ascontiguousarray(np.clip(colors, 0.0, 1.0) * 255, dtype=np.uint8).reshape(-1, 3)
    finite = np.isfinite(xyz).all(axis=1)
    xyz, rgb = xyz[finite], rgb[finite]
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


def _density_colors(log_probs: np.ndarray) -> np.ndarray:
    values = np.asarray(log_probs, dtype=np.float32).reshape(-1)
    lo, hi = np.percentile(values, (2, 98))
    unit = np.clip((values - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0)
    return np.column_stack((unit, 1.0 - np.abs(unit * 2.0 - 1.0), 1.0 - unit))


def _gaussian_arrays(gaussian) -> dict[str, np.ndarray]:
    return {
        "means": gaussian.get_xyz.detach().float().cpu().numpy(),
        "colors": np.clip(
            gaussian._features_dc.detach().float().cpu().numpy()[:, 0, :] * _SH_C0 + 0.5,
            0.0, 1.0,
        ),
        "opacities": gaussian.get_opacity.detach().float().cpu().numpy()[:, 0],
        "scales": gaussian.get_scaling.detach().float().cpu().numpy(),
        "quaternions": (
            gaussian._rotation + gaussian.rots_bias[None, :]
        ).detach().float().cpu().numpy(),
    }


def _gaussian_summary(arrays: dict[str, np.ndarray]) -> dict:
    means = arrays["means"]
    scales = arrays["scales"]
    opacity = arrays["opacities"]
    return {
        "count": len(means),
        "bounds_min": means.min(axis=0).tolist(),
        "bounds_max": means.max(axis=0).tolist(),
        "opacity": {
            "min": float(opacity.min()),
            "median": float(np.median(opacity)),
            "mean": float(opacity.mean()),
            "max": float(opacity.max()),
        },
        "scale": {
            "min": float(scales.min()),
            "median": float(np.median(scales)),
            "mean": float(scales.mean()),
            "max": float(scales.max()),
        },
    }


def _decode_detailed(pipe, latent: torch.Tensor, count: int, random_seed: int):
    from model import OctreeProbabilityFixedlenDecoder
    from triposplat import _build_gaussians

    cuda_rng = torch.cuda.get_rng_state()
    cpu_rng = torch.get_rng_state()
    try:
        torch.manual_seed(random_seed)
        anchor_count = max(1, count // pipe.decoder.gaussians_per_point)
        points = OctreeProbabilityFixedlenDecoder.sample(
            pipe.decoder.octree,
            latent,
            num_points=anchor_count,
            level=pipe.decoder._MAX_VOXEL_LEVEL,
            temperature=1.0,
            algo="systematic",
        )
        attributes = pipe.decoder.gs(x=points, cond=latent)
        gaussian = _build_gaussians(pipe.decoder.gs, points, attributes)[0]
        return gaussian, points
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)


def _look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    if np.linalg.norm(right) < 1e-5:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack((right, down, forward), axis=0)
    view = np.eye(4, dtype=np.float32)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ eye
    return view


def _rotate_axis(vector: np.ndarray, axis: np.ndarray, degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    unit_axis = np.asarray(axis, dtype=np.float32)
    unit_axis /= max(float(np.linalg.norm(unit_axis)), 1e-8)
    value = np.asarray(vector, dtype=np.float32)
    return (
        value * math.cos(angle)
        + np.cross(unit_axis, value) * math.sin(angle)
        + unit_axis * np.dot(unit_axis, value) * (1.0 - math.cos(angle))
    )


def _principal_axis(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = points - points.mean(axis=0, keepdims=True)
    eigenvalues, eigenvectors = np.linalg.eigh(np.cov(centered, rowvar=False))
    axis = eigenvectors[:, -1].astype(np.float32)
    if axis[2] < 0:
        axis = -axis
    return axis, eigenvalues.astype(np.float32)


def _align_vector_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    source /= max(float(np.linalg.norm(source)), 1e-8)
    target /= max(float(np.linalg.norm(target)), 1e-8)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1e-7:
        return np.eye(3, dtype=np.float32)
    skew = np.array([
        [0.0, -cross[2], cross[1]],
        [cross[2], 0.0, -cross[0]],
        [-cross[1], cross[0], 0.0],
    ], dtype=np.float32)
    return np.eye(3, dtype=np.float32) + skew + skew @ skew * ((1.0 - cosine) / sine**2)


def _render_gaussian(
    gaussian,
    view_directions: list[np.ndarray],
    size: int,
    *,
    fov_degrees: float = 40.0,
    up_direction: np.ndarray | None = None,
) -> list[Image.Image]:
    from gsplat import rasterization

    means = gaussian.get_xyz.float()
    colors = torch.clamp(gaussian._features_dc[:, 0, :].float() * _SH_C0 + 0.5, 0, 1)
    opacities = gaussian.get_opacity[:, 0].float()
    scales = gaussian.get_scaling.float()
    quats = (gaussian._rotation + gaussian.rots_bias[None, :]).float()
    bounds_min = means.amin(dim=0)
    bounds_max = means.amax(dim=0)
    center = ((bounds_min + bounds_max) * 0.5).cpu().numpy()
    extent = float((bounds_max - bounds_min).max())
    radius = max(2.0 * extent, 1.0)
    up = np.asarray(
        [0.0, 0.0, 1.0] if up_direction is None else up_direction,
        dtype=np.float32,
    )
    up /= max(float(np.linalg.norm(up)), 1e-8)
    views = []
    for direction in view_directions:
        unit = np.asarray(direction, dtype=np.float32)
        unit /= max(float(np.linalg.norm(unit)), 1e-8)
        # TripoSplat training and its camera-token codec use a Z-up world.
        views.append(_look_at(center + unit * radius, center, up))
    viewmats = torch.from_numpy(np.stack(views)).to(means.device)
    focal = 0.5 * size / math.tan(math.radians(fov_degrees) * 0.5)
    intrinsics = np.array([
        [focal, 0.0, size * 0.5],
        [0.0, focal, size * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    Ks = torch.from_numpy(np.repeat(intrinsics[None], len(views), axis=0)).to(means.device)
    rendered, alpha, _ = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=size,
        height=size,
        packed=True,
    )
    images = []
    rendered = rendered[..., :3] + (1.0 - alpha)
    for frame in rendered.detach().clamp(0, 1).cpu().numpy():
        images.append(Image.fromarray((frame * 255).astype(np.uint8), "RGB"))
    return images


def _contact_sheet(images: list[Image.Image], labels: list[str], columns: int) -> Image.Image:
    if not images:
        raise ValueError("contact sheet needs at least one image")
    width, height = images[0].size
    label_height = 34
    rows = math.ceil(len(images) / columns)
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "#1c1c1c")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=18)
    for index, (image, label) in enumerate(zip(images, labels)):
        x = (index % columns) * width
        y = (index // columns) * (height + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 10, y + height + 7), label, fill="white", font=font)
    return sheet


def _html_report(report: dict) -> str:
    stage_rows = "".join(
        "<tr>"
        f"<td>{html.escape(name)}</td>"
        f"<td>{values['seconds']:.2f} s</td>"
        f"<td>{values['cuda']['peak_allocated_mib']:.0f} MiB</td>"
        "</tr>"
        for name, values in report["stages"].items()
    )
    camera = report["camera_final"]
    return f"""<!doctype html>
<html lang=\"ru\"><head><meta charset=\"utf-8\"><title>TripoSplat process</title>
<style>
body{{font:16px/1.45 system-ui;background:#111;color:#ddd;max-width:1200px;margin:auto;padding:28px}}
h1,h2{{color:#fff}} code{{background:#262626;padding:.15em .35em;border-radius:4px}}
.flow{{display:grid;grid-template-columns:repeat(7,1fr);gap:8px;align-items:stretch}}
.flow div{{background:#232938;border:1px solid #46516b;padding:12px;border-radius:8px}}
img{{max-width:100%;background:#222}} table{{border-collapse:collapse;width:100%}}
td,th{{border-bottom:1px solid #444;text-align:left;padding:8px}}
.note{{border-left:4px solid #d89b35;padding:8px 14px;background:#251f16}}
</style></head><body>
<h1>TripoSplat — протокол процесса</h1>
<p>Вход: <code>{html.escape(report['input'])}</code>. Seed {report['parameters']['seed']},
{report['parameters']['steps']} Euler-шагов, CFG {report['parameters']['guidance']}.</p>
<div class=\"flow\"><div>RGB / alpha<br><small>BiRefNet + bbox</small></div><div>DINOv3<br><small>семантика</small></div>
<div>FLUX.2 VAE<br><small>детали</small></div><div>VecSeq flow<br><small>8192×16 + camera 5D</small></div>
<div>Octree PDF<br><small>8 уровней</small></div><div>Local decoder<br><small>32 GS / anchor</small></div>
<div>Gaussian scene<br><small>32k…262k</small></div></div>
<h2>Траектория flow</h2><img src=\"07-renders/trajectory-contact-sheet.jpg\">
<p>Каждый кадр — декодирование текущего noisy latent одним и тем же низким бюджетом 32k. Это
диагностическая визуализация; во время штатного inference декодируется только последний шаг.</p>
<h2>Финальная сцена, восемь направлений</h2><img src=\"07-renders/final-orbit-contact-sheet.jpg\">
<p>Это честная Z-up орбита. Наклон объекта показывает совместную неопределённость объекта и камеры.</p>
<h2>Та же сцена после фиксации вертикали по PCA тела</h2><img src=\"07-renders/body-axis-orbit-contact-sheet.jpg\">
<h2>Один latent, четыре бюджета плотности</h2><img src=\"07-renders/density-contact-sheet.jpg\">
<h2>Предсказанная камера</h2>
<p>Направление: <code>{html.escape(str(camera['unit_view_direction']))}</code>;
raw distance: <code>{camera['distance']}</code>; raw FOV: <code>{camera['fov_degrees']}</code>.</p>
<p class=\"note\">Авторы предупреждают, что bbox-нормализация делает distance и scale менее точными.
В рендерах выше используется направление camera token, но нормализованная дистанция и FOV 40°.</p>
<h2>Время и VRAM</h2><table><tr><th>Стадия</th><th>Время</th><th>Peak allocated</th></tr>{stage_rows}</table>
<h2>Артефакты</h2><ul>
<li><code>02-image-features.npz</code> — image embedding каждого входного вида;</li>
<li><code>03-flow-trajectory.npz</code> — latent и camera после каждого шага;</li>
<li><code>04-trajectory-gaussians/</code> — PLY эволюции для нативного просмотрщика;</li>
<li><code>05-density-anchors/</code> — центры октодерева, окрашенные по log-density;</li>
<li><code>06-final-gaussians/</code> — стандартные PLY/SPLAT и point-preview;</li>
<li><code>report.json</code> — параметры, численные характеристики и stage timings.</li>
</ul></body></html>"""


def main() -> int:
    args = _arguments()
    args.image = args.image.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    args.triposplat_root = args.triposplat_root.expanduser().resolve()
    args.checkpoints = args.checkpoints.expanduser().resolve()
    args.mask = args.mask.expanduser().resolve() if args.mask else None
    args.initial_latent = (
        args.initial_latent.expanduser().resolve()
        if args.initial_latent else None
    )
    args.conditioning_image = [
        path.expanduser().resolve() for path in args.conditioning_image
    ]
    args.conditioning_mask = [
        path.expanduser().resolve() for path in args.conditioning_mask
    ]
    if args.conditioning_mask and (
        len(args.conditioning_mask) != len(args.conditioning_image)
    ):
        raise SystemExit(
            "--conditioning-mask count must match --conditioning-image count"
        )
    if not args.conditioning_mask:
        args.conditioning_mask = [None] * len(args.conditioning_image)
    if not torch.cuda.is_available():
        raise SystemExit("TripoSplat requires CUDA")
    # The upstream stage methods are individually decorated with no_grad.  Our
    # expanded sampler loop calls the denoiser and decoder pieces directly, so
    # disable autograd globally before recording the trajectory.
    torch.set_grad_enabled(False)
    if args.steps <= 0 or args.render_size <= 0:
        raise SystemExit("steps and render-size must be positive")
    if (args.initial_latent is None) != (args.img2img_strength is None):
        raise SystemExit(
            "--initial-latent and --img2img-strength must be supplied together"
        )
    if (
        args.img2img_strength is not None
        and not 0.0 < args.img2img_strength <= 1.0
    ):
        raise SystemExit("--img2img-strength must be inside (0, 1]")
    if not 0 <= args.primary_view_warmup_steps <= args.steps:
        raise SystemExit(
            "primary-view-warmup-steps must be inside [0, steps]"
        )
    if (
        args.primary_view_warmup_steps
        and args.conditioning_fusion not in (
            "velocity-mean", "velocity-weighted", "camera-alternating"
        )
    ):
        raise SystemExit(
            "primary-view-warmup-steps requires "
            "--conditioning-fusion velocity-mean, velocity-weighted, or "
            "camera-alternating"
        )
    selected_steps = sorted(set(args.trajectory_steps) | {0, args.steps})
    if selected_steps[0] < 0 or selected_steps[-1] > args.steps:
        raise SystemExit("trajectory steps must be inside [0, steps]")

    generated_entries = (
        "00-source.png", "00-source-original.png", "00-source-original.webp",
        "00-source-original.jpg", "00-source-original.jpeg",
        "01-prepared-1024.png", "02-image-features.npz",
        "03-final-latent-camera.pt", "03-flow-trajectory.npz",
        "04-trajectory-gaussians", "05-density-anchors", "06-final-gaussians",
        "00-conditioning", "07-renders", "process.html", "report.json",
    )
    existing_generated = [
        args.output / name for name in generated_entries
        if (args.output / name).exists()
    ]
    if existing_generated and not args.overwrite:
        raise SystemExit(
            f"output already contains TripoSplat artifacts: {args.output}; "
            "pass --overwrite to replace them"
        )
    if args.overwrite:
        for entry in existing_generated:
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    args.output.mkdir(parents=True, exist_ok=True)
    trajectory_dir = args.output / "04-trajectory-gaussians"
    anchors_dir = args.output / "05-density-anchors"
    final_dir = args.output / "06-final-gaussians"
    renders_dir = args.output / "07-renders"
    conditioning_dir = args.output / "00-conditioning"
    for directory in (trajectory_dir, anchors_dir, final_dir, renders_dir):
        directory.mkdir(exist_ok=True)
    if args.conditioning_image:
        conditioning_dir.mkdir(exist_ok=True)

    input_views = [
        {"image": args.image, "mask": args.mask},
        *[
            {"image": image, "mask": mask}
            for image, mask in zip(
                args.conditioning_image, args.conditioning_mask
            )
        ],
    ]
    if args.image_feature_seed:
        if len(args.image_feature_seed) != len(input_views):
            raise SystemExit(
                "--image-feature-seed count must match the total input-view "
                f"count ({len(input_views)})"
            )
        image_feature_seeds = args.image_feature_seed
    else:
        image_feature_seeds = [
            args.seed + 10_000 + index for index in range(len(input_views))
        ]
    if args.view_weight:
        if len(args.view_weight) != len(input_views):
            raise SystemExit(
                "--view-weight count must match the total input-view count "
                f"({len(input_views)})"
            )
        if any(weight < 0.0 or not math.isfinite(weight)
               for weight in args.view_weight):
            raise SystemExit("view weights must be finite and non-negative")
        if not any(weight > 0.0 for weight in args.view_weight):
            raise SystemExit("at least one view weight must be positive")
        view_weights = args.view_weight
    else:
        view_weights = [1.0] * len(input_views)
    if args.conditioning_fusion == "velocity-weighted" and not args.view_weight:
        raise SystemExit("velocity-weighted requires repeated --view-weight")
    view_cycle_order = (
        list(args.alternating_view_index)
        if args.alternating_view_index
        else list(range(len(input_views)))
    )
    if args.alternating_view_index:
        if args.conditioning_fusion != "camera-alternating":
            raise SystemExit(
                "alternating-view-index requires "
                "--conditioning-fusion camera-alternating"
            )
        if args.alternating_view_order_seed is not None:
            raise SystemExit(
                "alternating-view-index and alternating-view-order-seed "
                "are mutually exclusive"
            )
        invalid_indices = [
            index for index in view_cycle_order
            if not 0 <= index < len(input_views)
        ]
        if invalid_indices:
            raise SystemExit(
                "alternating view indices must be inside [0, "
                f"{len(input_views) - 1}]; got {invalid_indices}"
            )
    if args.alternating_view_order_seed is not None:
        if args.conditioning_fusion != "camera-alternating":
            raise SystemExit(
                "alternating-view-order-seed requires "
                "--conditioning-fusion camera-alternating"
            )
        np.random.default_rng(
            args.alternating_view_order_seed
        ).shuffle(view_cycle_order)
    fixed_camera_values = None
    if args.fixed_camera_azimuth:
        if len(args.fixed_camera_azimuth) != len(input_views):
            raise SystemExit(
                "--fixed-camera-azimuth count must match the total input-view "
                f"count ({len(input_views)})"
            )
        if (
            len(input_views) > 1
            and args.conditioning_fusion not in (
                "velocity-mean", "velocity-weighted", "camera-alternating"
            )
        ):
            raise SystemExit(
                "fixed multiview cameras require "
                "--conditioning-fusion velocity-mean, velocity-weighted, or "
                "camera-alternating"
            )
        if not args.fixed_camera_elevation:
            fixed_elevations = [0.0] * len(input_views)
        elif len(args.fixed_camera_elevation) == 1:
            fixed_elevations = (
                args.fixed_camera_elevation * len(input_views)
            )
        elif len(args.fixed_camera_elevation) == len(input_views):
            fixed_elevations = args.fixed_camera_elevation
        else:
            raise SystemExit(
                "--fixed-camera-elevation must be omitted, passed once, or "
                "repeated for every input view"
            )
        if args.fixed_camera_distance <= 0.0:
            raise SystemExit("--fixed-camera-distance must be positive")
        if not 0.0 < args.fixed_camera_fov < 180.0:
            raise SystemExit("--fixed-camera-fov must be inside (0, 180)")
        fixed_camera_values = [
            _camera_target_values(
                azimuth,
                elevation,
                args.fixed_camera_distance,
                args.fixed_camera_fov,
            )
            for azimuth, elevation in zip(
                args.fixed_camera_azimuth, fixed_elevations
            )
        ]
    elif args.fixed_camera_elevation:
        raise SystemExit(
            "--fixed-camera-elevation requires --fixed-camera-azimuth"
        )
    elif args.fixed_camera_trajectory != "flow-inpaint":
        raise SystemExit(
            "--fixed-camera-trajectory requires --fixed-camera-azimuth"
        )

    report = {
        "schema": "diffusion-editor.triposplat-process.v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.image),
        "mask": str(args.mask) if args.mask else None,
        "input_views": [
            {
                "image": str(view["image"]),
                "mask": str(view["mask"]) if view["mask"] else None,
            }
            for view in input_views
        ],
        "upstream": {
            "root": str(args.triposplat_root),
            "git_commit": os.popen(
                f"git -C {str(args.triposplat_root)!r} rev-parse HEAD"
            ).read().strip(),
        },
        "checkpoints": str(args.checkpoints),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
        },
        "parameters": {
            "seed": args.seed,
            "steps": args.steps,
            "primary_view_warmup_steps": (
                args.primary_view_warmup_steps
            ),
            "guidance": args.guidance,
            "shift": args.shift,
            "counts": args.counts,
            "trajectory_steps": selected_steps,
            "trajectory_count": args.trajectory_count,
            "render_size": args.render_size,
            "conditioning_fusion": (
                args.conditioning_fusion
                if len(input_views) > 1 else "single-view"
            ),
            "view_count": len(input_views),
            "view_weights": view_weights,
            "alternating_view_order_seed": (
                args.alternating_view_order_seed
            ),
            "alternating_view_order": view_cycle_order,
            "alternating_view_sequence_explicit": bool(
                args.alternating_view_index
            ),
            "fixed_cameras": (
                {
                    "azimuth_degrees": args.fixed_camera_azimuth,
                    "elevation_degrees": fixed_elevations,
                    "distance": args.fixed_camera_distance,
                    "fov_degrees": args.fixed_camera_fov,
                    "target_tokens": fixed_camera_values,
                    "trajectory": args.fixed_camera_trajectory,
                }
                if fixed_camera_values is not None else None
            ),
            "img2img": (
                {
                    "initial_latent": str(args.initial_latent),
                    "initial_latent_key": args.initial_latent_key,
                    "strength": args.img2img_strength,
                    "noising_equation": (
                        "z_t = (1 - t) * z_clean + t * epsilon"
                    ),
                }
                if args.initial_latent is not None else None
            ),
        },
        "stages": {},
    }

    sys.path.insert(0, str(args.triposplat_root))
    from triposplat import FlowEulerCfgSampler, TripoSplatPipeline

    with _timed_stage(report, "load_models"):
        pipe = TripoSplatPipeline(
            ckpt_path=str(args.checkpoints / "diffusion_models/triposplat_fp16.safetensors"),
            decoder_path=str(args.checkpoints / "vae/triposplat_vae_decoder_fp16.safetensors"),
            dinov3_path=str(args.checkpoints / "clip_vision/dino_v3_vit_h.safetensors"),
            flux2_vae_encoder_path=str(args.checkpoints / "vae/flux2-vae.safetensors"),
            rmbg_path=str(args.checkpoints / "background_removal/birefnet.safetensors"),
            device="cuda",
        )

    # A supplied mask makes BiRefNet unnecessary during preprocessing.  Keep
    # the already constructed pipeline API intact, but move the unused model
    # away from scarce VRAM before allocating image encoder activations.
    if all(view["mask"] is not None for view in input_views):
        pipe.rmbg.to("cpu")
        torch.cuda.empty_cache()
        report["parameters"]["background_removal_device"] = (
            "cpu (external masks)"
        )
    else:
        report["parameters"]["background_removal_device"] = "cuda"

    sources = []
    for view_index, view in enumerate(input_views):
        source = _source_with_optional_mask(view["image"], view["mask"])
        sources.append(source)
        if view_index == 0:
            source.save(args.output / "00-source.png")
            shutil.copy2(
                view["image"],
                args.output / f"00-source-original{view['image'].suffix.lower()}",
            )
        else:
            prefix = conditioning_dir / f"view-{view_index:02d}"
            source.save(prefix.with_name(prefix.name + "-source.png"))
            shutil.copy2(
                view["image"],
                prefix.with_name(
                    prefix.name + f"-original{view['image'].suffix.lower()}"
                ),
            )
    with _timed_stage(report, "preprocess"):
        prepared_views = []
        for view_index, source in enumerate(sources):
            prepared = pipe.preprocess_image(source, erode_radius=1)
            prepared_views.append(prepared)
            if view_index == 0:
                prepared.save(args.output / "01-prepared-1024.png")
            else:
                prepared.save(
                    conditioning_dir / f"view-{view_index:02d}-prepared-1024.png"
                )

    with _timed_stage(report, "encode_image"):
        per_view_conditions = [
            pipe.encode_image(
                prepared,
                generator=torch.Generator(device="cuda").manual_seed(
                    image_feature_seeds[view_index]
                ),
            )
            for view_index, prepared in enumerate(prepared_views)
        ]
        conditions = {
            key: torch.cat(
                [view_conditions[key] for view_conditions in per_view_conditions],
                dim=1,
            )
            for key in per_view_conditions[0]
        }
        report["image_features"] = {
            "fusion": report["parameters"]["conditioning_fusion"],
            "per_view": [
                {
                    key: _tensor_summary(value)
                    for key, value in view_conditions.items()
                }
                for view_conditions in per_view_conditions
            ],
            "combined": {
                key: _tensor_summary(value)
                for key, value in conditions.items()
            },
        }
        feature_arrays = {
            f"view_{view_index:02d}_{key}": (
                value.detach().to(torch.float16).cpu().numpy()
            )
            for view_index, view_conditions in enumerate(per_view_conditions)
            for key, value in view_conditions.items()
        }
        np.savez_compressed(
            args.output / "02-image-features.npz",
            **feature_arrays,
        )

    # Only the fused conditioning tensors are needed by the denoiser.  Staging
    # the two image encoders on CPU leaves enough VRAM for the longer sequence.
    pipe.dinov3.to("cpu")
    pipe.vae_encoder.to("cpu")
    torch.cuda.empty_cache()

    # Keep the denoising start identical across single- and multiview runs.
    # The stochastic FLUX.2 VAE encodes above deliberately use separate RNGs.
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    report["parameters"]["random_streams"] = {
        "flow_noise_seed": args.seed,
        "image_feature_seeds": image_feature_seeds,
    }

    flow = pipe.flow_model
    velocity_fusion = (
        len(input_views) > 1
        and args.conditioning_fusion in (
            "velocity-mean", "velocity-weighted"
        )
    )
    weighted_velocity_fusion = (
        len(input_views) > 1
        and args.conditioning_fusion == "velocity-weighted"
    )
    alternating_fusion = (
        len(input_views) > 1
        and args.conditioning_fusion == "camera-alternating"
    )
    negative = {
        key: torch.zeros_like(value) for key, value in conditions.items()
    }
    per_view_negative = [
        {key: torch.zeros_like(value) for key, value in view.items()}
        for view in per_view_conditions
    ]
    latent_noise = torch.randn(
        1, flow.q_token_length, flow.in_channels,
        device=flow.device, generator=generator,
    )
    base_camera = torch.randn(
        1, 1, flow.cam_channels,
        device=flow.device, generator=generator,
    )
    camera_count = (
        len(input_views) if velocity_fusion or alternating_fusion else 1
    )
    camera_noise = base_camera.repeat(camera_count, 1, 1)
    camera_targets = (
        torch.tensor(
            fixed_camera_values,
            device=flow.device,
            dtype=camera_noise.dtype,
        ).unsqueeze(1)
        if fixed_camera_values is not None else None
    )
    clean_latent = None
    if args.initial_latent is not None:
        loaded_latent = torch.load(
            args.initial_latent, map_location="cpu", weights_only=False,
        )
        if isinstance(loaded_latent, dict):
            if args.initial_latent_key not in loaded_latent:
                raise SystemExit(
                    f"initial latent key {args.initial_latent_key!r} is absent "
                    f"from {args.initial_latent}; available keys: "
                    f"{sorted(loaded_latent)}"
                )
            loaded_latent = loaded_latent[args.initial_latent_key]
        if not isinstance(loaded_latent, torch.Tensor):
            raise SystemExit(
                "initial latent must be a tensor or a dictionary containing one"
            )
        expected_shape = (1, flow.q_token_length, flow.in_channels)
        if tuple(loaded_latent.shape) != expected_shape:
            raise SystemExit(
                f"initial latent has shape {tuple(loaded_latent.shape)}, "
                f"expected {expected_shape}"
            )
        clean_latent = loaded_latent.to(
            device=flow.device, dtype=latent_noise.dtype,
        )
        strength = float(args.img2img_strength)
        initial_latent = clean_latent * (1.0 - strength) + latent_noise * strength
        report["initial_latent"] = {
            "clean": _tensor_summary(clean_latent),
            "noise": _tensor_summary(latent_noise),
            "noised": _tensor_summary(initial_latent),
        }
    else:
        initial_latent = latent_noise
    sample = {
        "latent": initial_latent,
        "camera": camera_noise.clone(),
    }
    sampler = FlowEulerCfgSampler()
    start_timestep = (
        float(args.img2img_strength)
        if args.img2img_strength is not None else 1.0
    )
    # Preserve the upstream shifted schedule shape while starting at an exact
    # flow time.  This matters for img2img: strength=0.5 must mean t=0.5,
    # independent of --shift.
    schedule_base_start = start_timestep / (
        args.shift - start_timestep * (args.shift - 1.0)
    )
    schedule_base = np.linspace(schedule_base_start, 0, args.steps + 1)
    schedule = args.shift * schedule_base / (1 + (args.shift - 1) * schedule_base)
    if camera_targets is not None:
        sample["camera"] = _fixed_camera_state(
            camera_noise,
            camera_targets,
            schedule[0],
            args.fixed_camera_trajectory,
        )
    snapshots = {0: {key: value.detach().cpu() for key, value in sample.items()}}
    initial_cameras = _camera_summaries(sample["camera"])
    trajectory_metrics = [{
        "step": 0,
        "t": float(schedule[0]),
        "latent": _tensor_summary(sample["latent"]),
        "camera": initial_cameras[0],
        "cameras": initial_cameras,
    }]

    with _timed_stage(report, "flow_sampling"):
        previous_latent = sample["latent"].clone()
        for index, (timestep, previous_timestep) in enumerate(
            zip(schedule[:-1], schedule[1:]), start=1
        ):
            if camera_targets is not None:
                sample["camera"] = _fixed_camera_state(
                    camera_noise,
                    camera_targets,
                    timestep,
                    args.fixed_camera_trajectory,
                )
            velocity_diagnostics = None
            active_view_index = None
            if velocity_fusion:
                if index <= args.primary_view_warmup_steps:
                    active_view_index = 0
                    current = {
                        "latent": sample["latent"].clone(),
                        "camera": sample["camera"][0:1].clone(),
                    }
                    velocity = sampler._cfg_prediction(
                        flow,
                        current,
                        timestep,
                        per_view_conditions[0],
                        per_view_negative[0],
                        args.guidance,
                    )
                else:
                    per_view_velocity = []
                    for view_index, view_conditions in enumerate(
                        per_view_conditions
                    ):
                        current = {
                            "latent": sample["latent"].clone(),
                            "camera": sample["camera"][
                                view_index:view_index + 1
                            ].clone(),
                        }
                        per_view_velocity.append(sampler._cfg_prediction(
                            flow,
                            current,
                            timestep,
                            view_conditions,
                            per_view_negative[view_index],
                            args.guidance,
                        ))
                    latent_velocities = [
                        item["latent"] for item in per_view_velocity
                    ]
                    normalized_weights = torch.tensor(
                        view_weights,
                        device=latent_velocities[0].device,
                        dtype=latent_velocities[0].dtype,
                    )
                    if not weighted_velocity_fusion:
                        normalized_weights.fill_(1.0)
                    normalized_weights /= normalized_weights.sum()
                    velocity = {
                        "latent": (
                            torch.stack(latent_velocities, dim=0)
                            * normalized_weights.view(-1, 1, 1, 1)
                        ).sum(dim=0),
                        "camera": torch.cat(
                            [item["camera"] for item in per_view_velocity],
                            dim=0,
                        ),
                    }
                    velocity_diagnostics = _velocity_agreement(
                        latent_velocities
                    )
            elif alternating_fusion:
                if index <= args.primary_view_warmup_steps:
                    active_view_index = 0
                else:
                    active_view_index = (
                        index - args.primary_view_warmup_steps - 1
                    ) % len(view_cycle_order)
                    active_view_index = view_cycle_order[active_view_index]
                current = {
                    "latent": sample["latent"].clone(),
                    "camera": sample["camera"][
                        active_view_index:active_view_index + 1
                    ].clone(),
                }
                velocity = sampler._cfg_prediction(
                    flow,
                    current,
                    timestep,
                    per_view_conditions[active_view_index],
                    per_view_negative[active_view_index],
                    args.guidance,
                )
            else:
                current = {
                    key: value.clone() for key, value in sample.items()
                }
                velocity = sampler._cfg_prediction(
                    flow,
                    current,
                    timestep,
                    conditions,
                    negative,
                    args.guidance,
                )
            delta_t = timestep - previous_timestep
            sample["latent"] = (
                sample["latent"] - velocity["latent"] * delta_t
            )
            if camera_targets is None:
                if alternating_fusion:
                    camera_slice = slice(
                        active_view_index, active_view_index + 1
                    )
                    sample["camera"][camera_slice] = (
                        sample["camera"][camera_slice]
                        - velocity["camera"] * delta_t
                    )
                else:
                    sample["camera"] = (
                        sample["camera"] - velocity["camera"] * delta_t
                    )
            else:
                sample["camera"] = _fixed_camera_state(
                    camera_noise,
                    camera_targets,
                    previous_timestep,
                    args.fixed_camera_trajectory,
                )
            delta = sample["latent"] - previous_latent
            cameras = _camera_summaries(sample["camera"])
            metric = {
                "step": index,
                "t": float(previous_timestep),
                "latent": _tensor_summary(sample["latent"]),
                "latent_delta_l2": float(torch.linalg.vector_norm(delta.float())),
                "camera": cameras[0],
                "cameras": cameras,
            }
            if velocity_diagnostics is not None:
                metric["latent_velocity_agreement"] = velocity_diagnostics
            if active_view_index is not None:
                metric["active_view_index"] = active_view_index
                metric["active_view_image"] = str(
                    input_views[active_view_index]["image"]
                )
            trajectory_metrics.append(metric)
            previous_latent = sample["latent"].clone()
            if index in selected_steps:
                snapshots[index] = {
                    key: value.detach().cpu() for key, value in sample.items()
                }
            print(
                f"[flow] {index:02d}/{args.steps} t={previous_timestep:.5f} "
                f"latent_std={metric['latent']['std']:.4f} "
                f"active_view={active_view_index} "
                f"cameras={[item['raw'] for item in cameras]}",
                flush=True,
            )

    report["flow_trajectory"] = trajectory_metrics
    report["camera_final"] = trajectory_metrics[-1]["camera"]
    report["cameras_final"] = trajectory_metrics[-1]["cameras"]
    ordered_steps = list(range(args.steps + 1))
    latent_history = []
    camera_history = []
    # Full history is reconstructed from metrics only for cameras unless the
    # state was selected.  Save selected exact tensors and all numeric metrics.
    for step in selected_steps:
        latent_history.append(snapshots[step]["latent"].to(torch.float16).numpy())
        camera_history.append(snapshots[step]["camera"].float().numpy())
    np.savez_compressed(
        args.output / "03-flow-trajectory.npz",
        selected_steps=np.asarray(selected_steps, dtype=np.int32),
        timesteps=schedule[np.asarray(selected_steps)],
        latent=np.concatenate(latent_history, axis=0),
        camera=np.stack(camera_history, axis=0),
        all_steps=np.asarray(ordered_steps, dtype=np.int32),
        all_camera=np.asarray(
            [
                [camera["raw"] for camera in item["cameras"]]
                for item in trajectory_metrics
            ],
            np.float32,
        ),
    )

    final_direction = np.asarray(report["camera_final"]["unit_view_direction"], np.float32)
    trajectory_images = []
    trajectory_labels = []
    with _timed_stage(report, "decode_trajectory"):
        for step in selected_steps:
            latent = snapshots[step]["latent"].to(device="cuda")
            gaussian, points = _decode_detailed(
                pipe, latent, args.trajectory_count, args.seed + 1000 + step
            )
            arrays = _gaussian_arrays(gaussian)
            _write_point_ply(
                trajectory_dir / f"step-{step:03d}-points.ply",
                arrays["means"], arrays["colors"],
            )
            np.savez_compressed(
                trajectory_dir / f"step-{step:03d}-anchors.npz",
                points=points["points"].detach().float().cpu().numpy(),
                log_probs=points["log_probs"].detach().float().cpu().numpy(),
            )
            try:
                frame = _render_gaussian(
                    gaussian, [final_direction], args.render_size,
                )[0]
                frame.save(renders_dir / f"trajectory-step-{step:03d}.jpg", quality=94)
                trajectory_images.append(frame)
                trajectory_labels.append(f"flow step {step}/{args.steps}")
            except ImportError as exc:
                report.setdefault("warnings", []).append(f"gsplat unavailable: {exc}")
            del gaussian, points, latent
            torch.cuda.empty_cache()
    if trajectory_images:
        _contact_sheet(trajectory_images, trajectory_labels, columns=4).save(
            renders_dir / "trajectory-contact-sheet.jpg", quality=94
        )

    final_latent = sample["latent"]
    report["density_variants"] = {}
    final_gaussian = None
    density_images = []
    density_labels = []
    with _timed_stage(report, "decode_density_variants"):
        for count in args.counts:
            gaussian, points = _decode_detailed(pipe, final_latent, count, args.seed + 2000)
            arrays = _gaussian_arrays(gaussian)
            point_values = points["points"].detach().float().cpu().numpy()[0]
            log_probs = points["log_probs"].detach().float().cpu().numpy()[0]
            _write_point_ply(
                anchors_dir / f"anchors-{count:06d}.ply",
                point_values - 0.5,
                _density_colors(log_probs),
            )
            np.savez_compressed(
                anchors_dir / f"anchors-{count:06d}.npz",
                points=point_values,
                log_probs=log_probs,
            )
            gaussian.save_ply(final_dir / f"triposplat-{count:06d}.ply")
            gaussian.save_splat(final_dir / f"triposplat-{count:06d}.splat")
            _write_point_ply(
                final_dir / f"points-{count:06d}.ply",
                arrays["means"], arrays["colors"],
            )
            report["density_variants"][str(count)] = {
                "anchors": len(point_values),
                "gaussian": _gaussian_summary(arrays),
            }
            density_axis, _ = _principal_axis(arrays["means"])
            density_frame = _render_gaussian(
                gaussian, [final_direction], args.render_size,
                up_direction=density_axis,
            )[0]
            density_frame.save(renders_dir / f"density-{count:06d}.jpg", quality=94)
            density_images.append(density_frame)
            density_labels.append(f"{count:,} gaussians")
            final_gaussian = gaussian
            del points
            torch.cuda.empty_cache()

    _contact_sheet(density_images, density_labels, columns=4).save(
        renders_dir / "density-contact-sheet.jpg", quality=94
    )
    final_arrays = _gaussian_arrays(final_gaussian)
    body_axis, pca_eigenvalues = _principal_axis(final_arrays["means"])
    report["orientation_gauge"] = {
        "principal_body_axis": body_axis.tolist(),
        "pca_eigenvalues": pca_eigenvalues.tolist(),
        "axis_tilt_from_world_z_degrees": math.degrees(
            math.acos(float(np.clip(np.dot(body_axis, [0, 0, 1]), -1, 1)))
        ),
        "camera_axis_dot": float(np.dot(final_direction, body_axis)),
    }
    upright_rotation = _align_vector_rotation(
        body_axis, np.array([0, 0, 1], dtype=np.float32)
    )
    upright_center = final_arrays["means"].mean(axis=0, keepdims=True)
    upright_points = (
        final_arrays["means"] - upright_center
    ) @ upright_rotation.T
    _write_point_ply(
        final_dir / f"points-{len(upright_points):06d}-upright.ply",
        upright_points,
        final_arrays["colors"],
    )
    orbit_directions = [
        _rotate_axis(final_direction, np.array([0, 0, 1], np.float32), angle)
        for angle in range(0, 360, 45)
    ]
    with _timed_stage(report, "render_final_orbit"):
        orbit = _render_gaussian(final_gaussian, orbit_directions, args.render_size)
        orbit_labels = [f"camera + {angle}°" for angle in range(0, 360, 45)]
        for angle, frame in zip(range(0, 360, 45), orbit):
            frame.save(renders_dir / f"final-orbit-{angle:03d}.jpg", quality=94)
        _contact_sheet(orbit, orbit_labels, columns=4).save(
            renders_dir / "final-orbit-contact-sheet.jpg", quality=94
        )
        body_orbit_directions = [
            _rotate_axis(final_direction, body_axis, angle)
            for angle in range(0, 360, 45)
        ]
        body_orbit = _render_gaussian(
            final_gaussian,
            body_orbit_directions,
            args.render_size,
            up_direction=body_axis,
        )
        for angle, frame in zip(range(0, 360, 45), body_orbit):
            frame.save(renders_dir / f"body-axis-orbit-{angle:03d}.jpg", quality=94)
        _contact_sheet(body_orbit, orbit_labels, columns=4).save(
            renders_dir / "body-axis-orbit-contact-sheet.jpg", quality=94
        )

    torch.save(
        {
            "latent": final_latent.detach().to(torch.float16).cpu(),
            "camera": sample["camera"].detach().float().cpu(),
        },
        args.output / "03-final-latent-camera.pt",
    )
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )
    (args.output / "process.html").write_text(_html_report(report), encoding="utf-8")
    print(f"[done] {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
