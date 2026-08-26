#!/usr/bin/env python3
"""Replace a full-body TripoSPLAT head with a separately reconstructed bust.

The experiment performs a robust axis-aligned Sim(3) registration, removes the
shoulders/base of the bust, and compares a hard replacement with a narrow
optical-density feather at the neck.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch


ROOT = Path(__file__).resolve().parents[1]
ROUNDTRIP_SCRIPT = ROOT / "scripts/experiment-triposplat-vae-rendered-features-roundtrip.py"
HEAD_PATCH_SCRIPT = ROOT / "scripts/experiment-triposplat-head-patch-refinement.py"
EXPERIMENTS = Path("/home/mirmik/mnt-nvme/canonical-experiments")
DEFAULT_BODY = (
    EXPERIMENTS
    / "triposplat-process-vaan-4view-then-face5-warmup8-density262k-524k-"
    "render1024-steps33-seed42-run/06-final-gaussians/triposplat-524288.ply"
)
DEFAULT_HEAD = (
    EXPERIMENTS
    / "triposplat-head-img2img-renderbase-frontonly-strength100-steps24-seed42-v2/"
    "06-final-gaussians/triposplat-524288.ply"
)
DEFAULT_OUTPUT = EXPERIMENTS / "triposplat-vaan-opacity-head-merge-v2"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--body", type=Path, default=DEFAULT_BODY)
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--head-z-min",
        type=float,
        default=-0.12,
        help="Local Z cutoff that removes the reconstructed bust shoulders/base",
    )
    parser.add_argument(
        "--registration-head-z-min",
        type=float,
        default=-0.25,
        help="Wider source crop used only to estimate the Sim(3) registration",
    )
    parser.add_argument("--head-opacity-min", type=float, default=0.02)
    parser.add_argument("--body-z-min", type=float, default=0.315)
    parser.add_argument(
        "--registration-body-z-min",
        type=float,
        default=0.30,
        help="Body head crop used only to estimate the Sim(3) registration",
    )
    parser.add_argument("--body-abs-x", type=float, default=0.13)
    parser.add_argument("--body-abs-y", type=float, default=0.15)
    parser.add_argument(
        "--registration-quantile",
        type=float,
        default=0.01,
        help="Tail fraction ignored while matching robust head bounds",
    )
    parser.add_argument(
        "--body-feather",
        type=float,
        default=0.045,
        help="World-space Z width over which the old head optical density fades",
    )
    parser.add_argument(
        "--head-feather",
        type=float,
        default=0.10,
        help="Source-local Z width over which the inserted head fades in",
    )
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument(
        "--render-deps",
        type=Path,
        default=Path("/tmp/triposplat-render-deps"),
        help="Optional pip target containing gsplat",
    )
    return parser.parse_args()


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def smoothstep(edge0: float, edge1: float, value: torch.Tensor) -> torch.Tensor:
    if edge1 <= edge0:
        raise ValueError("smoothstep requires edge1 > edge0")
    t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def scale_optical_density(opacity: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Scale alpha through optical thickness rather than linear alpha."""
    alpha = opacity.clamp(0.0, 1.0 - 1e-7)
    tau = -torch.log1p(-alpha)
    return -torch.expm1(-tau * weight.clamp(0.0, 1.0))


def robust_bounds(points: torch.Tensor, quantile: float) -> tuple[np.ndarray, np.ndarray]:
    values = points.detach().float().cpu().numpy()
    low = np.quantile(values, quantile, axis=0).astype(np.float32)
    high = np.quantile(values, 1.0 - quantile, axis=0).astype(np.float32)
    return low, high


def register_head(
    body: dict[str, torch.Tensor],
    head: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict]:
    body_mask = (
        (body["means"][:, 2] > args.registration_body_z_min)
        & (body["means"][:, 0].abs() < args.body_abs_x)
        & (body["means"][:, 1].abs() < args.body_abs_y)
        & (body["opacities"] > args.head_opacity_min)
    )
    head_mask = (
        (head["means"][:, 2] > args.registration_head_z_min)
        & (head["opacities"] > args.head_opacity_min)
    )
    if int(body_mask.sum()) < 1_000 or int(head_mask.sum()) < 1_000:
        raise RuntimeError(
            f"Not enough head points: body={int(body_mask.sum())}, "
            f"insert={int(head_mask.sum())}"
        )

    body_low, body_high = robust_bounds(body["means"][body_mask], args.registration_quantile)
    head_low, head_high = robust_bounds(head["means"][head_mask], args.registration_quantile)
    body_extent = np.maximum(body_high - body_low, 1e-6)
    head_extent = np.maximum(head_high - head_low, 1e-6)
    axis_scale_ratios = body_extent / head_extent
    uniform_scale = float(np.median(axis_scale_ratios))
    body_center = (body_low + body_high) * 0.5
    head_center = (head_low + head_high) * 0.5
    translation = body_center - head_center * uniform_scale

    insert_mask = (
        (head["means"][:, 2] > args.head_z_min)
        & (head["opacities"] > args.head_opacity_min)
    )
    selected = {key: value[insert_mask].clone() for key, value in head.items()}
    selected["means"] = selected["means"] * uniform_scale + torch.as_tensor(
        translation, dtype=selected["means"].dtype, device=selected["means"].device
    )
    selected["scales"] = selected["scales"] * uniform_scale
    registration = {
        "kind": "robust-axis-aligned-sim3",
        "quantile": args.registration_quantile,
        "body_points": int(body_mask.sum()),
        "registration_head_points": int(head_mask.sum()),
        "inserted_head_points": int(insert_mask.sum()),
        "body_bounds": {"low": body_low.tolist(), "high": body_high.tolist()},
        "head_bounds": {"low": head_low.tolist(), "high": head_high.tolist()},
        "axis_scale_ratios": axis_scale_ratios.tolist(),
        "uniform_scale": uniform_scale,
        "translation": translation.tolist(),
    }
    return selected, insert_mask, registration


def hard_merge(
    body: dict[str, torch.Tensor],
    inserted: dict[str, torch.Tensor],
    args: argparse.Namespace,
) -> dict[str, torch.Tensor]:
    replaced = (
        (body["means"][:, 2] > args.body_z_min)
        & (body["means"][:, 0].abs() < args.body_abs_x)
        & (body["means"][:, 1].abs() < args.body_abs_y)
    )
    return {
        key: torch.cat((body[key][~replaced], inserted[key]), dim=0)
        for key in body
    }


def feather_merge(
    body: dict[str, torch.Tensor],
    inserted: dict[str, torch.Tensor],
    selected_head_local_z: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, torch.Tensor], dict]:
    replace_column = (
        (body["means"][:, 0].abs() < args.body_abs_x)
        & (body["means"][:, 1].abs() < args.body_abs_y)
    )
    body_replacement = smoothstep(
        args.body_z_min,
        args.body_z_min + args.body_feather,
        body["means"][:, 2],
    )
    body_keep_weight = torch.where(
        replace_column,
        1.0 - body_replacement,
        torch.ones_like(body_replacement),
    )
    inserted_weight = smoothstep(
        args.head_z_min,
        args.head_z_min + args.head_feather,
        selected_head_local_z,
    )

    faded_body = {key: value.clone() for key, value in body.items()}
    faded_body["opacities"] = scale_optical_density(
        body["opacities"], body_keep_weight
    )
    faded_inserted = {key: value.clone() for key, value in inserted.items()}
    faded_inserted["opacities"] = scale_optical_density(
        inserted["opacities"], inserted_weight
    )

    # Discard effectively invisible splats. They add cost but no useful transition.
    body_visible = faded_body["opacities"] > 1e-5
    inserted_visible = faded_inserted["opacities"] > 1e-5
    merged = {
        key: torch.cat(
            (faded_body[key][body_visible], faded_inserted[key][inserted_visible]),
            dim=0,
        )
        for key in body
    }
    diagnostics = {
        "body_keep_weight": {
            "min": float(body_keep_weight.min()),
            "mean": float(body_keep_weight.mean()),
            "zero_count": int((body_keep_weight == 0).sum()),
            "partial_count": int(
                ((body_keep_weight > 0) & (body_keep_weight < 1)).sum()
            ),
        },
        "inserted_weight": {
            "min": float(inserted_weight.min()),
            "mean": float(inserted_weight.mean()),
            "zero_count": int((inserted_weight == 0).sum()),
            "partial_count": int(
                ((inserted_weight > 0) & (inserted_weight < 1)).sum()
            ),
        },
        "visible_body_points": int(body_visible.sum()),
        "visible_inserted_points": int(inserted_visible.sum()),
    }
    return merged, diagnostics


@torch.inference_mode()
def render_views(rf, source: dict[str, torch.Tensor], cameras: list[dict]) -> list[Image.Image]:
    images = []
    for camera in cameras:
        tensor = rf.render_source(source, camera, int(camera["K"][0, 2] * 2))
        images.append(
            Image.fromarray(
                (tensor.permute(1, 2, 0) * 255).byte().cpu().numpy(), "RGB"
            )
        )
    return images


def contact_sheet(images: list[Image.Image], labels: list[str]) -> Image.Image:
    width, height = images[0].size
    columns = 3
    rows = (len(images) + columns - 1) // columns
    label_height = 24
    sheet = Image.new("RGB", (columns * width, rows * (height + label_height)), "#202020")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, (image, label) in enumerate(zip(images, labels)):
        x = index % columns * width
        y = index // columns * (height + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 6, y + height + 5), label, fill="white", font=font)
    return sheet


def main() -> int:
    args = arguments()
    if args.render_deps.is_dir():
        sys.path.insert(0, str(args.render_deps))
        bundled_bin = args.render_deps / "bin"
        if bundled_bin.is_dir():
            os.environ["PATH"] = f"{bundled_bin}:{os.environ.get('PATH', '')}"
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"Output directory is not empty: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    rf = load_module("triposplat_render_helpers", ROUNDTRIP_SCRIPT)
    hp = load_module("triposplat_head_patch_helpers", HEAD_PATCH_SCRIPT)
    base = rf.load_base_module()

    print("Loading full body and independently reconstructed head", flush=True)
    body = rf.load_source_gaussians(args.body, base)
    head = rf.load_source_gaussians(args.head, base)
    selected_head_local_z = head["means"][:, 2].clone()
    inserted, selected_mask, registration = register_head(body, head, args)
    selected_head_local_z = selected_head_local_z[selected_mask]
    print(json.dumps(registration, indent=2), flush=True)

    hard = hard_merge(body, inserted, args)
    feathered, feather_diagnostics = feather_merge(
        body, inserted, selected_head_local_z, args
    )

    hp.write_gaussian_ply(args.output / "01-hard-merged-gaussians.ply", hard, rf)
    hp.write_gaussian_ply(
        args.output / "02-feathered-merged-gaussians.ply", feathered, rf
    )
    for name, source in (("hard", hard), ("feathered", feathered)):
        base.write_point_ply(
            args.output / f"0{'1' if name == 'hard' else '2'}-{name}-centers.ply",
            source["means"].detach().cpu().numpy(),
            source["colors"].detach().cpu().numpy() * 255,
        )

    cameras = [
        camera
        for camera in rf.orbit_cameras(body["means"], args.render_size)
        if camera["elevation"] == 0
    ]
    body_views = render_views(rf, body, cameras)
    hard_views = render_views(rf, hard, cameras)
    feathered_views = render_views(rf, feathered, cameras)
    render_dir = args.output / "03-renders"
    render_dir.mkdir()
    comparison_images = []
    comparison_labels = []
    for index, camera in enumerate(cameras):
        azimuth = int(camera["azimuth"])
        body_views[index].save(render_dir / f"original-{azimuth:03d}.jpg", quality=94)
        hard_views[index].save(render_dir / f"hard-{azimuth:03d}.jpg", quality=94)
        feathered_views[index].save(
            render_dir / f"feathered-{azimuth:03d}.jpg", quality=94
        )
        comparison_images.extend(
            (body_views[index], hard_views[index], feathered_views[index])
        )
        comparison_labels.extend(
            (f"original {azimuth:03d}", f"hard {azimuth:03d}", f"feather {azimuth:03d}")
        )
    contact_sheet(comparison_images, comparison_labels).save(
        args.output / "03-original-vs-hard-vs-feathered-orbit.jpg", quality=94
    )

    closeup_cameras = [
        camera
        for camera in rf.orbit_cameras(inserted["means"], args.render_size)
        if camera["elevation"] == 0
    ]
    body_closeups = render_views(rf, body, closeup_cameras)
    hard_closeups = render_views(rf, hard, closeup_cameras)
    feathered_closeups = render_views(rf, feathered, closeup_cameras)
    closeup_images = []
    closeup_labels = []
    for index, camera in enumerate(closeup_cameras):
        azimuth = int(camera["azimuth"])
        body_closeups[index].save(
            render_dir / f"head-original-{azimuth:03d}.jpg", quality=94
        )
        hard_closeups[index].save(
            render_dir / f"head-hard-{azimuth:03d}.jpg", quality=94
        )
        feathered_closeups[index].save(
            render_dir / f"head-feathered-{azimuth:03d}.jpg", quality=94
        )
        closeup_images.extend(
            (body_closeups[index], hard_closeups[index], feathered_closeups[index])
        )
        closeup_labels.extend(
            (f"original {azimuth:03d}", f"hard {azimuth:03d}", f"feather {azimuth:03d}")
        )
    contact_sheet(closeup_images, closeup_labels).save(
        args.output / "04-head-original-vs-hard-vs-feathered-orbit.jpg", quality=94
    )

    report = {
        "schema": "diffusion-editor.triposplat-opacity-head-merge.v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "body": str(args.body),
        "head": str(args.head),
        "parameters": {
            "head_z_min": args.head_z_min,
            "registration_head_z_min": args.registration_head_z_min,
            "head_opacity_min": args.head_opacity_min,
            "body_z_min": args.body_z_min,
            "registration_body_z_min": args.registration_body_z_min,
            "body_abs_x": args.body_abs_x,
            "body_abs_y": args.body_abs_y,
            "registration_quantile": args.registration_quantile,
            "body_feather": args.body_feather,
            "head_feather": args.head_feather,
        },
        "registration": registration,
        "feather": feather_diagnostics,
        "counts": {
            "body": len(body["means"]),
            "head": len(head["means"]),
            "selected_head": len(inserted["means"]),
            "hard_merge": len(hard["means"]),
            "feathered_merge": len(feathered["means"]),
        },
    }
    (args.output / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
