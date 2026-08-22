#!/usr/bin/env python3
"""Run one reproducible joint DA3 multi-view reconstruction experiment.

This is deliberately an experiment driver rather than an editor feature.  It
keeps DA3's float depth, confidence, camera matrices, and processed RGB frames
and produces both a full-scene cloud and a chroma-keyed subject cloud without
normalizing their 3D coordinates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from scipy import ndimage


DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument(
        "--use-ray-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ref-view-strategy",
        choices=("first", "middle", "saddle_balanced", "saddle_sim_range"),
        default="first",
    )
    parser.add_argument("--max-full-points", type=int, default=1_000_000)
    parser.add_argument("--max-subject-points", type=int, default=750_000)
    parser.add_argument(
        "--subject-depth-tolerance",
        type=float,
        default=0.6,
        help=(
            "Post-inference rejection distance around each view's median "
            "subject depth; zero disables the filtered cloud."
        ),
    )
    parser.add_argument(
        "--nominal-orbit-radius",
        type=float,
        default=0.0,
        help="Provide an evenly spaced OpenCV camera orbit when greater than zero.",
    )
    parser.add_argument("--horizontal-fov-degrees", type=float, default=34.0)
    parser.add_argument(
        "--da3-source",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / ".venv-workers/share/diffusion-editor/depth-anything-3/src",
    )
    return parser.parse_args()


def _nominal_orbit_cameras(
    image_paths: list[Path],
    radius: float,
    horizontal_fov_degrees: float,
) -> tuple[np.ndarray, np.ndarray]:
    if radius <= 0.0:
        raise ValueError("nominal orbit radius must be positive")
    if not 1.0 < horizontal_fov_degrees < 179.0:
        raise ValueError("horizontal FOV must be between 1 and 179 degrees")

    extrinsics: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    world_down = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    for index, path in enumerate(image_paths):
        with Image.open(path) as image:
            width, height = image.size
        theta = 2.0 * math.pi * index / len(image_paths)
        center = np.array(
            (radius * math.sin(theta), 0.0, -radius * math.cos(theta)),
            dtype=np.float64,
        )
        forward = -center / np.linalg.norm(center)
        right = np.cross(world_down, forward)
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack((right, down, forward), axis=0)
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = rotation
        world_to_camera[:3, 3] = -rotation @ center
        extrinsics.append(world_to_camera.astype(np.float32))

        focal = (0.5 * width) / math.tan(
            math.radians(horizontal_fov_degrees) * 0.5
        )
        intrinsics.append(
            np.array(
                (
                    (focal, 0.0, (width - 1) * 0.5),
                    (0.0, focal, (height - 1) * 0.5),
                    (0.0, 0.0, 1.0),
                ),
                dtype=np.float32,
            )
        )
    return np.stack(extrinsics), np.stack(intrinsics)


def _camera_to_world_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    camera = np.stack(
        (
            (xx.astype(np.float32) - cx) * depth / fx,
            (yy.astype(np.float32) - cy) * depth / fy,
            depth,
        ),
        axis=-1,
    ).reshape(-1, 3)

    world_to_camera = np.eye(4, dtype=np.float64)
    rows, columns = extrinsics.shape
    if (rows, columns) not in ((3, 4), (4, 4)):
        raise ValueError(f"unexpected extrinsics shape: {extrinsics.shape}")
    world_to_camera[:rows, :columns] = extrinsics
    camera_to_world = np.linalg.inv(world_to_camera)
    world = (
        camera.astype(np.float64) @ camera_to_world[:3, :3].T
        + camera_to_world[:3, 3]
    )
    return np.ascontiguousarray(world, dtype=np.float32)


def _subject_mask(rgb: np.ndarray) -> np.ndarray:
    """Select the largest central non-green component without edge dilation."""

    values = rgb.astype(np.int16)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    green_screen = (
        (green - red >= 12)
        & (green - blue >= 12)
        & (green * 100 >= red * 112)
        & (green * 100 >= blue * 112)
    )
    candidates = ~green_screen
    labels, count = ndimage.label(candidates)
    if count == 0:
        return np.zeros(candidates.shape, dtype=bool)

    height, width = candidates.shape
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    best_label = 0
    best_score = -1.0
    for label in np.argsort(sizes)[::-1][:32]:
        if sizes[label] < 64:
            break
        ys, xs = np.nonzero(labels == label)
        center_x = float(xs.mean()) / max(width - 1, 1)
        center_y = float(ys.mean()) / max(height - 1, 1)
        centrality = max(0.0, 1.0 - abs(center_x - 0.5) * 1.5)
        verticality = max(0.0, 1.0 - abs(center_y - 0.52))
        score = float(sizes[label]) * (0.35 + centrality * verticality)
        if score > best_score:
            best_label = int(label)
            best_score = score
    return labels == best_label


def _limit_points(
    positions: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    maximum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if maximum < 1 or len(positions) <= maximum:
        return positions, colors, confidence
    indices = np.linspace(0, len(positions) - 1, maximum, dtype=np.int64)
    return positions[indices], colors[indices], confidence[indices]


def _save_cloud(
    path: Path,
    positions: list[np.ndarray],
    colors: list[np.ndarray],
    confidence: list[np.ndarray],
    maximum: int,
) -> int:
    xyz = np.concatenate(positions, axis=0)
    rgb = np.concatenate(colors, axis=0)
    conf = np.concatenate(confidence, axis=0)
    valid = np.isfinite(xyz).all(axis=1) & np.isfinite(conf)
    xyz, rgb, conf = xyz[valid], rgb[valid], conf[valid]
    xyz, rgb, conf = _limit_points(xyz, rgb, conf, maximum)
    np.savez_compressed(
        path,
        positions=np.ascontiguousarray(xyz, dtype=np.float32),
        colors=np.ascontiguousarray(rgb, dtype=np.float32),
        confidence=np.ascontiguousarray(conf, dtype=np.float32),
    )
    return len(xyz)


def main() -> int:
    args = _arguments()
    if len(args.inputs) < 2:
        raise SystemExit("joint multi-view inference needs at least two images")
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"missing input images: {missing}")
    if not (args.da3_source / "depth_anything_3/api.py").is_file():
        raise SystemExit(f"DA3 source is missing: {args.da3_source}")

    sys.path.insert(0, str(args.da3_source))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/diffusion-editor-matplotlib")
    from depth_anything_3.api import DepthAnything3

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.model}...", flush=True)
    model = DepthAnything3.from_pretrained(args.model).to("cuda").eval()
    print(
        f"Joint inference: {len(args.inputs)} views at {args.process_res}px, "
        f"ray_pose={args.use_ray_pose}...",
        flush=True,
    )
    input_extrinsics = None
    input_intrinsics = None
    if args.nominal_orbit_radius > 0.0:
        input_extrinsics, input_intrinsics = _nominal_orbit_cameras(
            args.inputs,
            args.nominal_orbit_radius,
            args.horizontal_fov_degrees,
        )
        print(
            f"Pose conditioning: nominal {args.nominal_orbit_radius:g} m "
            f"orbit, horizontal FOV {args.horizontal_fov_degrees:g}°",
            flush=True,
        )
    prediction = model.inference(
        image=[str(path) for path in args.inputs],
        extrinsics=input_extrinsics,
        intrinsics=input_intrinsics,
        process_res=args.process_res,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
    )

    depth = np.ascontiguousarray(prediction.depth, dtype=np.float32)
    confidence = np.ascontiguousarray(prediction.conf, dtype=np.float32)
    intrinsics = np.ascontiguousarray(prediction.intrinsics, dtype=np.float32)
    extrinsics = np.ascontiguousarray(prediction.extrinsics, dtype=np.float32)
    images = np.ascontiguousarray(prediction.processed_images, dtype=np.uint8)
    if not (
        len(depth) == len(confidence) == len(intrinsics)
        == len(extrinsics) == len(images) == len(args.inputs)
    ):
        raise RuntimeError("DA3 returned inconsistent multi-view dimensions")

    np.savez_compressed(
        args.output_dir / "prediction.npz",
        depth=depth,
        confidence=confidence,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        processed_images=images,
    )

    full_positions: list[np.ndarray] = []
    full_colors: list[np.ndarray] = []
    full_confidence: list[np.ndarray] = []
    subject_positions: list[np.ndarray] = []
    subject_colors: list[np.ndarray] = []
    subject_confidence: list[np.ndarray] = []
    filtered_positions: list[np.ndarray] = []
    filtered_colors: list[np.ndarray] = []
    filtered_confidence: list[np.ndarray] = []
    mask_counts: list[int] = []
    filtered_counts: list[int] = []
    for index in range(len(args.inputs)):
        xyz = _camera_to_world_points(
            depth[index], intrinsics[index], extrinsics[index]
        )
        rgb = images[index].reshape(-1, 3).astype(np.float32) / 255.0
        conf = confidence[index].reshape(-1)
        full_positions.append(xyz)
        full_colors.append(rgb)
        full_confidence.append(conf)

        mask = _subject_mask(images[index]).reshape(-1)
        mask_counts.append(int(mask.sum()))
        subject_positions.append(xyz[mask])
        subject_colors.append(rgb[mask])
        subject_confidence.append(conf[mask])
        if args.subject_depth_tolerance > 0.0:
            flat_depth = depth[index].reshape(-1)
            median_depth = float(np.median(flat_depth[mask]))
            filtered = mask & (
                np.abs(flat_depth - median_depth)
                <= args.subject_depth_tolerance
            )
        else:
            filtered = mask
        filtered_counts.append(int(filtered.sum()))
        filtered_positions.append(xyz[filtered])
        filtered_colors.append(rgb[filtered])
        filtered_confidence.append(conf[filtered])

    full_count = _save_cloud(
        args.output_dir / "full-cloud.npz",
        full_positions,
        full_colors,
        full_confidence,
        args.max_full_points,
    )
    subject_count = _save_cloud(
        args.output_dir / "subject-cloud.npz",
        subject_positions,
        subject_colors,
        subject_confidence,
        args.max_subject_points,
    )
    filtered_count = _save_cloud(
        args.output_dir / "subject-cloud-filtered.npz",
        filtered_positions,
        filtered_colors,
        filtered_confidence,
        args.max_subject_points,
    )
    manifest = {
        "model": args.model,
        "inputs": [str(path.resolve()) for path in args.inputs],
        "process_res": args.process_res,
        "use_ray_pose": args.use_ray_pose,
        "ref_view_strategy": args.ref_view_strategy,
        "camera_mode": (
            "nominal_orbit"
            if args.nominal_orbit_radius > 0.0
            else "predicted"
        ),
        "nominal_orbit_radius": (
            args.nominal_orbit_radius
            if args.nominal_orbit_radius > 0.0
            else None
        ),
        "horizontal_fov_degrees": (
            args.horizontal_fov_degrees
            if args.nominal_orbit_radius > 0.0
            else None
        ),
        "depth_shape": list(depth.shape),
        "is_metric": bool(getattr(prediction, "is_metric", False)),
        "scale_factor": (
            float(prediction.scale_factor)
            if getattr(prediction, "scale_factor", None) is not None
            else None
        ),
        "full_point_count": full_count,
        "subject_point_count": subject_count,
        "filtered_subject_point_count": filtered_count,
        "subject_pixels_per_view": mask_counts,
        "filtered_subject_pixels_per_view": filtered_counts,
        "subject_depth_tolerance": args.subject_depth_tolerance,
        "confidence_min": float(confidence.min()),
        "confidence_max": float(confidence.max()),
        "confidence_percentiles": {
            str(value): float(np.percentile(confidence, value))
            for value in (2, 10, 25, 50, 75, 90, 98)
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
