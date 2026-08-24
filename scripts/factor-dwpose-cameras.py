#!/usr/bin/env python3
"""Recover free 3D points and effective cameras from nine DWPose views."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diffusion_editor.generation.camera_factorization import (
    CameraFactorizationResult,
    camera_yaw_pitch_roll,
    factor_scaled_orthographic,
    reflect_depth,
    refine_scaled_orthographic,
)
from diffusion_editor.generation.pose_estimation import PoseEstimationResult
from diffusion_editor.workers.pose_backend import (
    COCO_BODY_NAMES,
    COCO_FOOT_NAMES,
    PoseBackend,
)


TRACK_NAMES = COCO_BODY_NAMES + COCO_FOOT_NAMES
TRACK_SETS = {
    "body13": (
        "nose",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ),
    "body17": COCO_BODY_NAMES,
    "body23": TRACK_NAMES,
}
ELEVATION_ORDER = (30, 0, -30)
AZIMUTH_ORDER = (-45, 0, 45)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument(
        "--track-set", choices=tuple(TRACK_SETS), default="body17")
    parser.add_argument("--soft-l1-pixels", type=float, default=7.0)
    parser.add_argument("--max-evaluations", type=int, default=1000)
    parser.add_argument(
        "--reuse-keypoints",
        action="store_true",
        help="Reuse output_dir/dwpose-keypoints.json when present.",
    )
    return parser.parse_args()


def signed_azimuth(value: float) -> float:
    return float((value + 180.0) % 360.0 - 180.0)


def load_records(manifest_path: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) < 3:
        raise SystemExit("manifest must contain at least three view jobs")
    records = []
    for job in jobs:
        path = Path(job["output"]).expanduser()
        if not path.is_file():
            path = manifest_path.parent / path.name
        if not path.is_file():
            raise SystemExit(f"missing view image: {path}")
        records.append({
            "name": path.stem,
            "path": path.resolve(),
            "nominal_yaw": signed_azimuth(float(job["azimuth_degrees"])),
            "nominal_pitch": float(job["elevation_degrees"]),
        })
    return manifest, records


def extract_keypoints(records: list[dict], cache_path: Path) -> list[dict]:
    backend = PoseBackend()
    payload = []
    for index, record in enumerate(records, start=1):
        print(f"DWPose {index}/{len(records)}: {record['name']}", flush=True)
        result = backend.estimate("dwpose", record["path"])
        if not result.poses:
            raise RuntimeError(f"DWPose found no person in {record['path']}")
        pose = max(
            result.poses,
            key=lambda item: sum(point.score for point in item.keypoints),
        )
        points = {point.name: point.to_dict() for point in pose.keypoints}
        missing = [name for name in TRACK_NAMES if name not in points]
        if missing:
            raise RuntimeError(
                f"{record['name']} is missing DWPose tracks: {missing}")
        payload.append({
            "name": record["name"],
            "path": str(record["path"]),
            "width": result.width,
            "height": result.height,
            "points": {name: points[name] for name in TRACK_NAMES},
        })
    cache_path.write_text(
        json.dumps({
            "schema": "diffusion-editor.dwpose-camera-tracks",
            "schema_version": 1,
            "track_names": TRACK_NAMES,
            "views": payload,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def load_or_extract_keypoints(
        records: list[dict], cache_path: Path, reuse: bool) -> list[dict]:
    if reuse and cache_path.is_file():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        views = payload.get("views")
        if (
                payload.get("track_names") == list(TRACK_NAMES)
                and isinstance(views, list)
                and [view.get("name") for view in views]
                == [record["name"] for record in records]):
            print(f"Reusing {cache_path}", flush=True)
            return views
        raise RuntimeError("cached DWPose tracks do not match this manifest")
    return extract_keypoints(records, cache_path)


def observation_arrays(
        records: list[dict],
        views: list[dict],
        confidence: float,
        track_names: tuple[str, ...]):
    observations = []
    weights = []
    sizes = []
    for record, view in zip(records, views):
        if record["name"] != view["name"]:
            raise RuntimeError("DWPose cache view order mismatch")
        width, height = int(view["width"]), int(view["height"])
        sizes.append((width, height))
        view_points = []
        view_weights = []
        for name in track_names:
            point = view["points"][name]
            view_points.append((
                (float(point["x"]) - width * 0.5) / height,
                (height * 0.5 - float(point["y"])) / height,
            ))
            score = float(point["score"])
            view_weights.append(score if score >= confidence else 0.0)
        observations.append(view_points)
        weights.append(view_weights)
    observations = np.asarray(observations, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    visible = np.sum(weights > 0.0, axis=0)
    if np.any(visible < 3):
        missing = [
            track_names[index] for index in np.flatnonzero(visible < 3)]
        raise RuntimeError(f"too few confident observations for {missing}")
    # Factorization needs a complete matrix. Low-confidence coordinates remain
    # usable for its initialization, while robust refinement gives them zero
    # weight.
    return observations, weights, sizes


def choose_depth_sign(
        result: CameraFactorizationResult,
        records: list[dict]) -> CameraFactorizationResult:
    nominal = np.array([record["nominal_yaw"] for record in records])
    candidates = (result, reflect_depth(result))

    def agreement(candidate: CameraFactorizationResult) -> float:
        fitted = np.array([
            camera_yaw_pitch_roll(camera.rotation)[0]
            for camera in candidate.cameras
        ])
        nonzero = np.abs(nominal) > 1.0
        if np.sum(nonzero) < 2:
            return 0.0
        return float(np.corrcoef(nominal[nonzero], fitted[nonzero])[0, 1])

    return max(candidates, key=agreement)


def pixel_predictions(
        result: CameraFactorizationResult,
        sizes: list[tuple[int, int]]) -> np.ndarray:
    normalized = result.project()
    pixels = []
    for view, (width, height) in enumerate(sizes):
        pixels.append(np.stack((
            normalized[view, :, 0] * height + width * 0.5,
            height * 0.5 - normalized[view, :, 1] * height,
        ), axis=1))
    return np.stack(pixels)


def observed_pixels(
        observations: np.ndarray,
        sizes: list[tuple[int, int]]) -> np.ndarray:
    pixels = []
    for view, (width, height) in enumerate(sizes):
        pixels.append(np.stack((
            observations[view, :, 0] * height + width * 0.5,
            height * 0.5 - observations[view, :, 1] * height,
        ), axis=1))
    return np.stack(pixels)


def residual_statistics(
        predicted: np.ndarray,
        observed: np.ndarray,
        weights: np.ndarray):
    distances = np.linalg.norm(predicted - observed, axis=2)
    views = []
    for view in range(distances.shape[0]):
        valid = weights[view] > 0.0
        weighted_rms = np.sqrt(
            np.sum(weights[view, valid] * distances[view, valid] ** 2)
            / np.sum(weights[view, valid])
        )
        views.append({
            "rms_pixels": float(np.sqrt(np.mean(distances[view, valid] ** 2))),
            "weighted_rms_pixels": float(weighted_rms),
            "median_pixels": float(np.median(distances[view, valid])),
            "max_pixels": float(np.max(distances[view, valid])),
        })
    valid = weights > 0.0
    return views, {
        "rms_pixels": float(np.sqrt(np.mean(distances[valid] ** 2))),
        "weighted_rms_pixels": float(np.sqrt(
            np.sum(weights[valid] * distances[valid] ** 2)
            / np.sum(weights[valid])
        )),
        "median_pixels": float(np.median(distances[valid])),
        "max_pixels": float(np.max(distances[valid])),
    }


def draw_overlays(
        output_dir: Path,
        records: list[dict],
        observed: np.ndarray,
        predicted: np.ndarray,
        view_stats: list[dict],
        cameras,
) -> list[Path]:
    paths = []
    for view, record in enumerate(records):
        image = Image.open(record["path"]).convert("RGBA")
        draw = ImageDraw.Draw(image, "RGBA")
        for actual, fitted in zip(observed[view], predicted[view]):
            draw.line((*actual, *fitted), fill=(255, 210, 35, 190), width=2)
            x, y = actual
            draw.ellipse((x - 5, y - 5, x + 5, y + 5),
                         fill=(60, 255, 110, 230), outline=(10, 30, 10, 255), width=2)
            x, y = fitted
            draw.line((x - 6, y, x + 6, y), fill=(255, 40, 210, 255), width=3)
            draw.line((x, y - 6, x, y + 6), fill=(255, 40, 210, 255), width=3)
        yaw, world_pitch, roll = camera_yaw_pitch_roll(
            cameras[view].rotation)
        elevation = -world_pitch
        label = (
            f"{record['name']}  nominal {record['nominal_yaw']:+.0f}/"
            f"{record['nominal_pitch']:+.0f}  fitted {yaw:+.1f}/"
            f"{elevation:+.1f}  "
            f"RMS {view_stats[view]['weighted_rms_pixels']:.1f}px"
        )
        draw.rectangle((8, 8, min(image.width - 8, 820), 38),
                       fill=(0, 0, 0, 190))
        draw.text((14, 14), label, fill=(255, 255, 255, 255))
        path = output_dir / f"{record['name']}-reprojection.png"
        image.save(path)
        paths.append(path)
    return paths


def comparison_sheet(
        paths: list[Path], records: list[dict], output: Path) -> None:
    cells = {}
    for path, record in zip(paths, records):
        key = (int(round(record["nominal_pitch"])),
               int(round(record["nominal_yaw"])))
        image = Image.open(path).convert("RGB")
        width = 360
        height = int(round(image.height * width / image.width))
        cells[key] = image.resize((width, height), Image.Resampling.LANCZOS)
    cell_width = 360
    cell_height = max(image.height for image in cells.values())
    sheet = Image.new("RGB", (cell_width * 3, cell_height * 3), (30, 30, 30))
    for row, pitch in enumerate(ELEVATION_ORDER):
        for column, yaw in enumerate(AZIMUTH_ORDER):
            image = cells.get((pitch, yaw))
            if image is not None:
                sheet.paste(image, (column * cell_width, row * cell_height))
    sheet.save(output)


def structure_sheet(
        result: CameraFactorizationResult,
        output: Path) -> None:
    canvas = Image.new("RGB", (1200, 420), (28, 28, 32))
    draw = ImageDraw.Draw(canvas)
    projections = ((0, 1, "front X/Y"), (2, 1, "side Z/Y"), (0, 2, "top X/Z"))
    color = (80, 220, 255)
    for panel, (first, second, title) in enumerate(projections):
        values = result.points[:, (first, second)]
        low = values.min(axis=0)
        high = values.max(axis=0)
        center = (low + high) * 0.5
        extent = max(float(np.max(high - low)), 1.0e-6)
        x0 = panel * 400
        draw.text((x0 + 12, 10), title, fill=(255, 255, 255))
        for index, point in enumerate(values):
            x = x0 + 200 + (point[0] - center[0]) / extent * 330
            y = 220 - (point[1] - center[1]) / extent * 330
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            draw.text((x + 7, y - 6), str(index), fill=(210, 210, 210))
    canvas.save(output)


def write_ply(
        result: CameraFactorizationResult,
        output: Path) -> None:
    lines = [
        "ply", "format ascii 1.0", f"element vertex {len(result.points)}",
        "property float x", "property float y", "property float z",
        "property uchar red", "property uchar green", "property uchar blue",
        "end_header",
    ]
    for index, point in enumerate(result.points):
        color = (60, 220, 255)
        lines.append(
            f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
            f"{color[0]} {color[1]} {color[2]}")
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be in [0, 1]")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest, records = load_records(args.manifest)
    track_names = TRACK_SETS[args.track_set]
    cache = args.output_dir / "dwpose-keypoints.json"
    views = load_or_extract_keypoints(records, cache, args.reuse_keypoints)
    observations, weights, sizes = observation_arrays(
        records, views, args.confidence, track_names)
    reference_view = next(
        index for index, record in enumerate(records)
        if record["nominal_yaw"] == 0.0 and record["nominal_pitch"] == 0.0
    )
    initial = factor_scaled_orthographic(
        observations, weights=weights, reference_view=reference_view)
    initial = choose_depth_sign(initial, records)
    mean_height = float(np.mean([height for _width, height in sizes]))
    fitted = refine_scaled_orthographic(
        initial,
        observations,
        weights=weights,
        reference_view=reference_view,
        soft_l1_scale=args.soft_l1_pixels / mean_height,
        max_evaluations=args.max_evaluations,
    )
    predicted = pixel_predictions(fitted, sizes)
    observed = observed_pixels(observations, sizes)
    view_stats, overall_stats = residual_statistics(
        predicted, observed, weights)
    camera_records = []
    for index, (record, camera, stats) in enumerate(
            zip(records, fitted.cameras, view_stats)):
        yaw, world_pitch, roll = camera_yaw_pitch_roll(camera.rotation)
        elevation = -world_pitch
        camera_record = {
            "name": record["name"],
            "path": str(record["path"]),
            "nominal_yaw_degrees": record["nominal_yaw"],
            "nominal_pitch_degrees": record["nominal_pitch"],
            "fitted_yaw_degrees": yaw,
            "fitted_elevation_degrees": elevation,
            "world_to_camera_pitch_degrees": world_pitch,
            "fitted_roll_degrees": roll,
            "scale": camera.scale,
            "translation_normalized": camera.translation.tolist(),
            "world_to_camera_rotation": camera.rotation.tolist(),
            **stats,
        }
        camera_records.append(camera_record)
        print(json.dumps(camera_record), flush=True)
    overlays = draw_overlays(
        args.output_dir, records, observed, predicted, view_stats, fitted.cameras)
    comparison = args.output_dir / "reprojection-grid.png"
    comparison_sheet(overlays, records, comparison)
    structure = args.output_dir / "structure-orthographic.png"
    structure_sheet(fitted, structure)
    ply = args.output_dir / "structure.ply"
    write_ply(fitted, ply)
    singular = fitted.singular_values
    nominal_yaw = np.array([
        record["nominal_yaw"] for record in records], dtype=np.float64)
    nominal_elevation = np.array([
        record["nominal_pitch"] for record in records], dtype=np.float64)
    fitted_yaw = np.array([
        camera["fitted_yaw_degrees"] for camera in camera_records])
    fitted_elevation = np.array([
        camera["fitted_elevation_degrees"] for camera in camera_records])
    lattice = {
        "yaw_correlation": float(np.corrcoef(nominal_yaw, fitted_yaw)[0, 1]),
        "yaw_mean_absolute_error_degrees": float(np.mean(
            np.abs(nominal_yaw - fitted_yaw))),
        "elevation_correlation": float(np.corrcoef(
            nominal_elevation, fitted_elevation)[0, 1]),
        "elevation_mean_absolute_error_degrees": float(np.mean(
            np.abs(nominal_elevation - fitted_elevation))),
        "fitted_elevation_row_means": {
            str(value): float(np.mean(fitted_elevation[nominal_elevation == value]))
            for value in sorted(set(nominal_elevation))
        },
    }
    report = {
        "schema": "diffusion-editor.dwpose-model-free-camera-factorization",
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "source_profile": manifest.get("profile"),
        "model": "scaled-orthographic free points; no body topology",
        "projection": (
            "normalized_xy = scale * (world_to_camera_rotation[:2] @ xyz) "
            "+ translation; pixel_x = normalized_x * image_height + width/2; "
            "pixel_y = height/2 - normalized_y * image_height"
        ),
        "track_set": args.track_set,
        "track_names": track_names,
        "reference_view": records[reference_view]["name"],
        "confidence_threshold": args.confidence,
        "soft_l1_pixels": args.soft_l1_pixels,
        "singular_values": singular.tolist(),
        "rank4_to_rank3_ratio": (
            None if len(singular) < 4 else float(singular[3] / singular[2])
        ),
        "optimization": {
            "cost": fitted.cost,
            "optimality": fitted.optimality,
            "evaluations": fitted.evaluations,
        },
        "overall": overall_stats,
        "camera_lattice": lattice,
        "cameras": camera_records,
        "points": [
            {"name": name, "xyz": point.tolist()}
            for name, point in zip(track_names, fitted.points)
        ],
        "artifacts": {
            "keypoints": str(cache),
            "reprojection_grid": str(comparison),
            "structure_sheet": str(structure),
            "structure_ply": str(ply),
        },
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "overall": overall_stats,
        "camera_lattice": lattice,
        "rank4_to_rank3_ratio": report["rank4_to_rank3_ratio"],
        "report": str(report_path),
        "reprojection_grid": str(comparison),
        "structure": str(structure),
        "ply": str(ply),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
