#!/usr/bin/env python3
"""Fuse saved canonical-head predictions into diagnostic PLY point clouds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


PREDICTION_PREFIX = "prediction_"
PROJECTIONS = (
    ("front", 0, 1, 2, 1.0, 1.0, 1.0),
    ("back", 0, 1, 2, -1.0, 1.0, -1.0),
    ("right", 2, 1, 0, -1.0, 1.0, 1.0),
    ("top", 0, 2, 1, 1.0, -1.0, 1.0),
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument(
        "--prediction",
        default="fusion_confidence_all",
        help=(
            "Saved prediction suffix, for example step_1 or "
            "fusion_confidence_all."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--confidence-keep",
        type=float,
        default=1.0,
        help="Per view, retain this lowest-predicted-error foreground fraction.",
    )
    parser.add_argument("--voxel-size", type=float, default=0.008)
    parser.add_argument("--projection-resolution", type=int, default=512)
    parser.add_argument("--point-radius", type=int, default=2)
    args = parser.parse_args()
    if not 0.0 < args.confidence_keep <= 1.0:
        parser.error("--confidence-keep must be in (0, 1]")
    if args.voxel_size <= 0.0:
        parser.error("--voxel-size must be positive")
    if args.projection_resolution <= 0 or args.point_radius <= 0:
        parser.error("projection resolution and point radius must be positive")
    return args


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rgb_grid(path: Path, width: int, height: int) -> np.ndarray:
    with Image.open(path) as source:
        image = source.convert("RGB").resize(
            (width, height), Image.Resampling.BILINEAR
        )
    return np.asarray(image, dtype=np.uint8)


def _canonical_colors(points: np.ndarray) -> np.ndarray:
    encoded = np.clip(points + 0.5, 0.0, 1.0)
    return np.round(encoded * 255.0).astype(np.uint8)


def _prediction_points(
    root: Path,
    manifest: dict,
    key: str,
    confidence_keep: float,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[dict],
    tuple[int, int],
]:
    points = []
    colors = []
    log_errors = []
    view_indices = []
    reports = []
    grid_size = None
    for view_index, view in enumerate(manifest["views"]):
        with np.load(root / view["predictions"]) as source:
            if key not in source:
                raise RuntimeError(
                    f"{view['id']} does not contain {key}; available: {source.files}"
                )
            prediction = np.asarray(source[key], dtype=np.float32)
        if prediction.ndim != 3 or prediction.shape[0] != 5:
            raise RuntimeError(f"{view['id']} has invalid prediction shape")
        height, width = prediction.shape[1:]
        if grid_size is None:
            grid_size = (height, width)
        elif grid_size != (height, width):
            raise RuntimeError("prediction views use inconsistent grid sizes")
        xyz = prediction[:3].transpose(1, 2, 0)
        mask = prediction[3] > 0.0
        finite = np.all(np.isfinite(xyz), axis=2) & np.isfinite(prediction[4])
        mask &= finite
        initial_count = int(mask.sum())
        threshold = None
        if initial_count and confidence_keep < 1.0:
            threshold = float(
                np.quantile(prediction[4][mask], confidence_keep)
            )
            mask &= prediction[4] <= threshold
        rgb = _rgb_grid(root / view["image"], width, height)
        points.append(xyz[mask])
        colors.append(rgb[mask])
        log_errors.append(prediction[4][mask])
        view_indices.append(np.full(int(mask.sum()), view_index, dtype=np.int16))
        reports.append(
            {
                "id": view["id"],
                "foreground_points": initial_count,
                "retained_points": int(mask.sum()),
                "log_error_threshold": threshold,
            }
        )
    if not points or not sum(len(item) for item in points):
        raise RuntimeError("prediction has no retained foreground points")
    return (
        np.concatenate(points),
        np.concatenate(colors),
        np.concatenate(log_errors),
        np.concatenate(view_indices),
        reports,
        grid_size,
    )


def _reference_points(
    dataset_root: Path,
    manifest: dict,
    selected_view_ids: set[str],
    grid_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = []
    colors = []
    view_indices = []
    selected_views = [
        view for view in manifest["views"] if view["id"] in selected_view_ids
    ]
    missing = selected_view_ids - {view["id"] for view in selected_views}
    if missing:
        raise RuntimeError(f"reference dataset is missing views: {sorted(missing)}")
    for view_index, view in enumerate(selected_views):
        with np.load(dataset_root / view["geometry"]) as source:
            mask = np.asarray(source["mask"], dtype=bool)
            xyz = np.asarray(source["canonical_xyz"], dtype=np.float32)
        source_height, source_width = mask.shape
        height, width = grid_size
        rows = np.floor(
            np.arange(height, dtype=np.float64) * source_height / height
        ).astype(np.intp)
        columns = np.floor(
            np.arange(width, dtype=np.float64) * source_width / width
        ).astype(np.intp)
        mask = mask[np.ix_(rows, columns)]
        xyz = xyz[rows[:, None], columns[None, :]]
        rgb = _rgb_grid(dataset_root / view["rgb"], width, height)
        points.append(xyz[mask])
        colors.append(rgb[mask])
        view_indices.append(np.full(int(mask.sum()), view_index, dtype=np.int16))
    return np.concatenate(points), np.concatenate(colors), np.concatenate(view_indices)


def _voxelize(
    points: np.ndarray,
    colors: np.ndarray,
    view_indices: np.ndarray,
    voxel_size: float,
    log_errors: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    voxel = np.floor(points / voxel_size).astype(np.int32)
    _keys, inverse, counts = np.unique(
        voxel, axis=0, return_inverse=True, return_counts=True
    )
    voxel_count = len(counts)

    def average(values: np.ndarray) -> np.ndarray:
        if values.ndim == 1:
            return np.bincount(inverse, weights=values) / counts
        return np.stack(
            [np.bincount(inverse, weights=values[:, axis]) / counts
             for axis in range(values.shape[1])],
            axis=1,
        )

    pairs = np.unique(np.column_stack((inverse, view_indices)), axis=0)
    observations = np.bincount(pairs[:, 0], minlength=voxel_count)
    result = {
        "points": average(points).astype(np.float32),
        "colors": np.round(average(colors)).clip(0, 255).astype(np.uint8),
        "samples": counts.astype(np.int32),
        "observations": observations.astype(np.int16),
    }
    if log_errors is not None:
        result["log_errors"] = average(log_errors).astype(np.float32)
    return result


def _write_ply(path: Path, cloud: dict[str, np.ndarray]) -> None:
    points = cloud["points"]
    colors = cloud["colors"]
    log_errors = cloud.get("log_errors")
    with path.open("w", encoding="ascii", newline="\n") as output:
        output.write("ply\nformat ascii 1.0\n")
        output.write(f"element vertex {len(points)}\n")
        output.write("property float x\nproperty float y\nproperty float z\n")
        output.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        output.write("property ushort source_views\nproperty uint source_samples\n")
        if log_errors is not None:
            output.write("property float predicted_log_error\n")
        output.write("end_header\n")
        for index, (point, color) in enumerate(zip(points, colors)):
            fields = [
                f"{point[0]:.8g}", f"{point[1]:.8g}", f"{point[2]:.8g}",
                str(int(color[0])), str(int(color[1])), str(int(color[2])),
                str(int(cloud["observations"][index])),
                str(int(cloud["samples"][index])),
            ]
            if log_errors is not None:
                fields.append(f"{log_errors[index]:.8g}")
            output.write(" ".join(fields) + "\n")


def _projection_limits(points: np.ndarray) -> np.ndarray:
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    span = maximum - minimum
    margin = np.maximum(span * 0.06, 0.015)
    return np.stack((minimum - margin, maximum + margin), axis=1)


def _render_projection(
    points: np.ndarray,
    colors: np.ndarray,
    spec: tuple,
    limits: np.ndarray,
    resolution: int,
    radius: int,
) -> Image.Image:
    _name, horizontal, vertical, depth_axis, h_sign, v_sign, d_sign = spec
    h_min, h_max = limits[horizontal]
    v_min, v_max = limits[vertical]
    h_values = points[:, horizontal] * h_sign
    v_values = points[:, vertical] * v_sign
    if h_sign < 0:
        h_min, h_max = -h_max, -h_min
    if v_sign < 0:
        v_min, v_max = -v_max, -v_min
    x = np.round((h_values - h_min) / (h_max - h_min) * (resolution - 1))
    y = np.round((v_max - v_values) / (v_max - v_min) * (resolution - 1))
    x = x.astype(np.int32)
    y = y.astype(np.int32)
    order = np.argsort(points[:, depth_axis] * d_sign)
    canvas = np.zeros((resolution, resolution, 3), dtype=np.uint8)
    rr = radius * radius
    offsets = [
        (dy, dx)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
        if dx * dx + dy * dy <= rr
    ]
    for index in order:
        for dy, dx in offsets:
            px = x[index] + dx
            py = y[index] + dy
            if 0 <= px < resolution and 0 <= py < resolution:
                canvas[py, px] = colors[index]
    return Image.fromarray(canvas, "RGB")


def _projection_sheet(
    points: np.ndarray,
    colors: np.ndarray,
    limits: np.ndarray,
    resolution: int,
    radius: int,
    title: str,
) -> Image.Image:
    label_height = 28
    sheet = Image.new("RGB", (2 * resolution, 2 * (resolution + label_height)))
    draw = ImageDraw.Draw(sheet)
    for index, spec in enumerate(PROJECTIONS):
        column = index % 2
        row = index // 2
        x = column * resolution
        y = row * (resolution + label_height)
        draw.text((x + 6, y + 6), f"{title}: {spec[0]}", fill=(235, 235, 235))
        image = _render_projection(
            points, colors, spec, limits, resolution, radius
        )
        sheet.paste(image, (x, y + label_height))
    return sheet


def _cloud_summary(cloud: dict[str, np.ndarray]) -> dict:
    points = cloud["points"]
    centered = points - points.mean(axis=0)
    covariance = centered.T @ centered / max(len(points) - 1, 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    observations = cloud["observations"]
    return {
        "voxel_points": len(points),
        "bounds": {
            "minimum": points.min(axis=0).tolist(),
            "maximum": points.max(axis=0).tolist(),
            "span": np.ptp(points, axis=0).tolist(),
        },
        "pca_variance": eigenvalues.tolist(),
        "source_views": {
            "maximum": int(observations.max()),
            "mean": float(observations.mean()),
            "fraction_at_least_2": float(np.mean(observations >= 2)),
            "fraction_at_least_3": float(np.mean(observations >= 3)),
        },
    }


def main() -> int:
    args = _arguments()
    root = args.capture_dir.resolve()
    manifest = _load_json(root / "manifest.json")
    if manifest.get("schema") != "diffusion-editor.qwen-canonical-head-denoising":
        raise SystemExit("unsupported capture manifest")
    prediction_key = (
        args.prediction
        if args.prediction.startswith(PREDICTION_PREFIX)
        else PREDICTION_PREFIX + args.prediction
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_points, raw_colors, log_errors, view_indices, view_reports, grid_size = (
        _prediction_points(
            root, manifest, prediction_key, args.confidence_keep
        )
    )
    prediction = _voxelize(
        raw_points,
        raw_colors,
        view_indices,
        args.voxel_size,
        log_errors,
    )

    dataset_root = Path(manifest["dataset"])
    dataset_manifest = _load_json(dataset_root / "manifest.json")
    reference_raw, reference_colors, reference_views = _reference_points(
        dataset_root,
        dataset_manifest,
        {view["id"] for view in manifest["views"]},
        grid_size,
    )
    reference = _voxelize(
        reference_raw,
        reference_colors,
        reference_views,
        args.voxel_size,
    )
    combined = np.concatenate((prediction["points"], reference["points"]))
    limits = _projection_limits(combined)

    _write_ply(args.output_dir / "prediction-rgb.ply", prediction)
    canonical_prediction = {**prediction, "colors": _canonical_colors(prediction["points"])}
    _write_ply(args.output_dir / "prediction-canonical.ply", canonical_prediction)
    canonical_reference = {**reference, "colors": _canonical_colors(reference["points"])}
    _write_ply(args.output_dir / "reference-canonical.ply", canonical_reference)

    view_directory = args.output_dir / "views"
    view_directory.mkdir(exist_ok=True)
    view_clouds = []
    for view_index, report in enumerate(view_reports):
        selected = view_indices == view_index
        per_view = _voxelize(
            raw_points[selected],
            raw_colors[selected],
            np.zeros(int(selected.sum()), dtype=np.int16),
            args.voxel_size,
            log_errors[selected],
        )
        view_id = report["id"].replace("/", "_")
        rgb_name = f"views/{view_id}-rgb.ply"
        canonical_name = f"views/{view_id}-canonical.ply"
        _write_ply(args.output_dir / rgb_name, per_view)
        _write_ply(
            args.output_dir / canonical_name,
            {**per_view, "colors": _canonical_colors(per_view["points"])},
        )
        view_clouds.append(
            {
                "id": report["id"],
                "raw_points": int(selected.sum()),
                "voxel_points": len(per_view["points"]),
                "rgb_ply": rgb_name,
                "canonical_ply": canonical_name,
            }
        )

    for name, cloud, colors in (
        ("prediction-rgb", prediction, prediction["colors"]),
        ("prediction-canonical", prediction, canonical_prediction["colors"]),
        ("reference-canonical", reference, canonical_reference["colors"]),
    ):
        sheet = _projection_sheet(
            cloud["points"],
            colors,
            limits,
            args.projection_resolution,
            args.point_radius,
            name,
        )
        sheet.save(args.output_dir / f"{name}-projections.png")

    summary = {
        "schema": "diffusion-editor.canonical-head-pointcloud",
        "schema_version": 1,
        "capture": str(root),
        "dataset": str(dataset_root),
        "prediction": prediction_key,
        "confidence_keep": args.confidence_keep,
        "voxel_size": args.voxel_size,
        "point_grid": list(grid_size),
        "raw_prediction_points": len(raw_points),
        "raw_reference_points": len(reference_raw),
        "view_reports": view_reports,
        "view_clouds": view_clouds,
        "prediction_summary": _cloud_summary(prediction),
        "reference_summary": _cloud_summary(reference),
        "predicted_log_error_quantiles": {
            str(quantile): float(np.quantile(log_errors, quantile))
            for quantile in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "artifacts": {
            "prediction_rgb_ply": "prediction-rgb.ply",
            "prediction_canonical_ply": "prediction-canonical.ply",
            "reference_canonical_ply": "reference-canonical.ply",
            "prediction_rgb_projections": "prediction-rgb-projections.png",
            "prediction_canonical_projections": "prediction-canonical-projections.png",
            "reference_canonical_projections": "reference-canonical-projections.png",
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
