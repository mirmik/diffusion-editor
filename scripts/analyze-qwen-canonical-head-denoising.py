#!/usr/bin/env python3
"""Analyze saved generated canonical point maps without rerunning Qwen."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _reference_surface(dataset_root: Path, manifest: dict) -> np.ndarray:
    points = []
    for view in manifest["views"]:
        with np.load(dataset_root / view["geometry"]) as geometry:
            mask = np.asarray(geometry["mask"], dtype=bool)
            points.append(
                np.asarray(geometry["canonical_xyz"], dtype=np.float32)[mask]
            )
    return np.concatenate(points)


def _points_and_pixels(prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xyz = prediction[:3].transpose(1, 2, 0)
    mask = prediction[3] > 0.0
    rows, columns = np.nonzero(mask)
    height, width = mask.shape
    pixels = np.column_stack(
        (
            (columns.astype(np.float64) + 0.5) / width,
            (rows.astype(np.float64) + 0.5) / height,
        )
    )
    return xyz[mask].astype(np.float64), pixels


def _image_foreground_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as source:
        rgb = np.asarray(
            source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        )
    # The controlled canonical experiment uses a nominally black vacuum background.
    # A low threshold includes antialiased silhouette pixels without treating
    # compression-level black variation as foreground.
    return np.max(rgb, axis=2) > 12


def _projective_reprojection(points: np.ndarray, pixels: np.ndarray) -> dict:
    if len(points) < 12:
        return {"valid": False}
    center = points.mean(axis=0)
    scale = np.maximum(points.std(axis=0), 1.0e-4)
    normalized = (points - center) / scale
    homogeneous = np.column_stack((normalized, np.ones(len(normalized))))
    if len(points) > 1024:
        fit_indices = np.linspace(0, len(points) - 1, 1024, dtype=np.int64)
    else:
        fit_indices = np.arange(len(points))
    fit_points = homogeneous[fit_indices]
    fit_pixels = pixels[fit_indices]
    zeros = np.zeros_like(fit_points)
    matrix = np.concatenate(
        (
            np.concatenate(
                (fit_points, zeros, -fit_pixels[:, :1] * fit_points), axis=1
            ),
            np.concatenate(
                (zeros, fit_points, -fit_pixels[:, 1:] * fit_points), axis=1
            ),
        ),
        axis=0,
    )
    _u, _singular, vh = np.linalg.svd(matrix, full_matrices=False)
    camera = vh[-1].reshape(3, 4)
    projected = homogeneous @ camera.T
    denominator = projected[:, 2]
    valid = np.abs(denominator) > 1.0e-8
    if valid.mean() < 0.99:
        return {"valid": False}
    image = projected[valid, :2] / denominator[valid, None]
    residual = np.linalg.norm(image - pixels[valid], axis=1)
    return {
        "valid": True,
        "median_image_fraction": float(np.median(residual)),
        "p90_image_fraction": float(np.quantile(residual, 0.90)),
        "p95_image_fraction": float(np.quantile(residual, 0.95)),
    }


def _write_generated_contact_sheet(capture_root: Path, views: list[dict]) -> Path:
    cell = 256
    label_height = 24
    columns = 8
    rows = (len(views) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + label_height)), "black")
    draw = ImageDraw.Draw(sheet)
    for index, view in enumerate(views):
        with Image.open(capture_root / view["image"]) as source:
            image = source.convert("RGB")
            image.thumbnail((cell, cell), Image.Resampling.LANCZOS)
        column = index % columns
        row = index // columns
        x = column * cell + (cell - image.width) // 2
        y = row * (cell + label_height) + (cell - image.height) // 2
        sheet.paste(image, (x, y))
        draw.text(
            (column * cell + 4, row * (cell + label_height) + cell + 4),
            view["id"],
            fill="white",
        )
    path = capture_root / "generated-contact-sheet.jpg"
    sheet.save(path, quality=92)
    return path


def _write_xyz_contact_sheet(
    capture_root: Path,
    views: list[dict],
    label: str,
) -> Path:
    cell = 128
    label_height = 20
    columns = 8
    rows = (len(views) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * cell, rows * (cell + label_height)), "black")
    draw = ImageDraw.Draw(sheet)
    for index, view in enumerate(views):
        path = capture_root / f"{view['id']}-{label}-xyz.png"
        with Image.open(path) as source:
            image = source.convert("RGB").resize(
                (cell, cell), Image.Resampling.NEAREST
            )
        column = index % columns
        row = index // columns
        x = column * cell
        y = row * (cell + label_height)
        sheet.paste(image, (x, y))
        draw.text((x + 3, y + cell + 2), view["id"], fill="white")
    path = capture_root / f"{label}-xyz-contact-sheet.png"
    sheet.save(path)
    return path


def main() -> int:
    args = _arguments()
    capture_root = args.capture_dir.resolve()
    manifest = _load_json(capture_root / "manifest.json")
    dataset_root = Path(manifest["dataset"])
    dataset_manifest = _load_json(dataset_root / "manifest.json")
    reference = _reference_surface(dataset_root, dataset_manifest)
    reference_tree = cKDTree(reference)
    steps = int(manifest["steps"])
    prediction_sets = [
        {
            "name": f"step_{step}",
            "npz_key": f"prediction_step_{step}",
            "diagnostic_label": f"step{step}",
        }
        for step in range(steps)
    ]
    prediction_sets.extend(
        {
            "name": f"fusion_{name}",
            "npz_key": f"prediction_fusion_{name}",
            "diagnostic_label": f"stepfusion-{name}",
        }
        for name in manifest.get("fusions", [])
    )
    per_view = []
    aggregate = {item["name"]: [] for item in prediction_sets}
    for view in manifest["views"]:
        image_mask = None
        with np.load(capture_root / view["predictions"]) as predictions:
            reports = {}
            for prediction_set in prediction_sets:
                prediction = np.asarray(predictions[prediction_set["npz_key"]])
                points, pixels = _points_and_pixels(prediction)
                predicted_mask = prediction[3] > 0.0
                if image_mask is None:
                    image_mask = _image_foreground_mask(
                        capture_root / view["image"],
                        (predicted_mask.shape[1], predicted_mask.shape[0]),
                    )
                mask_union = np.logical_or(predicted_mask, image_mask).sum()
                name = prediction_set["name"]
                aggregate[name].append(points)
                reports[name] = {
                    "foreground_points": int(len(points)),
                    "image_foreground_iou": float(
                        np.logical_and(predicted_mask, image_mask).sum()
                        / max(mask_union, 1)
                    ),
                    "projective_reprojection": _projective_reprojection(
                        points, pixels
                    ),
                }
        per_view.append({"id": view["id"], "predictions": reports})

    prediction_summaries = []
    for prediction_set in prediction_sets:
        name = prediction_set["name"]
        cloud = np.concatenate(aggregate[name])
        cloud_tree = cKDTree(cloud)
        point_to_surface = reference_tree.query(cloud, workers=-1)[0]
        surface_to_cloud = cloud_tree.query(reference, workers=-1)[0]
        projective = [
            item["predictions"][name]["projective_reprojection"]
            for item in per_view
        ]
        valid_projective = [item for item in projective if item["valid"]]
        image_ious = [
            item["predictions"][name]["image_foreground_iou"]
            for item in per_view
        ]
        prediction_summaries.append(
            {
                "name": name,
                "point_count": int(len(cloud)),
                "aggregate_point_to_surface_median": float(
                    np.median(point_to_surface)
                ),
                "aggregate_point_to_surface_p95": float(
                    np.quantile(point_to_surface, 0.95)
                ),
                "aggregate_points_within_5_percent": float(
                    np.mean(point_to_surface <= 0.05)
                ),
                "aggregate_surface_coverage_median": float(
                    np.median(surface_to_cloud)
                ),
                "aggregate_surface_coverage_p90": float(
                    np.quantile(surface_to_cloud, 0.90)
                ),
                "aggregate_surface_coverage_p95": float(
                    np.quantile(surface_to_cloud, 0.95)
                ),
                "projective_valid_views": len(valid_projective),
                "projective_reprojection_median_mean": (
                    float(
                        np.mean(
                            [item["median_image_fraction"] for item in valid_projective]
                        )
                    )
                    if valid_projective
                    else None
                ),
                "projective_reprojection_p95_max": (
                    float(max(item["p95_image_fraction"] for item in valid_projective))
                    if valid_projective
                    else None
                ),
                "image_foreground_iou_mean": float(np.mean(image_ious)),
                "image_foreground_iou_min": float(min(image_ious)),
            }
        )
    step_summaries = [
        {"step": step, **prediction_summaries[step]}
        for step in range(steps)
    ]
    fusion_summaries = {
        summary["name"].removeprefix("fusion_"): summary
        for summary in prediction_summaries[steps:]
    }
    report = {
        "schema": "diffusion-editor.qwen-canonical-head-denoising-analysis",
        "schema_version": 1,
        "capture": str(capture_root),
        "metric_note": (
            "Projective reprojection fits an unconstrained 3x4 camera per view. "
            "Aggregate coverage compares the union of generated point maps with "
            "the union of exact Blender visible surfaces. Image foreground IoU "
            "uses max(RGB) > 12 as a heuristic mask for the controlled black "
            "background experiment."
        ),
        "step_summaries": step_summaries,
        "fusion_summaries": fusion_summaries,
        "prediction_summaries": prediction_summaries,
        "per_view": per_view,
        "generated_contact_sheet": str(
            _write_generated_contact_sheet(capture_root, manifest["views"])
        ),
        "xyz_contact_sheets": [
            str(
                _write_xyz_contact_sheet(
                    capture_root,
                    manifest["views"],
                    prediction_set["diagnostic_label"],
                )
            )
            for prediction_set in prediction_sets
        ],
    }
    output = args.output or (capture_root / "analysis.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(prediction_summaries, indent=2))
    print(f"Complete: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
