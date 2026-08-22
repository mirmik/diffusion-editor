#!/usr/bin/env python3
"""Evaluate dense correspondences from captured Qwen spatial features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

import cv2
import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--da3-prediction", type=Path)
    parser.add_argument("--min-similarity", type=float, default=0.50)
    parser.add_argument("--ransac-threshold", type=float, default=2.0)
    parser.add_argument("--max-da3-error-m", type=float, default=0.03)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--centering",
        choices=("raw", "image", "position"),
        default="raw",
        help=(
            "Descriptor centering: per-image removes the image-wide common "
            "direction; per-position removes the component shared by all views."
        ),
    )
    return parser.parse_args()


def _grid_points(indices: np.ndarray, grid: tuple[int, int], size: tuple[int, int]) -> np.ndarray:
    grid_width, grid_height = grid
    width, height = size
    y, x = np.divmod(indices, grid_width)
    return np.stack(
        (
            (x.astype(np.float32) + 0.5) * width / grid_width - 0.5,
            (y.astype(np.float32) + 0.5) * height / grid_height - 0.5,
        ),
        axis=1,
    )


def _mutual_matches(
    features0: np.ndarray,
    features1: np.ndarray,
    mask0: np.ndarray,
    mask1: np.ndarray,
    min_similarity: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids0 = np.flatnonzero(mask0.reshape(-1))
    ids1 = np.flatnonzero(mask1.reshape(-1))
    tensor0 = torch.from_numpy(features0.reshape(-1, features0.shape[-1])[ids0].astype(np.float32)).to(device)
    tensor1 = torch.from_numpy(features1.reshape(-1, features1.shape[-1])[ids1].astype(np.float32)).to(device)
    similarity = tensor0 @ tensor1.T
    best1 = similarity.argmax(dim=1)
    best0 = similarity.argmax(dim=0)
    source = torch.arange(len(ids0), device=device)
    mutual = best0[best1] == source
    scores = similarity[source, best1]
    keep = mutual & (scores >= min_similarity)
    selected0 = source[keep].cpu().numpy()
    selected1 = best1[keep].cpu().numpy()
    selected_scores = scores[keep].cpu().numpy()
    return ids0[selected0], ids1[selected1], selected_scores


def _normalize_features(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32)
    norms = np.linalg.norm(features, axis=-1, keepdims=True)
    return (features / np.maximum(norms, 1e-8)).astype(np.float16)


def main() -> int:
    args = _arguments()
    manifest_path = args.capture_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"missing capture manifest: {manifest_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = manifest["views"]
    if len(views) < 3:
        raise SystemExit("at least three captured views are required")
    feature_keys = set(views[0]["feature_keys"])
    for view in views[1:]:
        feature_keys.intersection_update(view["feature_keys"])
    feature_keys = sorted(feature_keys)
    grid = tuple(int(value) for value in manifest["feature_grid"])
    size = tuple(int(value) for value in manifest["output_size"])

    helper_path = Path(__file__).with_name("experiment-multiview-keypoints.py")
    helpers = runpy.run_path(str(helper_path))
    subject_mask = helpers["_subject_mask"]
    estimate_pair = helpers["_estimate_pair"]
    depth_geometry_report = helpers["_depth_geometry_report"]
    draw_pair = helpers["_draw_pair"]

    images = [
        np.asarray(Image.open(view["image"]).convert("RGB")) for view in views
    ]
    masks = []
    for image in images:
        full_mask = subject_mask(image, 3).astype(np.uint8)
        masks.append(
            cv2.resize(full_mask, grid, interpolation=cv2.INTER_NEAREST).astype(bool)
        )
    archives = [np.load(view["features"]) for view in views]
    prepared_features: dict[str, list[np.ndarray]] = {}
    for feature_key in feature_keys:
        maps = [archive[feature_key].astype(np.float32) for archive in archives]
        if args.centering == "image":
            maps = [feature_map - feature_map.mean(axis=(0, 1), keepdims=True)
                    for feature_map in maps]
        elif args.centering == "position":
            position_mean = np.mean(np.stack(maps, axis=0), axis=0)
            maps = [feature_map - position_mean for feature_map in maps]
        prepared_features[feature_key] = [
            _normalize_features(feature_map) for feature_map in maps
        ]
    da3 = np.load(args.da3_prediction) if args.da3_prediction else None
    device = torch.device(args.device)

    width, height = size
    focal = (0.5 * width) / np.tan(np.radians(34.0) * 0.5)
    diagnostic_intrinsics = np.array(
        ((focal, 0.0, (width - 1) * 0.5),
         (0.0, focal, (height - 1) * 0.5),
         (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )

    reports = []
    cached_matches: dict[tuple[str, int], tuple[np.ndarray, ...]] = {}
    for feature_key in feature_keys:
        pair_reports = []
        for pair_index in range(len(views) - 1):
            ids0, ids1, scores = _mutual_matches(
                prepared_features[feature_key][pair_index],
                prepared_features[feature_key][pair_index + 1],
                masks[pair_index],
                masks[pair_index + 1],
                args.min_similarity,
                device,
            )
            points0 = _grid_points(ids0, grid, size)
            points1 = _grid_points(ids1, grid, size)
            displacements = np.linalg.norm(points1 - points0, axis=1)
            try:
                inliers, geometry = estimate_pair(
                    points0, points1, args.ransac_threshold, diagnostic_intrinsics
                )
            except cv2.error:
                # Some feature maps collapse to too few distinct image points;
                # OpenCV's USAC implementation asserts instead of reporting
                # that it could not instantiate a model.
                inliers = np.zeros(len(points0), dtype=bool)
                geometry = {
                    "fundamental_matrix": None,
                    "median_sampson_error_px": None,
                    "estimated_rotation_degrees": None,
                    "estimated_translation_direction": None,
                }
            depth_report = {}
            depth_errors = np.full(len(points0), np.nan, dtype=np.float64)
            azimuth0 = int(views[pair_index]["azimuth_degrees"])
            azimuth1 = int(views[pair_index + 1]["azimuth_degrees"])
            # The first three regenerated images reproduce the original ring
            # closely enough to share its saved DA3 prediction. The generated
            # 180-degree view is intentionally excluded: it differs from the
            # hand-edited Back anchor used by that prediction.
            if da3 is not None and azimuth1 <= 135:
                index0, index1 = azimuth0 // 45, azimuth1 // 45
                depth_report, depth_errors = depth_geometry_report(
                    points0,
                    points1,
                    inliers,
                    size,
                    size,
                    da3["depth"][index0],
                    da3["depth"][index1],
                    da3["intrinsics"][index0],
                    da3["intrinsics"][index1],
                    da3["extrinsics"][index0],
                    da3["extrinsics"][index1],
                )
            has_depth_check = bool(np.isfinite(depth_errors).any())
            strict = (
                inliers
                & np.isfinite(depth_errors)
                & (depth_errors <= args.max_da3_error_m)
                if has_depth_check
                else np.zeros_like(inliers)
            )
            cached_matches[(feature_key, pair_index)] = (
                points0, points1, inliers, strict, scores
            )
            pair_reports.append({
                "pair": [azimuth0, azimuth1],
                "match_count": int(len(points0)),
                "similarity_median": (
                    float(np.median(scores)) if len(scores) else None
                ),
                "displacement_median_px": (
                    float(np.median(displacements)) if len(displacements) else None
                ),
                "displacement_p90_px": (
                    float(np.percentile(displacements, 90))
                    if len(displacements)
                    else None
                ),
                "position_locked_ratio": (
                    float(np.mean(displacements < 1.0)) if len(displacements) else 0.0
                ),
                "magsac_inlier_count": int(inliers.sum()),
                "magsac_inlier_ratio": (
                    float(inliers.mean()) if len(inliers) else 0.0
                ),
                "strict_3d_count": int(strict.sum()),
                **geometry,
                **depth_report,
            })
        scored_pairs = pair_reports[:2]
        reports.append({
            "feature_key": feature_key,
            "score_strict_3d": int(sum(x["strict_3d_count"] for x in scored_pairs)),
            "score_magsac": int(sum(x["magsac_inlier_count"] for x in scored_pairs)),
            "position_locked_ratio": float(
                np.mean([x["position_locked_ratio"] for x in scored_pairs])
            ),
            "pairs": pair_reports,
        })
        print(
            feature_key,
            "strict",
            reports[-1]["score_strict_3d"],
            "MAGSAC",
            reports[-1]["score_magsac"],
            flush=True,
        )

    reports.sort(
        key=lambda item: (item["score_strict_3d"], item["score_magsac"]),
        reverse=True,
    )
    raw_best = reports[0]
    nontrivial = [item for item in reports if item["position_locked_ratio"] < 0.25]
    best = nontrivial[0] if nontrivial else raw_best
    visualizations = []
    for pair_index, pair_report in enumerate(best["pairs"]):
        points0, points1, inliers, strict, _scores = cached_matches[
            (best["feature_key"], pair_index)
        ]
        visualization = draw_pair(
            images[pair_index],
            images[pair_index + 1],
            points0,
            points1,
            inliers,
            strict,
            f"Qwen {best['feature_key']} · {pair_report['pair'][0]:03d}→{pair_report['pair'][1]:03d}",
            (
                f"MNN {len(points0)} · MAGSAC {int(inliers.sum())} · "
                f"strict {int(strict.sum())}"
            ),
        )
        path = args.output_dir / f"best-pair-{pair_index}.png"
        visualization.save(path)
        thumbnail = visualization.copy()
        thumbnail.thumbnail((960, 570))
        visualizations.append(thumbnail)
    sheet = Image.new("RGB", (960, 570 * len(visualizations)), (8, 8, 8))
    for index, visualization in enumerate(visualizations):
        sheet.paste(
            visualization,
            ((960 - visualization.width) // 2, index * 570),
        )
    sheet.save(args.output_dir / "best-contact.png")

    summary = {
        "capture_manifest": str(manifest_path.resolve()),
        "centering": args.centering,
        "min_similarity": args.min_similarity,
        "ransac_threshold_px": args.ransac_threshold,
        "max_da3_error_m": args.max_da3_error_m,
        "best_feature_key": best["feature_key"],
        "raw_best_feature_key": raw_best["feature_key"],
        "position_lock_rejection_threshold": 0.25,
        "reports": reports,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Best: {best['feature_key']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
