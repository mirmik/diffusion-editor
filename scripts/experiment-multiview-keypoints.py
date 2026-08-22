#!/usr/bin/env python3
"""Measure adjacent-view feature consistency for a generated object ring."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
import torch

from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-keypoints", type=int, default=4096)
    parser.add_argument("--ransac-threshold", type=float, default=1.5)
    parser.add_argument("--min-lightglue-score", type=float, default=0.30)
    parser.add_argument("--max-da3-error-m", type=float, default=0.03)
    parser.add_argument("--mask-erosion", type=int, default=3)
    parser.add_argument("--horizontal-fov-degrees", type=float, default=34.0)
    parser.add_argument(
        "--da3-prediction",
        type=Path,
        help=(
            "Optional prediction.npz from experiment-da3-multiview.py. "
            "Matched pixels are lifted through its float depth and cameras "
            "to measure cross-view 3D disagreement."
        ),
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _subject_mask(rgb: np.ndarray, erosion: int) -> np.ndarray:
    values = rgb.astype(np.int16)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    green_screen = (
        (green - red >= 12)
        & (green - blue >= 12)
        & (green * 100 >= red * 112)
        & (green * 100 >= blue * 112)
    )
    labels, count = ndimage.label(~green_screen)
    if count == 0:
        return np.zeros(green_screen.shape, dtype=bool)

    height, width = green_screen.shape
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
    mask = np.asarray(labels == best_label, dtype=np.uint8)
    if erosion > 0:
        kernel = np.ones((erosion * 2 + 1, erosion * 2 + 1), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask.astype(bool)


def _inside(mask: np.ndarray, points: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    xy = np.rint(points).astype(np.int64)
    x = np.clip(xy[:, 0], 0, width - 1)
    y = np.clip(xy[:, 1], 0, height - 1)
    return mask[y, x]


def _sampson_errors(
    matrix: np.ndarray, points0: np.ndarray, points1: np.ndarray
) -> np.ndarray:
    ones = np.ones((len(points0), 1), dtype=np.float64)
    x0 = np.concatenate((points0, ones), axis=1)
    x1 = np.concatenate((points1, ones), axis=1)
    fx0 = x0 @ matrix.T
    ftx1 = x1 @ matrix
    numerator = np.sum(x1 * fx0, axis=1) ** 2
    denominator = (
        fx0[:, 0] ** 2
        + fx0[:, 1] ** 2
        + ftx1[:, 0] ** 2
        + ftx1[:, 1] ** 2
    )
    return np.sqrt(numerator / np.maximum(denominator, 1e-12))


def _estimate_pair(
    points0: np.ndarray,
    points1: np.ndarray,
    threshold: float,
    intrinsics: np.ndarray,
) -> tuple[np.ndarray, dict[str, object]]:
    if len(points0) < 8:
        return np.zeros(len(points0), dtype=bool), {
            "fundamental_matrix": None,
            "median_sampson_error_px": None,
            "estimated_rotation_degrees": None,
            "estimated_translation_direction": None,
        }
    fundamental, raw_inliers = cv2.findFundamentalMat(
        points0,
        points1,
        cv2.USAC_MAGSAC,
        threshold,
        0.999,
        20_000,
    )
    if fundamental is None or fundamental.shape != (3, 3):
        inliers = np.zeros(len(points0), dtype=bool)
        return inliers, {
            "fundamental_matrix": None,
            "median_sampson_error_px": None,
            "estimated_rotation_degrees": None,
            "estimated_translation_direction": None,
        }
    inliers = raw_inliers.reshape(-1).astype(bool)
    errors = _sampson_errors(fundamental, points0, points1)

    rotation_degrees = None
    translation_direction = None
    essential, essential_mask = cv2.findEssentialMat(
        points0,
        points1,
        intrinsics,
        cv2.USAC_MAGSAC,
        0.999,
        threshold,
        maxIters=20_000,
    )
    if essential is not None and essential.shape == (3, 3):
        _count, rotation, translation, _pose_mask = cv2.recoverPose(
            essential,
            points0,
            points1,
            intrinsics,
            mask=essential_mask,
        )
        cosine = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
        rotation_degrees = math.degrees(math.acos(float(cosine)))
        translation_direction = translation.reshape(3).astype(float).tolist()
    return inliers, {
        "fundamental_matrix": fundamental.astype(float).tolist(),
        "median_sampson_error_px": (
            float(np.median(errors[inliers])) if np.any(inliers) else None
        ),
        "estimated_rotation_degrees": rotation_degrees,
        "estimated_translation_direction": translation_direction,
    }


def _draw_pair(
    image0: np.ndarray,
    image1: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    inliers: np.ndarray,
    strict_inliers: np.ndarray,
    title: str,
    stats: str,
) -> Image.Image:
    height = max(image0.shape[0], image1.shape[0])
    width0, width1 = image0.shape[1], image1.shape[1]
    canvas = Image.new("RGB", (width0 + width1, height + 42), (12, 12, 12))
    canvas.paste(Image.fromarray(image0), (0, 42))
    canvas.paste(Image.fromarray(image1), (width0, 42))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 7), title, fill=(255, 255, 255))
    draw.text((10, 23), stats, fill=(190, 220, 255))

    outlier_ids = np.flatnonzero(~inliers)
    uncertain_ids = np.flatnonzero(inliers & ~strict_inliers)
    inlier_ids = np.flatnonzero(strict_inliers)
    if len(outlier_ids) > 40:
        outlier_ids = outlier_ids[
            np.linspace(0, len(outlier_ids) - 1, 40, dtype=np.int64)
        ]
    if len(inlier_ids) > 160:
        inlier_ids = inlier_ids[
            np.linspace(0, len(inlier_ids) - 1, 160, dtype=np.int64)
        ]
    if len(uncertain_ids) > 120:
        uncertain_ids = uncertain_ids[
            np.linspace(0, len(uncertain_ids) - 1, 120, dtype=np.int64)
        ]
    for index in outlier_ids:
        p0 = tuple(points0[index] + np.array((0.0, 42.0)))
        p1 = tuple(points1[index] + np.array((width0, 42.0)))
        draw.line((p0, p1), fill=(180, 60, 60), width=1)
    for index in uncertain_ids:
        p0 = tuple(points0[index] + np.array((0.0, 42.0)))
        p1 = tuple(points1[index] + np.array((width0, 42.0)))
        draw.line((p0, p1), fill=(255, 170, 45), width=1)
    for index in inlier_ids:
        p0 = tuple(points0[index] + np.array((0.0, 42.0)))
        p1 = tuple(points1[index] + np.array((width0, 42.0)))
        draw.line((p0, p1), fill=(80, 255, 120), width=2)
        radius = 2
        for point in (p0, p1):
            draw.ellipse(
                (
                    point[0] - radius,
                    point[1] - radius,
                    point[0] + radius,
                    point[1] + radius,
                ),
                fill=(255, 235, 80),
            )
    return canvas


def _sample_depth_and_unproject(
    points: np.ndarray,
    original_size: tuple[int, int],
    depth: np.ndarray,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Lift original-image points into world space without depth quantization."""

    original_width, original_height = original_size
    processed_height, processed_width = depth.shape
    processed = points.astype(np.float64).copy()
    processed[:, 0] *= processed_width / float(original_width)
    processed[:, 1] *= processed_height / float(original_height)

    sampled_depth = cv2.remap(
        depth.astype(np.float32),
        processed[:, 0].astype(np.float32).reshape(-1, 1),
        processed[:, 1].astype(np.float32).reshape(-1, 1),
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=float("nan"),
    ).reshape(-1).astype(np.float64)

    fx, fy = float(intrinsics[0, 0]), float(intrinsics[1, 1])
    cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
    camera = np.stack(
        (
            (processed[:, 0] - cx) * sampled_depth / fx,
            (processed[:, 1] - cy) * sampled_depth / fy,
            sampled_depth,
        ),
        axis=1,
    )
    world_to_camera = np.eye(4, dtype=np.float64)
    rows, columns = extrinsics.shape
    if (rows, columns) not in ((3, 4), (4, 4)):
        raise ValueError(f"unexpected DA3 extrinsics shape: {extrinsics.shape}")
    world_to_camera[:rows, :columns] = extrinsics
    camera_to_world = np.linalg.inv(world_to_camera)
    world = (
        camera @ camera_to_world[:3, :3].T
        + camera_to_world[:3, 3]
    )
    valid = np.isfinite(world).all(axis=1) & (sampled_depth > 0.0)
    return world, valid


def _depth_geometry_report(
    points0: np.ndarray,
    points1: np.ndarray,
    inliers: np.ndarray,
    size0: tuple[int, int],
    size1: tuple[int, int],
    depth0: np.ndarray,
    depth1: np.ndarray,
    intrinsics0: np.ndarray,
    intrinsics1: np.ndarray,
    extrinsics0: np.ndarray,
    extrinsics1: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    all_distances = np.full(len(points0), np.nan, dtype=np.float64)
    selected0 = points0[inliers]
    selected1 = points1[inliers]
    if len(selected0) == 0:
        return {"da3_3d_correspondence_count": 0}, all_distances
    world0, valid0 = _sample_depth_and_unproject(
        selected0, size0, depth0, intrinsics0, extrinsics0
    )
    world1, valid1 = _sample_depth_and_unproject(
        selected1, size1, depth1, intrinsics1, extrinsics1
    )
    valid = valid0 & valid1
    distances = np.linalg.norm(world1[valid] - world0[valid], axis=1)
    selected_indices = np.flatnonzero(inliers)
    all_distances[selected_indices[valid]] = distances
    if len(distances) == 0:
        return {"da3_3d_correspondence_count": 0}, all_distances
    return {
        "da3_3d_correspondence_count": int(len(distances)),
        "da3_3d_error_median": float(np.median(distances)),
        "da3_3d_error_p90": float(np.percentile(distances, 90.0)),
        "da3_3d_inlier_ratio_2cm": float(np.mean(distances <= 0.02)),
        "da3_3d_inlier_ratio_5cm": float(np.mean(distances <= 0.05)),
        "da3_3d_inlier_ratio_10cm": float(np.mean(distances <= 0.10)),
    }, all_distances


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, item: tuple[int, int]) -> tuple[int, int]:
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: tuple[int, int], right: tuple[int, int]) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def main() -> int:
    args = _arguments()
    if len(args.inputs) < 3:
        raise SystemExit("a closed feature ring needs at least three views")
    missing = [str(path) for path in args.inputs if not path.is_file()]
    if missing:
        raise SystemExit(f"missing input images: {missing}")
    if args.da3_prediction is not None and not args.da3_prediction.is_file():
        raise SystemExit(f"missing DA3 prediction: {args.da3_prediction}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    da3 = None
    if args.da3_prediction is not None:
        da3 = np.load(args.da3_prediction)
        required = {"depth", "intrinsics", "extrinsics"}
        absent = required.difference(da3.files)
        if absent:
            raise SystemExit(f"DA3 prediction lacks arrays: {sorted(absent)}")
        if len(da3["depth"]) != len(args.inputs):
            raise SystemExit(
                "DA3 prediction view count does not match the input ring"
            )

    device = torch.device(args.device)
    extractor = SuperPoint(max_num_keypoints=args.max_keypoints).eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    rgb_images: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    features: list[dict[str, torch.Tensor]] = []
    print(f"Extracting SuperPoint features for {len(args.inputs)} views...", flush=True)
    with torch.inference_mode():
        for path in args.inputs:
            rgb = np.asarray(Image.open(path).convert("RGB"))
            rgb_images.append(rgb)
            masks.append(_subject_mask(rgb, args.mask_erosion))
            features.append(extractor.extract(load_image(path).to(device)))

    height, width = rgb_images[0].shape[:2]
    focal = (0.5 * width) / math.tan(
        math.radians(args.horizontal_fov_degrees) * 0.5
    )
    intrinsics = np.array(
        (
            (focal, 0.0, (width - 1) * 0.5),
            (0.0, focal, (height - 1) * 0.5),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )

    union_find = _UnionFind()
    pair_reports: list[dict[str, object]] = []
    saved_matches: dict[str, np.ndarray] = {}
    visualizations: list[Image.Image] = []
    with torch.inference_mode():
        for index0 in range(len(args.inputs)):
            index1 = (index0 + 1) % len(args.inputs)
            result = rbd(matcher({
                "image0": features[index0],
                "image1": features[index1],
            }))
            feature0 = rbd(features[index0])
            feature1 = rbd(features[index1])
            matches = result["matches"].detach().cpu().numpy()
            match_scores = result["scores"].detach().cpu().numpy()
            keypoints0 = feature0["keypoints"].detach().cpu().numpy()
            keypoints1 = feature1["keypoints"].detach().cpu().numpy()
            points0 = keypoints0[matches[:, 0]]
            points1 = keypoints1[matches[:, 1]]
            subject = _inside(masks[index0], points0) & _inside(
                masks[index1], points1
            )
            subject_matches = matches[subject]
            subject_scores = match_scores[subject]
            points0 = points0[subject]
            points1 = points1[subject]
            inliers, geometry = _estimate_pair(
                points0, points1, args.ransac_threshold, intrinsics
            )
            depth_geometry: dict[str, object] = {}
            depth_distances = np.full(len(points0), np.nan, dtype=np.float64)
            if da3 is not None:
                depth_geometry, depth_distances = _depth_geometry_report(
                    points0,
                    points1,
                    inliers,
                    (rgb_images[index0].shape[1], rgb_images[index0].shape[0]),
                    (rgb_images[index1].shape[1], rgb_images[index1].shape[0]),
                    da3["depth"][index0],
                    da3["depth"][index1],
                    da3["intrinsics"][index0],
                    da3["intrinsics"][index1],
                    da3["extrinsics"][index0],
                    da3["extrinsics"][index1],
                )
            strict_inliers = inliers & (
                subject_scores >= args.min_lightglue_score
            )
            if da3 is not None:
                strict_inliers &= (
                    np.isfinite(depth_distances)
                    & (depth_distances <= args.max_da3_error_m)
                )
            for match in subject_matches[strict_inliers]:
                union_find.union(
                    (index0, int(match[0])),
                    (index1, int(match[1])),
                )

            key = f"{index0:03d}_{index1:03d}"
            saved_matches[f"{key}_points0"] = points0.astype(np.float32)
            saved_matches[f"{key}_points1"] = points1.astype(np.float32)
            saved_matches[f"{key}_inliers"] = inliers.astype(np.uint8)
            saved_matches[f"{key}_strict_inliers"] = strict_inliers.astype(np.uint8)
            saved_matches[f"{key}_lightglue_scores"] = subject_scores.astype(np.float32)
            saved_matches[f"{key}_da3_3d_errors"] = depth_distances.astype(np.float32)
            inlier_count = int(inliers.sum())
            strict_count = int(strict_inliers.sum())
            report = {
                "pair": [index0, index1],
                "raw_match_count": int(len(matches)),
                "subject_match_count": int(len(points0)),
                "ransac_inlier_count": inlier_count,
                "strict_inlier_count": strict_count,
                "strict_inlier_ratio": (
                    strict_count / len(points0) if len(points0) else 0.0
                ),
                "ransac_inlier_ratio": (
                    inlier_count / len(points0) if len(points0) else 0.0
                ),
                "median_displacement_px": (
                    float(np.median(np.linalg.norm(points1 - points0, axis=1)))
                    if len(points0) else None
                ),
                **geometry,
                **depth_geometry,
            }
            pair_reports.append(report)
            stats = (
                f"subject {len(points0)} · MAGSAC {inlier_count} "
                f"· strict {strict_count} ({report['strict_inlier_ratio']:.1%}) · "
                f"Sampson {report['median_sampson_error_px']} px"
            )
            visualization = _draw_pair(
                rgb_images[index0],
                rgb_images[index1],
                points0,
                points1,
                inliers,
                strict_inliers,
                f"View {index0} → {index1}",
                stats,
            )
            visualization.save(args.output_dir / f"matches-{key}.png")
            visualizations.append(visualization)
            print(stats, flush=True)

    tracks: dict[tuple[int, int], set[tuple[int, int]]] = {}
    for node in union_find.parent:
        tracks.setdefault(union_find.find(node), set()).add(node)
    track_lengths = [len({view for view, _keypoint in nodes}) for nodes in tracks.values()]
    track_histogram = {
        str(length): int(sum(value == length for value in track_lengths))
        for length in range(2, len(args.inputs) + 1)
    }
    report = {
        "inputs": [str(path.resolve()) for path in args.inputs],
        "extractor": "SuperPoint",
        "matcher": "LightGlue(superpoint)",
        "max_keypoints": args.max_keypoints,
        "mask_erosion": args.mask_erosion,
        "ransac": "OpenCV USAC_MAGSAC fundamental matrix",
        "ransac_threshold_px": args.ransac_threshold,
        "min_lightglue_score": args.min_lightglue_score,
        "max_da3_error_m": args.max_da3_error_m,
        "horizontal_fov_degrees_for_pose_diagnostic": (
            args.horizontal_fov_degrees
        ),
        "da3_prediction": (
            str(args.da3_prediction.resolve())
            if args.da3_prediction is not None
            else None
        ),
        "keypoints_per_view": [
            int(rbd(feature)["keypoints"].shape[0]) for feature in features
        ],
        "subject_mask_pixels_per_view": [int(mask.sum()) for mask in masks],
        "pairs": pair_reports,
        "track_count": len(track_lengths),
        "track_length_histogram": track_histogram,
        "tracks_with_at_least_3_views": int(sum(x >= 3 for x in track_lengths)),
        "tracks_with_at_least_4_views": int(sum(x >= 4 for x in track_lengths)),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    np.savez_compressed(args.output_dir / "matches.npz", **saved_matches)

    thumbs: list[Image.Image] = []
    for visualization in visualizations:
        thumbnail = visualization.copy()
        thumbnail.thumbnail((720, 430))
        thumbs.append(thumbnail)
    sheet = Image.new("RGB", (1440, 430 * 4), (8, 8, 8))
    for index, thumbnail in enumerate(thumbs):
        x = (index % 2) * 720 + (720 - thumbnail.width) // 2
        y = (index // 2) * 430 + (430 - thumbnail.height) // 2
        sheet.paste(thumbnail, (x, y))
    sheet.save(args.output_dir / "contact.png")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
