#!/usr/bin/env python3
"""Compare raw, AdaLN-modulated, channel-discarded, and anchor Q/K features."""

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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-similarity", type=float, default=0.40)
    parser.add_argument("--ransac-threshold", type=float, default=2.0)
    parser.add_argument("--anchor-topk", type=int, default=64)
    parser.add_argument("--anchor-temperature", type=float, default=0.05)
    return parser.parse_args()


def _layer_norm(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32)
    mean = features.mean(axis=-1, keepdims=True)
    variance = np.mean((features - mean) ** 2, axis=-1, keepdims=True)
    return (features - mean) / np.sqrt(variance + 1e-6)


def _l2(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32)
    return features / np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)


def _grid_points(
    indices: np.ndarray,
    grid: tuple[int, int],
    size: tuple[int, int],
) -> np.ndarray:
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


def _mutual_from_features(
    features0: np.ndarray,
    features1: np.ndarray,
    mask0: np.ndarray,
    mask1: np.ndarray,
    minimum: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids0 = np.flatnonzero(mask0.reshape(-1))
    ids1 = np.flatnonzero(mask1.reshape(-1))
    tensor0 = torch.from_numpy(features0.reshape(-1, features0.shape[-1])[ids0]).to(
        device=device, dtype=torch.float32
    )
    tensor1 = torch.from_numpy(features1.reshape(-1, features1.shape[-1])[ids1]).to(
        device=device, dtype=torch.float32
    )
    similarity = tensor0 @ tensor1.T
    best1 = similarity.argmax(dim=1)
    best0 = similarity.argmax(dim=0)
    source = torch.arange(len(ids0), device=device)
    scores = similarity[source, best1]
    keep = (best0[best1] == source) & (scores >= minimum)
    selected0 = source[keep].cpu().numpy()
    selected1 = best1[keep].cpu().numpy()
    return ids0[selected0], ids1[selected1], scores[keep].cpu().numpy()


def _anchor_descriptors(
    q: np.ndarray,
    k: np.ndarray,
    target_mask: np.ndarray,
    anchor_mask: np.ndarray,
    topk: int,
    temperature: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    target_ids = np.flatnonzero(target_mask.reshape(-1))
    anchor_ids = np.flatnonzero(anchor_mask.reshape(-1))
    query = torch.from_numpy(q[target_ids].astype(np.float32)).to(device)
    keys = torch.from_numpy(k[anchor_ids].astype(np.float32)).to(device)
    # norm_q/norm_k normalize each head. Flattening and normalizing computes
    # the mean head-wise pre-RoPE affinity.
    query = torch.nn.functional.normalize(query.flatten(1), dim=-1)
    keys = torch.nn.functional.normalize(keys.flatten(1), dim=-1)
    affinity = query @ keys.T
    count = min(topk, affinity.shape[1])
    values, indices = torch.topk(affinity, count, dim=1)
    weights = torch.softmax(values / temperature, dim=1)
    descriptors = torch.zeros(
        (len(target_ids), len(anchor_ids)), device=device, dtype=torch.float32
    )
    descriptors.scatter_(1, indices, weights)
    descriptors = torch.nn.functional.normalize(descriptors, dim=-1)
    return target_ids, descriptors.cpu().numpy()


def _mutual_preselected(
    ids0: np.ndarray,
    features0: np.ndarray,
    ids1: np.ndarray,
    features1: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tensor0 = torch.from_numpy(features0).to(device=device, dtype=torch.float32)
    tensor1 = torch.from_numpy(features1).to(device=device, dtype=torch.float32)
    similarity = tensor0 @ tensor1.T
    best1 = similarity.argmax(dim=1)
    best0 = similarity.argmax(dim=0)
    source = torch.arange(len(ids0), device=device)
    scores = similarity[source, best1]
    keep = best0[best1] == source
    selected0 = source[keep].cpu().numpy()
    selected1 = best1[keep].cpu().numpy()
    return ids0[selected0], ids1[selected1], scores[keep].cpu().numpy()


def _evaluate(
    name: str,
    ids0: np.ndarray,
    ids1: np.ndarray,
    scores: np.ndarray,
    grid: tuple[int, int],
    size: tuple[int, int],
    threshold: float,
    intrinsics: np.ndarray,
    estimate_pair,
) -> tuple[dict[str, object], tuple[np.ndarray, ...]]:
    points0 = _grid_points(ids0, grid, size)
    points1 = _grid_points(ids1, grid, size)
    try:
        inliers, geometry = estimate_pair(points0, points1, threshold, intrinsics)
    except cv2.error:
        inliers = np.zeros(len(points0), dtype=bool)
        geometry = {}
    rows0, rows1 = ids0 // grid[0], ids1 // grid[0]
    columns0, columns1 = ids0 % grid[0], ids1 % grid[0]
    displacement = np.linalg.norm(points1 - points0, axis=1)
    report = {
        "name": name,
        "match_count": int(len(ids0)),
        "similarity_median": float(np.median(scores)) if len(scores) else None,
        "magsac_inlier_count": int(inliers.sum()),
        "magsac_inlier_ratio": float(inliers.mean()) if len(inliers) else 0.0,
        "same_token_ratio": (
            float(np.mean((rows0 == rows1) & (columns0 == columns1)))
            if len(ids0) else 0.0
        ),
        "same_row_ratio": float(np.mean(rows0 == rows1)) if len(ids0) else 0.0,
        "inlier_same_row_ratio": (
            float(np.mean(rows0[inliers] == rows1[inliers])) if inliers.any() else 0.0
        ),
        "median_displacement_px": (
            float(np.median(displacement)) if len(displacement) else None
        ),
        **geometry,
    }
    return report, (points0, points1, inliers)


def main() -> int:
    args = _arguments()
    manifest_path = args.capture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = manifest["views"]
    if len(views) != 2:
        raise SystemExit("focused DiTF comparison expects exactly two views")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archives = [np.load(view["features"]) for view in views]
    images = [np.asarray(Image.open(view["image"]).convert("RGB")) for view in views]
    grid = tuple(int(value) for value in manifest["feature_grid"])
    size = tuple(int(value) for value in manifest["output_size"])
    width, height = size

    helpers = runpy.run_path(str(Path(__file__).with_name("experiment-multiview-keypoints.py")))
    subject_mask = helpers["_subject_mask"]
    estimate_pair = helpers["_estimate_pair"]
    draw_pair = helpers["_draw_pair"]
    masks = [
        cv2.resize(subject_mask(image, 3).astype(np.uint8), grid,
                   interpolation=cv2.INTER_NEAREST).astype(bool)
        for image in images
    ]
    focal = (0.5 * width) / np.tan(np.radians(34.0) * 0.5)
    intrinsics = np.array(
        ((focal, 0.0, (width - 1) * 0.5),
         (0.0, focal, (height - 1) * 0.5),
         (0.0, 0.0, 1.0)), dtype=np.float64,
    )
    device = torch.device(args.device)

    steps = sorted(
        int(key.removeprefix("raw_step_"))
        for key in archives[0].files if key.startswith("raw_step_")
    )
    channel_magnitude = np.zeros(manifest["feature_channels"], dtype=np.float64)
    for step in steps:
        for archive in archives:
            raw = archive[f"raw_step_{step}"].reshape(-1, manifest["feature_channels"])
            channel_magnitude += np.mean(np.abs(raw), axis=0)
    global_channel_order = np.argsort(channel_magnitude)[::-1]

    reports = []
    visual_cache: dict[str, tuple[np.ndarray, ...]] = {}
    for step in steps:
        raw_maps = [archive[f"raw_step_{step}"] for archive in archives]
        scale = [archive[f"scale_step_{step}"] for archive in archives]
        shift = [archive[f"shift_step_{step}"] for archive in archives]
        variants: dict[str, list[np.ndarray]] = {
            f"raw-step-{step}": [_l2(raw) for raw in raw_maps],
            f"ln-step-{step}": [_l2(_layer_norm(raw)) for raw in raw_maps],
            f"adaln-step-{step}": [
                _l2(_layer_norm(raw) * (1.0 + view_scale) + view_shift)
                for raw, view_scale, view_shift in zip(raw_maps, scale, shift)
            ],
        }
        for count in (1, 4, 16):
            discarded = global_channel_order[:count]
            maps = []
            for raw, view_scale, view_shift in zip(raw_maps, scale, shift):
                filtered = raw.copy()
                filtered[..., discarded] = 0.0
                maps.append(
                    _l2(_layer_norm(filtered) * (1.0 + view_scale) + view_shift)
                )
            variants[f"adaln-cd{count}-step-{step}"] = maps

        for name, feature_maps in variants.items():
            ids0, ids1, scores = _mutual_from_features(
                feature_maps[0], feature_maps[1], masks[0], masks[1],
                args.min_similarity, device,
            )
            report, visual = _evaluate(
                name, ids0, ids1, scores, grid, size,
                args.ransac_threshold, intrinsics, estimate_pair,
            )
            reports.append(report)
            visual_cache[name] = visual
            print(
                name, "MNN", report["match_count"],
                "MAGSAC", report["magsac_inlier_count"],
                "same-row", f"{report['same_row_ratio']:.3f}",
                "rotation", report.get("estimated_rotation_degrees"),
                flush=True,
            )

    with Image.open(manifest["front"]) as source:
        front_image = np.asarray(source.convert("RGB"))
    with Image.open(manifest["back"]) as source:
        back_image = np.asarray(source.convert("RGB"))
    anchor_masks = []
    for image in (front_image, back_image):
        mask = subject_mask(image, 3).astype(np.uint8)
        anchor_masks.append(
            cv2.resize(mask, grid, interpolation=cv2.INTER_NEAREST).astype(bool)
        )
    anchor_mask = np.concatenate([mask.reshape(-1) for mask in anchor_masks])
    anchor_ids = []
    anchor_features = []
    for archive, mask in zip(archives, masks):
        ids, descriptors = _anchor_descriptors(
            archive["target_q"], archive["anchor_k"], mask, anchor_mask,
            args.anchor_topk, args.anchor_temperature, device,
        )
        anchor_ids.append(ids)
        anchor_features.append(descriptors)
    ids0, ids1, scores = _mutual_preselected(
        anchor_ids[0], anchor_features[0], anchor_ids[1], anchor_features[1], device
    )
    anchor_name = "anchor-qk-step-1"
    report, visual = _evaluate(
        anchor_name, ids0, ids1, scores, grid, size,
        args.ransac_threshold, intrinsics, estimate_pair,
    )
    reports.append(report)
    visual_cache[anchor_name] = visual
    print(
        anchor_name, "MNN", report["match_count"],
        "MAGSAC", report["magsac_inlier_count"],
        "same-row", f"{report['same_row_ratio']:.3f}",
        "rotation", report.get("estimated_rotation_degrees"),
        flush=True,
    )

    selected_names = [
        "raw-step-1",
        "adaln-step-1",
        "adaln-cd1-step-1",
        "adaln-cd4-step-1",
        "adaln-cd16-step-1",
        anchor_name,
    ]
    panels = []
    report_by_name = {report["name"]: report for report in reports}
    for name in selected_names:
        points0, points1, inliers = visual_cache[name]
        report = report_by_name[name]
        visualization = draw_pair(
            images[0], images[1], points0, points1, inliers, inliers,
            f"{name} · elevated 045 -> 090 · green=MAGSAC only",
            (
                f"MNN {report['match_count']} · MAGSAC {report['magsac_inlier_count']} · "
                f"same-row {report['same_row_ratio']:.1%}"
            ),
        )
        path = args.output_dir / f"{name}.png"
        visualization.save(path)
        panel = visualization.copy()
        panel.thumbnail((960, 544))
        panels.append(panel)
    sheet = Image.new("RGB", (1920, 544 * 3), (8, 8, 8))
    for index, panel in enumerate(panels):
        x = (index % 2) * 960 + (960 - panel.width) // 2
        y = (index // 2) * 544 + (544 - panel.height) // 2
        sheet.paste(panel, (x, y))
    sheet.save(args.output_dir / "contact-comparison.png")

    summary = {
        "capture_manifest": str(manifest_path.resolve()),
        "global_channel_order_top32": [int(value) for value in global_channel_order[:32]],
        "global_channel_magnitude_top32": [
            float(channel_magnitude[value]) for value in global_channel_order[:32]
        ],
        "notes": {
            "green": "MAGSAC-only visualization; not independent ground truth",
            "adaln": "LayerNorm(raw) * (1 + scale) + shift",
            "channel_discard": "zero raw channels before LayerNorm/AdaLN",
            "anchor_qk": "pre-RoPE normalized target-Q affinity to common front/back anchor-K positions",
        },
        "reports": reports,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
