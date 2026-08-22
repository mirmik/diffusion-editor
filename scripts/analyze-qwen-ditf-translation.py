#!/usr/bin/env python3
"""Evaluate Qwen DiT features against an exact known raster translation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy

import cv2
import numpy as np
from PIL import Image
import torch


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mask-erosion", type=int, default=3)
    return parser.parse_args()


def _layer_norm(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32)
    mean = features.mean(axis=-1, keepdims=True)
    variance = np.mean((features - mean) ** 2, axis=-1, keepdims=True)
    return (features - mean) / np.sqrt(variance + 1e-6)


def _l2(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32)
    return features / np.maximum(
        np.linalg.norm(features, axis=-1, keepdims=True), 1e-8
    )


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


def _evaluate(
    name: str,
    features0: np.ndarray,
    features1: np.ndarray,
    masks: list[np.ndarray],
    shift: tuple[int, int],
    device: torch.device,
) -> tuple[dict[str, object], tuple[np.ndarray, ...]]:
    grid_height, grid_width = masks[0].shape
    shift_x, shift_y = shift
    source_y, source_x = np.nonzero(masks[0])
    expected_x = source_x + shift_x
    expected_y = source_y + shift_y
    valid = (
        (expected_x >= 0)
        & (expected_x < grid_width)
        & (expected_y >= 0)
        & (expected_y < grid_height)
    )
    valid_ids = np.flatnonzero(valid)
    valid[valid_ids] &= masks[1][expected_y[valid_ids], expected_x[valid_ids]]
    source_x = source_x[valid]
    source_y = source_y[valid]
    expected_x = expected_x[valid]
    expected_y = expected_y[valid]
    source_ids = source_y * grid_width + source_x
    expected_ids = expected_y * grid_width + expected_x

    target_ids = np.flatnonzero(masks[1].reshape(-1))
    target_lookup = np.full(grid_height * grid_width, -1, dtype=np.int64)
    target_lookup[target_ids] = np.arange(len(target_ids), dtype=np.int64)
    expected_local = target_lookup[expected_ids]
    if np.any(expected_local < 0):
        raise RuntimeError("expected translated tokens escaped the target mask")

    tensor0 = torch.from_numpy(
        features0.reshape(-1, features0.shape[-1])[source_ids]
    ).to(device=device, dtype=torch.float32)
    tensor1 = torch.from_numpy(
        features1.reshape(-1, features1.shape[-1])[target_ids]
    ).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        similarity = tensor0 @ tensor1.T
        predicted_local = similarity.argmax(dim=1)
        predicted_ids = torch.from_numpy(target_ids).to(device)[predicted_local]
        best_source = similarity.argmax(dim=0)
        source_local = torch.arange(len(source_ids), device=device)
        mutual = best_source[predicted_local] == source_local
        expected_local_tensor = torch.from_numpy(expected_local).to(device)
        expected_scores = similarity[source_local, expected_local_tensor]
        ranks = (similarity > expected_scores[:, None]).sum(dim=1) + 1
        best_scores = similarity[source_local, predicted_local]

    predicted_ids_np = predicted_ids.cpu().numpy()
    predicted_y, predicted_x = np.divmod(predicted_ids_np, grid_width)
    error_x = predicted_x - expected_x
    error_y = predicted_y - expected_y
    chebyshev = np.maximum(np.abs(error_x), np.abs(error_y))
    exact = chebyshev == 0
    within1 = chebyshev <= 1
    mutual_np = mutual.cpu().numpy()
    ranks_np = ranks.cpu().numpy()
    best_scores_np = best_scores.cpu().numpy()
    expected_scores_np = expected_scores.cpu().numpy()
    same_position = (predicted_x == source_x) & (predicted_y == source_y)

    report: dict[str, object] = {
        "name": name,
        "query_count": int(len(source_ids)),
        "exact_top1_count": int(exact.sum()),
        "exact_top1_ratio": float(exact.mean()),
        "within_one_token_count": int(within1.sum()),
        "within_one_token_ratio": float(within1.mean()),
        "expected_top5_ratio": float(np.mean(ranks_np <= 5)),
        "expected_top10_ratio": float(np.mean(ranks_np <= 10)),
        "expected_median_rank": float(np.median(ranks_np)),
        "same_position_ratio": float(same_position.mean()),
        "mutual_count": int(mutual_np.sum()),
        "mutual_coverage": float(mutual_np.mean()),
        "mutual_exact_ratio": (
            float(exact[mutual_np].mean()) if mutual_np.any() else 0.0
        ),
        "mutual_within_one_ratio": (
            float(within1[mutual_np].mean()) if mutual_np.any() else 0.0
        ),
        "median_predicted_shift_tokens": [
            float(np.median(predicted_x - source_x)),
            float(np.median(predicted_y - source_y)),
        ],
        "median_absolute_error_tokens": [
            float(np.median(np.abs(error_x))),
            float(np.median(np.abs(error_y))),
        ],
        "best_similarity_median": float(np.median(best_scores_np)),
        "expected_similarity_median": float(np.median(expected_scores_np)),
        "expected_similarity_margin_median": float(
            np.median(expected_scores_np - best_scores_np)
        ),
    }
    return report, (source_ids, predicted_ids_np, within1, exact)


def main() -> int:
    args = _arguments()
    manifest_path = args.capture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    archives = [np.load(path) for path in manifest["features"]]
    images = [
        np.asarray(Image.open(path).convert("RGB")) for path in manifest["images"]
    ]
    grid = tuple(int(value) for value in manifest["feature_grid"])
    size = tuple(int(value) for value in manifest["output_size"])
    shift_float = tuple(float(value) for value in manifest["translation_tokens"])
    shift = tuple(int(round(value)) for value in shift_float)
    if any(abs(value - rounded) > 1e-6 for value, rounded in zip(shift_float, shift)):
        raise SystemExit("translation must be an integer number of 16-pixel tokens")

    helpers = runpy.run_path(
        str(Path(__file__).with_name("experiment-multiview-keypoints.py"))
    )
    subject_mask = helpers["_subject_mask"]
    draw_pair = helpers["_draw_pair"]
    masks = [
        cv2.resize(
            subject_mask(image, args.mask_erosion).astype(np.uint8),
            grid,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        for image in images
    ]

    raw_maps = [archive["raw"] for archive in archives]
    scales = [archive["scale"] for archive in archives]
    shifts = [archive["shift"] for archive in archives]
    channel_magnitude = sum(
        np.mean(np.abs(raw.reshape(-1, raw.shape[-1])), axis=0)
        for raw in raw_maps
    )
    channel_order = np.argsort(channel_magnitude)[::-1]
    variants: dict[str, list[np.ndarray]] = {
        "raw": [_l2(raw) for raw in raw_maps],
        "layernorm": [_l2(_layer_norm(raw)) for raw in raw_maps],
        "adaln": [
            _l2(_layer_norm(raw) * (1.0 + scale) + view_shift)
            for raw, scale, view_shift in zip(raw_maps, scales, shifts)
        ],
    }
    for count in (1, 4, 16):
        discarded = channel_order[:count]
        maps = []
        for raw, scale, view_shift in zip(raw_maps, scales, shifts):
            filtered = raw.copy()
            filtered[..., discarded] = 0.0
            maps.append(
                _l2(_layer_norm(filtered) * (1.0 + scale) + view_shift)
            )
        variants[f"adaln-cd{count}"] = maps

    reports = []
    panels = []
    device = torch.device(args.device)
    for name, feature_maps in variants.items():
        report, visual = _evaluate(
            name, feature_maps[0], feature_maps[1], masks, shift, device
        )
        reports.append(report)
        source_ids, predicted_ids, within1, exact = visual
        points0 = _grid_points(source_ids, grid, size)
        points1 = _grid_points(predicted_ids, grid, size)
        visualization = draw_pair(
            images[0],
            images[1],
            points0,
            points1,
            within1,
            exact,
            f"{name} · exact raster shift {shift[0]}x{shift[1]} tokens",
            (
                f"green=exact {report['exact_top1_ratio']:.1%} · "
                f"orange=within 1 {report['within_one_token_ratio']:.1%} · "
                f"expected median rank {report['expected_median_rank']:g}"
            ),
        )
        visualization.save(args.output_dir / f"{name}.png")
        panel = visualization.copy()
        panel.thumbnail((960, 544))
        panels.append(panel)
        print(
            f"{name:12s} exact={report['exact_top1_ratio']:.3f} "
            f"within1={report['within_one_token_ratio']:.3f} "
            f"top5={report['expected_top5_ratio']:.3f} "
            f"median-rank={report['expected_median_rank']:g} "
            f"same-position={report['same_position_ratio']:.3f} "
            f"median-shift={report['median_predicted_shift_tokens']}",
            flush=True,
        )

    columns = 2
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * 960, rows * 544), (8, 8, 8))
    for index, panel in enumerate(panels):
        x = (index % columns) * 960 + (960 - panel.width) // 2
        y = (index // columns) * 544 + (544 - panel.height) // 2
        sheet.paste(panel, (x, y))
    sheet.save(args.output_dir / "contact-comparison.png")

    summary = {
        "capture_manifest": str(manifest_path.resolve()),
        "ground_truth_shift_tokens": list(shift),
        "mask_erosion": args.mask_erosion,
        "global_channel_order_top32": [int(value) for value in channel_order[:32]],
        "reports": reports,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
