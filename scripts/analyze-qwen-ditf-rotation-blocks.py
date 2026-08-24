#!/usr/bin/env python3
"""Compare Qwen DiTF-style feature blocks across a generated camera turn."""

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
    parser.add_argument("--min-similarity", type=float, default=0.40)
    parser.add_argument("--ransac-threshold", type=float, default=2.0)
    parser.add_argument(
        "--mask-mode",
        choices=("foreground", "subject", "full"),
        default="foreground",
        help=(
            "foreground keeps every non-green object, including room rails; "
            "subject selects one connected component; full keeps every token"
        ),
    )
    parser.add_argument("--mask-erosion", type=int, default=0)
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


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    values = rgb.astype(np.int16)
    red, green, blue = values[..., 0], values[..., 1], values[..., 2]
    green_screen = (
        (green - red >= 12)
        & (green - blue >= 12)
        & (green * 100 >= red * 112)
        & (green * 100 >= blue * 112)
    )
    return ~green_screen


def _mutual_matches(
    features0: np.ndarray,
    features1: np.ndarray,
    mask0: np.ndarray,
    mask1: np.ndarray,
    minimum: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ids0 = np.flatnonzero(mask0.reshape(-1))
    ids1 = np.flatnonzero(mask1.reshape(-1))
    tensor0 = torch.from_numpy(
        features0.reshape(-1, features0.shape[-1])[ids0]
    ).to(device=device, dtype=torch.float32)
    tensor1 = torch.from_numpy(
        features1.reshape(-1, features1.shape[-1])[ids1]
    ).to(device=device, dtype=torch.float32)
    with torch.inference_mode():
        similarity = tensor0 @ tensor1.T
        best1 = similarity.argmax(dim=1)
        best0 = similarity.argmax(dim=0)
        source = torch.arange(len(ids0), device=device)
        scores = similarity[source, best1]
        keep = (best0[best1] == source) & (scores >= minimum)
        selected0 = source[keep].cpu().numpy()
        selected1 = best1[keep].cpu().numpy()
        selected_scores = scores[keep].cpu().numpy()
    return ids0[selected0], ids1[selected1], selected_scores


def main() -> int:
    args = _arguments()
    manifest_path = args.capture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = manifest["views"]
    if len(views) != 2:
        raise SystemExit("rotation block comparison expects exactly two views")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_sources = []
    for view in views:
        specification = view["features"]
        if isinstance(specification, str):
            feature_sources.append(("npz", np.load(specification)))
        else:
            feature_sources.append((
                "memmap",
                {
                    name: np.load(specification[name], mmap_mode="r")
                    for name in ("raw", "shift", "scale")
                },
            ))
    images = [
        np.asarray(Image.open(view["image"]).convert("RGB")) for view in views
    ]
    blocks = [int(value) for value in manifest["blocks"]]
    block_slots = {block_index: slot for slot, block_index in enumerate(blocks)}

    def feature(source, name: str, block_index: int) -> np.ndarray:
        storage, arrays = source
        if storage == "npz":
            return arrays[f"{name}_block_{block_index}"]
        return arrays[name][block_slots[block_index]]
    grid = tuple(int(value) for value in manifest["feature_grid"])
    size = tuple(int(value) for value in manifest["output_size"])
    width, height = size

    helpers = runpy.run_path(
        str(Path(__file__).with_name("experiment-multiview-keypoints.py"))
    )
    subject_mask = helpers["_subject_mask"]
    estimate_pair = helpers["_estimate_pair"]
    draw_pair = helpers["_draw_pair"]
    if args.mask_mode == "foreground":
        full_masks = [_foreground_mask(image) for image in images]
    elif args.mask_mode == "subject":
        full_masks = [
            subject_mask(image, args.mask_erosion) for image in images
        ]
    else:
        full_masks = [
            np.ones(image.shape[:2], dtype=bool) for image in images
        ]
    masks = [
        cv2.resize(
            mask.astype(np.uint8),
            grid,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        for mask in full_masks
    ]
    focal = (0.5 * width) / np.tan(np.radians(34.0) * 0.5)
    intrinsics = np.array(
        (
            (focal, 0.0, (width - 1) * 0.5),
            (0.0, focal, (height - 1) * 0.5),
            (0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )
    device = torch.device(args.device)
    reports = []
    visual_cache = {}

    for block_index in blocks:
        raw_maps = [
            feature(source, "raw", block_index) for source in feature_sources
        ]
        scales = [
            feature(source, "scale", block_index) for source in feature_sources
        ]
        shifts = [
            feature(source, "shift", block_index) for source in feature_sources
        ]
        variants = {
            "raw": [_l2(raw) for raw in raw_maps],
            "adaln": [
                _l2(_layer_norm(raw) * (1.0 + scale) + view_shift)
                for raw, scale, view_shift in zip(raw_maps, scales, shifts)
            ],
        }
        for variant, feature_maps in variants.items():
            ids0, ids1, scores = _mutual_matches(
                feature_maps[0],
                feature_maps[1],
                masks[0],
                masks[1],
                args.min_similarity,
                device,
            )
            points0 = _grid_points(ids0, grid, size)
            points1 = _grid_points(ids1, grid, size)
            try:
                inliers, geometry = estimate_pair(
                    points0,
                    points1,
                    args.ransac_threshold,
                    intrinsics,
                )
            except cv2.error:
                inliers = np.zeros(len(points0), dtype=bool)
                geometry = {}
            rows0, rows1 = ids0 // grid[0], ids1 // grid[0]
            columns0, columns1 = ids0 % grid[0], ids1 % grid[0]
            displacement = points1 - points0
            report = {
                "block": block_index,
                "variant": variant,
                "match_count": int(len(ids0)),
                "similarity_median": (
                    float(np.median(scores)) if len(scores) else None
                ),
                "magsac_inlier_count": int(inliers.sum()),
                "magsac_inlier_ratio": (
                    float(inliers.mean()) if len(inliers) else 0.0
                ),
                "same_token_ratio": (
                    float(np.mean((rows0 == rows1) & (columns0 == columns1)))
                    if len(ids0) else 0.0
                ),
                "same_row_ratio": (
                    float(np.mean(rows0 == rows1)) if len(ids0) else 0.0
                ),
                "median_displacement_px": (
                    [
                        float(np.median(displacement[:, 0])),
                        float(np.median(displacement[:, 1])),
                    ]
                    if len(displacement) else None
                ),
                **geometry,
            }
            reports.append(report)
            visual_cache[(block_index, variant)] = (
                points0, points1, inliers, report
            )
            print(
                f"block {block_index:02d} {variant:5s}: MNN={len(ids0):4d} "
                f"MAGSAC={int(inliers.sum()):4d} "
                f"same-row={report['same_row_ratio']:.3f} "
                f"rotation={report.get('estimated_rotation_degrees')}",
                flush=True,
            )

    def render(block_index: int, variant: str) -> Image.Image:
        points0, points1, inliers, report = visual_cache[(block_index, variant)]
        return draw_pair(
            images[0],
            images[1],
            points0,
            points1,
            inliers,
            inliers,
            (
                f"Qwen block {block_index} {variant} · elevated "
                f"{views[0]['azimuth_degrees']} -> "
                f"{views[1]['azimuth_degrees']}"
            ),
            (
                f"green=MAGSAC hypothesis, not ground truth · MNN "
                f"{report['match_count']} · inliers "
                f"{report['magsac_inlier_count']} · same-row "
                f"{report['same_row_ratio']:.1%}"
            ),
        )

    adaln_reports = [
        report for report in reports
        if report["variant"] == "adaln" and report["match_count"] >= 100
    ]
    ranked = sorted(adaln_reports, key=lambda report: report["same_row_ratio"])
    fixed = [value for value in (0, 15, 30, 45, 59) if value in blocks]
    selected_blocks = sorted(set(fixed + [r["block"] for r in ranked[:5]]))

    # A compact atlas keeps every block inspectable without creating a
    # 32,000-pixel-tall contact sheet. Full-resolution panels are retained for
    # the characteristic blocks and the five least row-locked candidates.
    columns = 5
    cell_width, cell_height = 384, 218
    rows = (len(blocks) + columns - 1) // columns
    adaln_sheet = Image.new(
        "RGB", (columns * cell_width, rows * cell_height), (8, 8, 8)
    )
    for index, block_index in enumerate(blocks):
        panel = render(block_index, "adaln")
        panel.thumbnail((cell_width, cell_height))
        x = (index % columns) * cell_width + (cell_width - panel.width) // 2
        y = (index // columns) * cell_height + (cell_height - panel.height) // 2
        adaln_sheet.paste(panel, (x, y))
    adaln_sheet.save(args.output_dir / "contact-adaln-all-blocks.png")

    selected_sheet = Image.new(
        "RGB", (1920, len(selected_blocks) * 544), (8, 8, 8)
    )
    for row, block_index in enumerate(selected_blocks):
        for column, variant in enumerate(("raw", "adaln")):
            visualization = render(block_index, variant)
            visualization.save(
                args.output_dir / f"block-{block_index:02d}-{variant}.png"
            )
            panel = visualization.copy()
            panel.thumbnail((960, 544))
            x = column * 960 + (960 - panel.width) // 2
            y = row * 544 + (544 - panel.height) // 2
            selected_sheet.paste(panel, (x, y))
    selected_sheet.save(args.output_dir / "contact-selected-raw-vs-adaln.png")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    colors = {"raw": "#d95f02", "adaln": "#1b9e77"}
    for variant in ("raw", "adaln"):
        variant_reports = sorted(
            (report for report in reports if report["variant"] == variant),
            key=lambda report: report["block"],
        )
        x = [report["block"] for report in variant_reports]
        axes[0, 0].plot(
            x, [report["same_row_ratio"] for report in variant_reports],
            label=variant, color=colors[variant],
        )
        axes[0, 1].plot(
            x, [report["match_count"] for report in variant_reports],
            label=variant, color=colors[variant],
        )
        axes[1, 0].plot(
            x, [report["magsac_inlier_count"] for report in variant_reports],
            label=variant, color=colors[variant],
        )
        axes[1, 1].plot(
            x,
            [
                report.get("estimated_rotation_degrees")
                if report.get("estimated_rotation_degrees") is not None
                else np.nan
                for report in variant_reports
            ],
            label=variant,
            color=colors[variant],
        )
    axes[0, 0].set_title("Same-row ratio (lower = less position lock)")
    axes[0, 1].set_title("Reciprocal NN matches")
    axes[1, 0].set_title("MAGSAC inliers (hypotheses, not GT)")
    axes[1, 1].set_title("Recovered rotation, degrees")
    axes[1, 1].axhline(45.0, color="#7570b3", linestyle="--", label="nominal 45°")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel("Qwen transformer block")
        axis.legend()
    figure.tight_layout()
    figure.savefig(args.output_dir / "block-metrics.png", dpi=160)
    plt.close(figure)

    summary = {
        "capture_manifest": str(manifest_path.resolve()),
        "min_similarity": args.min_similarity,
        "ransac_threshold": args.ransac_threshold,
        "mask_mode": args.mask_mode,
        "mask_erosion": args.mask_erosion,
        "selected_blocks": selected_blocks,
        "least_row_locked_adaln_blocks": [r["block"] for r in ranked[:10]],
        "reports": reports,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
