#!/usr/bin/env python3
"""Match dense Qwen2.5-VL vision features across two final rasters."""

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
    parser.add_argument(
        "--masks",
        nargs=2,
        type=Path,
        help=(
            "Optional foreground masks corresponding to the two images. "
            "When omitted, use image alpha if present and fall back to the "
            "generated-image background heuristic."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-similarity", type=float, default=0.40)
    parser.add_argument(
        "--max-matches",
        type=int,
        default=0,
        help="Keep only this many highest-similarity reciprocal matches; 0 keeps all.",
    )
    parser.add_argument("--ransac-threshold", type=float, default=2.0)
    return parser.parse_args()


def _l2(features: np.ndarray) -> np.ndarray:
    features = features.astype(np.float32)
    return features / np.maximum(
        np.linalg.norm(features, axis=-1, keepdims=True), 1e-8
    )


def _subject_mask(image: Image.Image, fallback) -> tuple[np.ndarray, str]:
    if "A" in image.getbands():
        alpha = np.asarray(image.getchannel("A"))
        if np.any(alpha < 255):
            return alpha > 127, "image alpha"
    return fallback(np.asarray(image.convert("RGB"))), "non-green heuristic"


def main() -> int:
    args = _arguments()
    manifest_path = args.capture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    views = manifest["views"]
    if len(views) != 2:
        raise SystemExit("vision feature comparison expects exactly two views")
    grids = [tuple(int(value) for value in view["feature_grid"])
             for view in views]
    if grids[0] != grids[1]:
        raise SystemExit(f"view feature grids differ: {grids}")
    grid = grids[0]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_images = [Image.open(view["image"]) for view in views]
    images = [np.asarray(source.convert("RGB")) for source in source_images]
    blocks = [
        np.load(view["blocks"], mmap_mode="r") for view in views
    ]
    patches = [
        np.load(view["patch"], mmap_mode="r") for view in views
    ]
    q_global = [
        np.load(view["q_global"], mmap_mode="r") for view in views
    ]
    k_global = [
        np.load(view["k_global"], mmap_mode="r") for view in views
    ]

    ditf_helpers = runpy.run_path(
        str(Path(__file__).with_name("analyze-qwen-ditf-rotation-blocks.py"))
    )
    foreground_mask = ditf_helpers["_foreground_mask"]
    mutual_matches = ditf_helpers["_mutual_matches"]
    grid_points = ditf_helpers["_grid_points"]
    geometry_helpers = runpy.run_path(
        str(Path(__file__).with_name("experiment-multiview-keypoints.py"))
    )
    estimate_pair = geometry_helpers["_estimate_pair"]
    draw_pair = geometry_helpers["_draw_pair"]
    if args.masks:
        full_masks = [
            np.asarray(Image.open(path).convert("L")) > 127
            for path in args.masks
        ]
        mask_description = "explicit masks: " + ", ".join(
            str(path.resolve()) for path in args.masks
        )
    else:
        resolved = [
            _subject_mask(source, foreground_mask) for source in source_images
        ]
        full_masks = [mask for mask, _description in resolved]
        mask_description = ", ".join(
            description for _mask, description in resolved
        )
    masks = [
        cv2.resize(
            mask.astype(np.uint8),
            grid,
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
        for mask in full_masks
    ]
    device = torch.device(args.device)
    width, height = images[0].shape[1], images[0].shape[0]
    if images[1].shape[:2] != images[0].shape[:2]:
        raise SystemExit("pair visualization currently expects equal image sizes")
    size = (width, height)
    focal = (0.5 * width) / np.tan(np.radians(34.0) * 0.5)
    intrinsics = np.array(
        ((focal, 0.0, (width - 1) * 0.5),
         (0.0, focal, (height - 1) * 0.5),
         (0.0, 0.0, 1.0)),
        dtype=np.float64,
    )

    reports = []
    visual_cache = {}

    def evaluate(name: str, feature_maps: list[np.ndarray]) -> None:
        ids0, ids1, scores = mutual_matches(
            _l2(feature_maps[0]),
            _l2(feature_maps[1]),
            masks[0],
            masks[1],
            args.min_similarity,
            device,
        )
        if args.max_matches > 0 and len(scores) > args.max_matches:
            keep = np.argsort(scores)[-args.max_matches:]
            ids0, ids1, scores = ids0[keep], ids1[keep], scores[keep]
        points0 = grid_points(ids0, grid, size)
        points1 = grid_points(ids1, grid, size)
        try:
            inliers, geometry = estimate_pair(
                points0, points1, args.ransac_threshold, intrinsics
            )
        except cv2.error:
            inliers = np.zeros(len(points0), dtype=bool)
            geometry = {}
        rows0, rows1 = ids0 // grid[0], ids1 // grid[0]
        cols0, cols1 = ids0 % grid[0], ids1 % grid[0]
        displacement = points1 - points0
        report = {
            "name": name,
            "match_count": int(len(ids0)),
            "similarity_median": (
                float(np.median(scores)) if len(scores) else None
            ),
            "magsac_inlier_count": int(inliers.sum()),
            "magsac_inlier_ratio": (
                float(inliers.mean()) if len(inliers) else 0.0
            ),
            "same_token_ratio": (
                float(np.mean((rows0 == rows1) & (cols0 == cols1)))
                if len(ids0) else 0.0
            ),
            "same_row_ratio": (
                float(np.mean(rows0 == rows1)) if len(ids0) else 0.0
            ),
            "median_displacement_px": (
                [float(np.median(displacement[:, 0])),
                 float(np.median(displacement[:, 1]))]
                if len(displacement) else None
            ),
            **geometry,
        }
        reports.append(report)
        visual_cache[name] = (points0, points1, inliers, report)
        print(
            f"{name:12s}: MNN={len(ids0):4d} "
            f"MAGSAC={int(inliers.sum()):4d} "
            f"same-row={report['same_row_ratio']:.3f} "
            f"same-token={report['same_token_ratio']:.3f} "
            f"rotation={report.get('estimated_rotation_degrees')}",
            flush=True,
        )

    evaluate("patch", patches)
    for block_index in manifest["blocks"]:
        evaluate(
            f"block-{block_index:02d}",
            [source[block_index] for source in blocks],
        )
    for slot, block_index in enumerate(manifest["global_attention_blocks"]):
        evaluate(
            f"q-{block_index:02d}",
            [source[slot] for source in q_global],
        )
        evaluate(
            f"k-{block_index:02d}",
            [source[slot] for source in k_global],
        )

    def render(name: str) -> Image.Image:
        points0, points1, inliers, report = visual_cache[name]
        return draw_pair(
            images[0], images[1], points0, points1, inliers, inliers,
            f"Qwen2.5-VL {name} · final rasters · "
            f"{views[0]['label']} -> {views[1]['label']}",
            (
                f"green=MAGSAC hypothesis, not ground truth · MNN "
                f"{report['match_count']} · inliers "
                f"{report['magsac_inlier_count']} · same-row "
                f"{report['same_row_ratio']:.1%}"
            ),
        )

    block_reports = [
        report for report in reports if report["name"].startswith("block-")
    ]
    ranked = sorted(block_reports, key=lambda report: report["same_row_ratio"])
    selected_names = {"patch"}
    selected_names.update(
        f"block-{index:02d}" for index in manifest["global_attention_blocks"]
    )
    selected_names.update(report["name"] for report in ranked[:5])
    selected_names.update(
        report["name"] for report in reports
        if report["name"].startswith(("q-", "k-"))
    )
    for name in sorted(selected_names):
        render(name).save(args.output_dir / f"{name}.png")

    columns = 4
    cell_width, cell_height = 480, 272
    rows = (len(block_reports) + columns - 1) // columns
    atlas = Image.new(
        "RGB", (columns * cell_width, rows * cell_height), (8, 8, 8)
    )
    for index, report in enumerate(block_reports):
        panel = render(report["name"])
        panel.thumbnail((cell_width, cell_height))
        x = (index % columns) * cell_width + (cell_width - panel.width) // 2
        y = (index // columns) * cell_height + (cell_height - panel.height) // 2
        atlas.paste(panel, (x, y))
    atlas.save(args.output_dir / "contact-blocks-all.png")

    qk_names = [
        report["name"] for report in reports
        if report["name"].startswith(("q-", "k-"))
    ]
    qk_sheet = Image.new("RGB", (1920, 4 * 544), (8, 8, 8))
    for index, name in enumerate(qk_names):
        panel = render(name)
        panel.thumbnail((960, 544))
        x = (index % 2) * 960 + (960 - panel.width) // 2
        y = (index // 2) * 544 + (544 - panel.height) // 2
        qk_sheet.paste(panel, (x, y))
    qk_sheet.save(args.output_dir / "contact-global-qk.png")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = [int(report["name"].split("-")[1]) for report in block_reports]
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes[0, 0].plot(x, [r["same_row_ratio"] for r in block_reports])
    axes[0, 0].set_title("Same-row ratio")
    axes[0, 1].plot(x, [r["same_token_ratio"] for r in block_reports])
    axes[0, 1].set_title("Same-token ratio")
    axes[1, 0].plot(x, [r["match_count"] for r in block_reports])
    axes[1, 0].set_title("Reciprocal NN matches")
    axes[1, 1].plot(
        x,
        [r.get("estimated_rotation_degrees", np.nan) for r in block_reports],
    )
    axes[1, 1].axhline(45.0, color="#7570b3", linestyle="--")
    axes[1, 1].set_title("Recovered rotation, degrees")
    for axis in axes.flat:
        axis.grid(alpha=0.25)
        axis.set_xlabel("Qwen2.5-VL vision block")
    figure.tight_layout()
    figure.savefig(args.output_dir / "block-metrics.png", dpi=160)
    plt.close(figure)

    summary = {
        "capture_manifest": str(manifest_path.resolve()),
        "min_similarity": args.min_similarity,
        "max_matches": args.max_matches,
        "ransac_threshold": args.ransac_threshold,
        "mask": mask_description,
        "least_row_locked_blocks": [r["name"] for r in ranked[:10]],
        "reports": reports,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"Complete: {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
