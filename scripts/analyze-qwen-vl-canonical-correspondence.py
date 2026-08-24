#!/usr/bin/env python3
"""Measure whether Qwen2.5-VL visual tokens encode canonical correspondence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--sources", required=True, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--matchable-distance", type=float, default=0.075)
    return parser.parse_args()


def _normalize(features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    return values / np.maximum(
        np.linalg.norm(values, axis=-1, keepdims=True), 1.0e-8
    )


def _geometry_at_grid(path: Path, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as archive:
        xyz = np.asarray(archive["canonical_xyz"], dtype=np.float32)
        mask = np.asarray(archive["mask"], dtype=np.uint8)
    resized_xyz = cv2.resize(xyz, grid, interpolation=cv2.INTER_NEAREST)
    resized_mask = cv2.resize(mask, grid, interpolation=cv2.INTER_NEAREST) != 0
    return resized_xyz, resized_mask


def _nearest_errors(
    target_features: np.ndarray,
    target_xyz: np.ndarray,
    source_features: np.ndarray,
    source_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    scores = _normalize(target_features) @ _normalize(source_features).T
    selected = np.argmax(scores, axis=1)
    errors = np.linalg.norm(target_xyz - source_xyz[selected], axis=1)
    similarities = scores[np.arange(len(selected)), selected]
    return errors, similarities


def _oracle_errors(target_xyz: np.ndarray, source_xyz: np.ndarray) -> np.ndarray:
    distances = np.linalg.norm(
        target_xyz[:, None, :] - source_xyz[None, :, :], axis=2
    )
    return distances.min(axis=1)


def _metrics(errors: np.ndarray, similarities: np.ndarray, matchable: np.ndarray) -> dict:
    selected = errors[matchable]
    selected_similarity = similarities[matchable]
    result = {
        "target_tokens": int(len(errors)),
        "matchable_tokens": int(matchable.sum()),
        "matchable_fraction": float(matchable.mean()),
    }
    if not len(selected):
        return result
    result.update(
        {
            "canonical_error_median": float(np.median(selected)),
            "canonical_error_p95": float(np.quantile(selected, 0.95)),
            "within_2_5_percent": float(np.mean(selected <= 0.025)),
            "within_5_percent": float(np.mean(selected <= 0.05)),
            "within_10_percent": float(np.mean(selected <= 0.10)),
            "similarity_median": float(np.median(selected_similarity)),
        }
    )
    return result


def main() -> int:
    args = _arguments()
    capture = json.loads(
        (args.capture_dir / "manifest.json").read_text(encoding="utf-8")
    )
    dataset = json.loads(
        (args.dataset / "manifest.json").read_text(encoding="utf-8")
    )
    captured = {view["label"]: view for view in capture["views"]}
    dataset_views = {view["id"]: view for view in dataset["views"]}
    requested = [args.target, *args.sources]
    missing_capture = sorted(set(requested) - captured.keys())
    missing_dataset = sorted(set(requested) - dataset_views.keys())
    if missing_capture or missing_dataset:
        raise SystemExit(
            f"missing captured={missing_capture}, dataset={missing_dataset}"
        )
    grid = tuple(int(value) for value in captured[args.target]["feature_grid"])
    if any(tuple(captured[label]["feature_grid"]) != grid for label in requested):
        raise SystemExit("all feature grids must match")

    geometry = {}
    for label in requested:
        record = dataset_views[label]
        geometry[label] = _geometry_at_grid(
            args.dataset / record["geometry"], grid
        )
    target_xyz_map, target_mask = geometry[args.target]
    target_xyz = target_xyz_map[target_mask]
    source_xyz = np.concatenate(
        [geometry[label][0][geometry[label][1]] for label in args.sources]
    )
    oracle = _oracle_errors(target_xyz, source_xyz)
    matchable = oracle <= args.matchable_distance
    if not np.any(matchable):
        raise SystemExit("no target tokens are geometrically visible in sources")

    reports = []

    def evaluate(name: str, key: str, index: int | None = None) -> None:
        target_values = np.load(captured[args.target][key], mmap_mode="r")
        source_values = [
            np.load(captured[label][key], mmap_mode="r")
            for label in args.sources
        ]
        if index is not None:
            target_values = target_values[index]
            source_values = [values[index] for values in source_values]
        target_features = np.asarray(target_values[target_mask])
        source_features = np.concatenate(
            [
                np.asarray(values[geometry[label][1]])
                for label, values in zip(args.sources, source_values)
            ]
        )
        errors, similarities = _nearest_errors(
            target_features, target_xyz, source_features, source_xyz
        )
        report = {"name": name, **_metrics(errors, similarities, matchable)}
        reports.append(report)
        print(
            f"{name:12s}: matchable={report['matchable_tokens']:3d}/"
            f"{report['target_tokens']:3d} median="
            f"{report.get('canonical_error_median', float('nan')):.4f} "
            f"p95={report.get('canonical_error_p95', float('nan')):.4f} "
            f"within5={report.get('within_5_percent', 0.0):.3f}",
            flush=True,
        )

    evaluate("patch", "patch")
    for index in capture["blocks"]:
        evaluate(f"block-{index:02d}", "blocks", index)
    for slot, index in enumerate(capture["global_attention_blocks"]):
        evaluate(f"q-{index:02d}", "q_global", slot)
        evaluate(f"k-{index:02d}", "k_global", slot)

    ranked = sorted(
        reports,
        key=lambda report: report.get("canonical_error_median", float("inf")),
    )
    summary = {
        "schema": "diffusion-editor.qwen-vl-canonical-correspondence",
        "capture": str((args.capture_dir / "manifest.json").resolve()),
        "dataset": str((args.dataset / "manifest.json").resolve()),
        "target": args.target,
        "sources": args.sources,
        "feature_grid": list(grid),
        "matchable_distance": args.matchable_distance,
        "oracle_error_median": float(np.median(oracle)),
        "oracle_error_p95": float(np.quantile(oracle, 0.95)),
        "ranking": [report["name"] for report in ranked],
        "reports": reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Complete: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
