#!/usr/bin/env python3
"""Probe whether Qwen image readouts encode the rendered camera azimuth.

The probe deliberately receives no camera vector, ray map, view identifier, or
prompt text.  Its input is only the spatial mean of one captured target-token
readout block.  A prompt-conflict cache can therefore distinguish camera/image
evidence from merely decoding the requested angle from the prompt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Canonical identity experiment JSON")
    parser.add_argument(
        "--experiment-root",
        type=Path,
        required=True,
        help="Root containing features/<identity> for identities without explicit paths",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--generated-cache",
        type=Path,
        action="append",
        default=[],
        help="Generated Qwen feature cache to inspect; repeat as needed",
    )
    parser.add_argument(
        "--conflict-cache",
        type=Path,
        action="append",
        default=[],
        help="Exact cache whose semantic prompt angle differs from actual geometry",
    )
    parser.add_argument("--elevation", type=float, default=0.0)
    parser.add_argument(
        "--target",
        choices=("residual", "absolute"),
        default="residual",
        help="Predict actual-minus-prompt camera error or absolute actual azimuth",
    )
    parser.add_argument(
        "--pool-grid",
        type=int,
        default=4,
        help="Preserve a grid of spatially pooled readout cells per block",
    )
    parser.add_argument(
        "--ridge",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0, 100.0, 1000.0],
        help="Candidate dual-ridge regularization values",
    )
    return parser.parse_args()


def circular_delta_degrees(predicted: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Signed shortest delta predicted-target in [-180, 180)."""
    return (np.asarray(predicted) - np.asarray(target) + 180.0) % 360.0 - 180.0


def angle_targets(angles: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(np.asarray(angles, dtype=np.float64))
    return np.stack((np.sin(radians), np.cos(radians)), axis=1)


def vectors_to_angles(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    return np.rad2deg(np.arctan2(vectors[:, 0], vectors[:, 1])) % 360.0


@dataclass
class RidgeModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    train_scaled: np.ndarray
    dual_weights: np.ndarray
    target_mean: np.ndarray

    def predict_vectors(self, features: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(features, dtype=np.float64) - self.feature_mean) / self.feature_scale
        return (scaled @ self.train_scaled.T) @ self.dual_weights + self.target_mean

    def predict_angles(self, features: np.ndarray) -> np.ndarray:
        return vectors_to_angles(self.predict_vectors(features))


def fit_dual_ridge(features: np.ndarray, targets: np.ndarray, ridge: float) -> RidgeModel:
    features = np.asarray(features, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale = np.maximum(feature_scale, 1e-6)
    train_scaled = (features - feature_mean) / feature_scale
    target_mean = targets.mean(axis=0)
    centered_targets = targets - target_mean
    gram = train_scaled @ train_scaled.T
    gram.flat[:: gram.shape[0] + 1] += float(ridge)
    dual_weights = np.linalg.solve(gram, centered_targets)
    return RidgeModel(feature_mean, feature_scale, train_scaled, dual_weights, target_mean)


def _cache_key(feature_root: Path) -> str:
    return hashlib.sha256(str(feature_root.resolve()).encode("utf-8")).hexdigest()[:12]


def _descriptor_cache_path(
    output_dir: Path, label: str, feature_root: Path, pool_grid: int
) -> Path:
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in label)
    return output_dir / "descriptors" / (
        f"{safe_label}-{_cache_key(feature_root)}-grid{pool_grid}.npz"
    )


def extract_descriptors(
    feature_root: Path,
    label: str,
    output_dir: Path,
    elevation: float,
    pool_grid: int,
) -> dict[str, np.ndarray]:
    cache_path = _descriptor_cache_path(output_dir, label, feature_root, pool_grid)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cached:
            return {key: cached[key] for key in cached.files}

    manifest_path = feature_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    blocks = np.asarray(manifest["blocks"], dtype=np.int64)
    descriptors: list[np.ndarray] = []
    actual_angles: list[float] = []
    prompt_angles: list[float] = []
    steps: list[int] = []
    view_ids: list[str] = []

    for view in manifest["views"]:
        if abs(float(view["elevation_degrees"]) - elevation) > 1e-4:
            continue
        actual_angle = float(view["azimuth_degrees"]) % 360.0
        prompt_angle = float(view.get("prompt_azimuth_degrees", actual_angle)) % 360.0
        for sample in view["samples"]:
            source = np.load(feature_root / sample["features"], mmap_mode="r")
            if source.ndim != 4 or source.shape[0] != len(blocks):
                raise ValueError(f"Unexpected feature shape {source.shape} in {sample['features']}")
            if source.shape[1] % pool_grid or source.shape[2] % pool_grid:
                raise ValueError(
                    f"Feature grid {source.shape[1:3]} is not divisible by {pool_grid}"
                )
            pooled = np.stack(
                [
                    np.asarray(source[index], dtype=np.float32)
                    .reshape(
                        pool_grid,
                        source.shape[1] // pool_grid,
                        pool_grid,
                        source.shape[2] // pool_grid,
                        source.shape[3],
                    )
                    .mean(axis=(1, 3))
                    .reshape(-1)
                    for index in range(len(blocks))
                ]
            )
            descriptors.append(pooled)
            actual_angles.append(actual_angle)
            prompt_angles.append(prompt_angle)
            steps.append(int(sample["step"]))
            view_ids.append(str(view["id"]))

    if not descriptors:
        raise ValueError(f"No elevation {elevation:g} views in {manifest_path}")
    result = {
        "descriptors": np.asarray(descriptors, dtype=np.float32),
        "actual_angles": np.asarray(actual_angles, dtype=np.float32),
        "prompt_angles": np.asarray(prompt_angles, dtype=np.float32),
        "steps": np.asarray(steps, dtype=np.int64),
        "blocks": blocks,
        "view_ids": np.asarray(view_ids),
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **result)
    return result


def _identity_feature_root(identity: dict[str, Any], experiment_root: Path) -> Path:
    explicit = identity.get("features")
    return Path(explicit) if explicit else experiment_root / "features" / identity["id"]


def _combine(records: Iterable[tuple[str, dict[str, np.ndarray]]]) -> dict[str, np.ndarray]:
    items = list(records)
    if not items:
        raise ValueError("Cannot combine an empty record list")
    combined = {
        key: np.concatenate([record[key] for _, record in items], axis=0)
        for key in ("descriptors", "actual_angles", "prompt_angles", "steps", "view_ids")
    }
    combined["identities"] = np.concatenate(
        [np.repeat(identity, len(record["steps"])) for identity, record in items]
    )
    combined["blocks"] = items[0][1]["blocks"]
    return combined


def _block_index(data: dict[str, np.ndarray], block: int) -> int:
    matches = np.flatnonzero(data["blocks"] == int(block))
    if len(matches) != 1:
        raise ValueError(f"Expected block {block} once, found {matches.tolist()}")
    return int(matches[0])


def _target_angles(data: dict[str, np.ndarray], target: str) -> np.ndarray:
    if target == "absolute":
        return data["actual_angles"]
    return circular_delta_degrees(data["actual_angles"], data["prompt_angles"])


def _subset(
    data: dict[str, np.ndarray], block_index: int, step: int, target: str
) -> tuple[np.ndarray, np.ndarray]:
    mask = data["steps"] == step
    return data["descriptors"][mask, block_index], _target_angles(data, target)[mask]


def _metrics(predicted: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    errors = np.abs(circular_delta_degrees(predicted, actual))
    predicted_signed = circular_delta_degrees(predicted, 0.0)
    actual_signed = circular_delta_degrees(actual, 0.0)
    correlation = (
        float(np.corrcoef(predicted_signed, actual_signed)[0, 1])
        if len(actual_signed) > 1
        and np.std(predicted_signed) > 1e-9
        and np.std(actual_signed) > 1e-9
        else 0.0
    )
    return {
        "mae_degrees": float(errors.mean()),
        "median_degrees": float(np.median(errors)),
        "within_22_5_percent": float((errors <= 22.5).mean() * 100.0),
        "within_45_percent": float((errors <= 45.0).mean() * 100.0),
        "sign_accuracy_percent": float(
            (np.sign(predicted_signed) == np.sign(actual_signed)).mean() * 100.0
        ),
        "pearson_correlation": correlation,
    }


def _evaluate(
    model: RidgeModel,
    data: dict[str, np.ndarray],
    block_index: int,
    step: int,
    target: str,
) -> dict[str, float]:
    features, labels = _subset(data, block_index, step, target)
    return _metrics(model.predict_angles(features), labels)


def _prediction_rows(
    model: RidgeModel,
    data: dict[str, np.ndarray],
    block_index: int,
    step: int,
    target: str,
) -> list[dict[str, Any]]:
    mask = data["steps"] == step
    predicted = model.predict_angles(data["descriptors"][mask, block_index])
    rows = []
    for index, prediction in enumerate(predicted):
        actual = float(data["actual_angles"][mask][index])
        prompt = float(data["prompt_angles"][mask][index])
        true_residual = float(circular_delta_degrees(actual, prompt))
        predicted_target = (
            float(circular_delta_degrees(prediction, 0.0))
            if target == "residual"
            else float(prediction)
        )
        target_value = true_residual if target == "residual" else actual
        rows.append(
            {
                "identity": str(data["identities"][mask][index]),
                "view": str(data["view_ids"][mask][index]),
                "step": int(step),
                "actual_or_nominal_degrees": actual,
                "prompt_degrees": prompt,
                "true_residual_degrees": true_residual,
                "predicted_target_degrees": predicted_target,
                "target_error_degrees": float(
                    circular_delta_degrees(predicted_target, target_value)
                ),
            }
        )
    return rows


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    selected = report["selected"]
    lines = [
        "# Qwen camera readout probe",
        "",
        f"The probe predicts `{report['target']}` from a {report['pool_grid']}×{report['pool_grid']} spatially pooled target-token readout. It does not receive camera FiLM, ray maps, view IDs, or prompt text.",
        "",
        "## Selected probe",
        "",
        f"- Block: `{selected['block']}`",
        f"- Denoising step: `{selected['step']}`",
        f"- Ridge: `{selected['ridge']}`",
        f"- Validation target MAE: `{selected['validation']['mae_degrees']:.2f}°`",
        f"- Zero-correction validation MAE: `{report['zero_correction_baseline']['mae_degrees']:.2f}°`",
        f"- Validation sign accuracy: `{selected['validation']['sign_accuracy_percent']:.1f}%`",
        f"- Validation correlation: `{selected['validation']['pearson_correlation']:.3f}`",
        f"- Test target MAE: `{selected['test']['mae_degrees']:.2f}°`" if selected.get("test") else "- Test split: unavailable",
        "",
        "## Candidate matrix",
        "",
        "| block | step | ridge | train MAE | validation MAE |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["candidates"]:
        lines.append(
            f"| {row['block']} | {row['step']} | {row['ridge']:g} | "
            f"{row['train']['mae_degrees']:.2f}° | {row['validation']['mae_degrees']:.2f}° |"
        )
    if report["conflicts"]:
        lines.extend(
            [
                "",
                "## Additional controls",
                "",
                "| cache | target MAE |",
                "| --- | ---: |",
            ]
        )
        for conflict in report["conflicts"]:
            lines.append(
                f"| {conflict['name']} | {conflict['target_mae_degrees']:.2f}° |"
            )
    if report["generated"]:
        lines.extend(
            [
                "",
                "## Generated rings",
                "",
                "These have no geometric ground truth; values are predicted camera corrections relative to the requested view.",
                "",
                "| cache | mean absolute correction | signed mean correction |",
                "| --- | ---: | ---: |",
            ]
        )
        for generated in report["generated"]:
            lines.append(
                f"| {generated['name']} | {generated['mean_absolute_predicted_target_degrees']:.2f}° | "
                f"{generated['signed_mean_predicted_target_degrees']:.2f}° |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[tuple[str, dict[str, np.ndarray]]]] = {
        "train": [],
        "validation": [],
        "test": [],
    }
    for identity in config["identities"]:
        split = identity["split"]
        if split not in by_split:
            continue
        root = _identity_feature_root(identity, args.experiment_root)
        print(f"Pooling {identity['id']} ({split}) from {root}", flush=True)
        record = extract_descriptors(
            root, identity["id"], args.output_dir, args.elevation, args.pool_grid
        )
        by_split[split].append((identity["id"], record))

    train = _combine(by_split["train"])
    validation = _combine(by_split["validation"])
    test = _combine(by_split["test"]) if by_split["test"] else None
    if not np.array_equal(train["blocks"], validation["blocks"]):
        raise ValueError("Train and validation caches use different block lists")

    candidates: list[dict[str, Any]] = []
    common_steps = sorted(set(train["steps"].tolist()) & set(validation["steps"].tolist()))
    for block_index, block in enumerate(train["blocks"].tolist()):
        for step in common_steps:
            train_features, train_angles = _subset(
                train, block_index, step, args.target
            )
            targets = angle_targets(train_angles)
            for ridge in args.ridge:
                model = fit_dual_ridge(train_features, targets, ridge)
                candidates.append(
                    {
                        "block": int(block),
                        "block_index": block_index,
                        "step": int(step),
                        "ridge": float(ridge),
                        "train": _evaluate(
                            model, train, block_index, step, args.target
                        ),
                        "validation": _evaluate(
                            model, validation, block_index, step, args.target
                        ),
                    }
                )
    candidates.sort(key=lambda item: item["validation"]["mae_degrees"])
    best = candidates[0]
    block_index = int(best["block_index"])
    selected_train_features, selected_train_angles = _subset(
        train, block_index, best["step"], args.target
    )
    selected_model = fit_dual_ridge(
        selected_train_features,
        angle_targets(selected_train_angles),
        best["ridge"],
    )
    selected = {
        **{key: value for key, value in best.items() if key != "block_index"},
        "test": (
            _evaluate(
                selected_model, test, block_index, best["step"], args.target
            )
            if test
            else None
        ),
    }
    validation_predictions = _prediction_rows(
        selected_model,
        validation,
        block_index,
        best["step"],
        args.target,
    )
    validation_by_identity = {}
    for identity in sorted({row["identity"] for row in validation_predictions}):
        rows = [row for row in validation_predictions if row["identity"] == identity]
        predicted = np.asarray([row["predicted_target_degrees"] for row in rows])
        actual = np.asarray(
            [
                row["true_residual_degrees"]
                if args.target == "residual"
                else row["actual_or_nominal_degrees"]
                for row in rows
            ]
        )
        validation_by_identity[identity] = _metrics(predicted, actual)

    artifact_path = args.output_dir / "selected-probe.npz"
    np.savez_compressed(
        artifact_path,
        block=np.asarray(best["block"]),
        step=np.asarray(best["step"]),
        ridge=np.asarray(best["ridge"]),
        feature_mean=selected_model.feature_mean,
        feature_scale=selected_model.feature_scale,
        train_scaled=selected_model.train_scaled,
        dual_weights=selected_model.dual_weights,
        target_mean=selected_model.target_mean,
    )

    conflicts = []
    for conflict_root in args.conflict_cache:
        record = extract_descriptors(
            conflict_root,
            conflict_root.name,
            args.output_dir,
            args.elevation,
            args.pool_grid,
        )
        data = _combine([(conflict_root.name, record)])
        data_block_index = _block_index(data, best["block"])
        rows = _prediction_rows(
            selected_model, data, data_block_index, best["step"], args.target
        )
        conflicts.append(
            {
                "name": conflict_root.name,
                "path": str(conflict_root),
                "target_mae_degrees": float(
                    np.mean(np.abs([row["target_error_degrees"] for row in rows]))
                ),
                "predictions": rows,
            }
        )

    generated = []
    for generated_root in args.generated_cache:
        record = extract_descriptors(
            generated_root,
            generated_root.name,
            args.output_dir,
            args.elevation,
            args.pool_grid,
        )
        data = _combine([(generated_root.name, record)])
        data_block_index = _block_index(data, best["block"])
        rows = _prediction_rows(
            selected_model, data, data_block_index, best["step"], args.target
        )
        predictions = np.asarray(
            [row["predicted_target_degrees"] for row in rows]
        )
        generated.append(
            {
                "name": generated_root.name,
                "path": str(generated_root),
                "mean_absolute_predicted_target_degrees": float(
                    np.mean(np.abs(predictions))
                ),
                "signed_mean_predicted_target_degrees": float(
                    np.mean(predictions)
                ),
                "predictions": rows,
            }
        )

    report = {
        "schema": "diffusion-editor.qwen-camera-readout-probe",
        "schema_version": 1,
        "config": str(args.config),
        "elevation_degrees": args.elevation,
        "target": args.target,
        "pool_grid": args.pool_grid,
        "input": "spatially pooled target-token readout only",
        "excluded_inputs": ["camera FiLM", "ray map", "view identifier", "prompt text"],
        "selection_split": "validation identities",
        "zero_correction_baseline": _metrics(
            np.zeros_like(_target_angles(validation, args.target)),
            _target_angles(validation, args.target),
        ),
        "selected": selected,
        "validation_by_identity": validation_by_identity,
        "validation_predictions": validation_predictions,
        "candidates": [{key: value for key, value in row.items() if key != "block_index"} for row in candidates],
        "conflicts": conflicts,
        "generated": generated,
        "artifact": str(artifact_path),
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(report, args.output_dir / "report.md")
    print(json.dumps(selected, indent=2), flush=True)
    print(f"Wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
