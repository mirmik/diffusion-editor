#!/usr/bin/env python3
"""Replay cached generated Qwen features through canonical-head checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from diffusion_editor.training.canonical_pointmap import (  # noqa: E402
    canonical_camera_ray_map,
    normalized_camera_vector,
)
from diffusion_editor.training.generated_domain_gate import (  # noqa: E402
    rank_generated_domain_candidates,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "feature_caches",
        nargs="+",
        type=Path,
        help="Generated Qwen feature-cache directories for validation identities.",
    )
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Repeat in the predeclared candidate order.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--voxel-size", type=float, default=0.008)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.voxel_size <= 0.0:
        parser.error("--voxel-size must be positive")
    return args


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_checkpoints(values: list[str]) -> list[tuple[str, Path]]:
    parsed = []
    labels = set()
    for value in values:
        if "=" not in value:
            raise SystemExit(f"checkpoint must be LABEL=PATH: {value}")
        label, raw_path = value.split("=", 1)
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise SystemExit(f"unsafe checkpoint label: {label}")
        if label in labels:
            raise SystemExit(f"duplicate checkpoint label: {label}")
        labels.add(label)
        parsed.append((label, Path(raw_path).resolve()))
    return parsed


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dataset_identity(dataset_manifest: dict, fallback: str) -> str:
    asset = dataset_manifest.get("asset", {})
    return str(asset.get("identity_id") or asset.get("name") or fallback)


def _reference_surface(dataset_root: Path, dataset_manifest: dict) -> np.ndarray:
    points = []
    for view in dataset_manifest["views"]:
        with np.load(dataset_root / view["geometry"]) as geometry:
            mask = np.asarray(geometry["mask"], dtype=bool)
            points.append(
                np.asarray(geometry["canonical_xyz"], dtype=np.float32)[mask]
            )
    return np.concatenate(points)


def _image_foreground_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as source:
        rgb = np.asarray(
            source.convert("RGB").resize(size, Image.Resampling.LANCZOS)
        )
    return np.max(rgb, axis=2) > 12


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


def _projective_reprojection(points: np.ndarray, pixels: np.ndarray) -> dict:
    if len(points) < 12:
        return {"valid": False}
    center = points.mean(axis=0)
    scale = np.maximum(points.std(axis=0), 1.0e-4)
    homogeneous = np.column_stack(((points - center) / scale, np.ones(len(points))))
    indices = (
        np.linspace(0, len(points) - 1, 1024, dtype=np.int64)
        if len(points) > 1024
        else np.arange(len(points))
    )
    fit_points = homogeneous[indices]
    fit_pixels = pixels[indices]
    zeros = np.zeros_like(fit_points)
    matrix = np.concatenate(
        (
            np.concatenate((fit_points, zeros, -fit_pixels[:, :1] * fit_points), axis=1),
            np.concatenate((zeros, fit_points, -fit_pixels[:, 1:] * fit_points), axis=1),
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
    residual = np.linalg.norm(
        projected[valid, :2] / denominator[valid, None] - pixels[valid], axis=1
    )
    return {
        "valid": True,
        "median": float(np.median(residual)),
        "p95": float(np.quantile(residual, 0.95)),
    }


def _fuse_predictions(predictions: list[np.ndarray]) -> np.ndarray:
    stacked = np.stack(predictions).astype(np.float32, copy=False)
    mask_probability = 1.0 / (1.0 + np.exp(-np.clip(stacked[:, 3], -20.0, 20.0)))
    scores = np.log(mask_probability + 1.0e-6) - np.clip(stacked[:, 4], -9.0, 2.0)
    scores -= np.max(scores, axis=0, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=0, keepdims=True).clip(1.0e-6)
    fused = np.empty_like(stacked[0])
    fused[:3] = np.sum(weights[:, None] * stacked[:, :3], axis=0)
    fused[3] = np.mean(stacked[:, 3], axis=0)
    fused[4] = np.sum(weights * stacked[:, 4], axis=0)
    return fused


def _camera_inputs(camera: dict, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = grid
    vector = normalized_camera_vector(
        camera_from_canonical=camera["camera_from_canonical"],
        intrinsics=camera["intrinsics"],
        image_size=camera["image_size"],
    )
    rays = canonical_camera_ray_map(
        camera_from_canonical=camera["camera_from_canonical"],
        intrinsics=camera["intrinsics"],
        image_size=camera["image_size"],
        grid_height=height,
        grid_width=width,
    )
    return vector, rays


def _identity_metrics(
    predictions: list[np.ndarray],
    image_paths: list[Path],
    view_indices: list[int],
    reference: np.ndarray,
    voxel_size: float,
) -> dict:
    point_sets = []
    pixel_sets = []
    image_ious = []
    point_view_indices = []
    for prediction, image_path, view_index in zip(
        predictions, image_paths, view_indices
    ):
        points, pixels = _points_and_pixels(prediction)
        predicted_mask = prediction[3] > 0.0
        image_mask = _image_foreground_mask(
            image_path, (predicted_mask.shape[1], predicted_mask.shape[0])
        )
        union = np.logical_or(predicted_mask, image_mask).sum()
        image_ious.append(
            float(np.logical_and(predicted_mask, image_mask).sum() / max(union, 1))
        )
        point_sets.append(points)
        pixel_sets.append(pixels)
        point_view_indices.append(np.full(len(points), view_index, dtype=np.int16))
    cloud = np.concatenate(point_sets)
    cloud_views = np.concatenate(point_view_indices)
    reference_tree = cKDTree(reference)
    cloud_tree = cKDTree(cloud)
    point_to_surface = reference_tree.query(cloud, workers=-1)[0]
    surface_to_cloud = cloud_tree.query(reference, workers=-1)[0]
    projective = [
        _projective_reprojection(points, pixels)
        for points, pixels in zip(point_sets, pixel_sets)
    ]
    valid_projective = [item for item in projective if item["valid"]]
    voxels = np.floor(cloud / voxel_size).astype(np.int32)
    _keys, inverse = np.unique(voxels, axis=0, return_inverse=True)
    pairs = np.unique(np.column_stack((inverse, cloud_views)), axis=0)
    observations = np.bincount(pairs[:, 0], minlength=len(_keys))
    return {
        "surface_distance_median": float(np.median(point_to_surface)),
        "surface_distance_p95": float(np.quantile(point_to_surface, 0.95)),
        "surface_coverage_p95": float(np.quantile(surface_to_cloud, 0.95)),
        "projective_reprojection_median_mean": float(
            np.mean([item["median"] for item in valid_projective])
        ),
        "projective_reprojection_p95_max": float(
            max(item["p95"] for item in valid_projective)
        ),
        "image_foreground_iou_mean": float(np.mean(image_ious)),
        "voxel_fraction_at_least_2_views": float(np.mean(observations >= 2)),
        "point_count": int(len(cloud)),
        "voxel_count": int(len(_keys)),
        "projective_valid_views": len(valid_projective),
    }


def _evaluate_identity(
    torch,
    head,
    checkpoint: dict,
    cache_root: Path,
    cache: dict,
    output: Path,
    voxel_size: float,
    checkpoint_record: dict,
) -> dict:
    dataset_root = Path(cache["dataset"])
    dataset_manifest = _load_json(dataset_root / "manifest.json")
    dataset_views = {view["id"]: view for view in dataset_manifest["views"]}
    reference = _reference_surface(dataset_root, dataset_manifest)
    target_resolution = int(checkpoint["target_resolution"])
    predictions = []
    image_paths = []
    output_views = []
    started = time.monotonic()
    for index, view in enumerate(cache["views"]):
        dataset_view = dataset_views[view["id"]]
        camera_record = _load_json(dataset_root / dataset_view["camera"])
        samples = list(view["samples"])
        grid_height, grid_width = map(int, samples[0]["shape"][1:3])
        camera_vector, rays = _camera_inputs(
            camera_record, (grid_height, grid_width)
        )
        camera = torch.from_numpy(camera_vector).unsqueeze(0).to("cuda")
        ray_tensor = (
            torch.from_numpy(rays).permute(2, 0, 1).unsqueeze(0).to("cuda")
        )
        step_predictions = []
        for sample in samples:
            feature_array = np.load(cache_root / sample["features"], mmap_mode="r")
            features = (
                torch.from_numpy(np.array(feature_array, copy=True))
                .permute(0, 3, 1, 2)
                .unsqueeze(0)
                .to("cuda", dtype=torch.float32)
            )
            timestep = torch.tensor(
                [float(sample["normalized_timestep"])],
                device="cuda",
                dtype=torch.float32,
            )
            with torch.inference_mode():
                prediction = head(
                    features,
                    camera,
                    ray_tensor,
                    target_resolution,
                    timestep=timestep,
                )[0].float().cpu().numpy()
            step_predictions.append(prediction)
        fused = _fuse_predictions(step_predictions)
        predictions.append(fused)
        image_paths.append(Path(view["image"]))
        output_views.append(
            {
                "id": view["id"],
                "image": str(Path(view["image"]).resolve()),
                "steps": len(samples),
            }
        )
        print(f"  [{index + 1}/{len(cache['views'])}] {view['id']}", flush=True)
    metrics = _identity_metrics(
        predictions,
        image_paths,
        list(range(len(predictions))),
        reference,
        voxel_size,
    )
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "diffusion-editor.generated-domain-checkpoint-evaluation",
        "schema_version": 1,
        "feature_cache": str(cache_root.resolve()),
        "dataset": str(dataset_root),
        "checkpoint": checkpoint_record,
        "fusion": "confidence_all",
        "voxel_size": voxel_size,
        "views": output_views,
        "metrics": metrics,
        "elapsed_seconds": time.monotonic() - started,
    }
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> int:
    args = _arguments()
    import torch
    from diffusion_editor.training.canonical_pointmap_head import (
        build_canonical_pointmap_head,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("generated-domain checkpoint evaluation requires CUDA")
    checkpoints = _parse_checkpoints(args.checkpoint)
    caches = []
    identities = set()
    for root in args.feature_caches:
        root = root.resolve()
        cache = _load_json(root / "manifest.json")
        if cache.get("schema") != "diffusion-editor.qwen-generated-canonical-features":
            raise SystemExit(f"unsupported generated feature cache: {root}")
        dataset_manifest = _load_json(Path(cache["dataset"]) / "manifest.json")
        identity = _dataset_identity(dataset_manifest, root.name)
        if identity in identities:
            raise SystemExit(f"duplicate validation identity: {identity}")
        identities.add(identity)
        caches.append((identity, root, cache))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    started = time.monotonic()
    for candidate_index, (label, checkpoint_path) in enumerate(checkpoints, start=1):
        print(f"[{candidate_index}/{len(checkpoints)}] checkpoint {label}", flush=True)
        checkpoint_record = {
            "label": label,
            "path": str(checkpoint_path),
            "sha256": _sha256(checkpoint_path),
        }
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        blocks = [int(value) for value in checkpoint["blocks"]]
        head = build_canonical_pointmap_head(
            str(checkpoint.get("architecture", "local-v1")),
            len(blocks),
            int(checkpoint["feature_channels"]),
            int(checkpoint["projection_channels"]),
            int(checkpoint["hidden_channels"]),
            int(checkpoint.get("ray_channels", 0)),
            int(checkpoint.get("timestep_channels", 0)),
        )
        head.load_state_dict(checkpoint["model"])
        head.eval().to("cuda")
        results[label] = {}
        for identity, cache_root, cache in caches:
            if [int(value) for value in cache["blocks"]] != blocks:
                raise RuntimeError(f"{identity}: cache/checkpoint block mismatch")
            output = args.output_dir / "evaluations" / label / identity
            report_path = output / "metrics.json"
            if report_path.exists() and not args.force:
                print(f"  resume {identity}: {report_path}", flush=True)
                report = _load_json(report_path)
                if report.get("checkpoint") != checkpoint_record:
                    raise RuntimeError(
                        f"stale resumed evaluation for {label}/{identity}; "
                        "remove it or pass --force"
                    )
            else:
                print(f"  evaluate {identity}", flush=True)
                report = _evaluate_identity(
                    torch,
                    head,
                    checkpoint,
                    cache_root,
                    cache,
                    output,
                    args.voxel_size,
                    checkpoint_record,
                )
            results[label][identity] = report["metrics"]
        head.to("cpu")
        torch.cuda.empty_cache()
    ranking = rank_generated_domain_candidates(
        results, candidate_order=[label for label, _path in checkpoints]
    )
    gate = {
        "schema": "diffusion-editor.generated-domain-checkpoint-gate",
        "schema_version": 1,
        "validation_identities": [identity for identity, _root, _cache in caches],
        "final_test_identity": "Termin (excluded from this evaluation)",
        "seed": [int(cache["seed"]) for _identity, _root, cache in caches],
        "checkpoints": [
            {"label": label, "path": str(path), "sha256": _sha256(path)}
            for label, path in checkpoints
        ],
        "metrics": results,
        "ranking": ranking,
        "elapsed_seconds": time.monotonic() - started,
    }
    gate_path = args.output_dir / "gate.json"
    gate_path.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ranking, indent=2), flush=True)
    print(f"Complete: {gate_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
