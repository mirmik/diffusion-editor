#!/usr/bin/env python3
"""Run a trained canonical point-map head inside real Qwen denoising."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from diffusion_editor.generation.image_edit_profiles import (  # noqa: E402
    QWEN_IMAGE_EDIT_PROFILE_ID,
    image_edit_profile,
    qwen_multiple_angles_lora_adapter,
)
from diffusion_editor.training.canonical_pointmap import (  # noqa: E402
    canonical_camera_ray_map,
    normalized_camera_vector,
)
from diffusion_editor.training.canonical_pointmap_head import (  # noqa: E402
    build_canonical_pointmap_head,
)
from diffusion_editor.workers.ml_backend import RealMlBackend  # noqa: E402


AZIMUTHS = {
    0: "front view",
    45: "front-right quarter view",
    90: "right side view",
    135: "back-right quarter view",
    180: "back view",
    225: "back-left quarter view",
    270: "left side view",
    315: "front-left quarter view",
}
ELEVATIONS = {
    "low": (-30, "low-angle shot"),
    "eye": (0, "eye-level shot"),
    "elevated": (30, "elevated shot"),
}
FUSION_NAMES = ("uniform_all", "confidence_all")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--elevation",
        action="append",
        choices=tuple(ELEVATIONS),
        dest="elevations",
        help="Repeat to select rings; defaults to eye level.",
    )
    parser.add_argument(
        "--azimuth",
        action="append",
        type=int,
        choices=tuple(AZIMUTHS),
        dest="azimuths",
        help="Repeat to select azimuths; defaults to the complete ring.",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument(
        "--feature-cache-dir",
        type=Path,
        help=(
            "Persist the generated conditional Qwen block features as float16 so "
            "other head checkpoints can be evaluated without regenerating views."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    return args


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rgb(path: Path) -> Image.Image:
    with Image.open(path) as source:
        if source.mode != "RGBA":
            return source.convert("RGB")
        background = Image.new("RGBA", source.size, (0, 0, 0, 255))
        background.alpha_composite(source)
        return background.convert("RGB")


def _camera_inputs(camera: dict, grid: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    grid_height, grid_width = grid
    vector = normalized_camera_vector(
        camera_from_canonical=camera["camera_from_canonical"],
        intrinsics=camera["intrinsics"],
        image_size=camera["image_size"],
    )
    rays = canonical_camera_ray_map(
        camera_from_canonical=camera["camera_from_canonical"],
        intrinsics=camera["intrinsics"],
        image_size=camera["image_size"],
        grid_height=grid_height,
        grid_width=grid_width,
    )
    return vector, rays


def _reference_points(dataset_root: Path, views: list[dict]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    all_points = []
    by_id = {}
    for view in views:
        with np.load(dataset_root / view["geometry"]) as geometry:
            mask = np.asarray(geometry["mask"], dtype=bool)
            points = np.asarray(geometry["canonical_xyz"], dtype=np.float32)[mask]
        by_id[view["id"]] = points
        all_points.append(points)
    return np.concatenate(all_points), by_id


def _safe_correlation(first: np.ndarray, second: np.ndarray) -> float:
    if len(first) < 2 or np.std(first) <= 1.0e-8 or np.std(second) <= 1.0e-8:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _write_prediction_diagnostics(
    output_dir: Path,
    identifier: str,
    step: int,
    prediction: np.ndarray,
) -> None:
    xyz = prediction[:3].transpose(1, 2, 0)
    mask = prediction[3] > 0.0
    encoded = np.round(np.clip(xyz + 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
    encoded[~mask] = 0
    Image.fromarray(encoded, "RGB").save(
        output_dir / f"{identifier}-step{step}-xyz.png"
    )
    mask_rgb = np.repeat(mask[:, :, None], 3, axis=2).astype(np.uint8) * 255
    Image.fromarray(mask_rgb, "RGB").save(
        output_dir / f"{identifier}-step{step}-mask.png"
    )


def _fuse_predictions(predictions: list[np.ndarray]) -> dict[str, np.ndarray]:
    stacked = np.stack(predictions).astype(np.float32, copy=False)
    uniform = np.mean(stacked, axis=0)

    mask_probability = 1.0 / (1.0 + np.exp(-np.clip(stacked[:, 3], -20.0, 20.0)))
    scores = np.log(mask_probability + 1.0e-6) - np.clip(
        stacked[:, 4], -9.0, 2.0
    )
    scores -= np.max(scores, axis=0, keepdims=True)
    weights = np.exp(scores)
    weights /= np.sum(weights, axis=0, keepdims=True).clip(1.0e-6)
    confidence = np.empty_like(uniform)
    confidence[:3] = np.sum(weights[:, None] * stacked[:, :3], axis=0)
    confidence[3] = np.mean(stacked[:, 3], axis=0)
    confidence[4] = np.sum(weights * stacked[:, 4], axis=0)
    return {
        "uniform_all": uniform,
        "confidence_all": confidence,
    }


def _prediction_metrics(
    prediction: np.ndarray,
    rays: np.ndarray,
    surface_tree: cKDTree,
    visible_reference: np.ndarray,
) -> dict:
    xyz = prediction[:3].transpose(1, 2, 0)
    mask = prediction[3] > 0.0
    log_error = np.clip(prediction[4], -9.0, 2.0)
    points = xyz[mask]
    if not len(points):
        return {
            "foreground_points": 0,
            "foreground_fraction": 0.0,
            "valid": False,
        }
    off_surface = surface_tree.query(points, workers=-1)[0]
    predicted_tree = cKDTree(points)
    coverage = predicted_tree.query(visible_reference, workers=-1)[0]
    origins = rays[..., :3][mask]
    directions = rays[..., 3:][mask]
    relative = points - origins
    ray_depth = np.sum(relative * directions, axis=1)
    perpendicular = np.linalg.norm(
        relative - ray_depth[:, None] * directions,
        axis=1,
    )
    uncertainty = log_error[mask]
    return {
        "foreground_points": int(len(points)),
        "foreground_fraction": float(mask.mean()),
        "valid": True,
        "surface_distance_median": float(np.median(off_surface)),
        "surface_distance_p90": float(np.quantile(off_surface, 0.90)),
        "surface_distance_p95": float(np.quantile(off_surface, 0.95)),
        "surface_within_5_percent": float(np.mean(off_surface <= 0.05)),
        "visible_coverage_median": float(np.median(coverage)),
        "visible_coverage_p95": float(np.quantile(coverage, 0.95)),
        "ray_distance_median": float(np.median(perpendicular)),
        "ray_distance_p95": float(np.quantile(perpendicular, 0.95)),
        "behind_camera_fraction": float(np.mean(ray_depth <= 0.0)),
        "uncertainty_mean": float(np.mean(uncertainty)),
        "uncertainty_surface_log_error_correlation": _safe_correlation(
            uncertainty,
            np.log(off_surface + 1.0e-6),
        ),
    }


def _summaries(evaluations: list[dict], steps: int) -> list[dict]:
    summaries = []
    for step in range(steps):
        values = [item["steps"][step] for item in evaluations]
        valid = [item for item in values if item["valid"]]
        summaries.append(
            {
                "step": step,
                "valid_views": len(valid),
                "foreground_fraction_mean": float(
                    np.mean([item["foreground_fraction"] for item in values])
                ),
                "surface_distance_median_mean": (
                    float(np.mean([item["surface_distance_median"] for item in valid]))
                    if valid
                    else None
                ),
                "surface_distance_p95_max": (
                    float(max(item["surface_distance_p95"] for item in valid))
                    if valid
                    else None
                ),
                "surface_within_5_percent_mean": (
                    float(np.mean([item["surface_within_5_percent"] for item in valid]))
                    if valid
                    else None
                ),
                "visible_coverage_p95_max": (
                    float(max(item["visible_coverage_p95"] for item in valid))
                    if valid
                    else None
                ),
                "ray_distance_median_mean": (
                    float(np.mean([item["ray_distance_median"] for item in valid]))
                    if valid
                    else None
                ),
                "ray_distance_p95_max": (
                    float(max(item["ray_distance_p95"] for item in valid))
                    if valid
                    else None
                ),
                "uncertainty_surface_log_error_correlation_mean": (
                    float(
                        np.mean(
                            [
                                item["uncertainty_surface_log_error_correlation"]
                                for item in valid
                            ]
                        )
                    )
                    if valid
                    else None
                ),
            }
        )
    return summaries


def main() -> int:
    args = _arguments()
    output_manifest = args.output_dir / "manifest.json"
    if output_manifest.exists() and not args.force:
        raise SystemExit(f"output already contains {output_manifest}; pass --force")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_cache_manifest = None
    if args.feature_cache_dir is not None:
        feature_cache_manifest = args.feature_cache_dir / "manifest.json"
        if feature_cache_manifest.exists() and not args.force:
            raise SystemExit(
                f"feature cache already contains {feature_cache_manifest}; pass --force"
            )
        args.feature_cache_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
        calculate_dimensions,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen denoising experiment requires CUDA")
    dataset_root = args.dataset.resolve()
    dataset_manifest = _load_json(dataset_root / "manifest.json")
    dataset_views = list(dataset_manifest["views"])
    view_by_camera = {
        (
            int(round(float(view["azimuth_degrees"]))) % 360,
            int(round(float(view["elevation_degrees"]))),
        ): view
        for view in dataset_views
    }
    elevations = args.elevations or ["eye"]
    azimuths = args.azimuths or list(AZIMUTHS)
    requested = []
    for elevation_name in elevations:
        elevation_degrees, elevation_descriptor = ELEVATIONS[elevation_name]
        for azimuth in azimuths:
            view = view_by_camera.get((azimuth, elevation_degrees))
            if view is None:
                raise SystemExit(
                    f"dataset lacks azimuth {azimuth}, elevation {elevation_degrees}"
                )
            requested.append((view, elevation_name, elevation_descriptor))

    front_view = view_by_camera[(0, 0)]
    back_view = view_by_camera[(180, 0)]
    front = _rgb(dataset_root / front_view["rgb"])
    back = _rgb(dataset_root / back_view["rgb"])
    surface_points, visible_by_id = _reference_points(dataset_root, dataset_views)
    surface_tree = cKDTree(surface_points)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    blocks = [int(value) for value in checkpoint["blocks"]]
    head_architecture = str(checkpoint.get("architecture", "local-v1"))
    head = build_canonical_pointmap_head(
        head_architecture,
        len(blocks),
        int(checkpoint["feature_channels"]),
        int(checkpoint["projection_channels"]),
        int(checkpoint["hidden_channels"]),
        int(checkpoint.get("ray_channels", 0)),
        int(checkpoint.get("timestep_channels", 0)),
    )
    head.load_state_dict(checkpoint["model"])
    head.eval().to("cuda")
    target_resolution = int(checkpoint["target_resolution"])

    profile = image_edit_profile(QWEN_IMAGE_EDIT_PROFILE_ID)
    parameters = profile.defaults()
    parameters.update(seed=args.seed, steps=args.steps)
    adapters = [
        adapter.to_dict()
        for adapter in (
            *profile.default_lora_adapters,
            qwen_multiple_angles_lora_adapter(),
        )
    ]
    backend = RealMlBackend()
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    print("Loading Qwen Multiple Angles and canonical head...", flush=True)
    loaded = backend.load_image_edit(
        {
            "profile_id": profile.stable_id,
            "parameters": parameters,
            "lora_adapters": adapters,
        }
    )
    pipe = backend._instruct_pipe
    if pipe is None:
        raise RuntimeError("Qwen pipeline did not load")
    transformer = pipe.transformer
    if blocks[-1] >= len(transformer.transformer_blocks):
        raise RuntimeError("checkpoint block is outside the loaded transformer")
    width, height = calculate_dimensions(1024 * 1024, front.width / front.height)
    multiple = pipe.vae_scale_factor * 2
    width = width // multiple * multiple
    height = height // multiple * multiple
    grid_width = width // pipe.vae_scale_factor // 2
    grid_height = height // pipe.vae_scale_factor // 2
    target_tokens = grid_width * grid_height

    capture = {
        "enabled": False,
        "features": {},
        "predictions": {},
        "step_by_timestep": {},
        "completed_steps": set(),
        "camera": None,
        "rays": None,
        "feature_samples": {},
        "view_slug": None,
    }

    def step_index() -> int:
        timestep = float(pipe._current_timestep.detach().cpu())
        key = f"{timestep:.8f}"
        mapping = capture["step_by_timestep"]
        return mapping.setdefault(key, len(mapping))

    def make_hook(block_index: int):
        def hook(_module, _inputs, output):
            if not capture["enabled"]:
                return
            step = step_index()
            if step in capture["completed_steps"]:
                return
            key = (step, block_index)
            # True CFG invokes the transformer a second time for the negative
            # prompt. The first occurrence is the production conditional pass.
            if key in capture["features"]:
                return
            feature = output[0, :target_tokens].reshape(
                grid_height, grid_width, -1
            )
            capture["features"][key] = feature.detach()
            if block_index != blocks[-1]:
                return
            missing = [
                block for block in blocks if (step, block) not in capture["features"]
            ]
            if missing:
                raise RuntimeError(f"step {step} missed feature blocks {missing}")
            feature_tensor = torch.stack(
                [
                    capture["features"][(step, block)].permute(2, 0, 1)
                    for block in blocks
                ]
            ).unsqueeze(0).float()
            if args.feature_cache_dir is not None:
                feature_name = f"{capture['view_slug']}-step-{step:02d}.npy"
                feature_path = args.feature_cache_dir / feature_name
                feature_array = (
                    feature_tensor[0]
                    .permute(0, 2, 3, 1)
                    .to(dtype=torch.float16)
                    .cpu()
                    .numpy()
                )
                np.save(feature_path, feature_array, allow_pickle=False)
                capture["feature_samples"][step] = {
                    "step": step,
                    "timestep": float(pipe._current_timestep.detach().cpu()),
                    "normalized_timestep": (
                        float(pipe._current_timestep.detach().cpu()) / 1000.0
                    ),
                    "features": feature_name,
                    "shape": list(feature_array.shape),
                    "dtype": str(feature_array.dtype),
                }
            with torch.inference_mode():
                normalized_timestep = torch.tensor(
                    [float(pipe._current_timestep.detach().cpu()) / 1000.0],
                    device=feature_tensor.device,
                    dtype=feature_tensor.dtype,
                )
                prediction = head(
                    feature_tensor,
                    capture["camera"],
                    capture["rays"],
                    target_resolution,
                    timestep=normalized_timestep,
                )[0]
            capture["predictions"][step] = prediction.detach().float().cpu().numpy()
            capture["completed_steps"].add(step)
            for block in blocks:
                del capture["features"][(step, block)]

        return hook

    handles = [
        transformer.transformer_blocks[index].img_norm2.register_forward_hook(
            make_hook(index)
        )
        for index in blocks
    ]
    evaluations = []
    output_views = []
    feature_cache_views = []
    try:
        for index, (view, elevation_name, elevation_descriptor) in enumerate(
            requested, start=1
        ):
            azimuth = int(round(float(view["azimuth_degrees"]))) % 360
            prompt = (
                f"<sks> {AZIMUTHS[azimuth]} {elevation_descriptor} medium shot"
            )
            camera_record = _load_json(dataset_root / view["camera"])
            camera_vector, rays = _camera_inputs(
                camera_record,
                (grid_height, grid_width),
            )
            output_rays = canonical_camera_ray_map(
                camera_from_canonical=camera_record["camera_from_canonical"],
                intrinsics=camera_record["intrinsics"],
                image_size=camera_record["image_size"],
                grid_height=target_resolution,
                grid_width=target_resolution,
            )
            capture["enabled"] = True
            capture["features"] = {}
            capture["predictions"] = {}
            capture["step_by_timestep"] = {}
            capture["completed_steps"] = set()
            capture["feature_samples"] = {}
            capture["view_slug"] = view["id"].replace("/", "_")
            capture["camera"] = torch.from_numpy(camera_vector).unsqueeze(0).to(
                device="cuda", dtype=torch.float32
            )
            capture["rays"] = (
                torch.from_numpy(rays)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device="cuda", dtype=torch.float32)
            )
            view_parameters = dict(parameters)
            view_parameters["prompt"] = prompt
            print(
                f"[{index}/{len(requested)}] generating {view['id']} and "
                f"running head at {args.steps} steps",
                flush=True,
            )
            result, result_seed, provenance = backend.image_edit(
                {
                    "profile_id": profile.stable_id,
                    "parameters": view_parameters,
                    "lora_adapters": adapters,
                },
                front,
                back,
            )
            capture["enabled"] = False
            if sorted(capture["predictions"]) != list(range(args.steps)):
                raise RuntimeError(
                    f"{view['id']} captured steps {sorted(capture['predictions'])}"
                )
            image_path = args.output_dir / f"{view['id']}.png"
            prediction_path = args.output_dir / f"{view['id']}-head.npz"
            result.save(image_path)
            fusion_predictions = _fuse_predictions(
                [capture["predictions"][step] for step in range(args.steps)]
            )
            np.savez_compressed(
                prediction_path,
                **{
                    f"prediction_step_{step}": capture["predictions"][step]
                    for step in range(args.steps)
                },
                **{
                    f"prediction_fusion_{name}": prediction
                    for name, prediction in fusion_predictions.items()
                },
            )
            for step in range(args.steps):
                _write_prediction_diagnostics(
                    args.output_dir,
                    view["id"],
                    step,
                    capture["predictions"][step],
                )
            for name, prediction in fusion_predictions.items():
                _write_prediction_diagnostics(
                    args.output_dir,
                    view["id"],
                    f"fusion-{name}",
                    prediction,
                )
            metrics = [
                _prediction_metrics(
                    capture["predictions"][step],
                    output_rays,
                    surface_tree,
                    visible_by_id[view["id"]],
                )
                for step in range(args.steps)
            ]
            fusion_metrics = {
                name: _prediction_metrics(
                    prediction,
                    output_rays,
                    surface_tree,
                    visible_by_id[view["id"]],
                )
                for name, prediction in fusion_predictions.items()
            }
            evaluations.append(
                {
                    "id": view["id"],
                    "steps": metrics,
                    "fusions": fusion_metrics,
                }
            )
            output_views.append(
                {
                    "id": view["id"],
                    "azimuth_degrees": azimuth,
                    "elevation_degrees": int(round(float(view["elevation_degrees"]))),
                    "prompt": prompt,
                    "seed": int(result_seed),
                    "image": image_path.name,
                    "predictions": prediction_path.name,
                    "timesteps": {
                        value: int(step)
                        for value, step in capture["step_by_timestep"].items()
                    },
                    "provenance": provenance,
                }
            )
            if args.feature_cache_dir is not None:
                feature_cache_views.append(
                    {
                        "id": view["id"],
                        "azimuth_degrees": azimuth,
                        "elevation_degrees": int(
                            round(float(view["elevation_degrees"]))
                        ),
                        "prompt": prompt,
                        "seed": int(result_seed),
                        "image": str(image_path.resolve()),
                        "samples": [
                            capture["feature_samples"][step]
                            for step in range(args.steps)
                        ],
                    }
                )
            print(
                f"[{index}/{len(requested)}] step metrics: "
                + ", ".join(
                    (
                        f"{step}:surface-med="
                        f"{metrics[step].get('surface_distance_median', float('nan')):.3f}"
                    )
                    for step in range(args.steps)
                ),
                flush=True,
            )
            capture["camera"] = None
            capture["rays"] = None
            torch.cuda.empty_cache()
    finally:
        capture["enabled"] = False
        for handle in handles:
            handle.remove()
        head.to("cpu")
        backend.unload_image_edit()

    summaries = _summaries(evaluations, args.steps)
    fusion_summaries = {
        name: _summaries(
            [
                {"steps": [evaluation["fusions"][name]]}
                for evaluation in evaluations
            ],
            1,
        )[0]
        for name in FUSION_NAMES
    }
    manifest = {
        "schema": "diffusion-editor.qwen-canonical-head-denoising",
        "schema_version": 1,
        "dataset": str(dataset_root),
        "checkpoint": str(args.checkpoint.resolve()),
        "profile": profile.stable_id,
        "loaded": loaded,
        "blocks": blocks,
        "steps": args.steps,
        "fusions": list(FUSION_NAMES),
        "seed": args.seed,
        "conditioning_views": [front_view["id"], back_view["id"]],
        "output_size": [width, height],
        "feature_grid": [grid_width, grid_height],
        "head_output_resolution": target_resolution,
        "head_architecture": head_architecture,
        "elapsed_seconds": time.monotonic() - started,
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "views": output_views,
        "evaluation": evaluations,
        "step_summaries": summaries,
        "fusion_summaries": fusion_summaries,
        "metric_note": (
            "Generated pixels have no exact Blender correspondence. Surface distance "
            "uses the nearest canonical dataset surface; coverage uses the requested view's "
            "visible Blender surface; ray distance uses the nominal requested camera."
        ),
    }
    output_manifest.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    if feature_cache_manifest is not None:
        feature_cache = {
            "schema": "diffusion-editor.qwen-generated-canonical-features",
            "schema_version": 1,
            "dataset": str(dataset_root),
            "source_capture": str(args.output_dir.resolve()),
            "profile": profile.stable_id,
            "blocks": blocks,
            "steps": args.steps,
            "seed": args.seed,
            "conditioning_views": [front_view["id"], back_view["id"]],
            "output_size": [width, height],
            "feature_grid": [grid_width, grid_height],
            "views": feature_cache_views,
        }
        feature_cache_manifest.write_text(
            json.dumps(feature_cache, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Feature cache: {feature_cache_manifest}", flush=True)
    print(json.dumps(summaries, indent=2), flush=True)
    print(f"Complete: {output_manifest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
