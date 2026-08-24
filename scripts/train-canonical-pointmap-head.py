#!/usr/bin/env python3
"""Train a small canonical point-map head on one or more Qwen feature caches."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import sys
import time

import numpy as np
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from diffusion_editor.training.canonical_pointmap import (
    canonical_camera_ray_map,
    normalized_camera_vector,
)
from diffusion_editor.training.checkpointing import periodic_epoch_checkpoint_due


BOTTLENECK_DECODER_PREFIXES = (
    "bottleneck.",
    "attention.",
    "up32.",
    "merge32.",
    "decoder32.",
    "up64.",
    "merge64.",
    "decoder64.",
    "refine128.",
    "condition128.",
    "refine256.",
    "condition256.",
    "auxiliary64.",
    "auxiliary128.",
    "output.",
)


def _parameter_in_trainable_scope(name: str, scope: str) -> bool:
    if scope == "all":
        return True
    if scope == "vl-only":
        return name.startswith("vl_context.")
    if scope == "bottleneck-decoder":
        return name.startswith(("vl_context.", *BOTTLENECK_DECODER_PREFIXES))
    raise ValueError(f"unknown trainable scope: {scope}")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "features",
        nargs="+",
        type=Path,
        help="One or more compatible Qwen canonical-feature directories.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--view", action="append", dest="views")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--target-resolution", type=int, default=128)
    parser.add_argument("--projection-channels", type=int, default=48)
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument(
        "--architecture",
        choices=("local-v1", "multiscale-v2", "multiscale-vl-v3"),
        default="local-v1",
    )
    parser.add_argument(
        "--vl-conditioning",
        nargs="+",
        type=Path,
        help=(
            "Qwen2.5-VL conditioning cache roots paired in order with the "
            "positional feature roots."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--vl-learning-rate",
        type=float,
        help=(
            "Optional separate learning rate for vl_context parameters. "
            "Other trainable parameters use --learning-rate."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument(
        "--no-ray-features",
        action="store_true",
        help="Ablation: omit per-token canonical camera origin/direction.",
    )
    parser.add_argument(
        "--timestep-conditioning",
        action="store_true",
        help="Condition a newly initialized head on normalized FlowMatch time.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Initialize from a saved head; use --steps 0 for evaluation only.",
    )
    parser.add_argument(
        "--initialize-backbone-checkpoint",
        type=Path,
        help=(
            "Initialize matching non-VL weights of a new multiscale-vl-v3 "
            "head from a multiscale-v2 checkpoint."
        ),
    )
    parser.add_argument(
        "--evaluation-feature-ablation",
        choices=("zero", "spatial-roll"),
        help="Corrupt cached features during evaluation without retraining.",
    )
    parser.add_argument(
        "--evaluation-vl-ablation",
        choices=("zero", "spatial-roll", "source-swap"),
        help="Corrupt Qwen2.5-VL context during evaluation without retraining.",
    )
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument(
        "--checkpoint-every-epochs",
        type=int,
        default=0,
        help=(
            "Save a model checkpoint for resumable periodic evaluation after "
            "this many complete epochs; zero disables periodic retention."
        ),
    )
    parser.add_argument(
        "--host-cache-samples",
        type=int,
        default=8,
        help=(
            "Maximum decoded feature samples retained in host RAM. Use 0 for "
            "fully streaming reads or -1 for the legacy eager loader."
        ),
    )
    parser.add_argument(
        "--identity-balanced-sampling",
        action="store_true",
        help="Sample identities uniformly when roots contain unequal sample counts.",
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help=(
            "Train only the VL context branch of multiscale-vl-v3. This "
            "isolates information added by Qwen2.5-VL."
        ),
    )
    parser.add_argument(
        "--trainable-scope",
        choices=("all", "vl-only", "bottleneck-decoder"),
        default="all",
        help=(
            "Select trainable modules. bottleneck-decoder also includes "
            "vl_context when present."
        ),
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.steps < 0 or args.batch_size <= 0:
        parser.error("--steps must be non-negative and --batch-size positive")
    if args.steps == 0 and args.checkpoint is None:
        parser.error("--steps 0 requires --checkpoint")
    if args.checkpoint and args.initialize_backbone_checkpoint:
        parser.error(
            "--checkpoint and --initialize-backbone-checkpoint are mutually exclusive"
        )
    if args.architecture == "multiscale-vl-v3" and not args.vl_conditioning:
        parser.error("multiscale-vl-v3 requires --vl-conditioning")
    if args.vl_conditioning and len(args.vl_conditioning) != len(args.features):
        parser.error("--vl-conditioning must have one root per feature root")
    if args.target_resolution <= 0:
        parser.error("--target-resolution must be positive")
    if args.host_cache_samples < -1:
        parser.error("--host-cache-samples must be -1, zero, or positive")
    if args.checkpoint_every_epochs < 0:
        parser.error("--checkpoint-every-epochs must be non-negative")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.vl_learning_rate is not None and args.vl_learning_rate <= 0.0:
        parser.error("--vl-learning-rate must be positive")
    if args.freeze_backbone and args.trainable_scope != "all":
        parser.error("--freeze-backbone cannot be combined with --trainable-scope")
    return args


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_rgb(path: Path, values: np.ndarray) -> None:
    Image.fromarray(values.astype(np.uint8), "RGB").save(path)


def _xyz_image(xyz: np.ndarray, mask: np.ndarray) -> np.ndarray:
    encoded = np.clip(xyz + 0.5, 0.0, 1.0) * 255.0
    encoded[~mask] = 0.0
    return np.round(encoded).astype(np.uint8)


def _identity_balanced_weights(samples: list[dict]) -> list[float]:
    counts = Counter(sample["identity_id"] for sample in samples)
    return [1.0 / counts[sample["identity_id"]] for sample in samples]


class CanonicalFeatureDataset:
    """Decode samples on demand with an optional bounded host-memory LRU."""

    def __init__(self, samples: list[dict], host_cache_samples: int = 8):
        self.samples = list(samples)
        self.host_cache_samples = host_cache_samples
        self._cache: OrderedDict[int, dict] = OrderedDict()
        self.requests = 0
        self.hits = 0
        self.misses = 0
        self.peak_entries = 0
        if host_cache_samples == -1:
            for index in range(len(self.samples)):
                self._cache[index] = self._load(self.samples[index])
            self.peak_entries = len(self._cache)

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def _load(feature_sample: dict) -> dict:
        feature_root = feature_sample["feature_root"]
        dataset_root = feature_sample["dataset_root"]
        view = feature_sample["dataset_view"]
        features = np.load(
            feature_root / feature_sample["features"],
            mmap_mode="r",
            allow_pickle=False,
        )
        # Copy only the requested sample out of its read-only memmap. Keeping
        # every sample resident made host RAM grow with total corpus size.
        features = np.array(features, dtype=np.float16, copy=True)
        with np.load(dataset_root / view["geometry"]) as geometry:
            xyz = np.array(geometry["canonical_xyz"], copy=True)
            mask = np.array(geometry["mask"], dtype=np.float32, copy=True)
        camera_record = _load_json(dataset_root / view["camera"])
        camera = normalized_camera_vector(
            camera_from_canonical=camera_record["camera_from_canonical"],
            intrinsics=camera_record["intrinsics"],
            image_size=camera_record["image_size"],
        )
        rays = canonical_camera_ray_map(
            camera_from_canonical=camera_record["camera_from_canonical"],
            intrinsics=camera_record["intrinsics"],
            image_size=camera_record["image_size"],
            grid_height=features.shape[1],
            grid_width=features.shape[2],
        )
        result = {
            "id": feature_sample["sample_id"],
            "identity_id": feature_sample["identity_id"],
            "pose_id": feature_sample["pose_id"],
            "view_id": feature_sample["view_id"],
            "replica": int(feature_sample.get("replica", 0)),
            "schedule_step": int(feature_sample.get("step", 0)),
            "timestep": np.float32(
                feature_sample.get("normalized_timestep", 0.0)
            ),
            "features": np.transpose(features, (0, 3, 1, 2)),
            "xyz": np.transpose(xyz, (2, 0, 1)),
            "mask": mask[None, :, :],
            "camera": camera,
            "rays": np.transpose(rays, (2, 0, 1)),
        }
        if "vl_conditioning" in feature_sample:
            result["vl_conditioning"] = np.array(
                np.load(
                    feature_sample["vl_root"]
                    / feature_sample["vl_conditioning"],
                    mmap_mode="r",
                    allow_pickle=False,
                ),
                dtype=np.float16,
                copy=True,
            )
        return result

    def __getitem__(self, index):
        self.requests += 1
        if index in self._cache:
            self.hits += 1
            value = self._cache.pop(index)
            self._cache[index] = value
            return value
        self.misses += 1
        value = self._load(self.samples[index])
        if self.host_cache_samples > 0:
            self._cache[index] = value
            while len(self._cache) > self.host_cache_samples:
                self._cache.popitem(last=False)
            self.peak_entries = max(self.peak_entries, len(self._cache))
        return value

    def cache_info(self) -> dict:
        return {
            "configured_samples": self.host_cache_samples,
            "resident_samples": len(self._cache),
            "peak_resident_samples": self.peak_entries,
            "requests": self.requests,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / self.requests if self.requests else 0.0,
        }


def main() -> int:
    args = _arguments()
    import torch
    import torch.nn.functional as functional
    from diffusion_editor.training.canonical_pointmap_head import (
        build_canonical_pointmap_head,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("canonical point-map training requires CUDA")
    output_manifest_path = args.output_dir / "training.json"
    if output_manifest_path.exists() and not args.force:
        raise SystemExit(
            f"output already contains {output_manifest_path}; pass --force"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = args.output_dir / "diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    feature_roots = [path.resolve() for path in args.features]
    feature_manifests = []
    selected_view_ids = set(args.views or [])
    feature_samples = []
    selected_feature_view_count = 0
    observed_view_ids = set()
    identity_records = []
    vl_roots = [path.resolve() for path in (args.vl_conditioning or [])]
    for source_index, feature_root in enumerate(feature_roots):
        feature_manifest = _load_json(feature_root / "manifest.json")
        if feature_manifest.get("schema") != "diffusion-editor.qwen-canonical-features":
            raise SystemExit(f"unsupported Qwen feature manifest: {feature_root}")
        dataset_root = Path(feature_manifest["dataset"])
        dataset_manifest = _load_json(dataset_root / "manifest.json")
        dataset_views = {view["id"]: view for view in dataset_manifest["views"]}
        asset = dataset_manifest.get("asset", {})
        identity_id = str(asset.get("identity_id") or asset.get("name") or source_index)
        pose_id = str(asset.get("pose_id") or "unspecified")
        identity_records.append(
            {
                "identity_id": identity_id,
                "pose_id": pose_id,
                "features": str(feature_root),
                "dataset": str(dataset_root),
            }
        )
        feature_views = list(feature_manifest["views"])
        vl_root = vl_roots[source_index] if vl_roots else None
        vl_manifest = None
        vl_views = {}
        if vl_root is not None:
            vl_manifest = _load_json(vl_root / "manifest.json")
            if vl_manifest.get("schema") != (
                "diffusion-editor.qwen-vl-conditioning-features"
            ):
                raise SystemExit(f"unsupported VL manifest: {vl_root}")
            if Path(vl_manifest["dataset"]).resolve() != (
                dataset_root / "manifest.json"
            ).resolve():
                raise RuntimeError(
                    f"VL conditioning dataset differs for {feature_root}"
                )
            vl_views = {view["id"]: view for view in vl_manifest["views"]}
        observed_view_ids.update(view["id"] for view in feature_views)
        if selected_view_ids:
            feature_views = [
                view for view in feature_views if view["id"] in selected_view_ids
            ]
        selected_feature_view_count += len(feature_views)
        feature_manifests.append(feature_manifest)
        for feature_view in feature_views:
            if feature_view["id"] not in dataset_views:
                raise RuntimeError(
                    f"feature view {feature_view['id']} is absent from {dataset_root}"
                )
            if vl_root is not None and feature_view["id"] not in vl_views:
                continue
            samples = feature_view.get("samples")
            if not samples:
                sigma = float(feature_manifest.get("readout_sigma", 0.0))
                samples = [
                    {
                        "replica": 0,
                        "step": 0,
                        "timestep": sigma * 1000.0,
                        "normalized_timestep": sigma,
                        "sigma": sigma,
                        "features": feature_view["features"],
                        "shape": feature_view["shape"],
                    }
                ]
            for sample in samples:
                replica = int(sample.get("replica", 0))
                schedule_step = int(sample.get("step", 0))
                feature_samples.append(
                    {
                        **sample,
                        "feature_root": feature_root,
                        "dataset_root": dataset_root,
                        "dataset_view": dataset_views[feature_view["id"]],
                        "identity_id": identity_id,
                        "pose_id": pose_id,
                        "view_id": feature_view["id"],
                        "sample_id": (
                            f"{identity_id}-{feature_view['id']}-"
                            f"r{replica:02d}-s{schedule_step:02d}"
                        ),
                        **(
                            {
                                "vl_root": vl_root,
                                "vl_conditioning": vl_views[
                                    feature_view["id"]
                                ]["conditioning"],
                                "vl_shape": vl_views[feature_view["id"]]["shape"],
                            }
                            if vl_root is not None
                            else {}
                        ),
                    }
                )
    if selected_view_ids:
        missing = sorted(selected_view_ids - observed_view_ids)
        if missing:
            raise SystemExit(f"unknown feature views: {missing}")
    if not feature_samples:
        raise SystemExit("no feature samples selected")
    reference_manifest = feature_manifests[0]
    for feature_root, feature_manifest in zip(
        feature_roots[1:], feature_manifests[1:]
    ):
        if list(feature_manifest["blocks"]) != list(reference_manifest["blocks"]):
            raise RuntimeError(f"feature block lists differ: {feature_root}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    dataset = CanonicalFeatureDataset(
        feature_samples, host_cache_samples=args.host_cache_samples
    )
    generator = torch.Generator().manual_seed(args.seed)
    sampler = None
    if args.identity_balanced_sampling:
        sampler = torch.utils.data.WeightedRandomSampler(
            _identity_balanced_weights(feature_samples),
            num_samples=len(feature_samples),
            replacement=True,
            generator=generator,
        )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=0,
        generator=generator,
    )
    feature_shape = feature_samples[0]["shape"]
    if any(list(sample["shape"]) != list(feature_shape) for sample in feature_samples):
        raise RuntimeError("all cached feature samples must have the same shape")
    block_count = int(feature_shape[0])
    feature_channels = int(feature_shape[-1])
    context_shape = feature_samples[0].get("vl_shape")
    if context_shape is not None:
        if any(sample.get("vl_shape") != context_shape for sample in feature_samples):
            raise RuntimeError("all cached VL conditioning shapes must match")
        if len(context_shape) != 5 or int(context_shape[1]) != 2:
            raise RuntimeError(
                "VL conditioning must have shape "
                "(layers, 2, height, width, channels)"
            )
        context_layers = int(context_shape[0])
        context_channels = int(context_shape[-1])
    else:
        context_layers = 0
        context_channels = 0
    checkpoint_source = None
    backbone_source = None
    if args.checkpoint is not None:
        checkpoint_source = torch.load(args.checkpoint, map_location="cpu")
        if list(checkpoint_source["blocks"]) != list(reference_manifest["blocks"]):
            raise RuntimeError("checkpoint and feature block lists differ")
        if int(checkpoint_source["feature_channels"]) != feature_channels:
            raise RuntimeError("checkpoint and feature channel counts differ")
        projection_channels = int(checkpoint_source["projection_channels"])
        hidden_channels = int(checkpoint_source["hidden_channels"])
        target_resolution = int(checkpoint_source["target_resolution"])
        ray_channels = int(checkpoint_source.get("ray_channels", 0))
        timestep_channels = int(checkpoint_source.get("timestep_channels", 0))
        architecture = str(checkpoint_source.get("architecture", "local-v1"))
        context_layers = int(checkpoint_source.get("context_layers", 0))
        context_channels = int(checkpoint_source.get("context_channels", 0))
    else:
        if args.initialize_backbone_checkpoint is not None:
            backbone_source = torch.load(
                args.initialize_backbone_checkpoint, map_location="cpu"
            )
            if list(backbone_source["blocks"]) != list(
                reference_manifest["blocks"]
            ):
                raise RuntimeError("backbone and feature block lists differ")
            if int(backbone_source["feature_channels"]) != feature_channels:
                raise RuntimeError(
                    "backbone and feature channel counts differ"
                )
            projection_channels = int(backbone_source["projection_channels"])
            hidden_channels = int(backbone_source["hidden_channels"])
            target_resolution = int(backbone_source["target_resolution"])
            ray_channels = int(backbone_source.get("ray_channels", 0))
            timestep_channels = int(
                backbone_source.get("timestep_channels", 0)
            )
        else:
            projection_channels = args.projection_channels
            hidden_channels = args.hidden_channels
            target_resolution = args.target_resolution
            ray_channels = 0 if args.no_ray_features else 6
            timestep_channels = (
                projection_channels if args.timestep_conditioning else 0
            )
        architecture = args.architecture
    device = torch.device("cuda")
    model = build_canonical_pointmap_head(
        architecture,
        block_count,
        feature_channels,
        projection_channels,
        hidden_channels,
        ray_channels,
        timestep_channels,
        context_layers,
        context_channels,
    ).to(device)
    if checkpoint_source is not None:
        model.load_state_dict(checkpoint_source["model"])
    elif args.initialize_backbone_checkpoint is not None:
        if architecture != "multiscale-vl-v3":
            raise RuntimeError(
                "--initialize-backbone-checkpoint requires multiscale-vl-v3"
            )
        incompatible = model.load_state_dict(
            backbone_source["model"], strict=False
        )
        unexpected = list(incompatible.unexpected_keys)
        invalid_missing = [
            key for key in incompatible.missing_keys
            if not key.startswith("vl_context.")
        ]
        if unexpected or invalid_missing:
            raise RuntimeError(
                "backbone checkpoint mismatch: "
                f"missing={invalid_missing}, unexpected={unexpected}"
            )
    trainable_scope = args.trainable_scope
    if args.freeze_backbone:
        if architecture != "multiscale-vl-v3":
            raise RuntimeError("--freeze-backbone requires multiscale-vl-v3")
        trainable_scope = "vl-only"
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(
            _parameter_in_trainable_scope(name, trainable_scope)
        )
    named_trainable_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    trainable_parameters = [
        parameter for _name, parameter in named_trainable_parameters
    ]
    if not trainable_parameters:
        raise RuntimeError(
            f"trainable scope {trainable_scope} selected no parameters"
        )
    vl_parameters = [
        parameter for name, parameter in named_trainable_parameters
        if name.startswith("vl_context.")
    ]
    regular_parameters = [
        parameter for name, parameter in named_trainable_parameters
        if not name.startswith("vl_context.")
    ]
    optimizer_groups = []
    if regular_parameters:
        optimizer_groups.append(
            {"params": regular_parameters, "lr": args.learning_rate}
        )
    if vl_parameters:
        optimizer_groups.append(
            {
                "params": vl_parameters,
                "lr": (
                    args.vl_learning_rate
                    if args.vl_learning_rate is not None
                    else args.learning_rate
                ),
            }
        )
    elif args.vl_learning_rate is not None:
        raise RuntimeError("--vl-learning-rate requires a trainable VL branch")
    optimizer = torch.optim.AdamW(
        optimizer_groups,
        weight_decay=args.weight_decay,
    )
    def learning_rate_multiplier(step: int) -> float:
        progress = min(float(step) / max(args.steps, 1), 1.0)
        return 0.02 + 0.98 * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=learning_rate_multiplier
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in trainable_parameters
    )
    checkpoint_metadata = {
        "architecture": architecture,
        "blocks": reference_manifest["blocks"],
        "feature_channels": feature_channels,
        "projection_channels": projection_channels,
        "hidden_channels": hidden_channels,
        "target_resolution": target_resolution,
        "ray_channels": ray_channels,
        "timestep_channels": timestep_channels,
        "context_layers": context_layers,
        "context_channels": context_channels,
    }
    auxiliary_loss_weights = (
        (0.125, 0.25) if architecture.startswith("multiscale-") else ()
    )
    action = "Evaluating" if args.steps == 0 else "Training"
    print(
        f"{action} {architecture} with {parameter_count:,} parameters "
        f"({trainable_parameter_count:,} trainable) on "
        f"{len(dataset)} samples "
        f"from {selected_feature_view_count} identity-views across "
        f"{len(feature_roots)} identities; "
        f"feature shape {tuple(feature_shape)}",
        flush=True,
    )

    history = []
    epoch_history = []
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    iterator = iter(loader)
    model.train()
    best_loss = float("inf")
    best_step = 0
    best_state = (
        {
            name: value.detach().cpu().clone()
            for name, value in model.state_dict().items()
        }
        if args.steps == 0
        else None
    )
    epoch_loss_sum = 0.0
    epoch_batch_count = 0
    epoch_primary_sums = {
        "xyz_loss": 0.0,
        "mask_loss": 0.0,
        "uncertainty_loss": 0.0,
    }
    epoch_gradient_norm_sum = 0.0
    epoch_gradient_norm_max = 0.0
    epoch_auxiliary_sums = [
        {
            "loss": 0.0,
            "xyz_loss": 0.0,
            "mask_loss": 0.0,
            "uncertainty_loss": 0.0,
        }
        for _weight in auxiliary_loss_weights
    ]
    sampled_ids = []
    periodic_checkpoints = []
    periodic_checkpoint_dir = args.output_dir / "checkpoints"

    def prediction_losses(prediction, truth_xyz, truth_mask):
        output_size = prediction.shape[-2:]
        if truth_xyz.shape[-2:] != output_size:
            truth_xyz = functional.interpolate(
                truth_xyz,
                size=output_size,
                mode="nearest",
            )
            truth_mask = functional.interpolate(
                truth_mask,
                size=output_size,
                mode="nearest",
            )
        predicted_xyz = prediction[:, :3]
        mask_logits = prediction[:, 3:4]
        predicted_log_error = prediction[:, 4:5].clamp(-9.0, 2.0)
        foreground = truth_mask.expand_as(truth_xyz) > 0.5
        xyz_loss = functional.smooth_l1_loss(
            predicted_xyz[foreground], truth_xyz[foreground]
        )
        foreground_fraction = truth_mask.mean().clamp_min(1.0e-4)
        positive_weight = (
            (1.0 - foreground_fraction) / foreground_fraction
        ).clamp(1.0, 20.0)
        mask_loss = functional.binary_cross_entropy_with_logits(
            mask_logits,
            truth_mask,
            pos_weight=positive_weight.detach(),
        )
        with torch.no_grad():
            error_target = torch.log(
                torch.mean(
                    torch.abs(predicted_xyz - truth_xyz),
                    dim=1,
                    keepdim=True,
                )
                + 1.0e-4
            ).clamp(-9.0, 2.0)
        uncertainty_loss = functional.smooth_l1_loss(
            predicted_log_error[truth_mask > 0.5],
            error_target[truth_mask > 0.5],
        )
        return {
            "loss": 10.0 * xyz_loss + mask_loss + 0.1 * uncertainty_loss,
            "xyz": predicted_xyz,
            "mask_logits": mask_logits,
            "log_error": predicted_log_error,
            "xyz_loss": xyz_loss,
            "mask_loss": mask_loss,
            "uncertainty_loss": uncertainty_loss,
            "target_xyz": truth_xyz,
            "target_mask": truth_mask,
        }

    for step in range(1, args.steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        sampled_ids.extend(batch["id"])
        features = batch["features"].to(
            device, dtype=torch.float32, non_blocking=True
        )
        target_xyz = batch["xyz"].to(device, non_blocking=True)
        target_mask = batch["mask"].to(device, non_blocking=True)
        camera = batch["camera"].to(device, non_blocking=True)
        rays = batch["rays"].to(device, non_blocking=True)
        timestep = batch["timestep"].to(device, non_blocking=True)
        vl_conditioning = (
            batch["vl_conditioning"].to(
                device, dtype=torch.float32, non_blocking=True
            )
            if "vl_conditioning" in batch
            else None
        )
        target_xyz = functional.interpolate(
            target_xyz,
            size=(target_resolution, target_resolution),
            mode="nearest",
        )
        target_mask = functional.interpolate(
            target_mask,
            size=(target_resolution, target_resolution),
            mode="nearest",
        )
        if architecture.startswith("multiscale-"):
            pyramid = model.forward_pyramid(
                features,
                camera,
                rays,
                target_resolution,
                timestep=timestep,
                context=vl_conditioning,
            )
            prediction = pyramid["prediction"]
        else:
            pyramid = None
            prediction = model(
                features,
                camera,
                rays,
                target_resolution,
                timestep=timestep,
            )
        primary_losses = prediction_losses(
            prediction,
            target_xyz,
            target_mask,
        )
        loss = primary_losses["loss"]
        auxiliary_losses = []
        if pyramid is not None:
            for weight, auxiliary in zip(
                auxiliary_loss_weights,
                pyramid["auxiliary"],
            ):
                auxiliary_result = prediction_losses(
                    auxiliary,
                    target_xyz,
                    target_mask,
                )
                auxiliary_losses.append(auxiliary_result)
                loss = loss + weight * auxiliary_result["loss"]
        predicted_xyz = primary_losses["xyz"]
        mask_logits = primary_losses["mask_logits"]
        predicted_log_error = primary_losses["log_error"]
        xyz_loss = primary_losses["xyz_loss"]
        mask_loss = primary_losses["mask_loss"]
        uncertainty_loss = primary_losses["uncertainty_loss"]
        target_xyz = primary_losses["target_xyz"]
        target_mask = primary_losses["target_mask"]
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            trainable_parameters, 1.0
        )
        if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
            raise FloatingPointError(
                f"non-finite training values at step {step}: "
                f"loss={float(loss.detach())}, "
                f"gradient_norm={float(gradient_norm.detach())}"
            )
        optimizer.step()
        scheduler.step()

        loss_value = float(loss.detach())
        gradient_norm_value = float(gradient_norm.detach())
        epoch_loss_sum += loss_value
        epoch_batch_count += 1
        epoch_gradient_norm_sum += gradient_norm_value
        epoch_gradient_norm_max = max(
            epoch_gradient_norm_max, gradient_norm_value
        )
        for key in epoch_primary_sums:
            epoch_primary_sums[key] += float(primary_losses[key].detach())
        for sums, auxiliary_result in zip(
            epoch_auxiliary_sums, auxiliary_losses
        ):
            sums["loss"] += float(auxiliary_result["loss"].detach())
            for key in ("xyz_loss", "mask_loss", "uncertainty_loss"):
                sums[key] += float(auxiliary_result[key].detach())
        full_epoch_complete = epoch_batch_count == len(loader)
        epoch_complete = full_epoch_complete or step == args.steps
        if epoch_complete:
            epoch_loss = epoch_loss_sum / epoch_batch_count
            epoch_record = {
                "end_step": step,
                "batch_count": epoch_batch_count,
                "loss_mean": epoch_loss,
                "gradient_norm_mean": (
                    epoch_gradient_norm_sum / epoch_batch_count
                ),
                "gradient_norm_max": epoch_gradient_norm_max,
                "primary": {
                    key: value / epoch_batch_count
                    for key, value in epoch_primary_sums.items()
                },
                "auxiliary": [
                    {
                        "resolution": int(
                            pyramid["auxiliary"][index].shape[-1]
                        ),
                        "weight": auxiliary_loss_weights[index],
                        **{
                            key: value / epoch_batch_count
                            for key, value in sums.items()
                        },
                    }
                    for index, sums in enumerate(epoch_auxiliary_sums)
                ],
            }
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_step = step
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }
                epoch_record["is_best"] = True
            else:
                epoch_record["is_best"] = False
            epoch_history.append(epoch_record)
            epoch_number = len(epoch_history)
            if periodic_epoch_checkpoint_due(
                epoch_number,
                full_epoch=full_epoch_complete,
                every_epochs=args.checkpoint_every_epochs,
            ):
                periodic_checkpoint_dir.mkdir(parents=True, exist_ok=True)
                periodic_path = (
                    periodic_checkpoint_dir
                    / f"epoch-{epoch_number:03d}-step-{step:06d}.pt"
                )
                torch.save(
                    {
                        "model": {
                            name: value.detach().cpu().clone()
                            for name, value in model.state_dict().items()
                        },
                        **checkpoint_metadata,
                        "training_step": step,
                        "training_epoch": epoch_number,
                    },
                    periodic_path,
                )
                periodic_checkpoints.append(
                    {
                        "epoch": epoch_number,
                        "step": step,
                        "path": str(periodic_path.resolve()),
                    }
                )
                epoch_record["checkpoint"] = str(periodic_path.resolve())
            epoch_loss_sum = 0.0
            epoch_batch_count = 0
            epoch_primary_sums = {
                key: 0.0 for key in epoch_primary_sums
            }
            epoch_gradient_norm_sum = 0.0
            epoch_gradient_norm_max = 0.0
            epoch_auxiliary_sums = [
                {key: 0.0 for key in sums}
                for sums in epoch_auxiliary_sums
            ]

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            with torch.no_grad():
                mask_prediction = mask_logits > 0.0
                mask_truth = target_mask > 0.5
                intersection = torch.logical_and(mask_prediction, mask_truth).sum()
                union = torch.logical_or(mask_prediction, mask_truth).sum().clamp_min(1)
                distances = torch.linalg.vector_norm(
                    predicted_xyz - target_xyz, dim=1
                )[mask_truth[:, 0]]
                record = {
                    "step": step,
                    "loss": loss_value,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "learning_rates": scheduler.get_last_lr(),
                    "xyz_loss": float(xyz_loss.detach()),
                    "mask_loss": float(mask_loss.detach()),
                    "uncertainty_loss": float(uncertainty_loss.detach()),
                    "gradient_norm": gradient_norm_value,
                    "auxiliary": [
                        {
                            "resolution": int(auxiliary.shape[-1]),
                            "weight": weight,
                            "loss": float(values["loss"].detach()),
                            "xyz_loss": float(values["xyz_loss"].detach()),
                            "mask_loss": float(values["mask_loss"].detach()),
                            "uncertainty_loss": float(
                                values["uncertainty_loss"].detach()
                            ),
                        }
                        for weight, auxiliary, values in zip(
                            auxiliary_loss_weights,
                            pyramid["auxiliary"] if pyramid else (),
                            auxiliary_losses,
                        )
                    ],
                    "mask_iou_batch": float(intersection / union),
                    "xyz_median_batch": float(distances.median()),
                    "xyz_p95_batch": float(torch.quantile(distances, 0.95)),
                }
                history.append(record)
                print(
                    f"step {step:05d}: loss {record['loss']:.5f}, "
                    f"XYZ med {record['xyz_median_batch']:.4f}, "
                    f"p95 {record['xyz_p95_batch']:.4f}, "
                    f"mask IoU {record['mask_iou_batch']:.4f}, "
                    f"grad {record['gradient_norm']:.3f}",
                    flush=True,
                )

    training_elapsed_seconds = time.monotonic() - started
    training_cache_info = dataset.cache_info()
    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")

    final_checkpoint_path = None
    if args.steps:
        final_checkpoint_path = args.output_dir / "head-final.pt"
        torch.save(
            {
                "model": {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                },
                **checkpoint_metadata,
            },
            final_checkpoint_path,
        )
    model.load_state_dict(best_state)
    checkpoint_path = args.output_dir / "head.pt"
    torch.save(
        {
            "model": model.state_dict(),
            **checkpoint_metadata,
        },
        checkpoint_path,
    )

    model.eval()
    evaluations = []
    with torch.no_grad():
        for sample in dataset:
            identifier = sample["id"]
            features = torch.from_numpy(sample["features"]).unsqueeze(0).to(
                device, dtype=torch.float32
            )
            if args.evaluation_feature_ablation == "zero":
                features.zero_()
            elif args.evaluation_feature_ablation == "spatial-roll":
                features = torch.roll(
                    features,
                    shifts=(features.shape[-2] // 2, features.shape[-1] // 2),
                    dims=(-2, -1),
                )
            camera = torch.from_numpy(sample["camera"]).unsqueeze(0).to(device)
            rays = torch.from_numpy(sample["rays"]).unsqueeze(0).to(device)
            timestep = torch.as_tensor(sample["timestep"]).unsqueeze(0).to(device)
            vl_conditioning = (
                torch.from_numpy(sample["vl_conditioning"])
                .unsqueeze(0)
                .to(device, dtype=torch.float32)
                if "vl_conditioning" in sample
                else None
            )
            if vl_conditioning is not None:
                if args.evaluation_vl_ablation == "zero":
                    vl_conditioning.zero_()
                elif args.evaluation_vl_ablation == "spatial-roll":
                    vl_conditioning = torch.roll(
                        vl_conditioning,
                        shifts=(
                            vl_conditioning.shape[3] // 2,
                            vl_conditioning.shape[4] // 2,
                        ),
                        dims=(3, 4),
                    )
                elif args.evaluation_vl_ablation == "source-swap":
                    vl_conditioning = vl_conditioning.flip(2)
            target_xyz = functional.interpolate(
                torch.from_numpy(sample["xyz"]).unsqueeze(0).to(device),
                size=(target_resolution, target_resolution),
                mode="nearest",
            )
            target_mask = functional.interpolate(
                torch.from_numpy(sample["mask"]).unsqueeze(0).to(device),
                size=(target_resolution, target_resolution),
                mode="nearest",
            ) > 0.5
            prediction = model(
                features,
                camera,
                rays,
                target_resolution,
                timestep=timestep,
                context=vl_conditioning,
            )
            predicted_xyz = prediction[:, :3]
            predicted_mask = prediction[:, 3:4] > 0.0
            predicted_log_error = prediction[:, 4]
            distances = torch.linalg.vector_norm(
                predicted_xyz - target_xyz, dim=1
            )[target_mask[:, 0]]
            uncertainty = predicted_log_error[target_mask[:, 0]]
            if distances.numel() > 1:
                uncertainty_correlation = float(
                    torch.corrcoef(
                        torch.stack(
                            (
                                uncertainty.float(),
                                torch.log(distances.float() + 1.0e-6),
                            )
                        )
                    )[0, 1]
                )
            else:
                uncertainty_correlation = 0.0
            uncertainty_cutoff = torch.quantile(uncertainty, 0.9)
            retained_distances = distances[uncertainty <= uncertainty_cutoff]
            intersection = torch.logical_and(predicted_mask, target_mask).sum()
            union = torch.logical_or(predicted_mask, target_mask).sum().clamp_min(1)
            evaluations.append(
                {
                    "id": identifier,
                    "view_id": sample["view_id"],
                    "replica": sample["replica"],
                    "schedule_step": sample["schedule_step"],
                    "normalized_timestep": float(sample["timestep"]),
                    "xyz_median": float(distances.median()),
                    "xyz_mean": float(distances.mean()),
                    "xyz_p90": float(torch.quantile(distances, 0.90)),
                    "xyz_p95": float(torch.quantile(distances, 0.95)),
                    "xyz_within_2_percent": float((distances <= 0.02).float().mean()),
                    "xyz_within_5_percent": float((distances <= 0.05).float().mean()),
                    "xyz_within_10_percent": float((distances <= 0.10).float().mean()),
                    "uncertainty_log_error_correlation": uncertainty_correlation,
                    "xyz_p95_after_dropping_top_10_percent_uncertainty": float(
                        torch.quantile(retained_distances, 0.95)
                    ),
                    "mask_iou": float(intersection / union),
                }
            )
            mask_array = predicted_mask[0, 0].cpu().numpy()
            xyz_array = predicted_xyz[0].permute(1, 2, 0).cpu().numpy()
            _write_rgb(
                diagnostics_dir / f"{identifier}-xyz.png",
                _xyz_image(xyz_array, mask_array),
            )
            mask_rgb = np.repeat(mask_array[:, :, None], 3, axis=2) * 255
            _write_rgb(diagnostics_dir / f"{identifier}-mask.png", mask_rgb)

    evaluation_by_schedule_step = {}
    for schedule_step in sorted({item["schedule_step"] for item in evaluations}):
        items = [
            item for item in evaluations
            if item["schedule_step"] == schedule_step
        ]
        evaluation_by_schedule_step[str(schedule_step)] = {
            "sample_count": len(items),
            "xyz_median_mean": float(
                np.mean([item["xyz_median"] for item in items])
            ),
            "xyz_p95_max": float(max(item["xyz_p95"] for item in items)),
            "xyz_within_5_percent_mean": float(
                np.mean([item["xyz_within_5_percent"] for item in items])
            ),
            "mask_iou_mean": float(np.mean([item["mask_iou"] for item in items])),
        }

    summary = {
        "schema": "diffusion-editor.canonical-pointmap-training",
        "schema_version": 1,
        "features": [str(path) for path in feature_roots],
        "vl_conditioning": [str(path) for path in vl_roots],
        "identities": identity_records,
        "views": sorted({sample["view_id"] for sample in feature_samples}),
        "identity_view_count": selected_feature_view_count,
        "sample_count": len(feature_samples),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "target_resolution": target_resolution,
        "ray_channels": ray_channels,
        "timestep_channels": timestep_channels,
        "learning_rate": args.learning_rate,
        "vl_learning_rate": args.vl_learning_rate,
        "weight_decay": args.weight_decay,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "freeze_backbone": args.freeze_backbone,
        "trainable_scope": trainable_scope,
        "optimizer_initial_learning_rates": [
            group["initial_lr"] for group in optimizer.param_groups
        ],
        "architecture": architecture,
        "auxiliary_loss_weights": list(auxiliary_loss_weights),
        "initial_checkpoint": (
            str(args.checkpoint.resolve()) if args.checkpoint else None
        ),
        "best_checkpoint": str(checkpoint_path.resolve()),
        "final_checkpoint": (
            str(final_checkpoint_path.resolve())
            if final_checkpoint_path is not None
            else None
        ),
        "checkpoint_every_epochs": args.checkpoint_every_epochs,
        "periodic_checkpoints": periodic_checkpoints,
        "best_step": best_step,
        "best_loss": best_loss if args.steps else None,
        "elapsed_seconds": time.monotonic() - started,
        "training_elapsed_seconds": training_elapsed_seconds,
        "training_samples_per_second": (
            len(sampled_ids) / training_elapsed_seconds
            if training_elapsed_seconds > 0.0
            else 0.0
        ),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "host_peak_rss_bytes": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        ),
        "loader": {
            "identity_balanced_sampling": args.identity_balanced_sampling,
            "training_cache": training_cache_info,
            "final_cache": dataset.cache_info(),
            "sample_order_sha256": hashlib.sha256(
                "\n".join(sampled_ids).encode("utf-8")
            ).hexdigest(),
            "sample_order_prefix": sampled_ids[:64],
        },
        "history": history,
        "epochs": epoch_history,
        "evaluation": evaluations,
        "evaluation_by_schedule_step": evaluation_by_schedule_step,
        "evaluation_summary": {
            "xyz_median_mean": float(
                np.mean([item["xyz_median"] for item in evaluations])
            ),
            "xyz_p95_max": float(max(item["xyz_p95"] for item in evaluations)),
            "xyz_p90_max": float(max(item["xyz_p90"] for item in evaluations)),
            "xyz_within_5_percent_mean": float(
                np.mean([item["xyz_within_5_percent"] for item in evaluations])
            ),
            "uncertainty_log_error_correlation_mean": float(
                np.mean(
                    [
                        item["uncertainty_log_error_correlation"]
                        for item in evaluations
                    ]
                )
            ),
            "xyz_p95_after_uncertainty_drop_max": float(
                max(
                    item["xyz_p95_after_dropping_top_10_percent_uncertainty"]
                    for item in evaluations
                )
            ),
            "mask_iou_mean": float(
                np.mean([item["mask_iou"] for item in evaluations])
            ),
            "mask_iou_min": float(min(item["mask_iou"] for item in evaluations)),
        },
        "checkpoint": checkpoint_path.name,
        "initialized_from_checkpoint": (
            str(args.checkpoint.resolve()) if args.checkpoint is not None else None
        ),
        "initialized_backbone_checkpoint": (
            str(args.initialize_backbone_checkpoint.resolve())
            if args.initialize_backbone_checkpoint is not None
            else None
        ),
        "evaluation_feature_ablation": args.evaluation_feature_ablation,
        "evaluation_vl_ablation": args.evaluation_vl_ablation,
    }
    output_manifest_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["evaluation_summary"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
