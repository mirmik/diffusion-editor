#!/usr/bin/env python3
"""Run a minimal Easy3E-style local Shape-SLat edit in TRELLIS.2.

The experiment deliberately keeps the sparse coordinates fixed.  It first
reconstructs a source Shape-SLat from ``--source-image`` and then integrates
the difference between target- and source-conditioned flow fields only in a
soft 3D head mask.  This isolates the most important question before we add
sparse-structure editing and local PBR refinement: can TRELLIS.2's shape flow
move the source geometry toward the target while preserving the body?
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--target-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--trellis2-root",
        type=Path,
        default=Path("/home/mirmik/soft/TRELLIS.2"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/home/mirmik/soft/TRELLIS.2/models/TRELLIS.2-4B"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-steps", type=int, default=12)
    parser.add_argument("--edit-steps", type=int, default=16)
    parser.add_argument(
        "--edit-mode",
        choices=("flowedit", "repaint"),
        default="flowedit",
    )
    parser.add_argument("--flow-averages", type=int, default=2)
    parser.add_argument("--cfg-source", type=float, default=5.0)
    parser.add_argument("--cfg-target", type=float, default=5.0)
    parser.add_argument(
        "--head-core",
        type=float,
        default=0.24,
        help="Top fraction of the object fully covered by the head mask.",
    )
    parser.add_argument(
        "--head-feather-end",
        type=float,
        default=0.33,
        help="Top fraction where the soft head mask reaches zero.",
    )
    return parser.parse_args()


def _save_sparse(path: Path, value) -> None:
    import numpy as np

    np.savez_compressed(
        path,
        coords=value.coords.detach().cpu().numpy(),
        feats=value.feats.detach().float().cpu().numpy(),
    )


def _soft_head_mask(coords, core: float, feather_end: float):
    """Return per-token weights for the top of TRELLIS.2's internal Z axis."""
    import torch

    xyz = coords[:, 1:].float()
    vertical = xyz[:, 2]
    upper = vertical.max()
    span = (upper - vertical.min()).clamp(min=1.0)
    from_top = (upper - vertical) / span
    weights = ((feather_end - from_top) / (feather_end - core)).clamp(0, 1)
    # Smoothstep makes the boundary less visible without expanding the mask.
    weights = weights * weights * (3 - 2 * weights)
    return weights[:, None]


def _cfg_velocity(sampler, model, state, time, condition, strength: float):
    return sampler._get_model_prediction(
        model,
        state,
        float(time),
        cond=condition["cond"],
        neg_cond=condition["neg_cond"],
        guidance_strength=float(strength),
        guidance_rescale=0.5,
        guidance_interval=(0.0, 1.0),
    )[2]


def _flow_edit_shape(
    pipeline,
    source_slat,
    source_condition,
    target_condition,
    mask,
    *,
    steps: int,
    averages: int,
    cfg_source: float,
    cfg_target: float,
):
    import numpy as np
    import torch

    sampler = pipeline.shape_slat_sampler
    model = pipeline.models["shape_slat_flow_model_512"]
    sigma_min = float(sampler.sigma_min)
    mean = torch.tensor(
        pipeline.shape_slat_normalization["mean"],
        device=source_slat.device,
    )[None]
    std = torch.tensor(
        pipeline.shape_slat_normalization["std"],
        device=source_slat.device,
    )[None]
    source = (source_slat - mean) / std
    sample = source.replace(source.feats.clone())

    times = np.linspace(1, 0, steps + 1)
    times = 3.0 * times / (1 + 2.0 * times)
    model.to(pipeline.device)
    try:
        for index, (time, previous) in enumerate(zip(times[:-1], times[1:]), 1):
            velocity = torch.zeros_like(source.feats)
            for _ in range(averages):
                epsilon = torch.randn_like(source.feats)
                noisy = (
                    (1.0 - float(time)) * source.feats
                    + (sigma_min + (1.0 - sigma_min) * float(time)) * epsilon
                )
                noisy_source = source.replace(
                    mask * noisy + (1.0 - mask) * source.feats
                )
                noisy_target = sample.replace(
                    sample.feats + noisy_source.feats - source.feats
                )
                source_velocity = _cfg_velocity(
                    sampler, model, noisy_source, time,
                    source_condition, cfg_source,
                )
                target_velocity = _cfg_velocity(
                    sampler, model, noisy_target, time,
                    target_condition, cfg_target,
                )
                velocity += (
                    target_velocity.feats - source_velocity.feats
                ) / float(averages)

            delta_time = float(time - previous)
            edited = sample.feats - delta_time * mask * velocity
            sample = sample.replace(mask * edited + (1.0 - mask) * source.feats)
            print(
                f"[shape-flowedit] {index:02d}/{steps}: "
                f"t={time:.4f}, |delta-v|={velocity.norm().item():.3f}",
                flush=True,
            )
    finally:
        if pipeline.low_vram:
            model.cpu()

    return sample * std + mean


def _repaint_shape(
    pipeline,
    source_slat,
    target_condition,
    mask,
    *,
    steps: int,
    cfg_target: float,
):
    import numpy as np
    import torch

    sampler = pipeline.shape_slat_sampler
    model = pipeline.models["shape_slat_flow_model_512"]
    sigma_min = float(sampler.sigma_min)
    mean = torch.tensor(
        pipeline.shape_slat_normalization["mean"],
        device=source_slat.device,
    )[None]
    std = torch.tensor(
        pipeline.shape_slat_normalization["std"],
        device=source_slat.device,
    )[None]
    source = (source_slat - mean) / std
    sample = source.replace(torch.randn_like(source.feats))
    source_epsilon = torch.randn_like(source.feats)

    times = np.linspace(1, 0, steps + 1)
    times = 3.0 * times / (1 + 2.0 * times)
    model.to(pipeline.device)
    try:
        for index, (time, previous) in enumerate(zip(times[:-1], times[1:]), 1):
            edited = sampler.sample_once(
                model,
                sample,
                float(time),
                float(previous),
                cond=target_condition["cond"],
                neg_cond=target_condition["neg_cond"],
                guidance_strength=float(cfg_target),
                guidance_rescale=0.5,
                guidance_interval=(0.0, 1.0),
            ).pred_x_prev
            source_previous = (
                (1.0 - float(previous)) * source.feats
                + (
                    sigma_min
                    + (1.0 - sigma_min) * float(previous)
                ) * source_epsilon
            )
            sample = sample.replace(
                mask * edited.feats + (1.0 - mask) * source_previous
            )
            print(
                f"[shape-repaint] {index:02d}/{steps}: t={time:.4f}",
                flush=True,
            )
    finally:
        if pipeline.low_vram:
            model.cpu()

    sample = sample.replace(mask * sample.feats + (1.0 - mask) * source.feats)
    return sample * std + mean


def main() -> int:
    args = _parse_args()
    if not 0 <= args.head_core < args.head_feather_end <= 1:
        raise ValueError("Expected 0 <= head-core < head-feather-end <= 1")
    for path in (args.source_image, args.target_image, args.model_path):
        if not path.exists():
            raise FileNotFoundError(path)

    # These must be selected before TRELLIS.2 imports its attention modules.
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    sys.path.insert(0, str(args.trellis2_root.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from diffusion_editor.workers.trellis2_staged_runner import (
        _point_preview,
        _shape_preview,
    )

    torch.set_grad_enabled(False)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.source_image, output / "source-condition.png")
    shutil.copy2(args.target_image, output / "target-condition.png")

    print("[load] TRELLIS.2 pipeline", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(args.model_path))
    pipeline.low_vram = True
    pipeline.cuda()

    source_image = pipeline.preprocess_image(Image.open(args.source_image))
    target_image = pipeline.preprocess_image(Image.open(args.target_image))
    source_image.save(output / "source-preprocessed.png")
    target_image.save(output / "target-preprocessed.png")
    source_condition = pipeline.get_cond([source_image], 512)
    target_condition = pipeline.get_cond([target_image], 512)

    torch.manual_seed(args.seed)
    sparse_params = {
        "steps": args.generation_steps,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.7,
        "guidance_interval": (0.6, 1.0),
        "rescale_t": 5.0,
    }
    shape_params = {
        "steps": args.generation_steps,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "guidance_interval": (0.6, 1.0),
        "rescale_t": 3.0,
    }
    print("[source] sparse structure", flush=True)
    coords = pipeline.sample_sparse_structure(source_condition, 32, 1, sparse_params)
    print(f"[source] shape SLat on {len(coords):,} tokens", flush=True)
    source_slat = pipeline.sample_shape_slat(
        source_condition,
        pipeline.models["shape_slat_flow_model_512"],
        coords,
        shape_params,
    )
    _save_sparse(output / "source-shape-slat.npz", source_slat)

    source_meshes, _ = pipeline.decode_shape_slat(source_slat, 512)
    _shape_preview(source_meshes, output / "source-shape.glb")
    del source_meshes
    torch.cuda.empty_cache()

    mask = _soft_head_mask(
        source_slat.coords,
        args.head_core,
        args.head_feather_end,
    ).to(source_slat.device, source_slat.feats.dtype)
    selected = mask[:, 0] > 0.01
    _point_preview(
        source_slat.coords[selected],
        32,
        output / "head-mask-tokens.glb",
    )
    print(
        f"[mask] {selected.sum().item():,}/{len(mask):,} tokens; "
        f"full-weight={(mask[:, 0] > 0.99).sum().item():,}",
        flush=True,
    )

    torch.manual_seed(args.seed + 1)
    if args.edit_mode == "flowedit":
        edited_slat = _flow_edit_shape(
            pipeline,
            source_slat,
            source_condition,
            target_condition,
            mask,
            steps=args.edit_steps,
            averages=args.flow_averages,
            cfg_source=args.cfg_source,
            cfg_target=args.cfg_target,
        )
    else:
        edited_slat = _repaint_shape(
            pipeline,
            source_slat,
            target_condition,
            mask,
            steps=args.edit_steps,
            cfg_target=args.cfg_target,
        )
    _save_sparse(output / "edited-shape-slat.npz", edited_slat)
    edited_meshes, _ = pipeline.decode_shape_slat(edited_slat, 512)
    _shape_preview(edited_meshes, output / "edited-shape.glb")

    feature_delta = (edited_slat.feats - source_slat.feats).norm(dim=1)
    outside = mask[:, 0] <= 0.0
    manifest = {
        "source_image": str(args.source_image.resolve()),
        "target_image": str(args.target_image.resolve()),
        "model_path": str(args.model_path.resolve()),
        "seed": args.seed,
        "generation_steps": args.generation_steps,
        "edit_steps": args.edit_steps,
        "edit_mode": args.edit_mode,
        "flow_averages": args.flow_averages,
        "cfg_source": args.cfg_source,
        "cfg_target": args.cfg_target,
        "shape_tokens": len(mask),
        "masked_tokens": int(selected.sum().item()),
        "feature_delta_mean": float(feature_delta.mean().item()),
        "feature_delta_masked_mean": float(feature_delta[selected].mean().item()),
        "feature_delta_outside_max": float(feature_delta[outside].max().item()),
        "scope": "shape SLat only; fixed sparse coordinates; no texture edit",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
