#!/usr/bin/env python3
"""Run Material Anything's image-to-PBR estimator on rendered mesh views.

This intentionally bypasses Material Anything's old PyTorch3D/Kaolin renderer
and its text-to-texture stage.  It is useful for comparing the learned PBR
estimator independently from UV baking and mesh topology problems.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stages_dir", type=Path, help="Directory with albedo_N.png and normal_N.png")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--views", default="0", help="Comma-separated view indices (default: 0)")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--normal-mode",
        choices=("rendered", "front-facing"),
        default="rendered",
        help="Use rendered normals or a flat front-facing normal inside the silhouette",
    )
    parser.add_argument(
        "--background-mode",
        choices=("white-preserved", "raw-free"),
        default="white-preserved",
        help="Match upstream white/locked background or diffuse the raw input frame",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/home/mirmik/soft/MaterialAnything"),
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/home/mirmik/soft/MaterialAnything/pretrained_models/material_estimator"),
    )
    return parser.parse_args()


def load_rgb(path: Path, size: int):
    from PIL import Image

    return Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)


def front_facing_normal(rendered_normal):
    """Make a neutral view-space normal while retaining the rendered silhouette."""
    import numpy as np
    from PIL import Image

    source = np.asarray(rendered_normal)
    # Material Anything's renderer uses white outside the object.
    object_mask = np.any(source < 250, axis=2)
    result = np.full_like(source, 255)
    result[object_mask] = (128, 128, 255)
    return Image.fromarray(result, "RGB")


def object_mask_from_normal(rendered_normal):
    import numpy as np

    source = np.asarray(rendered_normal)
    return np.any(source < 250, axis=2)


def white_background(image, object_mask):
    import numpy as np
    from PIL import Image

    result = np.asarray(image).copy()
    result[~object_mask] = 255
    return Image.fromarray(result, "RGB")


def image_tensor(image):
    import numpy as np
    import torch

    data = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(data).unsqueeze(0)


def main() -> int:
    args = parse_args()
    if not args.repo_root.is_dir():
        raise SystemExit(f"Material Anything repository not found: {args.repo_root}")
    if not args.model_root.is_dir():
        raise SystemExit(f"Material estimator weights not found: {args.model_root}")

    view_indices = [int(item.strip()) for item in args.views.split(",") if item.strip()]
    if not view_indices:
        raise SystemExit("At least one view index is required")

    for view_index in view_indices:
        for prefix in ("albedo", "normal"):
            path = args.stages_dir / f"{prefix}_{view_index}.png"
            if not path.is_file():
                raise SystemExit(f"Missing input: {path}")

    # The upstream repository uses absolute imports rooted at its checkout.
    sys.path.insert(0, str(args.repo_root))
    import torch
    from models.scheduling_ddpm import DDPMScheduler
    from pipelines.pipeline_stable_diffusion_switcher import StableDiffusionPipeline

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this experiment")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    print(f"[load] Material Anything estimator: {args.model_root}", flush=True)
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_root,
        torch_dtype=torch.float16,
    ).to("cuda")
    pipeline.scheduler = DDPMScheduler.from_pretrained(args.model_root, subfolder="scheduler")
    load_seconds = time.monotonic() - started

    results = []
    for view_index in view_indices:
        albedo = load_rgb(args.stages_dir / f"albedo_{view_index}.png", args.size)
        rendered_normal = load_rgb(args.stages_dir / f"normal_{view_index}.png", args.size)
        object_mask = object_mask_from_normal(rendered_normal)
        if args.background_mode == "white-preserved":
            albedo = white_background(albedo, object_mask)
        normal = (
            rendered_normal
            if args.normal_mode == "rendered"
            else front_facing_normal(rendered_normal)
        )

        albedo_tensor = image_tensor(albedo)
        neutral_rm = torch.ones_like(albedo_tensor)
        neutral_bump = torch.ones_like(albedo_tensor)
        init_materials = {
            "albedo": albedo_tensor,
            "roughness_metallic": neutral_rm,
            "bump": neutral_bump,
        }
        # One locks pixels to init_materials. Upstream locks the white background
        # while allowing the estimator to generate all visible object pixels.
        if args.background_mode == "white-preserved":
            masks = torch.from_numpy((~object_mask).astype("float32")).unsqueeze(0)
        else:
            masks = torch.zeros((1, args.size, args.size), dtype=torch.float32)

        view_dir = args.output_dir / f"view-{view_index}"
        view_dir.mkdir(parents=True, exist_ok=True)
        albedo.save(view_dir / "condition-albedo.png")
        normal.save(view_dir / "condition-normal.png")

        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        inference_started = time.monotonic()
        print(
            f"[estimate] view={view_index} normal={args.normal_mode} "
            f"background={args.background_mode} steps={args.steps} size={args.size}",
            flush=True,
        )
        with torch.inference_mode():
            output = pipeline(
                prompt=[""],
                cond_image=[albedo],
                normal_image=[normal],
                init_materials=init_materials,
                masks=masks,
                num_inference_steps=args.steps,
                guidance_scale=1.0,
                generator=generator,
                height=args.size,
                width=args.size,
            ).images
        inference_seconds = time.monotonic() - inference_started
        if len(output) != 3:
            raise RuntimeError(f"Expected 3 material images, got {len(output)}")

        estimated_albedo, roughness_metallic, bump = output
        estimated_albedo.save(view_dir / "albedo.png")
        roughness_metallic.save(view_dir / "roughness-metallic.png")
        bump.save(view_dir / "bump.png")
        rm_channels = roughness_metallic.convert("RGB").split()
        rm_channels[1].save(view_dir / "roughness.png")
        rm_channels[2].save(view_dir / "metallic.png")

        peak_gib = torch.cuda.max_memory_allocated() / (1024**3)
        print(
            f"[done] view={view_index} inference={inference_seconds:.2f}s "
            f"peak_allocated={peak_gib:.2f} GiB",
            flush=True,
        )
        results.append(
            {
                "view": view_index,
                "inference_seconds": inference_seconds,
                "peak_allocated_gib": peak_gib,
            }
        )

    manifest = {
        "backend": "Material Anything material_estimator",
        "repo_root": str(args.repo_root),
        "model_root": str(args.model_root),
        "source_stages": str(args.stages_dir),
        "views": view_indices,
        "normal_mode": args.normal_mode,
        "background_mode": args.background_mode,
        "steps": args.steps,
        "size": args.size,
        "seed": args.seed,
        "load_seconds": load_seconds,
        "total_seconds": time.monotonic() - started,
        "results": results,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
