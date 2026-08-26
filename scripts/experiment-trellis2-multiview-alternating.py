#!/usr/bin/env python3
"""Generate one TRELLIS.2 asset by alternating four image conditions.

This is deliberately not a native multi-view pipeline: TRELLIS.2 is still
given exactly one DINO image condition at every Euler step.  The first steps
use the front image as a warm-up; every later step cycles through front,
right, back and left.  The same schedule is applied independently to sparse
occupancy, low- and high-resolution shape, and texture flows.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


VIEW_NAMES = ("front", "right-090", "back", "left-270")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--front", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--back", type=Path, required=True)
    parser.add_argument("--left", type=Path, required=True)
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
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--max-num-tokens", type=int, default=49_152)
    parser.add_argument("--decimation-target", type=int, default=200_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    return parser.parse_args()


def _view_schedule(steps: int, warmup_steps: int) -> list[int]:
    return [
        0 if index < warmup_steps else (index - warmup_steps) % 4
        for index in range(steps)
    ]


def _alternating_sample(
    sampler,
    model,
    noise,
    conditions,
    *,
    stage: str,
    steps: int,
    warmup_steps: int,
    rescale_t: float,
    guidance_strength: float,
    guidance_rescale: float,
    guidance_interval: tuple[float, float],
    **model_kwargs,
):
    import numpy as np

    sample = noise
    times = np.linspace(1, 0, steps + 1)
    times = rescale_t * times / (1 + (rescale_t - 1) * times)
    schedule = _view_schedule(steps, warmup_steps)
    for index, ((current, previous), view_index) in enumerate(
        zip(zip(times[:-1], times[1:]), schedule),
        1,
    ):
        condition = conditions[view_index]
        out = sampler.sample_once(
            model,
            sample,
            float(current),
            float(previous),
            cond=condition["cond"],
            neg_cond=condition["neg_cond"],
            guidance_strength=guidance_strength,
            guidance_rescale=guidance_rescale,
            guidance_interval=guidance_interval,
            **model_kwargs,
        )
        sample = out.pred_x_prev
        print(
            f"[{stage}] {index:02d}/{steps}: {VIEW_NAMES[view_index]} "
            f"t={current:.4f}->{previous:.4f}",
            flush=True,
        )
    return sample


def _save_sparse(path: Path, value) -> None:
    import numpy as np

    np.savez_compressed(
        path,
        coords=value.coords.detach().cpu().numpy(),
        feats=value.feats.detach().float().cpu().numpy(),
    )


def _sample_sparse_structure(pipeline, conditions, args):
    import torch

    model = pipeline.models["sparse_structure_flow_model"]
    noise = torch.randn(
        1,
        model.in_channels,
        model.resolution,
        model.resolution,
        model.resolution,
        device=pipeline.device,
    )
    if pipeline.low_vram:
        model.to(pipeline.device)
    latent = _alternating_sample(
        pipeline.sparse_structure_sampler,
        model,
        noise,
        conditions,
        stage="occupancy",
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        rescale_t=5.0,
        guidance_strength=7.5,
        guidance_rescale=0.7,
        guidance_interval=(0.6, 1.0),
    )
    if pipeline.low_vram:
        model.cpu()

    decoder = pipeline.models["sparse_structure_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
    decoded = decoder(latent) > 0
    if pipeline.low_vram:
        decoder.cpu()
    if decoded.shape[2] != 32:
        ratio = decoded.shape[2] // 32
        decoded = torch.nn.functional.max_pool3d(
            decoded.float(), ratio, ratio, 0
        ) > 0.5
    return torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()


def _sample_shape(pipeline, conditions, model, coords, args, stage: str):
    import torch
    from trellis2.modules.sparse import SparseTensor

    noise = SparseTensor(
        feats=torch.randn(
            coords.shape[0], model.in_channels, device=pipeline.device
        ),
        coords=coords,
    )
    if pipeline.low_vram:
        model.to(pipeline.device)
    latent = _alternating_sample(
        pipeline.shape_slat_sampler,
        model,
        noise,
        conditions,
        stage=stage,
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        rescale_t=3.0,
        guidance_strength=7.5,
        guidance_rescale=0.5,
        guidance_interval=(0.6, 1.0),
    )
    if pipeline.low_vram:
        model.cpu()
    std = torch.tensor(
        pipeline.shape_slat_normalization["std"], device=latent.device
    )[None]
    mean = torch.tensor(
        pipeline.shape_slat_normalization["mean"], device=latent.device
    )[None]
    return latent * std + mean


def _sample_texture(pipeline, conditions, model, shape_slat, args):
    import torch

    std = torch.tensor(
        pipeline.shape_slat_normalization["std"], device=shape_slat.device
    )[None]
    mean = torch.tensor(
        pipeline.shape_slat_normalization["mean"], device=shape_slat.device
    )[None]
    normalized_shape = (shape_slat - mean) / std
    noise = normalized_shape.replace(
        feats=torch.randn(
            normalized_shape.coords.shape[0],
            model.in_channels - normalized_shape.feats.shape[1],
            device=pipeline.device,
        )
    )
    if pipeline.low_vram:
        model.to(pipeline.device)
    latent = _alternating_sample(
        pipeline.tex_slat_sampler,
        model,
        noise,
        conditions,
        stage="texture",
        steps=args.steps,
        warmup_steps=args.warmup_steps,
        rescale_t=3.0,
        guidance_strength=1.0,
        guidance_rescale=0.0,
        guidance_interval=(0.6, 0.9),
        concat_cond=normalized_shape,
    )
    if pipeline.low_vram:
        model.cpu()
    std = torch.tensor(
        pipeline.tex_slat_normalization["std"], device=latent.device
    )[None]
    mean = torch.tensor(
        pipeline.tex_slat_normalization["mean"], device=latent.device
    )[None]
    return latent * std + mean


def main() -> int:
    args = _parse_args()
    if args.steps <= 0 or not 0 <= args.warmup_steps <= args.steps:
        raise ValueError("Expected 0 <= warmup-steps <= steps and steps > 0")
    if args.resolution < 1024 or args.resolution % 128:
        raise ValueError("resolution must be a multiple of 128 and at least 1024")
    image_paths = (args.front, args.right, args.back, args.left)
    for path in (*image_paths, args.trellis2_root, args.model_path):
        if not path.exists():
            raise FileNotFoundError(path)

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    sys.path.insert(0, str(args.trellis2_root.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import numpy as np
    import o_voxel
    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.representations import MeshWithVoxel
    from diffusion_editor.workers.trellis2_staged_runner import (
        _occupancy_preview,
        _point_preview,
        _shape_preview,
    )

    torch.set_grad_enabled(False)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for name, path in zip(VIEW_NAMES, image_paths):
        shutil.copy2(path, output / f"input-{name}{path.suffix.lower()}")

    schedule_names = [VIEW_NAMES[index] for index in _view_schedule(
        args.steps, args.warmup_steps
    )]
    manifest = {
        "experiment": "TRELLIS.2 alternating four-view generation",
        "input_images": {
            name: str(path.resolve()) for name, path in zip(VIEW_NAMES, image_paths)
        },
        "model_path": str(args.model_path.resolve()),
        "seed": args.seed,
        "steps_per_flow": args.steps,
        "warmup_steps_per_flow": args.warmup_steps,
        "schedule_per_flow": schedule_names,
        "flows": ["occupancy", "shape-512", "shape-high-resolution", "texture"],
        "condition_combination": "one view per Euler step; no averaging",
        "refinement": "none",
        "requested_resolution": args.resolution,
        "status": "starting",
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print("[load] TRELLIS.2 pipeline", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(args.model_path))
    pipeline.low_vram = True
    pipeline.cuda()

    prepared = []
    for name, path in zip(VIEW_NAMES, image_paths):
        image = pipeline.preprocess_image(Image.open(path))
        image.save(output / f"prepared-{name}.png")
        prepared.append(image)

    torch.manual_seed(args.seed)
    print("[condition] extracting four 512px DINO conditions", flush=True)
    conditions_512 = [pipeline.get_cond([image], 512) for image in prepared]
    coords = _sample_sparse_structure(pipeline, conditions_512, args)
    manifest["occupancy_tokens"] = int(coords.shape[0])
    _occupancy_preview(coords, 32, output / "occupancy.glb")

    lr_shape = _sample_shape(
        pipeline,
        conditions_512,
        pipeline.models["shape_slat_flow_model_512"],
        coords,
        args,
        "shape-512",
    )
    _save_sparse(output / "shape-512-slat.npz", lr_shape)
    lr_meshes, _ = pipeline.decode_shape_slat(lr_shape, 512)
    _shape_preview(lr_meshes, output / "shape-512.glb")

    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    high_coords = decoder.upsample(lr_shape, upsample_times=4)
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False

    actual_resolution = args.resolution
    while True:
        grid_resolution = actual_resolution // 16
        quantized = torch.cat((
            high_coords[:, :1],
            ((high_coords[:, 1:] + 0.5) / 512 * grid_resolution).int(),
        ), dim=1)
        unique_coords = quantized.unique(dim=0)
        if len(unique_coords) < args.max_num_tokens or actual_resolution == 1024:
            break
        actual_resolution -= 128
    manifest["actual_resolution"] = actual_resolution
    manifest["high_resolution_tokens"] = int(unique_coords.shape[0])
    _point_preview(
        unique_coords,
        actual_resolution // 16,
        output / "high-resolution-coordinates.glb",
    )

    del conditions_512, coords, lr_meshes, high_coords, quantized
    torch.cuda.empty_cache()
    print("[condition] extracting four 1024px DINO conditions", flush=True)
    conditions_1024 = [pipeline.get_cond([image], 1024) for image in prepared]
    shape = _sample_shape(
        pipeline,
        conditions_1024,
        pipeline.models["shape_slat_flow_model_1024"],
        unique_coords,
        args,
        "shape-high-resolution",
    )
    _save_sparse(output / "shape-high-resolution-slat.npz", shape)
    shape_meshes, shape_subs = pipeline.decode_shape_slat(
        shape, actual_resolution
    )
    _shape_preview(shape_meshes, output / "shape-high-resolution.glb")

    texture = _sample_texture(
        pipeline,
        conditions_1024,
        pipeline.models["tex_slat_flow_model_1024"],
        shape,
        args,
    )
    _save_sparse(output / "texture-slat.npz", texture)
    del conditions_1024
    torch.cuda.empty_cache()

    tex_voxels = pipeline.decode_tex_slat(texture, shape_subs)
    meshes = []
    for mesh, voxels in zip(shape_meshes, tex_voxels):
        mesh.fill_holes()
        meshes.append(MeshWithVoxel(
            mesh.vertices,
            mesh.faces,
            origin=[-0.5, -0.5, -0.5],
            voxel_size=1 / actual_resolution,
            coords=voxels.coords[:, 1:],
            attrs=voxels.feats,
            voxel_shape=torch.Size([*voxels.shape, *voxels.spatial_shape]),
            layout=pipeline.pbr_attr_layout,
        ))

    result = meshes[0]
    print("[export] baking PBR GLB", flush=True)
    glb = o_voxel.postprocess.to_glb(
        vertices=result.vertices,
        faces=result.faces,
        attr_volume=result.attrs,
        coords=result.coords,
        attr_layout=result.layout,
        voxel_size=result.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0.0,
        verbose=True,
    )
    # o-voxel already returns a conventional glTF Y-up asset.  Keep that
    # version for Blender and other glTF viewers before also exporting the
    # editor-oriented convention used by the existing TRELLIS.2 backend.
    gltf_path = output / "final-pbr-gltf-y-up.glb"
    glb.export(gltf_path, extension_webp=True)
    glb.apply_transform(np.asarray((
        (-1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, -1, 0, 0),
        (0, 0, 0, 1),
    ), dtype=np.float64))
    final_path = output / "final-pbr.glb"
    glb.export(final_path, extension_webp=True)

    manifest["status"] = "complete"
    manifest["output"] = str(final_path)
    manifest["gltf_y_up_output"] = str(gltf_path)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
