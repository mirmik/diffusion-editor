#!/usr/bin/env python3
"""Texture an enlarged local TRELLIS.2 head from one close-up image.

This is the texture companion to experiment-trellis2-local-face-cube-refine.py.
It resumes from an already refined Shape-SLat, samples a fresh Texture-SLat
conditioned on the close-up, and exports both local-cube and body-space PBR
GLBs.  It deliberately does not blend the new material into the body yet.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape-slat", type=Path, required=True)
    parser.add_argument("--shape-manifest", type=Path, required=True)
    parser.add_argument("--head-image", type=Path, required=True)
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
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--decimation-target", type=int, default=200_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    return parser.parse_args()


def _load_sparse(path, sp, torch):
    import numpy as np

    with np.load(path, allow_pickle=False) as saved:
        coords = torch.from_numpy(saved["coords"]).int().cuda()
        feats = torch.from_numpy(saved["feats"]).float().cuda()
    return sp.SparseTensor(feats=feats, coords=coords)


def _save_sparse(path: Path, value) -> None:
    import numpy as np

    np.savez_compressed(
        path,
        coords=value.coords.detach().cpu().numpy(),
        feats=value.feats.detach().float().cpu().numpy(),
    )


def _sample_texture(pipeline, condition, shape_slat, *, steps, cfg, seed):
    import numpy as np
    import torch

    model = pipeline.models["tex_slat_flow_model_1024"]
    sampler = pipeline.tex_slat_sampler
    shape_std = torch.tensor(
        pipeline.shape_slat_normalization["std"], device=shape_slat.device
    )[None]
    shape_mean = torch.tensor(
        pipeline.shape_slat_normalization["mean"], device=shape_slat.device
    )[None]
    normalized_shape = (shape_slat - shape_mean) / shape_std

    torch.manual_seed(seed)
    sample = normalized_shape.replace(
        feats=torch.randn(
            normalized_shape.coords.shape[0],
            model.in_channels - normalized_shape.feats.shape[1],
            device=pipeline.device,
        )
    )
    times = np.linspace(1, 0, steps + 1)
    times = 3.0 * times / (1 + 2.0 * times)
    if pipeline.low_vram:
        model.to(pipeline.device)
    try:
        for index, (current, previous) in enumerate(
            zip(times[:-1], times[1:]), 1
        ):
            sample = sampler.sample_once(
                model,
                sample,
                float(current),
                float(previous),
                cond=condition["cond"],
                neg_cond=condition["neg_cond"],
                guidance_strength=cfg,
                guidance_rescale=0.0,
                guidance_interval=(0.6, 0.9),
                concat_cond=normalized_shape,
            ).pred_x_prev
            print(
                f"[texture] {index:02d}/{steps}: "
                f"t={current:.4f}->{previous:.4f}",
                flush=True,
            )
    finally:
        if pipeline.low_vram:
            model.cpu()

    tex_std = torch.tensor(
        pipeline.tex_slat_normalization["std"], device=sample.device
    )[None]
    tex_mean = torch.tensor(
        pipeline.tex_slat_normalization["mean"], device=sample.device
    )[None]
    return sample * tex_std + tex_mean


def main() -> int:
    args = _arguments()
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    for path in (
        args.shape_slat,
        args.shape_manifest,
        args.head_image,
        args.trellis2_root,
        args.model_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    sys.path.insert(0, str(args.trellis2_root.resolve()))

    import numpy as np
    import o_voxel
    from PIL import Image
    import torch
    from trellis2.modules import sparse as sp
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.representations import MeshWithVoxel

    torch.set_grad_enabled(False)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.head_image, output / "head-condition-source.png")
    shape_manifest = json.loads(args.shape_manifest.read_text(encoding="utf-8"))

    print("[load] TRELLIS.2 pipeline and refined Shape-SLat", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(args.model_path))
    pipeline.low_vram = True
    pipeline.cuda()
    shape_slat = _load_sparse(args.shape_slat, sp, torch)

    image = pipeline.preprocess_image(Image.open(args.head_image))
    image.save(output / "head-condition-prepared.png")
    condition = pipeline.get_cond([image], 1024)
    texture_slat = _sample_texture(
        pipeline,
        condition,
        shape_slat,
        steps=args.steps,
        cfg=args.cfg,
        seed=args.seed,
    )
    _save_sparse(output / "head-texture-slat.npz", texture_slat)

    print("[decode] shape and PBR attribute volumes", flush=True)
    shape_meshes, shape_subs = pipeline.decode_shape_slat(
        shape_slat, args.resolution
    )
    tex_voxels = pipeline.decode_tex_slat(texture_slat, shape_subs)
    mesh = shape_meshes[0]
    voxels = tex_voxels[0]
    mesh.fill_holes()
    result = MeshWithVoxel(
        mesh.vertices,
        mesh.faces,
        origin=[-0.5, -0.5, -0.5],
        voxel_size=1 / args.resolution,
        coords=voxels.coords[:, 1:],
        attrs=voxels.feats,
        voxel_shape=torch.Size([*voxels.shape, *voxels.spatial_shape]),
        layout=pipeline.pbr_attr_layout,
    )

    print("[export] baking local PBR head", flush=True)
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
    local_y_up = output / "head-refined-pbr-local-gltf-y-up.glb"
    glb.export(local_y_up, extension_webp=True)

    editor = glb.copy()
    editor.apply_transform(np.asarray((
        (-1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, -1, 0, 0),
        (0, 0, 0, 1),
    ), dtype=np.float64))
    local_editor = output / "head-refined-pbr-local-editor-z-up.glb"
    editor.export(local_editor, extension_webp=True)

    center = np.asarray(shape_manifest["local_center"], dtype=np.float64)
    scale = float(shape_manifest["local_scale"])
    to_world = np.eye(4, dtype=np.float64)
    to_world[:3, :3] /= scale
    to_world[:3, 3] = center
    editor.apply_transform(to_world)
    world_editor = output / "head-refined-pbr-body-space-editor-z-up.glb"
    editor.export(world_editor, extension_webp=True)

    manifest = {
        "experiment": "TRELLIS.2 local enlarged refined-head texturing",
        "shape_slat": str(args.shape_slat.resolve()),
        "shape_manifest": str(args.shape_manifest.resolve()),
        "head_image": str(args.head_image.resolve()),
        "resolution": args.resolution,
        "shape_tokens": int(len(shape_slat.coords)),
        "steps": args.steps,
        "cfg": args.cfg,
        "seed": args.seed,
        "decimation_target": args.decimation_target,
        "texture_size": args.texture_size,
        "local_gltf_y_up_output": str(local_y_up),
        "local_editor_output": str(local_editor),
        "body_space_editor_output": str(world_editor),
        "scope": "fresh texture generation on refined local head; no body material blending",
        "status": "complete",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
