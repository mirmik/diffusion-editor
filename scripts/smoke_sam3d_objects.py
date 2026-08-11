#!/usr/bin/env python3
"""Run a staged SAM 3D Objects smoke without touching a running editor."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import types


DEFAULT_ROOT = Path("/home/mirmik/soft/sam-3d-objects")


def _install_kaolin_shape_check_stub() -> None:
    """Avoid loading Kaolin's unrelated binary extension.

    SAM 3D Objects imports only ``kaolin.utils.testing.check_tensor`` from
    Kaolin.  The published Kaolin wheel is tied to the official Torch 2.5 ABI,
    while the RTX 5090 runtime needs current Torch/CUDA.  A shape check has no
    reason to pull in that binary dependency.
    """

    def check_tensor(value, shape, throw=True, **_kwargs):
        valid = len(value.shape) == len(shape) and all(
            expected is None or int(actual) == int(expected)
            for actual, expected in zip(value.shape, shape)
        )
        if throw and not valid:
            raise ValueError(f"expected tensor shape {shape}, got {value.shape}")
        return valid

    kaolin = types.ModuleType("kaolin")
    utils = types.ModuleType("kaolin.utils")
    testing = types.ModuleType("kaolin.utils.testing")
    testing.check_tensor = check_tensor
    utils.testing = testing
    kaolin.utils = utils
    sys.modules.update({
        "kaolin": kaolin,
        "kaolin.utils": utils,
        "kaolin.utils.testing": testing,
    })


def _save_point_cloud(path: Path, points, colors=None) -> int:
    import numpy as np
    import trimesh

    xyz = np.asarray(points.detach().float().cpu()).reshape(-1, 3)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    rgba = None
    if colors is not None:
        rgb = np.asarray(colors.detach().float().cpu()).reshape(-1, 3)[finite]
        rgb = np.clip(rgb, 0.0, 1.0)
        rgba = np.concatenate(
            ((rgb * 255.0).astype(np.uint8), np.full((len(rgb), 1), 255, np.uint8)),
            axis=1,
        )
    trimesh.PointCloud(xyz, colors=rgba).export(path)
    return len(xyz)


def _force_gsplat_texture_renderer() -> None:
    """Route texture-bake Gaussian views through the installed gsplat backend.

    The released ``render_multiview`` helper does not expose its renderer option
    and therefore always falls back to the optional Inria extension.  The rest
    of the renderer already has a maintained gsplat path.
    """
    from sam3d_objects.model.backbone.tdfy_dit.utils import (
        postprocessing_utils,
        render_utils,
    )

    def render_multiview(sample, resolution=512, nviews=30):
        cameras = [
            render_utils.sphere_hammersley_sequence(index, nviews)
            for index in range(nviews)
        ]
        extrinsics, intrinsics = (
            render_utils.yaw_pitch_r_fov_to_extrinsics_intrinsics(
                [camera[0] for camera in cameras],
                [camera[1] for camera in cameras],
                2,
                40,
            )
        )
        result = render_utils.render_frames(
            sample,
            extrinsics,
            intrinsics,
            {
                "resolution": resolution,
                "bg_color": (0, 0, 0),
                "backend": "gsplat",
            },
        )
        return result["color"], extrinsics, intrinsics

    postprocessing_utils.render_multiview = render_multiview


def run(args) -> dict:
    root = args.root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("LIDRA_SKIP_INIT", "true")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPCONV_ALGO", "native")
    os.environ.setdefault("CONDA_PREFIX", str(root / "venv"))
    sys.path.insert(0, str(root))
    os.chdir(root)
    _install_kaolin_shape_check_stub()

    import numpy as np
    import torch
    import trimesh
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from PIL import Image

    _force_gsplat_texture_renderer()

    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3D Objects requires a CUDA GPU")
    started = time.monotonic()
    timings = {}
    torch.cuda.reset_peak_memory_stats()

    config = OmegaConf.load(config_path)
    # Keep PyTorch3D as a CPU-only structures dependency on the Blackwell
    # runtime; nvdiffrast handles the actual CUDA rasterization.
    config.rendering_engine = "nvdiffrast"
    config.compile_model = False
    config.workspace_dir = str(config_path.parent)
    mark = time.monotonic()
    pipeline = instantiate(config)
    timings["load_models_seconds"] = time.monotonic() - mark

    rgba = np.asarray(Image.open(args.image).convert("RGBA"))
    if args.mask is not None:
        mask_image = Image.open(args.mask)
        if "A" in mask_image.getbands():
            alpha = np.asarray(mask_image.getchannel("A"))
        else:
            alpha = np.asarray(mask_image.convert("L"))
    else:
        alpha = rgba[..., 3]
    mask = alpha > args.mask_threshold
    if not mask.any():
        raise ValueError("the input mask is empty")
    rgba = rgba.copy()
    rgba[..., 3] = mask.astype(np.uint8) * 255
    Image.fromarray(rgba).save(output / "conditioning.png")
    Image.fromarray(rgba[..., 3]).save(output / "mask.png")

    with pipeline.device:
        mark = time.monotonic()
        pointmap_dict = pipeline.compute_pointmap(rgba, None)
        timings["pointmap_seconds"] = time.monotonic() - mark
        pointmap = pointmap_dict["pointmap"]
        point_count = _save_point_cloud(
            output / "pointmap.ply",
            pointmap.permute(1, 2, 0),
            pointmap_dict["pts_color"].permute(1, 2, 0),
        )

        ss_input = pipeline.preprocess_image(
            rgba, pipeline.ss_preprocessor, pointmap=pointmap
        )
        slat_input = pipeline.preprocess_image(rgba, pipeline.slat_preprocessor)
        torch.manual_seed(args.seed)
        mark = time.monotonic()
        sparse = pipeline.sample_sparse_structure(
            ss_input, inference_steps=args.stage1_steps
        )
        timings["sparse_structure_seconds"] = time.monotonic() - mark
        sparse.update(
            pipeline.pose_decoder(
                sparse,
                scene_scale=ss_input.get("pointmap_scale"),
                scene_shift=ss_input.get("pointmap_shift"),
            )
        )
        sparse["scale"] = sparse["scale"] * sparse["downsample_factor"]
        coords = sparse["coords"]
        voxel = coords[:, 1:].float() / 64.0 - 0.5
        _save_point_cloud(output / "sparse-occupancy.ply", voxel)
        torch.save(
            {key: value.detach().cpu() for key, value in sparse.items()
             if isinstance(value, torch.Tensor)},
            output / "sparse-structure.pt",
        )

        mark = time.monotonic()
        slat = pipeline.sample_slat(
            slat_input, coords, inference_steps=args.stage2_steps
        )
        timings["structured_latent_seconds"] = time.monotonic() - mark
        torch.save(
            {"coords": slat.coords.detach().cpu(), "feats": slat.feats.detach().cpu()},
            output / "structured-latent.pt",
        )

        mark = time.monotonic()
        decoded = pipeline.decode_slat(slat, ["mesh", "gaussian"])
        timings["decode_seconds"] = time.monotonic() - mark
        mesh = decoded["mesh"][0]
        raw = trimesh.Trimesh(
            mesh.vertices.detach().float().cpu().numpy(),
            mesh.faces.detach().cpu().numpy(),
            process=False,
        )
        raw.export(output / "raw-mesh.glb")
        decoded["gaussian"][0].save_ply(output / "gaussian-splat.ply")

        mark = time.monotonic()
        processed = pipeline.postprocess_slat_output(
            decoded,
            with_mesh_postprocess=not args.skip_mesh_postprocess,
            with_texture_baking=not args.skip_texture_baking,
            use_vertex_color=args.use_vertex_color,
        )
        timings["mesh_texture_postprocess_seconds"] = time.monotonic() - mark
        if processed["glb"] is None:
            raise RuntimeError("SAM 3D Objects produced no GLB")
        processed["glb"].export(output / "final.glb")

    stats = {
        "seed": args.seed,
        "stage1_steps": args.stage1_steps,
        "stage2_steps": args.stage2_steps,
        "point_count": point_count,
        "sparse_count": int(coords.shape[0]),
        "raw_vertices": int(len(raw.vertices)),
        "raw_faces": int(len(raw.faces)),
        "rotation": sparse["rotation"].detach().float().cpu().tolist(),
        "translation": sparse["translation"].detach().float().cpu().tolist(),
        "scale": sparse["scale"].detach().float().cpu().tolist(),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "total_seconds": time.monotonic() - started,
        "timings": timings,
    }
    (output / "stats.json").write_text(
        json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mask", type=Path)
    parser.add_argument("--mask-threshold", type=int, default=127)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--config", type=Path,
        default=DEFAULT_ROOT / "checkpoints/pipeline.yaml",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stage1-steps", type=int)
    parser.add_argument("--stage2-steps", type=int)
    parser.add_argument("--skip-mesh-postprocess", action="store_true")
    parser.add_argument("--skip-texture-baking", action="store_true")
    parser.add_argument("--use-vertex-color", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
