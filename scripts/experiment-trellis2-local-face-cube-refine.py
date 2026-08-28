#!/usr/bin/env python3
"""Refine a TRELLIS.2 head after enlarging it into a local 3D cube.

The experiment cuts the head from a decoded full-body shape, normalizes that
geometry into a unit cube, encodes it with TRELLIS.2's official 3D Shape VAE
encoder, and runs masked Shape-SLat RePaint from one enlarged front image.
The lower part of the local cube is kept on the encoded source trajectory so
the neck remains an anchor when the refined head is mapped back to the body.
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
    parser.add_argument("--base-shape-mesh", type=Path, required=True)
    parser.add_argument("--base-shape-slat", type=Path, required=True)
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
    parser.add_argument("--head-height-fraction", type=float, default=0.25)
    parser.add_argument("--cube-fill", type=float, default=1.0)
    parser.add_argument("--neck-freeze-fraction", type=float, default=0.24)
    parser.add_argument("--neck-feather-end", type=float, default=0.40)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument(
        "--strength",
        type=float,
        default=1.0,
        help=(
            "Fraction of the flow schedule used for img2img-style refinement; "
            "1 starts from pure noise and 0 preserves the encoded geometry"
        ),
    )
    parser.add_argument("--cfg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--preview-face-target", type=int, default=200_000)
    return parser.parse_args()


def _clip_mesh_z(mesh, cutoff: float, *, keep_above: bool):
    """Clip triangles against a horizontal plane without optional Shapely."""
    import numpy as np
    import trimesh

    vertices = mesh.vertices.tolist()
    faces = []

    def inside(point) -> bool:
        return point[2] >= cutoff if keep_above else point[2] <= cutoff

    for face in mesh.faces:
        polygon = [mesh.vertices[int(index)] for index in face]
        clipped = []
        for current, following in zip(polygon, polygon[1:] + polygon[:1]):
            current_inside = inside(current)
            following_inside = inside(following)
            if current_inside:
                clipped.append(current)
            if current_inside != following_inside:
                denominator = following[2] - current[2]
                if abs(float(denominator)) > 1e-12:
                    amount = (cutoff - current[2]) / denominator
                    clipped.append(current + amount * (following - current))
        if len(clipped) < 3:
            continue
        base = len(vertices)
        vertices.extend(np.asarray(clipped).tolist())
        for index in range(1, len(clipped) - 1):
            faces.append((base, base + index, base + index + 1))

    result = trimesh.Trimesh(
        vertices=np.asarray(vertices, dtype=np.float32),
        faces=np.asarray(faces, dtype=np.int64),
        process=False,
    )
    result.remove_unreferenced_vertices()
    result.merge_vertices(digits_vertex=7)
    if hasattr(result, "unique_faces"):
        result.update_faces(result.unique_faces())
    if hasattr(result, "nondegenerate_faces"):
        result.update_faces(result.nondegenerate_faces())
    result.remove_unreferenced_vertices()
    return result


def _load_internal_preview(path: Path):
    """Undo the editor preview transform used by trellis2_staged_runner."""
    import numpy as np
    import trimesh

    scene = trimesh.load(path, force="scene", process=False)
    mesh = trimesh.util.concatenate(tuple(scene.geometry.values()))
    mesh.apply_transform(np.diag((-1.0, 1.0, -1.0, 1.0)))
    return mesh


def _normalization(mesh, fill: float):
    """Fit uniformly while anchoring the neck cut to the cube's lower face."""
    lower, upper = mesh.bounds
    side = float((upper - lower).max())
    scale = float(fill / side)
    center = (lower + upper) * 0.5
    center[2] = lower[2] + 0.5 / scale
    return center, scale


def _transformed(mesh, transform):
    result = mesh.copy()
    result.apply_transform(transform)
    return result


def _local_mesh(mesh, center, scale):
    import numpy as np

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = -center * scale
    return _transformed(mesh, transform)


def _world_mesh(mesh, center, scale):
    import numpy as np

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] /= scale
    transform[:3, 3] = center
    return _transformed(mesh, transform)


def _gltf_y_up(mesh):
    import numpy as np

    return _transformed(mesh, np.asarray((
        (1, 0, 0, 0),
        (0, 0, 1, 0),
        (0, -1, 0, 0),
        (0, 0, 0, 1),
    ), dtype=np.float64))


def _save_sparse(path: Path, value) -> None:
    import numpy as np

    np.savez_compressed(
        path,
        coords=value.coords.detach().cpu().numpy(),
        feats=value.feats.detach().float().cpu().numpy(),
    )


def _mesh_to_shape_slat(mesh, encoder, resolution, sp, torch, o_voxel):
    vertices = torch.from_numpy(mesh.vertices.copy()).float()
    faces = torch.from_numpy(mesh.faces.copy()).long()
    voxel_indices, dual_vertices, intersected = (
        o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices,
            faces,
            grid_size=resolution,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
    )
    local_vertices = torch.clamp(
        dual_vertices * resolution - voxel_indices, 0, 1
    )
    coords = torch.cat((
        torch.zeros_like(voxel_indices[:, :1]),
        voxel_indices,
    ), dim=1).int()
    vertices_sparse = sp.SparseTensor(local_vertices.float(), coords)
    intersections_sparse = vertices_sparse.replace(intersected.bool())
    latent = encoder(
        vertices_sparse.cuda(),
        intersections_sparse.cuda(),
        sample_posterior=False,
    )
    return latent, len(voxel_indices)


def _neck_mask(coords, freeze_fraction: float, feather_end: float):
    vertical = coords[:, 3].float()
    minimum = vertical.min()
    span = (vertical.max() - minimum).clamp(min=1.0)
    from_bottom = (vertical - minimum) / span
    weights = (
        (from_bottom - freeze_fraction) / (feather_end - freeze_fraction)
    ).clamp(0, 1)
    weights = weights * weights * (3 - 2 * weights)
    return weights[:, None]


def _repaint(
    pipeline,
    source_slat,
    condition,
    mask,
    *,
    steps: int,
    strength: float,
    cfg: float,
    seed: int,
):
    import numpy as np
    import torch

    sampler = pipeline.shape_slat_sampler
    model = pipeline.models["shape_slat_flow_model_1024"]
    sigma_min = float(sampler.sigma_min)
    std = torch.tensor(
        pipeline.shape_slat_normalization["std"], device=source_slat.device
    )[None]
    mean = torch.tensor(
        pipeline.shape_slat_normalization["mean"], device=source_slat.device
    )[None]
    source = (source_slat - mean) / std
    torch.manual_seed(seed)
    source_epsilon = torch.randn_like(source.feats)
    times = np.linspace(strength, 0, steps + 1)
    times = 3.0 * times / (1 + 2.0 * times)
    start = float(times[0])
    start_sigma = sigma_min + (1.0 - sigma_min) * start
    sample = source.replace(
        (1.0 - start) * source.feats + start_sigma * source_epsilon
    )
    model.to(pipeline.device)
    try:
        for index, (current, previous) in enumerate(
            zip(times[:-1], times[1:]), 1
        ):
            edited = sampler.sample_once(
                model,
                sample,
                float(current),
                float(previous),
                cond=condition["cond"],
                neg_cond=condition["neg_cond"],
                guidance_strength=cfg,
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
                f"[local-repaint] {index:02d}/{steps}: "
                f"t={current:.4f}->{previous:.4f}",
                flush=True,
            )
    finally:
        if pipeline.low_vram:
            model.cpu()
    sample = sample.replace(
        mask * sample.feats + (1.0 - mask) * source.feats
    )
    return sample * std + mean


def _decoded_mesh(pipeline, slat, resolution, trimesh, face_target: int):
    meshes, _ = pipeline.decode_shape_slat(slat, resolution)
    mesh = meshes[0]
    if len(mesh.faces) > face_target:
        mesh.simplify(face_target)
    return trimesh.Trimesh(
        vertices=mesh.vertices.detach().float().cpu().numpy(),
        faces=mesh.faces.detach().long().cpu().numpy(),
        process=False,
    )


def _combined(body, head, trimesh):
    result = trimesh.util.concatenate((body.copy(), head.copy()))
    result.remove_unreferenced_vertices()
    return result


def main() -> int:
    args = _arguments()
    if not 0 < args.head_height_fraction < 1:
        raise ValueError("head-height-fraction must be in (0, 1)")
    if args.cube_fill != 1.0:
        raise ValueError("cube-fill must be exactly 1.0; padding is disabled")
    if not 0 <= args.neck_freeze_fraction < args.neck_feather_end <= 1:
        raise ValueError("invalid neck mask fractions")
    if not 0 <= args.strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    for path in (
        args.base_shape_mesh,
        args.base_shape_slat,
        args.head_image,
        args.trellis2_root,
        args.model_path,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    sys.path.insert(0, str(args.trellis2_root.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import o_voxel
    from PIL import Image
    import torch
    import trimesh
    import trellis2.models as models
    from trellis2.modules import sparse as sp
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from diffusion_editor.workers.trellis2_staged_runner import _point_preview

    torch.set_grad_enabled(False)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.head_image, output / "head-condition-source.png")

    base = _load_internal_preview(args.base_shape_mesh)
    vertical_min, vertical_max = base.bounds[:, 2]
    cutoff = float(
        vertical_max
        - args.head_height_fraction * (vertical_max - vertical_min)
    )
    head = _clip_mesh_z(base, cutoff, keep_above=True)
    body = _clip_mesh_z(base, cutoff, keep_above=False)
    center, scale = _normalization(head, args.cube_fill)
    head_local = _local_mesh(head, center, scale)
    base.export(output / "base-shape-editor-z-up.glb")
    head.export(output / "head-cut-editor-z-up.glb")
    head_local.export(output / "head-cut-local-cube.glb")
    _gltf_y_up(head_local).export(output / "head-cut-local-cube-gltf-y-up.glb")

    print("[load] TRELLIS.2 pipeline and Shape VAE encoder", flush=True)
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(args.model_path))
    pipeline.low_vram = True
    pipeline.cuda()
    encoder_path = (
        args.model_path / "ckpts" / "shape_enc_next_dc_f16c32_fp16"
    )
    encoder = models.from_pretrained(str(encoder_path)).eval().cuda()
    source_slat, source_voxels = _mesh_to_shape_slat(
        head_local,
        encoder,
        args.resolution,
        sp,
        torch,
        o_voxel,
    )
    encoder.cpu()
    del encoder
    torch.cuda.empty_cache()
    _save_sparse(output / "head-source-encoded-slat.npz", source_slat)

    source_roundtrip_local = _decoded_mesh(
        pipeline,
        source_slat,
        args.resolution,
        trimesh,
        args.preview_face_target,
    )
    source_roundtrip_world = _world_mesh(
        source_roundtrip_local, center, scale
    )
    source_roundtrip_local.export(output / "head-source-roundtrip-local.glb")
    _gltf_y_up(source_roundtrip_local).export(
        output / "head-source-roundtrip-local-gltf-y-up.glb"
    )
    source_roundtrip_world.export(
        output / "head-source-roundtrip-editor-z-up.glb"
    )

    head_image = pipeline.preprocess_image(Image.open(args.head_image))
    head_image.save(output / "head-condition-prepared.png")
    condition = pipeline.get_cond([head_image], 1024)
    mask = _neck_mask(
        source_slat.coords,
        args.neck_freeze_fraction,
        args.neck_feather_end,
    ).to(source_slat.device, source_slat.feats.dtype)
    selected = mask[:, 0] > 0.01
    _point_preview(
        source_slat.coords[selected],
        args.resolution // 16,
        output / "head-edit-mask-tokens.glb",
    )
    refined_slat = _repaint(
        pipeline,
        source_slat,
        condition,
        mask,
        steps=args.steps,
        strength=args.strength,
        cfg=args.cfg,
        seed=args.seed,
    )
    _save_sparse(output / "head-refined-slat.npz", refined_slat)
    refined_local = _decoded_mesh(
        pipeline,
        refined_slat,
        args.resolution,
        trimesh,
        args.preview_face_target,
    )
    refined_world = _world_mesh(refined_local, center, scale)
    refined_local.export(output / "head-refined-local.glb")
    _gltf_y_up(refined_local).export(
        output / "head-refined-local-gltf-y-up.glb"
    )
    refined_world.export(output / "head-refined-editor-z-up.glb")

    source_composite = _combined(body, source_roundtrip_world, trimesh)
    refined_composite = _combined(body, refined_world, trimesh)
    source_composite.export(output / "body-plus-source-head-editor-z-up.glb")
    refined_composite.export(output / "body-plus-refined-head-editor-z-up.glb")
    _gltf_y_up(source_composite).export(
        output / "body-plus-source-head-gltf-y-up.glb"
    )
    _gltf_y_up(refined_composite).export(
        output / "body-plus-refined-head-gltf-y-up.glb"
    )

    delta = (refined_slat.feats - source_slat.feats).norm(dim=1)
    frozen = mask[:, 0] <= 0.0
    manifest = {
        "experiment": "TRELLIS.2 enlarged local face cube RePaint",
        "base_shape_mesh": str(args.base_shape_mesh.resolve()),
        "base_shape_slat": str(args.base_shape_slat.resolve()),
        "head_image": str(args.head_image.resolve()),
        "shape_encoder": str(encoder_path),
        "resolution": args.resolution,
        "head_height_fraction": args.head_height_fraction,
        "world_cutoff_z": cutoff,
        "local_center": center.tolist(),
        "local_scale": scale,
        "cube_fill": args.cube_fill,
        "source_surface_voxels": int(source_voxels),
        "source_shape_tokens": int(len(source_slat.coords)),
        "edited_shape_tokens": int(selected.sum().item()),
        "fully_frozen_shape_tokens": int(frozen.sum().item()),
        "neck_freeze_fraction": args.neck_freeze_fraction,
        "neck_feather_end": args.neck_feather_end,
        "steps": args.steps,
        "strength": args.strength,
        "initial_rescaled_t": (
            3.0 * args.strength / (1.0 + 2.0 * args.strength)
        ),
        "cfg": args.cfg,
        "seed": args.seed,
        "preview_face_target": args.preview_face_target,
        "feature_delta_mean": float(delta.mean().item()),
        "feature_delta_edited_mean": float(delta[selected].mean().item()),
        "feature_delta_frozen_max": (
            float(delta[frozen].max().item()) if frozen.any() else 0.0
        ),
        "scope": "shape only; one front head view; no texture refinement",
        "status": "complete",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
