"""Experimental local-1536 Pixal3D geometry generation and registration.

The runner intentionally produces diagnostic patch and overlap artifacts.  It
does not replace the editor's compatible shape checkpoint until multi-scale
fusion has a proven latent representation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


EDITOR_ROOT = Path(__file__).resolve().parents[2]
if str(EDITOR_ROOT) not in sys.path:
    sys.path.insert(0, str(EDITOR_ROOT))

from diffusion_editor.generation.local_detail_geometry import (  # noqa: E402
    fit_local_detail_transform,
    local_roi_bounds,
    overlap_face_masks,
    project_pixal_points,
    sample_mask_bilinear,
)
from diffusion_editor.workers.pixal3d_staged_runner import (  # noqa: E402
    _detail_crop_box,
    _preprocess_refine_pair,
    _preview_transform,
)


def _load_base_shape(args, pipeline, sp, torch, np):
    """Return denormalized base shape latent and its saved camera."""
    path = Path(args.base_checkpoint)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as saved:
            resolution = int(saved["resolution"][0])
            coords = torch.from_numpy(saved["coords"].copy()).to("cuda")
            normalized = torch.from_numpy(
                saved["normalized_feats"].copy()
            ).to("cuda").float()
            camera = {
                "camera_angle_x": float(saved["camera_angle_x"][0]),
                "distance": float(saved["camera_distance"][0]),
                "mesh_scale": float(saved["mesh_scale"][0]),
            }
        std = torch.tensor(
            pipeline.shape_slat_normalization["std"], device="cuda"
        )[None]
        mean = torch.tensor(
            pipeline.shape_slat_normalization["mean"], device="cuda"
        )[None]
        latent = sp.SparseTensor(
            feats=normalized * std + mean,
            coords=coords,
        )
        return latent, resolution, camera

    if not args.base_metadata:
        raise ValueError("legacy .pt checkpoints require --base-metadata")
    saved = torch.load(path, map_location="cpu", weights_only=False)
    metadata = json.loads(
        Path(args.base_metadata).read_text(encoding="utf-8")
    )
    latent = sp.SparseTensor(
        feats=saved["feats"].to("cuda").float(),
        coords=saved["coords"].to("cuda"),
    )
    camera = {
        "camera_angle_x": float(metadata["camera_angle_x"]),
        "distance": float(metadata["camera_distance"]),
        "mesh_scale": float(metadata.get("mesh_scale", 1.0)),
    }
    return latent, int(saved["resolution"]), camera


def _load_legacy_base_context(args, torch):
    if not args.base_metadata:
        raise ValueError("legacy .pt checkpoints require --base-metadata")
    saved = torch.load(
        Path(args.base_checkpoint), map_location="cpu", weights_only=False
    )
    metadata = json.loads(
        Path(args.base_metadata).read_text(encoding="utf-8")
    )
    camera = {
        "camera_angle_x": float(metadata["camera_angle_x"]),
        "distance": float(metadata["camera_distance"]),
        "mesh_scale": float(metadata.get("mesh_scale", 1.0)),
    }
    return int(saved["resolution"]), camera


def _mesh_arrays(mesh, np):
    return (
        mesh.vertices.detach().float().cpu().numpy(),
        mesh.faces.detach().long().cpu().numpy(),
    )


def _load_mesh_tensors(path, torch, np):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    return (
        saved["vertices"].float().numpy().astype(np.float32, copy=False),
        saved["faces"].long().numpy().astype(np.int64, copy=False),
    )


def _decode_shape(pipeline, latent, resolution):
    meshes, _subdivisions = pipeline.decode_shape_slat(latent, resolution)
    return meshes[0]


def _estimate_camera(image, upstream, torch, output_dir):
    temporary = output_dir / "local-camera-input.png"
    image.save(temporary)
    moge = upstream.load_moge_model(device="cuda")
    camera = upstream.get_camera_params_wild_moge(
        str(temporary), moge, device="cuda"
    )
    moge.cpu()
    del moge
    torch.cuda.empty_cache()
    return camera


def _generate_local_shape(
        pipeline,
        image,
        camera,
        *,
        resolution,
        steps,
        seed,
        max_num_tokens,
        sp,
        torch,
):
    """Run the stock Pixal3D cascade through its HR shape latent only."""
    shape_sampler = {
        "steps": steps,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    }
    sparse_sampler = {
        **shape_sampler,
        "guidance_rescale": 0.7,
        "rescale_t": 5.0,
    }
    torch.manual_seed(seed)
    cond_ss = pipeline.get_proj_cond_ss([image], **camera)
    coords = pipeline.sample_sparse_structure(
        cond_ss, 32, 1, sparse_sampler
    )
    del cond_ss
    torch.cuda.empty_cache()

    cond_lr = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_512,
        [image],
        coords,
        **camera,
    )
    lr_slat = pipeline.sample_shape_slat(
        cond_lr,
        pipeline.models["shape_slat_flow_model_512"],
        coords,
        shape_sampler,
    )
    del cond_lr
    torch.cuda.empty_cache()

    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    hr_coords = decoder.upsample(lr_slat, upsample_times=4)
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False
    actual_resolution = int(resolution)
    while True:
        grid_resolution = actual_resolution // 16
        quantized = torch.cat((
            hr_coords[:, :1],
            ((hr_coords[:, 1:] + 0.5) / 512
             * (grid_resolution - 1)).round().int(),
        ), dim=1)
        unique_coords = quantized.unique(dim=0)
        if len(unique_coords) < max_num_tokens or actual_resolution == 1024:
            break
        actual_resolution -= 128
    del hr_coords, quantized, lr_slat
    torch.cuda.empty_cache()

    cond_hr = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [image],
        unique_coords,
        grid_resolution_override=actual_resolution // 16,
        **camera,
    )
    flow = pipeline.models["shape_slat_flow_model_1024"]
    noise = sp.SparseTensor(
        feats=torch.randn(
            len(unique_coords), flow.in_channels, device=pipeline.device
        ),
        coords=unique_coords,
    )
    if pipeline.low_vram:
        flow.to(pipeline.device)
    normalized = pipeline.shape_slat_sampler.sample(
        flow,
        noise,
        **cond_hr,
        **{**pipeline.shape_slat_sampler_params, **shape_sampler},
        verbose=True,
        tqdm_desc=f"Sampling local HR shape SLat ({actual_resolution})",
    ).samples
    if pipeline.low_vram:
        flow.cpu()
    std = torch.tensor(
        pipeline.shape_slat_normalization["std"], device=normalized.device
    )[None]
    mean = torch.tensor(
        pipeline.shape_slat_normalization["mean"], device=normalized.device
    )[None]
    shape = normalized.replace(normalized.feats * std + mean)
    return shape, actual_resolution


def _export_mesh(vertices, faces, path, trimesh, *, transform=True):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    if transform:
        mesh.apply_transform(_preview_transform())
    mesh.export(path)
    return mesh


def _simplify_mesh(vertices, faces, *, target_faces, torch):
    if len(faces) <= target_faces:
        return vertices, faces
    try:
        import cumesh

        gpu_mesh = cumesh.CuMesh()
        gpu_mesh.init(
            torch.from_numpy(vertices).float().cuda().contiguous(),
            torch.from_numpy(faces).int().cuda().contiguous(),
        )
        gpu_mesh.simplify(target_faces, verbose=True)
        simplified_vertices, simplified_faces = gpu_mesh.read()
        return (
            simplified_vertices.float().cpu().numpy(),
            simplified_faces.long().cpu().numpy(),
        )
    except (ImportError, RuntimeError) as error:
        print(
            "[WARNING] connected preview simplification unavailable; "
            f"exporting the full mesh: {error}",
            flush=True,
        )
        return vertices, faces


def _registration_scene(
        base_vertices,
        base_faces,
        local_vertices,
        local_faces,
    path,
    trimesh,
    torch,
    max_faces=300_000,
):
    base_vertices, base_faces = _simplify_mesh(
        base_vertices, base_faces, target_faces=max_faces, torch=torch
    )
    local_vertices, local_faces = _simplify_mesh(
        local_vertices, local_faces, target_faces=max_faces, torch=torch
    )
    base = trimesh.Trimesh(
        vertices=base_vertices, faces=base_faces, process=False
    )
    local = trimesh.Trimesh(
        vertices=local_vertices, faces=local_faces, process=False
    )
    base.remove_unreferenced_vertices()
    local.remove_unreferenced_vertices()
    base.apply_transform(_preview_transform())
    local.apply_transform(_preview_transform())
    base.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name="Base", baseColorFactor=[50, 110, 230, 255]
        )
    )
    local.visual = trimesh.visual.TextureVisuals(
        material=trimesh.visual.material.PBRMaterial(
            name="Local1536", baseColorFactor=[245, 70, 35, 255]
        )
    )
    trimesh.Scene({"base": base, "local": local}).export(path)


def _fuse_overlap_mesh(vertices, faces, *, resolution, band, project_back,
                       torch, np):
    """Reconstruct one surface around overlapping meshes in a narrow band."""
    from cumesh.remeshing import remesh_narrow_band_dc

    source_vertices = torch.from_numpy(vertices).float().cuda().contiguous()
    source_faces = torch.from_numpy(faces).int().cuda().contiguous()
    lower = source_vertices.amin(dim=0)
    upper = source_vertices.amax(dim=0)
    center = (lower + upper) * 0.5
    scale = float((upper - lower).max().item() * 1.02)
    fused_vertices, fused_faces = remesh_narrow_band_dc(
        source_vertices,
        source_faces,
        center,
        scale,
        resolution=resolution,
        band=band,
        project_back=project_back,
        verbose=True,
    )
    finite = torch.isfinite(fused_vertices).all(dim=1)
    valid_faces = finite[fused_faces.long()].all(dim=1)
    if not bool(finite.all()):
        fused_faces = fused_faces[valid_faces]
        referenced = torch.unique(fused_faces.long().reshape(-1))
        remap = torch.full(
            (fused_vertices.shape[0],),
            -1,
            dtype=torch.long,
            device=fused_vertices.device,
        )
        remap[referenced] = torch.arange(
            len(referenced), device=fused_vertices.device
        )
        fused_vertices = fused_vertices[referenced]
        fused_faces = remap[fused_faces.long()]
    return (
        fused_vertices.float().cpu().numpy().astype(np.float32, copy=False),
        fused_faces.long().cpu().numpy().astype(np.int64, copy=False),
    )


def run(args) -> None:
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    sys.path.insert(0, str(Path(args.pixal3d_root).resolve()))
    import numpy as np
    from PIL import Image
    import torch
    import trimesh

    torch.set_grad_enabled(False)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    registration_only = bool(
        args.base_mesh
        and args.local_mesh
        and args.inputs_preprocessed
        and Path(args.base_checkpoint).suffix.lower() == ".pt"
    )
    if registration_only:
        upstream = None
        sp = None
        pipeline = None
    else:
        import inference as upstream
        from pixal3d.modules import sparse as sp
        pipeline = upstream.init_pipeline(
            args.model_path, low_vram=args.low_vram
        )
    source = Image.open(args.image)
    source_mask = Image.open(args.mask).convert("L")
    if args.inputs_preprocessed:
        if source.size != source_mask.size:
            raise ValueError("preprocessed image and mask dimensions differ")
        condition, processed_mask = source.convert("RGB"), source_mask
    else:
        condition, processed_mask = _preprocess_refine_pair(
            pipeline, source, source_mask
        )
    condition.save(output_dir / "base-condition.png")
    processed_mask.save(output_dir / "base-mask.png")

    if args.base_mesh:
        if registration_only:
            base_latent = None
            base_resolution, base_camera = _load_legacy_base_context(
                args, torch
            )
        else:
            base_latent, base_resolution, base_camera = _load_base_shape(
                args, pipeline, sp, torch, np
            )
        base_vertices, base_faces = _load_mesh_tensors(
            args.base_mesh, torch, np
        )
    else:
        base_latent, base_resolution, base_camera = _load_base_shape(
            args, pipeline, sp, torch, np
        )
        base_vertices, base_faces = _mesh_arrays(
            _decode_shape(pipeline, base_latent, base_resolution), np
        )

    if args.detail_image:
        detail_source = Image.open(args.detail_image)
    else:
        crop_box = _detail_crop_box(processed_mask)
        detail_rgb = condition.crop(crop_box).convert("RGB")
        detail_alpha = processed_mask.crop(crop_box)
        detail_source = detail_rgb.convert("RGBA")
        detail_source.putalpha(detail_alpha)
    detail_source.save(output_dir / "local-source.png")

    if args.local_mesh:
        local_vertices, local_faces = _load_mesh_tensors(
            args.local_mesh, torch, np
        )
        local_resolution = args.resolution
        local_camera = None
    else:
        detail = pipeline.preprocess_image(detail_source)
        detail.save(output_dir / "local-condition.png")
        local_camera = _estimate_camera(
            detail, upstream, torch, output_dir
        )
        local_latent, local_resolution = _generate_local_shape(
            pipeline,
            detail,
            local_camera,
            resolution=args.resolution,
            steps=args.steps,
            seed=args.seed,
            max_num_tokens=args.max_num_tokens,
            sp=sp,
            torch=torch,
        )
        local_vertices, local_faces = _mesh_arrays(
            _decode_shape(pipeline, local_latent, local_resolution), np
        )
        torch.save({
            "coords": local_latent.coords.detach().cpu(),
            "feats": local_latent.feats.detach().cpu(),
            "resolution": local_resolution,
        }, output_dir / "local-shape-latent.pt")
        torch.save({
            "vertices": torch.from_numpy(local_vertices),
            "faces": torch.from_numpy(local_faces),
        }, output_dir / "local-shape-mesh.pt")

    pixels, _depth, valid = project_pixal_points(
        base_vertices,
        camera_angle_x=base_camera["camera_angle_x"],
        distance=base_camera["distance"],
        mesh_scale=base_camera.get("mesh_scale", 1.0),
        image_size=processed_mask.size,
    )
    weights = sample_mask_bilinear(np.asarray(processed_mask), pixels)
    weights[~valid] = 0.0
    bounds = local_roi_bounds(
        base_vertices,
        weights,
        threshold=args.mask_threshold,
        quantile=args.bounds_quantile,
        padding=args.roi_padding,
    )
    registration = fit_local_detail_transform(
        local_vertices, bounds, quantile=args.bounds_quantile
    )
    registered_local = registration.apply(local_vertices).astype(np.float32)

    _export_mesh(
        local_vertices,
        local_faces,
        output_dir / "local-1536.glb",
        trimesh,
    )
    _registration_scene(
        base_vertices,
        base_faces,
        registered_local,
        local_faces,
        output_dir / "registration.glb",
        trimesh,
        torch,
        max_faces=args.preview_faces,
    )
    keep_base, keep_local = overlap_face_masks(
        base_vertices,
        base_faces,
        registered_local,
        local_faces,
        bounds,
        base_inner_radius=args.base_inner_radius,
        local_outer_radius=args.local_outer_radius,
    )
    overlap_vertices = int(len(base_vertices) + len(registered_local))
    overlap_faces = int(keep_base.sum() + keep_local.sum())
    combined_vertices = None
    combined_faces = None
    if args.export_overlap or args.fuse:
        combined_vertices = np.concatenate((base_vertices, registered_local))
        combined_faces = np.concatenate((
            base_faces[keep_base],
            local_faces[keep_local] + len(base_vertices),
        ))
    if args.export_overlap:
        overlap = _export_mesh(
            combined_vertices,
            combined_faces,
            output_dir / "overlap.glb",
            trimesh,
        )
        overlap_vertices = int(len(overlap.vertices))
        overlap_faces = int(len(overlap.faces))
    fused_vertices = None
    fused_faces = None
    if args.fuse:
        fused_vertices, fused_faces = _fuse_overlap_mesh(
            combined_vertices,
            combined_faces,
            resolution=args.fusion_resolution or base_resolution,
            band=args.fusion_band,
            project_back=args.fusion_project_back,
            torch=torch,
            np=np,
        )
        _export_mesh(
            fused_vertices,
            fused_faces,
            output_dir / "fused.glb",
            trimesh,
        )
        preview_vertices, preview_faces = _simplify_mesh(
            fused_vertices,
            fused_faces,
            target_faces=args.preview_faces,
            torch=torch,
        )
        _export_mesh(
            preview_vertices,
            preview_faces,
            output_dir / "fused-preview.glb",
            trimesh,
        )

    metadata = {
        "protocol": 1,
        "base_resolution": base_resolution,
        "local_requested_resolution": args.resolution,
        "local_actual_resolution": local_resolution,
        "roi_bounds": bounds.tolist(),
        "registration": {
            "scale": registration.scale,
            "translation": list(registration.translation),
        },
        "selected_base_vertices": int(
            (weights > args.mask_threshold).sum()
        ),
        "base_faces_kept": int(keep_base.sum()),
        "local_faces_kept": int(keep_local.sum()),
        "overlap_vertices": overlap_vertices,
        "overlap_faces": overlap_faces,
        "fused_vertices": (
            int(len(fused_vertices)) if fused_vertices is not None else None
        ),
        "fused_faces": (
            int(len(fused_faces)) if fused_faces is not None else None
        ),
        "local_camera": local_camera,
        "artifacts": {
            "local": "local-1536.glb",
            "registration": "registration.glb",
            "overlap": "overlap.glb" if args.export_overlap else None,
            "fused": "fused.glb" if args.fuse else None,
            "fused_preview": "fused-preview.glb" if args.fuse else None,
        },
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pixal3d-root", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-metadata")
    parser.add_argument("--base-mesh")
    parser.add_argument("--local-mesh")
    parser.add_argument("--detail-image")
    parser.add_argument("--inputs-preprocessed", action="store_true")
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resolution", type=int, default=1536)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    parser.add_argument("--mask-threshold", type=float, default=0.05)
    parser.add_argument("--bounds-quantile", type=float, default=0.01)
    parser.add_argument("--roi-padding", type=float, default=0.15)
    parser.add_argument("--base-inner-radius", type=float, default=0.72)
    parser.add_argument("--local-outer-radius", type=float, default=1.08)
    parser.add_argument("--preview-faces", type=int, default=300_000)
    parser.add_argument("--export-overlap", action="store_true")
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--fusion-resolution", type=int)
    parser.add_argument("--fusion-band", type=float, default=1.5)
    parser.add_argument("--fusion-project-back", type=float, default=0.0)
    parser.add_argument("--low-vram", action="store_true")
    args = parser.parse_args()
    if args.resolution not in (1024, 1280, 1536):
        parser.error("--resolution must be 1024, 1280, or 1536")
    if not 1 <= args.steps <= 50:
        parser.error("--steps must be in [1, 50]")
    if args.preview_faces < 1:
        parser.error("--preview-faces must be positive")
    if args.fusion_resolution is not None and args.fusion_resolution < 32:
        parser.error("--fusion-resolution must be at least 32")
    if args.fusion_band <= 0.0:
        parser.error("--fusion-band must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
