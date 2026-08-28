#!/usr/bin/env python3
"""Heavy TRELLIS.2 worker that textures the current postprocessed GLB."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def _emit(message: str) -> None:
    print(message, flush=True)


def _save_sparse(path: Path, value) -> None:
    import numpy as np

    np.savez_compressed(
        path,
        coords=value.coords.detach().cpu().numpy(),
        feats=value.feats.detach().float().cpu().numpy(),
    )


def _load_sparse(path: Path, sparse, torch, np):
    with np.load(path, allow_pickle=False) as saved:
        coords = torch.from_numpy(saved["coords"].copy()).int()
        feats = torch.from_numpy(saved["feats"].copy()).float()
    return sparse.SparseTensor(feats, coords).cuda()


def _shape_guide_subs(shape_slat, levels: int, sparse, torch):
    """Recover decoder subdivision guides from the shape encoder hierarchy."""
    guides = []
    coords = shape_slat.coords
    scale = tuple(shape_slat._scale)
    cache_root = shape_slat._spatial_cache
    for _index in range(levels):
        cache = cache_root.get(str(scale), {}).get("channel2spatial_2")
        if cache is None:
            raise RuntimeError(
                f"shape SLat is missing channel2spatial_2 cache at scale {scale}"
            )
        next_coords, parent_indexes, child_indexes = cache
        children = torch.zeros(
            (len(coords), 8),
            dtype=torch.bool,
            device=coords.device,
        )
        children[parent_indexes, child_indexes] = True
        guides.append(sparse.SparseTensor(children, coords))
        coords = next_coords
        scale = tuple(value / 2 for value in scale)
    return tuple(guides)


def _save_guide_subs(path: Path, guides) -> None:
    import numpy as np

    payload = {"count": np.asarray([len(guides)], dtype=np.int32)}
    for index, guide in enumerate(guides):
        payload[f"coords_{index}"] = guide.coords.detach().cpu().numpy()
        payload[f"feats_{index}"] = guide.feats.detach().cpu().numpy()
    np.savez_compressed(path, **payload)


def _load_guide_subs(path: Path, sparse, torch, np):
    guides = []
    with np.load(path, allow_pickle=False) as saved:
        count = int(saved["count"][0])
        for index in range(count):
            coords = torch.from_numpy(saved[f"coords_{index}"].copy()).int()
            feats = torch.from_numpy(saved[f"feats_{index}"].copy()).bool()
            guides.append(sparse.SparseTensor(feats, coords).cuda())
    return tuple(guides)


def _texture_model_names(
    resolution: int,
    *,
    shape_cached: bool,
    guides_cached: bool,
    texture_cached: bool,
) -> tuple[str, ...]:
    names = ["tex_slat_decoder"]
    if not shape_cached or not guides_cached:
        names.append("shape_slat_encoder")
    if not texture_cached:
        names.append(f"tex_slat_flow_model_{resolution}")
    return tuple(names)


def _sample_texture(pipeline, conditions, schedule, shape_slat, *, seed: int):
    import numpy as np
    import torch

    resolution = int(pipeline.image_cond_model.image_size)
    model = pipeline.models[f"tex_slat_flow_model_{resolution}"]
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
    times = np.linspace(1, 0, len(schedule) + 1)
    times = 3.0 * times / (1 + 2.0 * times)
    if pipeline.low_vram:
        model.to(pipeline.device)
    try:
        for index, ((current, previous), view_id) in enumerate(
            zip(zip(times[:-1], times[1:]), schedule),
            1,
        ):
            condition = conditions[view_id]
            sample = sampler.sample_once(
                model,
                sample,
                float(current),
                float(previous),
                cond=condition["cond"],
                neg_cond=condition["neg_cond"],
                guidance_strength=1.0,
                guidance_rescale=0.0,
                guidance_interval=(0.6, 0.9),
                concat_cond=normalized_shape,
            ).pred_x_prev
            _emit(f"[texture] {index}/{len(schedule)} · {view_id}")
    finally:
        if pipeline.low_vram:
            model.cpu()

    texture_std = torch.tensor(
        pipeline.tex_slat_normalization["std"], device=sample.device
    )[None]
    texture_mean = torch.tensor(
        pipeline.tex_slat_normalization["mean"], device=sample.device
    )[None]
    return sample * texture_std + texture_mean


def _load_mesh(path: Path):
    import numpy as np
    import trimesh

    loaded = trimesh.load(path, force="scene", process=False)
    mesh = loaded.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or len(mesh.faces) == 0:
        raise ValueError(f"input GLB contains no triangle mesh: {path}")
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)
    edge_a = vertices[faces[:, 1]] - vertices[faces[:, 0]]
    edge_b = vertices[faces[:, 2]] - vertices[faces[:, 0]]
    exact_degenerate = np.all(np.cross(edge_a, edge_b) == 0, axis=1)
    removed = int(np.count_nonzero(exact_degenerate))
    if removed:
        mesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces[~exact_degenerate],
            process=False,
        )
        mesh.remove_unreferenced_vertices()
    return mesh, removed


def _bake_pbr_preserving_faces(
    mesh,
    pbr_voxel,
    *,
    resolution: int,
    texture_size: int,
    device,
):
    """UV-unwrap and bake without CuMesh's implicit topology cleanup."""
    import cv2
    import flex_gemm
    from PIL import Image
    import numpy as np
    import nvdiffrast.torch as dr
    import torch
    import trimesh
    from cumesh import Atlas

    vertices = np.ascontiguousarray(mesh.vertices, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.faces, dtype=np.int32)
    normals = np.ascontiguousarray(mesh.vertex_normals, dtype=np.float32)
    atlas = Atlas()
    atlas.add_mesh(
        torch.from_numpy(vertices),
        torch.from_numpy(faces),
        torch.from_numpy(normals),
    )
    atlas.compute_charts(fix_winding=False, verbose=False)
    atlas.pack_charts(
        resolution=texture_size,
        padding=2,
        bilinear=True,
        verbose=False,
    )
    vertex_map, out_faces, uvs = atlas.get_mesh(0)
    if len(out_faces) != len(faces):
        raise RuntimeError(
            "xatlas changed triangle count after exact-degenerate filtering: "
            f"{len(faces)} -> {len(out_faces)}"
        )
    out_vertices = torch.from_numpy(vertices)[vertex_map].cuda().contiguous()
    out_normals = torch.from_numpy(normals)[vertex_map].cuda().contiguous()
    out_faces = out_faces.cuda().contiguous()
    uvs = uvs.cuda().contiguous()

    ctx = dr.RasterizeCudaContext()
    uv_clip = torch.cat(
        (
            uvs * 2 - 1,
            torch.zeros_like(uvs[:, :1]),
            torch.ones_like(uvs[:, :1]),
        ),
        dim=-1,
    ).unsqueeze(0)
    rast, _ = dr.rasterize(
        ctx,
        uv_clip,
        out_faces,
        resolution=[texture_size, texture_size],
    )
    mask = rast[0, ..., 3] > 0
    positions = dr.interpolate(
        out_vertices.unsqueeze(0), rast, out_faces
    )[0][0]
    attrs = torch.zeros(
        texture_size,
        texture_size,
        pbr_voxel.shape[1],
        device=device,
    )
    attrs[mask] = flex_gemm.ops.grid_sample.grid_sample_3d(
        pbr_voxel.feats,
        pbr_voxel.coords,
        shape=torch.Size([*pbr_voxel.shape, *pbr_voxel.spatial_shape]),
        grid=((positions[mask] + 0.5) * resolution).reshape(1, -1, 3),
        mode="trilinear",
    )

    layout = {
        "base_color": slice(0, 3),
        "metallic": slice(3, 4),
        "roughness": slice(4, 5),
        "alpha": slice(5, 6),
    }
    attrs_np = attrs.detach().cpu().numpy()
    base_color = np.clip(
        attrs_np[..., layout["base_color"]] * 255, 0, 255
    ).astype(np.uint8)
    metallic = np.clip(
        attrs_np[..., layout["metallic"]] * 255, 0, 255
    ).astype(np.uint8)
    roughness = np.clip(
        attrs_np[..., layout["roughness"]] * 255, 0, 255
    ).astype(np.uint8)
    alpha = np.clip(attrs_np[..., layout["alpha"]] * 255, 0, 255).astype(
        np.uint8
    )
    holes = (~mask.detach().cpu().numpy()).astype(np.uint8)
    base_color = cv2.inpaint(base_color, holes, 3, cv2.INPAINT_TELEA)
    metallic = cv2.inpaint(metallic, holes, 1, cv2.INPAINT_TELEA)[..., None]
    roughness = cv2.inpaint(roughness, holes, 1, cv2.INPAINT_TELEA)[..., None]
    alpha = cv2.inpaint(alpha, holes, 1, cv2.INPAINT_TELEA)[..., None]

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(
            np.concatenate((base_color, alpha), axis=-1)
        ),
        baseColorFactor=np.asarray([255, 255, 255, 255], dtype=np.uint8),
        metallicRoughnessTexture=Image.fromarray(
            np.concatenate(
                (np.zeros_like(metallic), roughness, metallic), axis=-1
            )
        ),
        metallicFactor=1.0,
        roughnessFactor=1.0,
        alphaMode="OPAQUE",
        doubleSided=True,
    )

    out_vertices_np = out_vertices.detach().cpu().numpy()
    out_normals_np = out_normals.detach().cpu().numpy()
    out_faces_np = out_faces.detach().cpu().numpy()
    uvs_np = uvs.detach().cpu().numpy()
    out_vertices_np[:, 1], out_vertices_np[:, 2] = (
        out_vertices_np[:, 2].copy(),
        -out_vertices_np[:, 1].copy(),
    )
    out_normals_np[:, 1], out_normals_np[:, 2] = (
        out_normals_np[:, 2].copy(),
        -out_normals_np[:, 1].copy(),
    )
    uvs_np[:, 1] = 1 - uvs_np[:, 1]
    return trimesh.Trimesh(
        vertices=out_vertices_np,
        faces=out_faces_np,
        vertex_normals=out_normals_np,
        process=False,
        visual=trimesh.visual.TextureVisuals(uv=uvs_np, material=material),
    )


def _write_texture_result(
    request,
    output: Path,
    input_mesh_path: Path,
    source_mesh,
    removed_degenerate_faces: int,
    textured,
    shape_path: Path,
    np,
) -> None:
    source_bounds = np.asarray(source_mesh.bounds, dtype=np.float64)
    output_bounds = np.asarray(textured.bounds, dtype=np.float64)
    result = {
        "protocol": 1,
        "status": "complete",
        "texture_key": str(request["texture_key"]),
        "input_mesh": str(input_mesh_path),
        "shape": str(shape_path.resolve()),
        "shape_slat": str((output / "encoded-shape-slat.npz").resolve()),
        "texture_slat": str((output / "texture-slat.npz").resolve()),
        "source_vertices": int(len(source_mesh.vertices)),
        "source_faces": int(len(source_mesh.faces)),
        "removed_exact_degenerate_faces": removed_degenerate_faces,
        "output_vertices": int(len(textured.vertices)),
        "output_faces": int(len(textured.faces)),
        "source_bounds": source_bounds.tolist(),
        "output_bounds": output_bounds.tolist(),
        "resolution": int(request["resolution"]),
        "texture_size": int(request["texture_size"]),
        "views": list(dict.fromkeys(map(str, request["schedule"]))),
        "schedule": list(map(str, request["schedule"])),
        "material": {
            "base_color": True,
            "metallic_roughness": True,
            "normal": False,
        },
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: trellis_texture_runner.py REQUEST.json")
    request_path = Path(sys.argv[1]).expanduser().resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol") != 1:
        raise ValueError("unsupported texture request protocol")

    output = Path(request["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    input_mesh_path = Path(request["input_mesh"]).expanduser().resolve()
    schedule = tuple(str(value) for value in request["schedule"])
    if len(schedule) != int(request["steps"]) or not schedule:
        raise ValueError("texture schedule length does not match steps")
    view_specs = {
        str(item["id"]): item
        for item in request["views"]
    }
    missing = sorted(set(schedule) - view_specs.keys())
    if missing:
        raise ValueError(f"texture schedule refers to missing views: {missing}")

    trellis_root = Path(request["trellis_root"]).expanduser().resolve()
    model_path = Path(request["model_path"]).expanduser().resolve()
    sys.path.insert(0, str(trellis_root))
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import numpy as np
    import trimesh
    _emit(f"[mesh] loading final geometry: {input_mesh_path.name}")
    source_mesh, removed_degenerate_faces = _load_mesh(input_mesh_path)
    if removed_degenerate_faces:
        _emit(
            "[mesh] removed "
            f"{removed_degenerate_faces} exact zero-area input triangles"
        )
    shape_path = output / "textured.glb"
    shape_slat_path = output / "encoded-shape-slat.npz"
    shape_guides_path = output / "shape-guide-subs.npz"
    texture_slat_path = output / "texture-slat.npz"
    if all(
        path.is_file()
        for path in (shape_path, shape_slat_path, texture_slat_path)
    ):
        _emit("[resume] baked PBR GLB and SLat caches found; writing result")
        textured, _removed_output_faces = _load_mesh(shape_path)
        _write_texture_result(
            request,
            output,
            input_mesh_path,
            source_mesh,
            removed_degenerate_faces,
            textured,
            shape_path,
            np,
        )
        _emit(f"[complete] {shape_path}")
        return 0

    import torch
    from PIL import Image
    from trellis2.modules import sparse
    from trellis2.pipelines import Trellis2TexturingPipeline
    from trellis_refine_runner import (
        DECODER_ACTIVATION_CHUNK_BYTES,
        _enable_chunked_decoder,
        _enable_chunked_encoder_mlp,
    )

    torch.set_grad_enabled(False)
    resolution = int(request["resolution"])
    shape_cached = shape_slat_path.is_file()
    guides_cached = shape_guides_path.is_file()
    texture_cached = texture_slat_path.is_file()
    model_names = _texture_model_names(
        resolution,
        shape_cached=shape_cached,
        guides_cached=guides_cached,
        texture_cached=texture_cached,
    )
    _emit(
        "[load] minimal TRELLIS.2 texturing pipeline: "
        + ", ".join(model_names)
    )
    Trellis2TexturingPipeline.model_names_to_load = list(model_names)
    pipeline = Trellis2TexturingPipeline.from_pretrained(
        str(model_path),
        config_file="texturing_pipeline.json",
    )
    pipeline.low_vram = True
    pipeline.cuda()

    source_bounds = np.asarray(source_mesh.bounds, dtype=np.float64)
    center = source_bounds.mean(axis=0)
    extent = float(np.max(source_bounds[1] - source_bounds[0]))
    if not np.isfinite(extent) or extent <= 0:
        raise ValueError("input mesh has degenerate bounds")
    normalization_scale = 0.99999 / extent
    prepared_mesh = pipeline.preprocess_mesh(source_mesh)
    prepared_mesh.export(output / "geometry-normalized.glb")

    decoder = pipeline.models["tex_slat_decoder"]
    guide_levels = len(decoder.blocks) - 1
    if shape_cached and guides_cached:
        _emit("[resume] encoded Shape-SLat and subdivision guides found")
        shape_slat = _load_sparse(shape_slat_path, sparse, torch, np)
        guide_subs = _load_guide_subs(shape_guides_path, sparse, torch, np)
    else:
        if shape_cached:
            _emit(
                "[resume] legacy Shape-SLat cache has no subdivision guides; "
                "rebuilding shape hierarchy once"
            )
        encoder = pipeline.models["shape_slat_encoder"]
        chunked_blocks = _enable_chunked_encoder_mlp(encoder, torch)
        _emit(
            f"[low-vram] chunked {chunked_blocks} Shape VAE encoder MLP "
            "blocks"
        )
        _emit(f"[shape-encode] current mesh at resolution {resolution}")
        shape_slat = pipeline.encode_shape_slat(prepared_mesh, resolution)
        guide_subs = _shape_guide_subs(
            shape_slat, guide_levels, sparse, torch
        )
        _save_sparse(shape_slat_path, shape_slat)
        _save_guide_subs(shape_guides_path, guide_subs)
        torch.cuda.empty_cache()

    if texture_cached:
        _emit("[resume] texture SLat cache found; skipping conditioning/sampling")
        texture_slat = _load_sparse(texture_slat_path, sparse, torch, np)
    else:
        prepared_images = {}
        for view_id in dict.fromkeys(schedule):
            spec = view_specs[view_id]
            image_path = Path(spec["image"]).expanduser().resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as opened:
                source_image = opened.convert("RGB")
                crop_bounds = spec.get("bounds")
                if crop_bounds is not None:
                    x0, y0, x1, y1 = map(float, crop_bounds)
                    width, height = source_image.size
                    source_image = source_image.crop((
                        round(x0 * width),
                        round(y0 * height),
                        round(x1 * width),
                        round(y1 * height),
                    ))
                    source_image.save(output / f"patch-{view_id}.png")
                image = pipeline.preprocess_image(source_image)
            image.save(output / f"prepared-{view_id}.png")
            prepared_images[view_id] = image

        _emit(f"[condition-{resolution}] {len(prepared_images)} view(s)")
        conditions = {
            view_id: pipeline.get_cond([image], resolution)
            for view_id, image in prepared_images.items()
        }
        texture_slat = _sample_texture(
            pipeline,
            conditions,
            schedule,
            shape_slat,
            seed=int(request["seed"]),
        )
        _save_sparse(texture_slat_path, texture_slat)
        del conditions, prepared_images

    del shape_slat
    torch.cuda.empty_cache()

    _emit("[decode] sparse PBR attribute volume")
    decoder_mlps, decoder_upsamples = _enable_chunked_decoder(decoder, torch)
    _emit(
        f"[low-vram] chunked texture decoder: {decoder_mlps} MLP, "
        f"{decoder_upsamples} upsample blocks; activation budget "
        f"{DECODER_ACTIVATION_CHUNK_BYTES // (1024 * 1024)} MiB"
    )
    decoder.to(pipeline.device)
    try:
        pbr_voxel = decoder(texture_slat, guide_subs=guide_subs)
        pbr_voxel.feats.mul_(0.5).add_(0.5)
    finally:
        decoder.cpu()
    del texture_slat, guide_subs
    torch.cuda.empty_cache()
    _emit(f"[bake] {int(request['texture_size'])}px PBR atlas")
    textured = _bake_pbr_preserving_faces(
        prepared_mesh,
        pbr_voxel,
        resolution=resolution,
        texture_size=int(request["texture_size"]),
        device=pipeline.device,
    )

    # The upstream texturing pipeline normalizes every input mesh. Restore the
    # original GLB bounds after its own TRELLIS-Z-up -> glTF-Y-up conversion.
    restore = np.eye(4, dtype=np.float64)
    restore[:3, :3] /= normalization_scale
    restore[:3, 3] = center
    textured.apply_transform(restore)
    textured.export(shape_path, extension_webp=True)
    _write_texture_result(
        request,
        output,
        input_mesh_path,
        source_mesh,
        removed_degenerate_faces,
        textured,
        shape_path,
        np,
    )
    _emit(f"[complete] {shape_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
