#!/usr/bin/env python3
"""Heavy worker for isolated masked Shape-SLat region RePaint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import types


REFINE_PIPELINE_MODEL_NAMES = (
    "shape_slat_flow_model_1024",
    "shape_slat_decoder",
)
ENCODER_MLP_CHUNK_BYTES = 256 * 1024 * 1024
DECODER_ACTIVATION_CHUNK_BYTES = 256 * 1024 * 1024


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


def _gltf_center_to_trellis(center, np):
    x, y, z = map(float, center)
    return np.asarray((x, -z, y), dtype=np.float32)


def _enable_chunked_encoder_mlp(encoder, torch) -> int:
    """Bound peak inference activation memory without changing MLP math."""

    class ChunkedMlp(torch.nn.Module):
        def __init__(self, inner) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, features):
            first = self.inner[0]
            last = self.inner[-1]
            hidden = int(getattr(first, "out_features", features.shape[-1] * 4))
            output_features = int(
                getattr(last, "out_features", features.shape[-1])
            )
            # Linear output and non-inplace SiLU coexist at the peak.
            bytes_per_row = max(1, hidden * features.element_size() * 2)
            rows = max(1, ENCODER_MLP_CHUNK_BYTES // bytes_per_row)
            if len(features) <= rows:
                return self.inner(features)
            output = torch.empty(
                (len(features), output_features),
                dtype=features.dtype,
                device=features.device,
            )
            for start in range(0, len(features), rows):
                stop = min(start + rows, len(features))
                output[start:stop] = self.inner(features[start:stop])
            return output

    blocks = [
        module
        for module in encoder.modules()
        if module.__class__.__name__ == "SparseConvNeXtBlock3d"
    ]
    for block in blocks:
        block.mlp = ChunkedMlp(block.mlp)
    return len(blocks)


def _chunked_norm_silu(features, norm, torch):
    channels = int(features.shape[-1])
    # LayerNorm32 holds float32 input/output before casting back, then SiLU
    # creates another tensor. Bound those temporaries independently of N.
    bytes_per_row = max(1, channels * (4 + 4 + features.element_size()))
    rows = max(1, DECODER_ACTIVATION_CHUNK_BYTES // bytes_per_row)
    if len(features) <= rows:
        return torch.nn.functional.silu(norm(features))
    output = torch.empty_like(features)
    for start in range(0, len(features), rows):
        stop = min(start + rows, len(features))
        output[start:stop] = torch.nn.functional.silu(
            norm(features[start:stop])
        )
    return output


def _enable_chunked_decoder(decoder, torch) -> tuple[int, int]:
    """Chunk decoder row-wise activations while preserving sparse topology."""

    mlp_blocks = _enable_chunked_encoder_mlp(decoder, torch)
    upsample_blocks = [
        module
        for module in decoder.modules()
        if module.__class__.__name__ in {
            "SparseResBlockUpsample3d",
            "SparseResBlockC2S3d",
        }
    ]

    def upsample_forward(block, x, subdiv=None):
        if block.pred_subdiv:
            subdiv = block.to_subdiv(x)
        h = x.replace(_chunked_norm_silu(x.feats, block.norm1, torch))
        h = block.conv1(h)
        subdiv_binarized = (
            subdiv.replace(subdiv.feats > 0) if subdiv is not None else None
        )
        h = block.updown(h, subdiv_binarized)
        x = block.updown(x, subdiv_binarized)
        h = h.replace(_chunked_norm_silu(h.feats, block.norm2, torch))
        h = block.conv2(h)
        h = h + block.skip_connection(x)
        if block.pred_subdiv:
            return h, subdiv
        return h

    for block in upsample_blocks:
        block._forward = types.MethodType(upsample_forward, block)

    def decoder_forward(model, x, guide_subs=None, return_subs=False):
        assert guide_subs is None or model.pred_subdiv is False
        assert return_subs is False or model.pred_subdiv is True
        h = model.from_latent(x)
        h = h.type(model.dtype)
        subs_gt = []
        subs = []
        for i, res in enumerate(model.blocks):
            for j, block in enumerate(res):
                if i < len(model.blocks) - 1 and j == len(res) - 1:
                    if model.pred_subdiv:
                        if model.training:
                            subs_gt.append(h.get_spatial_cache("subdivision"))
                        h, sub = block(h)
                        subs.append(sub)
                    else:
                        h = block(
                            h,
                            subdiv=(
                                guide_subs[i]
                                if guide_subs is not None
                                else None
                            ),
                        )
                else:
                    h = block(h)

        channels = int(h.feats.shape[-1])
        bytes_per_row = max(1, channels * (4 + 4 + h.feats.element_size()))
        rows = max(1, DECODER_ACTIVATION_CHUNK_BYTES // bytes_per_row)
        output = torch.empty(
            (len(h.feats), model.output_layer.out_features),
            dtype=x.dtype,
            device=h.device,
        )
        for start in range(0, len(h.feats), rows):
            stop = min(start + rows, len(h.feats))
            features = h.feats[start:stop].type(x.dtype)
            features = torch.nn.functional.layer_norm(
                features, features.shape[-1:]
            )
            output[start:stop] = torch.nn.functional.linear(
                features,
                model.output_layer.weight,
                model.output_layer.bias,
            )
        h = h.replace(output)
        if model.training and model.pred_subdiv:
            return h, subs_gt, subs
        if return_subs:
            return h, subs
        return h

    decoder_base = next(
        cls
        for cls in decoder.__class__.__mro__
        if cls.__name__ == "SparseUnetVaeDecoder"
    )
    # Preserve FlexiDualGridVaeDecoder.forward: it converts the base decoder's
    # SparseTensor output into the final mesh. Patch only its base implementation.
    decoder_base.forward = decoder_forward
    return mlp_blocks, len(upsample_blocks)


def _to_gltf_y_up(mesh, np):
    result = mesh.copy()
    result.apply_transform(np.asarray((
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    ), dtype=np.float64))
    return result


def _mesh_to_shape_slat(mesh, encoder, resolution, sparse, torch, o_voxel):
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
    vertices_sparse = sparse.SparseTensor(local_vertices.float(), coords)
    intersections_sparse = vertices_sparse.replace(intersected.bool())
    latent = encoder(
        vertices_sparse.cuda(),
        intersections_sparse.cuda(),
        sample_posterior=False,
    )
    return latent, int(len(voxel_indices))


def _token_mask(source_slat, local_vertices, vertex_weights, resolution, torch):
    grid = resolution // 16
    vertices = torch.from_numpy(local_vertices).float().cuda()
    weights = torch.from_numpy(vertex_weights).float().cuda().clamp(0, 1)
    indexes = torch.floor((vertices + 0.5) * grid).long().clamp(0, grid - 1)
    linear = indexes[:, 0] * grid * grid + indexes[:, 1] * grid + indexes[:, 2]
    dense = torch.zeros(grid * grid * grid, device=weights.device)
    dense.scatter_reduce_(0, linear, weights, reduce="amax", include_self=True)
    dense = dense.reshape(1, 1, grid, grid, grid)
    dense = torch.nn.functional.max_pool3d(dense, 3, stride=1, padding=1)
    coords = source_slat.coords[:, 1:].long().clamp(0, grid - 1)
    return dense[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]][:, None]


def _repaint(
    pipeline,
    source_slat,
    conditions,
    schedule,
    mask,
    *,
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
    times = np.linspace(strength, 0, len(schedule) + 1)
    times = 3.0 * times / (1 + 2.0 * times)
    start = float(times[0])
    start_sigma = sigma_min + (1.0 - sigma_min) * start
    sample = source.replace(
        (1.0 - start) * source.feats + start_sigma * source_epsilon
    )
    if pipeline.low_vram:
        model.to(pipeline.device)
    try:
        for index, ((current, previous), view_id) in enumerate(
            zip(zip(times[:-1], times[1:]), schedule), 1
        ):
            condition = conditions[view_id]
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
            _emit(
                f"[region-refine] {index}/{len(schedule)} · {view_id} · "
                f"t={current:.4f}->{previous:.4f}"
            )
    finally:
        if pipeline.low_vram:
            model.cpu()
    sample = sample.replace(
        mask * sample.feats + (1.0 - mask) * source.feats
    )
    return sample * std + mean


def _decode_mesh_cache(pipeline, slat, resolution, path, np, torch):
    """Decode once and move the large decoder result out of CUDA memory."""

    if path.is_file():
        _emit(f"[resume] decoded mesh cache found: {path.name}")
        with np.load(path, allow_pickle=False) as saved:
            return (
                np.ascontiguousarray(saved["vertices"], dtype=np.float32),
                np.ascontiguousarray(saved["faces"], dtype=np.int32),
            )

    meshes, subdivisions = pipeline.decode_shape_slat(slat, resolution)
    decoded = meshes[0]
    vertices = np.ascontiguousarray(
        decoded.vertices.detach().float().cpu().numpy(), dtype=np.float32
    )
    faces = np.ascontiguousarray(
        decoded.faces.detach().int().cpu().numpy(), dtype=np.int32
    )
    # This cache is intentionally written before CuMesh. If postprocessing runs
    # out of memory, retrying must not repeat either sampling or shape decoding.
    np.savez(path, vertices=vertices, faces=faces)
    del decoded, meshes, subdivisions
    torch.cuda.empty_cache()
    _emit(
        f"[decode-cache] {len(vertices)} vertices, {len(faces)} faces: "
        f"{path.name}"
    )
    return vertices, faces


def _simplify_decoded_mesh(
    vertices, faces, face_target, cumesh, np, torch, trimesh
):
    if len(faces) > face_target:
        _emit(f"[simplify] CuMesh {len(faces)} -> {face_target} faces")
        vertices_cuda = torch.from_numpy(vertices).float().cuda()
        faces_cuda = torch.from_numpy(faces).int().cuda()
        mesh = cumesh.CuMesh()
        mesh.init(vertices_cuda, faces_cuda)
        mesh.simplify(face_target)
        simplified_vertices, simplified_faces = mesh.read()
        vertices = np.ascontiguousarray(
            simplified_vertices.detach().float().cpu().numpy(),
            dtype=np.float32,
        )
        faces = np.ascontiguousarray(
            simplified_faces.detach().int().cpu().numpy(),
            dtype=np.int32,
        )
        del (
            mesh,
            vertices_cuda,
            faces_cuda,
            simplified_vertices,
            simplified_faces,
        )
        torch.cuda.empty_cache()
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: trellis_refine_runner.py REQUEST.json")
    request_path = Path(sys.argv[1]).expanduser().resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol") != 1:
        raise ValueError("unsupported region refine request protocol")
    output = Path(request["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_path = Path(request["source_region"]).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    schedule = tuple(map(str, request["schedule"]))
    if len(schedule) != int(request["steps"]) or not schedule:
        raise ValueError("region refine schedule length does not match steps")

    trellis_root = Path(request["trellis_root"]).expanduser().resolve()
    model_path = Path(request["model_path"]).expanduser().resolve()
    sys.path.insert(0, str(trellis_root))
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import cumesh
    import numpy as np
    import o_voxel
    from PIL import Image
    import torch
    import trimesh
    import trellis2.models as models
    from trellis2.modules import sparse
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    torch.set_grad_enabled(False)
    with np.load(source_path, allow_pickle=False) as saved:
        project_vertices = np.ascontiguousarray(saved["vertices"], dtype=np.float32)
        faces = np.ascontiguousarray(saved["faces"], dtype=np.int64)
        vertex_weights = np.ascontiguousarray(saved["weights"], dtype=np.float32)
    cube = request["cube"]
    cube_center = _gltf_center_to_trellis(cube["center"], np)
    cube_side = float(cube["side"])
    bounds_min = project_vertices.min(axis=0)
    bounds_max = project_vertices.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    scale = float((bounds_max - bounds_min).max() / 0.96)
    if not scale > 0.0:
        raise ValueError("standalone refine region has zero spatial extent")
    local_vertices = (project_vertices - center) / scale
    source_project = trimesh.Trimesh(
        vertices=project_vertices, faces=faces, process=False
    )
    source_local = trimesh.Trimesh(
        vertices=local_vertices, faces=faces, process=False
    )
    source_region_path = output / "source-region.glb"
    source_local_path = output / "source-region-local.glb"
    _to_gltf_y_up(source_project, np).export(source_region_path)
    _to_gltf_y_up(source_local, np).export(source_local_path)

    patches = {str(item["id"]): item for item in request["patches"]}
    missing = sorted(set(schedule) - patches.keys())
    if missing:
        raise ValueError(f"schedule refers to missing patches: {missing}")

    source_slat_path = output / "source-shape-slat.npz"
    refined_slat_path = output / "refined-shape-slat.npz"
    token_mask_path = output / "shape-token-mask.npz"
    resume_decode = all(
        path.is_file()
        for path in (source_slat_path, refined_slat_path, token_mask_path)
    )
    model_names = (
        ("shape_slat_decoder",)
        if resume_decode
        else REFINE_PIPELINE_MODEL_NAMES
    )
    _emit(
        "[load] minimal TRELLIS.2 shape-refine pipeline: "
        + ", ".join(model_names)
    )
    Trellis2ImageTo3DPipeline.model_names_to_load = list(
        model_names
    )
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(model_path))
    pipeline.low_vram = True
    pipeline.cuda()
    resolution = int(request["resolution"])
    decoder = pipeline.models["shape_slat_decoder"]
    decoder_mlps, decoder_upsamples = _enable_chunked_decoder(decoder, torch)
    _emit(
        f"[low-vram] chunked decoder: {decoder_mlps} MLP, "
        f"{decoder_upsamples} upsample blocks; "
        f"activation budget {ENCODER_MLP_CHUNK_BYTES // (1024 * 1024)} MiB"
    )
    if resume_decode:
        _emit("[resume] refined Shape-SLat cache found; skipping encode/sampling")
        source_slat = _load_sparse(source_slat_path, sparse, torch, np)
        refined_slat = _load_sparse(refined_slat_path, sparse, torch, np)
        with np.load(token_mask_path, allow_pickle=False) as saved:
            mask = torch.from_numpy(saved["weights"].copy()).float().cuda()[:, None]
        mask = mask.to(source_slat.device, source_slat.feats.dtype)
        editable_tokens = int((mask[:, 0] > 1e-4).sum().item())
        surface_voxels = None
    else:
        encoder_path = model_path / "ckpts" / "shape_enc_next_dc_f16c32_fp16"
        encoder = models.from_pretrained(str(encoder_path)).eval()
        chunked_blocks = _enable_chunked_encoder_mlp(encoder, torch)
        _emit(
            f"[low-vram] chunked {chunked_blocks} Shape VAE encoder MLP "
            f"blocks; activation budget "
            f"{ENCODER_MLP_CHUNK_BYTES // (1024 * 1024)} MiB"
        )
        encoder = encoder.cuda()
        _emit(f"[shape-encode] standalone region at resolution {resolution}")
        source_slat, surface_voxels = _mesh_to_shape_slat(
            source_local, encoder, resolution, sparse, torch, o_voxel
        )
        encoder.cpu()
        del encoder
        torch.cuda.empty_cache()
        _save_sparse(source_slat_path, source_slat)

        mask = _token_mask(
            source_slat,
            local_vertices,
            vertex_weights,
            resolution,
            torch,
        ).to(source_slat.device, source_slat.feats.dtype)
        editable_tokens = int((mask[:, 0] > 1e-4).sum().item())
        if not editable_tokens:
            raise ValueError(
                "vertex mask does not reach any encoded Shape-SLat token"
            )
        np.savez_compressed(
            token_mask_path,
            coords=source_slat.coords.detach().cpu().numpy(),
            weights=mask[:, 0].detach().float().cpu().numpy(),
        )
        _emit(
            f"[mask] {editable_tokens}/{len(source_slat.coords)} "
            "editable Shape-SLat tokens"
        )

        prepared = {}
        for view_id in dict.fromkeys(schedule):
            item = patches[view_id]
            image_path = Path(item["image"]).expanduser().resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            with Image.open(image_path) as image:
                x0, y0, x1, y1 = map(float, item["bounds"])
                width, height = image.size
                crop = image.crop((
                    round(x0 * width),
                    round(y0 * height),
                    round(x1 * width),
                    round(y1 * height),
                ))
            crop.save(output / f"patch-{view_id}.png")
            prepared[view_id] = pipeline.preprocess_image(crop)
            prepared[view_id].save(output / f"prepared-{view_id}.png")
        _emit(f"[condition-1024] {len(prepared)} manual view patch(es)")
        conditions = {
            view_id: pipeline.get_cond([image], 1024)
            for view_id, image in prepared.items()
        }
        refined_slat = _repaint(
            pipeline,
            source_slat,
            conditions,
            schedule,
            mask,
            strength=float(request["strength"]),
            cfg=float(request["cfg"]),
            seed=int(request["seed"]),
        )
        _save_sparse(refined_slat_path, refined_slat)
        del conditions
        torch.cuda.empty_cache()

    delta = (refined_slat.feats - source_slat.feats).norm(dim=1)
    frozen = mask[:, 0] <= 0.0
    feature_delta_mean = float(delta.mean().item())
    feature_delta_frozen_max = (
        float(delta[frozen].max().item()) if frozen.any() else 0.0
    )
    fully_frozen_tokens = int(frozen.sum().item())
    shape_tokens = int(len(source_slat.coords))
    del delta, frozen

    decoded_cache_path = output / "decoded-region-raw.npz"
    _emit(f"[decode] standalone refined region at resolution {resolution}")
    decoded_vertices, decoded_faces = _decode_mesh_cache(
        pipeline,
        refined_slat,
        resolution,
        decoded_cache_path,
        np,
        torch,
    )
    # The decoder's sparse subdivision tensors are much larger than the final
    # mesh. Release the entire neural stage before CuMesh allocates its graph.
    del pipeline, decoder, source_slat, refined_slat, mask
    torch.cuda.empty_cache()
    _emit("[low-vram] TRELLIS tensors released before CuMesh")
    refined_local = _simplify_decoded_mesh(
        decoded_vertices,
        decoded_faces,
        int(request["preview_face_target"]),
        cumesh,
        np,
        torch,
        trimesh,
    )
    refined_project = refined_local.copy()
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= scale
    transform[:3, 3] = center
    refined_project.apply_transform(transform)
    refined_local_path = output / "refined-region-local.glb"
    refined_region_path = output / "refined-region.glb"
    _to_gltf_y_up(refined_local, np).export(refined_local_path)
    _to_gltf_y_up(refined_project, np).export(refined_region_path)

    result = {
        "protocol": 1,
        "status": "complete",
        "request_key": str(request["request_key"]),
        "geometry_fingerprint": str(request["geometry_fingerprint"]),
        "shape": str(refined_region_path.resolve()),
        "source_region": str(source_region_path.resolve()),
        "source_region_local": str(source_local_path.resolve()),
        "refined_region_local": str(refined_local_path.resolve()),
        "source_slat": str(source_slat_path.resolve()),
        "refined_slat": str(refined_slat_path.resolve()),
        "token_mask": str((output / "shape-token-mask.npz").resolve()),
        "resolution": resolution,
        "surface_voxels": surface_voxels,
        "shape_tokens": shape_tokens,
        "editable_tokens": editable_tokens,
        "fully_frozen_tokens": fully_frozen_tokens,
        "feature_delta_mean": feature_delta_mean,
        "feature_delta_frozen_max": feature_delta_frozen_max,
        "cube_center_gltf": list(map(float, cube["center"])),
        "cube_center_trellis": cube_center.tolist(),
        "cube_side": cube_side,
        "normalization_center_trellis": center.tolist(),
        "normalization_scale": scale,
        "patches": list(dict.fromkeys(schedule)),
        "schedule": list(schedule),
        "assembly": "not-run",
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _emit(f"[complete] standalone region: {refined_region_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
