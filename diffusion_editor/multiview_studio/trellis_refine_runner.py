#!/usr/bin/env python3
"""Main TRELLIS.2 shape cascade started from protected region latents."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import types


REFINE_PIPELINE_MODEL_NAMES = (
    "sparse_structure_flow_model",
    "sparse_structure_decoder",
    "shape_slat_flow_model_512",
    "shape_slat_flow_model_1024",
    "shape_slat_decoder",
)
SPARSE_STRUCTURE_ENCODER = (
    "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"
)
ENCODER_MLP_CHUNK_BYTES = 256 * 1024 * 1024
DECODER_ACTIVATION_CHUNK_BYTES = 256 * 1024 * 1024


def _region_stage_strengths(structure_strength: float) -> dict[str, float]:
    """Map UI strength to coarse structure and fully rebuild editable detail."""
    structure_strength = float(structure_strength)
    if not 0.0 <= structure_strength <= 1.0:
        raise ValueError("region structure strength must be from zero to one")
    return {
        "occupancy": structure_strength,
        "shape_512": 1.0,
        "shape_high_resolution": 1.0,
    }


def _emit(message: str) -> None:
    print(message, flush=True)


def _save_sparse(path: Path, value) -> None:
    import numpy as np

    np.savez_compressed(
        path,
        coords=np.ascontiguousarray(value.coords.detach().cpu().numpy()),
        feats=np.ascontiguousarray(
            value.feats.detach().float().cpu().numpy()
        ),
    )


def _load_sparse(path: Path, sparse, torch, np):
    with np.load(path, allow_pickle=False) as saved:
        coords = torch.from_numpy(
            np.ascontiguousarray(saved["coords"])
        ).int().contiguous()
        feats = torch.from_numpy(
            np.ascontiguousarray(saved["feats"])
        ).float().contiguous()
    return sparse.SparseTensor(feats, coords).cuda()


def _save_dense(path: Path, value) -> None:
    import numpy as np

    np.savez_compressed(
        path,
        value=np.ascontiguousarray(value.detach().float().cpu().numpy()),
    )


def _load_dense(path: Path, torch, np):
    with np.load(path, allow_pickle=False) as saved:
        value = np.ascontiguousarray(saved["value"], dtype=np.float32)
    return torch.from_numpy(value).cuda()


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


def _dense_vertex_mask(local_vertices, vertex_weights, grid, torch):
    vertices = torch.from_numpy(local_vertices).float().cuda()
    weights = torch.from_numpy(vertex_weights).float().cuda().clamp(0, 1)
    indexes = torch.floor((vertices + 0.5) * grid).long().clamp(0, grid - 1)
    linear = indexes[:, 0] * grid * grid + indexes[:, 1] * grid + indexes[:, 2]
    dense = torch.zeros(grid * grid * grid, device=weights.device)
    dense.scatter_reduce_(0, linear, weights, reduce="amax", include_self=True)
    dense = dense.reshape(1, 1, grid, grid, grid)
    return torch.nn.functional.max_pool3d(dense, 3, stride=1, padding=1)


def _encode_sparse_structure(mesh, encoder, torch, o_voxel):
    """Encode the same 64^3 surface occupancy consumed by Main generation."""
    vertices = torch.from_numpy(mesh.vertices.copy()).float()
    faces = torch.from_numpy(mesh.faces.copy()).long()
    voxel_indices, _dual_vertices, _intersected = (
        o_voxel.convert.mesh_to_flexible_dual_grid(
            vertices,
            faces,
            grid_size=64,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            face_weight=1.0,
            boundary_weight=0.2,
            regularization_weight=1e-2,
            timing=True,
        )
    )
    occupancy = torch.zeros(
        (1, 1, 64, 64, 64), dtype=torch.float32, device="cuda"
    )
    indexes = voxel_indices.long().cuda()
    occupancy[0, 0, indexes[:, 0], indexes[:, 1], indexes[:, 2]] = 1.0
    latent = encoder(occupancy, sample_posterior=False)
    return latent, int(len(voxel_indices))


def _structure_protection_mask(dense_vertex_mask, torch):
    mask = torch.nn.functional.max_pool3d(
        dense_vertex_mask, kernel_size=4, stride=4
    )
    mask = torch.nn.functional.max_pool3d(mask, 3, stride=1, padding=1)
    return (mask > 1e-6).to(mask.dtype)


def _remap_source_slat(
    source,
    target_coords,
    dense_protection,
    sparse,
    torch,
):
    """Map source features onto Main-generated coords without merging coords."""
    grid = int(dense_protection.shape[-1])
    source_xyz = source.coords[:, 1:].long().clamp(0, grid - 1)
    target_xyz = target_coords[:, 1:].long().clamp(0, grid - 1)
    source_linear = (
        source_xyz[:, 0] * grid * grid
        + source_xyz[:, 1] * grid
        + source_xyz[:, 2]
    )
    target_linear = (
        target_xyz[:, 0] * grid * grid
        + target_xyz[:, 1] * grid
        + target_xyz[:, 2]
    )
    lookup = torch.full(
        (grid * grid * grid,), -1, dtype=torch.long, device=source.device
    )
    lookup[source_linear] = torch.arange(
        len(source_linear), device=source.device
    )
    source_indexes = lookup[target_linear]
    found = source_indexes >= 0
    feats = torch.zeros(
        (len(target_coords), source.feats.shape[1]),
        dtype=source.feats.dtype,
        device=source.device,
    )
    feats[found] = source.feats[source_indexes[found]]
    mapped = sparse.SparseTensor(feats, target_coords.contiguous())
    protected = (
        (_mask_at_coords(dense_protection, target_coords) > 1e-6)
        & found[:, None]
    ).to(feats.dtype)
    return mapped, protected, found


def _mask_at_coords(dense_mask, coords):
    grid = int(dense_mask.shape[-1])
    coords = coords[:, 1:].long().clamp(0, grid - 1)
    return dense_mask[
        0, 0, coords[:, 0], coords[:, 1], coords[:, 2]
    ][:, None]


def _restore_protected(source, candidate, protected_mask):
    return (
        (1.0 - protected_mask) * candidate
        + protected_mask * source
    )


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


def _reuse_decoded_shape_cache(
    stage_result_path: Path,
    decoded_cache_path: Path,
    shape_request_key: str,
) -> bool:
    """Reuse only the raw neural decode and migrate legacy stage manifests."""
    if not stage_result_path.is_file() or not decoded_cache_path.is_file():
        return False
    cached_stage = json.loads(stage_result_path.read_text(encoding="utf-8"))
    if cached_stage.get("shape_request_key") != shape_request_key:
        return False
    raw_cache = str(decoded_cache_path.resolve())
    cached_stage["shape"] = raw_cache
    cached_stage["postprocess_cache"] = raw_cache
    cached_stage.pop("pre_simplification", None)
    stage_result_path.write_text(
        json.dumps(cached_stage, indent=2) + "\n", encoding="utf-8"
    )
    return True


def _postprocess_refined_mesh(
    cache_path, output, settings, resolution
):
    from trellis_mesh_postprocess import run_mesh_postprocess

    return run_mesh_postprocess(
        cache_path,
        output,
        settings,
        resolution,
        progress=_emit,
    )


def _mesh_contract(mesh, np) -> dict[str, object]:
    from trellis_mesh_postprocess import _topology_counts

    faces = np.asarray(mesh.faces, dtype=np.int64)
    normals = np.asarray(mesh.vertex_normals)
    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(faces)),
        "components": int(len(mesh.split(only_watertight=False))),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "finite_vertex_normals": bool(np.isfinite(normals).all()),
        **_topology_counts(faces),
    }


def _project_from_local_gltf(mesh, center_trellis, scale, np):
    """Apply the region denormalization to an already glTF Y-up mesh."""
    center_gltf = np.asarray(
        (center_trellis[0], center_trellis[2], -center_trellis[1]),
        dtype=np.float64,
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] *= float(scale)
    transform[:3, 3] = center_gltf
    result = mesh.copy()
    result.apply_transform(transform)
    return result


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: trellis_refine_runner.py REQUEST.json")
    request_path = Path(sys.argv[1]).expanduser().resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol") != 1:
        raise ValueError("unsupported region refine request protocol")
    output = Path(request["output_dir"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    stage_result_path = output / "shape-stage-result.json"
    decoded_cache_path = output / "decoded-region-raw.npz"
    if _reuse_decoded_shape_cache(
        stage_result_path,
        decoded_cache_path,
        str(request.get("shape_request_key", "")),
    ):
        _emit(f"[shape-stage-cache] raw decode: {decoded_cache_path.name}")
        return 0
    source_path = Path(request["source_region"]).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    schedule = tuple(map(str, request["schedule"]))
    if len(schedule) != int(request["steps"]) or not schedule:
        raise ValueError("region refine schedule length does not match steps")
    warmup_steps = int(request.get("warmup_steps", 0))
    if not 0 <= warmup_steps <= len(schedule):
        raise ValueError("region refine warmup is outside the schedule")
    if any(view_id != "eye-000" for view_id in schedule[:warmup_steps]):
        raise ValueError("region refine warmup must use eye-000")
    _emit(
        f"[schedule] {warmup_steps} front warmup step(s), "
        f"{len(schedule) - warmup_steps} alternating step(s) per stage"
    )

    trellis_root = Path(request["trellis_root"]).expanduser().resolve()
    model_path = Path(request["model_path"]).expanduser().resolve()
    sys.path.insert(0, str(trellis_root))
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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

    source_structure_path = output / "source-structure-latent.npz"
    refined_structure_path = output / "refined-structure-latent.npz"
    structure_mask_path = output / "structure-protection-mask.npz"
    source_shape_512_path = output / "source-shape-512-slat.npz"
    refined_shape_512_path = output / "refined-shape-512-slat.npz"
    source_slat_path = output / "source-shape-slat.npz"
    repaint_source_slat_path = output / "refine-start-shape-slat.npz"
    refined_slat_path = output / "refined-shape-slat.npz"
    token_mask_path = output / "shape-protection-mask.npz"
    cascade_state_path = output / "cascade-state.json"
    resume_decode = all(
        path.is_file()
        for path in (
            source_slat_path,
            repaint_source_slat_path,
            refined_slat_path,
            token_mask_path,
            cascade_state_path,
        )
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
    requested_resolution = int(request["resolution"])
    actual_resolution = requested_resolution
    stage_strengths = _region_stage_strengths(request["strength"])
    decoder = pipeline.models["shape_slat_decoder"]
    decoder_mlps, decoder_upsamples = _enable_chunked_decoder(decoder, torch)
    _emit(
        f"[low-vram] chunked decoder: {decoder_mlps} MLP, "
        f"{decoder_upsamples} upsample blocks; "
        f"activation budget {ENCODER_MLP_CHUNK_BYTES // (1024 * 1024)} MiB"
    )
    if resume_decode:
        _emit(
            "[resume] complete hierarchical latent caches found; "
            "skipping source encode and Main cascade"
        )
        source_slat = _load_sparse(source_slat_path, sparse, torch, np)
        repaint_source_slat = _load_sparse(
            repaint_source_slat_path, sparse, torch, np
        )
        refined_slat = _load_sparse(refined_slat_path, sparse, torch, np)
        with np.load(token_mask_path, allow_pickle=False) as saved:
            protected_mask = (
                torch.from_numpy(saved["protected"].copy()).float().cuda()[:, None]
            )
        protected_mask = protected_mask.to(
            refined_slat.device, refined_slat.feats.dtype
        )
        cascade_state = json.loads(
            cascade_state_path.read_text(encoding="utf-8")
        )
        actual_resolution = int(cascade_state["actual_resolution"])
        protected_tokens = int((protected_mask[:, 0] > 0.5).sum().item())
        editable_tokens = int(len(refined_slat.coords) - protected_tokens)
        surface_voxels = cascade_state.get("surface_voxels")
        structure_voxels = cascade_state.get("structure_voxels")
        occupancy_tokens = int(cascade_state["occupancy_tokens"])
        shape_512_tokens = int(cascade_state["shape_512_tokens"])
        protected_source_tokens_missing = int(
            cascade_state.get("protected_source_tokens_missing", 0)
        )
        structure_delta_mean = float(cascade_state["structure_delta_mean"])
        structure_delta_frozen_max = float(
            cascade_state["structure_delta_frozen_max"]
        )
    else:
        try:
            from .trellis_shape_runner import (
                _decode_sparse_coords,
                _quantize_high_coords,
                _sample_shape,
                _sample_sparse_latent,
            )
        except ImportError:
            from trellis_shape_runner import (
                _decode_sparse_coords,
                _quantize_high_coords,
                _sample_shape,
                _sample_sparse_latent,
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

        _emit("[source-encode] sparse structure occupancy at resolution 64")
        structure_encoder = models.from_pretrained(
            SPARSE_STRUCTURE_ENCODER
        ).eval().cuda()
        source_structure, structure_voxels = _encode_sparse_structure(
            source_local, structure_encoder, torch, o_voxel
        )
        structure_encoder.cpu()
        del structure_encoder
        torch.cuda.empty_cache()
        _save_dense(source_structure_path, source_structure)
        dense_protection_64 = _dense_vertex_mask(
            local_vertices, vertex_weights, 64, torch
        )
        structure_mask = _structure_protection_mask(
            dense_protection_64, torch
        ).to(source_structure.dtype)
        _save_dense(structure_mask_path, structure_mask)
        if bool((structure_mask > 0.5).all().item()):
            raise ValueError(
                "protection mask covers the complete structural latent"
            )

        encoder_path = model_path / "ckpts" / "shape_enc_next_dc_f16c32_fp16"
        encoder = models.from_pretrained(str(encoder_path)).eval()
        chunked_blocks = _enable_chunked_encoder_mlp(encoder, torch)
        _emit(
            f"[low-vram] chunked {chunked_blocks} Shape VAE encoder MLP "
            f"blocks; activation budget "
            f"{ENCODER_MLP_CHUNK_BYTES // (1024 * 1024)} MiB"
        )
        encoder = encoder.cuda()
        _emit("[source-encode] Shape-SLat at resolution 512")
        source_shape_512, _source_512_voxels = _mesh_to_shape_slat(
            source_local, encoder, 512, sparse, torch, o_voxel
        )
        encoder.cpu()
        torch.cuda.empty_cache()
        _save_sparse(source_shape_512_path, source_shape_512)

        cfg = float(request["cfg"])
        torch.manual_seed(int(request["seed"]))
        _emit(f"[condition-512] {len(prepared)} manual view patch(es)")
        conditions_512 = {
            view_id: pipeline.get_cond([image], 512)
            for view_id, image in prepared.items()
        }
        _emit(
            "[stage-strengths] "
            f"occupancy={stage_strengths['occupancy']:.4f}, "
            f"shape-512={stage_strengths['shape_512']:.4f}, "
            "shape-high-resolution="
            f"{stage_strengths['shape_high_resolution']:.4f}"
        )
        _emit(
            "[occupancy-start] source+noise "
            f"strength={stage_strengths['occupancy']:.4f}; "
            "protected latent remains clean"
        )
        refined_structure = _sample_sparse_latent(
            pipeline,
            conditions_512,
            schedule,
            guidance_strength=cfg,
            source=source_structure,
            protected_mask=structure_mask,
            strength=stage_strengths["occupancy"],
        )
        _save_dense(refined_structure_path, refined_structure)
        structure_delta = (
            refined_structure - source_structure
        ).norm(dim=1, keepdim=True)
        structure_delta_mean = float(structure_delta.mean().item())
        structure_frozen = structure_mask[:, 0] > 0.5
        structure_delta_frozen_max = (
            float(structure_delta[structure_frozen[:, None]].max().item())
            if structure_frozen.any()
            else 0.0
        )
        del structure_delta, structure_frozen, source_structure, structure_mask
        coords_512 = _decode_sparse_coords(
            pipeline, refined_structure, 32
        )
        occupancy_tokens = int(len(coords_512))
        del refined_structure
        torch.cuda.empty_cache()

        dense_protection_32 = _dense_vertex_mask(
            local_vertices, vertex_weights, 32, torch
        )
        repaint_source_512, protected_512, _found_512 = (
            _remap_source_slat(
                source_shape_512,
                coords_512,
                dense_protection_32,
                sparse,
                torch,
            )
        )
        _emit(
            f"[shape-512-start] generated coords={len(coords_512)}, "
            f"protected={int(protected_512.sum().item())}, "
            f"editable-strength={stage_strengths['shape_512']:.4f}"
        )
        refined_shape_512 = _sample_shape(
            pipeline,
            conditions_512,
            schedule,
            pipeline.models["shape_slat_flow_model_512"],
            coords_512,
            "shape-512",
            guidance_strength=cfg,
            source=repaint_source_512,
            protected_mask=protected_512,
            strength=stage_strengths["shape_512"],
        )
        _save_sparse(refined_shape_512_path, refined_shape_512)
        shape_512_tokens = int(len(refined_shape_512.coords))
        del (
            conditions_512,
            coords_512,
            repaint_source_512,
            protected_512,
            source_shape_512,
            dense_protection_32,
        )
        torch.cuda.empty_cache()

        if pipeline.low_vram:
            decoder.to(pipeline.device)
            decoder.low_vram = True
        high_coords = decoder.upsample(refined_shape_512, upsample_times=4)
        if pipeline.low_vram:
            decoder.cpu()
            decoder.low_vram = False
        unique_coords, actual_resolution = _quantize_high_coords(
            high_coords, requested_resolution
        )
        del refined_shape_512, high_coords
        torch.cuda.empty_cache()

        encoder = encoder.cuda()
        _emit(
            f"[source-encode] Shape-SLat at resolution {actual_resolution}"
        )
        source_slat, surface_voxels = _mesh_to_shape_slat(
            source_local,
            encoder,
            actual_resolution,
            sparse,
            torch,
            o_voxel,
        )
        encoder.cpu()
        del encoder
        torch.cuda.empty_cache()
        _save_sparse(source_slat_path, source_slat)

        high_grid = actual_resolution // 16
        dense_protection_high = _dense_vertex_mask(
            local_vertices, vertex_weights, high_grid, torch
        )
        source_protected = (
            _mask_at_coords(dense_protection_high, source_slat.coords) > 1e-6
        )
        repaint_source_slat, protected_mask, _found_high = (
            _remap_source_slat(
                source_slat,
                unique_coords,
                dense_protection_high,
                sparse,
                torch,
            )
        )
        protected_source_tokens_missing = max(
            0,
            int(source_protected.sum().item())
            - int(protected_mask.sum().item()),
        )
        _save_sparse(repaint_source_slat_path, repaint_source_slat)
        protected_tokens = int(protected_mask[:, 0].sum().item())
        editable_tokens = int(len(unique_coords) - protected_tokens)
        if not editable_tokens:
            raise ValueError(
                "protection mask covers every high-resolution Shape-SLat token"
            )
        np.savez_compressed(
            token_mask_path,
            coords=np.ascontiguousarray(
                unique_coords.detach().cpu().numpy()
            ),
            protected=np.ascontiguousarray(
                protected_mask[:, 0].detach().float().cpu().numpy()
            ),
        )
        _emit(
            f"[protection-mask-high] protected={protected_tokens}, "
            f"editable={editable_tokens}, "
            f"missing-source-protected={protected_source_tokens_missing}, "
            "editable-strength="
            f"{stage_strengths['shape_high_resolution']:.4f}"
        )

        _emit(f"[condition-1024] {len(prepared)} manual view patch(es)")
        conditions_1024 = {
            view_id: pipeline.get_cond([image], 1024)
            for view_id, image in prepared.items()
        }
        refined_slat = _sample_shape(
            pipeline,
            conditions_1024,
            schedule,
            pipeline.models["shape_slat_flow_model_1024"],
            unique_coords,
            "shape-high-resolution",
            guidance_strength=cfg,
            source=repaint_source_slat,
            protected_mask=protected_mask,
            strength=stage_strengths["shape_high_resolution"],
        )
        _save_sparse(refined_slat_path, refined_slat)
        cascade_state = {
            "requested_resolution": requested_resolution,
            "actual_resolution": actual_resolution,
            "structure_voxels": structure_voxels,
            "surface_voxels": surface_voxels,
            "occupancy_tokens": occupancy_tokens,
            "shape_512_tokens": shape_512_tokens,
            "stage_strengths": stage_strengths,
            "protected_source_tokens_missing": (
                protected_source_tokens_missing
            ),
            "structure_delta_mean": structure_delta_mean,
            "structure_delta_frozen_max": structure_delta_frozen_max,
        }
        cascade_state_path.write_text(
            json.dumps(cascade_state, indent=2) + "\n", encoding="utf-8"
        )
        del (
            conditions_1024,
            unique_coords,
            source_protected,
            dense_protection_high,
            dense_protection_64,
        )
        torch.cuda.empty_cache()

    if not torch.equal(refined_slat.coords, repaint_source_slat.coords):
        raise RuntimeError("high-resolution sampler changed its coordinate support")
    delta = (refined_slat.feats - repaint_source_slat.feats).norm(dim=1)
    frozen = protected_mask[:, 0] > 0.5
    feature_delta_mean = float(delta.mean().item())
    feature_delta_frozen_max = (
        float(delta[frozen].max().item()) if frozen.any() else 0.0
    )
    fully_frozen_tokens = protected_tokens
    shape_tokens = int(len(refined_slat.coords))
    del delta, frozen

    _emit(
        f"[decode] standalone refined region at resolution {actual_resolution}"
    )
    decoded_vertices, decoded_faces = _decode_mesh_cache(
        pipeline,
        refined_slat,
        actual_resolution,
        decoded_cache_path,
        np,
        torch,
    )
    del decoded_vertices, decoded_faces
    # The decoder's sparse subdivision tensors are much larger than the final
    # mesh. Release the entire neural stage before CuMesh allocates its graph.
    del (
        pipeline,
        decoder,
        source_slat,
        repaint_source_slat,
        refined_slat,
        protected_mask,
    )
    torch.cuda.empty_cache()
    _emit("[low-vram] TRELLIS tensors released before raw mesh handoff")
    result = {
        "protocol": 1,
        "status": "shape-stage-complete",
        "request_key": str(request["request_key"]),
        "shape_request_key": str(request["shape_request_key"]),
        "geometry_fingerprint": str(request["geometry_fingerprint"]),
        "shape": str(decoded_cache_path.resolve()),
        "postprocess_cache": str(decoded_cache_path.resolve()),
        "source_region": str(source_region_path.resolve()),
        "source_region_local": str(source_local_path.resolve()),
        "source_slat": str(source_slat_path.resolve()),
        "refined_slat": str(refined_slat_path.resolve()),
        "token_mask": str(token_mask_path.resolve()),
        "requested_resolution": requested_resolution,
        "resolution": actual_resolution,
        "actual_resolution": actual_resolution,
        "structure_voxels": structure_voxels,
        "surface_voxels": surface_voxels,
        "occupancy_tokens": occupancy_tokens,
        "shape_512_tokens": shape_512_tokens,
        "shape_tokens": shape_tokens,
        "stage_strengths": stage_strengths,
        "protected_tokens": protected_tokens,
        "editable_tokens": editable_tokens,
        "fully_frozen_tokens": fully_frozen_tokens,
        "feature_delta_mean": feature_delta_mean,
        "feature_delta_frozen_max": feature_delta_frozen_max,
        "structure_delta_mean": structure_delta_mean,
        "structure_delta_frozen_max": structure_delta_frozen_max,
        "protected_source_tokens_missing": protected_source_tokens_missing,
        "source_structure": str(source_structure_path.resolve()),
        "refined_structure": str(refined_structure_path.resolve()),
        "structure_mask": str(structure_mask_path.resolve()),
        "source_shape_512": str(source_shape_512_path.resolve()),
        "refined_shape_512": str(refined_shape_512_path.resolve()),
        "repaint_source_slat": str(repaint_source_slat_path.resolve()),
        "cube_center_gltf": list(map(float, cube["center"])),
        "cube_center_trellis": cube_center.tolist(),
        "cube_side": cube_side,
        "normalization_center_trellis": center.tolist(),
        "normalization_scale": scale,
        "patches": list(dict.fromkeys(schedule)),
        "schedule": list(schedule),
        "assembly": "not-run",
    }
    stage_result_path.write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _emit(f"[shape-stage-complete] postprocess cache: {decoded_cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
