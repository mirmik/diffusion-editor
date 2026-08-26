#!/usr/bin/env python3
"""Heavy TRELLIS.2 shape-only worker used by Multiview Shape Studio."""

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


def _alternating_sample(
    sampler,
    model,
    noise,
    conditions,
    schedule,
    *,
    stage: str,
    rescale_t: float,
    guidance_strength: float,
    guidance_rescale: float,
    guidance_interval: tuple[float, float],
):
    import numpy as np

    sample = noise
    times = np.linspace(1, 0, len(schedule) + 1)
    times = rescale_t * times / (1 + (rescale_t - 1) * times)
    for index, ((current, previous), view_id) in enumerate(
        zip(zip(times[:-1], times[1:]), schedule), 1
    ):
        condition = conditions[view_id]
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
        )
        sample = out.pred_x_prev
        _emit(f"[{stage}] {index}/{len(schedule)} · {view_id}")
    return sample


def _sample_sparse(pipeline, conditions, schedule):
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
        schedule,
        stage="occupancy",
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


def _sample_shape(pipeline, conditions, schedule, model, coords, stage: str):
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
        schedule,
        stage=stage,
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


def _cache_decoded_meshes(meshes, path: Path) -> dict[str, int]:
    import numpy as np

    vertices = []
    faces = []
    vertex_offset = 0
    for mesh in meshes:
        mesh_vertices = mesh.vertices.detach().float().cpu().numpy()
        mesh_faces = mesh.faces.detach().int().cpu().numpy()
        vertices.append(mesh_vertices)
        faces.append(mesh_faces + vertex_offset)
        vertex_offset += len(mesh_vertices)
    combined_vertices = np.ascontiguousarray(
        np.concatenate(vertices, axis=0), dtype=np.float32
    )
    combined_faces = np.ascontiguousarray(
        np.concatenate(faces, axis=0), dtype=np.int32
    )
    np.savez(
        path,
        vertices=combined_vertices,
        faces=combined_faces,
    )
    return {
        "source_vertices": int(len(combined_vertices)),
        "source_faces": int(len(combined_faces)),
    }


def _postprocess_shape(
    cache_path: Path,
    output: Path,
    settings: dict,
    actual_resolution: int,
):
    from trellis_mesh_postprocess import run_mesh_postprocess

    return run_mesh_postprocess(
        cache_path,
        output,
        settings,
        actual_resolution,
        progress=_emit,
    )


def _gltf_y_up_transform():
    import numpy as np

    return np.asarray(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
            (0.0, -1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        dtype=np.float64,
    )


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: trellis_shape_runner.py REQUEST.json")
    request_path = Path(sys.argv[1]).resolve()
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if request.get("protocol") != 1:
        raise ValueError("unsupported shape request protocol")
    output = Path(request["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    schedule = tuple(str(value) for value in request["schedule"])
    if len(schedule) != int(request["steps"]):
        raise ValueError("schedule length does not match steps")
    if not schedule:
        raise ValueError("empty TRELLIS.2 schedule")

    trellis_root = Path(request["trellis_root"]).resolve()
    model_path = Path(request["model_path"]).resolve()
    sys.path.insert(0, str(trellis_root))
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "xformers")

    import torch
    from PIL import Image
    from trellis2.pipelines import Trellis2ImageTo3DPipeline

    torch.set_grad_enabled(False)
    _emit("[load] TRELLIS.2 pipeline")
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(model_path))
    pipeline.low_vram = True
    pipeline.cuda()

    images = {str(item["id"]): Path(item["image"]) for item in request["views"]}
    images.setdefault("eye-000", Path(request["front"]))
    missing = sorted(set(schedule) - images.keys())
    if missing:
        raise ValueError(f"schedule refers to missing views: {missing}")
    prepared = {}
    for view_id in dict.fromkeys(schedule):
        source = images[view_id]
        if not source.is_file():
            raise FileNotFoundError(source)
        image = pipeline.preprocess_image(Image.open(source))
        image.save(output / f"prepared-{view_id}.png")
        prepared[view_id] = image

    torch.manual_seed(int(request["seed"]))
    _emit(f"[condition-512] {len(prepared)} view(s)")
    conditions_512 = {
        view_id: pipeline.get_cond([image], 512)
        for view_id, image in prepared.items()
    }
    coords = _sample_sparse(pipeline, conditions_512, schedule)
    lr_shape = _sample_shape(
        pipeline,
        conditions_512,
        schedule,
        pipeline.models["shape_slat_flow_model_512"],
        coords,
        "shape-512",
    )
    _save_sparse(output / "shape-512-slat.npz", lr_shape)

    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    high_coords = decoder.upsample(lr_shape, upsample_times=4)
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False

    requested_resolution = int(request["resolution"])
    actual_resolution = requested_resolution
    max_num_tokens = 49_152
    while True:
        grid_resolution = actual_resolution // 16
        quantized = torch.cat(
            (
                high_coords[:, :1],
                ((high_coords[:, 1:] + 0.5) / 512 * grid_resolution).int(),
            ),
            dim=1,
        )
        unique_coords = quantized.unique(dim=0)
        if len(unique_coords) < max_num_tokens or actual_resolution == 1024:
            break
        actual_resolution -= 128
    del conditions_512, coords, high_coords, quantized
    torch.cuda.empty_cache()

    _emit(f"[condition-1024] {len(prepared)} view(s)")
    conditions_1024 = {
        view_id: pipeline.get_cond([image], 1024)
        for view_id, image in prepared.items()
    }
    shape = _sample_shape(
        pipeline,
        conditions_1024,
        schedule,
        pipeline.models["shape_slat_flow_model_1024"],
        unique_coords,
        "shape-high-resolution",
    )
    slat_path = output / "shape-high-resolution-slat.npz"
    _save_sparse(slat_path, shape)
    _emit(f"[decode] resolution {actual_resolution}")
    meshes, _shape_subs = pipeline.decode_shape_slat(shape, actual_resolution)
    cache_path = output / "decoded-mesh-z-up.npz"
    counts = _cache_decoded_meshes(meshes, cache_path)
    shape_path, postprocess_report = _postprocess_shape(
        cache_path,
        output,
        dict(request["postprocess"]),
        actual_resolution,
    )
    result = {
        "protocol": 1,
        "status": "complete",
        "shape": str(shape_path.resolve()),
        "shape_cache": str(cache_path.resolve()),
        "shape_slat": str(slat_path.resolve()),
        "generation_key": str(request["generation_key"]),
        "requested_resolution": requested_resolution,
        "actual_resolution": actual_resolution,
        "occupancy_tokens": int(unique_coords.shape[0]),
        "postprocess": postprocess_report,
        **counts,
    }
    (output / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    _emit(f"[complete] {shape_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
