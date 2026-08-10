"""Editor-owned staged entry point for an installed TRELLIS.2 runtime."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys


STAGES = (
    "source_image",
    "sparse_occupancy",
    "lr_shape_flow",
    "lr_shape_latent",
    "hr_coordinates",
    "hr_shape_flow",
    "hr_shape_latent",
    "texture_flow",
    "texture_latent",
    "final_mesh",
)


class StageReporter:
    def __init__(self, path: Path) -> None:
        self._path = path

    def emit(
        self,
        stage: str,
        status: str,
        *,
        progress: int = 0,
        total: int = 0,
        artifact_path: Path | None = None,
        preview_kind: str | None = None,
    ) -> None:
        payload = {
            "protocol": 1,
            "stage": stage,
            "status": status,
            "progress": int(progress),
            "total": int(total),
        }
        if artifact_path is not None:
            payload["artifact_path"] = str(artifact_path)
            payload["preview_kind"] = preview_kind or "mesh"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def _reached(target: str, stage: str) -> bool:
    return STAGES.index(target) <= STAGES.index(stage)


def _tensor_numpy(value):
    return value.detach().to("cpu").numpy()


def _preview_transform():
    import numpy as np

    return np.diag((-1.0, 1.0, -1.0, 1.0))


def _merge_occupied_cells(xyz):
    remaining = {tuple(map(int, point)) for point in xyz}
    cuboids = []
    while remaining:
        x0, y0, z0 = min(remaining)
        x1 = x0 + 1
        while (x1, y0, z0) in remaining:
            x1 += 1
        y1 = y0 + 1
        while all((x, y1, z0) in remaining for x in range(x0, x1)):
            y1 += 1
        z1 = z0 + 1
        while all(
            (x, y, z1) in remaining
            for x in range(x0, x1)
            for y in range(y0, y1)
        ):
            z1 += 1
        for x in range(x0, x1):
            for y in range(y0, y1):
                for z in range(z0, z1):
                    remaining.remove((x, y, z))
        cuboids.append(((x0, y0, z0), (x1, y1, z1)))
    return cuboids


def _occupancy_preview(coords, resolution: int, path: Path) -> Path:
    import numpy as np
    import trimesh

    xyz = _tensor_numpy(coords[:, -3:]).astype(np.int64)
    corners = np.asarray((
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1),
    ), dtype=np.float32)
    box_faces = np.asarray((
        (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7),
        (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5),
        (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7),
    ), dtype=np.int64)
    vertices = []
    faces = []
    for index, (minimum, maximum) in enumerate(_merge_occupied_cells(xyz)):
        bounds = np.asarray((minimum, maximum), dtype=np.float32)
        vertices.append(
            (bounds[0] + corners * (bounds[1] - bounds[0])) / resolution - 0.5
        )
        faces.append(box_faces + index * 8)
    mesh = trimesh.Trimesh(
        vertices=np.concatenate(vertices),
        faces=np.concatenate(faces),
        process=False,
    )
    mesh.apply_transform(_preview_transform())
    mesh.export(path)
    return path


def _point_preview(coords, resolution: int, path: Path) -> Path:
    import numpy as np
    import trimesh

    xyz = _tensor_numpy(coords[:, -3:]).astype(np.float32)
    if len(xyz) > 20_000:
        xyz = xyz[::math.ceil(len(xyz) / 20_000)]
    xyz = (xyz + 0.5) / max(float(resolution), 1.0) - 0.5
    radius = 0.32 / max(float(resolution), 1.0)
    offsets = np.asarray((
        (radius, radius, radius), (radius, -radius, -radius),
        (-radius, radius, -radius), (-radius, -radius, radius),
    ), dtype=np.float32)
    vertices = (xyz[:, None, :] + offsets[None]).reshape(-1, 3)
    base = np.arange(len(xyz), dtype=np.int64)[:, None, None] * 4
    tetra_faces = np.asarray((
        (0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2),
    ), dtype=np.int64)
    faces = (base + tetra_faces[None]).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.apply_transform(_preview_transform())
    mesh.export(path)
    return path


def _shape_preview(meshes, path: Path, face_target: int = 100_000) -> Path:
    import trimesh
    from trellis2.representations import Mesh

    scene = trimesh.Scene()
    for index, mesh in enumerate(meshes):
        preview = Mesh(mesh.vertices.clone(), mesh.faces.clone())
        if len(preview.faces) > face_target:
            preview.simplify(face_target)
        geometry = trimesh.Trimesh(
            vertices=_tensor_numpy(preview.vertices),
            faces=_tensor_numpy(preview.faces),
            process=False,
        )
        geometry.apply_transform(_preview_transform())
        scene.add_geometry(geometry, node_name=f"shape-{index}")
    scene.export(path)
    return path


def run(args) -> None:
    sys.path.insert(0, str(Path(args.trellis2_root).resolve()))
    import numpy as np
    import o_voxel
    import torch
    from easydict import EasyDict as edict
    from PIL import Image
    from trellis2.modules import sparse as sp
    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines.samplers.flow_euler import FlowEulerSampler
    from trellis2.representations import MeshWithVoxel

    # The upstream end-to-end pipeline is decorated with ``torch.no_grad``.
    # Our staged runner calls the same operations individually, so establish
    # the equivalent process-wide inference context before creating tensors.
    torch.set_grad_enabled(False)

    reporter = StageReporter(Path(args.events))
    active_flow = {"stage": ""}

    def tracked_sample(
        sampler,
        model,
        noise,
        cond=None,
        steps=50,
        rescale_t=1.0,
        verbose=True,
        tqdm_desc="Sampling",
        **kwargs,
    ):
        del verbose, tqdm_desc
        sample = noise
        times = np.linspace(1, 0, steps + 1)
        times = rescale_t * times / (1 + (rescale_t - 1) * times)
        result = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        pairs = list(zip(times[:-1], times[1:]))
        for index, (current, previous) in enumerate(pairs, 1):
            out = sampler.sample_once(
                model, sample, float(current), float(previous), cond, **kwargs
            )
            sample = out.pred_x_prev
            result.pred_x_t.append(out.pred_x_prev)
            result.pred_x_0.append(out.pred_x_0)
            reporter.emit(
                active_flow["stage"], "running",
                progress=index, total=len(pairs),
            )
        result.samples = sample
        return result

    FlowEulerSampler.sample = tracked_sample
    artifact_root = Path(args.output).resolve().parent
    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model_path)
    pipeline.low_vram = args.low_vram
    pipeline.cuda()
    image = pipeline.preprocess_image(Image.open(args.image))
    image.save(artifact_root / "preprocessed.png")
    torch.manual_seed(args.seed)

    shape_sampler = {
        "steps": args.steps,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    }
    sparse_sampler = {
        **shape_sampler,
        "guidance_rescale": 0.7,
        "rescale_t": 5.0,
    }
    texture_sampler = {
        "steps": args.steps,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }
    cond_512 = pipeline.get_cond([image], 512)

    active_flow["stage"] = "sparse_occupancy"
    reporter.emit("sparse_occupancy", "running", total=args.steps)
    coords = pipeline.sample_sparse_structure(
        cond_512, 32, 1, sparse_sampler
    )
    sparse_preview = _occupancy_preview(
        coords, 32, artifact_root / "sparse-occupancy.glb"
    )
    reporter.emit(
        "sparse_occupancy", "ready",
        progress=args.steps, total=args.steps,
        artifact_path=sparse_preview, preview_kind="mesh",
    )
    if _reached(args.target_stage, "sparse_occupancy"):
        return

    active_flow["stage"] = "lr_shape_flow"
    reporter.emit("lr_shape_flow", "running", total=args.steps)
    lr_slat = pipeline.sample_shape_slat(
        cond_512,
        pipeline.models["shape_slat_flow_model_512"],
        coords,
        shape_sampler,
    )
    reporter.emit(
        "lr_shape_flow", "ready", progress=args.steps, total=args.steps
    )
    if _reached(args.target_stage, "lr_shape_flow"):
        return
    lr_meshes, _lr_subs = pipeline.decode_shape_slat(lr_slat, 512)
    lr_preview = _shape_preview(lr_meshes, artifact_root / "lr-shape.glb")
    reporter.emit(
        "lr_shape_latent", "ready",
        artifact_path=lr_preview, preview_kind="mesh",
    )
    if _reached(args.target_stage, "lr_shape_latent"):
        return

    decoder = pipeline.models["shape_slat_decoder"]
    if pipeline.low_vram:
        decoder.to(pipeline.device)
        decoder.low_vram = True
    hr_coords = decoder.upsample(lr_slat, upsample_times=4)
    if pipeline.low_vram:
        decoder.cpu()
        decoder.low_vram = False
    actual_resolution = int(args.resolution)
    while True:
        grid_resolution = actual_resolution // 16
        quantized = torch.cat((
            hr_coords[:, :1],
            ((hr_coords[:, 1:] + 0.5) / 512 * grid_resolution).int(),
        ), dim=1)
        hr_coords_unique = quantized.unique(dim=0)
        if (
            len(hr_coords_unique) < args.max_num_tokens
            or actual_resolution == 1024
        ):
            break
        actual_resolution -= 128
    coordinates_preview = _point_preview(
        hr_coords_unique,
        actual_resolution // 16,
        artifact_root / "hr-coordinates.glb",
    )
    reporter.emit(
        "hr_coordinates", "ready",
        artifact_path=coordinates_preview, preview_kind="mesh",
    )
    if _reached(args.target_stage, "hr_coordinates"):
        return

    del cond_512, coords, lr_meshes, _lr_subs, lr_slat
    del hr_coords, quantized
    torch.cuda.empty_cache()
    cond_1024 = pipeline.get_cond([image], 1024)
    active_flow["stage"] = "hr_shape_flow"
    reporter.emit("hr_shape_flow", "running", total=args.steps)
    shape_slat = pipeline.sample_shape_slat(
        cond_1024,
        pipeline.models["shape_slat_flow_model_1024"],
        hr_coords_unique,
        shape_sampler,
    )
    reporter.emit(
        "hr_shape_flow", "ready", progress=args.steps, total=args.steps
    )
    if _reached(args.target_stage, "hr_shape_flow"):
        return
    shape_meshes, shape_subs = pipeline.decode_shape_slat(
        shape_slat, actual_resolution
    )
    shape_preview = _shape_preview(
        shape_meshes, artifact_root / "hr-shape.glb"
    )
    reporter.emit(
        "hr_shape_latent", "ready",
        artifact_path=shape_preview, preview_kind="mesh",
    )
    if _reached(args.target_stage, "hr_shape_latent"):
        return

    active_flow["stage"] = "texture_flow"
    reporter.emit("texture_flow", "running", total=args.steps)
    texture_slat = pipeline.sample_tex_slat(
        cond_1024,
        pipeline.models["tex_slat_flow_model_1024"],
        shape_slat,
        texture_sampler,
    )
    reporter.emit(
        "texture_flow", "ready", progress=args.steps, total=args.steps
    )
    del cond_1024
    torch.cuda.empty_cache()
    if _reached(args.target_stage, "texture_flow"):
        return
    reporter.emit("texture_latent", "ready")
    if _reached(args.target_stage, "texture_latent"):
        return

    tex_voxels = pipeline.decode_tex_slat(texture_slat, shape_subs)
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
    mesh = meshes[0]
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0.0,
        verbose=True,
    )
    glb.apply_transform(np.asarray((
        (-1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, -1, 0, 0),
        (0, 0, 0, 1),
    ), dtype=np.float64))
    glb.export(args.output, extension_webp=True)
    reporter.emit(
        "final_mesh", "ready",
        artifact_path=Path(args.output), preview_kind="mesh",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trellis2-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--target-stage", choices=STAGES, required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decimation-target", type=int, default=200_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--max-num-tokens", type=int, default=49_152)
    parser.add_argument("--low-vram", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
