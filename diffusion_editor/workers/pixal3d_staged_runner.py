"""Project-owned staged entry point for an installed Pixal3D runtime.

This file is executed by Pixal3D's Python, not by the editor environment.
It deliberately imports the upstream package from ``--pixal3d-root`` while
keeping the stage protocol and stopping semantics under editor ownership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time


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
    """Align raw Pixal3D previews with ``to_glb`` in glTF Y-up space.

    The editor performs the standard glTF Y-up to engine Z-up conversion when
    loading every artifact. Applying Pixal3D's final export transform here as
    well would rotate point previews onto their backs. Raw decoded meshes only
    need the 180-degree Y-axis correction introduced by ``o_voxel.to_glb``.
    """
    import numpy as np

    return np.diag((-1.0, 1.0, -1.0, 1.0))


def _merge_occupied_cells(
    xyz,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int]]]:
    """Greedily cover occupied grid cells with non-overlapping cuboids.

    Returned maxima are exclusive. The occupancy remains exact; merging only
    removes internal faces from the human-readable preview.
    """
    remaining = {tuple(map(int, point)) for point in xyz}
    cuboids = []
    while remaining:
        x0, y0, z0 = min(remaining)
        x1 = x0 + 1
        while (x1, y0, z0) in remaining:
            x1 += 1

        y1 = y0 + 1
        while all(
            (x, y1, z0) in remaining
            for x in range(x0, x1)
        ):
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
    """Export exact sparse occupancy as merged axis-aligned cuboids."""
    import numpy as np
    import trimesh

    xyz = _tensor_numpy(coords[:, -3:]).astype(np.int64)
    cuboids = _merge_occupied_cells(xyz)
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
    scale = max(float(resolution), 1.0)
    for minimum, maximum in cuboids:
        bounds = np.asarray((minimum, maximum), dtype=np.float32)
        box = bounds[0] + corners * (bounds[1] - bounds[0])
        vertices.append(box / scale - 0.5)
        faces.append(box_faces + (len(vertices) - 1) * 8)
    mesh = trimesh.Trimesh(
        vertices=np.concatenate(vertices, axis=0),
        faces=np.concatenate(faces, axis=0),
        process=False,
    )
    mesh.apply_transform(_preview_transform())
    mesh.export(path)
    return path


def _point_preview(coords, resolution: int, path: Path) -> Path:
    """Export bounded tetrahedra as a portable GLB coordinate preview."""
    import numpy as np
    import trimesh

    xyz = _tensor_numpy(coords[:, -3:]).astype(np.float32)
    if len(xyz) > 20_000:
        xyz = xyz[:: math.ceil(len(xyz) / 20_000)]
    xyz = (xyz + 0.5) / max(float(resolution), 1.0) - 0.5
    radius = 0.32 / max(float(resolution), 1.0)
    offsets = np.asarray((
        (radius, radius, radius),
        (radius, -radius, -radius),
        (-radius, radius, -radius),
        (-radius, -radius, radius),
    ), dtype=np.float32)
    vertices = (xyz[:, None, :] + offsets[None, :, :]).reshape(-1, 3)
    base = np.arange(len(xyz), dtype=np.int64)[:, None] * 4
    faces = (base[:, :, None] + np.asarray((
        (0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2),
    ), dtype=np.int64)[None, :, :]).reshape(-1, 3)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.apply_transform(_preview_transform())
    mesh.export(path)
    return path


def _shape_preview(meshes, path: Path, face_target: int = 100_000) -> Path:
    import trimesh
    from pixal3d.representations import Mesh

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
        # Bake the transform into POSITION data. NativeGLBDocument deliberately
        # builds a mesh primitive without instantiating the glTF scene graph.
        geometry.apply_transform(_preview_transform())
        scene.add_geometry(geometry, node_name=f"shape-{index}")
    scene.export(path)
    return path


def _lr_conditioner(pipeline, resolution: int):
    """Select a trained conditioner while retaining LR grid coordinates."""
    if resolution == 512:
        return pipeline.image_cond_model_shape_512, None
    if resolution == 1024:
        return pipeline.image_cond_model_shape_1024, 32
    raise ValueError(f"Unsupported LR conditioning resolution: {resolution}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _save_shape_checkpoint(
    path: Path,
    normalized_latent,
    resolution: int,
    camera: dict,
    source_path: Path,
    model_path: str,
) -> Path:
    """Write a pickle-free, session-local checkpoint for masked refinement."""
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        protocol_version=np.asarray([1], dtype=np.int32),
        coords=_tensor_numpy(normalized_latent.coords),
        normalized_feats=_tensor_numpy(normalized_latent.feats),
        resolution=np.asarray([resolution], dtype=np.int32),
        camera_angle_x=np.asarray(
            [float(camera["camera_angle_x"])], dtype=np.float64
        ),
        camera_distance=np.asarray(
            [float(camera["distance"])], dtype=np.float64
        ),
        mesh_scale=np.asarray(
            [float(camera.get("mesh_scale", 1.0))], dtype=np.float64
        ),
        source_sha256=np.asarray(_file_sha256(source_path)),
        model_path=np.asarray(str(Path(model_path).resolve())),
    )
    return path


def _project_mask_to_tokens(mask_image, coords, resolution: int, camera: dict):
    """Project an image-space soft mask onto fixed Pixal3D HR coordinates."""
    import numpy as np
    import torch
    import torch.nn.functional as functional
    from PIL import Image
    from pixal3d.trainers.flow_matching.mixins.image_conditioned_proj import (
        project_points_to_image_batch,
    )

    projection_resolution = max(mask_image.size)
    mask = mask_image.convert("L").resize(
        (projection_resolution, projection_resolution),
        resample=Image.Resampling.BILINEAR,
    )
    mask_values = torch.from_numpy(
        np.asarray(mask, dtype=np.float32) / 255.0
    ).to(coords.device)[None, None]

    grid_resolution = resolution // 16
    raw_points = coords[:, 1:].float() / (grid_resolution - 1) * 2.0 - 1.0
    rotation = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        device=coords.device,
    )
    points = raw_points @ rotation.T
    points = points / float(camera.get("mesh_scale", 1.0)) / 2.0
    transform = torch.tensor(
        (
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, -1.0, -float(camera["distance"])),
            (0.0, 1.0, 0.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
        device=coords.device,
    )[None]
    image_points, _depth, valid = project_points_to_image_batch(
        points,
        transform,
        torch.tensor([float(camera["camera_angle_x"])], device=coords.device),
        projection_resolution,
    )
    queries = (image_points + 0.5) / projection_resolution * 2.0 - 1.0
    weights = functional.grid_sample(
        mask_values,
        queries.view(1, -1, 1, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).view(-1)
    return weights * valid.view(-1).to(weights.dtype)


def _run_refine(args, reporter, upstream, sp, torch, np) -> None:
    from PIL import Image

    checkpoint_path = Path(args.refine_checkpoint)
    with np.load(checkpoint_path, allow_pickle=False) as saved:
        if int(saved["protocol_version"][0]) != 1:
            raise ValueError("unsupported Pixal3D refine checkpoint version")
        expected_model = str(Path(args.model_path).resolve())
        if str(saved["model_path"].item()) != expected_model:
            raise ValueError("refine checkpoint belongs to a different model")
        resolution = int(saved["resolution"][0])
        coords = torch.from_numpy(saved["coords"].copy()).to("cuda")
        base_feats = torch.from_numpy(
            saved["normalized_feats"].copy()
        ).to("cuda").float()
        camera = {
            "camera_angle_x": float(saved["camera_angle_x"][0]),
            "distance": float(saved["camera_distance"][0]),
            "mesh_scale": float(saved["mesh_scale"][0]),
        }

    condition_image = Image.open(args.image).convert("RGB")
    mask_image = Image.open(args.refine_mask).convert("L")
    if condition_image.size != mask_image.size:
        raise ValueError("refine mask must match conditioning image dimensions")

    pipeline = upstream.init_pipeline(args.model_path, low_vram=args.low_vram)
    token_mask = _project_mask_to_tokens(
        mask_image, coords, resolution, camera
    )
    if not bool((token_mask > 0.0).any()):
        raise ValueError("refine mask does not select any visible 3D tokens")
    cond = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [condition_image],
        coords,
        grid_resolution_override=resolution // 16,
        **camera,
    )
    x0 = sp.SparseTensor(feats=base_feats, coords=coords)
    generator = torch.Generator(device="cuda").manual_seed(args.refine_seed)
    noise = torch.randn(
        base_feats.shape,
        generator=generator,
        device=base_feats.device,
        dtype=base_feats.dtype,
    )
    sampler = pipeline.shape_slat_sampler
    sigma_min = sampler.sigma_min
    raw_times = np.linspace(args.refine_strength, 0.0, args.refine_steps + 1)
    times = (
        args.refine_rescale_t * raw_times
        / (1 + (args.refine_rescale_t - 1) * raw_times)
    )

    def reference_at(value: float):
        sigma = sigma_min + (1 - sigma_min) * value
        return (1 - value) * base_feats + sigma * noise

    sample = x0.replace(reference_at(float(times[0])))
    flow = pipeline.models["shape_slat_flow_model_1024"]
    if pipeline.low_vram:
        flow.to(pipeline.device)
    blend = token_mask[:, None].to(base_feats.dtype)
    reporter.emit("hr_shape_flow", "running", total=args.refine_steps)
    with torch.inference_mode():
        for index, (current, previous) in enumerate(
            zip(times[:-1], times[1:]), 1
        ):
            out = sampler.sample_once(
                flow,
                sample,
                float(current),
                float(previous),
                cond["cond"],
                neg_cond=cond["neg_cond"],
                guidance_strength=args.refine_guidance,
                guidance_rescale=0.5,
                guidance_interval=(0.6, 1.0),
            )
            reference = reference_at(float(previous))
            sample = out.pred_x_prev.replace(
                out.pred_x_prev.feats * blend + reference * (1 - blend)
            )
            reporter.emit(
                "hr_shape_flow", "running",
                progress=index, total=args.refine_steps,
            )
    if pipeline.low_vram:
        flow.cpu()
    sample = sample.replace(sample.feats * blend + base_feats * (1 - blend))
    reporter.emit(
        "hr_shape_flow", "ready",
        progress=args.refine_steps, total=args.refine_steps,
    )

    refined_checkpoint = Path(args.checkpoint)
    _save_shape_checkpoint(
        refined_checkpoint,
        sample,
        resolution,
        camera,
        Path(args.image),
        args.model_path,
    )
    std = torch.tensor(
        pipeline.shape_slat_normalization["std"], device=sample.device
    )[None]
    mean = torch.tensor(
        pipeline.shape_slat_normalization["mean"], device=sample.device
    )[None]
    decoded = sample.replace(sample.feats * std + mean)
    meshes, _subdivisions = pipeline.decode_shape_slat(decoded, resolution)
    output = _shape_preview(meshes, Path(args.output))
    reporter.emit(
        "final_mesh", "ready", artifact_path=output, preview_kind="mesh"
    )


def run(args) -> None:
    sys.path.insert(0, str(Path(args.pixal3d_root).resolve()))
    import torch
    from PIL import Image
    from pixal3d.modules import sparse as sp
    from pixal3d.pipelines.samplers.flow_euler import FlowEulerSampler
    from easydict import EasyDict as edict
    import inference as upstream
    import o_voxel
    import numpy as np

    torch.set_grad_enabled(False)
    reporter = StageReporter(Path(args.events))
    if args.refine_checkpoint:
        _run_refine(args, reporter, upstream, sp, torch, np)
        return
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
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        pairs = list(zip(t_seq[:-1].tolist(), t_seq[1:].tolist()))
        result = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for index, (t, t_prev) in enumerate(pairs, 1):
            out = sampler.sample_once(model, sample, t, t_prev, cond, **kwargs)
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
    target = args.target_stage
    artifact_root = Path(args.output).resolve().parent
    pipeline = upstream.init_pipeline(args.model_path, low_vram=args.low_vram)
    image = pipeline.preprocess_image(Image.open(args.image))
    preprocessed = artifact_root / "preprocessed.png"
    image.save(preprocessed)

    if args.manual_fov > 0:
        camera_angle_x = float(args.manual_fov)
        grid_point = torch.tensor([-1.0, 0.0, 0.0])
        distance = upstream.distance_from_fov(
            camera_angle_x,
            grid_point,
            torch.tensor([0, args.image_resolution - 1]),
            1.0,
            args.image_resolution,
        )["distance_from_x"]
        camera = {
            "camera_angle_x": camera_angle_x,
            "distance": distance,
            "mesh_scale": 1.0,
        }
    else:
        moge = upstream.load_moge_model(device="cuda")
        camera = upstream.get_camera_params_wild_moge(
            str(preprocessed),
            moge,
            image_resolution=args.image_resolution,
        )
        moge.cpu()
        del moge
        torch.cuda.empty_cache()

    torch.manual_seed(args.seed)
    sampler = {
        "steps": args.steps,
        "guidance_strength": 7.5,
        "guidance_rescale": 0.5,
        "rescale_t": 3.0,
    }
    ss_sampler = {**sampler, "guidance_rescale": 0.7, "rescale_t": 5.0}
    tex_sampler = {
        "steps": args.steps,
        "guidance_strength": 1.0,
        "guidance_rescale": 0.0,
        "rescale_t": 3.0,
    }
    hr_resolution = int(args.resolution)

    reporter.emit("sparse_occupancy", "running", total=args.steps)
    active_flow["stage"] = "sparse_occupancy"
    cond_ss = pipeline.get_proj_cond_ss([image], **camera)
    coords = pipeline.sample_sparse_structure(cond_ss, 32, 1, ss_sampler)
    del cond_ss
    torch.cuda.empty_cache()
    sparse_preview = _occupancy_preview(
        coords, 32, artifact_root / "sparse-occupancy.glb"
    )
    reporter.emit(
        "sparse_occupancy", "ready", progress=args.steps, total=args.steps,
        artifact_path=sparse_preview, preview_kind="mesh",
    )
    if _reached(target, "sparse_occupancy"):
        return

    reporter.emit("lr_shape_flow", "running", total=args.steps)
    active_flow["stage"] = "lr_shape_flow"
    lr_conditioner, lr_grid_override = _lr_conditioner(
        pipeline, args.lr_conditioning_resolution
    )
    cond_lr = pipeline.get_proj_cond_shape(
        lr_conditioner,
        [image],
        coords,
        grid_resolution_override=lr_grid_override,
        **camera,
    )
    lr_slat = pipeline.sample_shape_slat(
        cond_lr, pipeline.models["shape_slat_flow_model_512"], coords, sampler
    )
    del cond_lr
    torch.cuda.empty_cache()
    reporter.emit("lr_shape_flow", "ready", progress=args.steps, total=args.steps)
    if _reached(target, "lr_shape_flow"):
        return
    lr_meshes, _ = pipeline.decode_shape_slat(lr_slat, 512)
    lr_preview = _shape_preview(lr_meshes, artifact_root / "lr-shape.glb")
    reporter.emit(
        "lr_shape_latent", "ready", artifact_path=lr_preview,
        preview_kind="mesh",
    )
    if _reached(target, "lr_shape_latent"):
        return

    if pipeline.low_vram:
        pipeline.models["shape_slat_decoder"].to(pipeline.device)
        pipeline.models["shape_slat_decoder"].low_vram = True
    hr_coords = pipeline.models["shape_slat_decoder"].upsample(
        lr_slat, upsample_times=4
    )
    if pipeline.low_vram:
        pipeline.models["shape_slat_decoder"].cpu()
        pipeline.models["shape_slat_decoder"].low_vram = False
    actual_hr_resolution = hr_resolution
    while True:
        grid_res = actual_hr_resolution // 16
        quantized = torch.cat((
            hr_coords[:, :1],
            ((hr_coords[:, 1:] + 0.5) / 512 * (grid_res - 1)).round().int(),
        ), dim=1)
        hr_coords_unique = quantized.unique(dim=0)
        if len(hr_coords_unique) < args.max_num_tokens or actual_hr_resolution == 1024:
            break
        actual_hr_resolution -= 128
    del lr_meshes, hr_coords, quantized
    torch.cuda.empty_cache()
    coords_preview = _point_preview(
        hr_coords_unique,
        actual_hr_resolution // 16,
        artifact_root / "hr-coordinates.glb",
    )
    reporter.emit(
        "hr_coordinates", "ready", artifact_path=coords_preview,
        preview_kind="mesh",
    )
    if _reached(target, "hr_coordinates"):
        return

    reporter.emit("hr_shape_flow", "running", total=args.steps)
    active_flow["stage"] = "hr_shape_flow"
    cond_hr = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_shape_1024,
        [image],
        hr_coords_unique,
        grid_resolution_override=actual_hr_resolution // 16,
        **camera,
    )
    noise = sp.SparseTensor(
        feats=torch.randn(
            len(hr_coords_unique),
            pipeline.models["shape_slat_flow_model_1024"].in_channels,
            device=pipeline.device,
        ),
        coords=hr_coords_unique,
    )
    flow = pipeline.models["shape_slat_flow_model_1024"]
    if pipeline.low_vram:
        flow.to(pipeline.device)
    hr_slat = pipeline.shape_slat_sampler.sample(
        flow, noise, **cond_hr,
        **{**pipeline.shape_slat_sampler_params, **sampler},
        verbose=True,
        tqdm_desc=f"Sampling HR shape SLat ({actual_hr_resolution})",
    ).samples
    if pipeline.low_vram:
        flow.cpu()
    if args.checkpoint:
        _save_shape_checkpoint(
            Path(args.checkpoint),
            hr_slat,
            actual_hr_resolution,
            camera,
            preprocessed,
            args.model_path,
        )
    std = torch.tensor(pipeline.shape_slat_normalization["std"])[None].to(hr_slat.device)
    mean = torch.tensor(pipeline.shape_slat_normalization["mean"])[None].to(hr_slat.device)
    shape_slat = hr_slat * std + mean
    del cond_hr, noise, hr_slat, hr_coords_unique, lr_slat
    torch.cuda.empty_cache()
    reporter.emit("hr_shape_flow", "ready", progress=args.steps, total=args.steps)
    if _reached(target, "hr_shape_flow"):
        return
    shape_meshes, shape_subs = pipeline.decode_shape_slat(
        shape_slat, actual_hr_resolution
    )
    shape_preview = _shape_preview(
        shape_meshes, artifact_root / "hr-shape.glb"
    )
    reporter.emit(
        "hr_shape_latent", "ready", artifact_path=shape_preview,
        preview_kind="mesh",
    )
    if _reached(target, "hr_shape_latent"):
        return

    reporter.emit("texture_flow", "running", total=args.steps)
    active_flow["stage"] = "texture_flow"
    cond_tex = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image],
        shape_slat.coords,
        grid_resolution_override=actual_hr_resolution // 16,
        **camera,
    )
    tex_slat = pipeline.sample_tex_slat(
        cond_tex,
        pipeline.models["tex_slat_flow_model_1024"],
        shape_slat,
        tex_sampler,
    )
    del cond_tex
    torch.cuda.empty_cache()
    reporter.emit("texture_flow", "ready", progress=args.steps, total=args.steps)
    if _reached(target, "texture_flow"):
        return
    reporter.emit("texture_latent", "ready")
    if _reached(target, "texture_latent"):
        return

    tex_voxels = pipeline.decode_tex_slat(tex_slat, shape_subs)
    meshes = []
    from pixal3d.representations import MeshWithVoxel
    for mesh, voxels in zip(shape_meshes, tex_voxels):
        mesh.fill_holes()
        meshes.append(MeshWithVoxel(
            mesh.vertices,
            mesh.faces,
            origin=[-0.5, -0.5, -0.5],
            voxel_size=1 / actual_hr_resolution,
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
        attr_layout=pipeline.pbr_attr_layout,
        grid_size=actual_hr_resolution,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=args.decimation_target,
        texture_size=args.texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0.0,
        use_tqdm=True,
    )
    glb.apply_transform(np.asarray((
        (-1, 0, 0, 0),
        (0, 0, -1, 0),
        (0, -1, 0, 0),
        (0, 0, 0, 1),
    ), dtype=np.float64))
    glb.export(args.output, extension_webp=True)
    reporter.emit(
        "final_mesh", "ready", artifact_path=Path(args.output),
        preview_kind="mesh",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pixal3d-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--refine-checkpoint")
    parser.add_argument("--refine-mask")
    parser.add_argument("--refine-strength", type=float, default=0.35)
    parser.add_argument("--refine-steps", type=int, default=8)
    parser.add_argument("--refine-seed", type=int, default=123)
    parser.add_argument("--refine-rescale-t", type=float, default=3.0)
    parser.add_argument("--refine-guidance", type=float, default=7.5)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--target-stage", choices=STAGES, default="final_mesh")
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument(
        "--lr-conditioning-resolution",
        type=int,
        choices=(512, 1024),
        default=512,
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decimation-target", type=int, default=200_000)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-num-tokens", type=int, default=49152)
    parser.add_argument("--manual-fov", type=float, default=-1.0)
    parser.add_argument("--low_vram", action="store_true")
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
