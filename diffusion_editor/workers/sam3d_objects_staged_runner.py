"""Editor-owned staged entry point for SAM 3D Objects."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import types


STAGES = (
    "source_image",
    "point_cloud",
    "sparse_occupancy",
    "hr_shape_flow",
    "hr_shape_latent",
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


def _stop_at(target: str, stage: str) -> bool:
    return STAGES.index(target) <= STAGES.index(stage)


def _install_kaolin_shape_check_stub() -> None:
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


def _force_gsplat_texture_renderer() -> None:
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


def _save_points(path: Path, points, colors=None) -> int:
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
            ((rgb * 255).astype(np.uint8), np.full((len(rgb), 1), 255, np.uint8)),
            axis=1,
        )
    trimesh.PointCloud(xyz, colors=rgba).export(path)
    return len(xyz)


def _export_mesh_for_gltf(mesh, path: Path):
    import numpy as np
    import trimesh

    # SAM's decoded space is Z-up. Store glTF as Y-up so Termin's importer
    # restores the same Z-up pose used by the PLY previews.
    vertices = mesh.vertices.detach().float().cpu().numpy()
    vertices = vertices @ np.asarray(
        [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
    )
    faces = mesh.faces.detach().cpu().numpy()
    preview = trimesh.Trimesh(vertices, faces, process=False)
    preview.export(path)
    return preview


def _prepare_source(path: Path, output: Path):
    import numpy as np
    from PIL import Image

    source = Image.open(path).convert("RGBA")
    alpha = np.asarray(source.getchannel("A"))
    if np.all(alpha == 255):
        from rembg import remove

        source = remove(source).convert("RGBA")
    alpha = np.asarray(source.getchannel("A"))
    if not np.any(alpha > 127):
        raise RuntimeError("SAM 3D Objects received an empty foreground mask")
    source.save(output)
    return np.asarray(source)


def run(args) -> None:
    root = Path(args.sam3d_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    output_path = Path(args.output).resolve()
    artifact_root = output_path.parent
    artifact_root.mkdir(parents=True, exist_ok=True)
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
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise RuntimeError("SAM 3D Objects requires a CUDA GPU")
    torch.set_grad_enabled(False)
    reporter = StageReporter(Path(args.events))
    rgba = _prepare_source(Path(args.image), artifact_root / "conditioning.png")

    config = OmegaConf.load(config_path)
    config.rendering_engine = "nvdiffrast"
    config.compile_model = False
    config.workspace_dir = str(config_path.parent)
    config.ss_inference_steps = args.sparse_steps
    config.slat_inference_steps = args.slat_steps
    config.ss_cfg_strength = args.sparse_guidance
    config.slat_cfg_strength = args.slat_guidance
    pipeline = instantiate(config)
    _force_gsplat_texture_renderer()

    with pipeline.device:
        reporter.emit("point_cloud", "running")
        pointmap_dict = pipeline.compute_pointmap(rgba, None)
        pointmap = pointmap_dict["pointmap"]
        point_path = artifact_root / "pointmap.ply"
        _save_points(
            point_path,
            pointmap.permute(1, 2, 0),
            pointmap_dict["pts_color"].permute(1, 2, 0),
        )
        reporter.emit(
            "point_cloud", "ready",
            artifact_path=point_path, preview_kind="points",
        )
        if _stop_at(args.target_stage, "point_cloud"):
            return

        ss_input = pipeline.preprocess_image(
            rgba, pipeline.ss_preprocessor, pointmap=pointmap
        )
        slat_input = pipeline.preprocess_image(rgba, pipeline.slat_preprocessor)
        torch.manual_seed(args.seed)
        reporter.emit(
            "sparse_occupancy", "running", total=args.sparse_steps
        )
        sparse = pipeline.sample_sparse_structure(
            ss_input, inference_steps=args.sparse_steps
        )
        sparse.update(pipeline.pose_decoder(
            sparse,
            scene_scale=ss_input.get("pointmap_scale"),
            scene_shift=ss_input.get("pointmap_shift"),
        ))
        sparse["scale"] = sparse["scale"] * sparse["downsample_factor"]
        coords = sparse["coords"]
        sparse_path = artifact_root / "sparse-occupancy.ply"
        _save_points(sparse_path, coords[:, 1:].float() / 64.0 - 0.5)
        torch.save(
            {
                key: value.detach().cpu()
                for key, value in sparse.items()
                if isinstance(value, torch.Tensor)
            },
            artifact_root / "sparse-structure.pt",
        )
        reporter.emit(
            "sparse_occupancy", "ready",
            progress=args.sparse_steps, total=args.sparse_steps,
            artifact_path=sparse_path, preview_kind="points",
        )
        if _stop_at(args.target_stage, "sparse_occupancy"):
            return

        reporter.emit("hr_shape_flow", "running", total=args.slat_steps)
        slat = pipeline.sample_slat(
            slat_input, coords, inference_steps=args.slat_steps
        )
        slat_path = artifact_root / "structured-latent.pt"
        torch.save(
            {
                "coords": slat.coords.detach().cpu(),
                "feats": slat.feats.detach().cpu(),
            },
            slat_path,
        )
        reporter.emit(
            "hr_shape_flow", "ready",
            progress=args.slat_steps, total=args.slat_steps,
        )
        if _stop_at(args.target_stage, "hr_shape_flow"):
            return

        reporter.emit("hr_shape_latent", "running")
        decoded = pipeline.decode_slat(slat, ["mesh", "gaussian"])
        raw_path = artifact_root / "raw-mesh.glb"
        _export_mesh_for_gltf(decoded["mesh"][0], raw_path)
        reporter.emit(
            "hr_shape_latent", "ready",
            artifact_path=raw_path, preview_kind="mesh",
        )
        if _stop_at(args.target_stage, "hr_shape_latent"):
            return

        reporter.emit("texture_latent", "running")
        gaussian = decoded["gaussian"][0]
        gaussian_path = artifact_root / "gaussian-splat.ply"
        gaussian.save_ply(str(gaussian_path))
        # A conventional colored PLY remains useful until Termin grows a
        # native anisotropic Gaussian-splat material.
        dc = gaussian.get_features[:, 0, :]
        colors = torch.clamp(dc * 0.28209479177387814 + 0.5, 0, 1)
        gaussian_preview = artifact_root / "gaussian-preview.ply"
        _save_points(gaussian_preview, gaussian.get_xyz, colors)
        reporter.emit(
            "texture_latent", "ready",
            artifact_path=gaussian_preview, preview_kind="points",
        )
        if _stop_at(args.target_stage, "texture_latent"):
            return

        reporter.emit("final_mesh", "running")
        from sam3d_objects.model.backbone.tdfy_dit.utils import (
            postprocessing_utils,
        )

        final = postprocessing_utils.to_glb(
            gaussian,
            decoded["mesh"][0],
            simplify=args.simplify,
            texture_size=args.texture_size,
            verbose=False,
            with_mesh_postprocess=True,
            with_texture_baking=True,
            use_vertex_color=False,
            rendering_engine="nvdiffrast",
        )
        final.export(output_path)
        reporter.emit(
            "final_mesh", "ready",
            artifact_path=output_path, preview_kind="mesh",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam3d-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--target-stage", choices=STAGES, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sparse-steps", type=int, default=25)
    parser.add_argument("--slat-steps", type=int, default=25)
    parser.add_argument("--sparse-guidance", type=float, default=7.0)
    parser.add_argument("--slat-guidance", type=float, default=1.0)
    parser.add_argument("--simplify", type=float, default=0.95)
    parser.add_argument("--texture-size", type=int, default=1024)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
