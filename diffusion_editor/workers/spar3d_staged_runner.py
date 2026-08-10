"""Editor-owned staged entry point for an installed SPAR3D runtime."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import random
import sys


STAGES = ("source_image", "point_cloud", "final_mesh")


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


def _prepare_batch(model, image, device):
    from spar3d.utils import create_intrinsic_from_fov_rad, default_cond_c2w

    mask_cond, rgb_cond = model.prepare_image(image)
    c2w = default_cond_c2w(model.cfg.default_distance).to(device)
    intrinsic, intrinsic_normed = create_intrinsic_from_fov_rad(
        model.cfg.default_fovy_rad,
        model.cfg.cond_image_size,
        model.cfg.cond_image_size,
    )
    return {
        "rgb_cond": rgb_cond.unsqueeze(0),
        "mask_cond": mask_cond.unsqueeze(0),
        "c2w_cond": c2w.view(1, 1, 4, 4),
        "intrinsic_cond": intrinsic.to(device).view(1, 1, 3, 3),
        "intrinsic_normed_cond": intrinsic_normed.to(device).view(1, 1, 3, 3),
    }


def run(args) -> None:
    sys.path.insert(0, str(Path(args.spar3d_root).resolve()))
    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    from transparent_background import Remover

    from spar3d.system import SPAR3D
    from spar3d.utils import (
        foreground_crop,
        get_device,
        normalize_pc_bbox,
        remove_background,
    )

    torch.set_grad_enabled(False)
    reporter = StageReporter(Path(args.events))
    artifact_root = Path(args.output).resolve().parent
    device = get_device()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = SPAR3D.from_pretrained(
        args.model_path,
        config_name="config.yaml",
        weight_name="model.safetensors",
        low_vram_mode=args.low_vram,
    )
    model.to(device)
    model.eval()
    remover = Remover(device=device)
    image = remove_background(
        Image.open(args.image).convert("RGBA"), remover
    )
    image = foreground_crop(image, 1.3)
    image.save(artifact_root / "preprocessed.png")
    batch = _prepare_batch(model, image, device)

    point_steps = int(model.cfg.inference_time_steps)
    reporter.emit("point_cloud", "running", total=point_steps)
    autocast = (
        torch.autocast(device_type=device, dtype=torch.bfloat16)
        if "cuda" in device else nullcontext()
    )
    with autocast:
        cond_tokens = model.forward_pdiff_cond(batch)
        sample_iter = model.sampler.sample_batch_progressive(
            1,
            cond_tokens,
            guidance_scale=args.guidance_scale,
            device=device,
        )
        samples = None
        progress = 0
        for progress, sample in enumerate(sample_iter, 1):
            samples = sample["xstart"]
            reporter.emit(
                "point_cloud", "running", progress=progress, total=point_steps
            )
        if samples is None:
            raise RuntimeError("SPAR3D point sampler produced no samples")
        batch["pc_cond"] = normalize_pc_bbox(
            samples.permute(0, 2, 1).float()
        )

        xyz = batch["pc_cond"][0, :, :3].detach().cpu().numpy()
        colors = (
            batch["pc_cond"][0, :, 3:6].clamp(0, 1).detach().cpu().numpy()
            * 255
        ).astype(np.uint8)
        conditioning_path = artifact_root / "point-conditioning.ply"
        trimesh.PointCloud(vertices=xyz, colors=colors).export(
            conditioning_path
        )
        # ``SPAR3D.generate_mesh`` applies the first two transforms to its GLB.
        # The editor's GLB importer then converts glTF Y-up coordinates to its
        # Z-up viewport. Apply the complete display transform to the PLY too,
        # so switching between points and the mesh preserves orientation.
        rotation_x = trimesh.transformations.rotation_matrix(
            np.radians(-90), [1, 0, 0]
        )
        rotation_y = trimesh.transformations.rotation_matrix(
            np.radians(90), [0, 1, 0]
        )
        gltf_to_z_up = np.asarray(
            [
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.float64,
        )
        preview = trimesh.PointCloud(vertices=xyz, colors=colors)
        preview.apply_transform(gltf_to_z_up @ rotation_y @ rotation_x)
        point_path = artifact_root / "points.ply"
        preview.export(point_path)
        reporter.emit(
            "point_cloud",
            "ready",
            progress=progress,
            total=point_steps,
            artifact_path=point_path,
            preview_kind="points",
        )
        if args.target_stage == "point_cloud":
            return

        reporter.emit("final_mesh", "running")
        meshes, _metadata = model.generate_mesh(
            batch,
            args.texture_size,
            remesh="none",
            vertex_count=-1,
            estimate_illumination=False,
        )
        if not meshes:
            raise RuntimeError("SPAR3D mesh decoder produced no meshes")
        meshes[0].export(args.output, include_normals=True)
    reporter.emit(
        "final_mesh",
        "ready",
        artifact_path=Path(args.output),
        preview_kind="mesh",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spar3d-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--target-stage", choices=("point_cloud", "final_mesh"), required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--low-vram", action="store_true")
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
