"""Editor-owned staged entry point for an installed Hi3DGen runtime."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


STAGES = (
    "source_image",
    "normal_map",
    "sparse_occupancy",
    "hr_shape_flow",
    "hr_shape_latent",
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
    mesh.export(path)
    return path


def run(args) -> None:
    sys.path.insert(0, str(Path(args.hi3dgen_root).resolve()))
    import numpy as np
    import torch
    from PIL import Image

    import hi3dgen.pipelines.hi3dgen as hi3dgen_module
    from hi3dgen.pipelines.samplers.flow_euler import FlowEulerSampler

    # Stable3DGen c29f668 forgot this import in hi3dgen.py.
    hi3dgen_module.os = os
    torch.set_grad_enabled(False)
    reporter = StageReporter(Path(args.events))
    artifact_root = Path(args.output).resolve().parent
    active_flow = {"stage": ""}

    def tracked_sample(
        sampler,
        model,
        noise,
        cond=None,
        steps=50,
        rescale_t=1.0,
        verbose=True,
        **kwargs,
    ):
        del verbose
        sample = noise
        times = np.linspace(1, 0, steps + 1)
        times = rescale_t * times / (1 + (rescale_t - 1) * times)
        result = {"samples": None, "pred_x_t": [], "pred_x_0": []}
        pairs = list(zip(times[:-1], times[1:]))
        for index, (current, previous) in enumerate(pairs, 1):
            out = sampler.sample_once(
                model, sample, float(current), float(previous), cond, **kwargs
            )
            sample = out["pred_x_prev"]
            result["pred_x_t"].append(out["pred_x_prev"])
            result["pred_x_0"].append(out["pred_x_0"])
            reporter.emit(
                active_flow["stage"],
                "running",
                progress=index,
                total=len(pairs),
            )
        result["samples"] = sample
        return result

    FlowEulerSampler.sample = tracked_sample
    pipeline = hi3dgen_module.Hi3DGenPipeline.from_pretrained(args.model_path)
    pipeline.cuda()

    source = Image.open(args.image)
    reporter.emit("normal_map", "running")
    preprocessed = pipeline.preprocess_image(source, resolution=1024)
    preprocessed.save(artifact_root / "preprocessed.png")
    if getattr(pipeline, "birefnet_model", None) is not None:
        del pipeline.birefnet_model
        torch.cuda.empty_cache()

    normal_predictor = torch.hub.load(
        str(Path(args.stable_normal_root).resolve()),
        "StableNormal_turbo",
        source="local",
        yoso_version="yoso-normal-v1-8-1",
        local_cache_dir=str(Path(args.hi3dgen_root).resolve() / "weights"),
    )
    normal = normal_predictor(
        preprocessed,
        resolution=args.normal_resolution,
        match_input_resolution=True,
        data_type="object",
    )
    normal_path = artifact_root / "normal.png"
    normal.save(normal_path)
    reporter.emit(
        "normal_map", "ready", artifact_path=normal_path, preview_kind="image"
    )
    if _reached(args.target_stage, "normal_map"):
        return

    del normal_predictor
    torch.cuda.empty_cache()
    cond = pipeline.get_cond([normal])
    torch.manual_seed(args.seed)

    active_flow["stage"] = "sparse_occupancy"
    reporter.emit("sparse_occupancy", "running", total=args.sparse_steps)
    coords = pipeline.sample_sparse_structure(
        cond,
        sampler_params={
            "steps": args.sparse_steps,
            "cfg_strength": args.guidance_scale,
        },
    )
    sparse_preview = _occupancy_preview(
        coords,
        pipeline.models["sparse_structure_flow_model"].resolution,
        artifact_root / "sparse-occupancy.glb",
    )
    reporter.emit(
        "sparse_occupancy",
        "ready",
        progress=args.sparse_steps,
        total=args.sparse_steps,
        artifact_path=sparse_preview,
        preview_kind="mesh",
    )
    if _reached(args.target_stage, "sparse_occupancy"):
        return

    active_flow["stage"] = "hr_shape_flow"
    reporter.emit("hr_shape_flow", "running", total=args.slat_steps)
    slat = pipeline.sample_slat(
        cond,
        coords,
        sampler_params={
            "steps": args.slat_steps,
            "cfg_strength": args.guidance_scale,
        },
    )
    reporter.emit(
        "hr_shape_flow",
        "ready",
        progress=args.slat_steps,
        total=args.slat_steps,
    )
    if _reached(args.target_stage, "hr_shape_flow"):
        return

    mesh = pipeline.decode_slat(slat, ["mesh"])["mesh"][0]
    geometry = mesh.to_trimesh(transform_pose=True)
    shape_path = artifact_root / "hr-shape.glb"
    geometry.export(shape_path)
    reporter.emit(
        "hr_shape_latent",
        "ready",
        artifact_path=shape_path,
        preview_kind="mesh",
    )
    if _reached(args.target_stage, "hr_shape_latent"):
        return

    if len(geometry.faces) > args.decimation_target:
        geometry = geometry.simplify_quadric_decimation(
            face_count=args.decimation_target
        )
    geometry.export(args.output)
    reporter.emit(
        "final_mesh",
        "ready",
        artifact_path=Path(args.output),
        preview_kind="mesh",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hi3dgen-root", required=True)
    parser.add_argument("--stable-normal-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--target-stage", choices=STAGES, required=True)
    parser.add_argument("--sparse-steps", type=int, default=12)
    parser.add_argument("--slat-steps", type=int, default=6)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--normal-resolution", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decimation-target", type=int, default=200_000)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
