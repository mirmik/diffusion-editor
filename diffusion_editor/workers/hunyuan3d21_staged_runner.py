"""Editor-owned staged entry point for Hunyuan3D 2.1 shape and PBR paint."""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


STAGES = (
    "source_image",
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


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def _save_image(value, path: Path) -> None:
    import numpy as np
    import torch
    from PIL import Image

    if isinstance(value, Image.Image):
        image = value
    else:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        array = np.asarray(value).squeeze()
        if array.dtype != np.uint8:
            if array.size and float(np.nanmax(array)) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        image = Image.fromarray(array)
    image.save(path)


def _load_shape_pipeline(args, offload_device):
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    config_path = Path(args.node_root) / "configs/dit_config_2_1.yaml"
    config_text = _require_file(config_path, "DiT config").read_text(
        encoding="utf-8"
    ).replace(".hy3dshape.hy3dshape.", "hy3dshape.")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", encoding="utf-8"
    ) as config:
        config.write(config_text)
        config.flush()
        return Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            config_path=config.name,
            ckpt_path=str(Path(args.dit).resolve()),
            offload_device=offload_device,
            attention_mode="sdpa",
        )


def _decode_shape(args, latents, device):
    import numpy as np
    import torch
    import trimesh
    from comfy.utils import load_torch_file
    from hy3dshape.models.autoencoders import ShapeVAE

    vae = ShapeVAE(
        num_latents=4096,
        embed_dim=64,
        num_freqs=8,
        include_pi=False,
        heads=16,
        width=1024,
        num_encoder_layers=8,
        num_decoder_layers=16,
        qkv_bias=False,
        qk_norm=True,
        scale_factor=1.0039506158752403,
        geo_decoder_mlp_expand_ratio=4,
        geo_decoder_downsample_ratio=1,
        geo_decoder_ln_post=True,
        point_feats=4,
        pc_size=81920,
        pc_sharpedge_size=0,
    )
    vae.load_state_dict(load_torch_file(str(Path(args.vae).resolve())))
    vae.eval().to(torch.float16).to(device)
    # FlashVDM is deliberately disabled.  The installed 2.1 checkpoint produced
    # non-finite vertices with it, while the hierarchical MC decoder is stable.
    vae.enable_flashvdm_decoder(enabled=False, mc_algo="mc")
    with torch.inference_mode():
        decoded = vae.decode(latents.to(device))
        result = vae.latents2mesh(
            decoded,
            output_type="trimesh",
            bounds=1.01,
            mc_level=0.0,
            num_chunks=8000,
            octree_resolution=args.octree_resolution,
            mc_algo="mc",
            enable_pbar=True,
        )[0]
    if result is None:
        raise RuntimeError(
            "Hunyuan3D 2.1 produced no surface; increase shape steps or change seed"
        )
    vertices = np.asarray(result.mesh_v)
    faces = np.asarray(result.mesh_f)[:, ::-1]
    if not vertices.size or not faces.size or not np.isfinite(vertices).all():
        raise RuntimeError("Hunyuan3D 2.1 decoded an empty or non-finite mesh")
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    del vae, decoded, result
    return mesh


def _normalize_glb(gltf_transform: str, generated: Path, output: Path) -> None:
    executable = Path(gltf_transform).expanduser()
    if not executable.is_file():
        resolved = shutil.which(gltf_transform)
        if resolved is None:
            raise FileNotFoundError(f"gltf-transform is unavailable: {gltf_transform}")
        executable = Path(resolved)
    with tempfile.TemporaryDirectory(
        prefix="hunyuan3d21-glb-", dir=output.parent
    ) as temporary:
        normalized = Path(temporary) / output.name
        subprocess.run(
            [str(executable), "copy", str(generated), str(normalized)],
            check=True,
        )
        shutil.copy2(normalized, output)


def run(args) -> None:
    comfy_root = Path(args.comfy_root).resolve()
    node_root = Path(args.node_root).resolve()
    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(node_root / "hy3dshape"))
    sys.path.insert(0, str(node_root.parent))
    os.chdir(comfy_root)

    import numpy as np
    import torch
    import trimesh
    from PIL import Image
    import comfy.model_management as mm
    from hy3dshape.postprocessors import FaceReducer
    from hy3dshape.rembg import BackgroundRemover

    if not torch.cuda.is_available():
        raise RuntimeError("Hunyuan3D 2.1 requires a CUDA GPU")
    torch.set_grad_enabled(False)
    reporter = StageReporter(Path(args.events))
    output_path = Path(args.output).resolve()
    artifact_root = output_path.parent
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = Image.open(args.image).convert("RGBA")
    source = BackgroundRemover()(source)
    preprocessed_path = artifact_root / "preprocessed.png"
    source.save(preprocessed_path)

    device = mm.get_torch_device()
    reporter.emit("hr_shape_flow", "running", total=args.shape_steps)
    pipeline = _load_shape_pipeline(args, mm.unet_offload_device())

    def report_shape_step(step_index, _timestep, _outputs) -> None:
        reporter.emit(
            "hr_shape_flow",
            "running",
            progress=min(int(step_index) + 1, args.shape_steps),
            total=args.shape_steps,
        )

    latents = pipeline(
        image=source,
        num_inference_steps=args.shape_steps,
        guidance_scale=args.shape_guidance,
        generator=torch.manual_seed(args.seed % (2**32)),
        callback=report_shape_step,
        callback_steps=1,
    )
    latent_path = Path(args.latent_output).resolve()
    torch.save(latents.detach().cpu(), latent_path)
    reporter.emit(
        "hr_shape_flow", "ready",
        progress=args.shape_steps, total=args.shape_steps,
    )
    del pipeline
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
    if _reached(args.target_stage, "hr_shape_flow"):
        return

    reporter.emit("hr_shape_latent", "running")
    mesh = _decode_shape(args, latents, device)
    raw_mesh_path = artifact_root / "raw-mesh.glb"
    mesh.export(raw_mesh_path)
    reporter.emit(
        "hr_shape_latent", "ready",
        artifact_path=raw_mesh_path, preview_kind="mesh",
    )
    del latents
    mm.soft_empty_cache()
    torch.cuda.empty_cache()
    gc.collect()
    if _reached(args.target_stage, "hr_shape_latent"):
        return

    if len(mesh.faces) > args.decimation_target:
        mesh = FaceReducer()(mesh, max_facenum=args.decimation_target)

    plugin = node_root.name
    texture_module = importlib.import_module(
        f"{plugin}.hy3dpaint.textureGenPipeline"
    )
    config = texture_module.Hunyuan3DPaintConfig(
        512,
        [0, 90, 180, 270, 0, 180],
        [0, 0, 0, 0, 90, -90],
        [1.0, 0.1, 0.5, 0.1, 0.05, 0.05],
        1.0,
        args.texture_size,
    )
    paint = texture_module.Hunyuan3DPaintPipeline(config)
    reporter.emit("texture_flow", "running", total=args.texture_steps)
    albedo_views, mr_views, normal_views, position_views = paint(
        mesh=mesh,
        image_path=source,
        output_mesh_path=str(output_path.with_suffix(".obj")),
        num_steps=args.texture_steps,
        guidance_scale=args.texture_guidance,
        unwrap=True,
        seed=args.seed % (2**32),
    )
    stages_dir = artifact_root / "texture-stages"
    stages_dir.mkdir(parents=True, exist_ok=True)
    for label, views in (
        ("albedo", albedo_views),
        ("mr", mr_views),
        ("normal", normal_views),
        ("position", position_views),
    ):
        for index, view in enumerate(views):
            _save_image(view, stages_dir / f"{label}_{index}.png")
    reporter.emit(
        "texture_flow", "ready",
        progress=args.texture_steps, total=args.texture_steps,
    )
    if _reached(args.target_stage, "texture_flow"):
        paint.clean_memory()
        return

    reporter.emit("texture_latent", "running")
    azimuths = [0, 90, 180, 270, 0, 180]
    elevations = [0, 0, 0, 0, 90, -90]
    weights = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05]
    albedo, albedo_mask, mr, mr_mask = paint.bake_from_multiview(
        albedo_views, mr_views, elevations, azimuths, weights
    )
    for label, value in (
        ("albedo_baked", albedo), ("albedo_mask", albedo_mask),
        ("mr_baked", mr), ("mr_mask", mr_mask),
    ):
        _save_image(value, stages_dir / f"{label}.png")
    albedo, mr = paint.inpaint(albedo, albedo_mask, mr, mr_mask, True, "NS")
    albedo_final = stages_dir / "albedo_final.png"
    _save_image(albedo, albedo_final)
    _save_image(mr, stages_dir / "mr_final.png")
    paint.set_texture_albedo(albedo)
    paint.set_texture_mr(mr)
    generated = Path(paint.save_mesh(str(output_path.with_suffix(".obj"))))
    textured_preview = artifact_root / "textured-preview.glb"
    _normalize_glb(args.gltf_transform, generated, textured_preview)
    reporter.emit(
        "texture_latent", "ready",
        artifact_path=textured_preview, preview_kind="mesh",
    )
    if _reached(args.target_stage, "texture_latent"):
        paint.clean_memory()
        return

    shutil.copy2(textured_preview, output_path)
    paint.clean_memory()
    reporter.emit(
        "final_mesh", "ready", artifact_path=output_path, preview_kind="mesh"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfy-root", required=True)
    parser.add_argument("--node-root", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--dit", required=True)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--latent-output", required=True)
    parser.add_argument("--target-stage", choices=STAGES, required=True)
    parser.add_argument("--shape-steps", type=int, default=30)
    parser.add_argument("--shape-guidance", type=float, default=5.0)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--texture-steps", type=int, default=10)
    parser.add_argument("--texture-guidance", type=float, default=3.0)
    parser.add_argument("--texture-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decimation-target", type=int, default=200_000)
    parser.add_argument("--gltf-transform", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
