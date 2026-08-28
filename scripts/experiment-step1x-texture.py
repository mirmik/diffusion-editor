#!/usr/bin/env python3
"""Run isolated Step1X-3D texture experiments on an existing mesh."""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import time
import types
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--render-only", action="store_true")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("/home/mirmik/soft/Step1X-3D"),
    )
    parser.add_argument("--model", default="stepfun-ai/Step1X-3D")
    return parser.parse_args()


def install_texture_only_geometry_stub(repo_root: Path) -> None:
    """Avoid importing Step1X geometry's training/data dependency graph.

    The texture pipeline only asks that graph for smart_load_model(), whose
    small download/path-resolution behavior is reproduced here.
    """
    geometry_root = repo_root / "step1x3d_geometry"
    package_paths = (
        ("step1x3d_geometry", geometry_root),
        ("step1x3d_geometry.models", geometry_root / "models"),
        ("step1x3d_geometry.models.pipelines", geometry_root / "models" / "pipelines"),
    )
    for name, path in package_paths:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module

    loader_module_name = "step1x3d_geometry.models.pipelines.pipeline_utils"
    loader_module = types.ModuleType(loader_module_name)

    def smart_load_model(model_path: str, subfolder: str = "") -> str:
        local = Path(model_path)
        candidate = local / subfolder if subfolder else local
        if candidate.exists():
            return str(candidate)

        from huggingface_hub import snapshot_download

        patterns = [f"{subfolder}/*"] if subfolder else None
        snapshot = Path(snapshot_download(repo_id=model_path, allow_patterns=patterns))
        return str(snapshot / subfolder if subfolder else snapshot)

    loader_module.smart_load_model = smart_load_model
    sys.modules[loader_module_name] = loader_module


def preload_cupy_cuda12() -> None:
    """Expose pip-installed CUDA 12 libraries to CuPy without global env changes."""
    site_packages = (
        Path(sys.prefix)
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
        / "nvidia"
    )
    libraries = (
        site_packages / "cuda_runtime/lib/libcudart.so.12",
        site_packages / "cuda_nvrtc/lib/libnvrtc-builtins.so.12.8",
        site_packages / "cuda_nvrtc/lib/libnvrtc.so.12",
    )
    for library in libraries:
        if not library.is_file():
            raise FileNotFoundError(f"Required CuPy CUDA library not found: {library}")
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def load_trimesh(path: Path):
    import trimesh

    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"Expected a triangle mesh, got {type(mesh).__name__}")
    return mesh


def mesh_summary(mesh) -> dict:
    import numpy as np

    return {
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
        "components": int(len(mesh.split(only_watertight=False))),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "bounds": np.asarray(mesh.bounds).tolist(),
    }


def save_step1x_controls(repo_root: Path, mesh, output_dir: Path) -> dict:
    import torch

    sys.path.insert(0, str(repo_root))
    from step1x3d_texture.utils import get_orthogonal_camera, tensor_to_image
    from step1x3d_texture.utils.render import (
        NVDiffRastContextWrapper,
        load_mesh,
        render,
    )

    cameras = get_orthogonal_camera(
        elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
        distance=[1.8] * 6,
        left=-0.55,
        right=0.55,
        bottom=-0.55,
        top=0.55,
        azimuth_deg=[-90, 0, 90, 180, 90, 90],
        device="cuda",
    )
    context = NVDiffRastContextWrapper(device="cuda", context_type="cuda")
    render_mesh, _ = load_mesh(mesh, rescale=True, device="cuda")
    rendered = render(
        context,
        render_mesh,
        cameras,
        height=768,
        width=768,
        render_attr=False,
        normal_background=0.0,
    )
    normal_images = tensor_to_image(
        (rendered.normal / 2 + 0.5).clamp(0, 1), batched=True
    )
    position_images = tensor_to_image((rendered.pos + 0.5).clamp(0, 1), batched=True)
    for index, image in enumerate(normal_images):
        image.save(output_dir / f"normal-{index}.png")
    for index, image in enumerate(position_images):
        image.save(output_dir / f"position-{index}.png")
    return {"views": len(normal_images), "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3}


def main() -> int:
    args = parse_args()
    if not args.mesh.is_file():
        raise SystemExit(f"Mesh not found: {args.mesh}")
    if not args.reference.is_file():
        raise SystemExit(f"Reference image not found: {args.reference}")
    if not args.repo_root.is_dir():
        raise SystemExit(f"Step1X repository not found: {args.repo_root}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    mesh = load_trimesh(args.mesh)
    manifest = {
        "backend": "Step1X-3D Texture",
        "mesh": str(args.mesh),
        "reference": str(args.reference),
        "repo_root": str(args.repo_root),
        "model": args.model,
        "seed": args.seed,
        "steps": args.steps,
        "vae_slicing": True,
        "vae_tiling": True,
        "mesh_summary": mesh_summary(mesh),
    }

    started = time.monotonic()
    print("[render] Step1X geometric controls", flush=True)
    manifest["control_render"] = save_step1x_controls(args.repo_root, mesh.copy(), args.output_dir)
    if args.render_only:
        manifest["total_seconds"] = time.monotonic() - started
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return 0

    import torch

    preload_cupy_cuda12()
    install_texture_only_geometry_stub(args.repo_root)
    from step1x3d_texture.pipelines.step1x_3d_texture_synthesis_pipeline import (
        Step1X3DTexturePipeline,
    )

    print(f"[load] {args.model}/Step1X-3D-Texture", flush=True)
    load_started = time.monotonic()
    pipeline = Step1X3DTexturePipeline.from_pretrained(
        args.model,
        subfolder="Step1X-3D-Texture",
    )
    pipeline.config.num_inference_steps = args.steps
    # Encoding six 768px control views as one VAE batch exceeds a 32 GiB GPU.
    # These are exact memory-saving execution modes; they do not reduce output
    # resolution or change the diffusion schedule.
    pipeline.ig2mv_pipe.enable_vae_slicing()
    pipeline.ig2mv_pipe.enable_vae_tiling()
    manifest["load_seconds"] = time.monotonic() - load_started

    original_run = pipeline.run_ig2mv_pipeline

    def saving_run(*run_args, **run_kwargs):
        result = original_run(*run_args, **run_kwargs)
        images, positions, normals, reference, *_ = result
        for index, image in enumerate(images):
            image.save(args.output_dir / f"generated-{index}.png")
        for index, image in enumerate(positions):
            image.save(args.output_dir / f"generated-position-{index}.png")
        for index, image in enumerate(normals):
            image.save(args.output_dir / f"generated-normal-{index}.png")
        reference.save(args.output_dir / "reference-preprocessed.png")
        return result

    pipeline.run_ig2mv_pipeline = saving_run
    torch.cuda.reset_peak_memory_stats()
    inference_started = time.monotonic()
    print(f"[texture] steps={args.steps} seed={args.seed}", flush=True)
    textured_mesh = pipeline(str(args.reference), mesh.copy(), remove_bg=False, seed=args.seed)
    textured_path = args.output_dir / "textured.glb"
    textured_mesh.export(textured_path)
    material = getattr(textured_mesh.visual, "material", None)
    atlas = getattr(material, "baseColorTexture", None)
    if atlas is None:
        atlas = getattr(material, "image", None)
    if atlas is None:
        exported = load_trimesh(textured_path)
        material = getattr(exported.visual, "material", None)
        atlas = getattr(material, "baseColorTexture", None)
    if atlas is not None:
        atlas.save(args.output_dir / "texture-atlas.png")
    manifest["inference_seconds"] = time.monotonic() - inference_started
    manifest["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 1024**3
    manifest["total_seconds"] = time.monotonic() - started
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
