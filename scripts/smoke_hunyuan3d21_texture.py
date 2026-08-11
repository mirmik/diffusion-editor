#!/usr/bin/env python3
"""Texture an existing mesh with the local Hunyuan3D 2.1 PBR pipeline."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


DEFAULT_COMFY_ROOT = Path("/home/mirmik/soft/ComfyUI")
DEFAULT_NODE_ROOT = DEFAULT_COMFY_ROOT / "custom_nodes/ComfyUI-Hunyuan3d-2-1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("output", type=Path, help="Output GLB")
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--node-root", type=Path, default=DEFAULT_NODE_ROOT)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=12646)
    parser.add_argument("--view-size", type=int, default=512)
    parser.add_argument("--texture-size", type=int, default=1024)
    parser.add_argument("--ortho-scale", type=float, default=1.0)
    parser.add_argument("--remove-background", action="store_true")
    parser.add_argument(
        "--gltf-transform",
        type=Path,
        default=Path(shutil.which("gltf-transform") or "gltf-transform"),
        help="Used to embed data-URI textures as GLB bufferViews for Termin",
    )
    return parser


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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_path = _require_file(args.image, "input image")
    mesh_path = _require_file(args.mesh, "input mesh")
    output_path = args.output.expanduser().resolve()
    if output_path.suffix.lower() != ".glb":
        raise ValueError("output must have a .glb suffix")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    comfy_root = args.comfy_root.expanduser().resolve()
    custom_nodes_root = args.node_root.expanduser().resolve().parent
    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(custom_nodes_root))
    os.chdir(comfy_root)

    import trimesh
    from PIL import Image

    plugin = args.node_root.name
    texture_module = importlib.import_module(
        f"{plugin}.hy3dpaint.textureGenPipeline"
    )
    Hunyuan3DPaintConfig = texture_module.Hunyuan3DPaintConfig
    Hunyuan3DPaintPipeline = texture_module.Hunyuan3DPaintPipeline

    image = Image.open(image_path).convert("RGBA")
    if args.remove_background:
        print("[hunyuan3d21-texture] removing background", flush=True)
        rembg_module = importlib.import_module(
            f"{plugin}.hy3dshape.hy3dshape.rembg"
        )
        image = rembg_module.BackgroundRemover()(image)

    mesh = trimesh.load(mesh_path, force="mesh")
    camera_azimuths = [0, 90, 180, 270, 0, 180]
    camera_elevations = [0, 0, 0, 0, 90, -90]
    view_weights = [1.0, 0.1, 0.5, 0.1, 0.05, 0.05]
    config = Hunyuan3DPaintConfig(
        args.view_size,
        camera_azimuths,
        camera_elevations,
        view_weights,
        args.ortho_scale,
        args.texture_size,
    )
    pipeline = Hunyuan3DPaintPipeline(config)

    print("[hunyuan3d21-texture] generating PBR multiviews", flush=True)
    albedo_views, mr_views, normal_views, position_views = pipeline(
        mesh=mesh,
        image_path=image,
        output_mesh_path=str(output_path.with_suffix(".obj")),
        num_steps=args.steps,
        guidance_scale=args.guidance,
        unwrap=True,
        seed=args.seed % (2**32),
    )
    preview_dir = output_path.parent / f"{output_path.stem}-stages"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for label, views in (
        ("albedo", albedo_views),
        ("mr", mr_views),
        ("normal", normal_views),
        ("position", position_views),
    ):
        for index, view in enumerate(views):
            _save_image(view, preview_dir / f"{label}_{index}.png")

    print("[hunyuan3d21-texture] baking multiviews", flush=True)
    albedo, albedo_mask, mr, mr_mask = pipeline.bake_from_multiview(
        albedo_views,
        mr_views,
        camera_elevations,
        camera_azimuths,
        view_weights,
    )
    _save_image(albedo, preview_dir / "albedo_baked.png")
    _save_image(albedo_mask, preview_dir / "albedo_mask.png")
    _save_image(mr, preview_dir / "mr_baked.png")
    _save_image(mr_mask, preview_dir / "mr_mask.png")

    print("[hunyuan3d21-texture] inpainting texture seams", flush=True)
    albedo, mr = pipeline.inpaint(
        albedo,
        albedo_mask,
        mr,
        mr_mask,
        True,
        "NS",
    )
    _save_image(albedo, preview_dir / "albedo_final.png")
    _save_image(mr, preview_dir / "mr_final.png")
    pipeline.set_texture_albedo(albedo)
    pipeline.set_texture_mr(mr)
    generated_path = Path(pipeline.save_mesh(str(output_path.with_suffix(".obj"))))
    gltf_transform = args.gltf_transform.expanduser()
    if not gltf_transform.is_absolute():
        resolved = shutil.which(str(gltf_transform))
        if resolved is None:
            raise FileNotFoundError(f"gltf-transform is not available: {gltf_transform}")
        gltf_transform = Path(resolved)
    with tempfile.TemporaryDirectory(
        prefix="hunyuan3d21-glb-", dir=output_path.parent
    ) as temporary:
        normalized_path = Path(temporary) / output_path.name
        subprocess.run(
            [str(gltf_transform), "copy", str(generated_path), str(normalized_path)],
            check=True,
        )
        shutil.copy2(normalized_path, output_path)
    pipeline.clean_memory()
    print(f"[hunyuan3d21-texture] exported {output_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
