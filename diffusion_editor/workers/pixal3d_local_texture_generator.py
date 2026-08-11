"""Generate a textured Pixal3D GLB from saved local-geometry stages.

The three heavy phases run as separate invocations so model weights and sparse
activations cannot accumulate in one Python process.  The ``all`` command
orchestrates those editor-owned phase commands.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def _prepare_imports(pixal3d_root: str) -> None:
    os.environ.setdefault("ATTN_BACKEND", "sdpa")
    os.environ.setdefault("SPARSE_ATTN_BACKEND", "sdpa")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    sys.path.insert(0, str(Path(pixal3d_root).resolve()))


def _generate(args) -> None:
    _prepare_imports(args.pixal3d_root)
    import torch
    from PIL import Image
    from inference import init_pipeline
    from pixal3d.modules.sparse import SparseTensor

    saved = torch.load(args.shape_latent, map_location="cpu", weights_only=False)
    resolution = int(saved["resolution"])
    shape = SparseTensor(
        feats=saved["feats"].cuda(), coords=saved["coords"].cuda()
    )
    image = Image.open(args.image).convert("RGB")
    pipeline = init_pipeline(args.model_path, low_vram=True)
    condition = pipeline.get_proj_cond_shape(
        pipeline.image_cond_model_tex_1024,
        [image],
        shape.coords,
        camera_angle_x=args.camera_angle_x,
        distance=args.camera_distance,
        mesh_scale=1.0,
        grid_resolution_override=resolution // 16,
    )
    torch.manual_seed(args.seed)
    texture = pipeline.sample_tex_slat(
        condition,
        pipeline.models["tex_slat_flow_model_1024"],
        shape,
        {
            "steps": args.steps,
            "guidance_strength": 1.0,
            "guidance_rescale": 0.0,
            "rescale_t": 3.0,
        },
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "coords": texture.coords.cpu(),
        "feats": texture.feats.cpu(),
        "resolution": resolution,
    }, args.output)
    print(f"[Saved] {args.output}: {len(texture.coords)} texture tokens", flush=True)


def _decode(args) -> None:
    _prepare_imports(args.pixal3d_root)
    import torch
    import torch.nn.functional as functional
    from pixal3d.modules.sparse import SparseTensor
    from pixal3d.models.sc_vaes.sparse_unet_vae import SparseUnetVaeDecoder
    from pixal3d.pipelines import Pixal3DImageTo3DPipeline

    def run_module(module, value):
        module.cuda()
        result = module(value)
        module.cpu()
        torch.cuda.empty_cache()
        return result

    def guide_to_cuda(saved):
        return SparseTensor(
            feats=saved["feats"].cuda(),
            coords=saved["coords"].cuda(),
            shape=saved["shape"],
            scale=saved["scale"],
        )

    def streaming_forward(self, value, guide_subs=None, return_subs=False):
        if guide_subs is None:
            raise ValueError("texture decode requires shape subdivisions")
        hidden = run_module(self.from_latent, value).type(self.dtype)
        for stage_index, blocks in enumerate(self.blocks):
            for block_index, block in enumerate(blocks):
                block.cuda()
                is_upsample = (
                    stage_index < len(self.blocks) - 1
                    and block_index == len(blocks) - 1
                )
                if is_upsample:
                    guide = guide_to_cuda(guide_subs[stage_index])
                    hidden = block(hidden, subdiv=guide)
                    del guide
                else:
                    hidden = block(hidden)
                block.cpu()
                torch.cuda.empty_cache()
                print(
                    f"[Texture decode] stage={stage_index} "
                    f"block={block_index} tokens={len(hidden.coords)}",
                    flush=True,
                )
        hidden = hidden.type(value.dtype)
        hidden = hidden.replace(functional.layer_norm(
            hidden.feats, hidden.feats.shape[-1:]
        ))
        return run_module(self.output_layer, hidden)

    saved = torch.load(args.texture_latent, map_location="cpu", weights_only=False)
    guides = torch.load(args.subdivisions, map_location="cpu", weights_only=False)
    latent = SparseTensor(
        feats=saved["feats"].cuda(), coords=saved["coords"].cuda()
    )
    pipeline = Pixal3DImageTo3DPipeline.from_pretrained(args.model_path)
    decoder = pipeline.models["tex_slat_decoder"]
    pipeline.models = {"tex_slat_decoder": decoder}
    SparseUnetVaeDecoder.forward = streaming_forward
    with torch.inference_mode():
        decoded = decoder(latent, guide_subs=guides)
        pbr = decoded * 0.5 + 0.5
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "coords": pbr.coords.cpu(),
        "attrs": pbr.feats.cpu(),
        "resolution": int(saved["resolution"]),
    }, args.output)
    print(f"[Saved] {args.output}: {len(pbr.coords)} PBR voxels", flush=True)


def _export(args) -> None:
    _prepare_imports(args.pixal3d_root)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    import numpy as np
    import o_voxel
    import torch

    mesh = torch.load(args.mesh, map_location="cpu", weights_only=False)
    pbr = torch.load(args.pbr_volume, map_location="cpu", weights_only=False)
    resolution = int(pbr["resolution"])
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh["vertices"].cuda(),
        faces=mesh["faces"].int().cuda(),
        attr_volume=pbr["attrs"].cuda(),
        coords=pbr["coords"][:, 1:].cuda(),
        attr_layout={
            "base_color": slice(0, 3),
            "metallic": slice(3, 4),
            "roughness": slice(4, 5),
            "alpha": slice(5, 6),
        },
        grid_size=resolution,
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
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    glb.export(args.output, extension_webp=True)
    print(f"[Saved] {args.output}", flush=True)


def _all(args) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    common = [sys.executable, str(Path(__file__).resolve())]
    environment = os.environ.copy()
    generate = common + [
        "generate",
        "--pixal3d-root", args.pixal3d_root,
        "--model-path", args.model_path,
        "--image", args.image,
        "--shape-latent", args.shape_latent,
        "--output", str(output_dir / "texture-latent.pt"),
        "--camera-angle-x", str(args.camera_angle_x),
        "--camera-distance", str(args.camera_distance),
        "--steps", str(args.steps),
        "--seed", str(args.seed),
    ]
    decode = common + [
        "decode",
        "--pixal3d-root", args.pixal3d_root,
        "--model-path", args.model_path,
        "--texture-latent", str(output_dir / "texture-latent.pt"),
        "--subdivisions", args.subdivisions,
        "--output", str(output_dir / "pbr-volume.pt"),
    ]
    export = common + [
        "export",
        "--pixal3d-root", args.pixal3d_root,
        "--mesh", args.mesh,
        "--pbr-volume", str(output_dir / "pbr-volume.pt"),
        "--output", str(output_dir / "local-textured.glb"),
        "--decimation-target", str(args.decimation_target),
        "--texture-size", str(args.texture_size),
    ]
    for command in (generate, decode, export):
        subprocess.run(command, check=True, env=environment)


def _common_model(parser):
    parser.add_argument("--pixal3d-root", required=True)
    parser.add_argument("--model-path", required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    generate = subparsers.add_parser("generate")
    _common_model(generate)
    generate.add_argument("--image", required=True)
    generate.add_argument("--shape-latent", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--camera-angle-x", type=float, required=True)
    generate.add_argument("--camera-distance", type=float, required=True)
    generate.add_argument("--steps", type=int, default=12)
    generate.add_argument("--seed", type=int, default=42)

    decode = subparsers.add_parser("decode")
    _common_model(decode)
    decode.add_argument("--texture-latent", required=True)
    decode.add_argument("--subdivisions", required=True)
    decode.add_argument("--output", required=True)

    export = subparsers.add_parser("export")
    export.add_argument("--pixal3d-root", required=True)
    export.add_argument("--mesh", required=True)
    export.add_argument("--pbr-volume", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--decimation-target", type=int, default=1_000_000)
    export.add_argument("--texture-size", type=int, default=4096)

    all_phases = subparsers.add_parser("all")
    _common_model(all_phases)
    all_phases.add_argument("--image", required=True)
    all_phases.add_argument("--shape-latent", required=True)
    all_phases.add_argument("--subdivisions", required=True)
    all_phases.add_argument("--mesh", required=True)
    all_phases.add_argument("--output-dir", required=True)
    all_phases.add_argument("--camera-angle-x", type=float, required=True)
    all_phases.add_argument("--camera-distance", type=float, required=True)
    all_phases.add_argument("--steps", type=int, default=12)
    all_phases.add_argument("--seed", type=int, default=42)
    all_phases.add_argument("--decimation-target", type=int, default=1_000_000)
    all_phases.add_argument("--texture-size", type=int, default=4096)
    args = parser.parse_args()
    if hasattr(args, "steps") and not 1 <= args.steps <= 50:
        parser.error("--steps must be in [1, 50]")
    if hasattr(args, "texture_size") and args.texture_size < 64:
        parser.error("--texture-size must be at least 64")
    if hasattr(args, "decimation_target") and args.decimation_target < 1:
        parser.error("--decimation-target must be positive")
    {"generate": _generate, "decode": _decode, "export": _export, "all": _all}[
        args.phase
    ](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
