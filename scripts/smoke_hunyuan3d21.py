#!/usr/bin/env python3
"""Run a standalone Hunyuan3D 2.1 geometry smoke test.

This intentionally uses the already installed ComfyUI Hunyuan3D 2.1 runtime,
but does not start ComfyUI or touch a running Diffusion Editor instance.
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
import sys
import tempfile


DEFAULT_COMFY_ROOT = Path("/home/mirmik/soft/ComfyUI")
DEFAULT_NODE_ROOT = DEFAULT_COMFY_ROOT / "custom_nodes/ComfyUI-Hunyuan3d-2-1"
DEFAULT_DIT = DEFAULT_COMFY_ROOT / "models/diffusion_models/hunyuan3d-dit-v2-1.ckpt"
DEFAULT_VAE = DEFAULT_COMFY_ROOT / "models/vae/hunyuan3d-vae-v2-1.ckpt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Input image")
    parser.add_argument("output", type=Path, help="Output GLB or OBJ")
    parser.add_argument("--comfy-root", type=Path, default=DEFAULT_COMFY_ROOT)
    parser.add_argument("--node-root", type=Path, default=DEFAULT_NODE_ROOT)
    parser.add_argument("--dit", type=Path, default=DEFAULT_DIT)
    parser.add_argument("--vae", type=Path, default=DEFAULT_VAE)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=12646)
    parser.add_argument("--octree-resolution", type=int, default=384)
    parser.add_argument("--num-chunks", type=int, default=8000)
    parser.add_argument("--bounds", type=float, default=1.01)
    parser.add_argument("--mc-level", type=float, default=0.0)
    parser.add_argument("--mc-algorithm", choices=("mc", "dmc"), default="mc")
    parser.add_argument("--remove-background", action="store_true")
    parser.add_argument("--disable-flash-vdm", action="store_true")
    parser.add_argument(
        "--latent-input",
        type=Path,
        help="Reuse a previously saved shape latent instead of sampling DiT",
    )
    parser.add_argument(
        "--latent-output",
        type=Path,
        help="Shape latent path (defaults next to the output mesh)",
    )
    return parser


def _require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    input_path = _require_file(args.input, "input image")
    dit_path = _require_file(args.dit, "shape DiT checkpoint")
    vae_path = _require_file(args.vae, "shape VAE checkpoint")
    comfy_root = args.comfy_root.expanduser().resolve()
    node_root = args.node_root.expanduser().resolve()
    shape_root = node_root / "hy3dshape"
    config_path = node_root / "configs/dit_config_2_1.yaml"
    _require_file(config_path, "DiT config")

    sys.path.insert(0, str(comfy_root))
    sys.path.insert(0, str(shape_root))
    os.chdir(comfy_root)

    import torch
    import trimesh
    from PIL import Image

    from comfy.utils import load_torch_file
    import comfy.model_management as mm
    from hy3dshape.models.autoencoders import ShapeVAE
    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    from hy3dshape.rembg import BackgroundRemover

    if not torch.cuda.is_available():
        raise RuntimeError("Hunyuan3D 2.1 requires a CUDA GPU")

    device = mm.get_torch_device()
    offload_device = mm.unet_offload_device()
    if args.latent_input is not None:
        latent_input = _require_file(args.latent_input, "shape latent")
        print(f"[hunyuan3d21] loading shape latent {latent_input}", flush=True)
        latents = torch.load(latent_input, map_location=device, weights_only=True)
    else:
        image = Image.open(input_path).convert("RGBA")
        if args.remove_background:
            print("[hunyuan3d21] removing background", flush=True)
            image = BackgroundRemover()(image)

        # The ComfyUI custom-node loader imports the plugin under its directory
        # name, so its bundled config uses package-relative targets.  A standalone
        # process imports ``hy3dshape`` normally and needs absolute targets.
        config_text = config_path.read_text(encoding="utf-8").replace(
            ".hy3dshape.hy3dshape.", "hy3dshape."
        )
        print("[hunyuan3d21] loading shape DiT", flush=True)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", encoding="utf-8") as config:
            config.write(config_text)
            config.flush()
            pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
                config_path=config.name,
                ckpt_path=str(dit_path),
                offload_device=offload_device,
                attention_mode="sdpa",
            )
        print("[hunyuan3d21] sampling shape latent", flush=True)
        latents = pipeline(
            image=image,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            generator=torch.manual_seed(args.seed % (2**32)),
        )
        del pipeline
        mm.soft_empty_cache()
        torch.cuda.empty_cache()
        gc.collect()

        latent_output = (
            args.latent_output.expanduser().resolve()
            if args.latent_output is not None
            else args.output.expanduser().resolve().with_suffix(".shape-latent.pt")
        )
        latent_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(latents.detach().cpu(), latent_output)
        print(f"[hunyuan3d21] saved shape latent {latent_output}", flush=True)

    print("[hunyuan3d21] loading shape VAE", flush=True)
    vae_config = {
        "num_latents": 4096,
        "embed_dim": 64,
        "num_freqs": 8,
        "include_pi": False,
        "heads": 16,
        "width": 1024,
        "num_encoder_layers": 8,
        "num_decoder_layers": 16,
        "qkv_bias": False,
        "qk_norm": True,
        "scale_factor": 1.0039506158752403,
        "geo_decoder_mlp_expand_ratio": 4,
        "geo_decoder_downsample_ratio": 1,
        "geo_decoder_ln_post": True,
        "point_feats": 4,
        "pc_size": 81920,
        "pc_sharpedge_size": 0,
    }
    vae = ShapeVAE(**vae_config)
    vae.load_state_dict(load_torch_file(str(vae_path)))
    vae.eval().to(torch.float16).to(device)
    vae.enable_flashvdm_decoder(
        enabled=not args.disable_flash_vdm,
        mc_algo=args.mc_algorithm,
    )

    print("[hunyuan3d21] decoding latent to mesh", flush=True)
    with torch.inference_mode():
        decoded = vae.decode(latents)
        result = vae.latents2mesh(
            decoded,
            output_type="trimesh",
            bounds=args.bounds,
            mc_level=args.mc_level,
            num_chunks=args.num_chunks,
            octree_resolution=args.octree_resolution,
            mc_algo=args.mc_algorithm,
            enable_pbar=True,
        )[0]
    faces = result.mesh_f[:, ::-1]
    mesh = trimesh.Trimesh(result.mesh_v, faces, process=False)

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(output_path)
    print(
        f"[hunyuan3d21] exported {output_path} "
        f"({len(mesh.vertices)} vertices, {len(mesh.faces)} faces)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
