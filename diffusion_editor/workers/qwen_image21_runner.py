"""Private JSON-lines worker using Comfy's inference libraries, without a server.

Run with the dedicated Qwen 2.1 interpreter. Never import this runtime into the
editor or the shared Diffusers worker; Comfy owns its process-global settings.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback


def _watch_parent(parent_pid):
    # Cancellation kills the outer ML worker. Do not leave its GPU child alive,
    # including while the child is busy loading weights or denoising.
    while True:
        if os.getppid() != parent_pid:
            os._exit(1)
        time.sleep(0.5)


def import_runtime(parameters):
    root = Path(parameters["runtime_root"]).expanduser().resolve()
    sys.path.insert(0, str(root))
    import torch
    from comfy.cli_args import args

    args.cpu = parameters["device"] == "cpu"
    args.lowvram = parameters["memory_mode"] == "lowvram" and not args.cpu
    args.disable_metadata = True
    # Installed xFormers kernels do not support masked attention on RTX 5090.
    args.disable_xformers = True
    args.use_pytorch_cross_attention = True
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("Qwen Image 2.1 requires CUDA, but the NVIDIA driver/GPU is unavailable")
    import comfy.sd
    import comfy.model_management
    import comfy.utils
    import nodes
    from comfy_extras.nodes_qwen import TextEncodeQwenImage21

    torch.set_grad_enabled(False)
    try:
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL, timeout=5,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unknown"
    return {
        "provider": "comfy.qwen_image21", "comfy_revision": revision,
        "python": sys.version.split()[0], "torch_version": torch.__version__,
        "device": parameters["device"], "dtype": parameters["dtype"],
        "memory_mode": parameters["memory_mode"], "attention_backend": "pytorch_sdpa",
    }


class Runtime:
    def __init__(self, parameters, adapters):
        self.info = import_runtime(parameters)
        import torch
        import comfy.sd as sd
        import comfy.utils

        dtype = getattr(torch, parameters["dtype"])
        self.model = sd.load_diffusion_model(
            parameters["model"], model_options={"dtype": dtype})
        if self.model is None or type(self.model.model).__name__ != "QwenImage21":
            raise ValueError("Transformer checkpoint is not a Qwen Image 2.1 model")
        self.clip = sd.load_clip(
            ckpt_paths=[parameters["text_encoder_checkpoint"]],
            clip_type=sd.CLIPType.QWEN_IMAGE,
        )
        if not hasattr(self.clip.cond_stage_model, "qwen3vl_8b"):
            raise ValueError("Qwen Image 2.1 requires its Qwen3-VL-8B text encoder")
        self.vae = sd.VAE(
            sd=comfy.utils.load_torch_file(parameters["vae_checkpoint"], safe_load=True),
            dtype=dtype,
        )
        if self.vae.latent_channels != 64 or self.vae.output_channels != 4:
            raise ValueError("Qwen Image 2.1 requires the 64-channel RGBA VAE")
        for adapter in adapters:
            if adapter["enabled"] and adapter["source"]:
                weights = comfy.utils.load_torch_file(adapter["source"], safe_load=True)
                self.model, self.clip = sd.load_lora_for_models(
                    self.model, self.clip, weights,
                    float(adapter["weight"]), float(adapter["weight"]),
                )

    def generate(self, request):
        import numpy as np
        import torch
        from PIL import Image
        import comfy.model_management as mm
        import nodes
        from comfy_extras.nodes_qwen import TextEncodeQwenImage21

        p = request["parameters"]
        width, height = int(request["width"]), int(request["height"])
        if width < 32 or height < 32 or width % 32 or height % 32:
            raise ValueError("Qwen Image 2.1 inference dimensions must be positive multiples of 32")
        images = {}
        for i, path in enumerate(request.get("images", [])):
            with Image.open(path) as source:
                image = source.convert("RGBA")
            # The first reference and output must use the same geometry.
            size = (width, height) if i == 0 else (
                max(32, round(image.width / 32) * 32),
                max(32, round(image.height / 32) * 32),
            )
            if image.size != size:
                image = image.resize(size, Image.Resampling.LANCZOS)
            images[f"image_{i + 1}"] = torch.from_numpy(
                np.array(image, dtype=np.float32) / 255.0).unsqueeze(0)
        with torch.inference_mode():
            encoded = TextEncodeQwenImage21.execute(
                self.clip, str(p["prompt"]), str(p["negative_prompt"]),
                vae=self.vae, resolution=0, images=images,
            )
            positive, negative = encoded[0], encoded[1]
            latent = {"samples": torch.zeros(
                [1, 64, height // 16, width // 16], device=mm.intermediate_device())}
            # Match the official Comfy workflow: native sampling configuration,
            # Euler/simple, CFG 1.0. Do not reuse the old 2511 Lightning schedule.
            sampled = nodes.common_ksampler(
                self.model, int(request["seed"]), int(p["steps"]),
                float(p["true_cfg_scale"]), "euler", "simple",
                positive, negative, latent,
            )[0]["samples"]
            pixels = (self.vae.decode_tiled(sampled) if p["vae_tiling"]
                      else self.vae.decode(sampled))
            # VAE may expose a singleton frame dimension.
            while pixels.ndim > 4 and pixels.shape[1] == 1:
                pixels = pixels.squeeze(1)
            if pixels.ndim != 4 or pixels.shape[-1] != 4:
                raise RuntimeError(f"Unexpected Qwen RGBA output shape: {tuple(pixels.shape)}")
            array = pixels[0].float().clamp(0, 1).cpu().numpy()
            image = Image.fromarray(np.rint(array * 255).astype(np.uint8), "RGBA")
            if p["output_mode"] == "RGB":
                background = Image.new("RGBA", image.size, "white")
                image = Image.alpha_composite(background, image).convert("RGB")
            image.save(request["output"], format="PNG")
        return {"width": image.width, "height": image.height, "mode": image.mode}


def main():
    protocol = sys.stdout
    sys.stdout = sys.stderr  # Third-party progress/prints must not corrupt IPC.
    threading.Thread(target=_watch_parent, args=(os.getppid(),), daemon=True).start()
    runtime = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            operation = request["operation"]
            if operation == "check":
                result = import_runtime(request["parameters"])
            elif operation == "load":
                if runtime is not None:
                    raise RuntimeError("Start a fresh Qwen process to load another configuration")
                runtime = Runtime(request["parameters"], request.get("adapters", []))
                result = runtime.info
            elif operation == "generate":
                if runtime is None:
                    raise RuntimeError("Qwen Image 2.1 is not loaded")
                result = runtime.generate(request)
            else:
                raise ValueError(f"Unknown Qwen operation: {operation}")
            response = {"ok": True, "result": result}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        protocol.write(json.dumps(response) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
