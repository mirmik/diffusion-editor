"""One-shot SDXL + Depth ControlNet + IPAdapter worker. No server or Comfy imports."""
import json
import math
import os
from pathlib import Path
import sys
import time


def model_path(folder, name):
    root = Path(os.environ.get("DIFFUSION_EDITOR_STABLEGEN_MODELS", "/home/mirmik/soft/ComfyUI/models"))
    path = Path(name).expanduser()
    path = path if path.is_absolute() else root / folder / path
    if not path.is_file():
        raise FileNotFoundError(f"StableGen model is missing: {path}")
    return path.resolve()


def main(root):
    # Never download models implicitly during an editor operation.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download
    from transformers import CLIPVisionConfig, CLIPVisionModelWithProjection, CLIPImageProcessor
    from diffusers import ControlNetModel, StableDiffusionXLControlNetInpaintPipeline, EulerDiscreteScheduler
    from accelerate import init_empty_weights

    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    start = time.monotonic()
    settings = json.loads((root / "request.json").read_text())["settings"]
    paths = {key: model_path(folder, settings.get(key, default)) for key, folder, default in [
        ("checkpoint", "checkpoints", "RealVisXL_V5.0_fp16.safetensors"),
        ("lora", "loras", "sdxl_lightning_8step_lora.safetensors"),
        ("depth_model", "controlnet", "controlnet_depth_sdxl.safetensors"),
        ("ip_adapter", "ipadapter", "ip-adapter-plus_sdxl_vit-h.safetensors"),
        ("image_encoder", "clip_vision", "CLIP-ViT-H-14-laion2B-s32B-b79K.safetensors"),
    ]}
    config = os.environ.get("DIFFUSION_EDITOR_STABLEGEN_SDXL_CONFIG")
    if not config:
        config = snapshot_download("stabilityai/stable-diffusion-xl-base-1.0", local_files_only=True)

    # Architecture of diffusers/controlnet-depth-sdxl-1.0; the local file is
    # already a Diffusers state dict. Strict loading rejects incompatible weights.
    with init_empty_weights():
        controlnet = ControlNetModel(
            in_channels=4, conditioning_channels=3,
            down_block_types=("DownBlock2D", "CrossAttnDownBlock2D", "CrossAttnDownBlock2D"),
            block_out_channels=(320, 640, 1280), layers_per_block=2,
            cross_attention_dim=2048, attention_head_dim=(5, 10, 20),
            use_linear_projection=True, addition_embed_type="text_time",
            addition_time_embed_dim=256, projection_class_embeddings_input_dim=2816,
            transformer_layers_per_block=(1, 2, 10),
        )
        encoder = CLIPVisionModelWithProjection(CLIPVisionConfig(
            hidden_size=1280, intermediate_size=5120, num_hidden_layers=32,
            num_attention_heads=16, image_size=224, patch_size=14,
            projection_dim=1024, hidden_act="gelu",
        ))
    print("Loading Depth ControlNet and CLIP ViT-H", flush=True)
    controlnet.load_state_dict(load_file(str(paths["depth_model"])), strict=True, assign=True)
    state = load_file(str(paths["image_encoder"]))
    # Old transformers checkpoints persisted this deterministic buffer.
    state.pop("vision_model.embeddings.position_ids", None)
    encoder.load_state_dict(state, strict=True, assign=True)
    del state
    controlnet.to(dtype=torch.float16)
    encoder.to(dtype=torch.float16)

    class MaskedPipeline(StableDiffusionXLControlNetInpaintPipeline):
        def prepare_latents(self, *args, **kwargs):
            result = super().prepare_latents(*args, **kwargs)
            self.pass_noise, self.pass_image_latents = result[1:3]
            return result

        def prepare_mask_latents(self, *args, **kwargs):
            mask, masked_image = super().prepare_mask_latents(*args, **kwargs)
            self.pass_mask = mask[:1].clone()
            # The callback applies a time-dependent mask instead of the default
            # constant blend. Keep the pipeline's own blend neutral.
            return torch.ones_like(mask), masked_image

        def get_timesteps(self, *args, **kwargs):
            result = super().get_timesteps(*args, **kwargs)
            self.pass_timesteps = result[0]
            return result

    print("Loading SDXL checkpoint", flush=True)
    pipe = MaskedPipeline.from_single_file(
        str(paths["checkpoint"]), config=config, local_files_only=True,
        controlnet=controlnet, image_encoder=encoder,
        feature_extractor=CLIPImageProcessor(), torch_dtype=torch.float16,
        add_watermarker=False,
    )
    if pipe.unet.config.in_channels != 4:
        raise ValueError("StableGen requires a standard four-channel SDXL checkpoint")
    pipe.load_lora_weights(str(paths["lora"].parent), weight_name=paths["lora"].name, local_files_only=True)
    pipe.fuse_lora()
    pipe.unload_lora_weights()
    pipe.load_ip_adapter(str(paths["ip_adapter"].parent), subfolder="",
                         weight_name=paths["ip_adapter"].name, image_encoder_folder=None,
                         local_files_only=True)
    pipe.set_ip_adapter_scale(settings["ip_strength"])
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.mask_processor.register_to_config(do_binarize=False)
    pipe.vae.enable_slicing()
    pipe.to("cuda")

    image = Image.open(root / "input-rgb.png").convert("RGB")
    mask = Image.open(root / "mask.png").convert("L")
    depth = Image.open(root / "control-depth.png").convert("RGB")
    reference = Image.open(root / "reference.png").convert("RGB")
    if mask.size != image.size or depth.size != image.size:
        raise ValueError("RGB, depth and mask must have identical dimensions")
    if not np.asarray(mask).any():
        raise ValueError("Empty generation mask")

    def differential_mask(pipeline, index, timestep, tensors):
        count = len(pipeline.pass_timesteps)
        # High mask values participate earlier; zero remains locked throughout.
        threshold = 1 - (index + 1) / count
        active = (pipeline.pass_mask > threshold).to(tensors["latents"].dtype)
        original = pipeline.pass_image_latents
        if index + 1 < count:
            next_t = pipeline.pass_timesteps[index + 1].reshape(1)
            original = pipeline.scheduler.add_noise(original, pipeline.pass_noise, next_t)
        tensors["latents"] = active * tensors["latents"] + (1 - active) * original
        print(f"Sampling {index + 1}/{count}", flush=True)
        return tensors

    # Preserve the requested number of executed steps at reduced denoise.
    total_steps = math.ceil(settings["steps"] / settings["denoise"])
    result = pipe(
        prompt=settings["prompt"], negative_prompt=settings["negative"],
        image=image, mask_image=mask, control_image=depth, ip_adapter_image=reference,
        width=image.width, height=image.height,
        num_inference_steps=total_steps, strength=settings["denoise"],
        guidance_scale=settings["cfg"], controlnet_conditioning_scale=settings["control_strength"],
        generator=torch.Generator(device="cuda").manual_seed(settings["seed"]),
        callback_on_step_end=differential_mask,
    ).images[0].convert("RGB")
    # VAE reconstruction also changes pixels outside the mask. Restore them
    # exactly in the 2D candidate; UV projection performs the soft final blend.
    pixels = np.array(result)
    pixels[np.asarray(mask) == 0] = np.asarray(image)[np.asarray(mask) == 0]
    Image.fromarray(pixels).save(root / "candidate.png")
    import diffusers
    (root / "generation.json").write_text(json.dumps({
        "status": "success", "backend": "diffusers", "diffusers_version": diffusers.__version__,
        "scheduler": "EulerDiscreteScheduler", "timestep_spacing": "trailing",
        "mask": "step-dependent", "executed_steps": len(pipe.pass_timesteps),
        "models": {k: str(v) for k, v in paths.items()},
        "elapsed_seconds": time.monotonic() - start,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
    }, indent=2))
    print("Image complete; worker exiting and releasing CUDA memory", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
