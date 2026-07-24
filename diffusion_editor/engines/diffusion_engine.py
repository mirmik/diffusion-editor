import os
from PIL import Image
from tcbase import log

from ..generation.types import (
    DiffusionInferenceResult,
    DiffusionRequest,
)
from .threaded_lifecycle import EngineTaskQueue

# Filename hints for v-prediction models
_VPRED_HINTS = ("vpred", "v-pred", "v_pred", "vprediction", "v-prediction", "v_prediction")


def _guess_prediction_type(path: str) -> str | None:
    """Guess prediction type from filename. Returns None if unknown."""
    name = os.path.basename(path).lower()
    for hint in _VPRED_HINTS:
        if hint in name:
            return "v_prediction"
    return None


def _import_torch():
    import torch
    return torch


def _import_sdxl_loaders():
    from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
    return StableDiffusionXLPipeline, DPMSolverMultistepScheduler


def _import_sdxl_mode_pipelines():
    from diffusers import (
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
    )
    return (
        StableDiffusionXLPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
    )


class DiffusionEngine:
    def __init__(self):
        self._pipe = None
        self._model_path = None
        self._tasks = EngineTaskQueue()
        self._pipe_mode = None  # "img2img" или "inpaint"
        self._ip_adapter_loaded = False
        self.model_info = {}  # диагностика — заполняется после загрузки

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    @property
    def is_busy(self) -> bool:
        return self._tasks.is_busy

    @property
    def model_path(self) -> str | None:
        return self._model_path

    def load_model(self, safetensors_path: str, prediction_type: str | None = None):
        torch = _import_torch()
        StableDiffusionXLPipeline, DPMSolverMultistepScheduler = _import_sdxl_loaders()

        self.unload()

        guessed = _guess_prediction_type(safetensors_path)
        chosen_prediction = prediction_type or guessed or "epsilon"

        # Load model with its default scheduler config (correct betas etc.)
        self._pipe = StableDiffusionXLPipeline.from_single_file(
            safetensors_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
        )
        # DPM++ 2M SDE Karras — inherit base config from model
        # timestep_spacing="trailing" matches A1111/Forge behaviour and
        # ensures the first timestep is at full noise (no white borders).
        self._pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self._pipe.scheduler.config,
            prediction_type=chosen_prediction,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            timestep_spacing="trailing",
        )
        self._pipe.to("cuda")
        self._model_path = safetensors_path
        self._pipe_mode = "txt2img"

        # Собираем диагностику
        sched = self._pipe.scheduler
        self.model_info = {
            "path": os.path.basename(safetensors_path),
            "scheduler": type(sched).__name__,
            "prediction_type": sched.config.get("prediction_type", "?"),
            "algorithm_type": sched.config.get("algorithm_type", "?"),
            "karras": sched.config.get("use_karras_sigmas", False),
            "guessed_from_name": guessed,
            "override": prediction_type,
        }
        log.info(f"[DiffusionEngine] Loaded: {self.model_info}")

    @property
    def ip_adapter_loaded(self) -> bool:
        return self._ip_adapter_loaded

    def load_ip_adapter(self):
        if self._pipe is None:
            raise RuntimeError("No model loaded")
        self._pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin",
        )
        self._ip_adapter_loaded = True
        log.info("[DiffusionEngine] IP-Adapter loaded")

    def unload(self):
        if self._pipe is not None:
            torch = _import_torch()
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()
        self._model_path = None
        self._pipe_mode = None
        self._ip_adapter_loaded = False

    def _ensure_pipeline(self, mode: str):
        (
            StableDiffusionXLPipeline,
            StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLInpaintPipeline,
        ) = _import_sdxl_mode_pipelines()

        if self._pipe is None:
            raise RuntimeError("No model loaded")
        if self._pipe_mode == mode:
            log.debug(f"[DiffusionEngine] _ensure_pipeline: already in {mode} mode")
            return
        log.debug(f"[DiffusionEngine] _ensure_pipeline: switching {self._pipe_mode} -> {mode}")
        components = dict(
            vae=self._pipe.vae,
            text_encoder=self._pipe.text_encoder,
            text_encoder_2=self._pipe.text_encoder_2,
            tokenizer=self._pipe.tokenizer,
            tokenizer_2=self._pipe.tokenizer_2,
            unet=self._pipe.unet,
            scheduler=self._pipe.scheduler,
        )
        none_components = [k for k, v in components.items() if v is None]
        if none_components:
            log.warn(f"[DiffusionEngine] None components: {none_components}")
        if self._ip_adapter_loaded:
            components["image_encoder"] = self._pipe.image_encoder
            components["feature_extractor"] = self._pipe.feature_extractor
        if mode == "txt2img":
            self._pipe = StableDiffusionXLPipeline(**components)
        elif mode == "inpaint":
            self._pipe = StableDiffusionXLInpaintPipeline(**components)
        else:
            self._pipe = StableDiffusionXLImg2ImgPipeline(**components)
        self._pipe_mode = mode

    def _img2img(self, image: Image.Image, prompt: str, negative_prompt: str,
                 strength: float, num_inference_steps: int,
                 guidance_scale: float, seed: int = -1,
                 ip_adapter_image: Image.Image = None,
                 ip_adapter_scale: float = 0.6) -> tuple[Image.Image, int]:
        self._ensure_pipeline("img2img")

        image = image.convert("RGB")

        # VAE работает с размерами кратными 8
        w, h = image.size
        w8 = (w // 8) * 8
        h8 = (h // 8) * 8
        if (w8, h8) != (w, h):
            image = image.resize((w8, h8), Image.LANCZOS)

        torch = _import_torch()
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        log.debug(
            "[DiffusionEngine] _img2img prompt=%r neg=%r size=%s strength=%s steps=%s cfg=%s seed=%s ip_adapter=%s scale=%s model=%s"
            % (
                prompt, negative_prompt, image.size, strength, num_inference_steps,
                guidance_scale, seed, ip_adapter_image is not None, ip_adapter_scale, self._model_path
            )
        )

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            original_size=(1024, 1024),
            target_size=(h8, w8),
            crops_coords_top_left=(0, 0),
        )
        if ip_adapter_image is not None and self._ip_adapter_loaded:
            self._pipe.set_ip_adapter_scale(ip_adapter_scale)
            kwargs["ip_adapter_image"] = ip_adapter_image

        result = self._pipe(**kwargs).images[0]

        log.debug(f"[DiffusionEngine] _img2img done, result size: {result.size}")
        return result, seed

    def _inpaint(self, image: Image.Image, mask_image: Image.Image,
                 prompt: str, negative_prompt: str,
                 strength: float, num_inference_steps: int,
                 guidance_scale: float, seed: int = -1,
                 ip_adapter_image: Image.Image = None,
                 ip_adapter_scale: float = 0.6,
                 masked_content: str = "original") -> tuple[Image.Image, int]:
        self._ensure_pipeline("inpaint")

        image = image.convert("RGB")
        mask_image = mask_image.convert("L")

        w, h = image.size
        w8 = (w // 8) * 8
        h8 = (h // 8) * 8
        if (w8, h8) != (w, h):
            image = image.resize((w8, h8), Image.LANCZOS)
            mask_image = mask_image.resize((w8, h8), Image.NEAREST)

        # Preprocess masked area based on masked_content mode
        if masked_content != "original":
            import numpy as np
            img_arr = np.array(image)
            mask_arr = np.array(mask_image).astype(np.float32) / 255.0
            if masked_content == "fill":
                from PIL import ImageFilter
                blurred = image.filter(ImageFilter.GaussianBlur(radius=32))
                blur_arr = np.array(blurred)
                mask_3d = mask_arr[:, :, None]
                img_arr = (blur_arr * mask_3d + img_arr * (1 - mask_3d)).astype(np.uint8)
            elif masked_content == "latent_noise":
                noise = np.random.randint(0, 256, img_arr.shape, dtype=np.uint8)
                mask_3d = mask_arr[:, :, None]
                img_arr = (noise * mask_3d + img_arr * (1 - mask_3d)).astype(np.uint8)
            elif masked_content == "latent_nothing":
                mask_3d = mask_arr[:, :, None]
                gray = np.full_like(img_arr, 127)
                img_arr = (gray * mask_3d + img_arr * (1 - mask_3d)).astype(np.uint8)
            image = Image.fromarray(img_arr, "RGB")

        torch = _import_torch()
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        log.debug(
            "[DiffusionEngine] _inpaint prompt=%r neg=%r size=%s mask=%s masked_content=%s strength=%s steps=%s cfg=%s seed=%s ip_adapter=%s scale=%s model=%s"
            % (
                prompt, negative_prompt, image.size, mask_image.size, masked_content, strength,
                num_inference_steps, guidance_scale, seed, ip_adapter_image is not None,
                ip_adapter_scale, self._model_path
            )
        )

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            mask_image=mask_image,
            width=w8,
            height=h8,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            original_size=(1024, 1024),
            target_size=(h8, w8),
            crops_coords_top_left=(0, 0),
        )
        if ip_adapter_image is not None and self._ip_adapter_loaded:
            self._pipe.set_ip_adapter_scale(ip_adapter_scale)
            kwargs["ip_adapter_image"] = ip_adapter_image

        log.debug(f"[DiffusionEngine] _inpaint: calling pipe({w8}x{h8}, steps={num_inference_steps})")
        result = self._pipe(**kwargs).images[0]

        log.debug(f"[DiffusionEngine] _inpaint done, result size: {result.size}")
        return result, seed

    def _txt2img(self, prompt: str, negative_prompt: str,
                 width: int, height: int,
                 num_inference_steps: int,
                 guidance_scale: float, seed: int = -1,
                 ip_adapter_image: Image.Image = None,
                 ip_adapter_scale: float = 0.6) -> tuple[Image.Image, int]:
        self._ensure_pipeline("txt2img")

        w8 = (width // 8) * 8
        h8 = (height // 8) * 8

        torch = _import_torch()
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        log.debug(
            "[DiffusionEngine] _txt2img prompt=%r neg=%r size=%sx%s steps=%s cfg=%s seed=%s ip_adapter=%s scale=%s"
            % (
                prompt, negative_prompt, w8, h8, num_inference_steps,
                guidance_scale, seed, ip_adapter_image is not None, ip_adapter_scale
            )
        )

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=w8,
            height=h8,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            # SDXL micro-conditioning: tell model this is a full 1024x1024 image
            original_size=(1024, 1024),
            target_size=(h8, w8),
            crops_coords_top_left=(0, 0),
        )
        if ip_adapter_image is not None and self._ip_adapter_loaded:
            self._pipe.set_ip_adapter_scale(ip_adapter_scale)
            kwargs["ip_adapter_image"] = ip_adapter_image

        result = self._pipe(**kwargs).images[0]

        log.debug(f"[DiffusionEngine] _txt2img done, result size: {result.size}")
        return result, seed

    def submit_request(self, request: DiffusionRequest, meta=None):
        return self._tasks.submit(
            "inference",
            lambda _cancel: self._run_inference(
                request.image,
                request.prompt,
                request.negative_prompt,
                request.strength,
                request.steps,
                request.guidance_scale,
                request.seed,
                request.mode,
                request.mask_image,
                request.masked_content,
                request.ip_adapter_image,
                request.ip_adapter_scale,
                request.width,
                request.height,
            ),
            meta=meta,
            name="diffusion-inference",
            on_error=lambda _exc: log.exception(
                f"Diffusion inference failed (mode={request.mode})"
            ),
        )

    def _run_inference(self, image, prompt, negative_prompt, strength, steps,
                       guidance_scale, seed, mode, mask_image, masked_content,
                       ip_adapter_image, ip_adapter_scale, width, height):
        log.debug(f"[DiffusionEngine] _run_inference thread started, mode={mode}")
        if mode == "txt2img":
            result_image, used_seed = self._txt2img(
                prompt, negative_prompt, width, height,
                steps, guidance_scale, seed,
                ip_adapter_image, ip_adapter_scale)
        elif mode == "inpaint" and mask_image is not None:
            result_image, used_seed = self._inpaint(
                image, mask_image, prompt, negative_prompt,
                strength, steps, guidance_scale, seed,
                ip_adapter_image, ip_adapter_scale,
                masked_content=masked_content)
        else:
            result_image, used_seed = self._img2img(
                image, prompt, negative_prompt,
                strength, steps, guidance_scale, seed,
                ip_adapter_image, ip_adapter_scale)
        log.debug("[DiffusionEngine] _run_inference OK")
        return DiffusionInferenceResult(image=result_image, seed=used_seed)

    def submit_load_ip_adapter(self):
        if self._pipe is None:
            return False
        return self._tasks.submit(
            "load_ip_adapter",
            lambda _cancel: self._run_load_ip_adapter(),
            name="diffusion-ip-adapter-load",
            on_error=lambda _exc: log.exception("IP-Adapter load failed"),
        )

    def _run_load_ip_adapter(self):
        self.load_ip_adapter()
        return True

    def submit_load(self, path: str, prediction_type: str | None = None):
        return self._tasks.submit(
            "load",
            lambda _cancel: self._run_load(path, prediction_type),
            meta=path,
            name="diffusion-model-load",
            on_error=lambda _exc: log.exception(f"Model load failed: {path}"),
        )

    def _run_load(self, path, prediction_type):
        self.load_model(path, prediction_type)
        return path

    def poll_event(self):
        return self._tasks.poll_event()

    def cancel(self) -> bool:
        return self._tasks.cancel()

    def shutdown(self, timeout: float = 1.0):
        if self._tasks.shutdown(timeout):
            self.unload()
