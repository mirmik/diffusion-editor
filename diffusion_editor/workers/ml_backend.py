"""Worker-only Diffusers and Transformers model implementation."""

from __future__ import annotations

from dataclasses import replace
import gc
import os
import json
from pathlib import Path
import secrets
import sys
import tempfile
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageFilter

from ..generation.provenance import (
    FrozenJsonObject,
    GenerationProvenance,
    ModelIdentity,
    ModelIdentityPolicy,
    ModelIdentityStatus,
    RequestProvenance,
    enforce_model_identity_policy,
    floating_model_identity,
    resolve_local_model_identity,
)
from ..generation.image_edit_profiles import (
    FLUX2_KLEIN_PROFILE_ID,
    LEGACY_INSTRUCT_PROFILE_ID,
    SENSENOVA_U15_PROFILE_ID,
    image_edit_profile,
    parse_float_list,
    parse_int_tuple,
)


_VPRED_HINTS = (
    "vpred", "v-pred", "v_pred", "vprediction", "v-prediction", "v_prediction"
)


class RealMlBackend:
    """Persistent model state. This module is imported only by the worker."""

    def __init__(self) -> None:
        self._diffusion_pipe = None
        self._diffusion_mode: str | None = None
        self._diffusion_path: str | None = None
        self._diffusion_identity: ModelIdentity | None = None
        self._diffusion_warnings: tuple[str, ...] = ()
        self._diffusion_device: str | None = None
        self._diffusion_dtype: str | None = None
        self._ip_adapter_loaded = False
        self._ip_adapter_identity: ModelIdentity | None = None
        self._instruct_pipe = None
        self._instruct_identity: ModelIdentity | None = None
        self._instruct_warnings: tuple[str, ...] = ()
        self._instruct_device: str | None = None
        self._instruct_dtype: str | None = None
        self._image_edit_profile_id: str | None = None
        self._image_edit_lora_adapters: tuple[dict[str, Any], ...] = ()
        self._dino_model = None
        self._dino_processor = None
        self._dino_key: tuple[str, str] | None = None
        self._sam_model = None
        self._sam_processor = None
        self._sam_key: tuple[str, str] | None = None
        self._depth_model = None
        self._depth_processor = None
        self._depth_key: tuple[str, str] | None = None

    @staticmethod
    def gpu_available() -> bool:
        import torch

        return bool(torch.cuda.is_available())

    @staticmethod
    def _device(requested: str | None = None) -> str:
        import torch

        if requested == "cpu":
            return "cpu"
        if requested in {"cuda", "rocm"}:
            if not torch.cuda.is_available():
                raise RuntimeError(f"Requested accelerator is unavailable: {requested}")
            return "cuda"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def _dtype(device: str):
        import torch

        return torch.float16 if device == "cuda" else torch.float32

    @staticmethod
    def _configured_dtype(name: str, device: str):
        import torch

        values = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = values.get(str(name))
        if dtype is None:
            raise ValueError(f"Unsupported image edit dtype: {name}")
        if device == "cpu" and dtype == torch.float16:
            raise ValueError("float16 image edit inference is not supported on CPU")
        return dtype

    def load_diffusion(self, data: dict[str, Any]) -> dict[str, Any]:
        import torch
        from diffusers import (
            DPMSolverMultistepScheduler,
            StableDiffusionXLPipeline,
        )

        self._unload_depth()
        self.unload_diffusion()
        model_path = str(data["model_path"])
        device = self._device(data.get("device"))
        identity, identity_warnings = resolve_local_model_identity(
            model_path,
            expected_content_hash=data.get("expected_content_hash"),
            policy=data.get(
                "model_identity_policy",
                ModelIdentityPolicy.WARN.value,
            ),
        )
        name = os.path.basename(model_path).lower()
        guessed = (
            "v_prediction"
            if any(hint in name for hint in _VPRED_HINTS)
            else None
        )
        chosen = data.get("prediction_type") or guessed or "epsilon"
        pipe = StableDiffusionXLPipeline.from_single_file(
            model_path,
            torch_dtype=self._dtype(device),
            use_safetensors=True,
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            prediction_type=chosen,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
            timestep_spacing="trailing",
        )
        pipe.to(device)
        self._diffusion_pipe = pipe
        self._diffusion_path = model_path
        self._diffusion_identity = identity
        self._diffusion_warnings = identity_warnings
        self._diffusion_device = device
        self._diffusion_dtype = str(self._dtype(device)).removeprefix("torch.")
        self._diffusion_mode = "txt2img"
        scheduler = pipe.scheduler
        return {
            "model_path": model_path,
            "model_info": {
                "path": os.path.basename(model_path),
                "scheduler": type(scheduler).__name__,
                "prediction_type": scheduler.config.get("prediction_type", "?"),
                "algorithm_type": scheduler.config.get("algorithm_type", "?"),
                "karras": scheduler.config.get("use_karras_sigmas", False),
                "guessed_from_name": guessed,
                "override": data.get("prediction_type"),
                "device": device,
                "dtype": self._diffusion_dtype,
                "pipeline": type(pipe).__name__,
                "model_identity": identity.to_dict(),
                "warnings": list(identity_warnings),
            },
        }

    def unload_diffusion(self) -> None:
        pipe = self._diffusion_pipe
        self._diffusion_pipe = None
        self._diffusion_path = None
        self._diffusion_mode = None
        self._diffusion_identity = None
        self._diffusion_warnings = ()
        self._diffusion_device = None
        self._diffusion_dtype = None
        self._ip_adapter_loaded = False
        self._ip_adapter_identity = None
        del pipe
        self._release_accelerator_memory()

    @staticmethod
    def _release_accelerator_memory() -> None:
        """Drop cyclic model references before returning cached CUDA blocks."""

        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def unload_image_edit(self) -> None:
        pipe = self._instruct_pipe
        self._instruct_pipe = None
        self._instruct_identity = None
        self._instruct_warnings = ()
        self._instruct_device = None
        self._instruct_dtype = None
        self._image_edit_profile_id = None
        self._image_edit_lora_adapters = ()
        del pipe
        self._release_accelerator_memory()

    def _unload_depth(self) -> None:
        model = self._depth_model
        had_model = model is not None
        self._depth_model = None
        self._depth_processor = None
        self._depth_key = None
        del model
        if had_model:
            self._release_accelerator_memory()

    def _prepare_depth_memory(self) -> None:
        """Give one high-quality depth model exclusive ML-worker VRAM."""

        if self._diffusion_pipe is not None:
            self.unload_diffusion()
        if self._instruct_pipe is not None:
            self.unload_image_edit()
        dino_model = self._dino_model
        sam_model = self._sam_model
        had_detection_model = dino_model is not None or sam_model is not None
        self._dino_model = None
        self._dino_processor = None
        self._dino_key = None
        self._sam_model = None
        self._sam_processor = None
        self._sam_key = None
        del dino_model, sam_model
        if had_detection_model:
            self._release_accelerator_memory()

    def load_ip_adapter(self) -> dict[str, Any]:
        if self._diffusion_pipe is None:
            raise RuntimeError("No diffusion model loaded")
        self._diffusion_pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin",
        )
        self._ip_adapter_loaded = True
        self._ip_adapter_identity = floating_model_identity(
            "huggingface",
            "h94/IP-Adapter",
        )
        return {
            "loaded": True,
            "model_identity": self._ip_adapter_identity.to_dict(),
            "warnings": [self._ip_adapter_identity.warning],
        }

    def _ensure_diffusion_mode(self, mode: str) -> None:
        from diffusers import (
            StableDiffusionXLImg2ImgPipeline,
            StableDiffusionXLInpaintPipeline,
            StableDiffusionXLPipeline,
        )

        pipe = self._diffusion_pipe
        if pipe is None:
            raise RuntimeError("No diffusion model loaded")
        if self._diffusion_mode == mode:
            return
        components = {
            "vae": pipe.vae,
            "text_encoder": pipe.text_encoder,
            "text_encoder_2": pipe.text_encoder_2,
            "tokenizer": pipe.tokenizer,
            "tokenizer_2": pipe.tokenizer_2,
            "unet": pipe.unet,
            "scheduler": pipe.scheduler,
        }
        if self._ip_adapter_loaded:
            components["image_encoder"] = pipe.image_encoder
            components["feature_extractor"] = pipe.feature_extractor
        pipeline_type = {
            "txt2img": StableDiffusionXLPipeline,
            "img2img": StableDiffusionXLImg2ImgPipeline,
            "inpaint": StableDiffusionXLInpaintPipeline,
        }[mode]
        self._diffusion_pipe = pipeline_type(**components)
        self._diffusion_mode = mode

    def diffusion(
        self,
        data: dict[str, Any],
        images: dict[str, Image.Image | None],
    ) -> tuple[Image.Image, int, dict[str, Any]]:
        import torch

        mode = str(data["mode"])
        self._ensure_diffusion_mode(mode)
        pipe = self._diffusion_pipe
        assert pipe is not None
        seed = self._resolve_seed(int(data["seed"]))
        generator = torch.Generator(device="cpu").manual_seed(seed)
        width = max(8, int(data["width"]) // 8 * 8)
        height = max(8, int(data["height"]) // 8 * 8)
        kwargs: dict[str, Any] = {
            "prompt": str(data["prompt"]),
            "negative_prompt": str(data["negative_prompt"]),
            "num_inference_steps": int(data["steps"]),
            "guidance_scale": float(data["guidance_scale"]),
            "generator": generator,
            "original_size": (1024, 1024),
            "target_size": (height, width),
            "crops_coords_top_left": (0, 0),
        }
        if mode != "txt2img":
            image = images["image"]
            if image is None:
                raise RuntimeError(f"{mode} requires an input image")
            image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            if mode == "inpaint" and data.get("masked_content") != "original":
                image = self._prepare_masked_content(
                    image,
                    images["mask"],
                    str(data["masked_content"]),
                    seed=seed,
                )
            kwargs["image"] = image
            kwargs["strength"] = float(data["strength"])
        if mode == "inpaint":
            mask = images["mask"]
            if mask is None:
                raise RuntimeError("inpaint requires a mask")
            kwargs.update(
                mask_image=mask.convert("L").resize(
                    (width, height), Image.Resampling.NEAREST
                ),
                width=width,
                height=height,
            )
        elif mode == "txt2img":
            kwargs.update(width=width, height=height)
        ip_image = images.get("ip_adapter")
        if ip_image is not None and self._ip_adapter_loaded:
            pipe.set_ip_adapter_scale(float(data["ip_adapter_scale"]))
            kwargs["ip_adapter_image"] = ip_image.convert("RGB")
        output = pipe(**kwargs).images[0]
        provenance = self._diffusion_provenance(
            data,
            pipe=pipe,
            seed=seed,
            width=width,
            height=height,
        )
        return output, seed, provenance.to_dict()

    @staticmethod
    def _prepare_masked_content(
        image: Image.Image,
        mask: Image.Image | None,
        mode: str,
        *,
        seed: int,
    ) -> Image.Image:
        if mask is None:
            return image
        mask_array = np.array(
            mask.convert("L").resize(image.size, Image.Resampling.NEAREST),
            dtype=np.float32,
        ) / 255.0
        image_array = np.array(image)
        if mode == "fill":
            replacement = np.array(image.filter(ImageFilter.GaussianBlur(32)))
        elif mode == "latent_noise":
            replacement = np.random.default_rng(seed).integers(
                0, 256, image_array.shape, dtype=np.uint8
            )
        elif mode == "latent_nothing":
            replacement = np.full_like(image_array, 127)
        else:
            return image
        alpha = mask_array[:, :, None]
        return Image.fromarray(
            (replacement * alpha + image_array * (1 - alpha)).astype(np.uint8),
            "RGB",
        )

    @staticmethod
    def _resolve_seed(request_seed: int) -> int:
        if request_seed == -1:
            return secrets.randbits(32)
        if request_seed < 0 or request_seed >= 2**32:
            raise ValueError("seed must be -1 or an unsigned 32-bit integer")
        return request_seed

    def load_image_edit(self, data: dict[str, Any]) -> dict[str, Any]:
        from diffusers import (
            EulerAncestralDiscreteScheduler,
            Flux2KleinPipeline,
            QwenImageEditPlusPipeline,
            StableDiffusionInstructPix2PixPipeline,
        )

        profile_id = str(data["profile_id"])
        profile = image_edit_profile(profile_id)
        parameters = profile.normalize(data.get("parameters"))
        lora_adapters = profile.normalize_lora_adapters(
            data.get("lora_adapters"))
        device = self._device(str(parameters["device"]))
        dtype = self._configured_dtype(str(parameters["dtype"]), device)
        model = str(parameters["model"])
        revision = str(parameters.get("revision", "")).strip() or None
        if Path(model).expanduser().exists():
            candidate = Path(model).expanduser().absolute()
            model_path = str(candidate)
            if candidate.is_file():
                identity, identity_warnings = resolve_local_model_identity(
                    model_path,
                    policy=ModelIdentityPolicy.WARN,
                )
            else:
                identity = floating_model_identity(
                    "local-directory",
                    candidate.name,
                    local_override=model_path,
                )
                identity_warnings = enforce_model_identity_policy(
                    identity, ModelIdentityPolicy.WARN.value)
            model = model_path
        else:
            identity = floating_model_identity(
                "huggingface", model, revision=revision)
            identity_warnings = enforce_model_identity_policy(
                identity, ModelIdentityPolicy.WARN.value)
        if profile_id == SENSENOVA_U15_PROFILE_ID:
            config_identity = identity
            config_warnings = identity_warnings
            gguf_checkpoint = str(
                parameters["gguf_checkpoint"]).strip()
            if not gguf_checkpoint:
                raise ValueError(
                    "SenseNova GGUF checkpoint path cannot be empty")
            identity, identity_warnings = resolve_local_model_identity(
                gguf_checkpoint,
                policy=ModelIdentityPolicy.WARN,
            )
            identity = replace(
                identity,
                extensions=FrozenJsonObject.capture({
                    "config_identity": config_identity.to_dict(),
                }),
            )
            identity_warnings = tuple(
                [*identity_warnings, *config_warnings])
        # A second heavyweight pipeline must never overlap the currently
        # loaded one in RAM/VRAM.  Keep the backend truthfully unloaded if any
        # part of the new load fails after this point.
        self._unload_depth()
        self.unload_image_edit()
        load_kwargs: dict[str, Any] = {
            "torch_dtype": dtype,
            "local_files_only": bool(
                parameters.get("local_files_only", False)),
        }
        if revision is not None:
            load_kwargs["revision"] = revision
        component_mode = "upstream"
        if profile.provider == "diffusers.qwen_image_edit_plus":
            transformer_path = str(
                parameters["transformer_checkpoint"]).strip()
            text_encoder_path = str(
                parameters["text_encoder_checkpoint"]).strip()
            if bool(transformer_path) != bool(text_encoder_path):
                raise ValueError(
                    "Qwen local FP8 loading requires both Transformer "
                    "checkpoint and Text encoder checkpoint"
                )
            if transformer_path:
                transformer, text_encoder = self._load_qwen_fp8_components(
                    model,
                    transformer_path=transformer_path,
                    text_encoder_path=text_encoder_path,
                    revision=revision,
                    local_files_only=bool(
                        parameters.get("local_files_only", False)),
                )
                load_kwargs.update(
                    transformer=transformer,
                    text_encoder=text_encoder,
                )
                component_mode = "local-scaled-fp8"
            pipe = QwenImageEditPlusPipeline.from_pretrained(
                model, **load_kwargs)
        elif profile_id == FLUX2_KLEIN_PROFILE_ID:
            pipe = Flux2KleinPipeline.from_pretrained(model, **load_kwargs)
        elif profile_id == SENSENOVA_U15_PROFILE_ID:
            from .sensenova_image_edit import SenseNovaImageEditPipeline

            pipe = SenseNovaImageEditPipeline(
                model_path=model,
                gguf_checkpoint=str(parameters["gguf_checkpoint"]),
                device=device,
                dtype=str(parameters["dtype"]),
                attention_backend=str(parameters["attention_backend"]),
                vram_mode=str(parameters["vram_mode"]),
                fast_vram_fraction=float(
                    parameters["fast_vram_fraction"]),
                fast_vram_headroom_gib=float(
                    parameters["fast_vram_headroom_gib"]),
                fast_activation_reserve_gib=float(
                    parameters["fast_activation_reserve_gib"]),
                fast_vram_budget_gib=float(
                    parameters["fast_vram_budget_gib"]),
            )
            component_mode = "gguf"
        elif profile_id == LEGACY_INSTRUCT_PROFILE_ID:
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                model, safety_checker=None, **load_kwargs)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config)
        else:  # protected by image_edit_profile; keeps type narrowing explicit
            raise ValueError(f"Unsupported image edit profile: {profile_id}")
        active_adapters = tuple(
            adapter for adapter in lora_adapters
            if adapter.enabled and adapter.source
        )
        if active_adapters:
            if profile_id == SENSENOVA_U15_PROFILE_ID:
                raise ValueError(
                    "SenseNova standalone runtime does not support LoRA "
                    "adapters"
                )
            adapter_names: list[str] = []
            adapter_weights: list[float] = []
            for index, adapter in enumerate(active_adapters):
                adapter_name = (
                    f"image_edit_{index}_"
                    f"{adapter.stable_id.replace('-', '_')}"
                )
                pipe.load_lora_weights(
                    str(Path(adapter.source).expanduser()),
                    adapter_name=adapter_name,
                )
                adapter_model = getattr(
                    pipe, "transformer", getattr(pipe, "unet", None))
                if adapter_model is None:
                    raise RuntimeError(
                        f"{profile.title} pipeline has no LoRA target model")
                self._cast_adapter_parameters(
                    adapter_model, adapter_name, dtype)
                adapter_names.append(adapter_name)
                adapter_weights.append(adapter.weight)
            pipe.set_adapters(
                adapter_names,
                adapter_weights=adapter_weights,
            )
        if profile_id != SENSENOVA_U15_PROFILE_ID:
            if bool(parameters["vae_tiling"]):
                pipe.vae.enable_tiling()
            if bool(parameters["cpu_offload"]):
                if device != "cuda":
                    raise ValueError("CPU offload requires the CUDA device")
                pipe.enable_model_cpu_offload()
            else:
                pipe.to(device)
        self._instruct_pipe = pipe
        self._instruct_identity = identity
        self._instruct_warnings = identity_warnings
        self._instruct_device = device
        self._instruct_dtype = str(dtype).removeprefix("torch.")
        self._image_edit_profile_id = profile_id
        self._image_edit_lora_adapters = tuple(
            adapter.to_dict() for adapter in lora_adapters)
        return {
            "loaded": True,
            "profile_id": profile_id,
            "profile_title": profile.title,
            "device": device,
            "dtype": self._instruct_dtype,
            "pipeline": type(pipe).__name__,
            "component_mode": component_mode,
            "model_identity": identity.to_dict(),
            "warnings": list(identity_warnings),
        }

    @staticmethod
    def _load_qwen_fp8_components(
        model: str,
        *,
        transformer_path: str,
        text_encoder_path: str,
        revision: str | None,
        local_files_only: bool,
    ):
        from accelerate import init_empty_weights
        from diffusers import QwenImageTransformer2DModel
        from transformers import (
            AutoConfig,
            Qwen2_5_VLForConditionalGeneration,
        )

        from .scaled_fp8 import load_scaled_fp8_checkpoint

        config_kwargs: dict[str, Any] = {
            "local_files_only": local_files_only,
        }
        if revision is not None:
            config_kwargs["revision"] = revision

        transformer_config = QwenImageTransformer2DModel.load_config(
            model, subfolder="transformer", **config_kwargs)
        with init_empty_weights():
            transformer = QwenImageTransformer2DModel.from_config(
                transformer_config)
        transformer = load_scaled_fp8_checkpoint(
            transformer, transformer_path)

        text_config = AutoConfig.from_pretrained(
            model, subfolder="text_encoder", **config_kwargs)
        with init_empty_weights():
            text_encoder = Qwen2_5_VLForConditionalGeneration(text_config)

        def text_encoder_key(key: str) -> str:
            if key.startswith("model."):
                return "model.language_model." + key.removeprefix("model.")
            if key.startswith("visual."):
                return "model.visual." + key.removeprefix("visual.")
            return key

        text_encoder = load_scaled_fp8_checkpoint(
            text_encoder,
            text_encoder_path,
            key_mapper=text_encoder_key,
        )
        return transformer, text_encoder

    @staticmethod
    def _cast_adapter_parameters(model, adapter_name: str, dtype) -> None:
        """Keep LoRA compute weights out of the FP8 storage dtype.

        PEFT initializes a new adapter with its base Linear dtype.  For our
        scaled-FP8 base layers that would incorrectly turn the BF16 Lightning
        LoRA into unscaled FP8.  Adapter parameters are small enough to retain
        the requested compute dtype.
        """
        marker = f".{adapter_name}."
        for name, parameter in model.named_parameters():
            if "lora_" in name and marker in name:
                parameter.data = parameter.data.to(dtype=dtype)

    def image_edit(
        self,
        data: dict[str, Any],
        image: Image.Image,
        reference_image: Image.Image | None = None,
    ) -> tuple[Image.Image, int, dict[str, Any]]:
        import torch

        profile_id = str(data["profile_id"])
        if (
                self._instruct_pipe is None
                or self._image_edit_profile_id != profile_id):
            raise RuntimeError(
                f"Image edit profile is not loaded: {profile_id}")
        profile = image_edit_profile(profile_id)
        parameters = profile.normalize(data.get("parameters"))
        requested_adapters = tuple(
            adapter.to_dict()
            for adapter in profile.normalize_lora_adapters(
                data.get("lora_adapters"))
        )
        if requested_adapters != self._image_edit_lora_adapters:
            raise RuntimeError(
                "Image edit LoRA configuration is not loaded")
        seed = self._resolve_seed(int(parameters["seed"]))
        if profile_id == SENSENOVA_U15_PROFILE_ID:
            result = self._instruct_pipe.edit(
                image,
                parameters,
                seed,
                reference_image=reference_image,
            )
            provenance = self._image_edit_provenance(
                profile_id, parameters, requested_adapters, seed, result.size)
            return result, seed, provenance.to_dict()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": str(parameters["prompt"]),
            "image": image.convert("RGB"),
            "num_inference_steps": int(parameters["steps"]),
            "generator": generator,
        }
        if reference_image is not None and profile.max_input_images > 1:
            kwargs["image"] = [
                image.convert("RGB"),
                reference_image.convert("RGB"),
            ]
        width = int(parameters.get("width", 0))
        height = int(parameters.get("height", 0))
        if width > 0:
            kwargs["width"] = width
        if height > 0:
            kwargs["height"] = height
        sigmas = parse_float_list(parameters.get("sigmas", ""))
        if sigmas is not None:
            kwargs["sigmas"] = sigmas
        attention_text = str(parameters.get("attention_kwargs", "")).strip()
        if attention_text:
            parsed = json.loads(attention_text)
            if not isinstance(parsed, dict):
                raise ValueError("attention_kwargs must be a JSON object")
            kwargs["attention_kwargs"] = parsed
        if profile.provider == "diffusers.qwen_image_edit_plus":
            kwargs.update({
                "negative_prompt": str(parameters["negative_prompt"]),
                "true_cfg_scale": float(parameters["true_cfg_scale"]),
                "guidance_scale": float(parameters["guidance_scale"]),
                "max_sequence_length": int(
                    parameters["max_sequence_length"]),
            })
        elif profile_id == FLUX2_KLEIN_PROFILE_ID:
            kwargs.update({
                "guidance_scale": float(parameters["guidance_scale"]),
                "max_sequence_length": int(
                    parameters["max_sequence_length"]),
                "text_encoder_out_layers": parse_int_tuple(
                    parameters["text_encoder_out_layers"]),
            })
        else:
            kwargs.update({
                "guidance_scale": float(parameters["guidance_scale"]),
                "image_guidance_scale": float(
                    parameters["image_guidance_scale"]),
            })
        result = self._instruct_pipe(**kwargs).images[0]
        provenance = self._image_edit_provenance(
            profile_id, parameters, requested_adapters, seed, result.size)
        return result, seed, provenance.to_dict()

    # Compatibility wrappers for older callers of the worker implementation.
    def load_instruct(self, data: dict[str, Any]) -> dict[str, Any]:
        parameters = image_edit_profile(
            LEGACY_INSTRUCT_PROFILE_ID).defaults()
        parameters.update({
            "revision": data.get("revision") or "",
            "device": data.get("device") or parameters["device"],
        })
        return self.load_image_edit({
            "profile_id": LEGACY_INSTRUCT_PROFILE_ID,
            "parameters": parameters,
        })

    def instruct(self, data, image):
        parameters = image_edit_profile(
            LEGACY_INSTRUCT_PROFILE_ID).defaults()
        parameters.update({
            "prompt": data["instruction"],
            "guidance_scale": data["guidance_scale"],
            "image_guidance_scale": data["image_guidance_scale"],
            "steps": data["steps"],
            "seed": data["seed"],
        })
        return self.image_edit({
            "profile_id": LEGACY_INSTRUCT_PROFILE_ID,
            "parameters": parameters,
        }, image)

    @staticmethod
    def _package_version(name: str) -> str | None:
        try:
            from importlib.metadata import version
            return version(name)
        except Exception:
            return None

    def _diffusion_provenance(
            self,
            data: dict[str, Any],
            *,
            pipe,
            seed: int,
            width: int,
            height: int,
    ) -> GenerationProvenance:
        identity = self._diffusion_identity or ModelIdentity(
            provider="local",
            repository=(
                Path(self._diffusion_path).name
                if self._diffusion_path else None
            ),
            revision=None,
            content_hash=None,
            local_override=self._diffusion_path,
            status=ModelIdentityStatus.UNKNOWN,
            warning="Diffusion model identity was not resolved",
        )
        warnings = list(self._diffusion_warnings)
        runtime: dict[str, Any] = {
            "pipeline": type(pipe).__name__,
            "scheduler": type(pipe.scheduler).__name__,
            "device": self._diffusion_device,
            "dtype": self._diffusion_dtype,
            "torch_version": self._package_version("torch"),
            "diffusers_version": self._package_version("diffusers"),
            "transformers_version": self._package_version("transformers"),
        }
        if self._ip_adapter_identity is not None:
            runtime["ip_adapter_model"] = self._ip_adapter_identity.to_dict()
            if self._ip_adapter_identity.warning:
                warnings.append(self._ip_adapter_identity.warning)
        request = RequestProvenance.capture(
            "diffusion",
            {
                key: data.get(key)
                for key in (
                    "prompt",
                    "negative_prompt",
                    "strength",
                    "steps",
                    "guidance_scale",
                    "seed",
                    "mode",
                    "masked_content",
                    "ip_adapter_scale",
                    "width",
                    "height",
                )
            },
        )
        return GenerationProvenance(
            operation="diffusion",
            model=identity,
            request=request,
            seed=seed,
            width=width,
            height=height,
            runtime=FrozenJsonObject.capture(runtime),
            warnings=tuple(warnings),
        )

    def _image_edit_provenance(
            self,
            profile_id: str,
            parameters: dict[str, Any],
            lora_adapters: tuple[dict[str, Any], ...],
            seed: int,
            size: tuple[int, int],
    ) -> GenerationProvenance:
        identity = self._instruct_identity or floating_model_identity(
            "huggingface",
            "timbrooks/instruct-pix2pix",
        )
        operation = (
            "instruct"
            if profile_id == LEGACY_INSTRUCT_PROFILE_ID
            else "image_edit"
        )
        request = RequestProvenance.capture(operation, {
            "model_profile_id": profile_id,
            "parameters": parameters,
            "lora_adapters": list(lora_adapters),
        })
        return GenerationProvenance(
            operation=operation,
            model=identity,
            request=request,
            seed=seed,
            width=int(size[0]),
            height=int(size[1]),
            runtime=FrozenJsonObject.capture({
                "pipeline": type(self._instruct_pipe).__name__,
                "model_profile_id": profile_id,
                "scheduler": (
                    type(self._instruct_pipe.scheduler).__name__
                    if getattr(self._instruct_pipe, "scheduler", None)
                    is not None else None
                ),
                "device": self._instruct_device,
                "dtype": self._instruct_dtype,
                "torch_version": self._package_version("torch"),
                "diffusers_version": self._package_version("diffusers"),
                "transformers_version": self._package_version("transformers"),
                "sensenova_u1_version": self._package_version(
                    "sensenova-u1"),
                "gguf_version": self._package_version("gguf"),
                "provider_runtime": (
                    self._instruct_pipe.runtime_info
                    if hasattr(self._instruct_pipe, "runtime_info") else {}
                ),
            }),
            warnings=self._instruct_warnings,
        )

    def depth(
        self,
        data: dict[str, Any],
        image: Image.Image,
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        backend = str(data.get("backend", "transformers"))
        if backend == "da3":
            return self._depth_da3(data, image, progress)
        if backend != "transformers":
            raise ValueError(f"Unsupported depth backend: {backend}")
        return self._depth_transformers(data, image, progress)

    def _depth_transformers(
        self,
        data: dict[str, Any],
        image: Image.Image,
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        import torch
        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        device = self._device(data.get("device"))
        model_id = str(data["model_id"])
        title = str(data.get("title", model_id.split("/")[-1]))
        key = (f"transformers:{model_id}", device)
        if self._depth_key != key:
            self._prepare_depth_memory()
            cached = try_to_load_from_cache(model_id, "config.json") is not None
            progress(
                f"Loading {model_id.split('/')[-1]} "
                f"{'from cache' if cached else '(first run downloads weights)'}..."
            )
            old_model = self._depth_model
            self._depth_model = None
            self._depth_processor = None
            self._depth_key = None
            del old_model
            self._release_accelerator_memory()
            self._depth_processor = AutoImageProcessor.from_pretrained(model_id)
            self._depth_model = (
                AutoModelForDepthEstimation.from_pretrained(
                    model_id,
                    dtype=self._dtype(device),
                )
                .to(device)
                .eval()
            )
            self._depth_key = key

        progress(f"{title}: estimating depth...")
        processor = self._depth_processor
        model = self._depth_model
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.inference_mode():
            outputs = model(**inputs)
        processed = processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )
        processed_depth = processed[0]
        depth = (
            processed_depth["predicted_depth"]
            .detach()
            .float()
            .cpu()
            .numpy()
            .squeeze()
        )
        if depth.shape != (image.height, image.width):
            raise RuntimeError(
                "Depth Anything returned an unexpected shape: "
                f"{depth.shape}"
            )
        if not np.isfinite(depth).all():
            raise RuntimeError("Depth Anything returned non-finite values")
        depth = np.ascontiguousarray(depth, dtype=np.float32)

        field_of_view = self._optional_depth_scalar(
            processed_depth.get("field_of_view"))
        focal_length = self._optional_depth_scalar(
            processed_depth.get("focal_length"))
        intrinsics = None
        if focal_length is not None:
            intrinsics = np.array([
                [focal_length, 0.0, image.width * 0.5],
                [0.0, focal_length, image.height * 0.5],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32)
        return {
            "depth": depth,
            "intrinsics": intrinsics,
            "confidence": None,
            "field_of_view_degrees": field_of_view,
            "scale_factor": None,
            "value_kind": str(data["value_kind"]),
        }

    def _depth_da3(
        self,
        data: dict[str, Any],
        image: Image.Image,
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        import torch
        from huggingface_hub import try_to_load_from_cache

        source = os.environ.get("DIFFUSION_EDITOR_DA3_SOURCE", "").strip()
        if source:
            source_path = Path(source).expanduser()
        else:
            source_path = (
                Path(sys.prefix)
                / "share"
                / "diffusion-editor"
                / "depth-anything-3"
                / "src"
            )
        if not (source_path / "depth_anything_3" / "api.py").is_file():
            raise RuntimeError(
                "Depth Anything 3 runtime not found. Run ./setup-workers.sh "
                "or set DIFFUSION_EDITOR_DA3_SOURCE."
            )
        source_text = str(source_path)
        if source_text not in sys.path:
            sys.path.insert(0, source_text)
        os.environ.setdefault(
            "MPLCONFIGDIR",
            str(Path(tempfile.gettempdir()) / "diffusion-editor-matplotlib"),
        )
        try:
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise RuntimeError(
                "Depth Anything 3 dependencies are incomplete. "
                "Run ./setup-workers.sh."
            ) from exc

        device = self._device(data.get("device"))
        model_id = str(data["model_id"])
        title = str(data.get("title", "DA3 Mono Large"))
        key = (f"da3:{model_id}", device)
        if self._depth_key != key:
            self._prepare_depth_memory()
            cached = try_to_load_from_cache(
                model_id, "model.safetensors") is not None
            progress(
                f"Loading {model_id.split('/')[-1]} "
                f"{'from cache' if cached else '(first run downloads weights)'}..."
            )
            old_model = self._depth_model
            self._depth_model = None
            self._depth_processor = None
            self._depth_key = None
            del old_model
            self._release_accelerator_memory()
            self._depth_model = (
                DepthAnything3.from_pretrained(model_id)
                .to(device)
                .eval()
            )
            self._depth_key = key

        resolution = int(data.get("process_resolution") or 504)
        progress(f"{title}: estimating depth at {resolution}px...")
        prediction = self._depth_model.inference(
            [image.convert("RGB")],
            process_res=resolution,
            use_ray_pose=bool(data.get("use_ray_pose", False)),
        )
        depth = np.asarray(prediction.depth[0], dtype=np.float32)
        if not np.isfinite(depth).all():
            raise RuntimeError("Depth Anything 3 returned non-finite values")
        intrinsics = (
            np.asarray(prediction.intrinsics[0], dtype=np.float32)
            if prediction.intrinsics is not None else None
        )
        confidence = (
            np.asarray(prediction.conf[0], dtype=np.float32)
            if prediction.conf is not None else None
        )
        if confidence is not None and not np.isfinite(confidence).all():
            raise RuntimeError(
                "Depth Anything 3 confidence contains non-finite values")
        if intrinsics is not None and (
                intrinsics.shape != (3, 3)
                or not np.isfinite(intrinsics).all()):
            raise RuntimeError(
                "Depth Anything 3 returned invalid camera intrinsics")
        scale_factor = self._optional_depth_scalar(prediction.scale_factor)
        return {
            "depth": np.ascontiguousarray(depth, dtype=np.float32),
            "intrinsics": (
                np.ascontiguousarray(intrinsics, dtype=np.float32)
                if intrinsics is not None else None
            ),
            "confidence": (
                np.ascontiguousarray(confidence, dtype=np.float32)
                if confidence is not None else None
            ),
            "field_of_view_degrees": None,
            "scale_factor": scale_factor,
            "value_kind": (
                "direct_metric"
                if bool(prediction.is_metric)
                else "direct_scale_ambiguous"
            ),
        }

    @staticmethod
    def _optional_depth_scalar(value: Any) -> float | None:
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().item()
        result = float(value)
        if not np.isfinite(result):
            raise RuntimeError("Depth model returned non-finite camera metadata")
        return result

    def grounding(
        self,
        data: dict[str, Any],
        image: Image.Image,
        progress: Callable[[str], None],
    ) -> list[dict[str, Any]]:
        import torch
        from huggingface_hub import try_to_load_from_cache
        from transformers import (
            GroundingDinoForObjectDetection,
            GroundingDinoProcessor,
        )

        device = self._device("cuda" if data["use_gpu"] else "cpu")
        model_id = str(data["model_id"])
        key = (model_id, device)
        if self._dino_key != key:
            self._unload_depth()
            cached = try_to_load_from_cache(model_id, "model.safetensors") is not None
            progress(
                f"Loading {model_id.split('/')[-1]} "
                f"{'from cache' if cached else '(download may take a few minutes)'}..."
            )
            self._dino_processor = GroundingDinoProcessor.from_pretrained(
                model_id, local_files_only=cached
            )
            self._dino_model = GroundingDinoForObjectDetection.from_pretrained(
                model_id, local_files_only=cached
            ).to(device).eval()
            self._dino_key = key
        progress("Grounding: detecting...")
        processor = self._dino_processor
        model = self._dino_model
        inputs = processor(
            images=image.convert("RGB"),
            text=str(data["prompt"]),
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        processed = processor.post_process_grounded_object_detection(
            outputs,
            input_ids=inputs.get("input_ids"),
            threshold=float(data["box_threshold"]),
            text_threshold=float(data["text_threshold"]),
            target_sizes=[(image.height, image.width)],
        )
        detections = []
        for result in processed:
            for score, box, label in zip(
                result["scores"].tolist(),
                result["boxes"].tolist(),
                result["labels"],
            ):
                x0, y0, x1, y1 = map(int, box)
                detections.append(
                    {
                        "label": label,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "score": float(score),
                        "mask": None,
                    }
                )
        detections.sort(key=lambda item: item["score"], reverse=True)
        if detections and data.get("sam2_model_id"):
            progress("SAM 2.1: segmenting...")
            masks = self._sam_masks(data, image, detections, device)
            for detection, mask in zip(detections, masks):
                detection["mask"] = mask
        return detections

    def _sam_masks(
        self,
        data: dict[str, Any],
        image: Image.Image,
        detections: list[dict[str, Any]],
        device: str,
    ) -> list[np.ndarray]:
        import torch
        from huggingface_hub import try_to_load_from_cache
        from transformers import Sam2Model, Sam2Processor

        model_id = str(data["sam2_model_id"])
        key = (model_id, device)
        if self._sam_key != key:
            self._unload_depth()
            cached = try_to_load_from_cache(model_id, "model.safetensors") is not None
            self._sam_processor = Sam2Processor.from_pretrained(
                model_id, local_files_only=cached
            )
            self._sam_model = Sam2Model.from_pretrained(
                model_id, local_files_only=cached
            ).to(device).eval()
            self._sam_key = key
        boxes = [[[
            item["x0"], item["y0"], item["x1"], item["y1"]
        ] for item in detections]]
        inputs = self._sam_processor(
            images=image.convert("RGB"),
            input_boxes=boxes,
            return_tensors="pt",
        )
        inputs = {name: value.to(device) for name, value in inputs.items()}
        with torch.no_grad():
            outputs = self._sam_model(
                **inputs, multimask_output=bool(data["multimask"])
            )
        masks = self._sam_processor.post_process_masks(
            outputs.pred_masks,
            original_sizes=[(image.height, image.width)],
            mask_threshold=float(data["mask_threshold"]),
            max_hole_area=int(data["max_hole_area"]),
            max_sprinkle_area=int(data["max_sprinkle_area"]),
            apply_non_overlapping_constraints=bool(data["non_overlap"]),
            binarize=True,
        )[0]
        result = []
        for mask in masks:
            if bool(data["multimask"]) and mask.dim() > 2:
                mask = mask[int(data["sam2_mask_channel"])]
            array = np.squeeze(mask.cpu().numpy()).astype(bool)
            if array.shape != (image.height, image.width):
                raise RuntimeError(
                    "SAM returned an unexpected mask shape: "
                    f"{array.shape}, expected {(image.height, image.width)}"
                )
            result.append(array)
        return result
