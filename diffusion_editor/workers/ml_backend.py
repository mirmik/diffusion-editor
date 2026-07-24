"""Worker-only Diffusers and Transformers model implementation."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageFilter


_VPRED_HINTS = (
    "vpred", "v-pred", "v_pred", "vprediction", "v-prediction", "v_prediction"
)


class RealMlBackend:
    """Persistent model state. This module is imported only by the worker."""

    def __init__(self) -> None:
        self._diffusion_pipe = None
        self._diffusion_mode: str | None = None
        self._diffusion_path: str | None = None
        self._ip_adapter_loaded = False
        self._instruct_pipe = None
        self._dino_model = None
        self._dino_processor = None
        self._dino_key: tuple[str, str] | None = None
        self._sam_model = None
        self._sam_processor = None
        self._sam_key: tuple[str, str] | None = None

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

    def load_diffusion(self, data: dict[str, Any]) -> dict[str, Any]:
        import torch
        from diffusers import (
            DPMSolverMultistepScheduler,
            StableDiffusionXLPipeline,
        )

        self.unload_diffusion()
        model_path = str(data["model_path"])
        device = self._device(data.get("device"))
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
            },
        }

    def unload_diffusion(self) -> None:
        self._diffusion_pipe = None
        self._diffusion_path = None
        self._diffusion_mode = None
        self._ip_adapter_loaded = False
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def load_ip_adapter(self) -> dict[str, Any]:
        if self._diffusion_pipe is None:
            raise RuntimeError("No diffusion model loaded")
        self._diffusion_pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin",
        )
        self._ip_adapter_loaded = True
        return {"loaded": True}

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
    ) -> tuple[Image.Image, int]:
        import torch

        mode = str(data["mode"])
        self._ensure_diffusion_mode(mode)
        pipe = self._diffusion_pipe
        assert pipe is not None
        seed = int(data["seed"])
        if seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())
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
        return pipe(**kwargs).images[0], seed

    @staticmethod
    def _prepare_masked_content(
        image: Image.Image,
        mask: Image.Image | None,
        mode: str,
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
            replacement = np.random.randint(
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

    def load_instruct(self, data: dict[str, Any]) -> dict[str, Any]:
        from diffusers import (
            EulerAncestralDiscreteScheduler,
            StableDiffusionInstructPix2PixPipeline,
        )

        device = self._device(data.get("device"))
        pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=self._dtype(device),
            safety_checker=None,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config
        )
        pipe.to(device)
        self._instruct_pipe = pipe
        return {"loaded": True, "device": device}

    def instruct(
        self,
        data: dict[str, Any],
        image: Image.Image,
    ) -> tuple[Image.Image, int]:
        import torch

        if self._instruct_pipe is None:
            raise RuntimeError("InstructPix2Pix model not loaded")
        seed = int(data["seed"])
        if seed == -1:
            seed = int(torch.randint(0, 2**32, (1,)).item())
        generator = torch.Generator(device="cpu").manual_seed(seed)
        result = self._instruct_pipe(
            prompt=str(data["instruction"]),
            image=image.convert("RGB"),
            num_inference_steps=int(data["steps"]),
            guidance_scale=float(data["guidance_scale"]),
            image_guidance_scale=float(data["image_guidance_scale"]),
            generator=generator,
        ).images[0]
        return result, seed

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
