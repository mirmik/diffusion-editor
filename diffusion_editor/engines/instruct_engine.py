from PIL import Image
from tcbase import log

from ..generation.types import (
    InstructInferenceResult,
    InstructRequest,
)
from .threaded_lifecycle import EngineTaskQueue


def _import_torch():
    import torch
    return torch


class InstructEngine:
    def __init__(self):
        self._pipe = None
        self._tasks = EngineTaskQueue()

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    @property
    def is_busy(self) -> bool:
        return self._tasks.is_busy

    def load_model(self):
        torch = _import_torch()
        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler
        self._pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "timbrooks/instruct-pix2pix",
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self._pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self._pipe.scheduler.config
        )
        self._pipe.to("cuda")
        log.info("[InstructEngine] Model loaded: timbrooks/instruct-pix2pix")

    def unload(self):
        if self._pipe is not None:
            torch = _import_torch()
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()

    def submit_load(self):
        return self._tasks.submit(
            "load",
            lambda _cancel: self._run_load(),
            name="instruct-load",
            on_error=lambda _exc: log.exception("InstructPix2Pix load failed"),
        )

    def _run_load(self):
        self.load_model()
        return True

    def submit_request(self, request: InstructRequest, meta=None):
        return self._tasks.submit(
            "inference",
            lambda _cancel: self._run_inference(
                request.image,
                request.instruction,
                request.guidance_scale,
                request.image_guidance_scale,
                request.steps,
                request.seed,
            ),
            meta=meta,
            name="instruct-inference",
            on_error=lambda _exc: log.exception(
                "InstructPix2Pix inference failed"
            ),
        )

    def _run_inference(self, image, instruction, guidance_scale,
                       image_guidance_scale, steps, seed):
        if self._pipe is None:
            raise RuntimeError("Model not loaded")

        image = image.convert("RGB")

        torch = _import_torch()
        if seed == -1:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        log.debug(
            "[InstructEngine] instruction=%r image_size=%s guidance_scale=%s image_guidance_scale=%s steps=%s seed=%s"
            % (instruction, image.size, guidance_scale, image_guidance_scale, steps, seed)
        )

        result = self._pipe(
            prompt=instruction,
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            image_guidance_scale=image_guidance_scale,
            generator=generator,
        ).images[0]

        log.debug(f"[InstructEngine] done, result size: {result.size}")
        return InstructInferenceResult(image=result, seed=seed)

    def poll_event(self):
        return self._tasks.poll_event()

    def cancel(self) -> bool:
        return self._tasks.cancel()

    def shutdown(self, timeout: float = 1.0):
        if self._tasks.shutdown(timeout):
            self.unload()
