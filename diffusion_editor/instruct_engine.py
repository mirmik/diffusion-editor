import threading
import torch
from PIL import Image


class InstructEngine:
    def __init__(self):
        self._pipe = None
        self._busy = False
        self._result = None
        self._error = None
        self._result_meta = None
        self._task_type = None  # "inference" or "load"
        self._thread = None

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    @property
    def is_busy(self) -> bool:
        return self._busy

    def load_model(self):
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
        print("[InstructEngine] Model loaded: timbrooks/instruct-pix2pix")

    def unload(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()

    def submit_load(self):
        if self._busy:
            return False
        self._busy = True
        self._result = None
        self._error = None
        self._result_meta = None
        self._task_type = "load"
        self._thread = threading.Thread(target=self._run_load, daemon=True)
        self._thread.start()
        return True

    def _run_load(self):
        try:
            self.load_model()
            self._result = True
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._error = str(e)
        self._busy = False

    def submit(self, image: Image.Image, instruction: str,
               guidance_scale: float = 7.0,
               image_guidance_scale: float = 1.5,
               steps: int = 20, seed: int = -1,
               meta=None):
        if self._busy:
            return False
        self._busy = True
        self._result = None
        self._error = None
        self._result_meta = meta
        self._task_type = "inference"
        self._thread = threading.Thread(
            target=self._run_inference,
            args=(image, instruction, guidance_scale, image_guidance_scale,
                  steps, seed),
            daemon=True,
        )
        self._thread.start()
        return True

    def _run_inference(self, image, instruction, guidance_scale,
                       image_guidance_scale, steps, seed):
        try:
            if self._pipe is None:
                raise RuntimeError("Model not loaded")

            image = image.convert("RGB")

            if seed == -1:
                seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device="cpu").manual_seed(seed)

            print(f"[InstructEngine] params:")
            print(f"  instruction:          {instruction!r}")
            print(f"  image size:           {image.size}")
            print(f"  guidance_scale:       {guidance_scale}")
            print(f"  image_guidance_scale: {image_guidance_scale}")
            print(f"  steps:                {steps}")
            print(f"  seed:                 {seed}")

            result = self._pipe(
                prompt=instruction,
                image=image,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                image_guidance_scale=image_guidance_scale,
                generator=generator,
            ).images[0]

            print(f"[InstructEngine] done, result size: {result.size}")
            self._result = (result, seed)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._error = str(e)
        self._busy = False

    def poll(self):
        """Returns (task_type, result, error, meta)."""
        if self._busy:
            return None, None, None, None

        task_type = self._task_type
        result = self._result
        error = self._error
        meta = self._result_meta

        if result is None and error is None:
            return None, None, None, None

        self._result = None
        self._error = None
        self._result_meta = None
        self._task_type = None
        return task_type, result, error, meta
