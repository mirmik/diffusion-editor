import os
import threading
import torch
from PIL import Image
from diffusers import StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler

# Filename hints for v-prediction models
_VPRED_HINTS = ("vpred", "v-pred", "v_pred", "vprediction", "v-prediction", "v_prediction")


def _guess_prediction_type(path: str) -> str | None:
    """Guess prediction type from filename. Returns None if unknown."""
    name = os.path.basename(path).lower()
    for hint in _VPRED_HINTS:
        if hint in name:
            return "v_prediction"
    return None


class DiffusionEngine:
    def __init__(self):
        self._pipe = None
        self._model_path = None
        self._busy = False
        self._result = None
        self._error = None
        self._result_meta = None
        self._task_type = None  # "inference" or "load"
        self._thread = None
        self.model_info = {}  # диагностика — заполняется после загрузки

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    @property
    def is_busy(self) -> bool:
        return self._busy

    def load_model(self, safetensors_path: str, prediction_type: str | None = None):
        self.unload()

        guessed = _guess_prediction_type(safetensors_path)
        chosen_prediction = prediction_type or guessed or "epsilon"

        # DPM++ 2M SDE Karras — как в A1111/Forge
        scheduler = DPMSolverMultistepScheduler(
            prediction_type=chosen_prediction,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )

        self._pipe = StableDiffusionXLImg2ImgPipeline.from_single_file(
            safetensors_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            scheduler=scheduler,
        )
        self._pipe.to("cuda")
        self._model_path = safetensors_path

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
        print(f"[DiffusionEngine] Loaded: {self.model_info}")

    def unload(self):
        if self._pipe is not None:
            del self._pipe
            self._pipe = None
            torch.cuda.empty_cache()
        self._model_path = None

    def _img2img(self, image: Image.Image, prompt: str, negative_prompt: str,
                 strength: float, num_inference_steps: int,
                 guidance_scale: float) -> Image.Image:
        if self._pipe is None:
            raise RuntimeError("No model loaded")

        image = image.convert("RGB")

        # VAE работает с размерами кратными 8
        w, h = image.size
        w8 = (w // 8) * 8
        h8 = (h // 8) * 8
        if (w8, h8) != (w, h):
            image = image.resize((w8, h8), Image.LANCZOS)

        result = self._pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        ).images[0]

        return result

    def submit(self, image: Image.Image, prompt: str, negative_prompt: str,
               strength: float, steps: int, guidance_scale: float, meta=None):
        if self._busy:
            return False
        self._busy = True
        self._result = None
        self._error = None
        self._result_meta = meta
        self._task_type = "inference"
        self._thread = threading.Thread(
            target=self._run_inference,
            args=(image, prompt, negative_prompt, strength, steps, guidance_scale),
            daemon=True,
        )
        self._thread.start()
        return True

    def _run_inference(self, image, prompt, negative_prompt, strength, steps, guidance_scale):
        try:
            result = self._img2img(image, prompt, negative_prompt,
                                   strength, steps, guidance_scale)
            self._result = result
        except Exception as e:
            self._error = str(e)
        self._busy = False

    def submit_load(self, path: str, prediction_type: str | None = None):
        if self._busy:
            return False
        self._busy = True
        self._result = None
        self._error = None
        self._result_meta = path
        self._task_type = "load"
        self._thread = threading.Thread(
            target=self._run_load, args=(path, prediction_type), daemon=True,
        )
        self._thread.start()
        return True

    def _run_load(self, path, prediction_type):
        try:
            self.load_model(path, prediction_type)
            self._result = path
        except Exception as e:
            self._error = str(e)
        self._busy = False

    def poll(self):
        """Check if background task is done.

        Returns (task_type, result_or_none, error_or_none, meta).
        Returns (None, None, None, None) if still busy or no pending result.
        """
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
