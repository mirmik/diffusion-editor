from PIL import Image
from tcbase import log

from ..generation.types import LamaRequest, LamaResult
from .threaded_lifecycle import EngineTaskQueue


class LamaEngine:
    def __init__(self):
        self._model = None
        self._tasks = EngineTaskQueue()

    @property
    def is_busy(self) -> bool:
        return self._tasks.is_busy

    def _ensure_loaded(self):
        if self._model is None:
            from simple_lama_inpainting import SimpleLama
            self._model = SimpleLama()

    def submit_request(self, request: LamaRequest):
        return self._tasks.submit(
            "inference",
            lambda _cancel: self._run(request.image, request.mask_image),
            name="lama-inference",
            on_error=lambda _exc: log.exception("LaMa inference failed"),
        )

    def _run(self, image: Image.Image, mask: Image.Image):
        log.debug("[LamaEngine] loading model...")
        self._ensure_loaded()
        log.debug("[LamaEngine] running inference...")
        image = image.convert("RGB")
        mask = mask.convert("L")
        result = self._model(image, mask)
        log.debug(f"[LamaEngine] done, result size: {result.size}")
        return LamaResult(image=result)

    def poll_event(self):
        return self._tasks.poll_event()

    def cancel(self) -> bool:
        return self._tasks.cancel()

    def unload(self):
        self._model = None

    def shutdown(self, timeout: float = 1.0):
        if self._tasks.shutdown(timeout):
            self.unload()
