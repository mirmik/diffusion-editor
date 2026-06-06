from threading import Thread
from PIL import Image
from tcbase import log

from ..generation.types import EnginePollEvent, LamaRequest, LamaResult


class LamaEngine:
    def __init__(self):
        self._model = None
        self._busy = False
        self._result = None   # PIL.Image or None
        self._error = None
        self._thread = None

    @property
    def is_busy(self) -> bool:
        return self._busy

    def _ensure_loaded(self):
        if self._model is None:
            from simple_lama_inpainting import SimpleLama
            self._model = SimpleLama()

    def submit_request(self, request: LamaRequest):
        if self._busy:
            return False
        self._busy = True
        self._result = None
        self._error = None
        self._thread = Thread(
            target=self._run,
            args=(request.image, request.mask_image),
            daemon=True,
        )
        self._thread.start()
        return True

    def _run(self, image: Image.Image, mask: Image.Image):
        try:
            log.debug("[LamaEngine] loading model...")
            self._ensure_loaded()
            log.debug("[LamaEngine] running inference...")
            image = image.convert("RGB")
            mask = mask.convert("L")
            result = self._model(image, mask)
            self._result = result
            log.debug(f"[LamaEngine] done, result size: {result.size}")
        except Exception as e:
            log.exception("LaMa inference failed")
            self._error = str(e)
        self._busy = False

    def poll_event(self) -> EnginePollEvent | None:
        if self._busy:
            return None
        result, error = self._result, self._error
        if result is None and error is None:
            return None
        self._result = None
        self._error = None
        event_result = None
        if result is not None:
            event_result = LamaResult(image=result)
        return EnginePollEvent(
            task_type="inference",
            result=event_result,
            error=error,
        )

    def unload(self):
        self._model = None

    def shutdown(self, timeout: float = 1.0):
        """Best-effort engine shutdown for app exit."""
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
        self.unload()
