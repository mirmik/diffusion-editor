from PIL import Image
from tcbase import log

from ..generation.types import LamaRequest, LamaResult
from .threaded_lifecycle import EngineTaskQueue
from ..workers.lama_process import LamaProcessClient


class LamaEngine:
    def __init__(self, client: LamaProcessClient | None = None):
        self._client = client or LamaProcessClient()
        self._tasks = EngineTaskQueue()

    @property
    def is_busy(self) -> bool:
        return self._tasks.is_busy

    def submit_request(self, request: LamaRequest):
        return self._tasks.submit(
            "inference",
            lambda cancel: self._run(
                request.image,
                request.mask_image,
                cancel,
            ),
            name="lama-inference",
            on_error=lambda _exc: log.exception("LaMa inference failed"),
        )

    def _run(self, image: Image.Image, mask: Image.Image, cancel):
        log.debug("[LamaEngine] running isolated inference...")
        result = self._client.inpaint(image, mask, cancel)
        log.debug(f"[LamaEngine] done, result size: {result.size}")
        return LamaResult(image=result)

    def poll_event(self):
        return self._tasks.poll_event()

    def cancel(self) -> bool:
        return self._tasks.cancel()

    def shutdown(self, timeout: float = 1.0):
        if self._tasks.shutdown(timeout):
            self._client.shutdown(timeout)
