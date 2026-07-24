import numpy as np
from PIL import Image
from tcbase import log

from ..generation.types import SegmentationRequest, SegmentationResult
from .threaded_lifecycle import EngineTaskQueue


class SegmentationEngine:
    def __init__(self):
        self._session = None
        self._tasks = EngineTaskQueue()

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def is_busy(self) -> bool:
        return self._tasks.is_busy

    def _ensure_loaded(self):
        """Lazy-load сессии при первом вызове."""
        if self._session is None:
            from rembg import new_session
            self._session = new_session("u2net")

    def submit_request(self, request: SegmentationRequest):
        return self._tasks.submit(
            "segmentation",
            lambda _cancel: self._run(request.image, request.invert),
            name="segmentation-inference",
            on_error=lambda _exc: log.exception("Segmentation failed"),
        )

    def _run(self, image_arr, invert):
        from rembg import remove
        log.debug("[Segmentation] loading model...")
        self._ensure_loaded()
        log.debug("[Segmentation] model loaded, running inference...")
        pil_img = Image.fromarray(image_arr[:, :, :3])
        fg_mask_pil = remove(pil_img, session=self._session, only_mask=True)
        fg_mask = np.array(fg_mask_pil, dtype=np.uint8)
        log.debug(
            f"[Segmentation] fg_mask shape={fg_mask.shape}, "
            f"min={fg_mask.min()}, max={fg_mask.max()}"
        )
        mask = 255 - fg_mask if invert else fg_mask
        result = mask.astype(np.uint8)
        log.debug(f"[Segmentation] done, result shape={result.shape}")
        return SegmentationResult(mask=result)

    def poll_event(self):
        return self._tasks.poll_event()

    def cancel(self) -> bool:
        return self._tasks.cancel()

    def unload(self):
        self._session = None

    def shutdown(self, timeout: float = 1.0):
        if self._tasks.shutdown(timeout):
            self.unload()
