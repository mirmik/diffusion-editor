import numpy as np
from tcbase import log

from ..generation.types import SegmentationRequest, SegmentationResult
from ..workers.segmentation_process import SegmentationProcessClient
from .threaded_lifecycle import EngineTaskQueue


class SegmentationEngine:
    def __init__(self, client: SegmentationProcessClient | None = None):
        self._client = client or SegmentationProcessClient()
        self._tasks = EngineTaskQueue()

    @property
    def is_loaded(self) -> bool:
        return self._client.is_running

    @property
    def is_busy(self) -> bool:
        return self._tasks.is_busy

    def submit_request(self, request: SegmentationRequest):
        return self._tasks.submit(
            "segmentation",
            lambda cancel: self._run(
                request.image,
                request.invert,
                cancel,
            ),
            name="segmentation-inference",
            on_error=lambda _exc: log.exception("Segmentation failed"),
        )

    def _run(self, image_arr, invert, cancel):
        log.debug("[Segmentation] running isolated inference...")
        fg_mask = self._client.segment(
            image_arr,
            cancel,
            on_progress=lambda message: log.debug(
                f"[Segmentation] {message}"
            ),
        )
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

    def shutdown(self, timeout: float = 1.0):
        if self._tasks.shutdown(timeout):
            self._client.shutdown(timeout)
