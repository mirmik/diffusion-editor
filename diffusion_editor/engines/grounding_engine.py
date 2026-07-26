"""Main-process facade for Grounding DINO and SAM inference."""

from __future__ import annotations

import threading

from tcbase import log

from ..grounding.types import (
    GroundingDetection,
    GroundingEngineEvent,
    GroundingRequest,
    GroundingResult,
)
from ..workers.ml_process import MlProcessClient
from .threaded_lifecycle import SingleWorkerEventQueue


class GroundingEngine:
    def __init__(self, client: MlProcessClient | None = None):
        self._client = client or MlProcessClient()
        self._tasks: SingleWorkerEventQueue[GroundingEngineEvent] = (
            SingleWorkerEventQueue()
        )

    @property
    def is_busy(self) -> bool:
        return self._tasks.is_busy

    def gpu_available(self) -> bool:
        try:
            result = self._client.request(
                "gpu_available",
                {},
                threading.Event(),
            )
            return bool(result["available"])
        except Exception as exc:
            log.error(f"Grounding: failed to query GPU availability: {exc}")
            return False

    def submit_request(self, request: GroundingRequest) -> bool:
        return self._tasks.submit(
            lambda cancel: self._run(request, cancel),
            name="grounding-inference",
        )

    def poll_event(self) -> GroundingEngineEvent | None:
        return self._tasks.poll_event()

    def _status(self, message: str) -> None:
        self._tasks.emit(GroundingEngineEvent(status=message))

    def _run(
        self,
        request: GroundingRequest,
        cancel,
    ) -> GroundingEngineEvent:
        params = request.params
        try:
            result = self._client.request(
                "grounding",
                {
                    "prompt": params.prompt,
                    "model_id": params.model_id,
                    "box_threshold": params.box_threshold,
                    "text_threshold": params.text_threshold,
                    "use_gpu": params.use_gpu,
                    "sam2_model_id": params.sam2_model_id,
                    "sam2_mask_channel": params.sam2_mask_channel,
                    "mask_threshold": params.mask_threshold,
                    "max_hole_area": params.max_hole_area,
                    "max_sprinkle_area": params.max_sprinkle_area,
                    "multimask": params.multimask,
                    "non_overlap": params.non_overlap,
                },
                cancel,
                images={"image": request.image},
                on_progress=self._status,
            )
            detections = tuple(
                GroundingDetection(
                    label=str(item["label"]),
                    x0=int(item["x0"]),
                    y0=int(item["y0"]),
                    x1=int(item["x1"]),
                    y1=int(item["y1"]),
                    score=float(item["score"]),
                    mask=item["mask"],
                )
                for item in result["detections"]
            )
            if not detections:
                return GroundingEngineEvent(
                    status="Grounding: nothing found"
                )
            found = ", ".join(
                f"{item.label} ({item.score:.0%})" for item in detections
            )
            return GroundingEngineEvent(
                status=f"Grounding: {len(detections)} hit(s): {found}",
                result=GroundingResult(
                    canvas_width=int(request.image.shape[1]),
                    canvas_height=int(request.image.shape[0]),
                    detections=detections,
                ),
            )
        except Exception as exc:
            log.error(f"Grounding: {exc}")
            return GroundingEngineEvent(error=str(exc))

    def cancel(self) -> bool:
        return self._tasks.cancel()

    def shutdown(self, timeout: float = 1.0) -> None:
        if self._tasks.shutdown(timeout):
            self._client.shutdown(timeout)
