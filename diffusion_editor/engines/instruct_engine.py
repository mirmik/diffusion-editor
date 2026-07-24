"""Main-process facade for InstructPix2Pix inference."""

from __future__ import annotations

from tcbase import log

from ..generation.types import InstructInferenceResult, InstructRequest
from ..workers.ml_process import MlProcessClient
from .threaded_lifecycle import EngineTaskQueue


class InstructEngine:
    def __init__(self, client: MlProcessClient | None = None):
        self._client = client or MlProcessClient()
        self._tasks = EngineTaskQueue()
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._client.is_running

    @property
    def is_busy(self) -> bool:
        return self._tasks.is_busy

    def submit_load(self):
        return self._tasks.submit(
            "load",
            lambda cancel: self._load(cancel),
            name="instruct-load",
            on_error=lambda _exc: log.exception(
                "InstructPix2Pix load failed"
            ),
        )

    def _load(self, cancel):
        self._client.request("load_instruct", {}, cancel)
        self._loaded = True
        return True

    def submit_request(self, request: InstructRequest, meta=None):
        return self._tasks.submit(
            "inference",
            lambda cancel: self._run_inference(request, cancel),
            meta=meta,
            name="instruct-inference",
            on_error=lambda _exc: log.exception(
                "InstructPix2Pix inference failed"
            ),
        )

    def _run_inference(self, request: InstructRequest, cancel):
        result = self._client.request(
            "instruct",
            {
                "instruction": request.instruction,
                "guidance_scale": request.guidance_scale,
                "image_guidance_scale": request.image_guidance_scale,
                "steps": request.steps,
                "seed": request.seed,
            },
            cancel,
            images={"image": request.image},
        )
        return InstructInferenceResult(
            image=result["image"],
            seed=int(result["seed"]),
        )

    def poll_event(self):
        return self._tasks.poll_event()

    def cancel(self) -> bool:
        return self._tasks.cancel()

    def shutdown(self, timeout: float = 1.0):
        if self._tasks.shutdown(timeout):
            self._client.shutdown(timeout)
            self._loaded = False
