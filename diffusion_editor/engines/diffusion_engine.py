"""Main-process facade for Stable Diffusion inference."""

from __future__ import annotations

from tcbase import log

from ..generation.types import DiffusionInferenceResult, DiffusionRequest
from ..workers.ml_process import MlProcessClient
from .threaded_lifecycle import EngineTaskQueue


class DiffusionEngine:
    def __init__(self, client: MlProcessClient | None = None):
        self._client = client or MlProcessClient()
        self._tasks = EngineTaskQueue()
        self._model_path: str | None = None
        self._ip_adapter_loaded = False
        self.model_info: dict = {}

    @property
    def is_loaded(self) -> bool:
        return self._model_path is not None and self._client.is_running

    @property
    def is_busy(self) -> bool:
        return self._tasks.is_busy

    @property
    def model_path(self) -> str | None:
        return self._model_path

    @property
    def ip_adapter_loaded(self) -> bool:
        return self._ip_adapter_loaded

    def submit_load(self, path: str, prediction_type: str | None = None):
        return self._tasks.submit(
            "load",
            lambda cancel: self._load(path, prediction_type, cancel),
            meta=path,
            name="diffusion-model-load",
            on_error=lambda _exc: log.exception(f"Model load failed: {path}"),
        )

    def _load(self, path, prediction_type, cancel):
        self._clear_state()
        result = self._client.request(
            "load_diffusion",
            {"model_path": path, "prediction_type": prediction_type},
            cancel,
        )
        self._model_path = str(result["model_path"])
        self.model_info = dict(result.get("model_info", {}))
        self._ip_adapter_loaded = False
        return self._model_path

    def submit_load_ip_adapter(self):
        if not self.is_loaded:
            return False
        return self._tasks.submit(
            "load_ip_adapter",
            lambda cancel: self._load_ip_adapter(cancel),
            name="diffusion-ip-adapter-load",
            on_error=lambda _exc: log.exception("IP-Adapter load failed"),
        )

    def _load_ip_adapter(self, cancel):
        try:
            self._client.request("load_ip_adapter", {}, cancel)
        except Exception:
            if not self._client.is_running:
                self._clear_state()
            raise
        self._ip_adapter_loaded = True
        return True

    def submit_request(self, request: DiffusionRequest, meta=None):
        return self._tasks.submit(
            "inference",
            lambda cancel: self._run_inference(request, cancel),
            meta=meta,
            name="diffusion-inference",
            on_error=lambda _exc: log.exception(
                f"Diffusion inference failed (mode={request.mode})"
            ),
        )

    def _run_inference(self, request: DiffusionRequest, cancel):
        try:
            result = self._client.request(
                "diffusion",
                {
                    "prompt": request.prompt,
                    "negative_prompt": request.negative_prompt,
                    "strength": request.strength,
                    "steps": request.steps,
                    "guidance_scale": request.guidance_scale,
                    "seed": request.seed,
                    "mode": request.mode,
                    "masked_content": request.masked_content,
                    "ip_adapter_scale": request.ip_adapter_scale,
                    "width": request.width,
                    "height": request.height,
                },
                cancel,
                images={
                    "image": request.image,
                    "mask": request.mask_image,
                    "ip_adapter": request.ip_adapter_image,
                },
            )
        except Exception:
            if not self._client.is_running:
                self._clear_state()
            raise
        return DiffusionInferenceResult(
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
            self._clear_state()

    def _clear_state(self) -> None:
        self._model_path = None
        self._ip_adapter_loaded = False
        self.model_info = {}
