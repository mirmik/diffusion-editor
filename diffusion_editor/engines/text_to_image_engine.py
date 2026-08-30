"""Main-process facade for provider-neutral text-to-image inference."""

from __future__ import annotations

from tcbase import log

from ..generation.provenance import GenerationProvenance
from ..generation.text_to_image_profiles import text_to_image_profile
from ..generation.types import (
    TextToImageInferenceResult,
    TextToImageRequest,
)
from ..workers.ml_process import MlProcessClient
from .threaded_lifecycle import EngineTaskQueue


class TextToImageEngine:
    supports_job_ids = True

    def __init__(self, client: MlProcessClient | None = None):
        self._client = client or MlProcessClient()
        self._tasks = EngineTaskQueue()
        self._loaded = False
        self._loaded_profile_id: str | None = None
        self._loaded_parameters: dict[str, object] = {}
        self._loaded_lora_adapters: tuple[dict[str, object], ...] = ()
        self.model_info: dict = {}

    @property
    def is_loaded(self) -> bool:
        return self._loaded and self._client.is_running

    @property
    def is_busy(self) -> bool:
        return self._tasks.is_busy

    @property
    def loaded_profile_id(self) -> str | None:
        return self._loaded_profile_id if self.is_loaded else None

    def loaded_configuration_matches(
            self, profile_id: str, parameters: dict[str, object],
            lora_adapters=None) -> bool:
        if self.loaded_profile_id != profile_id:
            return False
        profile = text_to_image_profile(profile_id)
        adapters = tuple(
            adapter.to_dict()
            for adapter in profile.normalize_lora_adapters(lora_adapters)
        )
        return (
            self._loaded_parameters == profile.load_values(parameters)
            and self._loaded_lora_adapters == adapters
        )

    def submit_load(
            self, profile_id: str, parameters: dict[str, object] | None = None,
            lora_adapters=None, *, job_id: str | None = None):
        profile = text_to_image_profile(profile_id)
        normalized = profile.normalize(parameters)
        adapters = tuple(
            adapter.to_dict()
            for adapter in profile.normalize_lora_adapters(lora_adapters)
        )
        return self._tasks.submit(
            "load",
            lambda cancel: self._load(
                profile_id, normalized, adapters, cancel),
            job_id=job_id,
            name="text-to-image-load",
            on_error=lambda _exc: log.exception(
                "Text-to-image model load failed"),
        )

    def _load(self, profile_id, parameters, adapters, cancel):
        result = self._client.request(
            "load_text_to_image",
            {
                "profile_id": profile_id,
                "parameters": parameters,
                "lora_adapters": list(adapters),
            },
            cancel,
        )
        self.model_info = dict(result)
        self._loaded = True
        self._loaded_profile_id = profile_id
        self._loaded_parameters = text_to_image_profile(
            profile_id).load_values(parameters)
        self._loaded_lora_adapters = tuple(adapters)
        return True

    def submit_request(
            self, request: TextToImageRequest, meta=None, *,
            job_id: str | None = None):
        return self._tasks.submit(
            "inference",
            lambda cancel: self._run_inference(request, cancel),
            meta=meta,
            job_id=job_id,
            name="text-to-image-inference",
            on_error=lambda _exc: log.exception(
                "Text-to-image inference failed"),
        )

    def _run_inference(self, request: TextToImageRequest, cancel):
        profile = text_to_image_profile(request.model_profile_id)
        adapters = [
            adapter.to_dict()
            for adapter in profile.normalize_lora_adapters(
                request.lora_adapters)
        ]
        result = self._client.request(
            "text_to_image",
            {
                "profile_id": request.model_profile_id,
                "parameters": request.parameters,
                "lora_adapters": adapters,
                "width": request.width,
                "height": request.height,
            },
            cancel,
        )
        image = result["image"]
        if image.size != (request.width, request.height):
            raise RuntimeError(
                "Text-to-image result size does not match target layer: "
                f"{image.width}x{image.height} != "
                f"{request.width}x{request.height}"
            )
        return TextToImageInferenceResult(
            image=image,
            seed=int(result["seed"]),
            provenance=(
                GenerationProvenance.from_dict(result["provenance"])
                if isinstance(result.get("provenance"), dict) else None
            ),
        )

    def poll_event(self):
        return self._tasks.poll_event()

    def cancel(self) -> bool:
        return self._tasks.cancel()

    def shutdown(self, timeout: float = 1.0):
        if self._tasks.shutdown(timeout):
            self._client.shutdown(timeout)
            self._loaded = False
            self._loaded_profile_id = None
            self._loaded_parameters = {}
            self._loaded_lora_adapters = ()
            self.model_info = {}
