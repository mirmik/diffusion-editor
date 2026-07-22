"""Toolkit-neutral owner of Diffusion Editor application state and workers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import os
from typing import Any, Callable, Protocol

import numpy as np
from tcbase import log

from ..agent.tools import create_editor_tool_registry
from ..document.document_service import DocumentService
from ..document.history import HistoryManager
from ..document.layer import Layer
from ..document.layer_stack import LayerStack
from ..engines.diffusion_engine import DiffusionEngine
from ..engines.grounding_engine import GroundingEngine
from ..engines.instruct_engine import InstructEngine
from ..engines.lama_engine import LamaEngine
from ..engines.segmentation_engine import SegmentationEngine
from ..generation.diffusion_controller import DiffusionGenerationController
from ..generation.instruct_controller import InstructGenerationController
from ..generation.lama_controller import LamaGenerationController
from ..generation.result_mapper import (
    map_diffusion_result,
    map_grounding_result,
    map_instruct_result,
    map_lama_result,
    map_segmentation_result,
)
from ..generation.segmentation_controller import SegmentationGenerationController
from ..grounding.controller import GroundingController
from .presentation import PanelUpdate, ViewPorts
from .settings import Settings

_BYTES_PER_GIB = 1024 * 1024 * 1024
DEFAULT_HISTORY_MEMORY_LIMIT_BYTES = 5 * _BYTES_PER_GIB
MIN_HISTORY_MEMORY_LIMIT_GIB = 0.25
MAX_HISTORY_MEMORY_LIMIT_GIB = 256.0
DEFAULT_MODELS_DIR = os.path.expanduser(
    "~/soft/stable-diffusion-webui-forge/models/Stable-diffusion/"
)


class SettingsStore(Protocol):
    def get(self, key: str, default: Any = None) -> Any: ...

    def set(self, key: str, value: Any) -> None: ...


@dataclass(frozen=True)
class EngineSet:
    diffusion: Any
    segmentation: Any
    lama: Any
    instruct: Any
    grounding: Any

    @classmethod
    def create_default(cls) -> "EngineSet":
        return cls(
            diffusion=DiffusionEngine(),
            segmentation=SegmentationEngine(),
            lama=LamaEngine(),
            instruct=InstructEngine(),
            grounding=GroundingEngine(),
        )


class ShutdownPhase(IntEnum):
    VIEW_WORKERS = 10
    ENGINE_WORKERS = 20
    GPU_RESOURCES = 30


@dataclass(frozen=True)
class _ShutdownResource:
    phase: ShutdownPhase
    order: int
    name: str
    close: Callable[[], None]


class EditorApplication:
    """Owns domain state, controllers and deterministic application shutdown.

    This module deliberately has no dependency on tcgui or termin-gui-native.
    A concrete toolkit binds explicit :class:`ViewPorts` after construction.
    """

    def __init__(
            self,
            *,
            settings: SettingsStore | None = None,
            engines: EngineSet | None = None) -> None:
        self.settings = settings if settings is not None else Settings()
        self.engines = engines if engines is not None else EngineSet.create_default()
        self.running = True
        self.closed = False
        self.project_path: str | None = None
        self.last_dir = str(self.settings.get("last_dir", ""))
        self.models_dir = self._load_models_dir()
        self.history_memory_limit_bytes = self._load_history_memory_limit_bytes()
        self.clipboard: np.ndarray | None = None
        self.clipboard_pos: tuple[int, int] | None = None
        self.history_replaying = False

        self.layer_stack = LayerStack()
        self.history = HistoryManager(
            self._apply_snapshot,
            max_memory_bytes=self.history_memory_limit_bytes,
        )
        self.document = DocumentService(
            self.layer_stack,
            self.history,
            self._apply_snapshot,
        )
        self.agent_tool_registry = create_editor_tool_registry()

        composite_below = self._composite_below
        self.diffusion_controller = DiffusionGenerationController(
            engine=self.engines.diffusion,
            layer_stack=self.layer_stack,
            composite_below=composite_below,
        )
        self.lama_controller = LamaGenerationController(
            engine=self.engines.lama,
            composite_below=composite_below,
        )
        self.instruct_controller = InstructGenerationController(
            engine=self.engines.instruct,
            composite_below=composite_below,
        )
        self.segmentation_controller = SegmentationGenerationController(
            engine=self.engines.segmentation,
            composite_below=composite_below,
        )
        self.grounding_controller = GroundingController(
            engine=self.engines.grounding,
            composite=lambda: self.layer_stack.composite(),
        )

        self._view = ViewPorts()
        self._snapshot_listeners: list[Callable[[], None]] = []
        self._shutdown_resources: list[_ShutdownResource] = []
        self._next_shutdown_order = 0
        self.shutdown_trace: list[str] = []
        for name, engine in (
                ("diffusion-engine", self.engines.diffusion),
                ("instruct-engine", self.engines.instruct),
                ("lama-engine", self.engines.lama),
                ("segmentation-engine", self.engines.segmentation),
                ("grounding-engine", self.engines.grounding)):
            self.register_shutdown_resource(
                ShutdownPhase.ENGINE_WORKERS,
                name,
                engine.shutdown,
            )

    def bind_view(self, ports: ViewPorts) -> None:
        self._view = ports
        if ports.status is not None:
            ports.status.set_status("Ready")

    def unbind_view(self) -> None:
        self._view = ViewPorts()

    def add_snapshot_listener(self, listener: Callable[[], None]) -> None:
        self._snapshot_listeners.append(listener)

    def register_shutdown_resource(
            self,
            phase: ShutdownPhase,
            name: str,
            close: Callable[[], None]) -> None:
        if self.closed:
            raise RuntimeError("cannot register a resource after application shutdown")
        self._shutdown_resources.append(_ShutdownResource(
            phase=phase,
            order=self._next_shutdown_order,
            name=name,
            close=close,
        ))
        self._next_shutdown_order += 1

    def set_status(self, text: str) -> None:
        if self._view.status is not None:
            self._view.status.set_status(text)

    def update_panel(self, panel_id: str, state: str, **payload: Any) -> None:
        if self._view.panels is not None:
            self._view.panels.update_panel(PanelUpdate(panel_id, state, payload))

    def set_history_memory_limit_bytes(self, limit_bytes: int) -> None:
        minimum = int(MIN_HISTORY_MEMORY_LIMIT_GIB * _BYTES_PER_GIB)
        limit_bytes = max(int(limit_bytes), minimum)
        self.history_memory_limit_bytes = limit_bytes
        self.document.set_history_memory_limit_bytes(limit_bytes)
        self.settings.set("history_memory_limit_bytes", limit_bytes)

    def set_models_dir(self, models_dir: str) -> None:
        value = os.path.expanduser(models_dir.strip()) or DEFAULT_MODELS_DIR
        self.models_dir = value
        self.settings.set("models_dir", value)

    def set_last_dir(self, directory: str) -> None:
        self.last_dir = directory
        self.settings.set("last_dir", directory)

    def clear_history(self) -> None:
        self.document.clear_history()

    def poll(self) -> None:
        """Poll controller events and project them without toolkit knowledge."""
        self._poll_segmentation()
        self._poll_lama()
        self._poll_instruct()
        self._poll_diffusion()
        self._poll_grounding()

    def request_stop(self) -> None:
        self.running = False

    def close(self) -> None:
        """Stop workers and GPU resources in stable phase/registration order."""
        if self.closed:
            return
        self.closed = True
        self.running = False
        resources = sorted(
            self._shutdown_resources,
            key=lambda resource: (resource.phase, resource.order),
        )
        for resource in resources:
            try:
                resource.close()
                self.shutdown_trace.append(resource.name)
            except Exception:
                log.exception(f"Application shutdown failed: {resource.name}")
        self.unbind_view()

    def _load_history_memory_limit_bytes(self) -> int:
        raw = self.settings.get(
            "history_memory_limit_bytes",
            DEFAULT_HISTORY_MEMORY_LIMIT_BYTES,
        )
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return DEFAULT_HISTORY_MEMORY_LIMIT_BYTES
        return value if value > 0 else DEFAULT_HISTORY_MEMORY_LIMIT_BYTES

    def _load_models_dir(self) -> str:
        raw = self.settings.get("models_dir", DEFAULT_MODELS_DIR)
        if not isinstance(raw, str):
            return DEFAULT_MODELS_DIR
        return os.path.expanduser(raw.strip()) or DEFAULT_MODELS_DIR

    def _apply_snapshot(self, snapshot: bytes) -> None:
        self.history_replaying = True
        try:
            self.layer_stack.load_state(snapshot)
        finally:
            self.history_replaying = False
            for listener in tuple(self._snapshot_listeners):
                listener()

    def _composite_below(self, layer: Layer) -> np.ndarray | None:
        return np.ascontiguousarray(self.layer_stack.composite(exclude_layer=layer))

    def _poll_segmentation(self) -> None:
        event = self.segmentation_controller.poll()
        if event is None:
            return
        if event.segmentation_result is not None:
            layer, seg_mask = event.segmentation_result
            command, status = map_segmentation_result(layer, seg_mask)
            if command is not None:
                self.document.execute(command)
            self.set_status(status)
        elif event.status:
            self.set_status(event.status)

    def _poll_lama(self) -> None:
        event = self.lama_controller.poll()
        if event is None:
            return
        if event.inference_result is not None:
            layer, result_image = event.inference_result
            command, status = map_lama_result(layer, result_image)
            if command is not None:
                self.document.execute(command)
            self.set_status(status)
        elif event.status:
            self.set_status(event.status)

    def _poll_instruct(self) -> None:
        event = self.instruct_controller.poll()
        if event is None:
            return
        if event.model_loading:
            self.update_panel("instruct", "model-loading")
        if event.model_error is not None:
            self.update_panel("instruct", "model-error", error=event.model_error)
        elif event.model_loaded:
            self.update_panel("instruct", "model-loaded")
        if event.inference_result is not None:
            layer, result_image, used_seed = event.inference_result
            command, status = map_instruct_result(layer, result_image, used_seed)
            if command is not None:
                self.document.execute(command)
            self.set_status(status)
        elif event.status:
            self.set_status(event.status)

    def _poll_diffusion(self) -> None:
        event = self.diffusion_controller.poll()
        if event is None:
            return
        if event.model_error is not None:
            self.update_panel("diffusion", "model-error", error=event.model_error)
        elif event.model_loaded_path is not None:
            self.update_panel(
                "diffusion",
                "model-loaded",
                path=event.model_loaded_path,
                info=self.engines.diffusion.model_info,
            )
        if event.ip_adapter_error is not None:
            self.update_panel("diffusion", "ip-adapter-error", error=event.ip_adapter_error)
        elif event.ip_adapter_loaded:
            self.update_panel("diffusion", "ip-adapter-loaded")
        if event.inference_result is not None:
            pending, result_image, used_seed = event.inference_result
            command, status = map_diffusion_result(pending, result_image, used_seed)
            if command is not None:
                self.document.execute(command)
            self.set_status(status)
        elif event.status is not None:
            self.set_status(event.status)

    def _poll_grounding(self) -> None:
        event = self.grounding_controller.poll()
        if event is None:
            return
        if event.grounding_result is not None:
            layer, result = event.grounding_result
            command, status = map_grounding_result(layer, result)
            if command is not None:
                self.document.execute(command)
            self.set_status(status)
        elif event.status:
            self.set_status(event.status)
