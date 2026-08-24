"""Toolkit-neutral owner of Diffusion Editor application state and workers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import os
import time
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image
from tcbase import log

from ..agent.tools import create_editor_tool_registry
from ..document.document_service import DocumentService
from ..document.commands import (
    AddDepthVisualizationCommand,
    AddLayerCommand,
    AddPoseOverlayCommand,
    SetLayerSelectionCommand,
)
from ..document.history import HistoryManager
from ..document.layer import Layer
from ..document.layer_stack import LayerStack
from ..document.change_event import DocumentChangeEvent
from ..document.session import DocumentSession, RecoveryRecord, RecoveryStore
from ..engines.diffusion_engine import DiffusionEngine
from ..engines.depth_engine import DepthEstimationEngine
from ..engines.grounding_engine import GroundingEngine
from ..engines.instruct_engine import InstructEngine
from ..engines.lama_engine import LamaEngine
from ..engines.pose_engine import PoseEstimationEngine
from ..engines.segmentation_engine import SegmentationEngine
from ..generation.diffusion_controller import DiffusionGenerationController
from ..generation.depth_controller import DepthGenerationController
from ..generation.depth_point_cloud import (
    DepthPointCloudData,
    project_depth_point_cloud,
)
from ..generation.depth_visualization import colorize_depth
from ..generation.instruct_controller import InstructGenerationController
from ..generation.job_context import (
    ApplyFrozenGeneratedResultCommand,
    FrozenArray,
    InferenceJobContext,
    JobDocumentState,
    ResultApplicationPolicy,
)
from ..generation.lama_controller import LamaGenerationController
from ..generation.pose_controller import PoseEstimationController
from ..generation.pose_estimation import (
    PoseEstimationResult,
    pose_estimator_profile,
    render_pose_overlay,
)
from ..generation.provenance import capture_tool_state
from ..generation.result_mapper import (
    map_grounding_result,
    map_segmentation_result,
)
from ..generation.segmentation_controller import SegmentationGenerationController
from ..generation.types import (
    DepthEstimationResult,
    DepthValueKind,
    depth_model_profile,
)
from ..grounding.controller import GroundingController
from .presentation import PanelUpdate, ViewPorts
from .settings import Settings

_BYTES_PER_GIB = 1024 * 1024 * 1024
DEFAULT_HISTORY_MEMORY_LIMIT_BYTES = 5 * _BYTES_PER_GIB
MIN_HISTORY_MEMORY_LIMIT_GIB = 0.25
MAX_HISTORY_MEMORY_LIMIT_GIB = 256.0
DEFAULT_RECOVERY_INITIAL_DELAY_SECONDS = 60.0
DEFAULT_RECOVERY_INTERVAL_SECONDS = 5.0 * 60.0
MIN_RECOVERY_DELAY_SECONDS = 60.0
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
    depth: Any | None = None
    pose: Any | None = None

    @classmethod
    def create_default(cls) -> "EngineSet":
        return cls(
            diffusion=DiffusionEngine(),
            segmentation=SegmentationEngine(),
            lama=LamaEngine(),
            instruct=InstructEngine(),
            grounding=GroundingEngine(),
            depth=DepthEstimationEngine(),
            pose=PoseEstimationEngine(),
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


@dataclass(frozen=True)
class _PendingSubjectDepth:
    profile_id: str
    layer_id: str


class EditorApplication:
    """Owns domain state, controllers and deterministic application shutdown.

    This module deliberately has no dependency on a UI toolkit.
    A concrete toolkit binds explicit :class:`ViewPorts` after construction.
    """

    def __init__(
            self,
            *,
            settings: SettingsStore | None = None,
            engines: EngineSet | None = None) -> None:
        self.settings = settings if settings is not None else Settings()
        self.engines = engines if engines is not None else EngineSet.create_default()
        self.depth_engine = self.engines.depth or DepthEstimationEngine()
        self.pose_engine = self.engines.pose or PoseEstimationEngine()
        self.running = True
        self.closed = False
        self.status_text = "Ready"
        self.window_title = "Diffusion Editor"
        self.command_states: dict[str, tuple[bool, bool]] = {}
        self.last_dir = str(self.settings.get("last_dir", ""))
        self.models_dir = self._load_models_dir()
        self.history_memory_limit_bytes = self._load_history_memory_limit_bytes()
        self.clipboard: np.ndarray | None = None
        self.clipboard_pos: tuple[int, int] | None = None
        self.history_replaying = False
        self._pending_subject_depth: _PendingSubjectDepth | None = None
        self.latest_depth_point_cloud: DepthPointCloudData | None = None
        self.latest_depth_point_cloud_error: str | None = None
        self.latest_depth_result: DepthEstimationResult | None = None
        self.latest_pose_result: PoseEstimationResult | None = None
        self._latest_depth_point_cloud_layer_ids: frozenset[str] = frozenset()

        self.layer_stack = LayerStack()
        self.session = DocumentSession()
        recovery_root = str(self.settings.get(
            "recovery_dir",
            os.path.expanduser("~/.cache/diffusion-editor/recovery"),
        ))
        self.recovery_store = RecoveryStore(recovery_root)
        self.recovery_initial_delay_seconds = max(
            MIN_RECOVERY_DELAY_SECONDS,
            float(self.settings.get(
                "recovery_initial_delay_seconds",
                DEFAULT_RECOVERY_INITIAL_DELAY_SECONDS,
            )),
        )
        self.recovery_interval_seconds = max(
            MIN_RECOVERY_DELAY_SECONDS,
            float(self.settings.get(
                "recovery_interval_seconds",
                DEFAULT_RECOVERY_INTERVAL_SECONDS,
            )),
        )
        self._recovery_due_at: float | None = None
        self._last_recovery_at: float | None = None
        self._last_recovery_revision: int | None = None
        self._session_subscription = self.layer_stack.subscribe(
            self._on_document_change)
        self.history = HistoryManager(
            self._apply_snapshot,
            max_memory_bytes=self.history_memory_limit_bytes,
        )
        self.document = DocumentService(
            self.layer_stack,
            self.history,
            self._apply_snapshot,
            after_history_navigation=self.reconcile_document_session,
        )
        self.agent_tool_registry = create_editor_tool_registry()

        composite_below = self._composite_below
        document_state = self._generation_document_state
        self.diffusion_controller = DiffusionGenerationController(
            engine=self.engines.diffusion,
            layer_stack=self.layer_stack,
            composite_below=composite_below,
            document_state=document_state,
        )
        self.lama_controller = LamaGenerationController(
            engine=self.engines.lama,
            composite_below=composite_below,
            document_state=document_state,
        )
        self.instruct_controller = InstructGenerationController(
            engine=self.engines.instruct,
            layer_stack=self.layer_stack,
            composite_below=composite_below,
            document_state=document_state,
        )
        self.segmentation_controller = SegmentationGenerationController(
            engine=self.engines.segmentation,
            composite_below=composite_below,
            composite=lambda: np.ascontiguousarray(
                self.layer_stack.composite()
            ),
            document_state=document_state,
        )
        self.grounding_controller = GroundingController(
            engine=self.engines.grounding,
            composite=lambda: self.layer_stack.composite(),
        )
        self.depth_controller = DepthGenerationController(
            engine=self.depth_engine,
            composite=lambda: np.ascontiguousarray(
                self.layer_stack.composite()
            ),
            selection=lambda: (
                None
                if self.layer_stack.selection.is_empty
                else self.layer_stack.selection.data
            ),
            document_state=document_state,
        )
        self.pose_controller = PoseEstimationController(
            self.pose_engine,
            composite=lambda: np.ascontiguousarray(
                self.layer_stack.composite()
            ),
            document_state=document_state,
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
                ("grounding-engine", self.engines.grounding),
                ("depth-engine", self.depth_engine),
                ("pose-engine", self.pose_engine)):
            self.register_shutdown_resource(
                ShutdownPhase.ENGINE_WORKERS,
                name,
                engine.shutdown,
            )

    def bind_view(self, ports: ViewPorts) -> None:
        self._view = ports
        if ports.status is not None:
            ports.status.set_status(self.status_text)
        if ports.window is not None:
            ports.window.set_window_title(self.window_title)
        if ports.commands is not None:
            for command_id, (enabled, checked) in self.command_states.items():
                ports.commands.set_command_state(
                    command_id,
                    enabled=enabled,
                    checked=checked,
                )

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
        self.status_text = text
        if self._view.status is not None:
            self._view.status.set_status(text)

    def set_window_title(self, title: str) -> None:
        self.window_title = title
        if self._view.window is not None:
            self._view.window.set_window_title(title)

    def set_command_state(
            self,
            command_id: str,
            *,
            enabled: bool,
            checked: bool = False) -> None:
        self.command_states[command_id] = (enabled, checked)
        if self._view.commands is not None:
            self._view.commands.set_command_state(
                command_id,
                enabled=enabled,
                checked=checked,
            )

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

    @property
    def document_session_id(self) -> str:
        return self.session.session_id

    @property
    def document_revision(self) -> int:
        return self.session.revision

    @property
    def project_path(self) -> str | None:
        return self.session.path

    @project_path.setter
    def project_path(self, value: str | None) -> None:
        self.session.path = value

    @property
    def document_dirty(self) -> bool:
        return self.session.dirty

    def mark_external_mutation(self) -> int:
        """Reserve a revision for a mutation outside DocumentService.

        Canvas transactions call this as soon as a direct gesture starts
        mutating document state, before its eventual history command is
        committed.
        """
        revision = self.session.mark_external_mutation()
        self._schedule_recovery()
        return revision

    def reset_document_session(
            self,
            project_path: str | None = None,
            *,
            clean: bool = True) -> None:
        """Invalidate document-bound jobs after New/Open/Import."""
        old_session_id = self.session.session_id
        snapshot = (
            self.layer_stack.serialize_state() if clean else b"")
        self.session.reset(
            revision=self.layer_stack.revision,
            path=project_path,
            snapshot=snapshot,
            clean=clean,
        )
        self.recovery_store.discard(old_session_id)
        self._recovery_due_at = None
        self._last_recovery_at = None
        self._last_recovery_revision = None
        if not clean:
            self._schedule_recovery()
        try:
            self.invalidate_generation_jobs("document was replaced")
        except Exception as exc:
            log.exception(
                f"Failed to cancel generation jobs after document reset: {exc}")

    def mark_document_saved(self, path: str) -> None:
        snapshot = self.layer_stack.serialize_state()
        old_session_id = self.session.session_id
        self.session.mark_saved(path, snapshot)
        self.recovery_store.discard(old_session_id)
        self._recovery_due_at = None
        self._last_recovery_at = None
        self._last_recovery_revision = None

    def reconcile_document_session(self) -> None:
        self.session.reconcile(self.layer_stack.serialize_state())

    @property
    def available_recovery(self) -> RecoveryRecord | None:
        return self.recovery_store.latest()

    def restore_recovery(self, record: RecoveryRecord) -> None:
        self.document.prepare_mutation()
        self.layer_stack.load_project(str(record.snapshot_path))
        self.clear_history()
        self.reset_document_session(None, clean=False)
        self.recovery_store.discard(record.session_id)
        self.set_status("Recovered unsaved document")

    def discard_recovery(self, record: RecoveryRecord) -> None:
        self.recovery_store.discard(record.session_id)

    def discard_unsaved_changes(self) -> None:
        self.recovery_store.discard(self.session.session_id)
        self.session.mark_discarded()
        self._recovery_due_at = None

    def invalidate_generation_jobs(self, reason: str) -> bool:
        invalidated = False
        for controller in self._generation_controllers():
            invalidate = getattr(controller, "invalidate_all", None)
            if callable(invalidate):
                invalidated = bool(invalidate(reason)) or invalidated
        return invalidated

    def invalidate_generation_jobs_for_layer(
            self,
            layer_id: str,
            reason: str = "target layer changed") -> bool:
        invalidated = False
        for controller in self._generation_controllers():
            invalidate = getattr(controller, "invalidate_layer", None)
            if callable(invalidate):
                invalidated = (
                    bool(invalidate(layer_id, reason)) or invalidated
                )
        return invalidated

    def poll(self) -> None:
        """Poll controller events and project them without toolkit knowledge."""
        if self.closed:
            return
        self._maybe_write_recovery()
        self._invalidate_stale_generation_jobs()
        self._poll_segmentation()
        self._poll_lama()
        self._poll_instruct()
        self._poll_diffusion()
        self._poll_grounding()
        self._poll_depth()
        self._poll_pose()

    def request_stop(self) -> None:
        self.running = False

    def depth_point_cloud_for_layer(
            self, layer: Layer | None) -> DepthPointCloudData | None:
        if (
                layer is None
                or layer.id not in self._latest_depth_point_cloud_layer_ids):
            return None
        return self.latest_depth_point_cloud

    def has_depth_point_cloud_context(self, layer: Layer | None) -> bool:
        return (
            layer is not None
            and layer.id in self._latest_depth_point_cloud_layer_ids
        )

    def start_subject_depth(
            self,
            layer: Layer,
            profile_id: str,
    ) -> str | None:
        """Segment the foreground, then start masked depth inference."""

        if (
                self._pending_subject_depth is not None
                or self.segmentation_controller.is_busy
                or self.depth_controller.is_busy):
            return None
        try:
            depth_model_profile(profile_id)
        except ValueError as exc:
            return str(exc)
        event = self.segmentation_controller.start_select_background_selection(
            layer)
        if event.status is None:
            return None
        self._pending_subject_depth = _PendingSubjectDepth(
            profile_id=profile_id,
            layer_id=layer.id,
        )
        return "Isolating subject for depth..."

    def close(self) -> None:
        """Stop workers and GPU resources in stable phase/registration order."""
        if self.closed:
            return
        self.invalidate_generation_jobs("application is closing")
        self.closed = True
        self.running = False
        if not self.session.dirty:
            self.recovery_store.discard(self.session.session_id)
        self._session_subscription.unsubscribe()
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

    def _generation_document_state(self) -> JobDocumentState:
        return JobDocumentState(
            session_id=self.session.session_id,
            revision=self.document_revision,
        )

    def _on_document_change(self, event: DocumentChangeEvent) -> None:
        self.session.record_change(event)
        self._schedule_recovery()

    def _schedule_recovery(self) -> None:
        if self._recovery_due_at is not None:
            return
        now = time.monotonic()
        due_at = now + self.recovery_initial_delay_seconds
        if self._last_recovery_at is not None:
            due_at = max(
                due_at,
                self._last_recovery_at + self.recovery_interval_seconds,
            )
        self._recovery_due_at = due_at

    def _maybe_write_recovery(self) -> None:
        now = time.monotonic()
        if self._recovery_due_at is None or now < self._recovery_due_at:
            return
        if not self.session.dirty or not self.layer_stack.layers:
            self._recovery_due_at = None
            return
        if self._last_recovery_revision == self.session.revision:
            self._recovery_due_at = None
            return
        try:
            self.recovery_store.write_project(
                self.session, self.layer_stack.save_project)
            self._last_recovery_at = now
            self._last_recovery_revision = self.session.revision
            self._recovery_due_at = None
        except Exception:
            log.exception("Failed to write document recovery snapshot")
            self._recovery_due_at = now + self.recovery_initial_delay_seconds

    def _generation_controllers(self) -> tuple[object, ...]:
        return (
            self.diffusion_controller,
            self.instruct_controller,
            self.lama_controller,
            self.segmentation_controller,
            self.depth_controller,
            self.pose_controller,
        )

    def _invalidate_stale_generation_jobs(self) -> None:
        status: str | None = None
        for controller in self._generation_controllers():
            contexts = tuple(getattr(
                controller, "pending_contexts", ()))
            for context in contexts:
                reason = self._generation_rejection_reason(context)
                if reason is None:
                    continue
                invalidate = getattr(controller, "invalidate_job", None)
                if callable(invalidate) and invalidate(
                        context.job_id, reason):
                    status = f"Generation cancelled: {reason}"
        if status is not None:
            self.set_status(status)

    def _generation_rejection_reason(
            self, context: InferenceJobContext) -> str | None:
        if context.application_policy != ResultApplicationPolicy.REJECT_STALE:
            return (
                "unsupported result application policy "
                f"'{context.application_policy.value}'"
            )
        if context.document_session_id != self.session.session_id:
            return "document session changed"
        layer = self.layer_stack.find_layer_by_id(context.layer_id)
        if layer is None:
            return "target layer no longer exists"
        current_tool_type = (
            str(layer.tool.tool_type)
            if layer.tool is not None
            else None
        )
        if current_tool_type != context.tool_type:
            return "target tool changed or was detached"
        if (
                context.tool_state_fingerprint
                and capture_tool_state(layer.tool).fingerprint
                != context.tool_state_fingerprint):
            return "generation request settings changed"
        if layer.bounds != context.layer_bounds:
            return "target layer geometry changed"
        if layer.pixel_revision != context.target_pixel_revision:
            return "target layer pixels changed"
        current_mask = FrozenArray.capture(layer.mask.to_uint8())
        if (
                context.target_mask is not None
                and current_mask != context.target_mask):
            return "target mask changed"
        if context.paste is not None and not context.paste.matches_layer(layer):
            return "request geometry or mask changed"
        if context.base_revision != self.document_revision:
            return "document revision changed"
        return None

    def _resolve_generation_target(
            self,
            context: InferenceJobContext,
    ) -> tuple[Layer | None, str | None]:
        reason = self._generation_rejection_reason(context)
        if reason is not None:
            return None, f"{context.kind.capitalize()} result rejected: {reason}"
        return self.layer_stack.find_layer_by_id(context.layer_id), None

    @staticmethod
    def _generation_success_status(
            base: str,
            context: InferenceJobContext,
    ) -> str:
        provenance = context.result_provenance
        if provenance is None or not provenance.warnings:
            return base
        warning = provenance.warnings[0]
        return f"{base}; reproducibility warning: {warning[:160]}"

    def _apply_snapshot(self, snapshot: bytes) -> None:
        self.history_replaying = True
        try:
            self.layer_stack.load_state(snapshot)
        finally:
            self.history_replaying = False
            for listener in tuple(self._snapshot_listeners):
                try:
                    listener()
                except Exception as exc:
                    log.exception(
                        f"Snapshot observer failed after commit: {exc}")

    def _composite_below(self, layer: Layer) -> np.ndarray | None:
        return np.ascontiguousarray(self.layer_stack.composite(exclude_layer=layer))

    def _poll_segmentation(self) -> None:
        event = self.segmentation_controller.poll()
        if event is None:
            return
        if event.segmentation_result is not None:
            context, seg_mask = event.segmentation_result
            layer, rejection = self._resolve_generation_target(context)
            if rejection is not None:
                self.set_status(rejection)
                return
            command, status = map_segmentation_result(layer, seg_mask)
            if command is not None:
                self.document.execute(command)
            self.set_status(status)
        elif event.selection_result is not None:
            context, seg_mask = event.selection_result
            layer, rejection = self._resolve_generation_target(context)
            if rejection is not None:
                self._pending_subject_depth = None
                self.set_status(rejection)
                return
            pending = self._pending_subject_depth
            if pending is not None:
                self._pending_subject_depth = None
                if layer is None or layer.id != pending.layer_id:
                    self.set_status(
                        "Subject depth cancelled: target layer changed"
                    )
                    return
                foreground_mask = 255 - np.asarray(
                    seg_mask, dtype=np.uint8)
                if not np.any(foreground_mask > 0):
                    self.set_status(
                        "Subject depth: foreground segmentation found nothing"
                    )
                    return
                depth_event = self.depth_controller.start(
                    layer,
                    pending.profile_id,
                    output_mask=foreground_mask,
                )
                self.set_status(
                    depth_event.status
                    or "Subject depth: failed to start depth estimation"
                )
                return
            self.document.execute(SetLayerSelectionCommand(
                mask=seg_mask,
                label="Select Background",
            ))
            self.set_status("Background selected")
        elif event.segmentation_error is not None:
            self._pending_subject_depth = None
            if event.status:
                self.set_status(event.status)
        elif event.status:
            if (
                    self._pending_subject_depth is not None
                    and not self.segmentation_controller.is_busy):
                self._pending_subject_depth = None
            self.set_status(event.status)

    def _poll_lama(self) -> None:
        event = self.lama_controller.poll()
        if event is None:
            return
        if event.inference_result is not None:
            context, result_image = event.inference_result
            layer, rejection = self._resolve_generation_target(context)
            if rejection is not None or layer is None or context.paste is None:
                status = rejection or "LaMa result rejected: invalid job context"
                self.update_panel("lama", "error", error=status)
                self.set_status(status)
                return
            self.document.execute(ApplyFrozenGeneratedResultCommand(
                layer=layer,
                result_image=result_image,
                paste=context.paste,
                label="Apply LaMa Result",
                provenance=context.result_provenance,
            ))
            self.update_panel("lama", "result")
            self.set_status(self._generation_success_status(
                "Objects removed (LaMa)",
                context,
            ))
        elif event.inference_error is not None:
            self.update_panel(
                "lama", "error", error=event.inference_error)
            if event.status:
                self.set_status(event.status)
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
            info = getattr(self.engines.instruct, "model_info", {})
            self.update_panel(
                "instruct",
                "model-loaded",
                profile_id=(
                    str(info.get("profile_id", ""))
                    if isinstance(info, dict) else ""
                ),
            )
        if (
                event.status
                and event.inference_result is None
                and event.inference_error is None
                and getattr(
                    self.instruct_controller,
                    "pending_context",
                    None,
                ) is not None):
            self.update_panel(
                "instruct", "running", status=event.status)
        if event.inference_result is not None:
            context, result_image, used_seed = event.inference_result
            layer, rejection = self._resolve_generation_target(context)
            if rejection is not None or layer is None or context.paste is None:
                status = (
                    rejection
                    or "Instruct result rejected: invalid job context"
                )
                self.update_panel(
                    "instruct", "inference-error", error=status)
                self.set_status(status)
                return
            self.document.execute(ApplyFrozenGeneratedResultCommand(
                layer=layer,
                result_image=result_image,
                paste=context.paste,
                label="Apply AI Edit Result",
                provenance=context.result_provenance,
            ))
            self.update_panel("instruct", "result")
            self.set_status(self._generation_success_status(
                f"AI edit applied (seed={used_seed})",
                context,
            ))
        elif event.inference_error is not None:
            self.update_panel(
                "instruct",
                "inference-error",
                error=event.inference_error,
            )
            if event.status:
                self.set_status(event.status)
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
        if (
                event.status
                and event.inference_result is None
                and event.inference_error is None
                and getattr(
                    self.diffusion_controller,
                    "pending_context",
                    None,
                ) is not None):
            self.update_panel(
                "diffusion", "running", status=event.status)
        if event.inference_result is not None:
            context, result_image, used_seed = event.inference_result
            layer, rejection = self._resolve_generation_target(context)
            if rejection is not None or layer is None or context.paste is None:
                status = (
                    rejection
                    or "Diffusion result rejected: invalid job context"
                )
                self.update_panel(
                    "diffusion", "inference-error", error=status)
                self.set_status(status)
                return
            self.document.execute(ApplyFrozenGeneratedResultCommand(
                layer=layer,
                result_image=result_image,
                paste=context.paste,
                label="Apply Diffusion Result",
                provenance=context.result_provenance,
            ))
            self.update_panel("diffusion", "result")
            self.set_status(self._generation_success_status(
                f"Regenerated (seed={used_seed})",
                context,
            ))
        elif event.inference_error is not None:
            self.update_panel(
                "diffusion",
                "inference-error",
                error=event.inference_error,
            )
            if event.status is not None:
                self.set_status(event.status)
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

    def _poll_depth(self) -> None:
        event = self.depth_controller.poll()
        if event is None:
            return
        if event.depth_result is not None:
            context, depth_result, output_mask = event.depth_result
            depth_map = depth_result.depth_map
            _layer, rejection = self._resolve_generation_target(context)
            if rejection is not None:
                self.set_status(rejection)
                return
            expected_shape = (self.layer_stack.height, self.layer_stack.width)
            if output_mask is None:
                alpha = np.full(expected_shape, 255, dtype=np.uint8)
            else:
                alpha = np.asarray(output_mask, dtype=np.uint8)
                if alpha.shape != expected_shape:
                    self.set_status(
                        "Depth result rejected: output mask size does not "
                        "match canvas"
                    )
                    return
            source = (
                context.input_array.to_array()
                if context.input_array is not None else None
            )
            depth_height, depth_width = depth_map.shape
            mask_for_depth = output_mask
            source_for_depth = source
            if depth_map.shape != expected_shape:
                if output_mask is not None:
                    mask_for_depth = np.asarray(
                        Image.fromarray(output_mask, "L").resize(
                            (depth_width, depth_height),
                            Image.Resampling.LANCZOS,
                        ),
                        dtype=np.uint8,
                    )
                if source is not None:
                    source_for_depth = np.asarray(
                        Image.fromarray(source).resize(
                            (depth_width, depth_height),
                            Image.Resampling.LANCZOS,
                        ),
                        dtype=np.uint8,
                    )
            preview = colorize_depth(
                depth_map,
                mask=mask_for_depth,
                near_is_high=(
                    depth_result.value_kind
                    is DepthValueKind.INVERSE_RELATIVE
                ),
            )
            preview_rgb = preview.rgb
            if preview_rgb.shape[:2] != expected_shape:
                preview_rgb = np.asarray(
                    Image.fromarray(preview_rgb, "RGB").resize(
                        (expected_shape[1], expected_shape[0]),
                        Image.Resampling.LANCZOS,
                    ),
                    dtype=np.uint8,
                )
            preview_rgba = np.dstack((preview_rgb, alpha))
            point_cloud = None
            point_cloud_error = None
            if source_for_depth is not None:
                try:
                    point_cloud = project_depth_point_cloud(
                        source_for_depth,
                        depth_map,
                        mask=mask_for_depth,
                        confidence=depth_result.confidence,
                        intrinsics=depth_result.intrinsics,
                        value_kind=depth_result.value_kind,
                        fallback_fov_y_degrees=(
                            55.0
                            if depth_result.value_kind
                            is DepthValueKind.INVERSE_RELATIVE
                            else None
                        ),
                    )
                except ValueError as exc:
                    point_cloud_error = str(exc)
                    log.error(f"Depth point-cloud projection failed: {exc}")
            profile_id = context.provenance("profile_id")
            try:
                profile = depth_model_profile(profile_id or "")
            except ValueError:
                self.set_status(
                    "Depth result rejected: unknown model profile"
                )
                return
            base_name = self.layer_stack.next_name(profile.layer_name)
            self.document.execute(AddDepthVisualizationCommand(
                name=f"{base_name} Preview (float32 source)",
                preview_image=np.ascontiguousarray(preview_rgba),
                label=f"Create Depth Map ({profile.title})",
            ))
            self.latest_depth_result = depth_result
            preview_layer = self.layer_stack.active_layer
            self._latest_depth_point_cloud_layer_ids = frozenset(
                layer.id
                for layer in (preview_layer,)
                if layer is not None
            )
            if point_cloud is not None:
                self.latest_depth_point_cloud = point_cloud
                self.latest_depth_point_cloud_error = None
            else:
                self.latest_depth_point_cloud = None
                self.latest_depth_point_cloud_error = (
                    point_cloud_error or "source image is unavailable"
                )
            if depth_result.value_kind is DepthValueKind.DIRECT_METRIC:
                depth_kind = "metric float32 with model camera calibration"
            elif depth_result.intrinsics is not None:
                depth_kind = (
                    "scale-ambiguous direct float32 with model camera "
                    "calibration"
                )
            elif depth_result.value_kind is DepthValueKind.INVERSE_RELATIVE:
                depth_kind = (
                    "inverse-relative float32; approximate 55° camera"
                )
            else:
                depth_kind = (
                    "scale-ambiguous direct float32; no camera calibration"
                )
            mask_status = (
                "; full-frame inference; output mask applied afterward"
                if output_mask is not None else ""
            )
            cloud_status = (
                ""
                if self.latest_depth_point_cloud_error is None
                else "; point cloud unavailable: "
                f"{self.latest_depth_point_cloud_error}"
            )
            self.set_status(
                f"{profile.title}: {depth_kind} depth map created "
                f"at raw {depth_width}x{depth_height} model resolution "
                f"(cold is farther, warm is closer; contrast uses "
                f"{preview.low:.6g}…{preview.high:.6g}; canonical values "
                f"remain unquantized"
                f"{mask_status}{cloud_status})"
            )
        elif event.status:
            self.set_status(event.status)

    def _poll_pose(self) -> None:
        event = self.pose_controller.poll()
        if event is None:
            return
        if event.pose_result is not None:
            context, result = event.pose_result
            layer, rejection = self._resolve_generation_target(context)
            if rejection is not None or layer is None:
                self.set_status(
                    rejection or "Pose result rejected: target layer is missing")
                return
            if (result.width, result.height) != (
                    self.layer_stack.width, self.layer_stack.height):
                self.set_status(
                    "Pose result rejected: result size does not match canvas")
                return
            profile = pose_estimator_profile(result.profile_id)
            overlay = render_pose_overlay(result)
            self.document.execute(AddPoseOverlayCommand(
                source_layer=layer,
                name=self.layer_stack.next_name(profile.layer_name),
                overlay_image=overlay,
                label=f"Create {profile.title} Overlay",
            ))
            self.latest_pose_result = result
            visible = sum(
                point.score >= 0.25
                for pose in result.poses
                for point in pose.keypoints
            )
            self.set_status(
                f"{profile.title}: {len(result.poses)} pose(s), "
                f"{visible} visible keypoints")
        elif event.status:
            self.set_status(event.status)
