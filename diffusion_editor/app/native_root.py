"""Hostless termin-gui-native composition for the editor migration path."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol
import zipfile

from PIL import Image
from tcbase import log
from termin.dispatch import Dispatcher, DispatchStats
from termin.display.window import WindowHandle, WindowManager, WindowedGraphicsSession
from termin.gui_native import (
    DynamicTextureLease,
    OffscreenGuiComposition,
    TcDocument,
    tc_ui_document_create,
    tc_ui_document_destroy,
)
from termin.gui_native.window import GuiWindowAdapter, dynamic_texture_lease
from tgfx import configure_default_shader_runtime

from ..sdk_runtime import resolve_sdk
from ..document.layer import Layer
from ..document.change_event import DocumentChangeKind
from ..document.commands import (
    AddReconstructionLayerCommand,
    ClearSelectionCommand,
    PublishReconstructionResultCommand,
    SelectReconstructionRunCommand,
    SetReconstructionRefinePlacementCommand,
)
from ..document.reconstruction import ReconstructionLayer, ReconstructionStatus
from ..document.tool import DiffusionTool, InstructTool, LamaTool
from ..generation.patch_resolver import (
    source_patch_at_center,
    source_patch_from_full_composite,
)
from ..generation.types import (
    RECONSTRUCTION_BACKEND_STAGES,
    ReconstructionBackend,
    ReconstructionLrVariant,
    ReconstructionParameters,
    ReconstructionRefinePlacement,
    ReconstructionRunKind,
    ReconstructionStage,
    ReconstructionStageStatus,
    pixal3d_resume_parameters_compatible,
)
from ..generation.local_detail_geometry import (
    euler_degrees_from_quaternion,
    quaternion_from_euler_degrees,
)
from ..generation.reconstruction_workspace import (
    LEGACY_OPERATION_TARGET_STAGES,
    PIXAL3D_OPERATION_PARAMETER_KEYS,
    WorkspacePreviewKind,
    build_legacy_workspace,
)
from .application import EditorApplication, ShutdownPhase
from .canvas_controls import (
    CanvasControlsCoordinator,
    SelectionControlAction,
    SelectionControlsIntent,
)
from .canvas_status import CanvasStatusCoordinator
from .editor_commands import EditorCommandCoordinator
from .generation_panels import GenerationPanelsCoordinator
from .native_canvas_controls import NativeCanvasControls
from .native_generation_panels import NativeGenerationPanels
from .dialogs import ApplicationDialogCoordinator
from .native_dialogs import NativeApplicationDialogs
from ..agent.chat import AgentChatCoordinator
from .native_agent_chat import NativeAgentChatPanel
from .layer_tree import LayerTreeCoordinator
from .native_layer_panel import NativeLayerPanel
from .native_shell import CommandHandler, NativeEditorView
from ..canvas.edit_transactions import CanvasEditTransactionCoordinator
from ..canvas.native_editor_canvas import NativeEditorCanvas
from ..engines.reconstruction_engine import ReconstructionEngine
from ..generation.reconstruction_controller import ReconstructionController
from ..automation import start_editor_automation
from .native_reconstruction_viewport import NativeReconstructionViewport


DEFAULT_NATIVE_WIDTH = 1280
DEFAULT_NATIVE_HEIGHT = 800
DEFAULT_DISPATCH_LIMIT = 256
DEFAULT_FONT_RELATIVE_PATH = Path("share/termin/fonts/DroidSans.ttf")


def _is_composite_shape_checkpoint(path: str | os.PathLike) -> bool:
    """Identify geometry-only composite checkpoints without importing NumPy."""
    try:
        with zipfile.ZipFile(path) as archive:
            return "composite_kind.npy" in archive.namelist()
    except (OSError, zipfile.BadZipFile):
        return False


def _hr_refine_source_run(node: ReconstructionLayer):
    """Return the nearest non-composite HR ancestor of the active run."""
    runs = {run.run_id: run for run in node.runs}
    candidate = node.active_run or node.base_run
    visited = set()
    while candidate is not None and candidate.run_id not in visited:
        visited.add(candidate.run_id)
        checkpoint = candidate.checkpoint_path
        if checkpoint and not _is_composite_shape_checkpoint(checkpoint):
            return candidate
        candidate = runs.get(candidate.parent_run_id or "")
    fallback = node.base_run
    if (
        fallback is not None
        and fallback.run_id not in visited
        and fallback.checkpoint_path
        and not _is_composite_shape_checkpoint(fallback.checkpoint_path)
    ):
        return fallback
    return None


class NativeComposition(Protocol):
    """Rendering/input owner consumed by :class:`NativeEditorRoot`."""

    document: TcDocument

    @property
    def should_close(self) -> bool: ...

    def pump_events(self) -> int: ...

    def request_repaint(self) -> None: ...

    def set_unhandled_key_handler(
            self,
            callback: Callable[[int, int], bool] | None) -> None: ...

    def render_frame(self) -> bool: ...

    def close(self) -> None: ...


class NativeViewFactory(Protocol):
    def __call__(
            self,
            document: TcDocument,
            request_repaint: Callable[[], None],
            set_window_title: Callable[[str], None],
            command_handlers: dict[str, CommandHandler]) -> NativeEditorView: ...


@dataclass(frozen=True)
class NativeTickResult:
    dispatched: int
    dispatch_failed: int
    dispatch_remaining: int
    events: int
    rendered: bool


def bundled_native_font_path(sdk_root: Path | None = None) -> Path:
    root = resolve_sdk() if sdk_root is None else sdk_root.expanduser().resolve()
    font_path = root / DEFAULT_FONT_RELATIVE_PATH
    if not font_path.is_file():
        raise RuntimeError(f"Termin SDK native UI font is missing: {font_path}")
    return font_path


class WindowedNativeComposition:
    """Application-owned window/session plus the borrowed native GUI adapter."""

    def __init__(
            self,
            *,
            title: str = "Diffusion Editor (native migration)",
            width: int = DEFAULT_NATIVE_WIDTH,
            height: int = DEFAULT_NATIVE_HEIGHT,
            sdk_root: Path | None = None,
            font_size: int = 14) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("native window dimensions must be positive")

        self._closed = False
        self._session: WindowedGraphicsSession | None = None
        self._manager: WindowManager | None = None
        self._handle: WindowHandle | None = None
        self._document: TcDocument | None = None
        self._adapter: GuiWindowAdapter | None = None

        root = resolve_sdk() if sdk_root is None else sdk_root.expanduser().resolve()
        if not configure_default_shader_runtime("diffusion-editor-native"):
            raise RuntimeError("Termin shader runtime is unavailable")

        try:
            self._session = WindowedGraphicsSession.create_native()
            self._manager = WindowManager(self._session)
            self._document = tc_ui_document_create()
            self._handle = self._manager.create_window(title, width, height)
            self._adapter = GuiWindowAdapter(
                self._manager,
                self._handle,
                self._document,
                font_path=str(bundled_native_font_path(root)),
                font_size=font_size,
            )
        except Exception:
            self.close()
            raise

    @property
    def document(self) -> TcDocument:
        if self._document is None or not self._document.valid:
            raise RuntimeError("native window document is closed")
        return self._document

    @property
    def should_close(self) -> bool:
        return self._adapter is None or self._adapter.should_close

    @property
    def graphics(self):
        if self._session is None:
            raise RuntimeError("native window composition is closed")
        return self._session.graphics

    def create_texture_lease(self):
        if self._adapter is None:
            raise RuntimeError("native window composition is closed")
        return dynamic_texture_lease(self._adapter)

    def pump_events(self) -> int:
        if self._manager is None or self._adapter is None or self._handle is None:
            raise RuntimeError("native window composition is closed")
        self._manager.pump_events()
        return self._adapter.consume_pending_events(self._manager, self._handle)

    def request_repaint(self) -> None:
        if self._adapter is None:
            raise RuntimeError("native window composition is closed")
        self._adapter.request_repaint()

    def set_unhandled_key_handler(
            self,
            callback: Callable[[int, int], bool] | None) -> None:
        if self._adapter is None:
            raise RuntimeError("native window composition is closed")
        self._adapter.set_unhandled_key_handler(callback)

    def set_window_title(self, title: str) -> None:
        if self._manager is None or self._handle is None:
            raise RuntimeError("native window composition is closed")
        self._manager.window(self._handle).set_title(title)

    def render_frame(self) -> bool:
        if self._adapter is None:
            raise RuntimeError("native window composition is closed")
        if not self._adapter.repaint_requested:
            return False
        return bool(self._adapter.render_frame())

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        adapter, self._adapter = self._adapter, None
        document, self._document = self._document, None
        manager, self._manager = self._manager, None
        handle, self._handle = self._handle, None
        session, self._session = self._session, None

        if adapter is not None:
            self._close_step("native adapter idle", adapter.wait_idle)
            self._close_step("native adapter", adapter.close)
        if document is not None and document.valid:
            self._close_step(
                "native document",
                lambda: tc_ui_document_destroy(document),
            )
        if manager is not None and handle is not None and manager.contains(handle):
            self._close_step(
                "native window",
                lambda: manager.destroy_window(handle),
            )
        if manager is not None:
            self._close_step("native window manager", manager.close)
        if session is not None:
            self._close_step("native graphics session", session.close)

    @staticmethod
    def _close_step(name: str, callback: Callable[[], Any]) -> None:
        try:
            callback()
        except Exception:
            log.exception(f"Failed to close {name}")


class NativeEditorRoot:
    """Own the native view, dispatcher phase and one rendering composition."""

    def __init__(
            self,
            application: EditorApplication,
            composition: NativeComposition,
            *,
            dispatcher: Dispatcher | None = None,
            dispatch_limit: int = DEFAULT_DISPATCH_LIMIT,
            command_handlers: Mapping[str, CommandHandler] | None = None,
            texture_lease_factory: Callable[[], Any] | None = None,
            reconstruction_engine: ReconstructionEngine | None = None,
            view_factory: NativeViewFactory = NativeEditorView) -> None:
        if dispatch_limit <= 0:
            raise ValueError("dispatch_limit must be positive")
        if application.closed:
            raise RuntimeError("cannot bind a closed editor application")

        self.application = application
        self.composition = composition
        self.dispatcher = dispatcher if dispatcher is not None else Dispatcher()
        self.dispatch_limit = dispatch_limit
        self.closed = False
        self._window_close_prompted = False
        self.discarded_on_close = 0
        handlers: dict[str, CommandHandler] = {
            "app.quit": application.request_stop,
        }
        if command_handlers is not None:
            handlers.update(command_handlers)
        self.canvas = None
        self.canvas_controls = None
        self.canvas_controls_coordinator = None
        self.canvas_status_coordinator = None
        self.canvas_edit_coordinator = None
        self.command_coordinator = None
        self.generation_panels = None
        self.generation_panels_coordinator = None
        self.layer_panel = None
        self.layer_tree_coordinator = None
        self.agent_chat = None
        self.agent_chat_coordinator = None
        self.dialogs = None
        self.dialog_coordinator = None
        self.reconstruction_engine = None
        self.reconstruction_controller = None
        self.reconstruction_viewport = None
        self.reconstruction_refine_viewport = None
        self._presented_refine_artifact_path: str | None = None
        self._presented_refine_reference_path: str | None = None
        self._presented_refine_run_id: str | None = None
        self._reconstruction_job_node_id: str | None = None
        self._reconstruction_parent_run_id: str | None = None
        self._presented_reconstruction_id: str | None = None
        self._active_reconstruction_workspace = None
        self.automation = None

        try:
            self.view = view_factory(
                composition.document,
                composition.request_repaint,
                self._set_window_title,
                handlers,
            )
            if hasattr(application, "layer_stack"):
                self.reconstruction_engine = (
                    reconstruction_engine or ReconstructionEngine()
                )
                self.reconstruction_controller = ReconstructionController(
                    self.reconstruction_engine
                )
                register_resource = getattr(
                    application, "register_shutdown_resource", None
                )
                if callable(register_resource):
                    register_resource(
                        ShutdownPhase.ENGINE_WORKERS,
                        "3d-reconstruction-engine",
                        self.reconstruction_engine.shutdown,
                    )
                set_handler = getattr(self.view, "set_command_handler", None)
                if callable(set_handler):
                    set_handler(
                        "layer.new_3d_reconstruction",
                        self._create_reconstruction_node,
                    )
                    set_handler("generation.3d", self._start_reconstruction)
                    set_handler(
                        "generation.3d_cancel", self._cancel_reconstruction
                    )
                    set_handler(
                        "view.3d_light_from_camera",
                        self._set_3d_light_from_camera,
                    )
                set_stage_handler = getattr(
                    self.view, "set_reconstruction_stage_handler", None
                )
                if callable(set_stage_handler):
                    set_stage_handler(self._select_reconstruction_stage)
                set_parameter_handler = getattr(
                    self.view, "set_reconstruction_parameter_handler", None
                )
                if callable(set_parameter_handler):
                    set_parameter_handler(self._set_reconstruction_parameter)
                set_refine_handler = getattr(
                    self.view, "set_reconstruction_refine_handler", None
                )
                if callable(set_refine_handler):
                    set_refine_handler(self._handle_reconstruction_refine)
                set_refine_placement_handler = getattr(
                    self.view,
                    "set_reconstruction_refine_placement_handler",
                    None,
                )
                if callable(set_refine_placement_handler):
                    set_refine_placement_handler(
                        self._handle_reconstruction_refine_placement
                    )
                set_workspace_handler = getattr(
                    self.view, "set_reconstruction_workspace_handler", None
                )
                if callable(set_workspace_handler):
                    set_workspace_handler(
                        self._handle_reconstruction_workspace
                    )
            mount_canvas = getattr(self.view, "mount_canvas", None)
            graphics = getattr(composition, "graphics", None)
            if mount_canvas is not None and graphics is not None:
                if texture_lease_factory is None:
                    raise RuntimeError(
                        "native canvas requires an explicit texture lease factory"
                    )
                self.canvas = NativeEditorCanvas(
                    composition.document,
                    application.layer_stack,
                    lease_factory=texture_lease_factory,
                    graphics_owner=graphics,
                    request_repaint=composition.request_repaint,
                )
                mount_canvas(self.canvas)
                application.register_shutdown_resource(
                    ShutdownPhase.GPU_RESOURCES,
                    "native-canvas",
                    self.canvas.close,
                )
                self.canvas_status_coordinator = CanvasStatusCoordinator(
                    application.layer_stack,
                    application.document,
                    self.canvas.controller,
                    application.set_status,
                    get_status=lambda: application.status_text,
                )
                application.register_shutdown_resource(
                    ShutdownPhase.VIEW_WORKERS,
                    "native-canvas-status",
                    self.canvas_status_coordinator.close,
                )
                self.canvas_edit_coordinator = (
                    CanvasEditTransactionCoordinator(
                        application.layer_stack,
                        application.document,
                        history_replaying=(
                            lambda: application.history_replaying),
                        on_history_changed=self._on_canvas_history_changed,
                        on_edit_cancelled=(
                            application.reconcile_document_session),
                        on_mutation_begin=application.mark_external_mutation,
                        cancel_interaction=(
                            self.canvas.cancel_pointer_interaction),
                    )
                )
                self.canvas_edit_coordinator.bind(self.canvas.controller)
                application.add_snapshot_listener(
                    self._on_snapshot_applied
                )
                application.register_shutdown_resource(
                    ShutdownPhase.VIEW_WORKERS,
                    "native-canvas-edit-transactions",
                    self.canvas_edit_coordinator.close,
                )
                mount_controls = getattr(
                    self.view, "mount_canvas_controls", None)
                if mount_controls is not None:
                    self.canvas_controls_coordinator = (
                        CanvasControlsCoordinator(
                            application.layer_stack,
                            application.document,
                            self.canvas.controller,
                        )
                    )
                    self.canvas_controls = NativeCanvasControls(
                        composition.document,
                        self.canvas_controls_coordinator.brush_state,
                        self.canvas_controls_coordinator.selection_state,
                        self.canvas_controls_coordinator.handle_brush_intent,
                        self.canvas_controls_coordinator.handle_selection_intent,
                        viewport_rect=lambda: self.view.root.bounds,
                    )
                    self.canvas_controls_coordinator.bind_view(
                        self.canvas_controls)
                    mount_controls(self.canvas_controls)
                    application.register_shutdown_resource(
                        ShutdownPhase.VIEW_WORKERS,
                        "native-canvas-controls-view",
                        self.canvas_controls.close,
                    )
                    application.register_shutdown_resource(
                        ShutdownPhase.VIEW_WORKERS,
                        "native-canvas-controls",
                        self.canvas_controls_coordinator.close,
                    )
                mount_generation_panels = getattr(
                    self.view, "mount_generation_panels", None)
                if mount_generation_panels is not None:
                    self.generation_panels_coordinator = (
                        GenerationPanelsCoordinator(
                            application,
                            self.canvas.controller,
                            canvas_controls=(
                                self.canvas_controls_coordinator),
                        )
                    )
                    self.generation_panels = NativeGenerationPanels(
                        composition.document,
                        self.generation_panels_coordinator.state,
                        self.generation_panels_coordinator.handle_intent,
                        composition.request_repaint,
                    )
                    self.generation_panels_coordinator.bind_view(
                        self.generation_panels)
                    mount_generation_panels(
                        self.generation_panels,
                        self.generation_panels_coordinator,
                    )
                mount_layer_panel = getattr(
                    self.view, "mount_layer_panel", None)
                if mount_layer_panel is not None:
                    self.layer_tree_coordinator = LayerTreeCoordinator(
                        application.layer_stack,
                        application.document,
                        tool_factory=self._create_default_layer_tool,
                        before_remove_layer=self._before_remove_layer,
                        before_detach_tool=self._before_detach_layer_tool,
                        set_status=application.set_status,
                    )
                    self.layer_panel = NativeLayerPanel(
                        composition.document,
                        self.layer_tree_coordinator.state,
                        self.layer_tree_coordinator.handle_intent,
                        viewport_rect=lambda: self.view.root.bounds,
                    )
                    self.layer_tree_coordinator.bind_view(self.layer_panel)
                    mount_layer_panel(self.layer_panel)
                    application.register_shutdown_resource(
                        ShutdownPhase.VIEW_WORKERS,
                        "native-layer-panel-view",
                        self.layer_panel.close,
                    )
                    application.register_shutdown_resource(
                        ShutdownPhase.VIEW_WORKERS,
                        "native-layer-tree",
                        self.layer_tree_coordinator.close,
                    )
                if self.generation_panels is not None:
                    application.register_shutdown_resource(
                        ShutdownPhase.VIEW_WORKERS,
                        "native-generation-panels-view",
                        self.generation_panels.close,
                    )
                    application.register_shutdown_resource(
                        ShutdownPhase.VIEW_WORKERS,
                        "native-generation-panels",
                        self.generation_panels_coordinator.close,
                    )
                mount_agent_chat = getattr(
                    self.view, "mount_agent_chat", None)
                if mount_agent_chat is not None:
                    self.agent_chat_coordinator = AgentChatCoordinator(
                        application.settings,
                        application.agent_tool_registry,
                        application.layer_stack,
                        application.document,
                        defer=self.defer,
                    )
                    self.agent_chat = NativeAgentChatPanel(
                        composition.document,
                        self.agent_chat_coordinator.state,
                        self.agent_chat_coordinator.handle_intent,
                        composition.request_repaint,
                    )
                    self.agent_chat_coordinator.bind_view(self.agent_chat)
                    mount_agent_chat(self.agent_chat)
                    application.register_shutdown_resource(
                        ShutdownPhase.VIEW_WORKERS,
                        "native-agent-chat",
                        self.agent_chat_coordinator.close,
                    )
                    application.register_shutdown_resource(
                        ShutdownPhase.VIEW_WORKERS,
                        "native-agent-chat-view",
                        self.agent_chat.close,
                    )
                self.dialog_coordinator = ApplicationDialogCoordinator(
                    application,
                    self.canvas,
                    on_models_dir_changed=(
                        self.generation_panels_coordinator.refresh_models
                        if self.generation_panels_coordinator is not None
                        else None),
                )
                self.dialogs = NativeApplicationDialogs(
                    composition.document,
                    viewport_rect=lambda: self.view.root.bounds,
                    request_repaint=composition.request_repaint,
                )
                self.dialog_coordinator.bind_view(self.dialogs)
                for command_id, handler in (
                        self.dialog_coordinator.command_handlers.items()):
                    self.view.set_command_handler(command_id, handler)
                self.command_coordinator = EditorCommandCoordinator(
                    application,
                    fit_in_view=self.canvas.fit_in_view,
                    request_remove_layer=(
                        self.layer_panel.request_active_delete
                        if self.layer_panel is not None else None
                    ),
                    before_mutation=self.canvas.cancel_pointer_interaction,
                )
                for command_id, handler in (
                        self.command_coordinator.handlers.items()):
                    self.view.set_command_handler(command_id, handler)
                if command_handlers is not None:
                    for command_id, handler in command_handlers.items():
                        self.view.set_command_handler(command_id, handler)
                application.register_shutdown_resource(
                    ShutdownPhase.VIEW_WORKERS,
                    "native-dialog-coordinator",
                    self.dialog_coordinator.close,
                )
                application.register_shutdown_resource(
                    ShutdownPhase.VIEW_WORKERS,
                    "native-dialogs",
                    self.dialogs.close,
                )
            else:
                self.canvas = None
            composition.set_unhandled_key_handler(self.view.dispatch_shortcut)
            application.bind_view(self.view.ports())
            self.automation = start_editor_automation(self)
            if self.automation is not None:
                application.register_shutdown_resource(
                    ShutdownPhase.VIEW_WORKERS,
                    "diffusion-editor-mcp",
                    self.automation.close,
                )
            composition.request_repaint()
        except Exception:
            if self.automation is not None:
                self.automation.close()
            if self.canvas_edit_coordinator is not None:
                self.canvas_edit_coordinator.close()
            if self.dialog_coordinator is not None:
                self.dialog_coordinator.close()
            if self.dialogs is not None:
                self.dialogs.close()
            if self.agent_chat_coordinator is not None:
                self.agent_chat_coordinator.close()
            if self.agent_chat is not None:
                self.agent_chat.close()
            if self.layer_panel is not None:
                self.layer_panel.close()
            if self.layer_tree_coordinator is not None:
                self.layer_tree_coordinator.close()
            if self.generation_panels is not None:
                self.generation_panels.close()
            if self.generation_panels_coordinator is not None:
                self.generation_panels_coordinator.close()
            if self.canvas_controls is not None:
                self.canvas_controls.close()
            if self.canvas_controls_coordinator is not None:
                self.canvas_controls_coordinator.close()
            if self.canvas_status_coordinator is not None:
                self.canvas_status_coordinator.close()
            if self.canvas is not None:
                self.canvas.close()
            self.dispatcher.close()
            self.dispatcher.discard_pending()
            composition.close()
            raise

    def _create_default_layer_tool(
            self, layer: Layer, tool_type: str):
        if self.canvas is None or not layer.accepts_pixel_edits:
            return None
        composite = self.canvas.get_composite_below(layer)
        if composite is None:
            composite = self.canvas.controller.get_composite()
        if composite is None:
            return None
        if tool_type == "instruct":
            patch = source_patch_from_full_composite(composite)
        else:
            center_x, center_y = self.canvas.view_center_image()
            patch = source_patch_at_center(composite, center_x, center_y)
        x0, y0, x1, y1 = patch.canvas_rect
        common = {
            "source_patch": patch.image,
            "patch_x": x0,
            "patch_y": y0,
            "patch_w": x1 - x0,
            "patch_h": y1 - y0,
        }
        if tool_type == "diffusion":
            engine = self.application.engines.diffusion
            return DiffusionTool(
                **common,
                prompt="",
                negative_prompt="",
                strength=0.3,
                guidance_scale=7.0,
                steps=20,
                seed=-1,
                model_path=str(getattr(engine, "model_path", "") or ""),
                prediction_type="",
                mode="inpaint",
            )
        if tool_type == "lama":
            return LamaTool(**common)
        if tool_type == "instruct":
            from ..generation.image_edit_profiles import (
                DEFAULT_IMAGE_EDIT_PROFILE_ID,
            )
            return InstructTool(
                **common,
                model_profile_id=DEFAULT_IMAGE_EDIT_PROFILE_ID,
            )
        return None

    def _before_detach_layer_tool(self, layer: Layer) -> None:
        for controller in (
                self.application.diffusion_controller,
                self.application.lama_controller,
                self.application.instruct_controller,
                self.application.segmentation_controller,
                self.application.grounding_controller):
            controller.clear_pending_layer(layer)

    def _before_remove_layer(self, layer: Layer) -> None:
        removed_layers = (layer, *layer.all_descendants())
        removed_ids = {removed.id for removed in removed_layers}
        if self._reconstruction_job_node_id in removed_ids:
            controller = self.reconstruction_controller
            if controller is not None:
                controller.cancel()
        if self._presented_reconstruction_id in removed_ids:
            if self.reconstruction_viewport is not None:
                self.reconstruction_viewport.clear_model()
            self._presented_reconstruction_id = None
        if any(isinstance(removed, ReconstructionLayer)
               for removed in removed_layers):
            self._hide_reconstruction_refine_output()
        for removed in removed_layers:
            self.application.invalidate_generation_jobs_for_layer(
                removed.id,
                "target layer was deleted",
            )

    @classmethod
    def create_headless(
            cls,
            application: EditorApplication,
            *,
            width: int = 320,
            height: int = 200,
            backend: str = "vulkan",
            dispatch_limit: int = DEFAULT_DISPATCH_LIMIT,
            command_handlers: Mapping[str, CommandHandler] | None = None,
            sdk_root: Path | None = None) -> "NativeEditorRoot":
        root = resolve_sdk() if sdk_root is None else sdk_root.expanduser().resolve()
        composition = OffscreenGuiComposition(
            width=width,
            height=height,
            backend=backend,
            font_path=str(bundled_native_font_path(root)),
            continuous_rendering=False,
            sdk_root=str(root),
        )
        return cls(
            application,
            composition,
            dispatch_limit=dispatch_limit,
            command_handlers=command_handlers,
            texture_lease_factory=partial(DynamicTextureLease, composition),
        )

    @classmethod
    def create_windowed(
            cls,
            application: EditorApplication,
            *,
            title: str = "Diffusion Editor (native migration)",
            width: int = DEFAULT_NATIVE_WIDTH,
            height: int = DEFAULT_NATIVE_HEIGHT,
            dispatch_limit: int = DEFAULT_DISPATCH_LIMIT,
            command_handlers: Mapping[str, CommandHandler] | None = None,
            sdk_root: Path | None = None) -> "NativeEditorRoot":
        composition = WindowedNativeComposition(
            title=title,
            width=width,
            height=height,
            sdk_root=sdk_root,
        )
        return cls(
            application,
            composition,
            dispatch_limit=dispatch_limit,
            command_handlers=command_handlers,
            texture_lease_factory=composition.create_texture_lease,
        )

    def _set_window_title(self, title: str) -> None:
        setter = getattr(self.composition, "set_window_title", None)
        if setter is not None:
            setter(title)

    def _on_canvas_history_changed(self) -> None:
        self._refresh_commands()

    def defer(self, callback: Callable[[], None]):
        if self.closed:
            raise RuntimeError("native editor root is closed")
        return self.dispatcher.defer(callback)

    def tick(self) -> NativeTickResult:
        if self.closed:
            raise RuntimeError("native editor root is closed")

        events = int(self.composition.pump_events())
        if self.automation is not None:
            self.automation.process_pending()
        stats: DispatchStats = self.dispatcher.run_pending(self.dispatch_limit)
        if stats.busy or stats.internal_error:
            raise RuntimeError("native dispatcher failed to drain cleanly")
        if self.composition.should_close and not self._window_close_prompted:
            self._window_close_prompted = True
            if self.dialog_coordinator is not None:
                self.dialog_coordinator.request_quit()
            else:
                self.application.request_stop()
        if self.application.running:
            self.application.poll()
        self._poll_reconstruction()
        if self.agent_chat is not None:
            self.agent_chat.poll()
        if self.canvas_status_coordinator is not None:
            self.canvas_status_coordinator.flush()
        self._refresh_commands()
        if self.reconstruction_viewport is not None:
            self.reconstruction_viewport.render_if_dirty()
        if self.reconstruction_refine_viewport is not None:
            self.reconstruction_refine_viewport.render_if_dirty()
        rendered = bool(self.composition.render_frame())
        return NativeTickResult(
            dispatched=stats.executed,
            dispatch_failed=stats.failed,
            dispatch_remaining=stats.remaining,
            events=events,
            rendered=rendered,
        )

    def _refresh_commands(self) -> None:
        if self.command_coordinator is not None:
            self.command_coordinator.refresh()
        view = getattr(self, "view", None)
        set_state = getattr(view, "set_command_state", None)
        controller = self.reconstruction_controller
        stack = getattr(self.application, "layer_stack", None)
        if callable(set_state) and controller is not None and stack is not None:
            busy = controller.is_busy
            has_composite = stack.width > 0 and stack.height > 0
            active = stack.active_layer
            active_reconstruction = (
                active if isinstance(active, ReconstructionLayer) else None
            )
            set_state(
                "layer.new_3d_reconstruction",
                enabled=has_composite and not busy,
            )
            set_state(
                "generation.3d",
                enabled=(
                    active_reconstruction is not None
                    and has_composite
                    and not busy
                ),
            )
            set_state("generation.3d_cancel", enabled=busy)
            set_state(
                "view.3d_light_from_camera",
                enabled=(
                    active_reconstruction is not None
                    and self.reconstruction_viewport is not None
                    and self.reconstruction_viewport.mesh_count > 0
                    and self._presented_reconstruction_id
                    == active_reconstruction.id
                ),
            )
            self._sync_reconstruction_selection(active_reconstruction)

    def _create_reconstruction_node(self) -> None:
        stack = self.application.layer_stack
        if stack.width <= 0 or stack.height <= 0:
            return
        self.application.document.execute(AddReconstructionLayerCommand(
            stack.next_name("3D Reconstruction")
        ))
        node = stack.active_layer
        if not isinstance(node, ReconstructionLayer):
            raise RuntimeError("failed to create a reconstruction node")
        self._presented_reconstruction_id = None
        self._sync_reconstruction_selection(node)
        self.application.set_status(f"Created: {node.name}")

    def _start_reconstruction(self) -> None:
        controller = self.reconstruction_controller
        if controller is None or controller.is_busy:
            return
        stack = self.application.layer_stack
        node = stack.active_layer
        if not isinstance(node, ReconstructionLayer):
            self.application.set_status(
                "Select a 3D Reconstruction object before generation"
            )
            return
        try:
            self._ensure_reconstruction_viewport()
            composite = stack.composite()
            image = Image.fromarray(composite, mode="RGBA")
            source_sha256 = hashlib.sha256(image.tobytes()).hexdigest()
            supported = RECONSTRUCTION_BACKEND_STAGES[
                node.generation_parameters.backend
            ]
            resume_stage = node.resume_stage
            compatible_parameters = pixal3d_resume_parameters_compatible(
                node.resume_parameters,
                node.generation_parameters,
                resume_stage,
            )
            can_resume = bool(
                node.generation_parameters.backend is ReconstructionBackend.PIXAL3D
                and resume_stage in supported
                and node.target_stage in supported
                and supported.index(node.target_stage) > supported.index(resume_stage)
                and node.resume_checkpoint_path
                and os.path.isfile(node.resume_checkpoint_path)
                and node.resume_source_sha256 == source_sha256
                and compatible_parameters
            )
            node.begin_staged_generation(resume_stage if can_resume else None)
            self._refresh_reconstruction_panel(node)
            start_kwargs = {
                "parameters": node.generation_parameters,
                "target_stage": node.target_stage,
            }
            if can_resume:
                start_kwargs["resume_checkpoint_path"] = (
                    node.resume_checkpoint_path
                )
            event = controller.start(image, **start_kwargs)
            if event.status:
                if not can_resume:
                    node.lr_variants = ()
                    node.accepted_lr_variant_id = None
                    node.selected_lr_refine_source_id = None
                self._reconstruction_job_source_sha256 = source_sha256
                self._reconstruction_job_parameters = node.generation_parameters
                self._reconstruction_job_node_id = node.id
                self._reconstruction_parent_run_id = None
                self._reconstruction_lr_parent_variant_id = None
                self._set_reconstruction_status(
                    node, ReconstructionStatus.GENERATING
                )
                self.application.set_status(event.status)
        except Exception as exc:
            log.exception("Failed to start 3D reconstruction")
            self.application.set_status(f"Cannot start 3D generation: {exc}")

    def start_reconstruction_refine(
        self,
        node: ReconstructionLayer,
        conditioning_image: Image.Image,
        mask_image: Image.Image,
        *,
        parameters=None,
        base_checkpoint_path: str | None = None,
    ):
        """Start a masked refinement from a canvas-sized source and mask."""
        controller = self.reconstruction_controller
        if controller is None or controller.is_busy:
            return None
        if self.application.layer_stack.find_layer_by_id(node.id) is not node:
            raise ValueError("reconstruction node does not belong to the document")
        parent = node.active_run or node.base_run
        if base_checkpoint_path:
            parent = next(
                (
                    run for run in node.runs
                    if run.checkpoint_path == base_checkpoint_path
                ),
                parent,
            )
        if (
            parent is not None
            and parent.backend is not ReconstructionBackend.PIXAL3D
        ):
            raise ValueError("masked refinement is not available for TRELLIS.2")
        if (
            parent is None
            and node.generation_parameters.backend
            is not ReconstructionBackend.PIXAL3D
        ):
            raise ValueError("masked refinement requires a Pixal3D checkpoint")
        checkpoint_path = (
            base_checkpoint_path
            or (parent.checkpoint_path if parent is not None else None)
        )
        if not checkpoint_path:
            raise ValueError("reconstruction has no HR refine checkpoint")
        if not os.path.isfile(checkpoint_path):
            raise ValueError("active reconstruction refine checkpoint is missing")
        event = controller.start_refine(
            conditioning_image,
            mask_image,
            checkpoint_path,
            parameters=parameters,
            generation_parameters=node.generation_parameters,
        )
        if event.status and event.error is None:
            self._reconstruction_job_node_id = node.id
            self._reconstruction_parent_run_id = (
                parent.run_id if parent is not None else None
            )
            self._set_reconstruction_status(node, ReconstructionStatus.GENERATING)
            self.application.set_status(event.status)
        elif event.status:
            self.application.set_status(event.status)
        return event

    def start_reconstruction_texture_refine(
        self,
        node: ReconstructionLayer,
        conditioning_image: Image.Image,
        mask_image: Image.Image,
        *,
        parameters=None,
        shape_checkpoint_path: str | None = None,
        texture_checkpoint_path: str | None = None,
    ):
        controller = self.reconstruction_controller
        if controller is None or controller.is_busy:
            return None
        if self.application.layer_stack.find_layer_by_id(node.id) is not node:
            raise ValueError("reconstruction node does not belong to the document")
        parent = node.active_run or node.base_run
        if (
            parent is not None
            and parent.backend is not ReconstructionBackend.PIXAL3D
        ):
            raise ValueError("texture refinement is not available for TRELLIS.2")
        if (
            parent is None
            and node.generation_parameters.backend
            is not ReconstructionBackend.PIXAL3D
        ):
            raise ValueError("texture refinement requires Pixal3D checkpoints")
        shape_checkpoint = (
            shape_checkpoint_path
            or (parent.checkpoint_path if parent is not None else None)
        )
        texture_checkpoint = (
            texture_checkpoint_path
            or (parent.texture_checkpoint_path if parent is not None else None)
        )
        if not shape_checkpoint:
            raise ValueError("active run has no shape checkpoint")
        if not texture_checkpoint:
            raise ValueError("active run has no texture checkpoint")
        for label, path in (
            ("shape", shape_checkpoint),
            ("texture", texture_checkpoint),
        ):
            if not os.path.isfile(path):
                raise ValueError(f"active run {label} checkpoint is missing")
        event = controller.start_texture_refine(
            conditioning_image,
            mask_image,
            shape_checkpoint,
            texture_checkpoint,
            parameters=parameters,
            generation_parameters=node.generation_parameters,
        )
        if event.status and event.error is None:
            self._reconstruction_job_node_id = node.id
            self._reconstruction_parent_run_id = (
                parent.run_id if parent is not None else None
            )
            self._set_reconstruction_status(node, ReconstructionStatus.GENERATING)
            self.application.set_status(event.status)
        elif event.status:
            self.application.set_status(event.status)
        return event

    def _start_selected_reconstruction_refine(
            self, node: ReconstructionLayer) -> None:
        stack = self.application.layer_stack
        parent = _hr_refine_source_run(node)
        source_path = (
            parent.source_path if parent is not None
            else node.intermediate_source_path
        )
        checkpoint_path = (
            parent.checkpoint_path if parent is not None
            else node.intermediate_shape_checkpoint_path
        )
        if not source_path:
            self.application.set_status(
                "Cannot refine: the selected version has no source image"
            )
            return
        if not os.path.isfile(source_path):
            self.application.set_status(
                "Cannot refine: the selected version source image is missing"
            )
            return
        if stack.selection.is_empty:
            self.application.set_status(
                "Paint a refine mask on the image first"
            )
            return
        try:
            with Image.open(source_path) as opened:
                source_image = opened.convert("RGBA").copy()
            if source_image.size != (stack.width, stack.height):
                raise ValueError(
                    "source image dimensions no longer match the canvas"
                )
            mask_image = Image.fromarray(
                stack.selection.to_mask().to_uint8(), mode="L"
            )
            self._set_reconstruction_mask_painting(False)
            self.start_reconstruction_refine(
                node,
                source_image,
                mask_image,
                parameters=node.refine_parameters,
                base_checkpoint_path=checkpoint_path,
            )
            self._refresh_reconstruction_panel(node)
        except Exception as exc:
            log.exception("Failed to start masked 3D refinement")
            self.application.set_status(f"Cannot start 3D refinement: {exc}")

    def _start_selected_reconstruction_lr_refine(
            self, node: ReconstructionLayer) -> None:
        stack = self.application.layer_stack
        controller = self.reconstruction_controller
        source_path = node.intermediate_source_path
        checkpoint_path = node.resume_checkpoint_path
        source_variant = node.selected_lr_refine_source
        if source_variant is not None:
            source_path = source_variant.source_path
            checkpoint_path = source_variant.checkpoint_path
        if controller is None or controller.is_busy:
            return
        if node.generation_parameters.backend is not ReconstructionBackend.PIXAL3D:
            self.application.set_status("LR refine requires the Pixal3D backend")
            return
        if node.resume_stage not in {
                ReconstructionStage.LR_SHAPE_FLOW,
                ReconstructionStage.LR_SHAPE_LATENT,
        }:
            self.application.set_status(
                "Generate through LR shape latent before refining LR"
            )
            return
        if not source_path or not os.path.isfile(source_path):
            self.application.set_status("Cannot refine LR: source image is missing")
            return
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            self.application.set_status("Cannot refine LR: checkpoint is missing")
            return
        if stack.selection.is_empty:
            self.application.set_status("Paint an LR refine mask first")
            return
        try:
            with Image.open(source_path) as opened:
                source_image = opened.convert("RGBA").copy()
            if source_image.size != (stack.width, stack.height):
                raise ValueError(
                    "source image dimensions no longer match the canvas"
                )
            mask_image = Image.fromarray(
                stack.selection.to_mask().to_uint8(), mode="L"
            )
            self._set_reconstruction_mask_painting(False)
            event = controller.start_lr_refine(
                source_image,
                mask_image,
                checkpoint_path,
                parameters=node.lr_refine_parameters,
                generation_parameters=node.generation_parameters,
            )
            if event.status and event.error is None:
                self._reconstruction_job_node_id = node.id
                self._reconstruction_parent_run_id = None
                self._reconstruction_lr_parent_variant_id = (
                    source_variant.variant_id if source_variant else None
                )
                self._reconstruction_job_source_sha256 = (
                    node.resume_source_sha256
                )
                self._reconstruction_job_parameters = node.generation_parameters
                self._set_reconstruction_status(
                    node, ReconstructionStatus.GENERATING
                )
                self.application.set_status(event.status)
            elif event.status:
                self.application.set_status(event.status)
            self._refresh_reconstruction_panel(node)
        except Exception as exc:
            log.exception("Failed to start masked LR refinement")
            self.application.set_status(f"Cannot start LR refinement: {exc}")

    def _start_selected_reconstruction_texture_refine(
            self,
            node: ReconstructionLayer,
            *,
            parameters=None,
    ) -> None:
        stack = self.application.layer_stack
        parent = node.active_run or node.base_run
        source_path = (
            parent.source_path if parent is not None
            else node.intermediate_source_path
        )
        shape_checkpoint = (
            parent.checkpoint_path if parent is not None
            else node.intermediate_shape_checkpoint_path
        )
        texture_checkpoint = (
            parent.texture_checkpoint_path if parent is not None
            else node.intermediate_texture_checkpoint_path
        )
        if not source_path:
            self.application.set_status(
                "Cannot refine texture: the selected version has no source image"
            )
            return
        if not os.path.isfile(source_path):
            self.application.set_status(
                "Cannot refine texture: source image is missing"
            )
            return
        if stack.selection.is_empty:
            self.application.set_status("Paint a texture refine mask first")
            return
        try:
            with Image.open(source_path) as opened:
                source_image = opened.convert("RGBA").copy()
            if source_image.size != (stack.width, stack.height):
                raise ValueError(
                    "source image dimensions no longer match the canvas"
                )
            mask_image = Image.fromarray(
                stack.selection.to_mask().to_uint8(), mode="L"
            )
            self._set_reconstruction_mask_painting(False)
            self.start_reconstruction_texture_refine(
                node,
                source_image,
                mask_image,
                parameters=parameters or node.refine_parameters,
                shape_checkpoint_path=shape_checkpoint,
                texture_checkpoint_path=texture_checkpoint,
            )
            self._refresh_reconstruction_panel(node)
        except Exception as exc:
            log.exception("Failed to start masked 3D texture refinement")
            self.application.set_status(
                f"Cannot start 3D texture refinement: {exc}"
            )

    def _cancel_reconstruction(self) -> None:
        controller = self.reconstruction_controller
        if controller is not None and controller.cancel():
            self.application.set_status("Cancelling 3D generation...")

    def _set_3d_light_from_camera(self) -> None:
        viewport = self.reconstruction_viewport
        active = self.application.layer_stack.active_layer
        if (
                viewport is None
                or viewport.mesh_count == 0
                or not isinstance(active, ReconstructionLayer)
                or self._presented_reconstruction_id != active.id):
            return
        viewport.light_from_camera()
        self.application.set_status("3D light set from current camera")

    def _poll_reconstruction(self) -> None:
        controller = self.reconstruction_controller
        if controller is None:
            return
        event = controller.poll()
        if event is None:
            return
        node_id = self._reconstruction_job_node_id
        node = self.application.layer_stack.find_layer_by_id(node_id or "")
        if not isinstance(node, ReconstructionLayer):
            self.application.set_status("Ignored 3D result for a deleted object")
            return
        if event.stage_event is not None:
            node.apply_stage_event(event.stage_event)
            self._refresh_reconstruction_panel(node)
            artifact = event.stage_event.artifact
            if (
                artifact is not None
                and event.stage_event.stage is node.selected_preview_stage
                and self.application.layer_stack.active_layer is node
            ):
                try:
                    if self._load_reconstruction_artifact(artifact):
                        self._presented_reconstruction_id = node.id
                except Exception:
                    log.exception("Failed to display reconstruction stage preview")
            return
        self._reconstruction_job_node_id = None
        parent_run_id = getattr(self, "_reconstruction_parent_run_id", None)
        self._reconstruction_parent_run_id = None
        lr_parent_variant_id = getattr(
            self, "_reconstruction_lr_parent_variant_id", None
        )
        self._reconstruction_lr_parent_variant_id = None
        if event.result is not None:
            if event.result.resume_checkpoint_path:
                node.resume_checkpoint_path = event.result.resume_checkpoint_path
                node.resume_stage = event.result.completed_stage
                node.resume_source_sha256 = getattr(
                    self, "_reconstruction_job_source_sha256", None
                )
                node.resume_parameters = getattr(
                    self, "_reconstruction_job_parameters", None
                )
            if event.result.source_path:
                node.intermediate_source_path = event.result.source_path
            if (
                    event.result.checkpoint_path
                    and not _is_composite_shape_checkpoint(
                        event.result.checkpoint_path
                    )):
                node.intermediate_shape_checkpoint_path = (
                    event.result.checkpoint_path
                )
            if event.result.texture_checkpoint_path:
                node.intermediate_texture_checkpoint_path = (
                    event.result.texture_checkpoint_path
                )
            if (
                    event.result.completed_stage
                    is ReconstructionStage.LR_SHAPE_LATENT
                    and event.result.resume_checkpoint_path):
                preview = next(
                    (
                        artifact for artifact in event.result.artifacts
                        if artifact.stage
                        is ReconstructionStage.LR_SHAPE_LATENT
                    ),
                    None,
                )
                if preview is not None:
                    if event.result.kind is ReconstructionRunKind.BASE:
                        variant = ReconstructionLrVariant(
                            "lr-base",
                            "Base LR",
                            event.result.resume_checkpoint_path,
                            event.result.source_path,
                            preview,
                        )
                        node.lr_variants = (variant,)
                        node.selected_lr_refine_source_id = variant.variant_id
                    else:
                        refine_number = 1 + sum(
                            item.parent_variant_id is not None
                            for item in node.lr_variants
                        )
                        variant = ReconstructionLrVariant(
                            f"lr-refined-{refine_number}",
                            f"Refined LR {refine_number}",
                            event.result.resume_checkpoint_path,
                            event.result.source_path,
                            preview,
                            parent_variant_id=lr_parent_variant_id,
                            refine_generated_path=(
                                event.result.refine_generated_path
                            ),
                        )
                        node.lr_variants = (*node.lr_variants, variant)
                    node.accepted_lr_variant_id = variant.variant_id
            if event.result.completed_stage is not ReconstructionStage.FINAL_MESH:
                self._set_reconstruction_status(node, ReconstructionStatus.READY)
                self._refresh_reconstruction_panel(node)
                self.application.set_status(
                    "3D generation stopped after "
                    f"{event.result.completed_stage.value.replace('_', ' ')}"
                )
                return
            try:
                vertices, triangles, meshes = self._read_reconstruction_glb_stats(
                    event.result.glb_path
                )
                self.application.document.execute(
                    PublishReconstructionResultCommand(
                        node,
                        event.result.glb_path,
                        vertices,
                        triangles,
                        meshes,
                        source_path=event.result.source_path,
                        conditioning_path=event.result.conditioning_path,
                        checkpoint_path=event.result.checkpoint_path,
                        texture_checkpoint_path=(
                            event.result.texture_checkpoint_path
                        ),
                        run_kind=event.result.kind,
                        parent_run_id=(
                            parent_run_id
                            if event.result.kind is not ReconstructionRunKind.BASE
                            else None
                        ),
                        backend=event.result.backend,
                        refine_generated_path=(
                            event.result.refine_generated_path
                        ),
                        refine_placement=event.result.refine_placement,
                        refine_placement_pivot=(
                            event.result.refine_placement_pivot
                        ),
                        refine_placement_accepted=(
                            event.result.refine_placement_accepted
                        ),
                    )
                )
                if (
                    self.application.layer_stack.active_layer is node
                    and node.selected_preview_stage
                    is ReconstructionStage.FINAL_MESH
                ):
                    self._ensure_reconstruction_viewport().load_glb(
                        event.result.glb_path
                    )
                    self._presented_reconstruction_id = node.id
                self._refresh_reconstruction_panel(node)
                self.application.set_status(
                    "3D model ready: "
                    f"{vertices:,} vertices, {triangles:,} triangles, "
                    f"{meshes} mesh(es)"
                )
            except Exception as exc:
                self._set_reconstruction_status(
                    node, ReconstructionStatus.FAILED
                )
                log.exception("Failed to display generated GLB")
                self.application.set_status(f"3D viewport error: {exc}")
        elif event.status:
            if event.error:
                for stage, stage_status in node.stage_statuses.items():
                    if stage_status is ReconstructionStageStatus.RUNNING:
                        node.stage_statuses[stage] = (
                            ReconstructionStageStatus.FAILED
                        )
                self._set_reconstruction_status(
                    node, ReconstructionStatus.FAILED
                )
                self._refresh_reconstruction_panel(node)
            self.application.set_status(event.status)

    @staticmethod
    def _read_reconstruction_glb_stats(path: str) -> tuple[int, int, int]:
        from ..native_glb import inspect_glb_stats

        return inspect_glb_stats(path)

    def _select_reconstruction_stage(self, stage: ReconstructionStage) -> None:
        node = self.application.layer_stack.active_layer
        if not isinstance(node, ReconstructionLayer):
            return
        node.selected_preview_stage = stage
        node.preview_stage_pinned = True
        artifact = node.stage_artifacts.get(stage)
        if artifact is not None:
            if self._load_reconstruction_artifact(artifact):
                self._presented_reconstruction_id = node.id
        else:
            node.target_stage = stage
        self._refresh_reconstruction_panel(node)

    def _set_reconstruction_mask_painting(self, enabled: bool) -> None:
        coordinator = self.canvas_controls_coordinator
        if coordinator is None:
            return
        coordinator.handle_selection_intent(SelectionControlsIntent(
            SelectionControlAction.EDIT_MODE,
            bool(enabled),
        ))
        if enabled:
            coordinator.handle_selection_intent(SelectionControlsIntent(
                SelectionControlAction.SHOW,
                True,
            ))

    def _handle_reconstruction_refine(self, action: str, value=None) -> None:
        node = self.application.layer_stack.active_layer
        controller = self.reconstruction_controller
        if (
            not isinstance(node, ReconstructionLayer)
            or (controller is not None and controller.is_busy)
        ):
            return
        coordinator = self.canvas_controls_coordinator
        try:
            if action == "paint":
                self._set_reconstruction_mask_painting(bool(value))
            elif action == "erase" and coordinator is not None:
                coordinator.handle_selection_intent(SelectionControlsIntent(
                    SelectionControlAction.ERASER, bool(value)
                ))
            elif action in {"brush_size", "brush_hardness", "brush_flow"}:
                if coordinator is None:
                    return
                selection_action = {
                    "brush_size": SelectionControlAction.SIZE,
                    "brush_hardness": SelectionControlAction.HARDNESS,
                    "brush_flow": SelectionControlAction.FLOW,
                }[action]
                coordinator.handle_selection_intent(SelectionControlsIntent(
                    selection_action, value
                ))
            elif action in {
                "strength", "steps", "seed", "resize_detail_to_1024"
            }:
                normalized = (
                    int(value)
                    if action in {"steps", "seed"}
                    else bool(value)
                    if action == "resize_detail_to_1024"
                    else float(value)
                )
                node.refine_parameters = replace(
                    node.refine_parameters, **{action: normalized}
                )
            elif action.startswith(("lr_", "hr_", "texture_")):
                scope, parameter_key = action.split("_", 1)
                if parameter_key not in {
                        "strength", "steps", "seed", "local_resolution",
                        "resize_detail_to_1024"}:
                    return
                normalized = (
                    int(value)
                    if parameter_key in {"steps", "seed", "local_resolution"}
                    and value is not None
                    else bool(value)
                    if parameter_key == "resize_detail_to_1024"
                    else float(value)
                    if parameter_key == "strength"
                    else None
                )
                attribute = {
                    "lr": "lr_refine_parameters",
                    "hr": "refine_parameters",
                    "texture": "texture_refine_parameters",
                }[scope]
                setattr(
                    node,
                    attribute,
                    replace(
                        getattr(node, attribute),
                        **{parameter_key: normalized},
                    ),
                )
            elif action == "clear":
                self.application.document.execute(ClearSelectionCommand())
            elif action == "run":
                self._start_selected_reconstruction_refine(node)
                return
            elif action == "run_lr":
                self._start_selected_reconstruction_lr_refine(node)
                return
            elif action == "run_texture":
                self._start_selected_reconstruction_texture_refine(node)
                return
            elif action == "run_texture_workspace":
                self._start_selected_reconstruction_texture_refine(
                    node, parameters=node.texture_refine_parameters
                )
                return
            elif action == "select_run" and value is not None:
                self.application.document.execute(
                    SelectReconstructionRunCommand(node, str(value))
                )
                artifact = node.stage_artifacts.get(
                    node.selected_preview_stage
                )
                if artifact is not None and self._load_reconstruction_artifact(
                    artifact
                ):
                    self._presented_reconstruction_id = node.id
                self.application.set_status("3D reconstruction version selected")
            else:
                return
        except (TypeError, ValueError) as exc:
            self.application.set_status(f"Invalid refine setting: {exc}")
            return
        self._refresh_reconstruction_panel(node)

    def _set_reconstruction_parameter(self, key: str, value) -> None:
        node = self.application.layer_stack.active_layer
        controller = self.reconstruction_controller
        if (
            not isinstance(node, ReconstructionLayer)
            or (controller is not None and controller.is_busy)
        ):
            return
        try:
            node.generation_parameters = replace(
                node.generation_parameters, **{key: value}
            )
            supported = RECONSTRUCTION_BACKEND_STAGES[
                node.generation_parameters.backend
            ]
            if node.target_stage not in supported:
                node.target_stage = ReconstructionStage.FINAL_MESH
            if node.selected_preview_stage not in supported:
                node.selected_preview_stage = ReconstructionStage.SOURCE_IMAGE
        except (TypeError, ValueError):
            return
        self._refresh_reconstruction_panel(node)

    def _refresh_reconstruction_panel(self, node: ReconstructionLayer) -> None:
        update = getattr(
            getattr(self, "view", None),
            "update_reconstruction_stages",
            None,
        )
        if callable(update):
            busy = bool(
                self.reconstruction_controller
                and self.reconstruction_controller.is_busy
            )
            update(
                node.stage_statuses,
                node.stage_progress,
                node.target_stage,
                node.selected_preview_stage,
                busy=busy,
                backend=node.generation_parameters.backend,
            )
            update_parameters = getattr(
                self.view, "update_reconstruction_parameters", None
            )
            if callable(update_parameters):
                update_parameters(node.generation_parameters, busy=busy)
            update_workspace = getattr(
                self.view, "update_reconstruction_workspace", None
            )
            if callable(update_workspace):
                workspace = build_legacy_workspace(
                    node.generation_parameters,
                    node.stage_statuses,
                    node.stage_artifacts,
                    node.runs,
                    node.active_run_id,
                    lr_variants=node.lr_variants,
                    selected_lr_refine_source_id=(
                        node.selected_lr_refine_source_id
                    ),
                )
                self._active_reconstruction_workspace = workspace
                update_workspace(
                    node.generation_parameters.backend,
                    workspace,
                    node.generation_parameters,
                    busy=busy,
                )
                self._sync_reconstruction_refine_output(
                    workspace,
                    getattr(
                        self.view,
                        "_selected_reconstruction_workspace_operation",
                        None,
                    ),
                    getattr(
                        self.view,
                        "_selected_reconstruction_workspace_operation_id",
                        None,
                    ),
                )
            update_refine = getattr(
                self.view, "update_reconstruction_refine", None
            )
            if callable(update_refine):
                parent = node.active_run or node.base_run
                hr_refine_source = _hr_refine_source_run(node)
                selected_backend = (
                    parent.backend
                    if parent
                    else node.generation_parameters.backend
                )
                refine_supported = (
                    selected_backend is ReconstructionBackend.PIXAL3D
                )
                if hr_refine_source is not None:
                    can_refine = bool(
                        hr_refine_source.backend
                        is ReconstructionBackend.PIXAL3D
                        and hr_refine_source.checkpoint_path
                        and os.path.isfile(hr_refine_source.checkpoint_path)
                        and hr_refine_source.source_path
                        and os.path.isfile(hr_refine_source.source_path)
                    )
                else:
                    can_refine = bool(
                        node.generation_parameters.backend
                        is ReconstructionBackend.PIXAL3D
                        and node.intermediate_shape_checkpoint_path
                        and os.path.isfile(
                            node.intermediate_shape_checkpoint_path
                        )
                        and not _is_composite_shape_checkpoint(
                            node.intermediate_shape_checkpoint_path
                        )
                        and node.intermediate_source_path
                        and os.path.isfile(node.intermediate_source_path)
                    )
                    can_texture_refine = bool(
                        can_refine
                        and node.intermediate_texture_checkpoint_path
                        and os.path.isfile(
                            node.intermediate_texture_checkpoint_path
                        )
                    )
                if parent is not None:
                    can_texture_refine = bool(
                        parent.backend is ReconstructionBackend.PIXAL3D
                        and parent.checkpoint_path
                        and os.path.isfile(parent.checkpoint_path)
                        and not _is_composite_shape_checkpoint(
                            parent.checkpoint_path
                        )
                        and parent.texture_checkpoint_path
                        and os.path.isfile(parent.texture_checkpoint_path)
                        and parent.source_path
                        and os.path.isfile(parent.source_path)
                    )
                lr_source = node.selected_lr_refine_source
                can_lr_refine = bool(
                    node.generation_parameters.backend
                    is ReconstructionBackend.PIXAL3D
                    and (
                        lr_source is not None
                        and os.path.isfile(lr_source.checkpoint_path)
                        and os.path.isfile(lr_source.source_path)
                        or lr_source is None
                        and node.resume_stage in {
                            ReconstructionStage.LR_SHAPE_FLOW,
                            ReconstructionStage.LR_SHAPE_LATENT,
                        }
                        and node.resume_checkpoint_path
                        and os.path.isfile(node.resume_checkpoint_path)
                        and node.intermediate_source_path
                        and os.path.isfile(node.intermediate_source_path)
                    )
                )
                selection = (
                    self.canvas_controls_coordinator.selection_state
                    if self.canvas_controls_coordinator is not None
                    else None
                )
                update_refine(
                    node.refine_parameters,
                    node.runs,
                    node.active_run_id,
                    lr_parameters=node.lr_refine_parameters,
                    texture_parameters=node.texture_refine_parameters,
                    mask_ready=(
                        not self.application.layer_stack.selection.is_empty
                    ),
                    refine_supported=refine_supported,
                    can_refine=can_refine,
                    can_texture_refine=can_texture_refine,
                    can_lr_refine=can_lr_refine,
                    paint_active=bool(selection and selection.edit_mode),
                    erase_active=bool(selection and selection.eraser),
                    brush_size=selection.size if selection else 50,
                    brush_hardness=selection.hardness if selection else 0.4,
                    brush_flow=selection.flow if selection else 1.0,
                    busy=busy,
                )

    def _handle_reconstruction_workspace(
            self, action: str, value=None) -> None:
        if action == "set_workspace_mode":
            if not value:
                self._hide_reconstruction_refine_output()
            else:
                self._sync_reconstruction_refine_output(
                    self._active_reconstruction_workspace,
                    getattr(
                        self.view,
                        "_selected_reconstruction_workspace_operation",
                        None,
                    ),
                    getattr(
                        self.view,
                        "_selected_reconstruction_workspace_operation_id",
                        None,
                    ),
                )
            return
        if action in {"select_operation", "select_operation_variant"}:
            operation_key = getattr(
                self.view,
                "_selected_reconstruction_workspace_operation",
                None,
            )
            operation_id = (
                str(value)
                if action == "select_operation_variant" and value
                else getattr(
                    self.view,
                    "_selected_reconstruction_workspace_operation_id",
                    None,
                )
            )
            self._sync_reconstruction_refine_output(
                self._active_reconstruction_workspace,
                operation_key,
                operation_id,
            )
            return
        if action == "select_refine_source":
            node = self.application.layer_stack.active_layer
            workspace = self._active_reconstruction_workspace
            if not isinstance(node, ReconstructionLayer) or workspace is None:
                return
            try:
                artifact = workspace.artifact(str(value))
                metadata = json.loads(artifact.metadata_json)
                variant_id = str(metadata["lr_variant_id"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                return
            if not any(
                    item.variant_id == variant_id
                    for item in node.lr_variants):
                return
            node.selected_lr_refine_source_id = variant_id
            self._refresh_reconstruction_panel(node)
            return
        if action == "set_operation_parameter":
            node = self.application.layer_stack.active_layer
            controller = self.reconstruction_controller
            if (
                not isinstance(node, ReconstructionLayer)
                or (controller is not None and controller.is_busy)
            ):
                return
            try:
                key, parameter_value = value
                node.generation_parameters = replace(
                    node.generation_parameters,
                    **{str(key): parameter_value},
                )
            except (TypeError, ValueError):
                return
            self._refresh_reconstruction_panel(node)
            return
        if action == "reset_operation_parameters":
            node = self.application.layer_stack.active_layer
            controller = self.reconstruction_controller
            if (
                not isinstance(node, ReconstructionLayer)
                or (controller is not None and controller.is_busy)
            ):
                return
            keys = PIXAL3D_OPERATION_PARAMETER_KEYS.get(str(value), ())
            defaults = ReconstructionParameters()
            replacements = {}
            for key in keys:
                if key.startswith("pixal3d_") and key.endswith("_seed"):
                    replacements[key] = -1
                elif key.startswith("pixal3d_") and key.endswith("_steps"):
                    replacements[key] = 0
                else:
                    replacements[key] = getattr(defaults, key)
            if replacements:
                node.generation_parameters = replace(
                    node.generation_parameters, **replacements
                )
                self._refresh_reconstruction_panel(node)
            return
        if action == "generate_to_operation":
            node = self.application.layer_stack.active_layer
            controller = self.reconstruction_controller
            if (
                    not isinstance(node, ReconstructionLayer)
                    or node.generation_parameters.backend
                    is not ReconstructionBackend.PIXAL3D
                    or (controller is not None and controller.is_busy)):
                return
            stage = LEGACY_OPERATION_TARGET_STAGES.get(str(value))
            if stage is None:
                return
            node.target_stage = stage
            self.application.set_status(
                "Starting legacy Pixal3D execution through "
                f"{str(value).replace('.', ' ')}"
            )
            self._start_reconstruction()
            return
        if action != "preview_artifact":
            return
        node = self.application.layer_stack.active_layer
        workspace = self._active_reconstruction_workspace
        if not isinstance(node, ReconstructionLayer) or workspace is None:
            return
        try:
            artifact = workspace.artifact(str(value))
            if not artifact.path or not os.path.isfile(artifact.path):
                self.application.set_status(
                    "Cannot preview graph artifact: file is missing"
                )
                return
            if artifact.preview_kind in {
                    WorkspacePreviewKind.MESH,
                    WorkspacePreviewKind.OVERLAY}:
                self._ensure_reconstruction_viewport().load_glb(artifact.path)
                self._presented_reconstruction_id = node.id
            elif artifact.preview_kind is WorkspacePreviewKind.POINTS:
                self._ensure_reconstruction_viewport().load_point_cloud(
                    artifact.path
                )
                self._presented_reconstruction_id = node.id
            elif (
                    artifact.preview_kind is WorkspacePreviewKind.IMAGE
                    and artifact.role == "source"):
                if self.canvas is not None:
                    self.canvas.fit_in_view()
            else:
                self.application.set_status(
                    "This artifact needs a viewer that is not connected yet"
                )
                return
            workspace.select_artifact(artifact.artifact_id)
            self.application.set_status(
                f"Previewing graph artifact: {artifact.role}"
            )
        except Exception as exc:
            log.exception("Failed to preview reconstruction graph artifact")
            self.application.set_status(f"Cannot preview graph artifact: {exc}")

    def _sync_reconstruction_selection(
            self, node: ReconstructionLayer | None) -> None:
        set_context = getattr(self.view, "set_reconstruction_context", None)
        if self.canvas is not None:
            self.canvas.set_selection_as_mask(node is not None)
        if node is None:
            coordinator = self.canvas_controls_coordinator
            if coordinator is not None and coordinator.selection_state.edit_mode:
                self._set_reconstruction_mask_painting(False)
            if callable(set_context):
                set_context(False)
            return
        viewport = self._ensure_reconstruction_viewport()
        if callable(set_context):
            set_context(True, node.reconstruction_status.value)
        self._refresh_reconstruction_panel(node)
        if self._presented_reconstruction_id == node.id:
            return
        artifact = node.stage_artifacts.get(node.selected_preview_stage)
        if artifact is not None and self._load_reconstruction_artifact(artifact):
            pass
        elif node.glb_path and os.path.isfile(node.glb_path):
            viewport.load_glb(node.glb_path)
        else:
            viewport.clear_model()
        self._presented_reconstruction_id = node.id

    def _load_reconstruction_artifact(self, artifact) -> bool:
        if not os.path.isfile(artifact.path):
            return False
        viewport = self._ensure_reconstruction_viewport()
        if artifact.preview_kind == "mesh":
            viewport.load_glb(artifact.path)
            return True
        if artifact.preview_kind == "points":
            viewport.load_point_cloud(artifact.path)
            return True
        return False

    def _set_reconstruction_status(
            self,
            node: ReconstructionLayer,
            status: ReconstructionStatus) -> None:
        if node.reconstruction_status == status:
            return
        node.reconstruction_status = status
        self.application.layer_stack.publish_change(
            DocumentChangeKind.METADATA,
            layers=(node,),
            operation="update 3D reconstruction status",
        )

    def _ensure_reconstruction_viewport(self) -> NativeReconstructionViewport:
        if self.reconstruction_viewport is not None:
            return self.reconstruction_viewport
        graphics = getattr(self.composition, "graphics", None)
        mount = getattr(self.view, "mount_reconstruction_viewport", None)
        if graphics is None or not callable(mount):
            raise RuntimeError("native 3D viewport is unavailable in this host")
        viewport = NativeReconstructionViewport(
            self.composition.document,
            graphics_owner=graphics,
            request_repaint=self.composition.request_repaint,
            resource_namespace="primary",
        )
        mount(viewport)
        self.reconstruction_viewport = viewport
        register_resource = getattr(
            self.application, "register_shutdown_resource", None
        )
        if callable(register_resource):
            register_resource(
                ShutdownPhase.GPU_RESOURCES,
                "native-reconstruction-viewport",
                viewport.close,
            )
        return viewport

    def _ensure_reconstruction_refine_viewport(
            self) -> NativeReconstructionViewport:
        existing = getattr(self, "reconstruction_refine_viewport", None)
        if existing is not None:
            return existing
        self._ensure_reconstruction_viewport()
        graphics = getattr(self.composition, "graphics", None)
        mount = getattr(
            self.view, "mount_reconstruction_refine_viewport", None
        )
        if graphics is None or not callable(mount):
            raise RuntimeError("native refine viewport is unavailable")
        viewport = NativeReconstructionViewport(
            self.composition.document,
            graphics_owner=graphics,
            request_repaint=self.composition.request_repaint,
            resource_namespace="refine",
        )
        mount(viewport)
        self.reconstruction_refine_viewport = viewport
        register_resource = getattr(
            self.application, "register_shutdown_resource", None
        )
        if callable(register_resource):
            register_resource(
                ShutdownPhase.GPU_RESOURCES,
                "native-reconstruction-refine-viewport",
                viewport.close,
            )
        return viewport

    def _hide_reconstruction_refine_output(self) -> None:
        set_visible = getattr(
            getattr(self, "view", None),
            "set_reconstruction_refine_view_visible",
            None,
        )
        if callable(set_visible):
            set_visible(False)
        update_placement = getattr(
            getattr(self, "view", None),
            "update_reconstruction_refine_placement",
            None,
        )
        if callable(update_placement):
            update_placement(None)
        self._presented_refine_artifact_path = None
        self._presented_refine_reference_path = None
        self._presented_refine_run_id = None
        viewport = getattr(self, "reconstruction_refine_viewport", None)
        if viewport is not None:
            viewport.clear_model()

    def _sync_reconstruction_refine_output(
            self, workspace, operation_key, operation_id=None) -> None:
        if (
                not getattr(self.view, "reconstruction_workspace_mode", False)
                or operation_key not in {"lr.refine", "hr.refine"}):
            self._hide_reconstruction_refine_output()
            return
        try:
            viewport = self._ensure_reconstruction_refine_viewport()
        except RuntimeError:
            return
        operation = None
        if workspace is not None and operation_id:
            try:
                candidate = workspace.operation(str(operation_id))
                if candidate.spec_key == operation_key:
                    operation = candidate
            except KeyError:
                pass
        if operation is None and workspace is not None:
            operations = workspace.operations_for_spec(operation_key)
            operation = operations[-1] if operations else None
        generated = None
        if operation is not None:
            generated = next(
                (
                    artifact
                    for artifact in workspace.artifacts_for_operation(
                        operation.operation_id
                    )
                    if artifact.role == "refine_generated_mesh"
                ),
                None,
            )
        path = generated.path if generated is not None else None
        if not path or not os.path.isfile(path):
            self._hide_reconstruction_refine_output()
            return
        set_visible = getattr(
            self.view, "set_reconstruction_refine_view_visible", None
        )
        if callable(set_visible):
            set_visible(True)
        application = getattr(self, "application", None)
        layer_stack = getattr(application, "layer_stack", None)
        node = getattr(layer_stack, "active_layer", None)
        run = None
        if isinstance(node, ReconstructionLayer) and operation is not None:
            prefix = "legacy:"
            suffix = f":{operation.spec_key}"
            operation_id_value = operation.operation_id
            if (
                    operation_id_value.startswith(prefix)
                    and operation_id_value.endswith(suffix)):
                run_id = operation_id_value[
                    len(prefix):-len(suffix)
                ]
                run = next(
                    (item for item in node.runs if item.run_id == run_id),
                    None,
                )
        reference = None
        if run is not None:
            runs_by_id = {item.run_id: item for item in node.runs}
            reference = runs_by_id.get(run.parent_run_id or "")
            visited = set()
            while (
                    reference is not None
                    and reference.kind is ReconstructionRunKind.MASKED_REFINE
                    and reference.run_id not in visited):
                visited.add(reference.run_id)
                ancestor = runs_by_id.get(reference.parent_run_id or "")
                if ancestor is None:
                    break
                reference = ancestor
            if reference is None:
                reference = node.base_run
        reference_path = (
            reference.glb_path
            if reference is not None
            and os.path.isfile(reference.glb_path)
            else None
        )
        if (
                path != getattr(
                    self, "_presented_refine_artifact_path", None
                )
                or reference_path != getattr(
                    self, "_presented_refine_reference_path", None
                )):
            load_comparison = getattr(
                viewport, "load_comparison_glbs", None
            )
            if reference_path is not None and callable(load_comparison):
                load_comparison(reference_path, path)
            else:
                viewport.load_glb(path)
            self._presented_refine_artifact_path = path
            self._presented_refine_reference_path = reference_path
        if run is not None:
            bind_placement = getattr(viewport, "bind_refine_placement", None)
            if callable(bind_placement):
                bind_placement(
                    run.refine_placement,
                    run.refine_placement_pivot,
                    lambda placement, run_id=run.run_id:
                    self._commit_reconstruction_refine_placement(
                        run_id, placement
                    ),
                )
            self._presented_refine_run_id = run.run_id
            update_placement = getattr(
                self.view,
                "update_reconstruction_refine_placement",
                None,
            )
            if callable(update_placement):
                controller = self.reconstruction_controller
                update_placement(
                    run.refine_placement,
                    busy=bool(controller and controller.is_busy),
                    accepted=run.refine_placement_accepted,
                )

    def _commit_reconstruction_refine_placement(
        self,
        run_id: str,
        placement: ReconstructionRefinePlacement,
    ) -> None:
        node = self.application.layer_stack.active_layer
        if not isinstance(node, ReconstructionLayer):
            return
        self.application.document.execute(
            SetReconstructionRefinePlacementCommand(
                node, run_id, placement
            )
        )
        self._refresh_reconstruction_panel(node)

    def _handle_reconstruction_refine_placement(
            self, action: str, value=None) -> None:
        node = self.application.layer_stack.active_layer
        run_id = self._presented_refine_run_id
        controller = self.reconstruction_controller
        if (
                not isinstance(node, ReconstructionLayer)
                or run_id is None
                or (controller is not None and controller.is_busy)):
            return
        run = next(
            (item for item in node.runs if item.run_id == run_id),
            None,
        )
        if run is None or run.refine_generated_path is None:
            return
        if action == "accept":
            self._accept_reconstruction_refine_placement(node, run)
            return
        placement = run.refine_placement
        if action == "reset":
            next_placement = ReconstructionRefinePlacement()
        else:
            translation = list(placement.translation)
            rotations = list(
                euler_degrees_from_quaternion(placement.orientation)
            )
            if action in {"x", "y", "z"}:
                translation[{"x": 0, "y": 1, "z": 2}[action]] = float(value)
            elif action in {"rx", "ry", "rz"}:
                rotations[{"rx": 0, "ry": 1, "rz": 2}[action]] = float(value)
            elif action != "scale":
                return
            next_placement = ReconstructionRefinePlacement(
                tuple(translation),
                quaternion_from_euler_degrees(*rotations),
                float(value) if action == "scale" else placement.scale,
            )
        self._commit_reconstruction_refine_placement(run.run_id, next_placement)
        viewport = getattr(self, "reconstruction_refine_viewport", None)
        preview_placement = getattr(
            viewport, "preview_refine_placement", None
        )
        if callable(preview_placement):
            preview_placement(
                next_placement, pivot=run.refine_placement_pivot
            )

    def _accept_reconstruction_refine_placement(
        self,
        node: ReconstructionLayer,
        run,
    ) -> None:
        controller = self.reconstruction_controller
        if controller is None or controller.is_busy:
            return
        checkpoint_path = run.checkpoint_path
        source_path = run.source_path
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            self.application.set_status(
                "Cannot fuse refine placement: proposal checkpoint is missing"
            )
            return
        if not _is_composite_shape_checkpoint(checkpoint_path):
            self.application.set_status(
                "Cannot fuse refine placement: checkpoint is not a local proposal"
            )
            return
        if not source_path or not os.path.isfile(source_path):
            self.application.set_status(
                "Cannot fuse refine placement: source image is missing"
            )
            return
        event = controller.start_refine_fusion(
            checkpoint_path,
            source_path,
            run.refine_placement,
            generation_parameters=node.generation_parameters,
        )
        if event.status and event.error is None:
            self._reconstruction_job_node_id = node.id
            self._reconstruction_parent_run_id = run.run_id
            self._set_reconstruction_status(
                node, ReconstructionStatus.GENERATING
            )
            self.application.set_status(event.status)
        elif event.status:
            self.application.set_status(event.status)
        self._refresh_reconstruction_panel(node)

    def _on_snapshot_applied(self) -> None:
        if self.canvas_edit_coordinator is not None:
            self.canvas_edit_coordinator.discard()
        if self.canvas is not None:
            self.canvas.cancel_pointer_interaction()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        # Application close joins every registered producer before queued UI
        # callbacks are discarded and the document/view lifetime ends.
        self.application.close()
        self.dispatcher.close()
        self.discarded_on_close = self.dispatcher.discard_pending()
        self.composition.set_unhandled_key_handler(None)
        self.view.close()
        self.composition.close()

    def __enter__(self) -> "NativeEditorRoot":
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> bool:
        self.close()
        return False
