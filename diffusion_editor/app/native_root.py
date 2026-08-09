"""Hostless termin-gui-native composition for the editor migration path."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import os
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol

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
    PublishReconstructionResultCommand,
)
from ..document.reconstruction import ReconstructionLayer, ReconstructionStatus
from ..document.tool import DiffusionTool, InstructTool, LamaTool
from ..generation.patch_resolver import source_patch_at_center
from .application import EditorApplication, ShutdownPhase
from .canvas_controls import CanvasControlsCoordinator
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
        self._reconstruction_job_node_id: str | None = None
        self._presented_reconstruction_id: str | None = None
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
                        "pixal3d-reconstruction-engine",
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
            return InstructTool(**common)
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
            event = controller.start(image, seed=42)
            if event.status:
                self._reconstruction_job_node_id = node.id
                self._set_reconstruction_status(
                    node, ReconstructionStatus.GENERATING
                )
                self.application.set_status(event.status)
        except Exception as exc:
            log.exception("Failed to start 3D reconstruction")
            self.application.set_status(f"Cannot start 3D generation: {exc}")

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
        node_id, self._reconstruction_job_node_id = (
            self._reconstruction_job_node_id, None
        )
        node = self.application.layer_stack.find_layer_by_id(node_id or "")
        if not isinstance(node, ReconstructionLayer):
            self.application.set_status("Ignored 3D result for a deleted object")
            return
        if event.result is not None:
            try:
                viewport = self._ensure_reconstruction_viewport()
                vertices, triangles, meshes = viewport.load_glb(
                    event.result.glb_path
                )
                self.application.document.execute(
                    PublishReconstructionResultCommand(
                        node,
                        event.result.glb_path,
                        vertices,
                        triangles,
                        meshes,
                    )
                )
                self._presented_reconstruction_id = node.id
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
                self._set_reconstruction_status(
                    node, ReconstructionStatus.FAILED
                )
            self.application.set_status(event.status)

    def _sync_reconstruction_selection(
            self, node: ReconstructionLayer | None) -> None:
        if node is None or self._presented_reconstruction_id == node.id:
            return
        viewport = self._ensure_reconstruction_viewport()
        if node.glb_path and os.path.isfile(node.glb_path):
            viewport.load_glb(node.glb_path)
        else:
            viewport.clear_model()
        self._presented_reconstruction_id = node.id

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
