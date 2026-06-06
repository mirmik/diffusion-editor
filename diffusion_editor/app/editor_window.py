"""EditorWindow — main orchestrator for the diffusion editor (tcgui version)."""

from __future__ import annotations

from dataclasses import dataclass
import os
import random

import numpy as np
from PIL import Image
from tcbase import log

from tcgui.widgets.ui import UI
from tcgui.widgets.vstack import VStack
from tcgui.widgets.hstack import HStack
from tcgui.widgets.panel import Panel
from tcgui.widgets.label import Label
from tcgui.widgets.button import Button
from tcgui.widgets.menu_bar import MenuBar
from tcgui.widgets.menu import MenuItem, Menu
from tcgui.widgets.tool_bar import ToolBar
from tcgui.widgets.status_bar import StatusBar
from tcgui.widgets.message_box import MessageBox, Buttons
from tcgui.widgets.dialog import Dialog
from tcgui.widgets.spin_box import SpinBox
from tcgui.widgets.text_input import TextInput
from tcgui.widgets.checkbox import Checkbox
from tcgui.widgets.units import px, pct
from tcgui.widgets.splitter import Splitter

from ..agent.chat_panel import DEFAULT_AGENT_BASE_URL, DEFAULT_AGENT_MODEL, AgentChatPanel
from ..agent.tools import create_editor_tool_registry
from ..document.layer_stack import LayerStack
from ..document.mask import coerce_mask_data
from ..document.layer import Layer
from ..document.tool import DiffusionTool, LamaTool, InstructTool
from ..canvas.brush import BrushToolMode
from ..canvas.editor_canvas import EditorCanvas
from ..ui.panels.layer_panel import LayerPanel
from ..ui.panels.brush_panel import BrushPanel
from ..ui.panels.diffusion_panel import DiffusionPanel
from ..generation.diffusion_controller import DiffusionGenerationController
from ..grounding.controller import GroundingController
from ..engines.grounding_engine import GroundingEngine
from ..grounding.types import GroundingParams
from ..generation.instruct_controller import InstructGenerationController
from ..ui.dialogs.ip_adapter_reference_dialog import show_ip_adapter_reference_dialog
from ..generation.lama_controller import LamaGenerationController
from ..ui.panels.lama_panel import LamaPanel
from ..ui.panels.instruct_panel import InstructPanel
from ..ui.panels.selection_panel import SelectionPanel
from ..generation.segmentation_controller import SegmentationGenerationController
from ..engines.lama_engine import LamaEngine
from ..engines.segmentation_engine import SegmentationEngine
from ..generation.patch_resolver import source_patch_at_center
from ..ui.dialogs.file_dialog import open_file_dialog, save_file_dialog, open_directory_dialog
from .settings import Settings
from ..document.history import HistoryManager
from ..document.document_service import DocumentService
from ..document.commands import (
    AddLayerCommand, RemoveLayerCommand,
    MoveLayerCommand, SetLayerVisibilityCommand, SetLayerOpacityCommand,
    SetLayerNameCommand,
    FlattenLayersCommand, ClearLayerMaskCommand,
    AttachLayerToolCommand, DetachLayerToolCommand,
    SetIpAdapterReferenceLayerCommand, ClearIpAdapterReferenceLayerCommand,
    SetLayerPatchRectCommand, ClearLayerPatchRectCommand,
    ClearSelectionCommand, InvertSelectionCommand, SelectAllCommand,
    SetLayerSelectionCommand,
)
from ..generation.result_mapper import (
    map_segmentation_result, map_lama_result,
    map_instruct_result, map_diffusion_result, map_grounding_result,
)

_BYTES_PER_GIB = 1024 * 1024 * 1024
_DEFAULT_HISTORY_MEMORY_LIMIT_BYTES = 5 * _BYTES_PER_GIB
_MIN_HISTORY_MEMORY_LIMIT_GIB = 0.25
_MAX_HISTORY_MEMORY_LIMIT_GIB = 256.0
_EXPORT_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass
class ExternalEditContext:
    label: str
    layer_path: str
    target: str
    before_arr: np.ndarray | None = None
    before_offset: tuple[int, int] | None = None


class EditorWindow:
    """Non-widget orchestrator: assembles UI, wires callbacks, handles logic."""

    def __init__(self, ctx=None):
        # Optional borrowed tgfx2 context. When given (typically by
        # main.py, which owns an SDLBackendWindow and wants every renderer
        # to share one IRenderDevice), the UI draws through it; when
        # None, UIRenderer falls back to its own owning Tgfx2Context.
        # Required to be non-None under TERMIN_BACKEND=vulkan because
        # cross-widget TextureHandle sharing and swapchain presentation
        # both assume a single process-global device.
        self._ctx = ctx
        self._running = True
        self._closed = False
        self._settings = Settings()
        self._project_path: str | None = None
        self._last_dir: str = self._settings.get("last_dir", "")
        self._history_memory_limit_bytes = self._load_history_memory_limit_bytes()
        self._models_dir = self._load_models_dir()
        self._clipboard: np.ndarray | None = None  # RGBA uint8 patch
        self._clipboard_pos: tuple[int, int] | None = None
        self._history_replaying = False
        self._external_edit_ctx: ExternalEditContext | None = None

        # Layer stack
        self._layer_stack = LayerStack()

        # Engines
        from ..engines.diffusion_engine import DiffusionEngine
        from ..engines.instruct_engine import InstructEngine

        self._engine = DiffusionEngine()
        self._seg_engine = SegmentationEngine()
        self._lama_engine = LamaEngine()
        self._instruct_engine = InstructEngine()
        self._grounding_engine = GroundingEngine()
        self._history = HistoryManager(
            self._apply_snapshot,
            max_memory_bytes=self._history_memory_limit_bytes,
        )
        self._document = DocumentService(
            self._layer_stack,
            self._history,
            self._apply_snapshot,
        )

        # Agent tool registry (created once, reused across chat clear/reset)
        self._agent_tool_registry = create_editor_tool_registry()

        # Build UI
        self._build_ui()
        self._diffusion_controller = DiffusionGenerationController(
            engine=self._engine,
            layer_stack=self._layer_stack,
            composite_below=lambda layer: self._canvas.get_composite_below(layer),
        )
        self._lama_controller = LamaGenerationController(
            engine=self._lama_engine,
            composite_below=lambda layer: self._canvas.get_composite_below(layer),
        )
        self._instruct_controller = InstructGenerationController(
            engine=self._instruct_engine,
            composite_below=lambda layer: self._canvas.get_composite_below(layer),
        )
        self._segmentation_controller = SegmentationGenerationController(
            engine=self._seg_engine,
            composite_below=lambda layer: self._canvas.get_composite_below(layer),
        )
        self._grounding_controller = GroundingController(
            engine=self._grounding_engine,
            composite=lambda: self._layer_stack.composite(),
        )

        # Wire callbacks
        self._wire_callbacks()
        self._menu_bar.register_shortcuts(self.ui)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        root = VStack()
        root.preferred_width = pct(100)
        root.preferred_height = pct(100)
        root.spacing = 0

        # Menu bar
        self._menu_bar = MenuBar()
        self._setup_menu()
        root.add_child(self._menu_bar)

        # Toolbar
        self._toolbar = ToolBar()
        self._toolbar.add_action(text="Open", on_click=self.open_file)
        self._toolbar.add_action(text="Save", on_click=self.save_file)
        self._toolbar.add_separator()
        self._toolbar.add_action(text="Fit", on_click=self._fit)
        self._toolbar.add_separator()
        self._toolbar.add_action(text="Sel All", on_click=self._select_all)
        self._toolbar.add_action(text="Sel Clear", on_click=self._clear_selection)
        self._toolbar.add_action(text="Sel Inv", on_click=self._invert_selection)
        root.add_child(self._toolbar)

        # Main content area: left panel | canvas | right panel
        main_area = HStack()
        main_area.stretch = True
        main_area.spacing = 0

        # Left panel container
        self._left_container = VStack()
        self._left_container.preferred_width = px(260)
        self._left_container.clip = True

        # Brush is always available; the selected layer's attached tool panel
        # appears below it when a tool is present.
        self._brush_panel = BrushPanel(self._canvas_placeholder_brush())
        self._diffusion_panel = DiffusionPanel()
        self._diffusion_panel.set_models_dir(self._models_dir)
        self._lama_panel = LamaPanel()
        self._instruct_panel = InstructPanel()
        self._selection_panel = SelectionPanel()
        self._selection_panel.visible = True

        self._left_container.spacing = 6
        self._brush_panel.visible = True
        self._left_container.add_child(self._brush_panel)
        self._left_container.add_child(self._selection_panel)
        for p in (self._diffusion_panel, self._lama_panel, self._instruct_panel):
            p.visible = False
            p.stretch = True
            self._left_container.add_child(p)

        main_area.add_child(self._left_container)
        main_area.add_child(Splitter(target=self._left_container, side="right"))

        # Canvas (center, stretches to fill remaining space)
        self._canvas = EditorCanvas(self._layer_stack, ctx=self._ctx)
        self._canvas.stretch = True
        # Give brush reference now
        self._brush_panel._brush = self._canvas.brush
        main_area.add_child(self._canvas)

        # Right panels: layer panel | agent chat
        self._layer_panel = LayerPanel(self._layer_stack)
        main_area.add_child(Splitter(target=self._layer_panel, side="left"))
        main_area.add_child(self._layer_panel)

        self._agent_chat_panel = AgentChatPanel(
            self._settings,
            tool_registry=self._agent_tool_registry,
            layer_stack=self._layer_stack,
            document_service=self._document,
        )
        self._agent_chat_panel.preferred_width = px(320)
        main_area.add_child(Splitter(target=self._agent_chat_panel, side="left"))
        main_area.add_child(self._agent_chat_panel)

        root.add_child(main_area)

        # Status bar
        self._statusbar = StatusBar()
        self._statusbar.text = "Ready"
        root.add_child(self._statusbar)

        self.ui = UI(graphics=self._ctx)
        self.ui.root = root

    def _canvas_placeholder_brush(self):
        """Temporary brush for panel construction (replaced after canvas created)."""
        from ..canvas.brush import Brush
        return Brush()

    def _setup_menu(self):
        # File menu
        file_menu = Menu()
        file_menu.add_item(MenuItem("New...", shortcut="Ctrl+N", on_click=self.new_project))
        file_menu.add_item(MenuItem("New From Image...", on_click=self.new_project_from_image))
        file_menu.add_item(MenuItem(separator=True))
        file_menu.add_item(MenuItem("Open...", shortcut="Ctrl+O", on_click=self.open_file))
        file_menu.add_item(MenuItem("Save", shortcut="Ctrl+S", on_click=self.save_file))
        file_menu.add_item(MenuItem("Save As...", shortcut="Ctrl+Shift+S", on_click=self.save_file_as))
        file_menu.add_item(MenuItem(separator=True))
        file_menu.add_item(MenuItem("Import Image...", shortcut="Ctrl+I", on_click=self.import_image))
        file_menu.add_item(MenuItem("Export Image...", shortcut="Ctrl+E", on_click=self.export_image))
        file_menu.add_item(MenuItem(separator=True))
        file_menu.add_item(MenuItem("Quit", shortcut="Ctrl+Q", on_click=self._quit))
        self._menu_bar.add_menu("File", file_menu)

        # Edit menu
        edit_menu = Menu()
        edit_menu.add_item(MenuItem("Undo", shortcut="Ctrl+Z", on_click=self.undo))
        edit_menu.add_item(MenuItem("Redo", shortcut="Ctrl+Shift+Z", on_click=self.redo))
        edit_menu.add_item(MenuItem("Redo", shortcut="Ctrl+Y", on_click=self.redo))
        edit_menu.add_item(MenuItem("Copy", shortcut="Ctrl+C", on_click=self._copy))
        edit_menu.add_item(MenuItem("Copy Visible", shortcut="Ctrl+Shift+C", on_click=self._copy_visible))
        edit_menu.add_item(MenuItem("Paste", shortcut="Ctrl+V", on_click=self._paste))
        edit_menu.add_item(MenuItem(separator=True))
        edit_menu.add_item(MenuItem("Settings...", on_click=self._show_settings_dialog))
        self._menu_bar.add_menu("Edit", edit_menu)

        # Select menu
        select_menu = Menu()
        select_menu.add_item(MenuItem("Select All", shortcut="Ctrl+A", on_click=self._select_all))
        select_menu.add_item(MenuItem("Clear Selection", shortcut="Ctrl+D", on_click=self._clear_selection))
        select_menu.add_item(MenuItem(separator=True))
        select_menu.add_item(MenuItem("Invert Selection", shortcut="Ctrl+Shift+I", on_click=self._invert_selection))
        self._menu_bar.add_menu("Select", select_menu)

        # Layer menu
        layer_menu = Menu()
        layer_menu.add_item(MenuItem("New Layer", shortcut="Ctrl+Shift+N", on_click=self._new_layer))
        layer_menu.add_item(MenuItem("Remove Layer", on_click=self._remove_layer))
        layer_menu.add_item(MenuItem(separator=True))
        layer_menu.add_item(MenuItem("Flatten", on_click=self._flatten_layers))
        layer_menu.add_item(MenuItem(separator=True))
        layer_menu.add_item(MenuItem("Detect Objects...", on_click=self._show_grounding_dialog))
        self._menu_bar.add_menu("Layer", layer_menu)

    # ------------------------------------------------------------------
    # Wiring
    # ------------------------------------------------------------------

    def _wire_callbacks(self):
        # Layer stack
        old_on_changed = self._layer_stack.on_changed

        def _on_stack_changed():
            if old_on_changed:
                old_on_changed()
            self._on_layer_changed()
            self._layer_panel.sync_from_stack()
        self._layer_stack.on_changed = _on_stack_changed

        # Canvas
        self._canvas.on_mouse_moved = self._on_mouse_moved
        self._canvas.on_color_picked = self._on_color_picked
        self._canvas.on_edit_begin = self._begin_external_edit
        self._canvas.on_edit_end = self._end_external_edit

        # Brush panel
        self._brush_panel.on_tool_changed = self._set_canvas_tool
        self._brush_panel.on_eraser_toggled = self._canvas.set_brush_eraser
        self._brush_panel.on_draw_patch_toggled = self._canvas.set_patch_rect_mode
        self._brush_panel.on_show_patch_toggled = self._canvas.set_show_patch_rect
        self._brush_panel.on_clear_patch = self._on_clear_patch_rect

        # Selection panel
        self._selection_panel.on_edit_mode_toggled = self._canvas.set_selection_mode
        self._selection_panel.on_rect_mode_toggled = self._canvas.set_selection_rect_mode
        self._selection_panel.on_brush_changed = self._canvas.set_selection_brush
        self._selection_panel.on_eraser_toggled = self._canvas.set_selection_eraser
        self._selection_panel.on_show_selection_toggled = self._canvas.set_show_selection

        # Diffusion panel
        self._diffusion_panel.on_load_model = self._on_load_model
        self._diffusion_panel.on_regenerate = self._on_regenerate
        self._diffusion_panel.on_new_seed = self._on_new_seed
        self._diffusion_panel.on_clear_mask = self._on_clear_mask
        self._diffusion_panel.on_mask_brush_changed = self._set_mask_brush
        self._diffusion_panel.on_mask_eraser_toggled = self._set_mask_eraser
        self._diffusion_panel.on_show_mask_toggled = self._canvas.set_show_mask
        self._diffusion_panel.on_load_ip_adapter = self._on_load_ip_adapter
        self._diffusion_panel.on_pick_ip_adapter_layer = self._on_pick_ip_adapter_layer
        self._diffusion_panel.on_clear_ip_adapter_layer = self._on_clear_ip_adapter_layer
        self._diffusion_panel.on_select_background = self._on_select_background
        self._canvas.on_patch_rect_drawn = self._on_patch_rect_drawn
        self._canvas.on_selection_rect_drawn = self._on_selection_rect_drawn

        # Layer panel
        self._layer_panel.on_attach_tool = self._attach_tool_to_layer
        self._layer_panel.on_detach_tool = self._detach_tool_from_layer
        self._layer_panel.on_add_layer = self._new_layer
        self._layer_panel.on_remove_layer = self._remove_layer
        self._layer_panel.on_flatten_layers = self._flatten_layers
        self._layer_panel.on_move_layer = self._move_layer
        self._layer_panel.on_toggle_visibility = self._set_layer_visibility
        self._layer_panel.on_toggle_solo = self._toggle_layer_solo
        self._layer_panel.on_opacity_changed = self._set_layer_opacity
        self._layer_panel.on_rename_layer = self._rename_layer

        # LaMa panel
        self._lama_panel.on_remove = self._on_lama_remove
        self._lama_panel.on_clear_mask = self._on_lama_clear_mask
        self._lama_panel.on_mask_brush_changed = self._set_mask_brush
        self._lama_panel.on_mask_eraser_toggled = self._set_mask_eraser
        self._lama_panel.on_show_mask_toggled = self._canvas.set_show_mask
        self._lama_panel.on_select_background = self._on_lama_select_background

        # Instruct panel
        self._instruct_panel.on_load_model = self._on_instruct_load_model
        self._instruct_panel.on_apply = self._on_instruct_apply
        self._instruct_panel.on_new_seed = self._on_instruct_new_seed
        self._instruct_panel.on_clear_mask = self._on_instruct_clear_mask
        self._instruct_panel.on_mask_brush_changed = self._set_mask_brush
        self._instruct_panel.on_mask_eraser_toggled = self._set_mask_eraser
        self._instruct_panel.on_show_mask_toggled = self._canvas.set_show_mask

    def _set_canvas_tool(self, mode: BrushToolMode | str):
        self._canvas.set_brush_tool(mode)
        self._brush_panel.set_tool_mode(self._canvas.brush_tool_mode)

    def _set_mask_brush(self, size: int, hardness: float, flow: float = 1.0):
        self._canvas.set_mask_brush(size, hardness, flow)
        self._brush_panel.set_tool_mode(self._canvas.brush_tool_mode)

    def _set_mask_eraser(self, eraser: bool):
        self._canvas.set_mask_eraser(eraser)
        self._brush_panel.set_tool_mode(self._canvas.brush_tool_mode)

    # ------------------------------------------------------------------
    # Panel switching
    # ------------------------------------------------------------------

    def _on_layer_changed(self):
        layer = self._layer_stack.active_layer
        self._brush_panel.visible = True
        self._diffusion_panel.visible = False
        self._lama_panel.visible = False
        self._instruct_panel.visible = False

        if layer is None:
            pass
        elif isinstance(layer.tool, DiffusionTool):
            self._diffusion_panel.show_diffusion_layer(layer)
            self._diffusion_panel.visible = True
        elif isinstance(layer.tool, LamaTool):
            self._lama_panel.show_lama_layer(layer)
            self._lama_panel.visible = True
        elif isinstance(layer.tool, InstructTool):
            self._instruct_panel.show_instruct_layer(layer)
            self._instruct_panel.visible = True
        self.ui.request_layout()

        # Debug: print visible panel coords after next layout
        for name, p in [("brush", self._brush_panel), ("diffusion", self._diffusion_panel),
                         ("lama", self._lama_panel), ("instruct", self._instruct_panel)]:
            if p.visible:
                log.debug(
                    f"[panel] {name}: x={p.x:.0f} y={p.y:.0f} w={p.width:.0f} h={p.height:.0f}"
                )

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_bytes(n: int) -> str:
        if n < 1024:
            return f"{n}B"
        if n < 1024 * 1024:
            return f"{n / 1024:.0f}K"
        return f"{n / (1024 * 1024):.1f}M"

    def _load_history_memory_limit_bytes(self) -> int:
        raw = self._settings.get("history_memory_limit_bytes", _DEFAULT_HISTORY_MEMORY_LIMIT_BYTES)
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return _DEFAULT_HISTORY_MEMORY_LIMIT_BYTES
        if value <= 0:
            return _DEFAULT_HISTORY_MEMORY_LIMIT_BYTES
        return value

    def _set_history_memory_limit_bytes(self, limit_bytes: int) -> None:
        limit_bytes = max(int(limit_bytes), int(_MIN_HISTORY_MEMORY_LIMIT_GIB * _BYTES_PER_GIB))
        self._history_memory_limit_bytes = limit_bytes
        self._document.set_history_memory_limit_bytes(limit_bytes)
        self._settings.set("history_memory_limit_bytes", limit_bytes)

    def _load_models_dir(self) -> str:
        raw = self._settings.get("models_dir", DiffusionPanel.default_models_dir())
        if not isinstance(raw, str):
            return DiffusionPanel.default_models_dir()
        value = os.path.expanduser(raw.strip())
        if not value:
            return DiffusionPanel.default_models_dir()
        return value

    def _set_models_dir(self, models_dir: str) -> None:
        value = os.path.expanduser(models_dir.strip())
        if not value:
            value = DiffusionPanel.default_models_dir()
        self._models_dir = value
        self._diffusion_panel.set_models_dir(value)
        self._settings.set("models_dir", value)

    def _show_settings_dialog(self):
        if self.ui is None:
            return

        dlg = Dialog()
        dlg.title = "Settings"
        dlg.buttons = ["OK", "Cancel"]
        dlg.default_button = "OK"
        dlg.cancel_button = "Cancel"

        content = VStack()
        content.spacing = 8

        models_title = Label()
        models_title.text = "Stable Diffusion models directory"
        content.add_child(models_title)

        models_row = HStack()
        models_row.spacing = 6

        models_dir_input = TextInput()
        models_dir_input.text = self._models_dir
        models_dir_input.preferred_width = px(420)
        models_row.add_child(models_dir_input)

        browse_btn = Button()
        browse_btn.text = "Browse..."
        browse_btn.preferred_width = px(90)

        def _browse_models_dir():
            def _on_result(selected):
                if selected:
                    models_dir_input.text = selected
                    self._last_dir = selected
                    self._settings.set("last_dir", self._last_dir)
            open_directory_dialog(
                self.ui, _on_result,
                title="Select models directory",
                directory=models_dir_input.text or self._last_dir,
            )

        browse_btn.on_click = _browse_models_dir
        models_row.add_child(browse_btn)
        content.add_child(models_row)

        title = Label()
        title.text = "Undo/Redo memory limit (GiB)"
        content.add_child(title)

        limit_input = SpinBox()
        limit_input.decimals = 2
        limit_input.step = 0.25
        limit_input.min_value = _MIN_HISTORY_MEMORY_LIMIT_GIB
        limit_input.max_value = _MAX_HISTORY_MEMORY_LIMIT_GIB
        limit_input.value = self._history_memory_limit_bytes / _BYTES_PER_GIB
        limit_input.preferred_width = px(220)
        content.add_child(limit_input)

        note = Label()
        note.text = "Older history entries are removed when the limit is exceeded."
        content.add_child(note)

        agent_title = Label()
        agent_title.text = "Agent Chat API"
        content.add_child(agent_title)

        agent_base_url_label = Label()
        agent_base_url_label.text = "Base URL"
        agent_base_url_label.font_size = 12
        content.add_child(agent_base_url_label)

        agent_base_url_input = TextInput()
        agent_base_url_input.text = str(
            self._settings.get("agent_api_base_url", DEFAULT_AGENT_BASE_URL)
        )
        agent_base_url_input.placeholder = DEFAULT_AGENT_BASE_URL
        agent_base_url_input.preferred_width = px(420)
        content.add_child(agent_base_url_input)

        agent_key_label = Label()
        agent_key_label.text = "API key"
        agent_key_label.font_size = 12
        content.add_child(agent_key_label)

        agent_key_input = TextInput()
        agent_key_input.text = str(self._settings.get("agent_api_key", ""))
        agent_key_input.placeholder = "API key"
        agent_key_input.preferred_width = px(420)
        content.add_child(agent_key_input)

        agent_model_label = Label()
        agent_model_label.text = "Model"
        agent_model_label.font_size = 12
        content.add_child(agent_model_label)

        agent_model_input = TextInput()
        agent_model_input.text = str(self._settings.get("agent_model", DEFAULT_AGENT_MODEL))
        agent_model_input.placeholder = DEFAULT_AGENT_MODEL
        agent_model_input.preferred_width = px(260)
        content.add_child(agent_model_input)

        agent_params_row = HStack()
        agent_params_row.spacing = 8

        temperature_box = VStack()
        temperature_box.spacing = 3
        temperature_label = Label()
        temperature_label.text = "Temperature"
        temperature_label.font_size = 12
        temperature_box.add_child(temperature_label)

        agent_temperature_input = SpinBox()
        agent_temperature_input.decimals = 2
        agent_temperature_input.step = 0.05
        agent_temperature_input.min_value = 0.0
        agent_temperature_input.max_value = 2.0
        agent_temperature_input.value = float(self._settings.get("agent_temperature", 0.7))
        agent_temperature_input.preferred_width = px(120)
        temperature_box.add_child(agent_temperature_input)
        agent_params_row.add_child(temperature_box)

        max_tokens_box = VStack()
        max_tokens_box.spacing = 3
        max_tokens_label = Label()
        max_tokens_label.text = "Max tokens"
        max_tokens_label.font_size = 12
        max_tokens_box.add_child(max_tokens_label)

        agent_max_tokens_input = SpinBox()
        agent_max_tokens_input.decimals = 0
        agent_max_tokens_input.step = 128
        agent_max_tokens_input.min_value = 0
        agent_max_tokens_input.max_value = 131072
        agent_max_tokens_input.value = float(self._settings.get("agent_max_tokens", 1024))
        agent_max_tokens_input.preferred_width = px(140)
        max_tokens_box.add_child(agent_max_tokens_input)
        agent_params_row.add_child(max_tokens_box)

        timeout_box = VStack()
        timeout_box.spacing = 3
        timeout_label = Label()
        timeout_label.text = "Timeout sec"
        timeout_label.font_size = 12
        timeout_box.add_child(timeout_label)

        agent_timeout_input = SpinBox()
        agent_timeout_input.decimals = 0
        agent_timeout_input.step = 5
        agent_timeout_input.min_value = 5
        agent_timeout_input.max_value = 600
        agent_timeout_input.value = float(self._settings.get("agent_timeout_seconds", 60))
        agent_timeout_input.preferred_width = px(120)
        timeout_box.add_child(agent_timeout_input)
        agent_params_row.add_child(timeout_box)

        content.add_child(agent_params_row)

        agent_stream_input = Checkbox()
        agent_stream_input.text = "Stream responses"
        agent_stream_input.checked = bool(self._settings.get("agent_stream", True))
        content.add_child(agent_stream_input)

        dlg.content = content

        def _apply(result: str):
            if result != "OK":
                return
            self._set_models_dir(models_dir_input.text)
            limit_bytes = int(limit_input.value * _BYTES_PER_GIB)
            self._set_history_memory_limit_bytes(limit_bytes)
            self._settings.set("agent_api_base_url", agent_base_url_input.text.strip())
            self._settings.set("agent_api_key", agent_key_input.text.strip())
            self._settings.set("agent_model", agent_model_input.text.strip())
            self._settings.set("agent_temperature", float(agent_temperature_input.value))
            self._settings.set("agent_max_tokens", int(agent_max_tokens_input.value))
            self._settings.set("agent_timeout_seconds", float(agent_timeout_input.value))
            self._settings.set("agent_stream", bool(agent_stream_input.checked))
            self._statusbar.text = (
                f"Saved settings: models dir, history limit {limit_input.value:.2f} GiB, agent API"
            )

        dlg.on_result = _apply
        dlg.show(self.ui)
        self.ui.set_focus(limit_input)

    def _memory_status(self) -> str:
        hist = self._document.memory_bytes()
        cache = self._layer_stack.cache_memory_bytes()
        return f"Hist:{self._fmt_bytes(hist)} Cache:{self._fmt_bytes(cache)}"

    def _on_mouse_moved(self, x, y):
        if self._layer_stack.width == 0:
            return
        w, h = self._layer_stack.width, self._layer_stack.height
        layer = self._layer_stack.active_layer
        name = layer.name if layer else "-"
        bs = self._canvas.brush.size
        tool = self._canvas.brush_tool_mode.value
        mem = self._memory_status()
        if 0 <= x < w and 0 <= y < h:
            self._statusbar.text = f"{w}x{h} | ({x},{y}) | {name} | {tool}:{bs}px | {mem}"
        else:
            self._statusbar.text = f"{w}x{h} | {name} | {tool}:{bs}px | {mem}"

    def _on_color_picked(self, r, g, b, a):
        self._canvas.brush.set_color(r, g, b, a)
        self._brush_panel.sync_from_brush()

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def _apply_snapshot(self, snapshot: bytes):
        self._history_replaying = True
        try:
            self._layer_stack.load_state(snapshot)
        finally:
            self._history_replaying = False
            self._external_edit_ctx = None

    def _clear_history(self):
        self._document.clear_history()
        self._external_edit_ctx = None

    def undo(self):
        label = self._document.undo()
        if label is not None:
            self._statusbar.text = f"Undo: {label}"

    def redo(self):
        label = self._document.redo()
        if label is not None:
            self._statusbar.text = f"Redo: {label}"

    def _begin_external_edit(self, label: str, layer: Layer, target: str):
        if self._history_replaying:
            return
        if self._external_edit_ctx is not None:
            return
        if target == "selection":
            before_arr = self._layer_stack.selection.data.copy()
            self._external_edit_ctx = ExternalEditContext(
                label=label,
                layer_path="",
                target=target,
                before_arr=before_arr,
            )
            return
        layer_path = self._layer_stack.get_layer_path(layer)
        if not layer_path:
            return
        if target == "transform":
            self._external_edit_ctx = ExternalEditContext(
                label=label,
                layer_path=layer_path,
                target=target,
                before_offset=(layer.x, layer.y),
            )
            return
        if target == "mask":
            before_arr = layer.mask.data.copy()
        else:
            before_arr = layer.image.copy()
        self._external_edit_ctx = ExternalEditContext(
            label=label,
            layer_path=layer_path,
            target=target,
            before_arr=before_arr,
        )

    def _apply_layer_patch(self, layer_path: str, target: str, rect, patch: np.ndarray):
        if target == "selection":
            x0, y0, x1, y1 = rect
            if x1 <= x0 or y1 <= y0:
                return
            self._layer_stack.selection.data[y0:y1, x0:x1] = coerce_mask_data(patch)
            if self._layer_stack.on_changed:
                self._layer_stack.on_changed()
            return
        layer = self._layer_stack.get_layer_by_path(layer_path)
        if layer is None:
            return
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            return
        if target == "mask":
            layer.mask.data[y0:y1, x0:x1] = coerce_mask_data(patch)
            if self._layer_stack.on_changed:
                self._layer_stack.on_changed()
            return
        layer.image[y0:y1, x0:x1] = patch
        self._layer_stack.mark_layer_dirty(
            layer, rect=layer.local_rect_to_canvas(rect))
        if self._layer_stack.on_changed:
            self._layer_stack.on_changed()

    def _end_external_edit(self, layer: Layer, target: str, dirty_rect):
        if self._history_replaying:
            self._external_edit_ctx = None
            return
        if self._external_edit_ctx is None:
            return
        ctx = self._external_edit_ctx
        self._external_edit_ctx = None
        if target == "selection":
            if dirty_rect is None:
                return
            x0, y0, x1, y1 = dirty_rect
            if x1 <= x0 or y1 <= y0:
                return
            before_arr = ctx.before_arr
            after_arr = self._layer_stack.selection.data
            before_patch = before_arr[y0:y1, x0:x1].copy()
            after_patch = after_arr[y0:y1, x0:x1].copy()
            if np.array_equal(before_patch, after_patch):
                return
            rect = (x0, y0, x1, y1)
            label = ctx.label
            self._document.push_callbacks(
                label=label,
                undo_fn=lambda: self._apply_layer_patch("", target, rect, before_patch),
                redo_fn=lambda: self._apply_layer_patch("", target, rect, after_patch),
                size_bytes=before_patch.nbytes + after_patch.nbytes,
            )
            return
        if layer is None:
            return
        layer_path = self._layer_stack.get_layer_path(layer)
        if layer_path != ctx.layer_path or target != ctx.target:
            return
        if target == "transform":
            if ctx.before_offset is None:
                return
            before = ctx.before_offset
            after = (layer.x, layer.y)
            if before == after:
                return
            label = ctx.label

            def _apply_offset(offset):
                target_layer = self._layer_stack.get_layer_by_path(layer_path)
                if target_layer is None:
                    return
                old_bounds = target_layer.bounds
                self._layer_stack.set_layer_offset(
                    target_layer,
                    offset[0],
                    offset[1],
                    old_bounds=old_bounds,
                )

            self._document.push_callbacks(
                label=label,
                undo_fn=lambda: _apply_offset(before),
                redo_fn=lambda: _apply_offset(after),
                size_bytes=16,
            )
            return
        if dirty_rect is None:
            return
        x0, y0, x1, y1 = dirty_rect
        if x1 <= x0 or y1 <= y0:
            return
        before_arr = ctx.before_arr
        if target == "mask":
            after_arr = layer.mask.data
        else:
            after_arr = layer.image
        before_patch = before_arr[y0:y1, x0:x1].copy()
        after_patch = after_arr[y0:y1, x0:x1].copy()
        if np.array_equal(before_patch, after_patch):
            return
        rect = (x0, y0, x1, y1)
        label = ctx.label
        self._document.push_callbacks(
            label=label,
            undo_fn=lambda: self._apply_layer_patch(layer_path, target, rect, before_patch),
            redo_fn=lambda: self._apply_layer_patch(layer_path, target, rect, after_patch),
            size_bytes=before_patch.nbytes + after_patch.nbytes,
        )

    # ------------------------------------------------------------------
    # File operations
    # ------------------------------------------------------------------

    def new_project(self):
        # Simple: create white 1024x1024
        white = np.full((1024, 1024, 4), 255, dtype=np.uint8)
        self._layer_stack.init_from_image(white)
        self._clear_history()
        self._canvas.fit_in_view()
        self._project_path = None

    def new_project_from_image(self):
        def _on_result(path):
            if not path:
                return
            self._last_dir = os.path.dirname(path)
            self._settings.set("last_dir", self._last_dir)
            img = Image.open(path).convert("RGBA")
            arr = np.array(img, dtype=np.uint8)
            self._layer_stack.init_from_image(arr)
            self._clear_history()
            self._canvas.fit_in_view()
            self._project_path = None
        open_file_dialog(
            self.ui, _on_result,
            title="New From Image", directory=self._last_dir,
            filter_str="Images | *.png *.jpg *.jpeg *.bmp *.tiff *.webp")

    def open_file(self):
        def _on_result(path):
            if not path:
                return
            self._last_dir = os.path.dirname(path)
            self._settings.set("last_dir", self._last_dir)
            self.open_file_path(path)
        open_file_dialog(
            self.ui, _on_result,
            title="Open Project", directory=self._last_dir,
            filter_str="Diffusion Editor Project | *.deproj")

    def open_file_path(self, path: str):
        try:
            self._layer_stack.load_project(path)
            self._clear_history()
            self._canvas.fit_in_view()
            self._project_path = path
            self._statusbar.text = f"Opened: {os.path.basename(path)}"
        except Exception as e:
            log.exception(f"Open project failed: {path}")
            self._statusbar.text = f"Open error: {e}"

    def import_image(self):
        def _on_result(path):
            if not path:
                return
            self._last_dir = os.path.dirname(path)
            self._settings.set("last_dir", self._last_dir)
            self.import_image_path(path)
        open_file_dialog(
            self.ui, _on_result,
            title="Import Image", directory=self._last_dir,
            filter_str="Images | *.png *.jpg *.jpeg *.bmp *.tiff *.webp")

    def import_image_path(self, path: str):
        img = Image.open(path).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)
        self._layer_stack.init_from_image(arr)
        self._clear_history()
        self._canvas.fit_in_view()
        self._project_path = None

    def _new_layer(self):
        self._document.execute(AddLayerCommand(
            name=self._layer_stack.next_name("Layer"),
        ))

    def _remove_layer(self, layer=None):
        if layer is None:
            layer = self._layer_stack.active_layer
        if layer is None:
            return

        def _on_result(btn: str):
            if btn == "Yes":
                self._document.execute(RemoveLayerCommand(layer=layer))

        MessageBox.question(
            self.ui,
            "Delete Layer",
            f"Delete layer \"{layer.name}\"?",
            buttons=Buttons.YES_NO,
            on_result=_on_result,
        )

    def _flatten_layers(self):
        self._document.execute(FlattenLayersCommand())

    def _select_all(self):
        self._document.execute(SelectAllCommand())

    def _clear_selection(self):
        self._document.execute(ClearSelectionCommand())

    def _invert_selection(self):
        self._document.execute(InvertSelectionCommand())

    def _copy(self):
        """Copy selected pixels from the active layer to clipboard."""
        layer = self._layer_stack.active_layer
        bbox = self._layer_stack.selection.bbox()
        if layer is None or bbox is None:
            self._statusbar.text = "Copy: nothing selected"
            return
        x0, y0, x1, y1 = bbox
        patch = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        ix0 = max(x0, layer.x)
        iy0 = max(y0, layer.y)
        ix1 = min(x1, layer.x + layer.width)
        iy1 = min(y1, layer.y + layer.height)
        if ix1 > ix0 and iy1 > iy0:
            src_x0 = ix0 - layer.x
            src_y0 = iy0 - layer.y
            src_x1 = src_x0 + (ix1 - ix0)
            src_y1 = src_y0 + (iy1 - iy0)
            dst_x0 = ix0 - x0
            dst_y0 = iy0 - y0
            dst_x1 = dst_x0 + (ix1 - ix0)
            dst_y1 = dst_y0 + (iy1 - iy0)
            patch[dst_y0:dst_y1, dst_x0:dst_x1] = (
                layer.image[src_y0:src_y1, src_x0:src_x1])
        mask = self._layer_stack.selection.data[y0:y1, x0:x1]
        patch[:, :, 3] = (patch[:, :, 3].astype(np.float32) * mask).astype(np.uint8)
        self._clipboard = patch
        self._clipboard_pos = (x0, y0)
        self._statusbar.text = f"Copied {x1 - x0}x{y1 - y0}"

    def _copy_visible(self):
        """Copy selected pixels from the composite of all visible layers to clipboard."""
        bbox = self._layer_stack.selection.bbox()
        if bbox is None:
            self._statusbar.text = "Copy Visible: nothing selected"
            return
        x0, y0, x1, y1 = bbox
        composite = self._layer_stack.composite()
        patch = composite[y0:y1, x0:x1].copy()
        mask = self._layer_stack.selection.data[y0:y1, x0:x1]
        patch[:, :, 3] = (patch[:, :, 3].astype(np.float32) * mask).astype(np.uint8)
        self._clipboard = patch
        self._clipboard_pos = (x0, y0)
        self._statusbar.text = f"Copied {x1 - x0}x{y1 - y0} (visible)"

    def _paste(self):
        """Paste clipboard as a new floating layer."""
        if self._clipboard is None:
            self._statusbar.text = "Paste: clipboard is empty"
            return
        ph, pw = self._clipboard.shape[:2]
        if self._clipboard_pos is None:
            ch, cw = self._layer_stack.height, self._layer_stack.width
            px = (cw - pw) // 2
            py = (ch - ph) // 2
        else:
            px, py = self._clipboard_pos
        self._document.execute(AddLayerCommand(
            name="Floating Selection",
            image=self._clipboard.copy(),
            x=px,
            y=py,
            label="Paste",
        ))
        self._statusbar.text = f"Pasted {pw}x{ph}"

    def _move_layer(self, layer: Layer, new_parent: Layer | None, index: int):
        self._document.execute(MoveLayerCommand(
            layer=layer,
            new_parent=new_parent,
            index=index,
        ))

    def _set_layer_visibility(self, layer: Layer, visible: bool):
        self._document.execute(SetLayerVisibilityCommand(
            layer=layer,
            visible=visible,
        ))

    def _toggle_layer_solo(self, layer: Layer):
        was_solo = self._layer_stack.solo_layer_id == layer.id
        self._layer_stack.toggle_solo_layer(layer)
        self._statusbar.text = "Solo off" if was_solo else f"Solo: {layer.name}"

    def _set_layer_opacity(self, layer: Layer, opacity: float):
        self._document.execute(SetLayerOpacityCommand(
            layer=layer,
            opacity=opacity,
        ))

    def _rename_layer(self, layer: Layer, name: str):
        self._document.execute(SetLayerNameCommand(layer=layer, name=name))

    def save_file(self):
        if self._project_path:
            try:
                self._layer_stack.save_project(self._project_path)
                self._statusbar.text = f"Saved: {self._project_path}"
            except Exception as e:
                log.exception(f"Save project failed: {self._project_path}")
                self._statusbar.text = f"Save error: {e}"
        else:
            self.save_file_as()

    def save_file_as(self):
        def _on_result(path):
            if not path:
                return
            if not path.lower().endswith(".deproj"):
                path += ".deproj"
            self._last_dir = os.path.dirname(path)
            self._settings.set("last_dir", self._last_dir)
            try:
                self._layer_stack.save_project(path)
                self._project_path = path
                self._statusbar.text = f"Saved: {os.path.basename(path)}"
            except Exception as e:
                log.exception(f"Save project failed: {path}")
                self._statusbar.text = f"Save error: {e}"
        save_file_dialog(
            self.ui, _on_result,
            title="Save Project", directory=self._last_dir,
            filter_str="Diffusion Editor Project | *.deproj")

    def _normalize_export_image_path(self, path: str) -> tuple[str | None, str | None]:
        _root, ext = os.path.splitext(path)
        if not ext:
            return path + ".png", None
        ext = ext.lower()
        if ext not in _EXPORT_IMAGE_EXTENSIONS:
            known = ", ".join(sorted(_EXPORT_IMAGE_EXTENSIONS))
            return None, f"Unknown export extension '{ext}'. Use {known}."
        return path, None

    def export_image_path(self, path: str) -> bool:
        normalized, error = self._normalize_export_image_path(path)
        if error is not None:
            self._statusbar.text = f"Export error: {error}"
            return False

        path = normalized
        arr = np.ascontiguousarray(self._layer_stack.composite())
        if arr is None or arr.size == 0:
            self._statusbar.text = "Export error: nothing to export"
            return False

        img = Image.fromarray(arr, "RGBA")
        if path.lower().endswith((".jpg", ".jpeg")):
            img = img.convert("RGB")
        try:
            img.save(path)
            self._statusbar.text = f"Exported: {os.path.basename(path)}"
            return True
        except Exception as e:
            log.exception(f"Export image failed: {path}")
            self._statusbar.text = f"Export error: {e}"
            return False

    def export_image(self):
        def _on_result(path):
            if not path:
                return
            normalized, error = self._normalize_export_image_path(path)
            if error is not None:
                self._statusbar.text = f"Export error: {error}"
                return
            path = normalized
            self._last_dir = os.path.dirname(path)
            self._settings.set("last_dir", self._last_dir)
            self.export_image_path(path)
        save_file_dialog(
            self.ui, _on_result,
            title="Export Image", directory=self._last_dir,
            filter_str="PNG | *.png;;JPEG | *.jpg *.jpeg")

    def _fit(self):
        self._canvas.fit_in_view()

    def _quit(self):
        self._running = False

    # ------------------------------------------------------------------
    # Layer tools
    # ------------------------------------------------------------------

    def _attach_tool_to_layer(self, layer: Layer, tool_type: str):
        if layer is None or layer.tool is not None:
            return
        maker = {
            "diffusion": self._make_diffusion_tool,
            "lama": self._make_lama_tool,
            "instruct": self._make_instruct_tool,
        }.get(tool_type)
        if maker is None:
            return
        tool = maker(layer)
        if tool is None:
            return
        labels = {
            "diffusion": "Attach Diffusion Tool",
            "lama": "Attach LaMa Tool",
            "instruct": "Attach Instruct Tool",
        }
        self._document.execute(AttachLayerToolCommand(
            layer=layer,
            tool=tool,
            label=labels[tool_type],
        ))

    def _detach_tool_from_layer(self, layer: Layer):
        if layer is None or layer.tool is None:
            return
        self._diffusion_controller.clear_pending_layer(layer)
        self._lama_controller.clear_pending_layer(layer)
        self._instruct_controller.clear_pending_layer(layer)
        self._segmentation_controller.clear_pending_layer(layer)
        self._grounding_controller.clear_pending_layer(layer)
        self._document.execute(DetachLayerToolCommand(layer=layer))

    def _tool_source_composite(self, layer: Layer) -> np.ndarray | None:
        composite = self._canvas.get_composite_below(layer)
        if composite is None:
            composite = self._canvas.get_composite()
        return composite

    # ------------------------------------------------------------------
    # Diffusion
    # ------------------------------------------------------------------

    def _on_load_model(self, path: str, prediction_type: str):
        event = self._diffusion_controller.submit_load_model(
            path,
            prediction_type,
        )
        if event.status:
            self._statusbar.text = event.status

    def _make_diffusion_tool(self, layer: Layer) -> DiffusionTool | None:
        mode = self._diffusion_panel.mode
        center_x, center_y = self._canvas.view_center_image()

        if mode == "txt2img":
            patch_pil = None
            ppx, ppy = 0, 0
            pw = self._layer_stack.width
            ph = self._layer_stack.height
        else:
            composite = self._tool_source_composite(layer)
            if composite is None:
                return None
            patch = source_patch_at_center(
                composite, center_x, center_y)
            ppx, ppy, px1, py1 = patch.canvas_rect
            patch_pil = patch.image
            pw, ph = px1 - ppx, py1 - ppy

        seed = self._diffusion_panel.seed
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
            self._diffusion_panel.set_seed(seed)

        return DiffusionTool(
            source_patch=patch_pil,
            patch_x=ppx, patch_y=ppy, patch_w=pw, patch_h=ph,
            prompt=self._diffusion_panel.prompt,
            negative_prompt=self._diffusion_panel.negative_prompt,
            strength=self._diffusion_panel.strength,
            guidance_scale=self._diffusion_panel.guidance_scale,
            steps=self._diffusion_panel.steps,
            seed=seed,
            model_path=self._engine.model_path or "",
            prediction_type=self._diffusion_panel.prediction_type,
            mode=mode,
        )

    def _on_clear_mask(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer.tool, DiffusionTool):
            self._document.execute(ClearLayerMaskCommand(
                layer=layer,
                label="Clear Diffusion Mask",
            ))

    def _sync_panel_to_layer(self, tool: DiffusionTool):
        tool.prompt = self._diffusion_panel.prompt
        tool.negative_prompt = self._diffusion_panel.negative_prompt
        tool.strength = self._diffusion_panel.strength
        tool.guidance_scale = self._diffusion_panel.guidance_scale
        tool.steps = self._diffusion_panel.steps
        tool.seed = self._diffusion_panel.seed
        tool.mode = self._diffusion_panel.mode
        tool.masked_content = self._diffusion_panel.masked_content
        tool.ip_adapter_scale = self._diffusion_panel.ip_adapter_scale
        tool.resize_to_model_resolution = self._diffusion_panel.resize_to_model_resolution
        if self._engine.model_path:
            tool.model_path = self._engine.model_path
        tool.prediction_type = self._diffusion_panel.prediction_type

    def _on_regenerate(self):
        layer = self._layer_stack.active_layer
        if layer is None or not isinstance(layer.tool, DiffusionTool):
            return
        if self._engine.is_busy:
            return
        tool = layer.tool
        self._sync_panel_to_layer(tool)

        event = self._diffusion_controller.start_regeneration(layer)
        if event.status:
            self._statusbar.text = event.status

    def _on_new_seed(self):
        layer = self._layer_stack.active_layer
        if layer is None or not isinstance(layer.tool, DiffusionTool):
            return
        new_seed = random.randint(0, 2**32 - 1)
        layer.tool.seed = new_seed
        self._diffusion_panel.set_seed(new_seed)
        self._on_regenerate()

    def _on_load_ip_adapter(self):
        event = self._diffusion_controller.submit_load_ip_adapter()
        if event.status:
            self._statusbar.text = event.status

    def _on_pick_ip_adapter_layer(self):
        layer = self._layer_stack.active_layer
        if layer is None or not isinstance(layer.tool, DiffusionTool):
            return

        def _apply(reference_layer: Layer) -> None:
            self._document.execute(SetIpAdapterReferenceLayerCommand(
                layer=layer,
                reference_layer_id=reference_layer.id,
                reference_layer_name_hint=reference_layer.name,
            ))
            self._statusbar.text = (
                f"IP-Adapter reference: {reference_layer.name}"
            )

        if self.ui is None:
            return
        shown = show_ip_adapter_reference_dialog(
            self.ui,
            self._layer_stack,
            layer,
            layer.tool.ip_adapter_layer_id,
            _apply,
        )
        if not shown:
            self._statusbar.text = "No reference layers available"

    def _on_clear_ip_adapter_layer(self):
        layer = self._layer_stack.active_layer
        if layer is not None and isinstance(layer.tool, DiffusionTool):
            self._document.execute(ClearIpAdapterReferenceLayerCommand(layer=layer))
            self._statusbar.text = "IP-Adapter reference cleared"

    def _on_patch_rect_drawn(self, x0, y0, x1, y1):
        layer = self._layer_stack.active_layer
        if layer is not None:
            self._brush_panel.set_draw_patch_checked(False)
            lx0, ly0, lx1, ly1 = layer.bounds
            x0, y0 = max(x0, lx0), max(y0, ly0)
            x1, y1 = min(x1, lx1), min(y1, ly1)
            if x1 - x0 <= 2 or y1 - y0 <= 2:
                return
            rect = layer.canvas_rect_to_local((x0, y0, x1, y1))
            self._document.execute(SetLayerPatchRectCommand(
                layer=layer,
                rect=rect,
                label="Set Patch Rect",
            ))

    def _on_clear_patch_rect(self):
        layer = self._layer_stack.active_layer
        if layer is not None:
            self._document.execute(ClearLayerPatchRectCommand(
                layer=layer,
                label="Clear Patch Rect",
            ))

    def _on_selection_rect_drawn(self, x0: int, y0: int, x1: int, y1: int):
        """Create a rectangular selection from the dragged rect."""
        h, w = self._layer_stack.height, self._layer_stack.width
        if h == 0 or w == 0:
            return
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return
        mask = np.zeros((h, w), dtype=np.float32)
        mask[y0:y1, x0:x1] = 1.0
        self._document.execute(SetLayerSelectionCommand(
            mask=mask,
            label="Rect Selection",
        ))
        # Uncheck the rect button in the panel
        self._selection_panel._rect_mode_cb.checked = False

    # ------------------------------------------------------------------
    # LaMa
    # ------------------------------------------------------------------

    def _make_lama_tool(self, layer: Layer) -> LamaTool | None:
        composite = self._tool_source_composite(layer)
        if composite is None:
            return None
        cx, cy = self._canvas.view_center_image()
        patch = source_patch_at_center(composite, cx, cy)
        ppx, ppy, px1, py1 = patch.canvas_rect
        return LamaTool(
            source_patch=patch.image,
            patch_x=ppx, patch_y=ppy, patch_w=px1 - ppx, patch_h=py1 - ppy,
        )

    def _on_lama_remove(self):
        layer = self._layer_stack.active_layer
        if layer is None or not isinstance(layer.tool, LamaTool):
            return
        event = self._lama_controller.start_remove(layer)
        if event.status:
            self._statusbar.text = event.status

    def _on_lama_clear_mask(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer.tool, LamaTool):
            self._document.execute(ClearLayerMaskCommand(
                layer=layer,
                label="Clear LaMa Mask",
            ))

    def _on_lama_select_background(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer.tool, LamaTool):
            return
        event = self._segmentation_controller.start_select_background(layer)
        if event.status:
            self._statusbar.text = event.status

    # ------------------------------------------------------------------
    # InstructPix2Pix
    # ------------------------------------------------------------------

    def _make_instruct_tool(self, layer: Layer) -> InstructTool | None:
        composite = self._tool_source_composite(layer)
        if composite is None:
            return None
        cx, cy = self._canvas.view_center_image()
        patch = source_patch_at_center(composite, cx, cy)
        ppx, ppy, px1, py1 = patch.canvas_rect
        seed = self._instruct_panel.seed
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
            self._instruct_panel.set_seed(seed)
        return InstructTool(
            source_patch=patch.image,
            patch_x=ppx, patch_y=ppy, patch_w=px1 - ppx, patch_h=py1 - ppy,
            instruction=self._instruct_panel.instruction,
            image_guidance_scale=self._instruct_panel.image_guidance_scale,
            guidance_scale=self._instruct_panel.guidance_scale,
            steps=self._instruct_panel.steps,
            seed=seed,
        )

    def _on_instruct_load_model(self):
        event = self._instruct_controller.submit_load_model()
        self._handle_instruct_event(event)

    def _on_instruct_apply(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer.tool, InstructTool):
            return
        tool = layer.tool

        tool.instruction = self._instruct_panel.instruction
        tool.image_guidance_scale = self._instruct_panel.image_guidance_scale
        tool.guidance_scale = self._instruct_panel.guidance_scale
        tool.steps = self._instruct_panel.steps
        tool.seed = self._instruct_panel.seed

        event = self._instruct_controller.start_apply(layer)
        self._handle_instruct_event(event)

    def _on_instruct_new_seed(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer.tool, InstructTool):
            return
        new_seed = random.randint(0, 2**32 - 1)
        layer.tool.seed = new_seed
        self._instruct_panel.set_seed(new_seed)
        self._on_instruct_apply()

    def _on_instruct_clear_mask(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer.tool, InstructTool):
            self._document.execute(ClearLayerMaskCommand(
                layer=layer,
                label="Clear Instruct Mask",
            ))

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _on_select_background(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer.tool, DiffusionTool):
            return
        event = self._segmentation_controller.start_select_background(layer)
        if event.status:
            self._statusbar.text = event.status

    # ------------------------------------------------------------------
    # Polling (called every frame from SDL main loop)
    # ------------------------------------------------------------------

    def poll(self):
        self._agent_chat_panel.poll()
        self._poll_segmentation()
        self._poll_lama()
        self._poll_instruct()
        self._poll_diffusion()
        self._poll_grounding()

    def _poll_segmentation(self):
        event = self._segmentation_controller.poll()
        if event is None:
            return
        if event.segmentation_result is not None:
            layer, seg_mask = event.segmentation_result
            command, status = map_segmentation_result(layer, seg_mask)
            if command is not None:
                self._document.execute(command)
            self._statusbar.text = status
            return
        if event.status:
            self._statusbar.text = event.status

    def _poll_lama(self):
        event = self._lama_controller.poll()
        if event is None:
            return
        if event.inference_result is not None:
            layer, result_image = event.inference_result
            command, status = map_lama_result(layer, result_image)
            if command is not None:
                self._document.execute(command)
            self._statusbar.text = status
        elif event.status:
            self._statusbar.text = event.status

    def _poll_instruct(self):
        event = self._instruct_controller.poll()
        if event is not None:
            self._handle_instruct_event(event)

    def _handle_instruct_event(self, event):
        if event.model_loading:
            self._instruct_panel.set_model_loading()
        if event.model_error is not None:
            self._instruct_panel.on_model_load_error(event.model_error)
        elif event.model_loaded:
            self._instruct_panel.on_model_loaded()
        if event.inference_result is not None:
            layer, result_image, used_seed = event.inference_result
            command, status = map_instruct_result(
                layer, result_image, used_seed)
            if command is not None:
                self._document.execute(command)
            self._statusbar.text = status
            return
        if event.status:
            self._statusbar.text = event.status

    def _poll_diffusion(self):
        event = self._diffusion_controller.poll()
        if event is None:
            return

        if event.model_error is not None:
            self._diffusion_panel.on_model_load_error(event.model_error)
        elif event.model_loaded_path is not None:
            self._diffusion_panel.on_model_loaded(
                event.model_loaded_path,
                self._engine.model_info,
            )

        if event.ip_adapter_error is not None:
            self._diffusion_panel.on_ip_adapter_load_error(event.ip_adapter_error)
        elif event.ip_adapter_loaded:
            self._diffusion_panel.on_ip_adapter_loaded()

        if event.inference_result is not None:
            pending, result_image, used_seed = event.inference_result
            command, status = map_diffusion_result(
                pending, result_image, used_seed)
            if command is not None:
                self._document.execute(command)
            self._statusbar.text = status
            return

        if event.status is not None:
            self._statusbar.text = event.status

    def _show_grounding_dialog(self) -> None:
        from ..ui.dialogs.grounding_dialog import GroundingDialog

        GroundingDialog(
            ui=self.ui,
            on_submit=self._on_grounding_detect,
            gpu_available=self._grounding_controller.gpu_available(),
        ).show()

    def _on_grounding_detect(self, params: GroundingParams) -> None:
        active = self._layer_stack.active_layer
        if active is None:
            log.warning("Grounding: no active layer")
            self._statusbar.text = "Grounding: no active layer"
            return
        event = self._grounding_controller.start_detection(active, params)
        if event.status:
            self._statusbar.text = event.status

    def _poll_grounding(self) -> None:
        event = self._grounding_controller.poll()
        if event is None:
            return
        if event.grounding_result is not None:
            layer, result = event.grounding_result
            command, status = map_grounding_result(layer, result)
            if command is not None:
                self._document.execute(command)
            self._statusbar.text = status
            return
        if event.status:
            self._statusbar.text = event.status

    # ------------------------------------------------------------------
    # Public: rendering
    # ------------------------------------------------------------------

    # Background colour painted by the UIRenderer's offscreen clear
    # so transparent UI regions show this through the composite.
    UI_BACKGROUND = (0.12, 0.12, 0.14, 1.0)

    def render(self, vw: int, vh: int):
        self.ui.render(vw, vh, background_color=self.UI_BACKGROUND)

    def render_compose(self, vw: int, vh: int):
        """Render the UI and return the composite TextureHandle.

        Preferred when the host owns an SDLBackendWindow and publishes the
        result via ``win.present(tex)`` — works on both OpenGL and
        Vulkan. ``None`` if the UI is empty.
        """
        return self.ui.render_compose(vw, vh, background_color=self.UI_BACKGROUND)

    @property
    def running(self) -> bool:
        return self._running

    def close(self):
        """Release runtime resources (GPU/engines). Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        self._running = False
        if self._canvas is not None:
            self._canvas.dispose()
        if self._agent_chat_panel is not None:
            self._agent_chat_panel.shutdown()
        self._engine.shutdown()
        self._instruct_engine.shutdown()
        self._lama_engine.shutdown()
        self._seg_engine.shutdown()
        self._grounding_engine.shutdown()
