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
from tcgui.widgets.units import px, pct
from tcgui.widgets.splitter import Splitter

from .layer_stack import LayerStack
from .layer import Layer, DiffusionLayer, LamaLayer, InstructLayer
from .editor_canvas import EditorCanvas
from .layer_panel import LayerPanel
from .brush_panel import BrushPanel
from .diffusion_panel import DiffusionPanel
from .lama_panel import LamaPanel
from .instruct_panel import InstructPanel
from .diffusion_engine import DiffusionEngine
from .lama_engine import LamaEngine
from .instruct_engine import InstructEngine
from .segmentation import SegmentationEngine
from .diffusion_brush import extract_patch, extract_mask_patch
from .file_dialog import open_file_dialog, save_file_dialog, open_directory_dialog
from .settings import Settings
from .history import HistoryManager
from .document_service import DocumentService
from .commands import (
    AddLayerCommand, InsertLayerCommand, RemoveLayerCommand,
    MoveLayerCommand, SetLayerVisibilityCommand, SetLayerOpacityCommand,
    FlattenLayersCommand, ClearLayerMaskCommand,
    SetDiffusionIpAdapterRectCommand, ClearDiffusionIpAdapterRectCommand,
    SetManualPatchRectCommand, ClearManualPatchRectCommand,
)
from .engine_result_mapper import (
    map_segmentation_result, map_lama_result,
    map_instruct_result, map_diffusion_result,
)

_BYTES_PER_GIB = 1024 * 1024 * 1024
_DEFAULT_HISTORY_MEMORY_LIMIT_BYTES = 5 * _BYTES_PER_GIB
_MIN_HISTORY_MEMORY_LIMIT_GIB = 0.25
_MAX_HISTORY_MEMORY_LIMIT_GIB = 256.0


@dataclass
class ExternalEditContext:
    label: str
    layer_path: str
    target: str
    before_arr: np.ndarray


class EditorWindow:
    """Non-widget orchestrator: assembles UI, wires callbacks, handles logic."""

    def __init__(self, graphics):
        self._running = True
        self._closed = False
        self._settings = Settings()
        self._project_path: str | None = None
        self._last_dir: str = self._settings.get("last_dir", "")
        self._history_memory_limit_bytes = self._load_history_memory_limit_bytes()
        self._models_dir = self._load_models_dir()
        self._pending_request = None
        self._pending_lama_layer = None
        self._pending_instruct_layer = None
        self._history_replaying = False
        self._external_edit_ctx: ExternalEditContext | None = None

        # Layer stack
        self._layer_stack = LayerStack()

        # Engines
        self._engine = DiffusionEngine()
        self._seg_engine = SegmentationEngine()
        self._lama_engine = LamaEngine()
        self._instruct_engine = InstructEngine()
        self._history = HistoryManager(
            self._apply_snapshot,
            max_memory_bytes=self._history_memory_limit_bytes,
        )
        self._document = DocumentService(
            self._layer_stack,
            self._history,
            self._apply_snapshot,
        )

        # Build UI
        self._build_ui(graphics)

        # Wire callbacks
        self._wire_callbacks()
        self._menu_bar.register_shortcuts(self.ui)

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self, graphics):
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
        root.add_child(self._toolbar)

        # Main content area: left panel | canvas | right panel
        main_area = HStack()
        main_area.stretch = True
        main_area.spacing = 0

        # Left panel container
        self._left_container = VStack()
        self._left_container.preferred_width = px(260)
        self._left_container.clip = True

        # Create all panels (only one visible at a time, stretch to fill)
        self._brush_panel = BrushPanel(self._canvas_placeholder_brush())
        self._diffusion_panel = DiffusionPanel()
        self._diffusion_panel.set_models_dir(self._models_dir)
        self._lama_panel = LamaPanel()
        self._instruct_panel = InstructPanel()

        for p in (self._brush_panel, self._diffusion_panel,
                  self._lama_panel, self._instruct_panel):
            p.visible = False
            p.stretch = True
            self._left_container.add_child(p)

        main_area.add_child(self._left_container)
        main_area.add_child(Splitter(target=self._left_container, side="left"))

        # Canvas (center, stretches to fill remaining space)
        self._canvas = EditorCanvas(self._layer_stack, graphics=graphics)
        self._canvas.stretch = True
        # Give brush reference now
        self._brush_panel._brush = self._canvas.brush
        main_area.add_child(self._canvas)

        # Right panel: layer panel
        self._layer_panel = LayerPanel(self._layer_stack)
        main_area.add_child(Splitter(target=self._layer_panel, side="right"))
        main_area.add_child(self._layer_panel)

        root.add_child(main_area)

        # Status bar
        self._statusbar = StatusBar()
        self._statusbar.text = "Ready"
        root.add_child(self._statusbar)

        # Create UI
        self.ui = UI(graphics)
        self.ui.root = root

    def _canvas_placeholder_brush(self):
        """Temporary brush for panel construction (replaced after canvas created)."""
        from .brush import Brush
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
        edit_menu.add_item(MenuItem(separator=True))
        edit_menu.add_item(MenuItem("Settings...", on_click=self._show_settings_dialog))
        self._menu_bar.add_menu("Edit", edit_menu)

        # Layer menu
        layer_menu = Menu()
        layer_menu.add_item(MenuItem("New Layer", shortcut="Ctrl+Shift+N", on_click=self._new_layer))
        layer_menu.add_item(MenuItem("Remove Layer", on_click=self._remove_layer))
        layer_menu.add_item(MenuItem(separator=True))
        layer_menu.add_item(MenuItem("Flatten", on_click=self._flatten_layers))
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
        self._brush_panel.on_eraser_toggled = self._canvas.set_brush_eraser

        # Diffusion panel
        self._diffusion_panel.on_load_model = self._on_load_model
        self._diffusion_panel.on_regenerate = self._on_regenerate
        self._diffusion_panel.on_new_seed = self._on_new_seed
        self._diffusion_panel.on_clear_mask = self._on_clear_mask
        self._diffusion_panel.on_mask_brush_changed = self._canvas.set_mask_brush
        self._diffusion_panel.on_mask_eraser_toggled = self._canvas.set_mask_eraser
        self._diffusion_panel.on_show_mask_toggled = self._canvas.set_show_mask
        self._diffusion_panel.on_load_ip_adapter = self._on_load_ip_adapter
        self._diffusion_panel.on_draw_rect_toggled = self._canvas.set_ref_rect_mode
        self._diffusion_panel.on_show_rect_toggled = self._canvas.set_show_ref_rect
        self._diffusion_panel.on_clear_rect = self._on_clear_ref_rect
        self._diffusion_panel.on_select_background = self._on_select_background
        self._diffusion_panel.on_draw_patch_toggled = self._canvas.set_patch_rect_mode
        self._diffusion_panel.on_clear_patch = self._on_clear_patch_rect
        self._canvas.on_ref_rect_drawn = self._on_ref_rect_drawn
        self._canvas.on_patch_rect_drawn = self._on_patch_rect_drawn

        # Layer panel
        self._layer_panel.on_create_diffusion = self._on_create_diffusion
        self._layer_panel.on_create_lama = self._on_create_lama
        self._layer_panel.on_create_instruct = self._on_create_instruct
        self._layer_panel.on_add_layer = self._new_layer
        self._layer_panel.on_remove_layer = self._remove_layer
        self._layer_panel.on_flatten_layers = self._flatten_layers
        self._layer_panel.on_move_layer = self._move_layer
        self._layer_panel.on_toggle_visibility = self._set_layer_visibility
        self._layer_panel.on_opacity_changed = self._set_layer_opacity

        # LaMa panel
        self._lama_panel.on_remove = self._on_lama_remove
        self._lama_panel.on_clear_mask = self._on_lama_clear_mask
        self._lama_panel.on_mask_brush_changed = self._canvas.set_mask_brush
        self._lama_panel.on_mask_eraser_toggled = self._canvas.set_mask_eraser
        self._lama_panel.on_show_mask_toggled = self._canvas.set_show_mask
        self._lama_panel.on_select_background = self._on_lama_select_background

        # Instruct panel
        self._instruct_panel.on_load_model = self._on_instruct_load_model
        self._instruct_panel.on_apply = self._on_instruct_apply
        self._instruct_panel.on_new_seed = self._on_instruct_new_seed
        self._instruct_panel.on_clear_mask = self._on_instruct_clear_mask
        self._instruct_panel.on_mask_brush_changed = self._canvas.set_mask_brush
        self._instruct_panel.on_mask_eraser_toggled = self._canvas.set_mask_eraser
        self._instruct_panel.on_show_mask_toggled = self._canvas.set_show_mask
        self._instruct_panel.on_draw_patch_toggled = self._canvas.set_patch_rect_mode
        self._instruct_panel.on_clear_patch = self._on_instruct_clear_patch_rect

    # ------------------------------------------------------------------
    # Panel switching
    # ------------------------------------------------------------------

    def _on_layer_changed(self):
        layer = self._layer_stack.active_layer
        self._brush_panel.visible = False
        self._diffusion_panel.visible = False
        self._lama_panel.visible = False
        self._instruct_panel.visible = False

        if layer is None:
            pass
        elif isinstance(layer, DiffusionLayer):
            self._diffusion_panel.show_diffusion_layer(layer)
            self._diffusion_panel.visible = True
        elif isinstance(layer, LamaLayer):
            self._lama_panel.show_lama_layer(layer)
            self._lama_panel.visible = True
        elif isinstance(layer, InstructLayer):
            self._instruct_panel.show_instruct_layer(layer)
            self._instruct_panel.visible = True
        else:
            self._brush_panel.visible = True

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

        dlg.content = content

        def _apply(result: str):
            if result != "OK":
                return
            self._set_models_dir(models_dir_input.text)
            limit_bytes = int(limit_input.value * _BYTES_PER_GIB)
            self._set_history_memory_limit_bytes(limit_bytes)
            self._statusbar.text = (
                f"Saved settings: models dir, history limit {limit_input.value:.2f} GiB"
            )

        dlg.on_result = _apply
        dlg.show(self.ui)
        self.ui.set_focus(limit_input)

    def _memory_status(self) -> str:
        hist = self._document.memory_bytes()
        cache = self._layer_stack._renderer.cache_memory_bytes()
        return f"Hist:{self._fmt_bytes(hist)} Cache:{self._fmt_bytes(cache)}"

    def _on_mouse_moved(self, x, y):
        if self._layer_stack.width == 0:
            return
        w, h = self._layer_stack.width, self._layer_stack.height
        layer = self._layer_stack.active_layer
        name = layer.name if layer else "-"
        bs = self._canvas.brush.size
        mem = self._memory_status()
        if 0 <= x < w and 0 <= y < h:
            self._statusbar.text = f"{w}x{h} | ({x},{y}) | {name} | Brush:{bs}px | {mem}"
        else:
            self._statusbar.text = f"{w}x{h} | {name} | Brush:{bs}px | {mem}"

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
        layer_path = self._layer_stack.get_layer_path(layer)
        if not layer_path:
            return
        if target == "mask":
            before_arr = layer.mask.copy()
        else:
            before_arr = layer.image.copy()
        self._external_edit_ctx = ExternalEditContext(
            label=label,
            layer_path=layer_path,
            target=target,
            before_arr=before_arr,
        )

    def _apply_layer_patch(self, layer_path: str, target: str, rect, patch: np.ndarray):
        layer = self._layer_stack.get_layer_by_path(layer_path)
        if layer is None:
            return
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            return
        if target == "mask" and isinstance(layer, (DiffusionLayer, LamaLayer, InstructLayer)):
            layer.mask[y0:y1, x0:x1] = patch
            if self._layer_stack.on_changed:
                self._layer_stack.on_changed()
            return
        layer.image[y0:y1, x0:x1] = patch
        self._layer_stack.mark_layer_dirty(layer, rect=rect)
        if self._layer_stack.on_changed:
            self._layer_stack.on_changed()

    def _end_external_edit(self, layer: Layer, target: str, dirty_rect):
        if self._history_replaying:
            self._external_edit_ctx = None
            return
        if self._external_edit_ctx is None:
            return
        if layer is None:
            self._external_edit_ctx = None
            return
        ctx = self._external_edit_ctx
        self._external_edit_ctx = None
        layer_path = self._layer_stack.get_layer_path(layer)
        if layer_path != ctx.layer_path or target != ctx.target:
            return
        if dirty_rect is None:
            return
        x0, y0, x1, y1 = dirty_rect
        if x1 <= x0 or y1 <= y0:
            return
        before_arr = ctx.before_arr
        if target == "mask" and isinstance(layer, (DiffusionLayer, LamaLayer, InstructLayer)):
            after_arr = layer.mask
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

    def _set_layer_opacity(self, layer: Layer, opacity: float):
        self._document.execute(SetLayerOpacityCommand(
            layer=layer,
            opacity=opacity,
        ))

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

    def export_image(self):
        def _on_result(path):
            if not path:
                return
            self._last_dir = os.path.dirname(path)
            self._settings.set("last_dir", self._last_dir)
            arr = self._canvas.get_composite()
            if arr is None:
                return
            img = Image.fromarray(arr, "RGBA")
            if path.lower().endswith((".jpg", ".jpeg")):
                img = img.convert("RGB")
            try:
                img.save(path)
                self._statusbar.text = f"Exported: {os.path.basename(path)}"
            except Exception as e:
                log.exception(f"Export image failed: {path}")
                self._statusbar.text = f"Export error: {e}"
        save_file_dialog(
            self.ui, _on_result,
            title="Export Image", directory=self._last_dir,
            filter_str="PNG | *.png;;JPEG | *.jpg *.jpeg")
        self._statusbar.text = f"Exported: {path}"

    def _fit(self):
        self._canvas.fit_in_view()

    def _quit(self):
        self._running = False

    # ------------------------------------------------------------------
    # Diffusion
    # ------------------------------------------------------------------

    def _on_load_model(self, path: str, prediction_type: str):
        if self._engine.is_busy:
            return
        pred = prediction_type if prediction_type else None
        self._engine.submit_load(path, pred)

    def _on_create_diffusion(self):
        composite = self._canvas.get_composite()
        if composite is None:
            return
        mode = self._diffusion_panel.mode
        center_x, center_y = self._canvas.view_center_image()

        if mode == "txt2img":
            patch_pil = None
            ppx, ppy = 0, 0
            pw = self._layer_stack.width
            ph = self._layer_stack.height
        else:
            patch_pil, ppx, ppy, pw, ph = extract_patch(
                composite, center_x, center_y)

        seed = self._diffusion_panel.seed
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
            self._diffusion_panel.set_seed(seed)

        dl = DiffusionLayer(
            name=self._layer_stack.next_name("Diffusion"),
            width=self._layer_stack.width,
            height=self._layer_stack.height,
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
            tile_size=self._layer_stack.tile_size,
        )
        self._document.execute(InsertLayerCommand(
            layer=dl,
            label="Create Diffusion Layer",
        ))

    def _on_clear_mask(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer, DiffusionLayer):
            self._document.execute(ClearLayerMaskCommand(
                layer=layer,
                label="Clear Diffusion Mask",
            ))

    def _sync_panel_to_layer(self, layer: DiffusionLayer):
        layer.prompt = self._diffusion_panel.prompt
        layer.negative_prompt = self._diffusion_panel.negative_prompt
        layer.strength = self._diffusion_panel.strength
        layer.guidance_scale = self._diffusion_panel.guidance_scale
        layer.steps = self._diffusion_panel.steps
        layer.seed = self._diffusion_panel.seed
        layer.mode = self._diffusion_panel.mode
        layer.masked_content = self._diffusion_panel.masked_content
        layer.ip_adapter_scale = self._diffusion_panel.ip_adapter_scale
        layer.resize_to_model_resolution = self._diffusion_panel.resize_to_model_resolution
        if self._engine.model_path:
            layer.model_path = self._engine.model_path
        layer.prediction_type = self._diffusion_panel.prediction_type

    def _on_regenerate(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer, DiffusionLayer):
            return
        if self._engine.is_busy:
            return
        self._sync_panel_to_layer(layer)

        if layer.mode != "txt2img":
            composite = self._canvas.get_composite_below(layer)
            if composite is None:
                return
            if layer.manual_patch_rect is not None:
                x0, y0, x1, y1 = layer.manual_patch_rect
                h, w = composite.shape[:2]
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(w, x1), min(h, y1)
                if x1 - x0 < 1 or y1 - y0 < 1:
                    return
                patch_pil = Image.fromarray(composite[y0:y1, x0:x1]).convert("RGB")
                layer.source_patch = patch_pil
                layer.patch_x, layer.patch_y = x0, y0
                layer.patch_w, layer.patch_h = x1 - x0, y1 - y0
            elif layer.has_mask():
                bbox = layer.mask_bbox()
                center = layer.mask_center()
                if bbox is not None and center is not None:
                    bx0, by0, bx1, by1 = bbox
                    ps = max(bx1 - bx0, by1 - by0)
                    ps = max(int(ps * 1.25), 512)
                    cx, cy = center
                    patch_pil, ppx, ppy, pw, ph = extract_patch(
                        composite, cx, cy, patch_size=ps)
                    layer.source_patch = patch_pil
                    layer.patch_x, layer.patch_y = ppx, ppy
                    layer.patch_w, layer.patch_h = pw, ph

            if layer.source_patch is None:
                return

        if layer.model_path and layer.model_path != self._engine.model_path:
            self._pending_request = layer
            pred = layer.prediction_type if layer.prediction_type else None
            self._engine.submit_load(layer.model_path, pred)
            self._statusbar.text = "Loading model for regeneration..."
            return

        if not self._engine.is_loaded:
            return
        self._submit_regenerate(layer)

    def _submit_regenerate(self, layer: DiffusionLayer):
        self._pending_request = layer
        mask_image = None
        if layer.mode == "inpaint":
            if not layer.has_mask():
                self._statusbar.text = "Inpaint requires a mask"
                return
            mask_image = extract_mask_patch(
                layer.mask, layer.patch_x, layer.patch_y,
                layer.patch_w, layer.patch_h)

        ip_adapter_image = None
        if layer.ip_adapter_rect is not None:
            if not self._engine.ip_adapter_loaded:
                self._pending_request = layer
                self._engine.submit_load_ip_adapter()
                self._statusbar.text = "Loading IP-Adapter..."
                return
            composite = self._canvas.get_composite_below(layer)
            if composite is not None:
                x0, y0, x1, y1 = layer.ip_adapter_rect
                h, w = composite.shape[:2]
                x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
                y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
                if x1 > x0 and y1 > y0:
                    crop = composite[y0:y1, x0:x1, :3]
                    ip_adapter_image = Image.fromarray(crop, "RGB")

        submit_image = layer.source_patch
        submit_mask = mask_image
        submit_w, submit_h = layer.patch_w, layer.patch_h
        MODEL_RES = 1024
        if layer.resize_to_model_resolution and submit_image is not None:
            longest = max(submit_w, submit_h)
            if longest != MODEL_RES:
                scale = MODEL_RES / longest
                submit_w = max(8, round(layer.patch_w * scale / 8) * 8)
                submit_h = max(8, round(layer.patch_h * scale / 8) * 8)
                submit_image = submit_image.resize((submit_w, submit_h), Image.LANCZOS)
                if submit_mask is not None:
                    submit_mask = submit_mask.resize((submit_w, submit_h), Image.NEAREST)

        self._engine.submit(
            image=submit_image,
            prompt=layer.prompt,
            negative_prompt=layer.negative_prompt,
            strength=layer.strength,
            steps=layer.steps,
            guidance_scale=layer.guidance_scale,
            seed=layer.seed,
            mode=layer.mode,
            mask_image=submit_mask,
            masked_content=layer.masked_content,
            ip_adapter_image=ip_adapter_image,
            ip_adapter_scale=layer.ip_adapter_scale,
            width=submit_w,
            height=submit_h,
        )
        self._statusbar.text = f"Regenerating ({submit_w}x{submit_h})..."

    def _on_new_seed(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer, DiffusionLayer):
            return
        new_seed = random.randint(0, 2**32 - 1)
        layer.seed = new_seed
        self._diffusion_panel.set_seed(new_seed)
        self._on_regenerate()

    def _on_load_ip_adapter(self):
        if self._engine.is_busy or not self._engine.is_loaded:
            return
        self._engine.submit_load_ip_adapter()
        self._statusbar.text = "Loading IP-Adapter..."

    def _on_ref_rect_drawn(self, x0, y0, x1, y1):
        layer = self._layer_stack.active_layer
        if isinstance(layer, DiffusionLayer):
            self._document.execute(SetDiffusionIpAdapterRectCommand(
                layer=layer,
                rect=(x0, y0, x1, y1),
            ))

    def _on_clear_ref_rect(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer, DiffusionLayer):
            self._document.execute(ClearDiffusionIpAdapterRectCommand(layer=layer))

    def _on_patch_rect_drawn(self, x0, y0, x1, y1):
        layer = self._layer_stack.active_layer
        if isinstance(layer, DiffusionLayer):
            self._diffusion_panel.set_draw_patch_checked(False)
            self._document.execute(SetManualPatchRectCommand(
                layer=layer,
                rect=(x0, y0, x1, y1),
                label="Set Diffusion Patch Rect",
            ))
        elif isinstance(layer, InstructLayer):
            self._instruct_panel.set_draw_patch_checked(False)
            self._document.execute(SetManualPatchRectCommand(
                layer=layer,
                rect=(x0, y0, x1, y1),
                label="Set Instruct Patch Rect",
            ))

    def _on_clear_patch_rect(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer, DiffusionLayer):
            self._document.execute(ClearManualPatchRectCommand(
                layer=layer,
                label="Clear Diffusion Patch Rect",
            ))

    # ------------------------------------------------------------------
    # LaMa
    # ------------------------------------------------------------------

    def _on_create_lama(self):
        composite = self._canvas.get_composite()
        if composite is None:
            return
        cx, cy = self._canvas.view_center_image()
        patch_pil, ppx, ppy, pw, ph = extract_patch(composite, cx, cy)
        ll = LamaLayer(
            name=self._layer_stack.next_name("LaMa"),
            width=self._layer_stack.width,
            height=self._layer_stack.height,
            source_patch=patch_pil,
            patch_x=ppx, patch_y=ppy, patch_w=pw, patch_h=ph,
            tile_size=self._layer_stack.tile_size,
        )
        self._document.execute(InsertLayerCommand(
            layer=ll,
            label="Create LaMa Layer",
        ))

    def _on_lama_remove(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer, LamaLayer):
            return
        if self._lama_engine.is_busy or not layer.has_mask():
            return

        bbox = layer.mask_bbox()
        center = layer.mask_center()
        if bbox is None or center is None:
            return
        composite = self._canvas.get_composite_below(layer)
        if composite is None:
            return

        bx0, by0, bx1, by1 = bbox
        ps = max(bx1 - bx0, by1 - by0)
        ps = max(int(ps * 1.25), 512)
        cx, cy = center
        patch_pil, ppx, ppy, pw, ph = extract_patch(composite, cx, cy, patch_size=ps)
        layer.source_patch = patch_pil
        layer.patch_x, layer.patch_y = ppx, ppy
        layer.patch_w, layer.patch_h = pw, ph

        mask_pil = extract_mask_patch(layer.mask, ppx, ppy, pw, ph)
        self._lama_engine.submit(patch_pil, mask_pil)
        self._pending_lama_layer = layer
        self._statusbar.text = "Removing objects (LaMa)..."

    def _on_lama_clear_mask(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer, LamaLayer):
            self._document.execute(ClearLayerMaskCommand(
                layer=layer,
                label="Clear LaMa Mask",
            ))

    def _on_lama_select_background(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer, LamaLayer):
            return
        if self._seg_engine.is_busy:
            return
        composite = self._canvas.get_composite_below(layer)
        if composite is None:
            return
        self._seg_engine.submit(composite, invert=True)
        self._statusbar.text = "Segmenting background..."

    # ------------------------------------------------------------------
    # InstructPix2Pix
    # ------------------------------------------------------------------

    def _on_create_instruct(self):
        composite = self._canvas.get_composite()
        if composite is None:
            return
        cx, cy = self._canvas.view_center_image()
        patch_pil, ppx, ppy, pw, ph = extract_patch(composite, cx, cy)
        seed = self._instruct_panel.seed
        if seed < 0:
            seed = random.randint(0, 2**32 - 1)
            self._instruct_panel.set_seed(seed)
        il = InstructLayer(
            name=self._layer_stack.next_name("Instruct"),
            width=self._layer_stack.width,
            height=self._layer_stack.height,
            source_patch=patch_pil,
            patch_x=ppx, patch_y=ppy, patch_w=pw, patch_h=ph,
            instruction=self._instruct_panel.instruction,
            image_guidance_scale=self._instruct_panel.image_guidance_scale,
            guidance_scale=self._instruct_panel.guidance_scale,
            steps=self._instruct_panel.steps,
            seed=seed,
            tile_size=self._layer_stack.tile_size,
        )
        self._document.execute(InsertLayerCommand(
            layer=il,
            label="Create Instruct Layer",
        ))

    def _on_instruct_load_model(self):
        if self._instruct_engine.is_busy:
            return
        self._instruct_panel.set_model_loading()
        self._instruct_engine.submit_load()
        self._statusbar.text = "Loading InstructPix2Pix model..."

    def _on_instruct_apply(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer, InstructLayer):
            return
        if self._instruct_engine.is_busy:
            return

        layer.instruction = self._instruct_panel.instruction
        layer.image_guidance_scale = self._instruct_panel.image_guidance_scale
        layer.guidance_scale = self._instruct_panel.guidance_scale
        layer.steps = self._instruct_panel.steps
        layer.seed = self._instruct_panel.seed

        if not self._instruct_engine.is_loaded:
            self._pending_instruct_layer = layer
            self._on_instruct_load_model()
            return

        composite = self._canvas.get_composite_below(layer)
        if composite is None:
            return

        if layer.manual_patch_rect is not None:
            x0, y0, x1, y1 = layer.manual_patch_rect
            h, w = composite.shape[:2]
            x0, y0 = max(0, x0), max(0, y0)
            x1, y1 = min(w, x1), min(h, y1)
            if x1 - x0 < 1 or y1 - y0 < 1:
                return
            patch_pil = Image.fromarray(composite[y0:y1, x0:x1]).convert("RGB")
            layer.source_patch = patch_pil
            layer.patch_x, layer.patch_y = x0, y0
            layer.patch_w, layer.patch_h = x1 - x0, y1 - y0
        elif layer.has_mask():
            bbox = layer.mask_bbox()
            center = layer.mask_center()
            if bbox is not None and center is not None:
                bx0, by0, bx1, by1 = bbox
                ps = max(bx1 - bx0, by1 - by0)
                ps = max(int(ps * 1.25), 512)
                cx, cy = center
                patch_pil, ppx, ppy, pw, ph = extract_patch(
                    composite, cx, cy, patch_size=ps)
                layer.source_patch = patch_pil
                layer.patch_x, layer.patch_y = ppx, ppy
                layer.patch_w, layer.patch_h = pw, ph
        else:
            cx = layer.patch_x + layer.patch_w // 2
            cy = layer.patch_y + layer.patch_h // 2
            patch_pil, ppx, ppy, pw, ph = extract_patch(
                composite, cx, cy, patch_size=max(layer.patch_w, layer.patch_h))
            layer.source_patch = patch_pil
            layer.patch_x, layer.patch_y = ppx, ppy
            layer.patch_w, layer.patch_h = pw, ph

        self._instruct_engine.submit(
            image=layer.source_patch,
            instruction=layer.instruction,
            guidance_scale=layer.guidance_scale,
            image_guidance_scale=layer.image_guidance_scale,
            steps=layer.steps,
            seed=layer.seed,
        )
        self._pending_instruct_layer = layer
        self._statusbar.text = "Applying instruction..."

    def _on_instruct_new_seed(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer, InstructLayer):
            return
        new_seed = random.randint(0, 2**32 - 1)
        layer.seed = new_seed
        self._instruct_panel.set_seed(new_seed)
        self._on_instruct_apply()

    def _on_instruct_clear_mask(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer, InstructLayer):
            self._document.execute(ClearLayerMaskCommand(
                layer=layer,
                label="Clear Instruct Mask",
            ))

    def _on_instruct_clear_patch_rect(self):
        layer = self._layer_stack.active_layer
        if isinstance(layer, InstructLayer):
            self._document.execute(ClearManualPatchRectCommand(
                layer=layer,
                label="Clear Instruct Patch Rect",
            ))

    # ------------------------------------------------------------------
    # Segmentation
    # ------------------------------------------------------------------

    def _on_select_background(self):
        layer = self._layer_stack.active_layer
        if not isinstance(layer, DiffusionLayer):
            return
        if self._seg_engine.is_busy:
            return
        composite = self._canvas.get_composite_below(layer)
        if composite is None:
            return
        self._seg_engine.submit(composite, invert=True)
        self._statusbar.text = "Segmenting background..."

    # ------------------------------------------------------------------
    # Polling (called every frame from SDL main loop)
    # ------------------------------------------------------------------

    def poll(self):
        self._poll_segmentation()
        self._poll_lama()
        self._poll_instruct()
        self._poll_diffusion()

    def _poll_segmentation(self):
        seg_mask, seg_error = self._seg_engine.poll()
        if seg_mask is not None:
            layer = self._layer_stack.active_layer
            command, status = map_segmentation_result(layer, seg_mask)
            if command is not None:
                self._document.execute(command)
            self._statusbar.text = status
        elif seg_error is not None:
            log.error(f"Segmentation error: {seg_error}")
            self._statusbar.text = f"Segmentation error: {seg_error[:80]}"

    def _poll_lama(self):
        result_image, lama_error = self._lama_engine.poll()
        if result_image is not None:
            layer = self._pending_lama_layer
            command, status = map_lama_result(layer, result_image)
            if command is not None:
                self._document.execute(command)
            self._statusbar.text = status
            self._pending_lama_layer = None
        elif lama_error is not None:
            log.error(f"LaMa error: {lama_error}")
            self._statusbar.text = f"LaMa error: {lama_error[:80]}"
            self._pending_lama_layer = None

    def _poll_instruct(self):
        task_type, result, error, meta = self._instruct_engine.poll()
        if task_type is None:
            return
        if task_type == "load":
            if error:
                log.error(f"InstructPix2Pix load error: {error}")
                self._instruct_panel.on_model_load_error(error)
                self._statusbar.text = f"InstructPix2Pix load error: {error[:80]}"
                self._pending_instruct_layer = None
            else:
                self._instruct_panel.on_model_loaded()
                self._statusbar.text = "InstructPix2Pix model loaded"
                if isinstance(self._pending_instruct_layer, InstructLayer):
                    self._on_instruct_apply()
        elif task_type == "inference":
            if error:
                log.error(f"InstructPix2Pix inference error: {error}")
                self._statusbar.text = f"InstructPix2Pix error: {error[:80]}"
                self._pending_instruct_layer = None
                return
            result_image, used_seed = result
            layer = self._pending_instruct_layer
            command, status = map_instruct_result(layer, result_image, used_seed)
            if command is not None:
                self._document.execute(command)
            self._statusbar.text = status
            self._pending_instruct_layer = None

    def _poll_diffusion(self):
        task_type, result, error, meta = self._engine.poll()
        if task_type is None:
            return

        log.debug(
            f"[_poll_diffusion] task_type={task_type}, error={error}, result_type={type(result)}"
        )

        if task_type == "load":
            if error:
                log.error(f"Diffusion model load error: {error}")
                self._diffusion_panel.on_model_load_error(error)
                self._statusbar.text = f"Model load error: {error[:80]}"
                self._pending_request = None
            else:
                self._diffusion_panel.on_model_loaded(result, self._engine.model_info)
                if isinstance(self._pending_request, DiffusionLayer):
                    self._submit_regenerate(self._pending_request)
                    return
                self._statusbar.text = "Model loaded"

        elif task_type == "load_ip_adapter":
            if error:
                log.error(f"IP-Adapter load error: {error}")
                self._diffusion_panel.on_ip_adapter_load_error(error)
                self._statusbar.text = f"IP-Adapter error: {error[:80]}"
                self._pending_request = None
            else:
                self._diffusion_panel.on_ip_adapter_loaded()
                if isinstance(self._pending_request, DiffusionLayer):
                    self._submit_regenerate(self._pending_request)
                    return
                self._statusbar.text = "IP-Adapter loaded"

        elif task_type == "inference":
            if error:
                log.error(f"Diffusion inference error: {error}")
                self._statusbar.text = f"Diffusion error: {error[:80]}"
                self._pending_request = None
                return

            result_image, used_seed = result
            log.debug(
                f"[_poll_diffusion] inference OK, seed={used_seed}, pending={type(self._pending_request).__name__}"
            )
            command, status = map_diffusion_result(
                self._pending_request, result_image, used_seed)
            if command is not None:
                self._document.execute(command)
            self._statusbar.text = status
            self._pending_request = None

    # ------------------------------------------------------------------
    # Public: rendering
    # ------------------------------------------------------------------

    def render(self, vw: int, vh: int):
        self.ui.render(vw, vh)

    @property
    def running(self) -> bool:
        return self._running

    def close(self):
        """Release runtime resources (GPU/engines). Safe to call multiple times."""
        if self._closed:
            return
        self._closed = True
        self._running = False
        if hasattr(self, "_canvas") and self._canvas is not None:
            self._canvas.dispose()
        self._engine.shutdown()
        self._instruct_engine.shutdown()
        self._lama_engine.shutdown()
        self._seg_engine.shutdown()
