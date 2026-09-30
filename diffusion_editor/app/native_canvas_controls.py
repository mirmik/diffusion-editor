"""Unified tool palette and contextual options for the native editor."""

from __future__ import annotations

from termin.gui_native import (
    CommandData, CommandModel, EdgeInsets, KeyCode, Rect, Size, SrgbColor,
    OverlayFlag, OverlayGeometry, OverlayPlacement, StyleField, TextWrapMode,
)

from ..canvas.brush import BrushToolMode
from .native_tool_icons import ToolGlyph
from .canvas_controls import (
    BrushControlAction, BrushControlsIntent, BrushControlsState,
    SelectionControlAction, SelectionControlsIntent, SelectionControlsState,
)


_TOOL_GROUPS = (
    ("Paint", (("paint", "Brush", "B"), ("eraser", "Eraser", "E"),
               ("smudge", "Smudge", "S"))),
    ("Selection", (("select_rect", "Rectangle", "M"),
                   ("select_brush", "Selection brush", "Q"))),
    ("Geometry", (("move", "Move layer", "V"),
                  ("transform", "Transform", "T"))),
)
_AI_TOOLS = (("mask", "Paint mask", ""), ("mask_eraser", "Erase mask", ""),
             ("patch", "Processing area", ""))
_LABELS = {key: label for _, items in _TOOL_GROUPS for key, label, _ in items}
_LABELS.update({key: label for key, label, _ in _AI_TOOLS})


class _Options:
    def __init__(self, document, name, on_intent, intent_type, *, horizontal=False):
        self._document = document
        self._on_intent = on_intent
        self._intent_type = intent_type
        self._syncing = False
        self._closed = False
        self._connections = []
        self.widget = document.create_vstack(name)
        self.widget.set_layout_spacing(4)
        self.row = (document.create_hstack if horizontal else document.create_vstack)(f"{name}Fields")
        self.row.set_layout_spacing(8)
        self.widget.add_preferred_child(self.row)

    def _emit(self, action, value=None):
        if not self._syncing and not self._closed:
            self._on_intent(self._intent_type(action, value))

    def _field(self, label, stable_id, value, minimum, maximum, decimals, action):
        cell = self._document.create_hstack("ToolParameter")
        cell.set_layout_spacing(4)
        cell.add_fixed_child(self._document.create_label(label), 68)
        control = self._document.create_spin_box(value)
        control.widget.stable_id = stable_id
        control.widget.min_size = Size(45, 24)
        control.set_range(minimum, maximum)
        control.step = 1 if decimals == 0 else .01
        control.decimals = decimals
        self._connections.append(control.connect_changed(lambda v: self._emit(action, v)))
        cell.add_flex_child(control.widget, 1)
        self.row.add_preferred_child(cell)
        return control

    def _checkbox(self, parent, label, stable_id, action):
        row = self._document.create_hstack("ToolOption")
        row.set_layout_spacing(3)
        control = self._document.create_checkbox(False)
        control.widget.stable_id = stable_id
        row.add_fixed_child(control.widget, 22)
        row.add_preferred_child(self._document.create_label(label))
        self._connections.append(control.connect_changed(lambda v: self._emit(action, v)))
        parent.add_preferred_child(row)
        return control

    def close(self):
        self._closed = True
        self._connections.clear()
        self._on_intent = lambda _intent: None


class NativeBrushPanel(_Options):
    def __init__(self, document, state, on_intent, viewport_rect):
        super().__init__(document, "BrushOptions", on_intent, BrushControlsIntent)
        self.widget.stable_id = "diffusion-editor.brush-panel"
        self._viewport_rect = viewport_rect
        self._state = state
        self.size = self._field("Size", "diffusion-editor.brush.size", state.size,
                                1, 500, 0, BrushControlAction.SIZE)
        self.hardness = self._field("Hardness", "diffusion-editor.brush.hardness", state.hardness,
                                    0, 1, 2, BrushControlAction.HARDNESS)
        self.flow = self._field("Flow", "diffusion-editor.brush.flow", state.flow,
                                0, 1, 2, BrushControlAction.FLOW)
        self.color_button = document.create_button("")
        self.color_button.widget.stable_id = "diffusion-editor.brush.color"
        self.widget.add_preferred_child(self.color_button.widget)
        self.color_hint = document.create_label("Ctrl + click to sample color")
        self.color_hint.set_wrap_mode(TextWrapMode.Word)
        self.widget.add_preferred_child(self.color_hint)
        self._connections.append(self.color_button.connect_clicked(self._show_color_dialog))
        self.color_dialog = document.create_color_dialog(
            SrgbColor(*(c / 255 for c in state.color)), show_alpha=True, title="Brush Color")
        self.color_dialog.widget.stable_id = "diffusion-editor.brush.color-dialog"
        self._connections.append(self.color_dialog.connect_color_finished(self._on_color_finished))
        self.apply_state(state)

    def apply_state(self, state):
        self._state = state
        self._syncing = True
        try:
            self.size.value = state.size
            self.hardness.value = state.hardness
            self.flow.value = state.flow
            r, g, b, a = state.color
            self.color_button.set_text(
                f"Color  #{r:02X}{g:02X}{b:02X}  A:{a}")
            self.color_button.widget.visible = state.tool == BrushToolMode.PAINT
            self.color_hint.visible = self.color_button.widget.visible
            self.widget.enabled = state.accepts_pixel_edits
        finally:
            self._syncing = False

    def _show_color_dialog(self):
        if self._closed or self.color_dialog.open:
            return
        self.color_dialog.color = SrgbColor(*(c / 255 for c in self._state.color))
        viewport = self._viewport_rect()
        if viewport.width <= 0 or viewport.height <= 0:
            viewport = Rect(0, 0, 640, 480)
        self.color_dialog.show(viewport)

    def _on_color_finished(self, color):
        if color is not None:
            self._emit(BrushControlAction.COLOR, tuple(
                max(0, min(round(c * 255), 255)) for c in (color.r, color.g, color.b, color.a)))


class NativeSelectionPanel(_Options):
    def __init__(self, document, state, on_intent):
        super().__init__(document, "SelectionOptions", on_intent, SelectionControlsIntent)
        self.widget.stable_id = "diffusion-editor.selection-panel"
        self.size = self._field("Size", "diffusion-editor.selection.size", state.size,
                                1, 500, 0, SelectionControlAction.SIZE)
        self.hardness = self._field("Hardness", "diffusion-editor.selection.hardness", state.hardness,
                                    0, 1, 2, SelectionControlAction.HARDNESS)
        self.flow = self._field("Flow", "diffusion-editor.selection.flow", state.flow,
                                0, 1, 2, SelectionControlAction.FLOW)
        self.flags = document.create_vstack("SelectionFlags")
        self.flags.set_layout_spacing(4)
        self.widget.add_preferred_child(self.flags)
        self.eraser = self._checkbox(
            self.flags, "Subtract", "diffusion-editor.selection.eraser",
            SelectionControlAction.ERASER)
        self.show = self._checkbox(
            self.flags, "Show selection", "diffusion-editor.selection.show",
            SelectionControlAction.SHOW)
        self.apply_state(state)

    def apply_state(self, state):
        self._syncing = True
        try:
            self.eraser.checked, self.show.checked = state.eraser, state.show
            self.size.value = state.size
            self.hardness.value = state.hardness
            self.flow.value = state.flow
            self.row.visible = state.edit_mode
            self.eraser.widget.parent.visible = state.edit_mode
        finally:
            self._syncing = False


class NativeCanvasControls:
    def __init__(self, document, brush_state, selection_state, on_brush_intent,
                 on_selection_intent, viewport_rect, activate_command=lambda _id: False):
        self._document = document
        self._viewport_rect = viewport_rect
        self._on_brush_intent = on_brush_intent
        self._on_selection_intent = on_selection_intent
        self._connections = []
        self._closed = False
        self._state = brush_state
        self._glyphs = []
        self.widget = document.create_vstack("NativeToolPalette")
        self.widget.stable_id = "diffusion-editor.canvas-controls"
        self.widget.set_layout_spacing(0)
        self.widget.set_layout_padding(EdgeInsets(4, 4, 4, 4))
        self._tooltip_key = None
        self.tooltip = document.create_vstack("ToolTooltip")
        self.tooltip.stable_id = "diffusion-editor.tool-tooltip"
        self.tooltip.set_layout_padding(EdgeInsets(8, 6, 8, 6))
        self.tooltip_text = document.create_label("")
        self.tooltip.add_preferred_child(self.tooltip_text)
        override = self.tooltip.style_override
        style = override.value
        style.background = SrgbColor(.10, .11, .13, 1)
        override.value = style
        override.fields = StyleField.Background.value
        self.tooltip.style_override = override
        self.tool_commands = {}
        self.toolbars = {}
        for title, items in (*_TOOL_GROUPS, ("AI region", _AI_TOOLS)):
            if title != "Paint":
                self.widget.add_fixed_child(document.create_separator(True), 8)
            for key, label, shortcut in items:
                self._tool(document, self.widget, key, label, shortcut)
        self.options_widget = document.create_vstack("ActiveToolOptions")
        self.options_widget.stable_id = "diffusion-editor.tool-options"
        self.options_widget.set_layout_spacing(4)
        self.options_widget.set_layout_padding(EdgeInsets(6, 4, 6, 6))
        self.caption = document.create_label("")
        self.caption.stable_id = "diffusion-editor.active-tool"
        self.options_widget.add_preferred_child(self.caption)
        self.brush = NativeBrushPanel(document, brush_state, on_brush_intent, viewport_rect)
        self.selection = NativeSelectionPanel(document, selection_state, on_selection_intent)
        self.options_widget.add_preferred_child(self.brush.widget)
        self.options_widget.add_preferred_child(self.selection.widget)
        self.selection_actions = document.create_vstack("SelectionActions")
        self.selection_actions.set_layout_spacing(4)
        for key, label in (("all", "Select all"), ("clear", "Deselect"), ("invert", "Invert")):
            button = document.create_button(label)
            button.widget.stable_id = f"diffusion-editor.selection.{key}"
            self._connections.append(button.connect_clicked(
                lambda key=key: activate_command(f"selection.{key}")))
            self.selection_actions.add_preferred_child(button.widget)
        self.options_widget.add_preferred_child(self.selection_actions)
        self.hint = document.create_label("")
        self.hint.set_wrap_mode(TextWrapMode.Word)
        self.options_widget.add_preferred_child(self.hint)
        # AI tools share the same selection as the main palette.
        self.ai_widget = document.create_vstack("AIRegionTools")
        self.ai_widget.stable_id = "diffusion-editor.ai-region-tools"
        self.ai_widget.set_layout_spacing(3)
        self.ai_widget.add_preferred_child(document.create_label("Processing region"))
        patch = _Options(document, "PatchOptions", on_brush_intent, BrushControlsIntent, horizontal=True)
        self.patch = patch
        self.show_patch = patch._checkbox(
            patch.row, "Show area", "diffusion-editor.brush.show-patch",
            BrushControlAction.SHOW_PATCH)
        clear = document.create_button("Clear area")
        clear.widget.stable_id = "diffusion-editor.brush.clear-patch"
        patch.row.add_preferred_child(clear.widget)
        self._connections.append(clear.connect_clicked(lambda: on_brush_intent(
            BrushControlsIntent(BrushControlAction.CLEAR_PATCH))))
        self.ai_widget.add_preferred_child(patch.widget)
        self.apply_brush_state(brush_state)
        self.apply_selection_state(selection_state)

    def _tool(self, document, parent, key, label, shortcut):
        model = CommandModel()
        command_id = model.append(CommandData(
            f"diffusion-editor.tool.{key}", "",
            tooltip=label + (f" ({shortcut})" if shortcut else ""), checkable=True))
        toolbar = document.create_tool_bar(model)
        toolbar.widget.stable_id = f"diffusion-editor.tool.{key}"
        toolbar.item_height, toolbar.padding = 36, 4
        self._connections.append(toolbar.connect_activated(lambda *_: self.activate_tool(key)))
        self.tool_commands[key] = model, command_id
        self.toolbars[key] = toolbar
        # The command toolbar owns interaction and checked state; the glyph
        # overlay is pointer-transparent and draws only native vector geometry.
        overlay = document.create_overlay_layout()
        overlay.add_child(toolbar.widget)
        glyph = ToolGlyph(key, model, command_id)
        glyph_handle = document.adopt(glyph, f"ToolGlyph.{key}")
        glyph.mouse_transparent = True
        overlay.add_child(document.ref(glyph_handle))
        self._glyphs.append(glyph)
        parent.add_fixed_child(overlay.widget, 44)

    def update_tooltip(self):
        if self._closed:
            return
        hovered = self._document.hovered_widget
        key = next((key for key, toolbar in self.toolbars.items()
                    if toolbar.handle == hovered and toolbar.hovered_tooltip), None)
        if self._document.pointer_capture.valid:
            key = None
        if key == self._tooltip_key:
            return
        if self._tooltip_key is not None:
            self._document.dismiss_overlay(self.tooltip.handle)
        self._tooltip_key = key
        if key is not None:
            toolbar = self.toolbars[key]
            self.tooltip_text.text = toolbar.hovered_tooltip
            geometry = OverlayGeometry()
            geometry.placement = OverlayPlacement.AnchorRight
            geometry.anchor = toolbar.handle
            geometry.margin = 6
            self._document.show_overlay_placed(
                self.tooltip.handle,
                int(OverlayFlag.Tooltip) | int(OverlayFlag.PointerTransparent),
                geometry, self._viewport_rect())

    def activate_tool(self, key):
        model, command_id = self.tool_commands[key]
        if self._closed or not model.command(command_id).data.enabled:
            return False
        self._on_brush_intent(BrushControlsIntent(BrushControlAction.ACTIVE_TOOL, key))
        self.apply_brush_state(self._state)
        return True

    def dispatch_shortcut(self, key, modifiers):
        if modifiers or self._closed:
            return False
        key = getattr(key, "value", key)
        if key in (ord("["), ord("]")):
            step = -5 if key == ord("[") else 5
            active = self._state.active_tool or self._state.tool.value
            if active == "select_brush":
                self._on_selection_intent(SelectionControlsIntent(
                    SelectionControlAction.SIZE, self.selection.size.value + step))
            elif active in ("paint", "eraser", "smudge", "mask", "mask_eraser"):
                self._on_brush_intent(BrushControlsIntent(
                    BrushControlAction.SIZE, self._state.size + step))
            return True
        shortcuts = {getattr(KeyCode, shortcut).value: tool
                     for _, items in _TOOL_GROUPS for tool, _, shortcut in items}
        tool = shortcuts.get(getattr(key, "value", key))
        return self.activate_tool(tool) if tool else False

    def apply_brush_state(self, state: BrushControlsState):
        self._state = state
        self.brush.apply_state(state)
        active = state.active_tool or state.tool.value
        transform = active == "transform"
        for tool, (model, command_id) in self.tool_commands.items():
            enabled = (state.can_move if tool == "move" else state.accepts_pixel_edits)
            model.set_enabled(command_id, enabled and not transform)
            model.set_checked(command_id, tool == active)
        self.caption.text = _LABELS.get(active, active)
        self.options_widget.visible = not transform
        self.brush.widget.enabled = state.accepts_pixel_edits and not transform
        self.selection.widget.enabled = not transform
        self.brush.widget.visible = active in ("paint", "eraser", "smudge", "mask", "mask_eraser")
        self.selection.widget.visible = active in ("select_rect", "select_brush")
        self.selection_actions.visible = active in ("select_rect", "select_brush")
        self.hint.visible = active in ("move", "patch", "select_rect")
        self.hint.text = {
            "move": "Drag to move the entire layer",
            "patch": "Drag a rectangle to set the AI processing area",
            "select_rect": "Drag a rectangle to replace the selection",
        }.get(active, "")
        self.patch._syncing = True
        try:
            self.show_patch.checked = state.show_patch
        finally:
            self.patch._syncing = False
        self.patch.widget.enabled = state.accepts_pixel_edits and not transform

    def apply_selection_state(self, state: SelectionControlsState):
        self.selection.apply_state(state)

    def close(self):
        self._closed = True
        if self._tooltip_key is not None:
            self._document.dismiss_overlay(self.tooltip.handle)
        self.brush.close()
        self.selection.close()
        self.patch.close()
        self._connections.clear()
