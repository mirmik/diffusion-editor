"""termin-gui-native projections for brush and selection controls."""

from __future__ import annotations

from typing import Callable

from termin.gui_native import (
    Color,
    CommandData,
    CommandModel,
    EdgeInsets,
    Rect,
    TcDocument,
)

from ..canvas.brush import BrushToolMode
from .canvas_controls import (
    BrushControlAction,
    BrushControlsIntent,
    BrushControlsState,
    SelectionControlAction,
    SelectionControlsIntent,
    SelectionControlsState,
)


_TOOL_LABELS = (
    (BrushToolMode.PAINT, "Paint"),
    (BrushToolMode.ERASER, "Erase"),
    (BrushToolMode.SMUDGE, "Smudge"),
    (BrushToolMode.MASK, "Mask"),
    (BrushToolMode.MASK_ERASER, "Unmask"),
    (BrushToolMode.MOVE, "Move"),
)


class NativeBrushPanel:
    def __init__(
            self,
            document: TcDocument,
            state: BrushControlsState,
            on_intent: Callable[[BrushControlsIntent], None],
            viewport_rect: Callable[[], Rect]) -> None:
        self._closed = False
        self._syncing = False
        self._state = state
        self._on_intent = on_intent
        self._viewport_rect = viewport_rect
        self._connections: list[object] = []

        self.group = document.create_group_box(
            "Brush", "DiffusionEditorNativeBrushPanel")
        self.group.widget.stable_id = "diffusion-editor.brush-panel"
        self.group.set_padding(EdgeInsets(6.0, 6.0, 6.0, 6.0))
        self.widget = self.group.widget
        content = document.create_vstack("NativeBrushPanelContent")
        content.set_layout_spacing(4.0)
        self.group.set_content(content)

        tools = document.create_vstack("NativeBrushToolModes")
        tools.set_layout_spacing(2.0)
        self.tool_commands = {}
        self.toolbars = []
        for row_index, row_modes in enumerate(
                (_TOOL_LABELS[:3], _TOOL_LABELS[3:])):
            model = CommandModel()
            for mode, label in row_modes:
                command_id = model.append(CommandData(
                    f"diffusion-editor.brush-tool.{mode.value}",
                    label,
                    tooltip=f"{label} brush mode",
                    checkable=True,
                ))
                self.tool_commands[mode] = (model, command_id)
            toolbar = document.create_tool_bar(model)
            toolbar.widget.stable_id = (
                f"diffusion-editor.brush-tools.{row_index}")
            toolbar.item_height = 26.0
            toolbar.padding = 2.0
            self._connections.append(toolbar.connect_activated(
                self._on_tool_activated))
            self.toolbars.append(toolbar)
            tools.add_preferred_child(toolbar.widget)
        content.add_preferred_child(tools)

        patch_row = document.create_hstack("NativeBrushPatchRow")
        patch_row.set_layout_spacing(4.0)
        self.draw_patch = document.create_checkbox(state.draw_patch)
        self.draw_patch.widget.stable_id = "diffusion-editor.brush.draw-patch"
        self.show_patch = document.create_checkbox(state.show_patch)
        self.show_patch.widget.stable_id = "diffusion-editor.brush.show-patch"
        self.clear_patch = document.create_button("Clear")
        self.clear_patch.widget.stable_id = "diffusion-editor.brush.clear-patch"
        self._connections.append(
            self.draw_patch.connect_changed(self._on_draw_patch_changed))
        self._connections.append(
            self.show_patch.connect_changed(self._on_show_patch_changed))
        self._connections.append(
            self.clear_patch.connect_clicked(self._on_clear_patch))
        self._add_checkbox_label(
            document, patch_row, self.draw_patch, "Draw", "draw-patch")
        self._add_checkbox_label(
            document, patch_row, self.show_patch, "Show", "show-patch")
        patch_row.add_preferred_child(self.clear_patch.widget)
        content.add_preferred_child(patch_row)

        self.color_button = document.create_button("")
        self.color_button.widget.stable_id = "diffusion-editor.brush.color"
        self._connections.append(
            self.color_button.connect_clicked(self._show_color_dialog))
        content.add_preferred_child(self.color_button.widget)

        initial = Color(*(channel / 255.0 for channel in state.color))
        self.color_dialog = document.create_color_dialog(
            initial,
            show_alpha=True,
            title="Brush Color",
        )
        self.color_dialog.widget.stable_id = (
            "diffusion-editor.brush.color-dialog")
        self._connections.append(
            self.color_dialog.connect_color_finished(
                self._on_color_finished))

        self.size = self._slider(
            document,
            content,
            "Size",
            "diffusion-editor.brush.size",
            state.size,
            1.0,
            500.0,
            1.0,
            0,
            lambda value: self._emit(BrushControlAction.SIZE, int(value)),
        )
        self.hardness = self._slider(
            document,
            content,
            "Hardness",
            "diffusion-editor.brush.hardness",
            state.hardness,
            0.0,
            1.0,
            0.01,
            2,
            lambda value: self._emit(BrushControlAction.HARDNESS, value),
        )
        self.flow = self._slider(
            document,
            content,
            "Flow",
            "diffusion-editor.brush.flow",
            state.flow,
            0.0,
            1.0,
            0.01,
            2,
            lambda value: self._emit(BrushControlAction.FLOW, value),
        )
        self.apply_state(state)

    def apply_state(self, state: BrushControlsState) -> None:
        if self._closed:
            return
        self._state = state
        self._syncing = True
        try:
            for mode, (model, command_id) in self.tool_commands.items():
                checked = mode == state.tool
                if model.command(command_id).data.checked != checked:
                    model.set_checked(command_id, checked)
            self.draw_patch.checked = state.draw_patch
            self.show_patch.checked = state.show_patch
            self.size.value = float(state.size)
            self.hardness.value = state.hardness
            self.flow.value = state.flow
            r, g, b, a = state.color
            self.color_button.set_text(
                f"Color  #{r:02X}{g:02X}{b:02X}  A:{a}")
            is_move = state.tool == BrushToolMode.MOVE
            self.group.title = "Move" if is_move else "Brush"
            self.color_button.widget.visible = not is_move
            self.size.widget.visible = not is_move
            self.hardness.widget.visible = not is_move
            self.flow.widget.visible = not is_move
        finally:
            self._syncing = False

    def close(self) -> None:
        self._closed = True
        self._on_intent = lambda _intent: None
        self._connections.clear()

    def _slider(
            self,
            document,
            parent,
            label,
            stable_id,
            value,
            minimum,
            maximum,
            step,
            decimals,
            callback):
        slider = document.create_slider_edit(float(value))
        slider.widget.stable_id = stable_id
        slider.label = label
        slider.set_range(minimum, maximum)
        slider.set_step(step)
        slider.set_decimals(decimals)
        self._connections.append(slider.connect_changed(
            lambda changed: None
            if self._syncing or self._closed
            else callback(changed)))
        parent.add_preferred_child(slider.widget)
        return slider

    @staticmethod
    def _add_checkbox_label(
            document, parent, checkbox, label, stable_suffix) -> None:
        cell = document.create_hstack("NativeBrushCheckboxCell")
        cell.set_layout_spacing(2.0)
        text = document.create_label(label, "NativeBrushCheckboxLabel")
        text.stable_id = (
            f"diffusion-editor.brush.{stable_suffix}.label")
        cell.add_preferred_child(checkbox.widget)
        cell.add_flex_child(text, 1.0)
        parent.add_flex_child(cell, 1.0)

    def _on_tool_activated(
            self, _index, _command_id, command) -> None:
        if self._syncing or self._closed:
            return
        mode = next(
            candidate
            for candidate in BrushToolMode
            if command.stable_id.endswith(f".{candidate.value}")
        )
        self._emit(BrushControlAction.TOOL, mode)
        # The toolbar implements generic independent checkable commands.
        # Re-project immediately so this application-level group remains
        # exactly-one-selected even if the active command was pressed again.
        self.apply_state(self._state)

    def _on_draw_patch_changed(self, checked: bool) -> None:
        self._emit(BrushControlAction.DRAW_PATCH, checked)

    def _on_show_patch_changed(self, checked: bool) -> None:
        self._emit(BrushControlAction.SHOW_PATCH, checked)

    def _on_clear_patch(self) -> None:
        self._emit(BrushControlAction.CLEAR_PATCH)

    def _show_color_dialog(self) -> None:
        if self._closed or self.color_dialog.open:
            return
        self.color_dialog.color = Color(
            *(channel / 255.0 for channel in self._state.color))
        viewport = self._viewport_rect()
        if viewport.width <= 0 or viewport.height <= 0:
            viewport = Rect(0.0, 0.0, 640.0, 480.0)
        self.color_dialog.show(viewport)

    def _on_color_finished(self, color: Color | None) -> None:
        if color is None or self._closed:
            return
        rgba = tuple(
            max(0, min(round(channel * 255.0), 255))
            for channel in (color.r, color.g, color.b, color.a)
        )
        self._emit(BrushControlAction.COLOR, rgba)

    def _emit(self, action: BrushControlAction, value=None) -> None:
        if not self._syncing and not self._closed:
            self._on_intent(BrushControlsIntent(action, value))


class NativeSelectionPanel:
    def __init__(
            self,
            document: TcDocument,
            state: SelectionControlsState,
            on_intent: Callable[[SelectionControlsIntent], None]) -> None:
        self._closed = False
        self._syncing = False
        self._on_intent = on_intent
        self._connections: list[object] = []

        self.group = document.create_group_box(
            "Selection", "DiffusionEditorNativeSelectionPanel")
        self.group.widget.stable_id = "diffusion-editor.selection-panel"
        self.group.set_padding(EdgeInsets(6.0, 6.0, 6.0, 6.0))
        self.widget = self.group.widget
        content = document.create_vstack("NativeSelectionPanelContent")
        content.set_layout_spacing(4.0)
        self.group.set_content(content)

        self.edit_mode = self._checkbox(
            document, content, "Edit selection",
            "diffusion-editor.selection.edit",
            lambda value: self._emit(SelectionControlAction.EDIT_MODE, value))
        self.rect_mode = self._checkbox(
            document, content, "Rectangle",
            "diffusion-editor.selection.rect",
            lambda value: self._emit(SelectionControlAction.RECT_MODE, value))
        self.eraser = self._checkbox(
            document, content, "Eraser",
            "diffusion-editor.selection.eraser",
            lambda value: self._emit(SelectionControlAction.ERASER, value))
        self.show = self._checkbox(
            document, content, "Show selection",
            "diffusion-editor.selection.show",
            lambda value: self._emit(SelectionControlAction.SHOW, value))
        self.size = self._slider(
            document, content, "Size", "diffusion-editor.selection.size",
            state.size, 1.0, 500.0, 1.0, 0,
            lambda value: self._emit(
                SelectionControlAction.SIZE, int(value)))
        self.hardness = self._slider(
            document, content, "Hardness",
            "diffusion-editor.selection.hardness",
            state.hardness, 0.0, 1.0, 0.01, 2,
            lambda value: self._emit(
                SelectionControlAction.HARDNESS, value))
        self.flow = self._slider(
            document, content, "Flow", "diffusion-editor.selection.flow",
            state.flow, 0.0, 1.0, 0.01, 2,
            lambda value: self._emit(SelectionControlAction.FLOW, value))
        self.apply_state(state)

    def apply_state(self, state: SelectionControlsState) -> None:
        if self._closed:
            return
        self._syncing = True
        try:
            self.edit_mode.checked = state.edit_mode
            self.rect_mode.checked = state.rect_mode
            self.eraser.checked = state.eraser
            self.show.checked = state.show
            self.size.value = float(state.size)
            self.hardness.value = state.hardness
            self.flow.value = state.flow
        finally:
            self._syncing = False

    def close(self) -> None:
        self._closed = True
        self._on_intent = lambda _intent: None
        self._connections.clear()

    def _checkbox(
            self, document, parent, label, stable_id, callback):
        checkbox = document.create_checkbox(False)
        checkbox.widget.stable_id = stable_id
        self._connections.append(checkbox.connect_changed(
            lambda checked: None
            if self._syncing or self._closed
            else callback(checked)))
        row = document.create_hstack("NativeSelectionCheckboxRow")
        row.set_layout_spacing(4.0)
        text = document.create_label(label, "NativeSelectionCheckboxLabel")
        text.stable_id = f"{stable_id}.label"
        row.add_preferred_child(checkbox.widget)
        row.add_flex_child(text, 1.0)
        parent.add_preferred_child(row)
        return checkbox

    def _slider(
            self,
            document,
            parent,
            label,
            stable_id,
            value,
            minimum,
            maximum,
            step,
            decimals,
            callback):
        slider = document.create_slider_edit(float(value))
        slider.widget.stable_id = stable_id
        slider.label = label
        slider.set_range(minimum, maximum)
        slider.set_step(step)
        slider.set_decimals(decimals)
        self._connections.append(slider.connect_changed(
            lambda changed: None
            if self._syncing or self._closed
            else callback(changed)))
        parent.add_preferred_child(slider.widget)
        return slider

    def _emit(self, action: SelectionControlAction, value=None) -> None:
        if not self._syncing and not self._closed:
            self._on_intent(SelectionControlsIntent(action, value))


class NativeCanvasControls:
    def __init__(
            self,
            document: TcDocument,
            brush_state: BrushControlsState,
            selection_state: SelectionControlsState,
            on_brush_intent: Callable[[BrushControlsIntent], None],
            on_selection_intent: Callable[[SelectionControlsIntent], None],
            viewport_rect: Callable[[], Rect]) -> None:
        self.widget = document.create_vstack("NativeCanvasControls")
        self.widget.stable_id = "diffusion-editor.canvas-controls"
        self.widget.set_layout_spacing(6.0)
        self.brush = NativeBrushPanel(
            document,
            brush_state,
            on_brush_intent,
            viewport_rect,
        )
        self.selection = NativeSelectionPanel(
            document,
            selection_state,
            on_selection_intent,
        )
        self.widget.add_preferred_child(self.brush.widget)
        self.widget.add_preferred_child(self.selection.widget)

    def apply_brush_state(self, state: BrushControlsState) -> None:
        self.brush.apply_state(state)

    def apply_selection_state(self, state: SelectionControlsState) -> None:
        self.selection.apply_state(state)

    def close(self) -> None:
        self.brush.close()
        self.selection.close()
