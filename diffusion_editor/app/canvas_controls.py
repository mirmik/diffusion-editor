"""Toolkit-neutral state and intents for brush/selection controls."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Protocol

import numpy as np

from ..canvas.brush import BrushToolMode
from ..canvas.editor_canvas_controller import EditorCanvasController
from ..document.commands import (
    ClearLayerPatchRectCommand,
    SetLayerPatchRectCommand,
    SetLayerSelectionCommand,
)
from ..document.document_service import DocumentService
from ..document.layer_stack import LayerStack

Rgba = tuple[int, int, int, int]


class BrushControlAction(str, Enum):
    TOOL = "tool"
    SIZE = "size"
    HARDNESS = "hardness"
    FLOW = "flow"
    COLOR = "color"
    DRAW_PATCH = "draw_patch"
    SHOW_PATCH = "show_patch"
    CLEAR_PATCH = "clear_patch"


class SelectionControlAction(str, Enum):
    EDIT_MODE = "edit_mode"
    RECT_MODE = "rect_mode"
    ERASER = "eraser"
    SIZE = "size"
    HARDNESS = "hardness"
    FLOW = "flow"
    SHOW = "show"


@dataclass(frozen=True)
class BrushControlsIntent:
    action: BrushControlAction
    value: object = None


@dataclass(frozen=True)
class SelectionControlsIntent:
    action: SelectionControlAction
    value: object = None


@dataclass(frozen=True)
class BrushControlsState:
    tool: BrushToolMode
    size: int
    hardness: float
    flow: float
    color: Rgba
    draw_patch: bool = False
    show_patch: bool = True


@dataclass(frozen=True)
class SelectionControlsState:
    edit_mode: bool = False
    rect_mode: bool = False
    eraser: bool = False
    size: int = 50
    hardness: float = 0.4
    flow: float = 1.0
    show: bool = True


class CanvasControlsPresentation(Protocol):
    def apply_brush_state(self, state: BrushControlsState) -> None: ...

    def apply_selection_state(self, state: SelectionControlsState) -> None: ...


class CanvasControlsCoordinator:
    """Coordinates native control intents with Canvas and document state."""

    def __init__(
            self,
            layer_stack: LayerStack,
            document: DocumentService,
            canvas: EditorCanvasController) -> None:
        self._layer_stack = layer_stack
        self._document = document
        self._canvas = canvas
        self._view: CanvasControlsPresentation | None = None
        self._closed = False
        brush = canvas.brush
        self._brush_state = BrushControlsState(
            tool=canvas.brush_tool_mode,
            size=brush.size,
            hardness=brush.hardness,
            flow=brush.flow,
            color=tuple(brush.color),
        )
        self._selection_state = SelectionControlsState()

        self._previous_color_picked = canvas.on_color_picked
        self._previous_patch_rect_drawn = canvas.on_patch_rect_drawn
        self._previous_selection_rect_drawn = canvas.on_selection_rect_drawn
        self._previous_brush_size_changed = canvas.on_brush_size_changed
        self._color_picked_callback = self._on_color_picked
        self._patch_rect_drawn_callback = self._on_patch_rect_drawn
        self._selection_rect_drawn_callback = self._on_selection_rect_drawn
        self._brush_size_changed_callback = self._on_brush_size_changed
        canvas.on_color_picked = self._color_picked_callback
        canvas.on_patch_rect_drawn = self._patch_rect_drawn_callback
        canvas.on_selection_rect_drawn = self._selection_rect_drawn_callback
        canvas.on_brush_size_changed = self._brush_size_changed_callback

    @property
    def brush_state(self) -> BrushControlsState:
        return self._brush_state

    @property
    def selection_state(self) -> SelectionControlsState:
        return self._selection_state

    def bind_view(self, view: CanvasControlsPresentation) -> None:
        self._require_open()
        self._view = view
        self._publish()

    def handle_brush_intent(self, intent: BrushControlsIntent) -> None:
        self._require_open()
        action, value = intent.action, intent.value
        state = self._brush_state
        if action == BrushControlAction.TOOL:
            mode = BrushToolMode(value)
            self._canvas.set_brush_tool(mode)
            self._deactivate_selection_modes()
            state = self._replace_brush(tool=mode)
            if mode in (BrushToolMode.MASK, BrushToolMode.MASK_ERASER):
                brush = self._canvas.brush
                self._canvas.set_mask_brush(
                    brush.size, brush.hardness, brush.flow)
        elif action == BrushControlAction.SIZE:
            self._canvas.brush.set_size(int(value))
            self._sync_mask_brush_if_active()
            state = self._replace_brush(size=self._canvas.brush.size)
        elif action == BrushControlAction.HARDNESS:
            self._canvas.brush.set_hardness(float(value))
            self._sync_mask_brush_if_active()
            state = self._replace_brush(
                hardness=self._canvas.brush.hardness)
        elif action == BrushControlAction.FLOW:
            self._canvas.brush.set_flow(float(value))
            self._sync_mask_brush_if_active()
            state = self._replace_brush(flow=self._canvas.brush.flow)
        elif action == BrushControlAction.COLOR:
            color = self._normalize_color(value)
            self._canvas.brush.set_color(*color)
            state = self._replace_brush(color=color)
        elif action == BrushControlAction.DRAW_PATCH:
            enabled = bool(value)
            if enabled:
                self._deactivate_selection_modes()
            self._canvas.set_patch_rect_mode(enabled)
            state = self._replace_brush(draw_patch=enabled)
        elif action == BrushControlAction.SHOW_PATCH:
            show = bool(value)
            self._canvas.set_show_patch_rect(show)
            state = self._replace_brush(show_patch=show)
        elif action == BrushControlAction.CLEAR_PATCH:
            self._clear_patch_rect()
        else:
            raise ValueError(f"unsupported brush control action: {action}")
        self._brush_state = state
        self._publish()

    def handle_selection_intent(self, intent: SelectionControlsIntent) -> None:
        self._require_open()
        action, value = intent.action, intent.value
        state = self._selection_state
        if action == SelectionControlAction.EDIT_MODE:
            enabled = bool(value)
            if enabled:
                self._disable_patch_mode()
                self._canvas.set_selection_rect_mode(False)
                self._canvas.set_selection_mode(True)
                state = self._replace_selection(
                    edit_mode=True, rect_mode=False)
            else:
                self._canvas.set_selection_mode(False)
                state = self._replace_selection(edit_mode=False)
        elif action == SelectionControlAction.RECT_MODE:
            enabled = bool(value)
            if enabled:
                self._disable_patch_mode()
                self._canvas.set_selection_mode(False)
                self._canvas.set_selection_rect_mode(True)
                state = self._replace_selection(
                    edit_mode=False, rect_mode=True)
            else:
                self._canvas.set_selection_rect_mode(False)
                state = self._replace_selection(rect_mode=False)
        elif action == SelectionControlAction.ERASER:
            eraser = bool(value)
            self._canvas.set_selection_eraser(eraser)
            state = self._replace_selection(eraser=eraser)
        elif action == SelectionControlAction.SIZE:
            size = max(1, min(int(value), 500))
            self._canvas.set_selection_brush(
                size, state.hardness, state.flow)
            state = self._replace_selection(size=size)
        elif action == SelectionControlAction.HARDNESS:
            hardness = max(0.0, min(float(value), 1.0))
            self._canvas.set_selection_brush(
                state.size, hardness, state.flow)
            state = self._replace_selection(hardness=hardness)
        elif action == SelectionControlAction.FLOW:
            flow = max(0.0, min(float(value), 1.0))
            self._canvas.set_selection_brush(
                state.size, state.hardness, flow)
            state = self._replace_selection(flow=flow)
        elif action == SelectionControlAction.SHOW:
            show = bool(value)
            self._canvas.set_show_selection(show)
            state = self._replace_selection(show=show)
        else:
            raise ValueError(f"unsupported selection control action: {action}")
        self._selection_state = state
        self._publish()

    def apply_generation_mask_brush(
            self,
            size: int,
            hardness: float,
            flow: float,
            eraser: bool) -> None:
        """Synchronize generation-panel mask controls with Canvas controls."""
        self._require_open()
        size = max(1, min(int(size), 500))
        hardness = max(0.0, min(float(hardness), 1.0))
        flow = max(0.0, min(float(flow), 1.0))
        mode = (
            BrushToolMode.MASK_ERASER
            if eraser else BrushToolMode.MASK
        )
        self._canvas.set_mask_brush(size, hardness, flow)
        self._canvas.set_mask_eraser(eraser)
        self._deactivate_selection_modes()
        self._brush_state = self._replace_brush(
            tool=mode,
            size=size,
            hardness=hardness,
            flow=flow,
        )
        self._publish()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._view = None
        if self._canvas.on_color_picked is self._color_picked_callback:
            self._canvas.on_color_picked = self._previous_color_picked
        if (
                self._canvas.on_patch_rect_drawn
                is self._patch_rect_drawn_callback):
            self._canvas.on_patch_rect_drawn = self._previous_patch_rect_drawn
        if (
                self._canvas.on_selection_rect_drawn
                is self._selection_rect_drawn_callback):
            self._canvas.on_selection_rect_drawn = (
                self._previous_selection_rect_drawn)
        if (
                self._canvas.on_brush_size_changed
                is self._brush_size_changed_callback):
            self._canvas.on_brush_size_changed = (
                self._previous_brush_size_changed)

    def _replace_brush(self, **changes) -> BrushControlsState:
        return replace(self._brush_state, **changes)

    def _replace_selection(self, **changes) -> SelectionControlsState:
        return replace(self._selection_state, **changes)

    def _publish(self) -> None:
        if self._view is None:
            return
        self._view.apply_brush_state(self._brush_state)
        self._view.apply_selection_state(self._selection_state)

    def _sync_mask_brush_if_active(self) -> None:
        if self._brush_state.tool in (
                BrushToolMode.MASK, BrushToolMode.MASK_ERASER):
            brush = self._canvas.brush
            self._canvas.set_mask_brush(
                brush.size, brush.hardness, brush.flow)

    def _deactivate_selection_modes(self) -> None:
        self._canvas.set_selection_mode(False)
        self._canvas.set_selection_rect_mode(False)
        self._selection_state = self._replace_selection(
            edit_mode=False,
            rect_mode=False,
        )

    def _disable_patch_mode(self) -> None:
        self._canvas.set_patch_rect_mode(False)
        self._brush_state = self._replace_brush(draw_patch=False)

    def _clear_patch_rect(self) -> None:
        layer = self._layer_stack.active_layer
        if layer is not None:
            self._document.execute(ClearLayerPatchRectCommand(
                layer=layer,
                label="Clear Patch Rect",
            ))

    def _on_color_picked(self, r: int, g: int, b: int, a: int) -> None:
        color = self._normalize_color((r, g, b, a))
        self._canvas.brush.set_color(*color)
        self._brush_state = self._replace_brush(color=color)
        self._publish()
        if self._previous_color_picked is not None:
            self._previous_color_picked(*color)

    def _on_brush_size_changed(self, size: int) -> None:
        self._brush_state = self._replace_brush(size=size)
        self._sync_mask_brush_if_active()
        self._publish()
        if self._previous_brush_size_changed is not None:
            self._previous_brush_size_changed(size)

    def _on_patch_rect_drawn(
            self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._disable_patch_mode()
        layer = self._layer_stack.active_layer
        if layer is not None:
            lx0, ly0, lx1, ly1 = layer.bounds
            clipped = (
                max(x0, lx0),
                max(y0, ly0),
                min(x1, lx1),
                min(y1, ly1),
            )
            cx0, cy0, cx1, cy1 = clipped
            if cx1 - cx0 > 2 and cy1 - cy0 > 2:
                self._document.execute(SetLayerPatchRectCommand(
                    layer=layer,
                    rect=layer.canvas_rect_to_local(clipped),
                    label="Set Patch Rect",
                ))
        self._publish()
        if self._previous_patch_rect_drawn is not None:
            self._previous_patch_rect_drawn(x0, y0, x1, y1)

    def _on_selection_rect_drawn(
            self, x0: int, y0: int, x1: int, y1: int) -> None:
        self._canvas.set_selection_rect_mode(False)
        self._selection_state = self._replace_selection(rect_mode=False)
        height, width = self._layer_stack.height, self._layer_stack.width
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(width, x1), min(height, y1)
        if x1 > x0 and y1 > y0:
            mask = np.zeros((height, width), dtype=np.float32)
            mask[y0:y1, x0:x1] = 1.0
            self._document.execute(SetLayerSelectionCommand(
                mask=mask,
                label="Rect Selection",
            ))
        self._publish()
        if self._previous_selection_rect_drawn is not None:
            self._previous_selection_rect_drawn(x0, y0, x1, y1)

    @staticmethod
    def _normalize_color(value) -> Rgba:
        if not isinstance(value, (tuple, list)) or len(value) != 4:
            raise ValueError("brush color must be an RGBA tuple")
        return tuple(max(0, min(int(channel), 255)) for channel in value)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("canvas controls coordinator is closed")
