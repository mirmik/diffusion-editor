"""Toolkit-neutral interaction controller for the editor canvas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from ..document.layer import Layer
from ..document.change_event import DocumentChangeEvent, DocumentChangeKind
from ..document.layer_stack import LayerStack
from .brush import Brush, BrushToolMode
from .canvas_composite import CanvasCompositeBridge
from .canvas_edit_session import CanvasEditSession
from .canvas_mask_erase import MaskEraseStrokeBuffer
from .canvas_mask_paint import CanvasMaskPainter
from .canvas_overlay import CanvasOverlayBridge
from .canvas_paint_stroke import PaintStrokeBuffer
from .canvas_rect_drag import CanvasRectDragController
from .canvas_selection_paint import CanvasSelectionPainter
from .canvas_smudge import SmudgeStrokeBuffer
from .canvas_tool_context import CanvasToolContext
from .canvas_tools import SelectionPaintTool, create_canvas_tools

Rect = tuple[int, int, int, int]


@dataclass(frozen=True)
class CanvasAnnotation:
    """An image-space rectangle drawn above the canvas textures."""

    kind: str
    rect: Rect


class EditorCanvasController:
    """Editor behavior independent from Termin GUI widgets."""

    LEFT_BUTTON = 0
    RIGHT_BUTTON = 1
    CTRL_MODIFIER = 2

    def __init__(
            self,
            layer_stack: LayerStack,
            *,
            gpu_compositing: bool = True,
            graphics=None,
            set_image: Callable[[np.ndarray | None], None],
            set_overlay: Callable[[np.ndarray | None], None],
            update_image_region: (
                Callable[[int, int, np.ndarray], None] | None
            ) = None,
            update_overlay_region: (
                Callable[[int, int, np.ndarray], None] | None
            ) = None,
            request_repaint: Callable[[], None] | None = None,
            set_cursor: Callable[[str], None] | None = None):
        self._layer_stack = layer_stack
        self._request_repaint = request_repaint or (lambda: None)
        self._set_cursor = set_cursor or (lambda _cursor: None)
        self._composite_bridge = CanvasCompositeBridge(
            layer_stack,
            gpu_compositing=gpu_compositing,
            graphics=graphics,
            set_image=set_image,
            update_image_region=update_image_region,
        )
        self._overlay_bridge = CanvasOverlayBridge(
            layer_stack,
            set_overlay=set_overlay,
            update_overlay_region=update_overlay_region,
        )

        self.brush = Brush()
        self._brush_tool_mode = BrushToolMode.PAINT
        self._mask_painter = CanvasMaskPainter()
        self._selection_mode = False
        self._selection_painter = CanvasSelectionPainter()
        self._rect_drags = CanvasRectDragController()
        self._paint_stroke = PaintStrokeBuffer()
        self._edit_session = CanvasEditSession()
        self._mask_erase_stroke = MaskEraseStrokeBuffer()
        self._smudge_stroke = SmudgeStrokeBuffer()
        self._tool_context = CanvasToolContext(
            layer_stack,
            self.brush,
            self._composite_bridge,
            self._overlay_bridge,
            self._paint_stroke,
            self._selection_painter,
            self._mask_painter,
            self._mask_erase_stroke,
            self._smudge_stroke,
        )
        self._stroke_tools = create_canvas_tools()
        self._active_stroke_tool = self._stroke_tools[self._brush_tool_mode]
        self._selection_tool = SelectionPaintTool()
        self._edit_tool = None
        self._edit_tool_started = False

        self.on_mouse_moved: Callable[[int, int], None] | None = None
        self.on_color_picked: Callable[[int, int, int, int], None] | None = None
        self.on_patch_rect_drawn: (
            Callable[[int, int, int, int], None] | None
        ) = None
        self.on_selection_rect_drawn: (
            Callable[[int, int, int, int], None] | None
        ) = None
        self.on_edit_begin: Callable[[str, Layer | None, str], None] | None = None
        self.on_edit_end: (
            Callable[[Layer | None, str, Rect | None], None] | None
        ) = None
        self.on_edit_cancel: (
            Callable[[Layer | None, str], None] | None
        ) = None
        self.on_brush_size_changed: Callable[[int], None] | None = None

    @property
    def composite_bridge(self) -> CanvasCompositeBridge:
        return self._composite_bridge

    @property
    def overlay_bridge(self) -> CanvasOverlayBridge:
        return self._overlay_bridge

    @property
    def brush_tool_mode(self) -> BrushToolMode:
        return self._brush_tool_mode

    @property
    def pointer_interaction_active(self) -> bool:
        return self._edit_session.active or self._rect_drags.dragging

    def refresh(self) -> None:
        if self._layer_stack.width <= 0 or self._layer_stack.height <= 0:
            self._composite_bridge.clear()
            self._overlay_bridge.clear_output()
            self._request_repaint()
            return
        self._composite_bridge.rebuild()
        self._composite_bridge.update_composite()
        self._overlay_bridge.clear()
        self._overlay_bridge.rebuild()
        self._request_repaint()

    def handle_document_change(self, event: DocumentChangeEvent) -> None:
        """Apply the least expensive refresh implied by a typed event."""
        if event.kind == DocumentChangeKind.PIXELS:
            layer = (
                self._layer_stack.find_layer_by_id(event.layer_ids[0])
                if len(event.layer_ids) == 1 else None
            )
            if layer is None:
                self.refresh()
                return
            self._composite_bridge.apply_published_pixel_change(
                layer, event.dirty_rect)
        elif event.kind in {
                DocumentChangeKind.TRANSFORM,
                DocumentChangeKind.VISIBILITY,
                DocumentChangeKind.OPACITY}:
            self._composite_bridge.apply_published_composite_change()
        elif event.kind in {
                DocumentChangeKind.STRUCTURE,
                DocumentChangeKind.SNAPSHOT_RESTORE}:
            self.refresh()
            return
        self._overlay_bridge.clear()
        self._overlay_bridge.rebuild()
        self._request_repaint()

    def get_composite(self) -> np.ndarray | None:
        return self._composite_bridge.get_composite()

    def get_composite_below(self, layer: Layer) -> np.ndarray | None:
        return self._composite_bridge.get_composite_below(layer)

    def set_mask_brush(
            self, size: int, hardness: float, flow: float = 1.0) -> None:
        self._mask_painter.set_brush(size, hardness, flow)
        if self._brush_tool_mode not in (
                BrushToolMode.MASK, BrushToolMode.MASK_ERASER):
            self.set_brush_tool(BrushToolMode.MASK)

    def set_mask_eraser(self, eraser: bool) -> None:
        self._mask_painter.set_eraser(eraser)
        self.set_brush_tool(
            BrushToolMode.MASK_ERASER if eraser else BrushToolMode.MASK)

    def set_brush_eraser(self, eraser: bool) -> None:
        self.set_brush_tool(
            BrushToolMode.ERASER if eraser else BrushToolMode.PAINT)

    def set_brush_tool(self, mode: BrushToolMode | str) -> None:
        self._brush_tool_mode = BrushToolMode(mode)
        self._mask_painter.set_eraser(
            self._brush_tool_mode == BrushToolMode.MASK_ERASER)
        self._active_stroke_tool = self._stroke_tools[self._brush_tool_mode]

    def set_selection_mode(self, on: bool) -> None:
        self._selection_mode = on
        self._set_cursor("crosshair" if on else "default")

    def set_selection_brush(
            self, size: int, hardness: float, flow: float = 1.0) -> None:
        self._selection_painter.set_brush(size, hardness, flow)

    def set_selection_eraser(self, eraser: bool) -> None:
        self._selection_painter.set_eraser(eraser)

    def set_selection_rect_mode(self, on: bool) -> None:
        self._rect_drags.set_selection_rect_mode(on)
        self._set_cursor("crosshair" if on else "default")

    def set_show_mask(self, show: bool) -> None:
        self._overlay_bridge.show_mask = show
        self._overlay_bridge.clear()
        self._overlay_bridge.rebuild()
        self._request_repaint()

    def set_show_selection(self, show: bool) -> None:
        self._overlay_bridge.show_selection = show
        self._overlay_bridge.rebuild()
        self._request_repaint()

    def set_patch_rect_mode(self, on: bool) -> None:
        self._rect_drags.set_patch_rect_mode(on)
        self._set_cursor("crosshair" if on else "default")

    def set_show_patch_rect(self, show: bool) -> None:
        self._rect_drags.set_show_patch_rect(show)
        self._request_repaint()

    def adjust_brush_size(self, delta: int) -> None:
        self.brush.set_size(self.brush.size + delta)
        if self.on_brush_size_changed is not None:
            self.on_brush_size_changed(self.brush.size)
        self._request_repaint()

    def pointer_down(
            self, ix: float, iy: float, button: int, modifiers: int = 0) -> None:
        x, y = int(ix), int(iy)
        if button == self.RIGHT_BUTTON or (
                button == self.LEFT_BUTTON
                and modifiers & self.CTRL_MODIFIER):
            self._pick_color(x, y)
            return
        if button != self.LEFT_BUTTON:
            return
        if self.pointer_interaction_active:
            self.pointer_cancel()

        layer = self._layer_stack.active_layer
        if layer is None:
            return
        if self._rect_drags.begin_selection_rect(x, y):
            self._request_repaint()
            return
        if self._selection_mode:
            self._begin_tool_edit(self._selection_tool, None, x, y)
            return
        if not self._can_edit_layer(layer):
            return
        if self._rect_drags.begin_patch_rect(x, y):
            self._request_repaint()
            return
        self._begin_tool_edit(self._active_stroke_tool, layer, x, y)

    def pointer_move(self, ix: float, iy: float) -> None:
        x, y = int(ix), int(iy)
        if self._rect_drags.move(x, y):
            self._request_repaint()
        elif self._edit_session.active:
            self._move_tool_edit(x, y)
        if self.on_mouse_moved:
            self.on_mouse_moved(x, y)

    def pointer_up(self, ix: float, iy: float) -> None:
        x, y = int(ix), int(iy)
        rect_result = self._rect_drags.finish(x, y)
        if rect_result.handled:
            self._set_cursor("default")
            if rect_result.rect is not None:
                if (
                        rect_result.target == "selection"
                        and self.on_selection_rect_drawn):
                    self.on_selection_rect_drawn(*rect_result.rect)
                elif (
                        rect_result.target == "patch"
                        and self.on_patch_rect_drawn):
                    self.on_patch_rect_drawn(*rect_result.rect)
            self._request_repaint()
            return
        if self._edit_session.active:
            self._finish_tool_edit()

    def pointer_cancel(self) -> None:
        rect_cancelled = self._rect_drags.cancel()
        edit_cancelled = self._edit_session.active
        errors: list[tuple[BaseException, object]] = []

        def cleanup(callback) -> None:
            try:
                callback()
            except BaseException as exc:
                errors.append((exc, exc.__traceback__))

        if edit_cancelled:
            tool = self._edit_tool
            layer = self._edit_session.layer
            target = self._edit_session.target
            if tool is not None and self._edit_tool_started:
                cleanup(lambda: tool.end(self._tool_context, layer))
            # Roll back persistent state before any presentation cleanup;
            # GPU/overlay failures must never suppress the transaction owner.
            if self.on_edit_cancel is not None:
                cleanup(lambda: self.on_edit_cancel(layer, target or ""))
            if target != "selection":
                cleanup(self._overlay_bridge.clear)
                cleanup(self._overlay_bridge.rebuild)
            self._edit_tool = None
            self._edit_tool_started = False
            self._edit_session.clear()
        if rect_cancelled or edit_cancelled:
            cleanup(lambda: self._set_cursor(
                "crosshair" if self._selection_mode else "default"))
            cleanup(self._request_repaint)
        if errors:
            error, traceback = errors[0]
            raise error.with_traceback(traceback)

    def annotations(self) -> tuple[CanvasAnnotation, ...]:
        result: list[CanvasAnnotation] = []
        layer = self._layer_stack.active_layer
        if layer is not None and layer.width > 0 and layer.height > 0:
            result.append(CanvasAnnotation("active-layer", layer.bounds))
        selection = self._rect_drags.selection_preview_rect()
        if selection is not None:
            result.append(CanvasAnnotation("selection", selection))
        patch = self._rect_drags.patch_preview_rect(layer)
        if patch is not None:
            result.append(CanvasAnnotation("patch", patch))
        return tuple(result)

    def dispose(self) -> None:
        try:
            self.pointer_cancel()
        finally:
            self._composite_bridge.dispose()

    def _can_edit_layer(self, layer: Layer | None) -> bool:
        return (
            layer is not None
            and self._layer_stack.is_layer_visible_for_composition(layer)
        )

    def _begin_tool_edit(self, tool, layer, x: int, y: int) -> None:
        self._edit_tool = tool
        self._edit_session.begin(
            label=tool.label,
            target=tool.target,
            layer=layer,
            pos=(x, y),
        )
        try:
            if self.on_edit_begin:
                self.on_edit_begin(tool.label, layer, tool.target)
            self._edit_tool_started = True
            self._edit_session.add_dirty(
                tool.begin(self._tool_context, layer, x, y))
        except BaseException as exc:
            try:
                self.pointer_cancel()
            except BaseException as cleanup_error:
                raise exc from cleanup_error
            raise

    def _move_tool_edit(self, x: int, y: int) -> None:
        tool = self._edit_tool
        if tool is None:
            return
        layer = None
        if self._edit_session.target != "selection":
            layer = self._edit_session.layer
            if (
                    layer is None
                    or self._layer_stack.find_layer_by_id(layer.id) is not layer):
                self.pointer_cancel()
                return
        try:
            dirty = tool.move(
                self._tool_context,
                layer,
                self._edit_session.last_pos,
                x,
                y,
            )
        except BaseException as exc:
            try:
                self.pointer_cancel()
            except BaseException as cleanup_error:
                raise exc from cleanup_error
            raise
        self._edit_session.add_dirty(dirty)
        self._edit_session.move_to((x, y))
        self._request_repaint()

    def _finish_tool_edit(self) -> None:
        tool = self._edit_tool
        layer = None
        if self._edit_session.target != "selection":
            layer = self._edit_session.layer
            if (
                    layer is None
                    or self._layer_stack.find_layer_by_id(layer.id) is not layer):
                self.pointer_cancel()
                return
        target = self._edit_session.target
        dirty_rect = self._edit_session.dirty_rect
        try:
            if tool is not None and self._edit_tool_started:
                tool.end(self._tool_context, layer)
                self._edit_tool_started = False
            if target != "selection":
                self._overlay_bridge.clear()
                self._overlay_bridge.rebuild()
            if self.on_edit_end:
                self.on_edit_end(
                    self._edit_session.layer,
                    target,
                    dirty_rect,
                )
        except BaseException as exc:
            try:
                if self.on_edit_cancel is not None:
                    self.on_edit_cancel(layer, target or "")
            except BaseException as cleanup_error:
                raise exc from cleanup_error
            raise
        finally:
            self._edit_tool = None
            self._edit_tool_started = False
            self._edit_session.clear()
            self._request_repaint()

    def _pick_color(self, x: int, y: int) -> None:
        if self._layer_stack.width == 0 or self._layer_stack.height == 0:
            return
        composite = self._layer_stack.composite()
        h, w = composite.shape[:2]
        if 0 <= x < w and 0 <= y < h and self.on_color_picked:
            r, g, b, a = composite[y, x]
            self.on_color_picked(int(r), int(g), int(b), int(a))
