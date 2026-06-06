"""Explicit runtime API exposed by EditorCanvas to brush-like tools."""

from __future__ import annotations

from .brush import Brush
from .canvas_composite import CanvasCompositeBridge
from .canvas_geometry import (
    canvas_to_layer_point,
    clip_layer_local_rect,
    union_rect,
)
from .canvas_image_erase import erase_alpha_region
from .canvas_mask_erase import MaskEraseStrokeBuffer
from .canvas_mask_paint import CanvasMaskPainter
from .canvas_overlay import CanvasOverlayBridge
from .canvas_paint_stroke import PaintStrokeBuffer
from .canvas_selection_paint import CanvasSelectionPainter
from .canvas_smudge import SmudgeStrokeBuffer
from ..document.layer import Layer
from ..document.layer_stack import LayerStack

Rect = tuple[int, int, int, int]


class CanvasToolContext:
    """Narrow host API for canvas tools.

    Tools should depend on this object instead of reaching into EditorCanvas
    internals. The context owns no state itself; it coordinates the canvas'
    runtime collaborators.
    """

    def __init__(
            self,
            layer_stack: LayerStack,
            brush: Brush,
            composite_bridge: CanvasCompositeBridge,
            overlay_bridge: CanvasOverlayBridge,
            paint_stroke: PaintStrokeBuffer,
            selection_painter: CanvasSelectionPainter,
            mask_painter: CanvasMaskPainter,
            mask_erase_stroke: MaskEraseStrokeBuffer,
            smudge_stroke: SmudgeStrokeBuffer):
        self._layer_stack = layer_stack
        self.brush = brush
        self._composite_bridge = composite_bridge
        self._overlay_bridge = overlay_bridge
        self._paint_stroke = paint_stroke
        self._selection_painter = selection_painter
        self._mask_painter = mask_painter
        self._mask_erase_stroke = mask_erase_stroke
        self._smudge_stroke = smudge_stroke

    def canvas_to_layer_point(self, layer: Layer, x: int, y: int) -> tuple[int, int]:
        return canvas_to_layer_point(layer, x, y)

    def union_rect(self, a: Rect | None, b: Rect | None) -> Rect | None:
        return union_rect(a, b)

    def mark_layer_dirty(self, layer: Layer, rect: Rect | None = None) -> None:
        self._layer_stack.mark_layer_dirty(layer, rect)

    def clip_layer_local_rect(
            self,
            layer: Layer,
            rect: Rect | None) -> tuple[Rect | None, Rect | None]:
        return clip_layer_local_rect(
            layer,
            rect,
            self._layer_stack.width,
            self._layer_stack.height,
        )

    def refresh_modified_layer_rect(self, layer: Layer, rect: Rect | None) -> None:
        local_rect, canvas_rect = self.clip_layer_local_rect(layer, rect)
        if local_rect is None:
            return
        self._composite_bridge.refresh_modified_layer_rect(
            layer,
            local_rect,
            canvas_rect,
        )

    def refresh_layer_transform(
            self,
            layer: Layer,
            dirty_canvas_rect: Rect | None) -> None:
        self._composite_bridge.refresh_layer_transform(layer, dirty_canvas_rect)

    def rebuild_overlay(self) -> None:
        self._overlay_bridge.rebuild()

    def clear_overlay(self) -> None:
        self._overlay_bridge.clear()

    def clear_paint_stroke(self) -> None:
        self._paint_stroke.clear()

    def begin_paint_stroke(self, layer: Layer) -> bool:
        return self._paint_stroke.begin(layer.image, tuple(self.brush.color))

    @property
    def paint_stroke_mask(self):
        return self._paint_stroke.mask

    def apply_paint_stroke_region(
            self,
            layer: Layer,
            dirty: Rect | None) -> Rect | None:
        dirty = self._paint_stroke.apply_region(layer.image, dirty)
        if dirty is None:
            return None
        self.refresh_modified_layer_rect(layer, dirty)
        return dirty

    def end_paint_stroke(self, layer: Layer | None) -> None:
        dirty = self._paint_stroke.live_dirty_rect
        if layer is not None and dirty is not None:
            self.mark_layer_dirty(layer, layer.local_rect_to_canvas(dirty))
        self._paint_stroke.clear()
        self._overlay_bridge.clear()
        self._overlay_bridge.rebuild()

    def begin_smudge(self, layer: Layer, x: int, y: int) -> None:
        self._smudge_stroke.begin(layer.image, self.brush, x, y)

    def smudge_line(
            self,
            layer: Layer,
            x0: int,
            y0: int,
            x1: int,
            y1: int) -> Rect | None:
        dirty = self._smudge_stroke.line(
            layer.image,
            self.brush,
            x0,
            y0,
            x1,
            y1,
        )
        self.refresh_modified_layer_rect(layer, dirty)
        return dirty

    def clear_smudge(self) -> None:
        self._smudge_stroke.clear()

    def selection_dab(self, x: int, y: int) -> Rect | None:
        dirty, _stamp = self._selection_painter.dab(
            self._layer_stack.selection.data,
            x,
            y,
        )
        self._overlay_bridge.rebuild()
        return dirty

    def selection_line(
            self,
            x0: int,
            y0: int,
            x1: int,
            y1: int) -> Rect | None:
        dirty, _stamp = self._selection_painter.line(
            self._layer_stack.selection.data,
            x0,
            y0,
            x1,
            y1,
        )
        self._overlay_bridge.rebuild()
        return dirty

    def mask_dab(
            self,
            layer: Layer,
            x: int,
            y: int,
            *,
            erase: bool) -> Rect | None:
        dirty, _stamp = self._mask_painter.dab(
            layer.mask.data,
            x,
            y,
            erase=erase,
        )
        self._overlay_bridge.update_mask_region(layer, dirty)
        return dirty

    def mask_line(
            self,
            layer: Layer,
            x0: int,
            y0: int,
            x1: int,
            y1: int,
            *,
            erase: bool) -> Rect | None:
        dirty, _stamp = self._mask_painter.line(
            layer.mask.data,
            x0,
            y0,
            x1,
            y1,
            erase=erase,
        )
        self._overlay_bridge.update_mask_region(layer, dirty)
        return dirty

    def begin_mask_erase(self, layer: Layer) -> None:
        self._mask_erase_stroke.begin(layer.height, layer.width)

    def mask_erase_dab(self, x: int, y: int) -> Rect | None:
        erase_mask = self._mask_erase_stroke.mask
        if erase_mask is None:
            return None
        dirty, _stamp = self._mask_painter.dab(erase_mask, x, y, erase=False)
        return dirty

    def mask_erase_line(
            self,
            x0: int,
            y0: int,
            x1: int,
            y1: int) -> Rect | None:
        erase_mask = self._mask_erase_stroke.mask
        if erase_mask is None:
            return None
        dirty, _stamp = self._mask_painter.line(
            erase_mask,
            x0,
            y0,
            x1,
            y1,
            erase=False,
        )
        return dirty

    def ensure_mask_erase(self, layer: Layer) -> None:
        if self._mask_erase_stroke.mask is None:
            self.begin_mask_erase(layer)

    def preview_mask_erase(self, layer: Layer, dirty: Rect | None) -> None:
        self._mask_erase_stroke.add_dirty(dirty)
        if dirty is None:
            return
        if self._overlay_bridge.show_mask:
            preview = self._mask_erase_stroke.preview_region(
                layer.mask.data,
                dirty,
            )
            self._overlay_bridge.update_mask_region_preview(layer, dirty, preview)

        local_rect, canvas_rect = self.clip_layer_local_rect(layer, dirty)
        if local_rect is None:
            return
        erase = self._mask_erase_stroke.erase_region(local_rect)
        if erase is None:
            return
        self._composite_bridge.preview_erased_layer_rect(
            layer,
            local_rect,
            canvas_rect,
            erase,
        )

    def finish_mask_erase(self, layer: Layer | None) -> None:
        dirty = self._mask_erase_stroke.dirty_rect
        if layer is not None and dirty is not None:
            self._mask_erase_stroke.apply_to_layer_mask(layer.mask.data)
            local_rect, canvas_rect = self.clip_layer_local_rect(layer, dirty)
            if local_rect is not None:
                erase = self._mask_erase_stroke.erase_region(local_rect)
                if erase is not None:
                    erase_alpha_region(layer.image, local_rect, erase)
                    self._composite_bridge.refresh_modified_layer_rect(
                        layer,
                        local_rect,
                        canvas_rect,
                    )
            self._overlay_bridge.update_mask_region(layer, dirty)
        self._mask_erase_stroke.clear()
