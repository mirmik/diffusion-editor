"""Brush-like canvas tools used by EditorCanvas."""

from __future__ import annotations

from .brush import BrushToolMode
from .canvas_image_erase import erase_dab, erase_line


class CanvasStrokeTool:
    mode: BrushToolMode
    label: str
    target: str

    def begin(self, context, layer, x: int, y: int):
        raise NotImplementedError

    def move(self, context, layer, last_pos: tuple[int, int] | None, x: int, y: int):
        raise NotImplementedError

    def end(self, context, layer):
        pass


class PaintTool(CanvasStrokeTool):
    mode = BrushToolMode.PAINT
    label = "Paint Stroke"
    target = "image"

    def begin(self, context, layer, x: int, y: int):
        x, y = context.canvas_to_layer_point(layer, x, y)
        context.begin_paint_stroke(layer)
        stroke_mask = context.paint_stroke_mask
        if stroke_mask is None:
            return None
        dirty = context.brush.dab_to_mask(stroke_mask, x, y)
        self._apply_region(context, layer, dirty)
        return dirty

    def move(self, context, layer, last_pos, x: int, y: int):
        stroke_mask = context.paint_stroke_mask
        if stroke_mask is None:
            return None
        x, y = context.canvas_to_layer_point(layer, x, y)
        if last_pos:
            lx, ly = context.canvas_to_layer_point(layer, *last_pos)
            dirty = context.brush.stroke_to_mask(stroke_mask, lx, ly, x, y)
        else:
            dirty = context.brush.dab_to_mask(stroke_mask, x, y)
        self._apply_region(context, layer, dirty)
        return dirty

    def end(self, context, layer):
        context.end_paint_stroke(layer)

    def _apply_region(self, context, layer, dirty):
        if dirty is None or layer is None:
            return
        context.apply_paint_stroke_region(layer, dirty)


class EraserTool(CanvasStrokeTool):
    mode = BrushToolMode.ERASER
    label = "Erase Stroke"
    target = "image"

    def begin(self, context, layer, x: int, y: int):
        context.clear_paint_stroke()
        x, y = context.canvas_to_layer_point(layer, x, y)
        dirty = erase_dab(layer.image, context.brush, x, y)
        context.refresh_modified_layer_rect(layer, dirty)
        return dirty

    def move(self, context, layer, last_pos, x: int, y: int):
        x, y = context.canvas_to_layer_point(layer, x, y)
        if last_pos:
            lx, ly = context.canvas_to_layer_point(layer, *last_pos)
            dirty = erase_line(layer.image, context.brush, lx, ly, x, y)
        else:
            dirty = erase_dab(layer.image, context.brush, x, y)
        context.refresh_modified_layer_rect(layer, dirty)
        return dirty

    def end(self, context, layer):
        if layer is not None:
            context.mark_layer_dirty(layer)


class SmudgeTool(CanvasStrokeTool):
    mode = BrushToolMode.SMUDGE
    label = "Smudge Stroke"
    target = "image"

    def __init__(self):
        self._dirty_rect = None

    def begin(self, context, layer, x: int, y: int):
        self._dirty_rect = None
        x, y = context.canvas_to_layer_point(layer, x, y)
        context.begin_smudge(layer, x, y)
        return None

    def move(self, context, layer, last_pos, x: int, y: int):
        if not last_pos:
            return None
        lx, ly = context.canvas_to_layer_point(layer, *last_pos)
        x, y = context.canvas_to_layer_point(layer, x, y)
        dirty = context.smudge_line(layer, lx, ly, x, y)
        self._dirty_rect = context.union_rect(self._dirty_rect, dirty)
        return dirty

    def end(self, context, layer):
        if layer is not None and self._dirty_rect is not None:
            context.mark_layer_dirty(
                layer, layer.local_rect_to_canvas(self._dirty_rect))
        self._dirty_rect = None
        context.clear_smudge()


class SelectionPaintTool(CanvasStrokeTool):
    mode = None
    label = "Selection Stroke"
    target = "selection"

    def begin(self, context, layer, x: int, y: int):
        return context.selection_dab(x, y)

    def move(self, context, layer, last_pos, x: int, y: int):
        if last_pos:
            lx, ly = last_pos
            return context.selection_line(lx, ly, x, y)
        return context.selection_dab(x, y)

    def end(self, context, layer):
        context.rebuild_overlay()


class MaskPaintTool(CanvasStrokeTool):
    mode = BrushToolMode.MASK
    label = "Mask Stroke"
    target = "mask"

    def begin(self, context, layer, x: int, y: int):
        x, y = context.canvas_to_layer_point(layer, x, y)
        return context.mask_dab(layer, x, y, erase=False)

    def move(self, context, layer, last_pos, x: int, y: int):
        x, y = context.canvas_to_layer_point(layer, x, y)
        if last_pos:
            lx, ly = context.canvas_to_layer_point(layer, *last_pos)
            return context.mask_line(layer, lx, ly, x, y, erase=False)
        return context.mask_dab(layer, x, y, erase=False)


class MaskEraserTool(CanvasStrokeTool):
    mode = BrushToolMode.MASK_ERASER
    label = "Mask Erase Stroke"
    target = "mask"

    def begin(self, context, layer, x: int, y: int):
        x, y = context.canvas_to_layer_point(layer, x, y)
        context.begin_mask_erase(layer)
        dirty = context.mask_erase_dab(x, y)
        context.preview_mask_erase(layer, dirty)
        return dirty

    def move(self, context, layer, last_pos, x: int, y: int):
        x, y = context.canvas_to_layer_point(layer, x, y)
        context.ensure_mask_erase(layer)
        if last_pos:
            lx, ly = context.canvas_to_layer_point(layer, *last_pos)
            dirty = context.mask_erase_line(lx, ly, x, y)
        else:
            dirty = context.mask_erase_dab(x, y)
        context.preview_mask_erase(layer, dirty)
        return dirty

    def end(self, context, layer):
        context.finish_mask_erase(layer)


class MoveTool(CanvasStrokeTool):
    mode = BrushToolMode.MOVE
    label = "Move Layer"
    target = "transform"

    def begin(self, context, layer, x: int, y: int):
        if layer is None:
            return None
        self._start_x = x
        self._start_y = y
        self._layer_start_x = layer.x
        self._layer_start_y = layer.y
        return layer.bounds

    def move(self, context, layer, last_pos, x: int, y: int):
        if layer is None:
            return None
        dx = x - self._start_x
        dy = y - self._start_y
        old_bounds = layer.bounds
        layer.x = self._layer_start_x + dx
        layer.y = self._layer_start_y + dy
        dirty = context.union_rect(old_bounds, layer.bounds)
        context.refresh_layer_transform(layer, dirty)
        return dirty

    def end(self, context, layer):
        pass


def create_canvas_tools() -> dict[BrushToolMode, CanvasStrokeTool]:
    return {
        BrushToolMode.PAINT: PaintTool(),
        BrushToolMode.ERASER: EraserTool(),
        BrushToolMode.SMUDGE: SmudgeTool(),
        BrushToolMode.MASK: MaskPaintTool(),
        BrushToolMode.MASK_ERASER: MaskEraserTool(),
        BrushToolMode.MOVE: MoveTool(),
    }
