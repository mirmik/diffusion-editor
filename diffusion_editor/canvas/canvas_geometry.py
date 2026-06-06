"""Geometry helpers for canvas/layer coordinate conversion."""

from __future__ import annotations

from ..document.layer import Layer

Rect = tuple[int, int, int, int]


def union_rect(a: Rect | None, b: Rect | None) -> Rect | None:
    if a is None:
        return b
    if b is None:
        return a
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return (min(ax0, bx0), min(ay0, by0), max(ax1, bx1), max(ay1, by1))


def canvas_to_layer_point(layer: Layer, x: int, y: int) -> tuple[int, int]:
    return int(x - layer.x), int(y - layer.y)


def clip_canvas_rect(rect: Rect, canvas_width: int, canvas_height: int) -> Rect:
    x0, y0, x1, y1 = rect
    return (
        max(0, x0),
        max(0, y0),
        min(canvas_width, x1),
        min(canvas_height, y1),
    )


def visible_layer_rect(
        layer: Layer,
        canvas_width: int,
        canvas_height: int) -> Rect:
    cx0, cy0, cx1, cy1 = clip_canvas_rect(
        layer.bounds,
        canvas_width,
        canvas_height,
    )
    if cx1 <= cx0 or cy1 <= cy0:
        return (0, 0, 0, 0)
    return (cx0 - layer.x, cy0 - layer.y, cx1 - layer.x, cy1 - layer.y)


def clip_layer_local_rect(
        layer: Layer,
        rect: Rect | None,
        canvas_width: int,
        canvas_height: int) -> tuple[Rect | None, Rect | None]:
    if rect is None:
        return None, None

    x0, y0, x1, y1 = rect
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(layer.width, x1)
    y1 = min(layer.height, y1)
    if x1 <= x0 or y1 <= y0:
        return None, None

    canvas_rect = layer.local_rect_to_canvas((x0, y0, x1, y1))
    cx0, cy0, cx1, cy1 = clip_canvas_rect(
        canvas_rect,
        canvas_width,
        canvas_height,
    )
    if cx1 <= cx0 or cy1 <= cy0:
        return None, None

    lx0 = cx0 - layer.x
    ly0 = cy0 - layer.y
    lx1 = lx0 + (cx1 - cx0)
    ly1 = ly0 + (cy1 - cy0)
    return (lx0, ly0, lx1, ly1), (cx0, cy0, cx1, cy1)
