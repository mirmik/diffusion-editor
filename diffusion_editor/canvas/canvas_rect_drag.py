"""Small state machine for canvas rectangle drag interactions."""

from __future__ import annotations

from dataclasses import dataclass

from .canvas_geometry import Rect


@dataclass(frozen=True)
class CanvasRectDragResult:
    rect: Rect


@dataclass(frozen=True)
class CanvasRectDragFinish:
    handled: bool
    target: str | None = None
    rect: Rect | None = None


class CanvasRectDrag:
    def __init__(self, *, include_end_pixel: bool, min_size: int):
        self._include_end_pixel = include_end_pixel
        self._min_size = min_size
        self._enabled = False
        self._dragging = False
        self._start: tuple[int, int] | None = None
        self._end: tuple[int, int] | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dragging(self) -> bool:
        return self._dragging

    def set_enabled(self, enabled: bool) -> None:
        self._enabled = enabled
        if not enabled:
            self.cancel_drag()

    def begin(self, x: int, y: int) -> bool:
        if not self._enabled:
            return False
        self._dragging = True
        self._start = (x, y)
        self._end = (x, y)
        return True

    def move(self, x: int, y: int) -> bool:
        if not self._dragging:
            return False
        self._end = (x, y)
        return True

    def finish(self, x: int, y: int) -> CanvasRectDragResult | None:
        if not self._dragging or self._start is None:
            return None
        sx, sy = self._start
        rect = self._normalized_rect(sx, sy, x, y)
        self._enabled = False
        self.cancel_drag()
        x0, y0, x1, y1 = rect
        if x1 - x0 <= self._min_size or y1 - y0 <= self._min_size:
            return None
        return CanvasRectDragResult(rect=rect)

    def cancel_drag(self) -> None:
        self._dragging = False
        self._start = None
        self._end = None

    def preview_rect(self) -> Rect | None:
        if not self._dragging or self._start is None or self._end is None:
            return None
        sx, sy = self._start
        ex, ey = self._end
        return sx, sy, ex, ey

    def _normalized_rect(self, sx: int, sy: int, ex: int, ey: int) -> Rect:
        x0, y0 = min(sx, ex), min(sy, ey)
        x1, y1 = max(sx, ex), max(sy, ey)
        if self._include_end_pixel:
            x1 += 1
            y1 += 1
        return x0, y0, x1, y1


class CanvasRectDragController:
    def __init__(self):
        self._selection_rect_drag = CanvasRectDrag(
            include_end_pixel=True,
            min_size=1,
        )
        self._patch_rect_drag = CanvasRectDrag(
            include_end_pixel=False,
            min_size=2,
        )
        self._show_patch_rect = True

    def set_selection_rect_mode(self, enabled: bool) -> None:
        self._selection_rect_drag.set_enabled(enabled)

    def set_patch_rect_mode(self, enabled: bool) -> None:
        self._patch_rect_drag.set_enabled(enabled)

    def set_show_patch_rect(self, show: bool) -> None:
        self._show_patch_rect = show

    def begin_selection_rect(self, x: int, y: int) -> bool:
        return self._selection_rect_drag.begin(x, y)

    def begin_patch_rect(self, x: int, y: int) -> bool:
        return self._patch_rect_drag.begin(x, y)

    def move(self, x: int, y: int) -> bool:
        return (
            self._selection_rect_drag.move(x, y)
            or self._patch_rect_drag.move(x, y)
        )

    def finish(self, x: int, y: int) -> CanvasRectDragFinish:
        if self._selection_rect_drag.dragging:
            result = self._selection_rect_drag.finish(x, y)
            return CanvasRectDragFinish(
                handled=True,
                target="selection",
                rect=result.rect if result is not None else None,
            )

        if self._patch_rect_drag.dragging:
            result = self._patch_rect_drag.finish(x, y)
            return CanvasRectDragFinish(
                handled=True,
                target="patch",
                rect=result.rect if result is not None else None,
            )

        return CanvasRectDragFinish(handled=False)

    def selection_preview_rect(self) -> Rect | None:
        return self._selection_rect_drag.preview_rect()

    def patch_preview_rect(self, layer) -> Rect | None:
        if not self._show_patch_rect or layer is None:
            return None
        rect = self._patch_rect_drag.preview_rect()
        if rect is None and layer.patch_rect:
            rect = layer.local_rect_to_canvas(layer.patch_rect)
        return rect
