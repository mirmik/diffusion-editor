"""Multi-gesture transform preview, committed once through DocumentService."""

from __future__ import annotations

from ..document.raster_transform import (
    ApplyRasterTransformCommand, RasterTransform, TransformPreviewRenderer,
)
from .canvas_geometry import union_rect


def rect_handles(rect):
    x0, y0, x1, y1 = rect
    cx, cy = (x0+x1)/2, (y0+y1)/2
    return {"nw": (x0, y0), "n": (cx, y0), "ne": (x1, y0),
            "e": (x1, cy), "se": (x1, y1), "s": (cx, y1),
            "sw": (x0, y1), "w": (x0, cy)}


class CanvasTransformController:
    def __init__(self, stack, document, canvas, *, on_committed=lambda: None):
        self.stack, self.document, self.canvas = stack, document, canvas
        self.on_committed = on_committed
        self.on_changed = lambda: None
        self.session = None
        self.drag = None
        self.keep_aspect = True
        self.hit_radius = 8.0
        self.message = ""
        self.error = None
        self._preview_renderer = None
        self._preview_image = None
        self._preview_rect = None
        self._remove_barrier = document.add_before_mutation_listener(self.cancel)
        self._subscription = stack.subscribe(self._document_changed)
        canvas.transform = self

    @property
    def active(self):
        return self.session is not None

    @property
    def dragging(self):
        return self.drag is not None

    def begin(self, target="auto"):
        self.document.prepare_mutation()
        self.canvas.pointer_cancel()
        try:
            session = RasterTransform(self.stack, target)
        except ValueError as exc:
            self.message = self.error = str(exc)
            self.on_changed()
            return False
        self.session = session
        self.error = None
        self.canvas.set_selection_mode(False)
        self.canvas.set_selection_rect_mode(False)
        self.canvas.set_patch_rect_mode(False)
        self.message = "Drag inside to move; drag handles to resize"
        self._render()
        return True

    def _document_changed(self, _event):
        if self.active:
            self.cancel()
        else:
            self.on_changed()

    def set_rect(self, rect):
        if not self.active:
            return
        previous_rect = self.session.rect
        try:
            self.session.set_rect(rect)
        except ValueError as exc:
            self.message = self.error = str(exc)
            self.on_changed()
            return
        self.message = "Enter: apply · Esc: cancel"
        self.error = None
        if self.session.rect == previous_rect:
            return
        self._render()

    def set_dimension(self, axis, value):
        if not self.active:
            return
        x0, y0, x1, y1 = self.session.rect
        w, h = x1-x0, y1-y0
        value = max(1, round(value))
        if axis == "width":
            h = max(1, round(h*value/w)) if self.keep_aspect else h
            w = value
        else:
            w = max(1, round(w*value/h)) if self.keep_aspect else w
            h = value
        self.set_rect((x0, y0, x0+w, y0+h))

    def pointer_down(self, x, y):
        if not self.active:
            return
        rect = self.session.rect
        handles = rect_handles(rect)
        handle = min(handles, key=lambda key: (handles[key][0]-x)**2 + (handles[key][1]-y)**2)
        hx, hy = handles[handle]
        if abs(hx-x) > self.hit_radius or abs(hy-y) > self.hit_radius:
            if not (rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]):
                return
            handle = "move"
        self.drag = (x, y, rect, handle)

    def pointer_move(self, x, y):
        if not self.drag or not self.active:
            return
        sx, sy, rect, handle = self.drag
        x0, y0, x1, y1 = rect
        dx, dy = round(x-sx), round(y-sy)
        if handle == "move":
            self.set_rect((x0+dx, y0+dy, x1+dx, y1+dy))
            return
        nx0, ny0, nx1, ny1 = x0, y0, x1, y1
        if "w" in handle: nx0 = min(x1-1, x0+dx)
        if "e" in handle: nx1 = max(x0+1, x1+dx)
        if "n" in handle: ny0 = min(y1-1, y0+dy)
        if "s" in handle: ny1 = max(y0+1, y1+dy)
        if self.keep_aspect:
            ratio = (x1-x0)/(y1-y0)
            vertical = (handle in ("n", "s") or
                        (len(handle) == 2 and abs(dy)/(y1-y0) > abs(dx)/(x1-x0)))
            if vertical:
                width = max(1, round((ny1-ny0)*ratio))
                if "w" in handle: nx0 = nx1-width
                elif "e" in handle: nx1 = nx0+width
                else:
                    nx0 = round((x0+x1-width)/2)
                    nx1 = nx0+width
            else:
                height = max(1, round((nx1-nx0)/ratio))
                if "n" in handle: ny0 = ny1-height
                elif "s" in handle: ny1 = ny0+height
                else:
                    ny0 = round((y0+y1-height)/2)
                    ny1 = ny0+height
        self.set_rect((nx0, ny0, nx1, ny1))

    def pointer_up(self, x, y):
        self.pointer_move(x, y)
        self.drag = None

    def cancel_drag(self):
        if self.drag and self.active:
            rect = self.drag[2]
            self.drag = None
            self.set_rect(rect)

    def apply(self):
        if not self.active:
            return
        session = self.session
        self.session, self.drag = None, None
        self.error = None
        try:
            self.document.execute(ApplyRasterTransformCommand(session))
            self.message = "Transform applied" if session.changed else "No changes"
            self.on_committed()
        finally:
            self._clear_preview_cache()
            self.canvas.composite_bridge.show_preview(None)
            self.canvas.refresh()
            self.on_changed()

    def cancel(self):
        if not self.active:
            if self.error is not None:
                self.error = None
                self.on_changed()
            return
        self.session, self.drag = None, None
        self.error = None
        self.message = "Transform cancelled"
        self._clear_preview_cache()
        self.canvas.composite_bridge.show_preview(None)
        self.canvas.refresh()
        self.on_changed()

    def _render(self):
        session = self.session
        if self._preview_renderer is None:
            self._preview_renderer = TransformPreviewRenderer(self.stack, session.layer, session)
            self._preview_image = self.stack.composite()
            self.canvas.overlay_bridge.clear_output()
            self.canvas.composite_bridge.show_preview(self._preview_image)
        else:
            renderer = self._preview_renderer
            old, new = self._preview_rect, session.rect
            # Separated positions need two small updates, not the empty path
            # between them. The hole at the source is already in the preview.
            overlaps = (old[0] <= new[2] and new[0] <= old[2]
                        and old[1] <= new[3] and new[1] <= old[3])
            regions = [union_rect(old, new)] if overlaps else [old, new]
            for dirty in regions:
                x0, y0 = max(0, dirty[0]), max(0, dirty[1])
                x1, y1 = min(self.stack.width, dirty[2]), min(self.stack.height, dirty[3])
                if x1 <= x0 or y1 <= y0:
                    continue
                dirty = (x0, y0, x1, y1)
                renderer.set_replacement(session, dirty)
                self._preview_image[y0:y1, x0:x1] = renderer.composite_preview_rect(dirty)
                self.canvas.composite_bridge.show_preview(self._preview_image, dirty)
        self._preview_rect = session.rect
        self.canvas._request_repaint()
        self.on_changed()

    def _clear_preview_cache(self):
        self._preview_renderer = self._preview_image = self._preview_rect = None

    def close(self):
        self.on_changed = lambda: None
        self.cancel()
        self._remove_barrier()
        self._subscription.unsubscribe()
        self.canvas.transform = None
