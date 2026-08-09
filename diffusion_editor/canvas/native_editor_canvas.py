"""termin-gui-native Canvas adapter for the editor canvas controller."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
from tcbase import log
from termin.gui_native import (
    CanvasTextureLayer,
    CursorIntent,
    DynamicTextureOwnership,
    Point,
    PointerEventType,
    Rect,
    Size,
    SrgbColor,
    TcDocument,
)
from tgfx import TextureEncoding, Tgfx2Context

from ..document.layer import Layer
from ..document.change_event import DocumentChangeEvent
from ..document.layer_stack import LayerStack
from .editor_canvas_controller import CanvasAnnotation, EditorCanvasController


class NativeEditorCanvas:
    """Owns a native Canvas and its image/overlay texture leases."""

    def __init__(
            self,
            document: TcDocument,
            layer_stack: LayerStack,
            *,
            lease_factory: Callable[[], Any],
            graphics_owner,
            request_repaint: Callable[[], None],
            gpu_compositing: bool = True) -> None:
        self._closed = False
        self._document = document
        self._request_repaint = request_repaint
        self._graphics_owner = graphics_owner
        self._graphics = None
        if gpu_compositing:
            try:
                self._graphics = Tgfx2Context.from_runtime(graphics_owner)
            except RuntimeError as exc:
                log.info(
                    "[native-canvas] GPU compositing unavailable; "
                    f"using owned CPU textures: {exc}"
                )
                gpu_compositing = False
        self.canvas = document.create_canvas()
        self.canvas.widget.stable_id = "diffusion-editor.canvas"
        self.widget = self.canvas.widget
        # Canvas itself reserves pointer capture for middle-button panning.
        # A zero-size live relay owns capture for edit gestures and forwards
        # raw window coordinates back through Canvas image-space conversion.
        self._capture_relay = document.create_scene_view()
        self._capture_relay.widget.stable_id = (
            "diffusion-editor.canvas.capture-relay"
        )
        self._capture_relay.widget.min_size = Size(0.0, 0.0)
        self._capture_relay.widget.preferred_size = Size(0.0, 0.0)
        self._capture_relay.set_pointer_handler(self._on_captured_pointer)
        self._image_lease = lease_factory()
        self._overlay_lease = lease_factory()
        self._image_lease.bind_canvas(self.canvas, CanvasTextureLayer.IMAGE)
        self._overlay_lease.bind_canvas(self.canvas, CanvasTextureLayer.OVERLAY)

        self.controller = EditorCanvasController(
            layer_stack,
            gpu_compositing=gpu_compositing,
            graphics=self._graphics,
            set_image=self._set_image,
            set_overlay=self._set_overlay,
            update_image_region=self._update_image_region,
            update_overlay_region=self._update_overlay_region,
            request_repaint=request_repaint,
            set_cursor=self._set_cursor,
        )
        self._layer_stack = layer_stack
        self._stack_subscription = layer_stack.subscribe(self._on_stack_changed)

        self._pointer_connection = self.canvas.connect_pointer_input(
            self._on_pointer_input)
        self.canvas.set_paint_callback(self._paint_annotations)
        self.controller.refresh()
        self._sync_gpu_image()

    @property
    def image_lease(self):
        return self._image_lease

    @property
    def overlay_lease(self):
        return self._overlay_lease

    @property
    def pointer_capture_widget(self):
        return self._capture_relay.widget

    def fit_in_view(self) -> None:
        self._require_open()
        self.canvas.fit_in_view()
        self._request_repaint()

    def get_composite_below(self, layer: Layer) -> np.ndarray | None:
        self._require_open()
        return self.controller.get_composite_below(layer)

    def view_center_image(self) -> tuple[int, int]:
        self._require_open()
        bounds = self.widget.bounds
        point = self.canvas.widget_to_image(Point(
            bounds.x + bounds.width * 0.5,
            bounds.y + bounds.height * 0.5,
        ))
        return int(point.x), int(point.y)

    def dispatch_shortcut(self, key: int, modifiers: int) -> bool:
        if modifiers != 0:
            return False
        if key == ord("["):
            self.controller.adjust_brush_size(-5)
            return True
        if key == ord("]"):
            self.controller.adjust_brush_size(5)
            return True
        return False

    def cancel_pointer_interaction(self) -> None:
        """Cancel an edit and release the relay's document-wide capture."""
        controller = getattr(self, "controller", None)
        cancel = getattr(controller, "pointer_cancel", None)
        first_error: BaseException | None = None
        first_traceback = None
        try:
            if cancel is not None:
                cancel()
        except BaseException as exc:
            first_error = exc
            first_traceback = exc.__traceback__
        document = getattr(self, "_document", None)
        relay = getattr(self, "_capture_relay", None)
        try:
            if (
                    document is not None
                    and relay is not None
                    and document.pointer_capture == relay.handle):
                document.release_pointer_capture(relay.handle)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
                first_traceback = exc.__traceback__
        if first_error is not None:
            raise first_error.with_traceback(first_traceback)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        errors: list[tuple[BaseException, object]] = []

        def cleanup(callback) -> None:
            try:
                callback()
            except BaseException as exc:
                errors.append((exc, exc.__traceback__))

        subscription = getattr(self, "_stack_subscription", None)
        if subscription is not None:
            cleanup(subscription.unsubscribe)
        cleanup(self.cancel_pointer_interaction)
        document = getattr(self, "_document", None)
        relay = getattr(self, "_capture_relay", None)
        if document is not None and relay is not None:
            if document.pointer_capture == relay.handle:
                cleanup(lambda: document.release_pointer_capture(relay.handle))
            cleanup(lambda: relay.set_pointer_handler(None))
        # The current Python binding advertises ``None`` but rejects it at
        # runtime, so detach application state with a capture-free no-op.
        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            cleanup(
                lambda: canvas.set_paint_callback(lambda _context: None))
        # The leases are released while their renderer and Canvas are alive.
        overlay_lease = getattr(self, "_overlay_lease", None)
        if overlay_lease is not None:
            cleanup(overlay_lease.close)
        image_lease = getattr(self, "_image_lease", None)
        if image_lease is not None:
            cleanup(image_lease.close)
        controller = getattr(self, "controller", None)
        if controller is not None:
            cleanup(controller.dispose)
        if errors:
            error, traceback = errors[0]
            raise error.with_traceback(traceback)

    def _on_stack_changed(self, event: DocumentChangeEvent) -> None:
        if self._closed:
            return
        self.controller.handle_document_change(event)
        self._sync_gpu_image()

    def _set_image(self, image: np.ndarray | None) -> None:
        if self._closed:
            return
        self._replace_owned(self._image_lease, image)

    def _set_overlay(self, overlay: np.ndarray | None) -> None:
        if self._closed:
            return
        self._replace_owned(self._overlay_lease, overlay)

    def _update_image_region(
            self, x: int, y: int, image: np.ndarray) -> None:
        if self._closed:
            return
        if (
                self._image_lease.empty
                or self._image_lease.width != self._layer_stack.width
                or self._image_lease.height != self._layer_stack.height
                or self._image_lease.ownership != DynamicTextureOwnership.OWNED):
            self._replace_owned(
                self._image_lease,
                self.controller.composite_bridge.composite,
            )
        else:
            self._image_lease.update_region_rgba8(
                x, y, np.ascontiguousarray(image))
            self._request_repaint()

    def _update_overlay_region(
            self, x: int, y: int, overlay: np.ndarray) -> None:
        if self._closed:
            return
        if (
                self._overlay_lease.empty
                or self._overlay_lease.width != self._layer_stack.width
                or self._overlay_lease.height != self._layer_stack.height
                or self._overlay_lease.ownership != DynamicTextureOwnership.OWNED):
            self._replace_owned(
                self._overlay_lease,
                self.controller.overlay_bridge.overlay,
            )
        else:
            self._overlay_lease.update_region_rgba8(
                x, y, np.ascontiguousarray(overlay))
            self._request_repaint()

    def _replace_owned(self, lease, data: np.ndarray | None) -> None:
        if data is None:
            if not lease.empty:
                lease.clear()
            return

        pixels = np.ascontiguousarray(data, dtype=np.uint8)
        height, width = pixels.shape[:2]
        if (
                lease.ownership == DynamicTextureOwnership.OWNED
                and lease.width == width
                and lease.height == height):
            lease.update_region_rgba8(0, 0, pixels)
            return

        if not lease.empty:
            lease.clear()
        lease.set_rgba8(pixels, TextureEncoding.SRGB)

    def _sync_gpu_image(self) -> None:
        bridge = self.controller.composite_bridge
        if not bridge.using_gpu:
            return
        texture = bridge.display_tex
        width, height = bridge.display_size()
        if texture is None or width <= 0 or height <= 0:
            self._replace_owned(self._image_lease, None)
            return
        if not self._image_lease.empty:
            self._image_lease.clear()
        self._image_lease.borrow(self._graphics_owner, texture)
        self._request_repaint()

    def _on_pointer_input(self, image_point: Point, event) -> None:
        if self._closed:
            return
        if event.type == PointerEventType.Down:
            self.controller.pointer_down(
                image_point.x,
                image_point.y,
                int(event.button),
                int(event.modifiers),
            )
            if self.controller.pointer_interaction_active:
                if not self._document.set_pointer_capture(
                        self._capture_relay.handle):
                    self.controller.pointer_cancel()
        elif event.type == PointerEventType.Move:
            self.controller.pointer_move(image_point.x, image_point.y)
        elif event.type == PointerEventType.Up:
            self.controller.pointer_up(image_point.x, image_point.y)
            self._sync_gpu_image()
        elif event.type == PointerEventType.Cancel:
            self.cancel_pointer_interaction()
            self._sync_gpu_image()

    def _on_captured_pointer(self, _world_point: Point, event) -> bool:
        if self._closed:
            return False
        image_point = self.canvas.widget_to_image(Point(event.x, event.y))
        try:
            self._on_pointer_input(image_point, event)
        finally:
            if event.type in (PointerEventType.Up, PointerEventType.Cancel):
                if self._document.pointer_capture == self._capture_relay.handle:
                    self._document.release_pointer_capture(
                        self._capture_relay.handle)
        return True

    def _paint_annotations(self, context) -> None:
        if self._closed:
            return
        for annotation in self.controller.annotations():
            self._paint_annotation(context, annotation)

    def _set_cursor(self, cursor: str) -> None:
        if self._closed:
            return
        self.widget.cursor_intent = (
            CursorIntent.Crosshair
            if cursor == "crosshair"
            else CursorIntent.Default
        )
        self._request_repaint()

    def _paint_annotation(
            self, context, annotation: CanvasAnnotation) -> None:
        x0, y0, x1, y1 = annotation.rect
        p0 = self.canvas.image_to_widget(Point(x0, y0))
        p1 = self.canvas.image_to_widget(Point(x1, y1))
        rect = Rect(
            min(p0.x, p1.x),
            min(p0.y, p1.y),
            abs(p1.x - p0.x),
            abs(p1.y - p0.y),
        )
        if rect.width <= 0 or rect.height <= 0:
            return
        if annotation.kind == "active-layer":
            context.stroke_rect(rect, SrgbColor(0.0, 0.0, 0.0, 0.85), 3.0)
            context.stroke_rect(rect, SrgbColor(1.0, 1.0, 1.0, 0.95), 1.0)
        elif annotation.kind == "selection":
            context.fill_rect(rect, SrgbColor(0.0, 0.8, 1.0, 0.12))
            context.stroke_rect(rect, SrgbColor(0.0, 0.8, 1.0, 0.7), 2.0)
        elif annotation.kind == "patch":
            context.stroke_rect(rect, SrgbColor(0.2, 0.78, 0.31, 0.8), 2.0)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("native editor canvas is closed")
