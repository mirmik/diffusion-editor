import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QPen, QColor

from .layer import LayerStack, Layer, DiffusionLayer, LamaLayer, InstructLayer
from .brush import Brush, composite_stroke, erase_stroke


class Canvas(QWidget):
    mouse_moved = pyqtSignal(int, int)
    color_picked = pyqtSignal(int, int, int, int)  # r, g, b, a
    ref_rect_drawn = pyqtSignal(int, int, int, int)  # x0, y0, x1, y1
    patch_rect_drawn = pyqtSignal(int, int, int, int)  # x0, y0, x1, y1

    def __init__(self, layer_stack: LayerStack, parent=None):
        super().__init__(parent)
        self._layer_stack = layer_stack
        self._composite = None
        self._qimage = None
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._panning = False
        self._pan_start = QPointF()
        self._painting = False
        self._last_paint_pos = None
        self.brush = Brush()
        self._brush_eraser = False
        self._mask_brush_size = 50
        self._mask_brush_hardness = 0.4
        self._mask_eraser = False
        self._show_mask = True
        self._mask_overlay = None
        self._ref_rect_mode = False
        self._ref_rect_dragging = False
        self._ref_rect_start = None
        self._ref_rect_end = None
        self._show_ref_rect = True
        self._patch_rect_mode = False
        self._patch_rect_dragging = False
        self._patch_rect_start = None
        self._patch_rect_end = None
        self._show_patch_rect = True

        # Stroke buffer: accumulates brush alpha via MAX during one stroke
        self._stroke_mask = None      # (H, W) uint8 — stroke opacity
        self._stroke_color = None     # (r, g, b, a) brush color at stroke start
        self._stroke_overlay = None   # (H, W, 4) uint8 — pre-allocated for paintEvent
        self._stroke_is_eraser = False

        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._layer_stack.changed.connect(self._on_stack_changed)

    def _is_mask_layer_active(self) -> bool:
        return isinstance(self._layer_stack.active_layer, (DiffusionLayer, LamaLayer, InstructLayer))

    def _on_stack_changed(self):
        self._update_composite()
        self.update()

    def _update_composite(self):
        self._composite = np.ascontiguousarray(self._layer_stack.composite())
        self._rebuild_qimage()

    def _rebuild_qimage(self):
        """Rebuild QImage from current composite buffer (no layer recomposite)."""
        if self._composite is None:
            return
        h, w = self._composite.shape[:2]
        self._qimage = QImage(self._composite.data, w, h, w * 4, QImage.Format.Format_RGBA8888)

    def get_composite(self) -> np.ndarray | None:
        return self._composite

    def get_composite_below(self, layer: Layer) -> np.ndarray | None:
        """Composite of all layers below the given layer (excluding it and above)."""
        return np.ascontiguousarray(
            self._layer_stack.composite(exclude_layer=layer))

    def image_size(self):
        if self._layer_stack.width > 0:
            return self._layer_stack.width, self._layer_stack.height
        return None

    def fit_in_view(self):
        size = self.image_size()
        if size is None:
            return
        w, h = size
        cw, ch = self.width(), self.height()
        if w == 0 or h == 0:
            return
        scale_x = cw / w
        scale_y = ch / h
        self._zoom = min(scale_x, scale_y) * 0.95
        self._offset = QPointF(
            (cw - w * self._zoom) / 2,
            (ch - h * self._zoom) / 2,
        )

    def view_center_image(self) -> tuple[int, int]:
        cx = self.width() / 2
        cy = self.height() / 2
        return self.widget_to_image(QPointF(cx, cy))

    def widget_to_image(self, pos: QPointF) -> tuple[int, int]:
        x = (pos.x() - self._offset.x()) / self._zoom
        y = (pos.y() - self._offset.y()) / self._zoom
        return int(x), int(y)

    # --- Mask painting ---

    def set_mask_brush(self, size: int, hardness: float):
        self._mask_brush_size = size
        self._mask_brush_hardness = hardness

    def set_mask_eraser(self, eraser: bool):
        self._mask_eraser = eraser

    def set_brush_eraser(self, eraser: bool):
        self._brush_eraser = eraser

    def set_show_mask(self, show: bool):
        self._show_mask = show
        self.update()

    def set_ref_rect_mode(self, on: bool):
        self._ref_rect_mode = on
        if on:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._ref_rect_dragging = False

    def set_show_ref_rect(self, show: bool):
        self._show_ref_rect = show
        self.update()

    def set_patch_rect_mode(self, on: bool):
        self._patch_rect_mode = on
        if on:
            self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self._patch_rect_dragging = False

    def set_show_patch_rect(self, show: bool):
        self._show_patch_rect = show
        self.update()

    def _dab_mask(self, mask: np.ndarray, cx: int, cy: int):
        d = self._mask_brush_size
        if d < 1:
            return

        y, x = np.ogrid[-d / 2:d / 2, -d / 2:d / 2]
        dist = np.sqrt(x * x + y * y)
        radius = d / 2

        if self._mask_brush_hardness >= 1.0:
            alpha_mask = (dist <= radius).astype(np.float32) * 255
        else:
            inner = radius * self._mask_brush_hardness
            alpha_mask = np.clip((radius - dist) / max(radius - inner, 0.001), 0, 1) * 255

        sh, sw = alpha_mask.shape
        ih, iw = mask.shape

        x0 = cx - sw // 2
        y0 = cy - sh // 2
        sx0 = max(0, -x0)
        sy0 = max(0, -y0)
        sx1 = min(sw, iw - x0)
        sy1 = min(sh, ih - y0)
        dx0 = max(0, x0)
        dy0 = max(0, y0)
        dx1 = dx0 + (sx1 - sx0)
        dy1 = dy0 + (sy1 - sy0)

        if dx0 >= dx1 or dy0 >= dy1:
            return

        stamp_slice = alpha_mask[sy0:sy1, sx0:sx1].astype(np.uint8)
        if self._mask_eraser:
            mask[dy0:dy1, dx0:dx1] = np.minimum(
                mask[dy0:dy1, dx0:dx1], (255 - stamp_slice))
        else:
            mask[dy0:dy1, dx0:dx1] = np.maximum(mask[dy0:dy1, dx0:dx1], stamp_slice)

    def _stroke_mask_line(self, mask: np.ndarray, x0: int, y0: int, x1: int, y1: int):
        dx = x1 - x0
        dy = y1 - y0
        dist = max(abs(dx), abs(dy), 1)
        spacing = max(1, self._mask_brush_size // 4)
        steps = max(1, int(dist / spacing))
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = int(x0 + dx * t)
            y = int(y0 + dy * t)
            self._dab_mask(mask, x, y)

    # --- Stroke buffer for brush painting ---

    def _begin_stroke(self):
        """Allocate stroke buffer at the start of a brush stroke."""
        h = self._layer_stack.height
        w = self._layer_stack.width
        if h == 0 or w == 0:
            return
        self._stroke_is_eraser = self._brush_eraser
        self._stroke_color = tuple(self.brush.color)
        if self._stroke_is_eraser:
            self._stroke_mask = None
            self._stroke_overlay = None
        else:
            self._stroke_mask = np.zeros((h, w), dtype=np.uint8)
            self._stroke_overlay = np.zeros((h, w, 4), dtype=np.uint8)
            r, g, b, _a = self._stroke_color
            self._stroke_overlay[:, :, 0] = r
            self._stroke_overlay[:, :, 1] = g
            self._stroke_overlay[:, :, 2] = b

    def _end_stroke(self):
        """Apply finished brush stroke onto layer+composite. No layer recomposite."""
        layer = self._layer_stack.active_layer
        if layer is not None and self._stroke_mask is not None:
            composite_stroke(layer.image, self._stroke_mask, self._stroke_color)
            if self._composite is not None:
                composite_stroke(self._composite, self._stroke_mask, self._stroke_color)
                self._rebuild_qimage()
        self._stroke_mask = None
        self._stroke_color = None
        self._stroke_overlay = None

    # --- Eraser (direct layer+composite modification) ---

    def _composite_rect_below(self, target_layer, dy0, dy1, dx0, dx1):
        """Composite all visible layers BELOW target_layer in a small rect."""
        rh, rw = dy1 - dy0, dx1 - dx0
        result = np.zeros((rh, rw, 4), dtype=np.float32)

        def _blend_rect(layer):
            if layer is target_layer:
                return True  # stop — don't include target
            if not layer.visible or layer.opacity <= 0:
                return False
            src = layer.image[dy0:dy1, dx0:dx1].astype(np.float32)
            alpha = src[:, :, 3:4] / 255.0 * layer.opacity
            inv_alpha = 1.0 - alpha
            result[:, :, :3] = src[:, :, :3] * alpha + result[:, :, :3] * inv_alpha
            result[:, :, 3:4] = alpha * 255.0 + result[:, :, 3:4] * inv_alpha
            return False

        for layer in reversed(self._layer_stack.layers):
            if _blend_rect(layer):
                break
        return result

    def _erase_dab(self, layer, cx: int, cy: int):
        """Erase one dab on layer, recompute composite in dab rect."""
        stamp = self.brush._alpha_stamp
        sh, sw = stamp.shape[:2]
        ih, iw = layer.image.shape[:2]

        x0 = cx - sw // 2
        y0 = cy - sh // 2
        sx0, sy0 = max(0, -x0), max(0, -y0)
        sx1, sy1 = min(sw, iw - x0), min(sh, ih - y0)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)
        if dx0 >= dx1 or dy0 >= dy1:
            return

        erase = stamp[sy0:sy1, sx0:sx1] * (self.brush.color[3] / 255.0)
        # Erase from layer (straight alpha — only reduce alpha)
        la = layer.image[dy0:dy1, dx0:dx1, 3].astype(np.float32)
        layer.image[dy0:dy1, dx0:dx1, 3] = np.clip(la * (1.0 - erase), 0, 255).astype(np.uint8)

        # Recompute composite in dab rect only
        if self._composite is not None:
            below = self._composite_rect_below(layer, dy0, dy1, dx0, dx1)
            above = layer.image[dy0:dy1, dx0:dx1].astype(np.float32)
            sa = above[:, :, 3:4] / 255.0
            inv_sa = 1.0 - sa
            da = below[:, :, 3:4] / 255.0
            out_a = sa + da * inv_sa
            safe_a = np.maximum(out_a, 1.0 / 255.0)
            out_rgb = (above[:, :, :3] * sa + below[:, :, :3] * da * inv_sa) / safe_a
            self._composite[dy0:dy1, dx0:dx1, :3] = np.clip(out_rgb, 0, 255).astype(np.uint8)
            self._composite[dy0:dy1, dx0:dx1, 3:4] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)

    def _erase_stroke_line(self, layer, x0: int, y0: int, x1: int, y1: int):
        dx = x1 - x0
        dy = y1 - y0
        dist = max(abs(dx), abs(dy), 1)
        spacing = max(1, self.brush.size // 4)
        steps = max(1, int(dist / spacing))
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = int(x0 + dx * t)
            y = int(y0 + dy * t)
            self._erase_dab(layer, x, y)
        self._rebuild_qimage()

    def _draw_stroke_overlay(self, painter: QPainter):
        """Draw current in-progress stroke as overlay."""
        if self._stroke_overlay is None or self._stroke_mask is None:
            return
        h, w = self._stroke_mask.shape
        # Copy stroke alpha into overlay's alpha channel
        self._stroke_overlay[:, :, 3] = self._stroke_mask
        qimg = QImage(self._stroke_overlay.data, w, h, w * 4,
                      QImage.Format.Format_RGBA8888)
        painter.drawImage(0, 0, qimg)

    # --- Overlay rendering ---

    def _draw_mask_overlay(self, painter: QPainter, mask: np.ndarray):
        h, w = mask.shape
        self._mask_overlay = np.zeros((h, w, 4), dtype=np.uint8)
        self._mask_overlay[:, :, 0] = 255
        self._mask_overlay[:, :, 1] = 50
        self._mask_overlay[:, :, 2] = 50
        self._mask_overlay[:, :, 3] = (mask.astype(np.float32) * 0.4).astype(np.uint8)
        self._mask_overlay = np.ascontiguousarray(self._mask_overlay)
        qimg = QImage(self._mask_overlay.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        painter.drawImage(0, 0, qimg)

    # --- Events ---

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.fillRect(self.rect(), Qt.GlobalColor.darkGray)
        if self._qimage is not None:
            painter.translate(self._offset)
            painter.scale(self._zoom, self._zoom)
            painter.drawImage(0, 0, self._qimage)

            # Brush stroke in progress — draw as overlay
            if self._stroke_mask is not None:
                self._draw_stroke_overlay(painter)

            layer = self._layer_stack.active_layer
            if self._show_mask and isinstance(layer, (DiffusionLayer, LamaLayer, InstructLayer)) and layer.has_mask():
                self._draw_mask_overlay(painter, layer.mask)

            # IP-Adapter reference rectangle (blue)
            if self._show_ref_rect and isinstance(layer, DiffusionLayer):
                rect = None
                if self._ref_rect_dragging and self._ref_rect_start and self._ref_rect_end:
                    rect = self._ref_rect_start + self._ref_rect_end
                elif layer.ip_adapter_rect:
                    rect = layer.ip_adapter_rect
                if rect:
                    x0, y0, x1, y1 = rect
                    pen = QPen(QColor(50, 120, 255, 200), 2.0 / self._zoom)
                    painter.setPen(pen)
                    painter.setBrush(QColor(50, 120, 255, 40))
                    painter.drawRect(x0, y0, x1 - x0, y1 - y0)

            # Manual patch rectangle (green)
            if self._show_patch_rect and isinstance(layer, (DiffusionLayer, InstructLayer)):
                rect = None
                if self._patch_rect_dragging and self._patch_rect_start and self._patch_rect_end:
                    rect = self._patch_rect_start + self._patch_rect_end
                elif layer.manual_patch_rect:
                    rect = layer.manual_patch_rect
                if rect:
                    x0, y0, x1, y1 = rect
                    pen = QPen(QColor(50, 200, 80, 200), 2.0 / self._zoom)
                    painter.setPen(pen)
                    painter.setBrush(QColor(50, 200, 80, 40))
                    painter.drawRect(x0, y0, x1 - x0, y1 - y0)

        painter.end()

    def wheelEvent(self, event):
        if self.image_size() is None:
            return
        pos = event.position()
        old_img = self.widget_to_image(pos)
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom *= factor
        self._zoom = max(0.01, min(self._zoom, 100.0))
        new_widget_x = pos.x() - old_img[0] * self._zoom
        new_widget_y = pos.y() - old_img[1] * self._zoom
        self._offset = QPointF(new_widget_x, new_widget_y)
        self.update()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_BracketRight:
            self.brush.set_size(self.brush.size + 5)
            self.update()
        elif event.key() == Qt.Key.Key_BracketLeft:
            self.brush.set_size(self.brush.size - 5)
            self.update()
        else:
            super().keyPressEvent(event)

    def _pick_color(self, event):
        """Eyedropper: sample color from composite under cursor."""
        if self._composite is None:
            return
        ix, iy = self.widget_to_image(event.position())
        h, w = self._composite.shape[:2]
        if 0 <= ix < w and 0 <= iy < h:
            r, g, b, a = self._composite[iy, ix]
            self.color_picked.emit(int(r), int(g), int(b), int(a))

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position() - self._offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            # Ctrl+Click = eyedropper
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._pick_color(event)
                return
            layer = self._layer_stack.active_layer
            if layer is None:
                return
            # Patch rect drawing mode
            if self._patch_rect_mode and isinstance(layer, (DiffusionLayer, InstructLayer)):
                ix, iy = self.widget_to_image(event.position())
                self._patch_rect_dragging = True
                self._patch_rect_start = (ix, iy)
                self._patch_rect_end = (ix, iy)
                return
            # Ref rect drawing mode
            if self._ref_rect_mode and self._is_mask_layer_active():
                ix, iy = self.widget_to_image(event.position())
                self._ref_rect_dragging = True
                self._ref_rect_start = (ix, iy)
                self._ref_rect_end = (ix, iy)
                return
            self._painting = True
            ix, iy = self.widget_to_image(event.position())
            if self._is_mask_layer_active():
                self._dab_mask(layer.mask, ix, iy)
            else:
                self._begin_stroke()
                if self._stroke_is_eraser:
                    self._erase_dab(layer, ix, iy)
                    self._rebuild_qimage()
                elif self._stroke_mask is not None:
                    self.brush.dab_to_mask(self._stroke_mask, ix, iy)
            self._last_paint_pos = (ix, iy)
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            if self._patch_rect_dragging:
                ix, iy = self.widget_to_image(event.position())
                sx, sy = self._patch_rect_start
                x0, y0 = min(sx, ix), min(sy, iy)
                x1, y1 = max(sx, ix), max(sy, iy)
                self._patch_rect_dragging = False
                self._patch_rect_start = None
                self._patch_rect_end = None
                self._patch_rect_mode = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                if x1 - x0 > 2 and y1 - y0 > 2:
                    self.patch_rect_drawn.emit(x0, y0, x1, y1)
                self.update()
                return
            if self._ref_rect_dragging:
                ix, iy = self.widget_to_image(event.position())
                sx, sy = self._ref_rect_start
                x0, y0 = min(sx, ix), min(sy, iy)
                x1, y1 = max(sx, ix), max(sy, iy)
                self._ref_rect_dragging = False
                self._ref_rect_start = None
                self._ref_rect_end = None
                self._ref_rect_mode = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                if x1 - x0 > 2 and y1 - y0 > 2:
                    self.ref_rect_drawn.emit(x0, y0, x1, y1)
                self.update()
                return
            if self._painting:
                if self._stroke_mask is not None:
                    self._end_stroke()  # applies to layer+composite, rebuilds QImage
                elif self._stroke_is_eraser:
                    self._rebuild_qimage()  # eraser already patched composite
                self.update()
            self._painting = False
            self._last_paint_pos = None

    def mouseMoveEvent(self, event):
        if self._panning:
            self._offset = event.position() - self._pan_start
            self.update()
        elif self._patch_rect_dragging:
            ix, iy = self.widget_to_image(event.position())
            self._patch_rect_end = (ix, iy)
            self.update()
        elif self._ref_rect_dragging:
            ix, iy = self.widget_to_image(event.position())
            self._ref_rect_end = (ix, iy)
            self.update()
        elif self._painting:
            layer = self._layer_stack.active_layer
            if layer is None:
                return
            ix, iy = self.widget_to_image(event.position())
            if self._is_mask_layer_active():
                if self._last_paint_pos:
                    lx, ly = self._last_paint_pos
                    self._stroke_mask_line(layer.mask, lx, ly, ix, iy)
                else:
                    self._dab_mask(layer.mask, ix, iy)
            else:
                if self._stroke_is_eraser:
                    if self._last_paint_pos:
                        lx, ly = self._last_paint_pos
                        self._erase_stroke_line(layer, lx, ly, ix, iy)
                    else:
                        self._erase_dab(layer, ix, iy)
                elif self._stroke_mask is not None:
                    if self._last_paint_pos:
                        lx, ly = self._last_paint_pos
                        self.brush.stroke_to_mask(
                            self._stroke_mask, lx, ly, ix, iy)
                    else:
                        self.brush.dab_to_mask(self._stroke_mask, ix, iy)
            self._last_paint_pos = (ix, iy)
            self.update()

        if self.image_size() is not None:
            ix, iy = self.widget_to_image(event.position())
            self.mouse_moved.emit(ix, iy)

    def resizeEvent(self, event):
        super().resizeEvent(event)
