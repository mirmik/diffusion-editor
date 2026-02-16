import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QPen, QColor

from .layer import LayerStack, DiffusionLayer
from .brush import Brush


class Canvas(QWidget):
    mouse_moved = pyqtSignal(int, int)
    color_picked = pyqtSignal(int, int, int, int)  # r, g, b, a

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
        self._mask_brush_size = 50
        self._mask_brush_hardness = 0.8
        self._mask_eraser = False
        self._show_mask = True
        self._mask_overlay = None
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._layer_stack.changed.connect(self._on_stack_changed)

    def _is_diffusion_active(self) -> bool:
        return isinstance(self._layer_stack.active_layer, DiffusionLayer)

    def _on_stack_changed(self):
        self._update_composite()
        self.update()

    def _update_composite(self):
        self._composite = np.ascontiguousarray(self._layer_stack.composite())
        h, w = self._composite.shape[:2]
        self._qimage = QImage(self._composite.data, w, h, w * 4, QImage.Format.Format_RGBA8888)

    def get_composite(self) -> np.ndarray | None:
        return self._composite

    def get_composite_below(self, layer_index: int) -> np.ndarray | None:
        """Composite of layers below the given index (excluding it and above)."""
        return np.ascontiguousarray(
            self._layer_stack.composite(exclude_above=layer_index))

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

    def set_show_mask(self, show: bool):
        self._show_mask = show
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
            mask[dy0:dy1, dx0:dx1] = np.clip(
                mask[dy0:dy1, dx0:dx1].astype(np.int16) - stamp_slice.astype(np.int16),
                0, 255).astype(np.uint8)
        else:
            mask[dy0:dy1, dx0:dx1] = np.maximum(mask[dy0:dy1, dx0:dx1], stamp_slice)

    def _stroke_mask(self, mask: np.ndarray, x0: int, y0: int, x1: int, y1: int):
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

    # --- Mask overlay rendering ---

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

            layer = self._layer_stack.active_layer
            if self._show_mask and isinstance(layer, DiffusionLayer) and layer.has_mask():
                self._draw_mask_overlay(painter, layer.mask)

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

    def _paint_brush(self, layer, ix, iy):
        """Dab on both layer and composite buffer (fast, no recomposite)."""
        self.brush.dab(layer.image, ix, iy)
        if self._composite is not None:
            self.brush.dab(self._composite, ix, iy)
        self.update()

    def _stroke_brush(self, layer, x0, y0, x1, y1):
        """Stroke on both layer and composite buffer (fast, no recomposite)."""
        self.brush.stroke(layer.image, x0, y0, x1, y1)
        if self._composite is not None:
            self.brush.stroke(self._composite, x0, y0, x1, y1)
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position() - self._offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            # Alt+Click = eyedropper
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self._pick_color(event)
                return
            layer = self._layer_stack.active_layer
            if layer is None:
                return
            self._painting = True
            ix, iy = self.widget_to_image(event.position())
            if self._is_diffusion_active():
                self._dab_mask(layer.mask, ix, iy)
                self.update()
            else:
                self._paint_brush(layer, ix, iy)
            self._last_paint_pos = (ix, iy)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            if self._painting:
                self._on_stack_changed()  # full recomposite
                self._layer_stack.changed.emit()
            self._painting = False
            self._last_paint_pos = None

    def mouseMoveEvent(self, event):
        if self._panning:
            self._offset = event.position() - self._pan_start
            self.update()
        elif self._painting:
            layer = self._layer_stack.active_layer
            if layer is None:
                return
            ix, iy = self.widget_to_image(event.position())
            if self._is_diffusion_active():
                if self._last_paint_pos:
                    lx, ly = self._last_paint_pos
                    self._stroke_mask(layer.mask, lx, ly, ix, iy)
                else:
                    self._dab_mask(layer.mask, ix, iy)
                self.update()
            else:
                if self._last_paint_pos:
                    lx, ly = self._last_paint_pos
                    self._stroke_brush(layer, *self._last_paint_pos, ix, iy)
                else:
                    self._paint_brush(layer, ix, iy)
            self._last_paint_pos = (ix, iy)

        if self.image_size() is not None:
            ix, iy = self.widget_to_image(event.position())
            self.mouse_moved.emit(ix, iy)

    def resizeEvent(self, event):
        super().resizeEvent(event)
