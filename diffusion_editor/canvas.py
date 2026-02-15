import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QPen, QColor

from .layer import LayerStack
from .brush import Brush


class Canvas(QWidget):
    mouse_moved = pyqtSignal(int, int)

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
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._layer_stack.changed.connect(self._on_stack_changed)

    def _on_stack_changed(self):
        self._update_composite()
        self.update()

    def _update_composite(self):
        self._composite = np.ascontiguousarray(self._layer_stack.composite())
        h, w = self._composite.shape[:2]
        self._qimage = QImage(self._composite.data, w, h, w * 4, QImage.Format.Format_RGBA8888)

    def get_composite(self) -> np.ndarray | None:
        return self._composite

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

    def widget_to_image(self, pos: QPointF) -> tuple[int, int]:
        x = (pos.x() - self._offset.x()) / self._zoom
        y = (pos.y() - self._offset.y()) / self._zoom
        return int(x), int(y)

    def _brush_cursor_rect(self):
        """Area around cursor to repaint for brush outline."""
        return self.rect()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        painter.fillRect(self.rect(), Qt.GlobalColor.darkGray)
        if self._qimage is not None:
            painter.translate(self._offset)
            painter.scale(self._zoom, self._zoom)
            painter.drawImage(0, 0, self._qimage)
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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position() - self._offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            layer = self._layer_stack.active_layer
            if layer is None:
                return
            self._painting = True
            ix, iy = self.widget_to_image(event.position())
            self.brush.dab(layer.image, ix, iy)
            self._last_paint_pos = (ix, iy)
            self._layer_stack.changed.emit()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
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
            if self._last_paint_pos:
                lx, ly = self._last_paint_pos
                self.brush.stroke(layer.image, lx, ly, ix, iy)
            else:
                self.brush.dab(layer.image, ix, iy)
            self._last_paint_pos = (ix, iy)
            self._layer_stack.changed.emit()

        if self.image_size() is not None:
            ix, iy = self.widget_to_image(event.position())
            self.mouse_moved.emit(ix, iy)

    def resizeEvent(self, event):
        super().resizeEvent(event)
