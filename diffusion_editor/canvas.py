import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QPointF, pyqtSignal
from PyQt6.QtGui import QPainter, QImage, QPixmap, QTransform


class Canvas(QWidget):
    mouse_moved = pyqtSignal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._image = None  # numpy array (H, W, 4) RGBA uint8
        self._qimage = None
        self._zoom = 1.0
        self._offset = QPointF(0, 0)
        self._panning = False
        self._pan_start = QPointF()
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

    def set_image(self, arr: np.ndarray):
        if arr.ndim == 2:
            arr = np.stack([arr, arr, arr, np.full_like(arr, 255)], axis=-1)
        elif arr.shape[2] == 3:
            alpha = np.full((*arr.shape[:2], 1), 255, dtype=np.uint8)
            arr = np.concatenate([arr, alpha], axis=-1)
        self._image = np.ascontiguousarray(arr.astype(np.uint8))
        h, w, _ = self._image.shape
        self._qimage = QImage(self._image.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
        self.fit_in_view()
        self.update()

    def get_image(self) -> np.ndarray | None:
        return self._image

    def image_size(self):
        if self._image is not None:
            h, w = self._image.shape[:2]
            return w, h
        return None

    def fit_in_view(self):
        if self._image is None:
            return
        h, w = self._image.shape[:2]
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
        if self._image is None:
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

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = True
            self._pan_start = event.position() - self._offset
            self.setCursor(Qt.CursorShape.ClosedHandCursor)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def mouseMoveEvent(self, event):
        if self._panning:
            self._offset = event.position() - self._pan_start
            self.update()
        if self._image is not None:
            ix, iy = self.widget_to_image(event.position())
            self.mouse_moved.emit(ix, iy)

    def resizeEvent(self, event):
        super().resizeEvent(event)
