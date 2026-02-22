from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QColorDialog,
)
from PyQt6.QtGui import QPixmap, QIcon, QColor
from PyQt6.QtCore import Qt, pyqtSignal

from .brush import Brush
from .slider_edit import SliderEdit


class BrushPanel(QDockWidget):
    eraser_toggled = pyqtSignal(bool)

    def __init__(self, brush: Brush, parent=None):
        super().__init__("Brush", parent)
        self._brush = brush
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setMinimumWidth(200)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Color + Eraser ---
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color:"))
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(40, 24)
        color_row.addWidget(self._color_btn)
        self._eraser_btn = QPushButton("Eraser")
        self._eraser_btn.setCheckable(True)
        color_row.addWidget(self._eraser_btn)
        color_row.addStretch()
        layout.addLayout(color_row)

        # --- Size ---
        layout.addWidget(QLabel("Size:"))
        self._size_slider = SliderEdit(1, 500, brush.size, decimals=0, step=1)
        layout.addWidget(self._size_slider)

        # --- Hardness ---
        layout.addWidget(QLabel("Hardness:"))
        self._hard_slider = SliderEdit(0.0, 1.0, brush.hardness, decimals=2, step=0.05)
        layout.addWidget(self._hard_slider)

        layout.addStretch()
        self.setWidget(container)

        # --- Connections ---
        self._color_btn.clicked.connect(self._pick_color)
        self._size_slider.valueChanged.connect(self._on_size_changed)
        self._hard_slider.valueChanged.connect(self._on_hard_changed)
        self._eraser_btn.toggled.connect(self.eraser_toggled.emit)

        self._update_color_icon()

    def _update_color_icon(self):
        r, g, b, _ = self._brush.color
        px = QPixmap(32, 16)
        px.fill(QColor(r, g, b))
        self._color_btn.setIcon(QIcon(px))

    def _pick_color(self):
        r, g, b, a = self._brush.color
        initial = QColor(r, g, b, a)
        color = QColorDialog.getColor(
            initial, self, "Brush Color",
            QColorDialog.ColorDialogOption.ShowAlphaChannel,
        )
        if color.isValid():
            self._brush.set_color(color.red(), color.green(), color.blue(), color.alpha())
            self._update_color_icon()

    def _on_size_changed(self, value):
        self._brush.set_size(int(value))

    def _on_hard_changed(self, value):
        self._brush.set_hardness(value)

    def sync_from_brush(self):
        self._size_slider.setValue(self._brush.size)
        self._hard_slider.setValue(self._brush.hardness)
        self._update_color_icon()
