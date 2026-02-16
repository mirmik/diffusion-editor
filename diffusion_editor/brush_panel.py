from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSlider, QSpinBox, QLabel, QGroupBox,
    QColorDialog,
)
from PyQt6.QtGui import QPixmap, QIcon, QColor
from PyQt6.QtCore import Qt

from .brush import Brush


class BrushPanel(QDockWidget):
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

        # --- Color ---
        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color:"))
        self._color_btn = QPushButton()
        self._color_btn.setFixedSize(40, 24)
        color_row.addWidget(self._color_btn)
        color_row.addStretch()
        layout.addLayout(color_row)

        # --- Size ---
        layout.addWidget(QLabel("Size:"))
        size_row = QHBoxLayout()
        self._size_slider = QSlider(Qt.Orientation.Horizontal)
        self._size_slider.setRange(1, 500)
        self._size_slider.setValue(brush.size)
        self._size_label = QLabel(str(brush.size))
        self._size_label.setFixedWidth(30)
        size_row.addWidget(self._size_slider)
        size_row.addWidget(self._size_label)
        layout.addLayout(size_row)

        # --- Hardness ---
        layout.addWidget(QLabel("Hardness:"))
        hard_row = QHBoxLayout()
        self._hard_slider = QSlider(Qt.Orientation.Horizontal)
        self._hard_slider.setRange(0, 100)
        self._hard_slider.setValue(int(brush.hardness * 100))
        self._hard_label = QLabel(f"{brush.hardness:.2f}")
        self._hard_label.setFixedWidth(30)
        hard_row.addWidget(self._hard_slider)
        hard_row.addWidget(self._hard_label)
        layout.addLayout(hard_row)

        layout.addStretch()
        self.setWidget(container)

        # --- Connections ---
        self._color_btn.clicked.connect(self._pick_color)
        self._size_slider.valueChanged.connect(self._on_size_changed)
        self._hard_slider.valueChanged.connect(self._on_hard_changed)

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
        self._brush.set_size(value)
        self._size_label.setText(str(value))

    def _on_hard_changed(self, value):
        h = value / 100.0
        self._brush.set_hardness(h)
        self._hard_label.setText(f"{h:.2f}")

    def sync_from_brush(self):
        self._size_slider.setValue(self._brush.size)
        self._hard_slider.setValue(int(self._brush.hardness * 100))
        self._update_color_icon()
