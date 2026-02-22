from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QGroupBox, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings

from .diffusion_panel import _make_collapsible
from .slider_edit import SliderEdit


class LamaPanel(QDockWidget):
    remove_requested = pyqtSignal()
    clear_mask_requested = pyqtSignal()
    mask_brush_changed = pyqtSignal(int, float)  # size, hardness
    select_background_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("LaMa Remove", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setMinimumWidth(250)

        self._settings = QSettings("DiffusionEditor", "DiffusionEditor")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Mask Brush ---
        mask_group = QGroupBox("Mask Brush")
        mask_layout = QVBoxLayout(mask_group)

        mask_layout.addWidget(QLabel("Size:"))
        self._mask_size_slider = SliderEdit(1, 500, 50, decimals=0, step=1)
        mask_layout.addWidget(self._mask_size_slider)

        mask_layout.addWidget(QLabel("Hardness:"))
        self._mask_hardness_slider = SliderEdit(0.0, 1.0, 0.40, decimals=2, step=0.05)
        mask_layout.addWidget(self._mask_hardness_slider)

        btn_row = QHBoxLayout()
        self._mask_eraser_btn = QPushButton("Eraser")
        self._mask_eraser_btn.setCheckable(True)
        self._show_mask_btn = QPushButton("Show Mask")
        self._show_mask_btn.setCheckable(True)
        self._show_mask_btn.setChecked(True)
        btn_row.addWidget(self._mask_eraser_btn)
        btn_row.addWidget(self._show_mask_btn)
        mask_layout.addLayout(btn_row)

        self._select_bg_btn = QPushButton("Select Background")
        mask_layout.addWidget(self._select_bg_btn)

        _make_collapsible(mask_group, self._settings, "lama_mask_brush")
        layout.addWidget(mask_group)

        # --- Actions ---
        action_group = QGroupBox("LaMa Layer")
        action_layout = QVBoxLayout(action_group)

        self._layer_info = QLabel("No LaMa layer selected")
        self._layer_info.setWordWrap(True)
        self._layer_info.setStyleSheet("font-size: 11px;")
        action_layout.addWidget(self._layer_info)

        self._remove_btn = QPushButton("Remove Objects")
        action_layout.addWidget(self._remove_btn)

        self._clear_mask_btn = QPushButton("Clear Mask")
        action_layout.addWidget(self._clear_mask_btn)

        _make_collapsible(action_group, self._settings, "lama_layer")
        layout.addWidget(action_group)

        layout.addStretch()
        scroll.setWidget(container)
        self.setWidget(scroll)

        # --- Connections ---
        self._mask_size_slider.valueChanged.connect(self._on_mask_brush_changed)
        self._mask_hardness_slider.valueChanged.connect(self._on_mask_brush_changed)
        self._remove_btn.clicked.connect(self.remove_requested.emit)
        self._clear_mask_btn.clicked.connect(self.clear_mask_requested.emit)
        self._select_bg_btn.clicked.connect(self.select_background_requested.emit)

    def _on_mask_brush_changed(self):
        size = int(self._mask_size_slider.value())
        hardness = self._mask_hardness_slider.value()
        self.mask_brush_changed.emit(size, hardness)

    def show_lama_layer(self, layer):
        mask_status = "has mask" if layer.has_mask() else "no mask"
        info = (
            f"patch: ({layer.patch_x},{layer.patch_y}) "
            f"{layer.patch_w}x{layer.patch_h}\n"
            f"mask: {mask_status}"
        )
        self._layer_info.setText(info)

    def clear_lama_layer(self):
        self._layer_info.setText("No LaMa layer selected")
