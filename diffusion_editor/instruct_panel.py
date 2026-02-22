import random
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QTextEdit, QLineEdit,
    QSpinBox, QGroupBox, QScrollArea,
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings

from .diffusion_panel import _make_collapsible
from .slider_edit import SliderEdit


class InstructPanel(QDockWidget):
    load_model_requested = pyqtSignal()
    apply_requested = pyqtSignal()
    new_seed_requested = pyqtSignal()
    clear_mask_requested = pyqtSignal()
    mask_brush_changed = pyqtSignal(int, float)  # size, hardness
    draw_patch_toggled = pyqtSignal(bool)
    clear_patch_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("InstructPix2Pix", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setMinimumWidth(280)

        self._settings = QSettings("DiffusionEditor", "DiffusionEditor")

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Model ---
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)

        self._load_btn = QPushButton("Load InstructPix2Pix")
        model_layout.addWidget(self._load_btn)

        self._model_status = QLabel("Not loaded")
        self._model_status.setStyleSheet("color: #888; font-size: 11px;")
        model_layout.addWidget(self._model_status)

        _make_collapsible(model_group, self._settings, "instruct_model")
        layout.addWidget(model_group)

        # --- Instruction ---
        instr_group = QGroupBox("Instruction")
        instr_layout = QVBoxLayout(instr_group)

        self._instruction = QTextEdit()
        self._instruction.setMaximumHeight(60)
        self._instruction.setPlaceholderText("make it snowy")
        instr_layout.addWidget(self._instruction)

        _make_collapsible(instr_group, self._settings, "instruct_instruction")
        layout.addWidget(instr_group)

        # --- Parameters ---
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        # Image Guidance Scale
        params_layout.addWidget(QLabel("Image Guidance Scale:"))
        self._img_guidance_slider = SliderEdit(1.0, 3.0, 1.5, decimals=1, step=0.1)
        params_layout.addWidget(self._img_guidance_slider)

        # CFG Scale
        params_layout.addWidget(QLabel("CFG Scale:"))
        self._cfg_slider = SliderEdit(1.0, 20.0, 7.0, decimals=1, step=0.5)
        params_layout.addWidget(self._cfg_slider)

        # Steps
        params_layout.addWidget(QLabel("Steps:"))
        self._steps_spin = QSpinBox()
        self._steps_spin.setRange(1, 50)
        self._steps_spin.setValue(20)
        params_layout.addWidget(self._steps_spin)

        # Seed
        params_layout.addWidget(QLabel("Seed:"))
        seed_row = QHBoxLayout()
        self._seed_edit = QLineEdit(str(random.randint(0, 2**32 - 1)))
        self._seed_edit.setPlaceholderText("seed")
        self._seed_random_btn = QPushButton("Rnd")
        self._seed_random_btn.setFixedWidth(40)
        seed_row.addWidget(self._seed_edit)
        seed_row.addWidget(self._seed_random_btn)
        params_layout.addLayout(seed_row)

        _make_collapsible(params_group, self._settings, "instruct_params")
        layout.addWidget(params_group)

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

        _make_collapsible(mask_group, self._settings, "instruct_mask_brush")
        layout.addWidget(mask_group)

        # --- Layer Info & Actions ---
        self._layer_group = QGroupBox("Instruct Layer")
        layer_layout = QVBoxLayout(self._layer_group)

        self._layer_info = QLabel("No instruct layer selected")
        self._layer_info.setWordWrap(True)
        self._layer_info.setStyleSheet("font-size: 11px;")
        layer_layout.addWidget(self._layer_info)

        action_row = QHBoxLayout()
        self._apply_btn = QPushButton("Apply Instruction")
        self._new_seed_btn = QPushButton("New Seed")
        action_row.addWidget(self._apply_btn)
        action_row.addWidget(self._new_seed_btn)
        layer_layout.addLayout(action_row)

        self._clear_mask_btn = QPushButton("Clear Mask")
        layer_layout.addWidget(self._clear_mask_btn)

        patch_row = QHBoxLayout()
        self._draw_patch_btn = QPushButton("Draw Patch")
        self._draw_patch_btn.setCheckable(True)
        self._clear_patch_btn = QPushButton("Clear Patch")
        patch_row.addWidget(self._draw_patch_btn)
        patch_row.addWidget(self._clear_patch_btn)
        layer_layout.addLayout(patch_row)

        _make_collapsible(self._layer_group, self._settings, "instruct_layer")
        self._layer_group.setVisible(False)
        layout.addWidget(self._layer_group)

        layout.addStretch()
        scroll.setWidget(container)
        self.setWidget(scroll)

        # --- Connections ---
        self._load_btn.clicked.connect(self.load_model_requested.emit)
        self._apply_btn.clicked.connect(self.apply_requested.emit)
        self._new_seed_btn.clicked.connect(self.new_seed_requested.emit)
        self._seed_random_btn.clicked.connect(
            lambda: self._seed_edit.setText(str(random.randint(0, 2**32 - 1)))
        )
        self._mask_size_slider.valueChanged.connect(self._on_mask_brush_changed)
        self._mask_hardness_slider.valueChanged.connect(self._on_mask_brush_changed)
        self._clear_mask_btn.clicked.connect(self.clear_mask_requested.emit)
        self._draw_patch_btn.toggled.connect(self.draw_patch_toggled.emit)
        self._clear_patch_btn.clicked.connect(self.clear_patch_requested.emit)

    def _on_mask_brush_changed(self):
        size = int(self._mask_size_slider.value())
        hardness = self._mask_hardness_slider.value()
        self.mask_brush_changed.emit(size, hardness)

    @property
    def instruction(self) -> str:
        return self._instruction.toPlainText()

    @property
    def image_guidance_scale(self) -> float:
        return self._img_guidance_slider.value()

    @property
    def guidance_scale(self) -> float:
        return self._cfg_slider.value()

    @property
    def steps(self) -> int:
        return self._steps_spin.value()

    @property
    def seed(self) -> int:
        try:
            return int(self._seed_edit.text())
        except ValueError:
            return -1

    def set_seed(self, seed: int):
        self._seed_edit.setText(str(seed))

    def on_model_loaded(self):
        self._model_status.setText("Loaded: instruct-pix2pix")
        self._load_btn.setEnabled(True)

    def on_model_load_error(self, error: str):
        self._model_status.setText(f"Error: {error[:60]}")
        self._load_btn.setEnabled(True)

    def show_instruct_layer(self, layer):
        self._layer_group.setVisible(True)
        self._instruction.setPlainText(layer.instruction)
        self._img_guidance_slider.setValue(layer.image_guidance_scale)
        self._cfg_slider.setValue(layer.guidance_scale)
        self._steps_spin.setValue(layer.steps)
        self._seed_edit.setText(str(layer.seed))
        mask_status = "has mask" if layer.has_mask() else "no mask"
        if layer.manual_patch_rect:
            r = layer.manual_patch_rect
            pw, ph = r[2] - r[0], r[3] - r[1]
            patch_info = f"manual ({r[0]},{r[1]})-({r[2]},{r[3]}) {pw}x{ph}"
        else:
            patch_info = f"auto ({layer.patch_x},{layer.patch_y}) {layer.patch_w}x{layer.patch_h}"
        info = (
            f"instruction: {layer.instruction[:60]}\n"
            f"image_guidance: {layer.image_guidance_scale:.1f}  "
            f"cfg: {layer.guidance_scale:.1f}\n"
            f"steps: {layer.steps}  seed: {layer.seed}\n"
            f"patch: {patch_info}\n"
            f"mask: {mask_status}"
        )
        self._layer_info.setText(info)

    def clear_instruct_layer(self):
        self._layer_group.setVisible(False)
        self._layer_info.setText("No instruct layer selected")
