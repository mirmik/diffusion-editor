import os
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QTextEdit, QLineEdit,
    QSlider, QSpinBox, QGroupBox, QScrollArea,
)
import random
from PyQt6.QtCore import Qt, pyqtSignal, QSettings


def _set_layout_visible(layout, visible):
    """Recursively show/hide all widgets inside a layout."""
    for i in range(layout.count()):
        item = layout.itemAt(i)
        if item.widget():
            item.widget().setVisible(visible)
        elif item.layout():
            _set_layout_visible(item.layout(), visible)


def _make_collapsible(group: QGroupBox, settings: QSettings, key: str):
    """Make a QGroupBox collapsible with state saved in QSettings."""
    group.setCheckable(True)
    collapsed = settings.value(f"collapsed/{key}", False, type=bool)
    group.setChecked(not collapsed)

    def on_toggled(checked):
        layout = group.layout()
        if layout:
            _set_layout_visible(layout, checked)
        settings.setValue(f"collapsed/{key}", not checked)

    group.toggled.connect(on_toggled)
    # Apply initial state
    if collapsed:
        on_toggled(False)

MODELS_DIR = os.path.expanduser(
    "~/soft/stable-diffusion-webui-forge/models/Stable-diffusion/"
)


class DiffusionPanel(QDockWidget):
    load_model_requested = pyqtSignal(str, str)  # path, prediction_type
    regenerate_requested = pyqtSignal()
    new_seed_requested = pyqtSignal()
    clear_mask_requested = pyqtSignal()
    mask_brush_changed = pyqtSignal(int, float)  # size, hardness
    load_ip_adapter_requested = pyqtSignal()
    draw_rect_toggled = pyqtSignal(bool)
    show_rect_toggled = pyqtSignal(bool)
    clear_rect_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__("Diffusion", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setMinimumWidth(280)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Model ---
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)

        self._settings = QSettings("DiffusionEditor", "DiffusionEditor")
        self._model_combo = QComboBox()
        self._scan_models()
        self._restore_last_model()
        model_layout.addWidget(self._model_combo)

        self._prediction_combo = QComboBox()
        self._prediction_combo.addItem("Auto (detect from name)", "")
        self._prediction_combo.addItem("epsilon", "epsilon")
        self._prediction_combo.addItem("v_prediction", "v_prediction")
        model_layout.addWidget(self._prediction_combo)

        self._load_btn = QPushButton("Load Model")
        model_layout.addWidget(self._load_btn)

        self._model_status = QLabel("No model loaded")
        model_layout.addWidget(self._model_status)

        self._model_diag = QLabel("")
        self._model_diag.setWordWrap(True)
        self._model_diag.setStyleSheet("color: #888; font-size: 11px;")
        model_layout.addWidget(self._model_diag)
        _make_collapsible(model_group, self._settings, "model")
        layout.addWidget(model_group)

        # --- Prompt ---
        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout(prompt_group)

        prompt_layout.addWidget(QLabel("Positive:"))
        self._prompt = QTextEdit()
        self._prompt.setMaximumHeight(60)
        self._prompt.setPlaceholderText("masterpiece, best quality")
        prompt_layout.addWidget(self._prompt)

        prompt_layout.addWidget(QLabel("Negative:"))
        self._negative_prompt = QTextEdit()
        self._negative_prompt.setMaximumHeight(40)
        self._negative_prompt.setPlaceholderText("worst quality, blurry")
        prompt_layout.addWidget(self._negative_prompt)
        _make_collapsible(prompt_group, self._settings, "prompt")
        layout.addWidget(prompt_group)

        # --- Parameters ---
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

        # Mode
        params_layout.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("txt2img", "txt2img")
        self._mode_combo.addItem("img2img", "img2img")
        self._mode_combo.addItem("inpaint", "inpaint")
        params_layout.addWidget(self._mode_combo)

        # Denoising strength
        params_layout.addWidget(QLabel("Denoising Strength:"))
        strength_row = QHBoxLayout()
        self._strength_slider = QSlider(Qt.Orientation.Horizontal)
        self._strength_slider.setRange(5, 80)
        self._strength_slider.setValue(30)
        self._strength_label = QLabel("0.30")
        strength_row.addWidget(self._strength_slider)
        strength_row.addWidget(self._strength_label)
        params_layout.addLayout(strength_row)

        # Steps
        params_layout.addWidget(QLabel("Steps:"))
        self._steps_spin = QSpinBox()
        self._steps_spin.setRange(1, 50)
        self._steps_spin.setValue(20)
        params_layout.addWidget(self._steps_spin)

        # CFG Scale
        params_layout.addWidget(QLabel("CFG Scale:"))
        cfg_row = QHBoxLayout()
        self._cfg_slider = QSlider(Qt.Orientation.Horizontal)
        self._cfg_slider.setRange(10, 200)
        self._cfg_slider.setValue(70)
        self._cfg_label = QLabel("7.0")
        cfg_row.addWidget(self._cfg_slider)
        cfg_row.addWidget(self._cfg_label)
        params_layout.addLayout(cfg_row)

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
        _make_collapsible(params_group, self._settings, "params")
        layout.addWidget(params_group)

        # --- Mask Brush ---
        mask_group = QGroupBox("Mask Brush")
        mask_layout = QVBoxLayout(mask_group)

        mask_layout.addWidget(QLabel("Size:"))
        size_row = QHBoxLayout()
        self._mask_size_slider = QSlider(Qt.Orientation.Horizontal)
        self._mask_size_slider.setRange(1, 500)
        self._mask_size_slider.setValue(50)
        self._mask_size_label = QLabel("50")
        size_row.addWidget(self._mask_size_slider)
        size_row.addWidget(self._mask_size_label)
        mask_layout.addLayout(size_row)

        mask_layout.addWidget(QLabel("Hardness:"))
        hard_row = QHBoxLayout()
        self._mask_hardness_slider = QSlider(Qt.Orientation.Horizontal)
        self._mask_hardness_slider.setRange(0, 100)
        self._mask_hardness_slider.setValue(80)
        self._mask_hardness_label = QLabel("0.80")
        hard_row.addWidget(self._mask_hardness_slider)
        hard_row.addWidget(self._mask_hardness_label)
        mask_layout.addLayout(hard_row)

        btn_row = QHBoxLayout()
        self._mask_eraser_btn = QPushButton("Eraser")
        self._mask_eraser_btn.setCheckable(True)
        self._show_mask_btn = QPushButton("Show Mask")
        self._show_mask_btn.setCheckable(True)
        self._show_mask_btn.setChecked(True)
        btn_row.addWidget(self._mask_eraser_btn)
        btn_row.addWidget(self._show_mask_btn)
        mask_layout.addLayout(btn_row)

        _make_collapsible(mask_group, self._settings, "mask_brush")
        layout.addWidget(mask_group)

        # --- IP-Adapter ---
        ip_group = QGroupBox("IP-Adapter")
        ip_layout = QVBoxLayout(ip_group)

        self._load_ip_btn = QPushButton("Load IP-Adapter")
        ip_layout.addWidget(self._load_ip_btn)

        self._ip_status = QLabel("Not loaded")
        self._ip_status.setStyleSheet("color: #888; font-size: 11px;")
        ip_layout.addWidget(self._ip_status)

        ip_layout.addWidget(QLabel("Scale:"))
        ip_scale_row = QHBoxLayout()
        self._ip_scale_slider = QSlider(Qt.Orientation.Horizontal)
        self._ip_scale_slider.setRange(0, 100)
        self._ip_scale_slider.setValue(60)
        self._ip_scale_label = QLabel("0.60")
        ip_scale_row.addWidget(self._ip_scale_slider)
        ip_scale_row.addWidget(self._ip_scale_label)
        ip_layout.addLayout(ip_scale_row)

        ip_btn_row = QHBoxLayout()
        self._draw_rect_btn = QPushButton("Draw Rect")
        self._draw_rect_btn.setCheckable(True)
        self._show_rect_btn = QPushButton("Show Rect")
        self._show_rect_btn.setCheckable(True)
        self._show_rect_btn.setChecked(True)
        self._clear_rect_btn = QPushButton("Clear Rect")
        ip_btn_row.addWidget(self._draw_rect_btn)
        ip_btn_row.addWidget(self._show_rect_btn)
        ip_btn_row.addWidget(self._clear_rect_btn)
        ip_layout.addLayout(ip_btn_row)

        _make_collapsible(ip_group, self._settings, "ip_adapter")
        layout.addWidget(ip_group)

        # --- Selected Diffusion Layer ---
        self._layer_group = QGroupBox("Diffusion Layer")
        layer_layout = QVBoxLayout(self._layer_group)

        self._layer_info = QLabel("No diffusion layer selected")
        self._layer_info.setWordWrap(True)
        self._layer_info.setStyleSheet("font-size: 11px;")
        layer_layout.addWidget(self._layer_info)

        regen_row = QHBoxLayout()
        self._regen_btn = QPushButton("Regenerate")
        self._new_seed_btn = QPushButton("New Seed")
        regen_row.addWidget(self._regen_btn)
        regen_row.addWidget(self._new_seed_btn)
        layer_layout.addLayout(regen_row)

        self._clear_mask_btn = QPushButton("Clear Mask")
        layer_layout.addWidget(self._clear_mask_btn)

        _make_collapsible(self._layer_group, self._settings, "diffusion_layer")
        self._layer_group.setVisible(False)
        layout.addWidget(self._layer_group)

        layout.addStretch()
        scroll.setWidget(container)
        self.setWidget(scroll)

        # --- Connections ---
        self._load_btn.clicked.connect(self._on_load)
        self._seed_random_btn.clicked.connect(
            lambda: self._seed_edit.setText(str(random.randint(0, 2**32 - 1)))
        )
        self._regen_btn.clicked.connect(self.regenerate_requested.emit)
        self._new_seed_btn.clicked.connect(self.new_seed_requested.emit)
        self._clear_mask_btn.clicked.connect(self.clear_mask_requested.emit)
        self._strength_slider.valueChanged.connect(
            lambda v: self._strength_label.setText(f"{v / 100:.2f}")
        )
        self._cfg_slider.valueChanged.connect(
            lambda v: self._cfg_label.setText(f"{v / 10:.1f}")
        )
        self._mask_size_slider.valueChanged.connect(self._on_mask_brush_changed)
        self._mask_hardness_slider.valueChanged.connect(self._on_mask_brush_changed)
        self._load_ip_btn.clicked.connect(self.load_ip_adapter_requested.emit)
        self._draw_rect_btn.toggled.connect(self.draw_rect_toggled.emit)
        self._show_rect_btn.toggled.connect(self.show_rect_toggled.emit)
        self._clear_rect_btn.clicked.connect(self.clear_rect_requested.emit)
        self._ip_scale_slider.valueChanged.connect(
            lambda v: self._ip_scale_label.setText(f"{v / 100:.2f}")
        )

    def _scan_models(self):
        self._model_combo.clear()
        if not os.path.isdir(MODELS_DIR):
            return
        for f in sorted(os.listdir(MODELS_DIR)):
            if f.endswith(".safetensors") and "flux" not in f.lower():
                self._model_combo.addItem(f, os.path.join(MODELS_DIR, f))

    def _restore_last_model(self):
        last = self._settings.value("last_model", "")
        if last:
            idx = self._model_combo.findText(last)
            if idx >= 0:
                self._model_combo.setCurrentIndex(idx)

    def _on_load(self):
        path = self._model_combo.currentData()
        if path:
            self._model_status.setText("Loading...")
            self._model_diag.setText("")
            self._load_btn.setEnabled(False)
            pred = self._prediction_combo.currentData()
            self.load_model_requested.emit(path, pred)

    def on_model_loaded(self, path: str, model_info: dict):
        name = os.path.basename(path)
        self._model_status.setText(f"Loaded: {name}")
        self._settings.setValue("last_model", name)
        self._load_btn.setEnabled(True)
        diag = (
            f"scheduler: {model_info.get('scheduler', '?')}\n"
            f"prediction: {model_info.get('prediction_type', '?')}\n"
            f"algorithm: {model_info.get('algorithm_type', '?')}\n"
            f"karras: {model_info.get('karras', '?')}\n"
            f"guessed: {model_info.get('guessed_from_name', 'none')}"
        )
        self._model_diag.setText(diag)

    def on_model_load_error(self, error: str):
        self._model_status.setText(f"Error: {error[:80]}")
        self._model_diag.setText("")
        self._load_btn.setEnabled(True)

    @property
    def prompt(self) -> str:
        return self._prompt.toPlainText()

    @property
    def negative_prompt(self) -> str:
        return self._negative_prompt.toPlainText()

    @property
    def strength(self) -> float:
        return self._strength_slider.value() / 100.0

    @property
    def steps(self) -> int:
        return self._steps_spin.value()

    @property
    def guidance_scale(self) -> float:
        return self._cfg_slider.value() / 10.0

    @property
    def seed(self) -> int:
        try:
            return int(self._seed_edit.text())
        except ValueError:
            return -1

    def set_seed(self, seed: int):
        self._seed_edit.setText(str(seed))

    @property
    def mask_eraser(self) -> bool:
        return self._mask_eraser_btn.isChecked()

    @property
    def mask_brush_size(self) -> int:
        return self._mask_size_slider.value()

    @property
    def mask_brush_hardness(self) -> float:
        return self._mask_hardness_slider.value() / 100.0

    def _on_mask_brush_changed(self):
        size = self._mask_size_slider.value()
        hardness = self._mask_hardness_slider.value() / 100.0
        self._mask_size_label.setText(str(size))
        self._mask_hardness_label.setText(f"{hardness:.2f}")
        self.mask_brush_changed.emit(size, hardness)

    @property
    def prediction_type(self) -> str:
        return self._prediction_combo.currentData() or ""

    @property
    def mode(self) -> str:
        return self._mode_combo.currentData()

    @property
    def ip_adapter_scale(self) -> float:
        return self._ip_scale_slider.value() / 100.0

    def on_ip_adapter_loaded(self):
        self._ip_status.setText("Loaded")
        self._load_ip_btn.setEnabled(True)

    def on_ip_adapter_load_error(self, error: str):
        self._ip_status.setText(f"Error: {error[:60]}")
        self._load_ip_btn.setEnabled(True)

    def show_diffusion_layer(self, layer):
        self._layer_group.setVisible(True)
        self._prompt.setPlainText(layer.prompt)
        self._negative_prompt.setPlainText(layer.negative_prompt)
        self._strength_slider.setValue(int(layer.strength * 100))
        self._steps_spin.setValue(layer.steps)
        self._cfg_slider.setValue(int(layer.guidance_scale * 10))
        self._seed_edit.setText(str(layer.seed))
        idx = self._mode_combo.findData(layer.mode)
        if idx >= 0:
            self._mode_combo.setCurrentIndex(idx)
        self._ip_scale_slider.setValue(int(layer.ip_adapter_scale * 100))
        model_name = os.path.basename(layer.model_path) if layer.model_path else "?"
        mask_status = "has mask" if layer.has_mask() else "no mask"
        if layer.ip_adapter_rect:
            r = layer.ip_adapter_rect
            ip_info = f"rect ({r[0]},{r[1]})-({r[2]},{r[3]}) scale={layer.ip_adapter_scale:.2f}"
        else:
            ip_info = "none"
        info = (
            f"model: {model_name}\n"
            f"mode: {layer.mode}\n"
            f"prompt: {layer.prompt[:60]}\n"
            f"negative: {layer.negative_prompt[:40]}\n"
            f"strength: {layer.strength:.2f}  steps: {layer.steps}  "
            f"cfg: {layer.guidance_scale:.1f}\n"
            f"seed: {layer.seed}\n"
            f"patch: ({layer.patch_x},{layer.patch_y}) {layer.patch_w}x{layer.patch_h}\n"
            f"mask: {mask_status}\n"
            f"ip_adapter: {ip_info}"
        )
        self._layer_info.setText(info)

    def clear_diffusion_layer(self):
        self._layer_group.setVisible(False)
        self._layer_info.setText("No diffusion layer selected")
