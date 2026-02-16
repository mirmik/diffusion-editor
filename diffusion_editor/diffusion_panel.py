import os
from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QTextEdit,
    QSlider, QSpinBox, QGroupBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

MODELS_DIR = os.path.expanduser(
    "~/soft/stable-diffusion-webui-forge/models/Stable-diffusion/"
)


class DiffusionPanel(QDockWidget):
    load_model_requested = pyqtSignal(str, str)  # path, prediction_type
    tool_toggled = pyqtSignal(bool)

    def __init__(self, parent=None):
        super().__init__("Diffusion", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.setMinimumWidth(280)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Model ---
        model_group = QGroupBox("Model")
        model_layout = QVBoxLayout(model_group)

        self._model_combo = QComboBox()
        self._scan_models()
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
        layout.addWidget(prompt_group)

        # --- Parameters ---
        params_group = QGroupBox("Parameters")
        params_layout = QVBoxLayout(params_group)

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
        layout.addWidget(params_group)

        # --- Toggle ---
        self._toggle_btn = QPushButton("Enable Diffusion Brush")
        self._toggle_btn.setCheckable(True)
        layout.addWidget(self._toggle_btn)

        layout.addStretch()
        self.setWidget(container)

        # --- Connections ---
        self._load_btn.clicked.connect(self._on_load)
        self._toggle_btn.toggled.connect(self._on_toggle)
        self._strength_slider.valueChanged.connect(
            lambda v: self._strength_label.setText(f"{v / 100:.2f}")
        )
        self._cfg_slider.valueChanged.connect(
            lambda v: self._cfg_label.setText(f"{v / 10:.1f}")
        )

    def _scan_models(self):
        self._model_combo.clear()
        if not os.path.isdir(MODELS_DIR):
            return
        for f in sorted(os.listdir(MODELS_DIR)):
            if f.endswith(".safetensors") and "flux" not in f.lower():
                self._model_combo.addItem(f, os.path.join(MODELS_DIR, f))

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

    def _on_toggle(self, checked: bool):
        self._toggle_btn.setText(
            "Disable Diffusion Brush" if checked else "Enable Diffusion Brush"
        )
        self.tool_toggled.emit(checked)

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
    def is_active(self) -> bool:
        return self._toggle_btn.isChecked()
