from PyQt6.QtWidgets import QWidget, QHBoxLayout, QSlider, QDoubleSpinBox
from PyQt6.QtCore import Qt, pyqtSignal


class SliderEdit(QWidget):
    """Slider + editable spinbox, synced together.

    Parameters
    ----------
    min_val, max_val : float
        Logical range (e.g. 0.0–1.0 for hardness, 1–500 for size).
    default : float
        Initial value.
    decimals : int
        Number of decimal places (0 for integer-like sliders).
    step : float
        Step size for the spinbox arrows / keyboard.
    """

    valueChanged = pyqtSignal(float)

    def __init__(self, min_val: float, max_val: float, default: float,
                 decimals: int = 0, step: float = 1.0, parent=None):
        super().__init__(parent)
        self._decimals = decimals
        self._min = min_val
        self._max = max_val
        self._updating = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setRange(0, 1000)
        layout.addWidget(self._slider, 1)

        self._spin = QDoubleSpinBox()
        self._spin.setDecimals(decimals)
        self._spin.setRange(min_val, max_val)
        self._spin.setSingleStep(step)
        self._spin.setFixedWidth(65)
        layout.addWidget(self._spin)

        self._slider.valueChanged.connect(self._slider_moved)
        self._spin.valueChanged.connect(self._spin_changed)

        self.setValue(default)

    def _val_to_slider(self, val: float) -> int:
        if self._max == self._min:
            return 0
        return int((val - self._min) / (self._max - self._min) * 1000)

    def _slider_to_val(self, pos: int) -> float:
        return self._min + (pos / 1000.0) * (self._max - self._min)

    def _slider_moved(self, pos):
        if self._updating:
            return
        self._updating = True
        val = self._slider_to_val(pos)
        self._spin.setValue(val)
        self._updating = False
        self.valueChanged.emit(val)

    def _spin_changed(self, val):
        if self._updating:
            return
        self._updating = True
        self._slider.setValue(self._val_to_slider(val))
        self._updating = False
        self.valueChanged.emit(val)

    def value(self) -> float:
        return self._spin.value()

    def setValue(self, val: float):
        self._updating = True
        self._spin.setValue(val)
        self._slider.setValue(self._val_to_slider(val))
        self._updating = False
