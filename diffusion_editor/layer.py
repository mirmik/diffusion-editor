import numpy as np
from PIL import Image
from PyQt6.QtCore import QObject, pyqtSignal


class Layer:
    def __init__(self, name: str, width: int, height: int, image: np.ndarray = None):
        self.name = name
        self.visible = True
        self.opacity = 1.0
        if image is not None:
            self.image = np.ascontiguousarray(image.astype(np.uint8))
        else:
            self.image = np.zeros((height, width, 4), dtype=np.uint8)

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]


class DiffusionLayer(Layer):
    def __init__(self, name: str, width: int, height: int,
                 source_patch: Image.Image,
                 patch_x: int, patch_y: int, patch_w: int, patch_h: int,
                 prompt: str, negative_prompt: str,
                 strength: float, guidance_scale: float, steps: int,
                 seed: int,
                 model_path: str = "", prediction_type: str = ""):
        super().__init__(name, width, height)
        self.source_patch = source_patch
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.seed = seed
        self.model_path = model_path
        self.prediction_type = prediction_type
        self.mask = np.zeros((height, width), dtype=np.uint8)

    def clear_mask(self):
        self.mask[:] = 0

    def has_mask(self) -> bool:
        return np.any(self.mask > 0)

    def mask_bbox(self) -> tuple[int, int, int, int] | None:
        """Return (x0, y0, x1, y1) bounding box of non-zero mask pixels."""
        rows = np.any(self.mask > 0, axis=1)
        cols = np.any(self.mask > 0, axis=0)
        if not np.any(rows):
            return None
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        return int(x0), int(y0), int(x1) + 1, int(y1) + 1

    def mask_center(self) -> tuple[int, int] | None:
        bbox = self.mask_bbox()
        if bbox is None:
            return None
        x0, y0, x1, y1 = bbox
        return (x0 + x1) // 2, (y0 + y1) // 2


class LayerStack(QObject):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layers: list[Layer] = []
        self._active_index = -1
        self._width = 0
        self._height = 0

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def layers(self):
        return self._layers

    @property
    def active_index(self):
        return self._active_index

    @active_index.setter
    def active_index(self, value):
        if 0 <= value < len(self._layers):
            self._active_index = value
            self.changed.emit()

    @property
    def active_layer(self) -> Layer | None:
        if 0 <= self._active_index < len(self._layers):
            return self._layers[self._active_index]
        return None

    def init_from_image(self, image: np.ndarray):
        self._layers.clear()
        h, w = image.shape[:2]
        self._width = w
        self._height = h
        layer = Layer("Background", w, h, image)
        self._layers.append(layer)
        self._active_index = 0
        self.changed.emit()

    def add_layer(self, name: str, image: np.ndarray = None):
        if self._width == 0 or self._height == 0:
            return
        layer = Layer(name, self._width, self._height, image)
        insert_at = self._active_index if self._active_index >= 0 else 0
        self._layers.insert(insert_at, layer)
        self._active_index = insert_at
        self.changed.emit()

    def insert_layer(self, layer: Layer):
        if self._width == 0 or self._height == 0:
            return
        insert_at = self._active_index if self._active_index >= 0 else 0
        self._layers.insert(insert_at, layer)
        self._active_index = insert_at
        self.changed.emit()

    def remove_layer(self, index: int):
        if len(self._layers) <= 1:
            return
        if 0 <= index < len(self._layers):
            self._layers.pop(index)
            if self._active_index >= len(self._layers):
                self._active_index = len(self._layers) - 1
            self.changed.emit()

    def move_layer(self, from_idx: int, to_idx: int):
        if from_idx == to_idx:
            return
        if not (0 <= from_idx < len(self._layers)):
            return
        to_idx = max(0, min(to_idx, len(self._layers) - 1))
        layer = self._layers.pop(from_idx)
        self._layers.insert(to_idx, layer)
        self._active_index = to_idx
        self.changed.emit()

    def reorder(self, new_indices: list[int]):
        if len(new_indices) != len(self._layers):
            return
        active_layer = self.active_layer
        self._layers = [self._layers[i] for i in new_indices]
        if active_layer in self._layers:
            self._active_index = self._layers.index(active_layer)
        self.changed.emit()

    def set_visibility(self, index: int, visible: bool):
        if 0 <= index < len(self._layers):
            self._layers[index].visible = visible
            self.changed.emit()

    def flatten(self):
        result = self.composite()
        self._layers.clear()
        layer = Layer("Background", self._width, self._height, result)
        self._layers.append(layer)
        self._active_index = 0
        self.changed.emit()

    def composite(self) -> np.ndarray:
        if not self._layers or self._width == 0:
            return np.zeros((1, 1, 4), dtype=np.uint8)

        result = np.zeros((self._height, self._width, 4), dtype=np.float32)

        for layer in reversed(self._layers):
            if not layer.visible or layer.opacity <= 0:
                continue
            src = layer.image.astype(np.float32)
            alpha = src[:, :, 3:4] / 255.0 * layer.opacity
            inv_alpha = 1.0 - alpha
            result[:, :, :3] = src[:, :, :3] * alpha + result[:, :, :3] * inv_alpha
            result[:, :, 3:4] = alpha * 255.0 + result[:, :, 3:4] * inv_alpha

        return np.clip(result, 0, 255).astype(np.uint8)
