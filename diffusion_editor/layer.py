import io
import json
import zipfile

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

    def to_dict(self, index: int) -> dict:
        return {
            "index": index,
            "type": "layer",
            "name": self.name,
            "visible": self.visible,
            "opacity": self.opacity,
            "image_file": f"layers/{index}_image.png",
        }

    def save_images_to_zip(self, zf: zipfile.ZipFile, index: int):
        img = Image.fromarray(self.image, "RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        zf.writestr(f"layers/{index}_image.png", buf.getvalue())

    @classmethod
    def from_dict(cls, d: dict, zf: zipfile.ZipFile) -> "Layer":
        image_data = zf.read(d["image_file"])
        img = Image.open(io.BytesIO(image_data)).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)
        layer = cls.__new__(cls)
        layer.name = d["name"]
        layer.visible = d["visible"]
        layer.opacity = d["opacity"]
        layer.image = np.ascontiguousarray(arr)
        return layer


class DiffusionLayer(Layer):
    def __init__(self, name: str, width: int, height: int,
                 source_patch: Image.Image,
                 patch_x: int, patch_y: int, patch_w: int, patch_h: int,
                 prompt: str, negative_prompt: str,
                 strength: float, guidance_scale: float, steps: int,
                 seed: int,
                 model_path: str = "", prediction_type: str = "",
                 mode: str = "img2img"):
        super().__init__(name, width, height)
        self.mode = mode
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

    def to_dict(self, index: int) -> dict:
        d = super().to_dict(index)
        d["type"] = "diffusion"
        d["mode"] = self.mode
        d["mask_file"] = f"layers/{index}_mask.png"
        d["source_file"] = f"layers/{index}_source.png" if self.source_patch is not None else None
        d["patch_x"] = self.patch_x
        d["patch_y"] = self.patch_y
        d["patch_w"] = self.patch_w
        d["patch_h"] = self.patch_h
        d["prompt"] = self.prompt
        d["negative_prompt"] = self.negative_prompt
        d["strength"] = self.strength
        d["guidance_scale"] = self.guidance_scale
        d["steps"] = self.steps
        d["seed"] = self.seed
        d["model_path"] = self.model_path
        d["prediction_type"] = self.prediction_type
        return d

    def save_images_to_zip(self, zf: zipfile.ZipFile, index: int):
        super().save_images_to_zip(zf, index)
        mask_img = Image.fromarray(self.mask, "L")
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        zf.writestr(f"layers/{index}_mask.png", buf.getvalue())
        if self.source_patch is not None:
            buf = io.BytesIO()
            self.source_patch.save(buf, format="PNG")
            zf.writestr(f"layers/{index}_source.png", buf.getvalue())

    @classmethod
    def from_dict(cls, d: dict, zf: zipfile.ZipFile) -> "DiffusionLayer":
        image_data = zf.read(d["image_file"])
        img = Image.open(io.BytesIO(image_data)).convert("RGBA")
        arr = np.array(img, dtype=np.uint8)
        h, w = arr.shape[:2]

        mask_data = zf.read(d["mask_file"])
        mask_img = Image.open(io.BytesIO(mask_data)).convert("L")
        mask_arr = np.array(mask_img, dtype=np.uint8)

        source_patch = None
        if d.get("source_file") and d["source_file"] in zf.namelist():
            source_data = zf.read(d["source_file"])
            source_patch = Image.open(io.BytesIO(source_data)).convert("RGB")

        layer = cls.__new__(cls)
        layer.name = d["name"]
        layer.visible = d["visible"]
        layer.opacity = d["opacity"]
        layer.image = np.ascontiguousarray(arr)
        layer.mask = np.ascontiguousarray(mask_arr)
        layer.source_patch = source_patch
        layer.patch_x = d["patch_x"]
        layer.patch_y = d["patch_y"]
        layer.patch_w = d["patch_w"]
        layer.patch_h = d["patch_h"]
        layer.prompt = d["prompt"]
        layer.negative_prompt = d["negative_prompt"]
        layer.strength = d["strength"]
        layer.guidance_scale = d["guidance_scale"]
        layer.steps = d["steps"]
        layer.seed = d["seed"]
        layer.model_path = d.get("model_path", "")
        layer.prediction_type = d.get("prediction_type", "")
        layer.mode = d.get("mode", "img2img")
        return layer


class LayerStack(QObject):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layers: list[Layer] = []
        self._active_index = -1
        self._width = 0
        self._height = 0

    def next_name(self, prefix: str) -> str:
        import re
        pattern = re.compile(rf'^{re.escape(prefix)} (\d+)$')
        max_n = -1
        for layer in self._layers:
            m = pattern.match(layer.name)
            if m:
                max_n = max(max_n, int(m.group(1)))
        return f"{prefix} {max_n + 1}"

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

    def composite(self, exclude_above: int = -1) -> np.ndarray:
        """Composite visible layers.

        If exclude_above >= 0, only layers with index > exclude_above
        are included (i.e. layers below the given index in the stack).
        """
        if not self._layers or self._width == 0:
            return np.zeros((1, 1, 4), dtype=np.uint8)

        result = np.zeros((self._height, self._width, 4), dtype=np.float32)

        for i, layer in enumerate(reversed(self._layers)):
            real_idx = len(self._layers) - 1 - i
            if exclude_above >= 0 and real_idx <= exclude_above:
                continue
            if not layer.visible or layer.opacity <= 0:
                continue
            src = layer.image.astype(np.float32)
            alpha = src[:, :, 3:4] / 255.0 * layer.opacity
            inv_alpha = 1.0 - alpha
            result[:, :, :3] = src[:, :, :3] * alpha + result[:, :, :3] * inv_alpha
            result[:, :, 3:4] = alpha * 255.0 + result[:, :, 3:4] * inv_alpha

        return np.clip(result, 0, 255).astype(np.uint8)

    FORMAT_VERSION = 1

    def save_project(self, path: str):
        manifest = {
            "format_version": self.FORMAT_VERSION,
            "canvas_width": self._width,
            "canvas_height": self._height,
            "active_index": self._active_index,
            "layers": [],
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, layer in enumerate(self._layers):
                manifest["layers"].append(layer.to_dict(i))
                layer.save_images_to_zip(zf, i)
            zf.writestr("manifest.json",
                        json.dumps(manifest, indent=2, ensure_ascii=False))

    def load_project(self, path: str):
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            version = manifest.get("format_version", 0)
            if version > self.FORMAT_VERSION:
                raise ValueError(
                    f"Project version {version} is newer than "
                    f"supported version {self.FORMAT_VERSION}")

            new_layers = []
            for layer_dict in manifest["layers"]:
                if layer_dict["type"] == "diffusion":
                    layer = DiffusionLayer.from_dict(layer_dict, zf)
                else:
                    layer = Layer.from_dict(layer_dict, zf)
                new_layers.append(layer)

        self._layers.clear()
        self._layers.extend(new_layers)
        self._width = manifest["canvas_width"]
        self._height = manifest["canvas_height"]
        self._active_index = manifest.get("active_index", 0)
        if not (0 <= self._active_index < len(self._layers)):
            self._active_index = 0
        self.changed.emit()
