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
        self.children: list['Layer'] = []
        self.parent: 'Layer | None' = None
        if image is not None:
            self.image = np.ascontiguousarray(image.astype(np.uint8))
        else:
            self.image = np.zeros((height, width, 4), dtype=np.uint8)

    def add_child(self, child: 'Layer', index: int | None = None):
        if child.parent is not None:
            child.parent.remove_child(child)
        child.parent = self
        if index is not None:
            self.children.insert(index, child)
        else:
            self.children.append(child)

    def remove_child(self, child: 'Layer'):
        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def all_descendants(self) -> list['Layer']:
        result = []
        for child in self.children:
            result.append(child)
            result.extend(child.all_descendants())
        return result

    @property
    def width(self):
        return self.image.shape[1]

    @property
    def height(self):
        return self.image.shape[0]

    def to_dict(self, path: str) -> dict:
        file_key = path.replace("/", "_")
        d = {
            "path": path,
            "type": "layer",
            "name": self.name,
            "visible": self.visible,
            "opacity": self.opacity,
            "image_file": f"layers/{file_key}_image.png",
            "children": [],
        }
        for i, child in enumerate(self.children):
            d["children"].append(child.to_dict(f"{path}/{i}"))
        return d

    def save_images_to_zip(self, zf: zipfile.ZipFile, path: str):
        file_key = path.replace("/", "_")
        img = Image.fromarray(self.image, "RGBA")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        zf.writestr(f"layers/{file_key}_image.png", buf.getvalue())
        for i, child in enumerate(self.children):
            child.save_images_to_zip(zf, f"{path}/{i}")

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
        layer.children = []
        layer.parent = None
        for child_dict in d.get("children", []):
            child = _layer_from_dict(child_dict, zf)
            child.parent = layer
            layer.children.append(child)
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
        self.ip_adapter_rect = None   # (x0, y0, x1, y1) or None
        self.ip_adapter_scale = 0.6
        self.masked_content = "original"  # original, fill, latent_noise, latent_nothing

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

    def to_dict(self, path: str) -> dict:
        d = super().to_dict(path)
        file_key = path.replace("/", "_")
        d["type"] = "diffusion"
        d["mode"] = self.mode
        d["mask_file"] = f"layers/{file_key}_mask.png"
        d["source_file"] = f"layers/{file_key}_source.png" if self.source_patch is not None else None
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
        d["ip_adapter_rect"] = list(self.ip_adapter_rect) if self.ip_adapter_rect else None
        d["ip_adapter_scale"] = self.ip_adapter_scale
        d["masked_content"] = self.masked_content
        return d

    def save_images_to_zip(self, zf: zipfile.ZipFile, path: str):
        super().save_images_to_zip(zf, path)
        file_key = path.replace("/", "_")
        mask_img = Image.fromarray(self.mask, "L")
        buf = io.BytesIO()
        mask_img.save(buf, format="PNG")
        zf.writestr(f"layers/{file_key}_mask.png", buf.getvalue())
        if self.source_patch is not None:
            buf = io.BytesIO()
            self.source_patch.save(buf, format="PNG")
            zf.writestr(f"layers/{file_key}_source.png", buf.getvalue())

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
        layer.children = []
        layer.parent = None
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
        rect = d.get("ip_adapter_rect")
        layer.ip_adapter_rect = tuple(rect) if rect else None
        layer.ip_adapter_scale = d.get("ip_adapter_scale", 0.6)
        layer.masked_content = d.get("masked_content", "original")
        for child_dict in d.get("children", []):
            child = _layer_from_dict(child_dict, zf)
            child.parent = layer
            layer.children.append(child)
        return layer


def _layer_from_dict(d: dict, zf: zipfile.ZipFile) -> Layer:
    """Dispatch layer deserialization by type."""
    if d.get("type") == "diffusion":
        return DiffusionLayer.from_dict(d, zf)
    return Layer.from_dict(d, zf)


class LayerStack(QObject):
    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layers: list[Layer] = []  # root-level layers
        self._active_layer: Layer | None = None
        self._width = 0
        self._height = 0

    # --- Tree traversal ---

    def _all_layers_flat(self) -> list[Layer]:
        """All layers in depth-first order (index 0 = topmost)."""
        result = []
        for layer in self._layers:
            result.append(layer)
            result.extend(layer.all_descendants())
        return result

    def next_name(self, prefix: str) -> str:
        import re
        pattern = re.compile(rf'^{re.escape(prefix)} (\d+)$')
        max_n = -1
        for layer in self._all_layers_flat():
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
    def active_layer(self) -> Layer | None:
        return self._active_layer

    @active_layer.setter
    def active_layer(self, layer: Layer | None):
        if layer is not self._active_layer:
            self._active_layer = layer
            self.changed.emit()

    def init_from_image(self, image: np.ndarray):
        self._layers.clear()
        h, w = image.shape[:2]
        self._width = w
        self._height = h
        layer = Layer("Background", w, h, image)
        self._layers.append(layer)
        self._active_layer = layer
        self.changed.emit()

    def _insert_near_active(self, layer: Layer):
        """Insert layer as a sibling above the active layer."""
        if self._active_layer is not None and self._active_layer.parent is not None:
            parent = self._active_layer.parent
            idx = parent.children.index(self._active_layer)
            parent.add_child(layer, idx)
        elif self._active_layer is not None and self._active_layer in self._layers:
            idx = self._layers.index(self._active_layer)
            self._layers.insert(idx, layer)
        else:
            self._layers.insert(0, layer)
        self._active_layer = layer

    def add_layer(self, name: str, image: np.ndarray = None):
        if self._width == 0 or self._height == 0:
            return
        layer = Layer(name, self._width, self._height, image)
        self._insert_near_active(layer)
        self.changed.emit()

    def insert_layer(self, layer: Layer):
        if self._width == 0 or self._height == 0:
            return
        self._insert_near_active(layer)
        self.changed.emit()

    def remove_layer(self, layer: Layer):
        """Remove layer and its entire subtree."""
        all_layers = self._all_layers_flat()
        if len(all_layers) <= 1:
            return
        if layer.parent is not None:
            parent = layer.parent
            idx = parent.children.index(layer)
            parent.remove_child(layer)
            if parent.children:
                self._active_layer = parent.children[min(idx, len(parent.children) - 1)]
            else:
                self._active_layer = parent
        elif layer in self._layers:
            idx = self._layers.index(layer)
            self._layers.remove(layer)
            layer.parent = None
            if self._layers:
                self._active_layer = self._layers[min(idx, len(self._layers) - 1)]
            else:
                self._active_layer = None
        self.changed.emit()

    def move_layer(self, layer: Layer, new_parent: Layer | None, index: int):
        """Move layer to new_parent at index (or root if new_parent is None)."""
        # Remove from current location
        if layer.parent is not None:
            layer.parent.remove_child(layer)
        elif layer in self._layers:
            self._layers.remove(layer)
        # Insert at new location
        if new_parent is not None:
            new_parent.add_child(layer, index)
        else:
            self._layers.insert(index, layer)
        self._active_layer = layer
        self.changed.emit()

    def set_visibility(self, layer: Layer, visible: bool):
        layer.visible = visible
        self.changed.emit()

    def flatten(self):
        result = self.composite()
        self._layers.clear()
        layer = Layer("Background", self._width, self._height, result)
        self._layers.append(layer)
        self._active_layer = layer
        self.changed.emit()

    # --- Compositing ---

    @staticmethod
    def _blend_image(image: np.ndarray, opacity: float, result: np.ndarray):
        src = image.astype(np.float32)
        alpha = src[:, :, 3:4] / 255.0 * opacity
        inv_alpha = 1.0 - alpha
        result[:, :, :3] = src[:, :, :3] * alpha + result[:, :, :3] * inv_alpha
        result[:, :, 3:4] = alpha * 255.0 + result[:, :, 3:4] * inv_alpha

    @staticmethod
    def _blend_buffer(src_buf: np.ndarray, opacity: float, result: np.ndarray):
        alpha = src_buf[:, :, 3:4] / 255.0 * opacity
        inv_alpha = 1.0 - alpha
        result[:, :, :3] = src_buf[:, :, :3] * alpha + result[:, :, :3] * inv_alpha
        result[:, :, 3:4] = alpha * 255.0 + result[:, :, 3:4] * inv_alpha

    def _composite_subtree(self, layer: Layer, result: np.ndarray):
        if not layer.visible or layer.opacity <= 0:
            return
        if layer.children:
            group_buf = np.zeros((self._height, self._width, 4), dtype=np.float32)
            for child in reversed(layer.children):
                self._composite_subtree(child, group_buf)
            self._blend_image(layer.image, 1.0, group_buf)
            self._blend_buffer(group_buf, layer.opacity, result)
        else:
            self._blend_image(layer.image, layer.opacity, result)

    def _composite_subtree_until(self, layer: Layer, result: np.ndarray,
                                  target: Layer) -> bool:
        """Composite bottom-up, stop when target is reached. Returns True if found."""
        if layer is target:
            # Target found — still composite its children (they render below
            # the target's own image), but skip the target's own image.
            if layer.visible and layer.opacity > 0 and layer.children:
                group_buf = np.zeros((self._height, self._width, 4), dtype=np.float32)
                for child in reversed(layer.children):
                    self._composite_subtree(child, group_buf)
                self._blend_buffer(group_buf, layer.opacity, result)
            return True
        if not layer.visible or layer.opacity <= 0:
            return False
        if layer.children:
            group_buf = np.zeros((self._height, self._width, 4), dtype=np.float32)
            for child in reversed(layer.children):  # bottom to top
                found = self._composite_subtree_until(child, group_buf, target)
                if found:
                    # Target found — blend accumulated children, skip parent image
                    self._blend_buffer(group_buf, layer.opacity, result)
                    return True
            # Target not in this subtree — composite normally
            self._blend_image(layer.image, 1.0, group_buf)
            self._blend_buffer(group_buf, layer.opacity, result)
            return False
        else:
            self._blend_image(layer.image, layer.opacity, result)
            return False

    def composite(self, exclude_layer: Layer | None = None) -> np.ndarray:
        """Composite visible layers in the tree.

        If exclude_layer is set, composites only what renders below that layer.
        """
        if not self._layers or self._width == 0:
            return np.zeros((1, 1, 4), dtype=np.uint8)

        result = np.zeros((self._height, self._width, 4), dtype=np.float32)
        if exclude_layer is None:
            for layer in reversed(self._layers):
                self._composite_subtree(layer, result)
        else:
            for layer in reversed(self._layers):  # bottom to top
                found = self._composite_subtree_until(layer, result, exclude_layer)
                if found:
                    break
        return np.clip(result, 0, 255).astype(np.uint8)

    # --- Serialization ---

    FORMAT_VERSION = 2

    def _find_layer_path(self, target: Layer | None) -> str | None:
        if target is None:
            return None
        def _search(layers, prefix):
            for i, layer in enumerate(layers):
                path = f"{prefix}/{i}" if prefix else str(i)
                if layer is target:
                    return path
                result = _search(layer.children, path)
                if result is not None:
                    return result
            return None
        return _search(self._layers, "")

    def _find_layer_by_path(self, path: str) -> Layer | None:
        if not path:
            return None
        parts = [int(p) for p in path.split("/")]
        layers = self._layers
        layer = None
        for idx in parts:
            if 0 <= idx < len(layers):
                layer = layers[idx]
                layers = layer.children
            else:
                return None
        return layer

    def save_project(self, path: str):
        manifest = {
            "format_version": self.FORMAT_VERSION,
            "canvas_width": self._width,
            "canvas_height": self._height,
            "active_layer_path": self._find_layer_path(self._active_layer),
            "layers": [],
        }
        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            for i, layer in enumerate(self._layers):
                layer_path = str(i)
                manifest["layers"].append(layer.to_dict(layer_path))
                layer.save_images_to_zip(zf, layer_path)
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
                layer = _layer_from_dict(layer_dict, zf)
                new_layers.append(layer)

        self._layers.clear()
        self._layers.extend(new_layers)
        self._width = manifest["canvas_width"]
        self._height = manifest["canvas_height"]

        # Restore active layer (v2: by path, v1: by index)
        active_path = manifest.get("active_layer_path")
        if active_path is not None:
            self._active_layer = self._find_layer_by_path(active_path)
        else:
            idx = manifest.get("active_index", 0)
            if 0 <= idx < len(self._layers):
                self._active_layer = self._layers[idx]
            else:
                self._active_layer = None

        if self._active_layer is None and self._layers:
            self._active_layer = self._layers[0]
        self.changed.emit()
