import io
import json
import logging
import zipfile

import numpy as np
from PIL import Image

from .tiles import DenseTileGrid

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Serialization helpers: raw numpy (.npy) with PNG fallback for old projects
# ---------------------------------------------------------------------------

def _save_array_to_zip(zf: zipfile.ZipFile, path: str, arr: np.ndarray):
    """Save numpy array into zip as .npy."""
    buf = io.BytesIO()
    np.save(buf, arr)
    zf.writestr(path, buf.getvalue())


def _load_array_from_zip(zf: zipfile.ZipFile, path: str,
                          mode: str | None = None) -> np.ndarray:
    """Load numpy array from zip. Supports .npy and legacy .png."""
    data = zf.read(path)
    if path.endswith('.npy'):
        return np.load(io.BytesIO(data))
    # Legacy PNG format
    img = Image.open(io.BytesIO(data))
    if mode:
        img = img.convert(mode)
    return np.array(img, dtype=np.uint8)


def _load_pil_from_zip(zf: zipfile.ZipFile, path: str,
                        mode: str = "RGB") -> Image.Image:
    """Load PIL Image from zip. Supports .npy and legacy .png."""
    data = zf.read(path)
    if path.endswith('.npy'):
        arr = np.load(io.BytesIO(data))
        return Image.fromarray(arr).convert(mode)
    return Image.open(io.BytesIO(data)).convert(mode)


class Layer:
    def __init__(self, name: str, width: int, height: int,
                 image: np.ndarray = None, tile_size: int = 256):
        self.name = name
        self.visible = True
        self.opacity = 1.0
        self.children: list['Layer'] = []
        self.parent: 'Layer | None' = None
        if image is not None:
            arr = np.ascontiguousarray(image.astype(np.uint8))
        else:
            arr = np.zeros((height, width, 4), dtype=np.uint8)
        self.content = DenseTileGrid.from_array(arr, tile_size=tile_size)
        # Keep .image for compatibility with existing tools (dense view)
        self.image = self.content.array

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
        return self.content.width

    @property
    def height(self):
        return self.content.height

    def to_dict(self, path: str) -> dict:
        file_key = path.replace("/", "_")
        d = {
            "path": path,
            "type": "layer",
            "name": self.name,
            "visible": self.visible,
            "opacity": self.opacity,
            "image_file": f"layers/{file_key}_image.npy",
            "children": [],
        }
        for i, child in enumerate(self.children):
            d["children"].append(child.to_dict(f"{path}/{i}"))
        return d

    def save_images_to_zip(self, zf: zipfile.ZipFile, path: str):
        file_key = path.replace("/", "_")
        _save_array_to_zip(zf, f"layers/{file_key}_image.npy", self.image)
        for i, child in enumerate(self.children):
            child.save_images_to_zip(zf, f"{path}/{i}")

    @classmethod
    def from_dict(cls, d: dict, zf: zipfile.ZipFile, tile_size: int = 256) -> "Layer":
        arr = _load_array_from_zip(zf, d["image_file"], mode="RGBA")
        layer = cls.__new__(cls)
        layer.name = d["name"]
        layer.visible = d["visible"]
        layer.opacity = d["opacity"]
        layer.content = DenseTileGrid.from_array(arr, tile_size=tile_size)
        layer.image = layer.content.array
        layer.children = []
        layer.parent = None
        for child_dict in d.get("children", []):
            child = _layer_from_dict(child_dict, zf, tile_size=tile_size)
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
                 mode: str = "inpaint",
                 tile_size: int = 256):
        super().__init__(name, width, height, tile_size=tile_size)
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
        self.manual_patch_rect = None  # (x0, y0, x1, y1) or None — explicit patch area
        self.resize_to_model_resolution = False

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
        d["mask_file"] = f"layers/{file_key}_mask.npy"
        d["source_file"] = f"layers/{file_key}_source.npy" if self.source_patch is not None else None
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
        d["manual_patch_rect"] = list(self.manual_patch_rect) if self.manual_patch_rect else None
        d["resize_to_model_resolution"] = self.resize_to_model_resolution
        return d

    def save_images_to_zip(self, zf: zipfile.ZipFile, path: str):
        super().save_images_to_zip(zf, path)
        file_key = path.replace("/", "_")
        _save_array_to_zip(zf, f"layers/{file_key}_mask.npy", self.mask)
        if self.source_patch is not None:
            _save_array_to_zip(zf, f"layers/{file_key}_source.npy",
                               np.array(self.source_patch))

    @classmethod
    def from_dict(cls, d: dict, zf: zipfile.ZipFile, tile_size: int = 256) -> "DiffusionLayer":
        arr = _load_array_from_zip(zf, d["image_file"], mode="RGBA")
        h, w = arr.shape[:2]

        mask_arr = _load_array_from_zip(zf, d["mask_file"], mode="L")

        source_patch = None
        if d.get("source_file") and d["source_file"] in zf.namelist():
            source_patch = _load_pil_from_zip(zf, d["source_file"], mode="RGB")

        layer = cls.__new__(cls)
        layer.name = d["name"]
        layer.visible = d["visible"]
        layer.opacity = d["opacity"]
        layer.content = DenseTileGrid.from_array(arr, tile_size=tile_size)
        layer.image = layer.content.array
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
        layer.mode = d.get("mode", "inpaint")
        rect = d.get("ip_adapter_rect")
        layer.ip_adapter_rect = tuple(rect) if rect else None
        layer.ip_adapter_scale = d.get("ip_adapter_scale", 0.6)
        layer.masked_content = d.get("masked_content", "original")
        mpr = d.get("manual_patch_rect")
        layer.manual_patch_rect = tuple(mpr) if mpr else None
        layer.resize_to_model_resolution = d.get("resize_to_model_resolution", False)
        for child_dict in d.get("children", []):
            child = _layer_from_dict(child_dict, zf, tile_size=tile_size)
            child.parent = layer
            layer.children.append(child)
        return layer


class LamaLayer(Layer):
    def __init__(self, name: str, width: int, height: int,
                 source_patch: Image.Image | None,
                 patch_x: int, patch_y: int, patch_w: int, patch_h: int,
                 tile_size: int = 256):
        super().__init__(name, width, height, tile_size=tile_size)
        self.source_patch = source_patch
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.mask = np.zeros((height, width), dtype=np.uint8)

    def clear_mask(self):
        self.mask[:] = 0

    def has_mask(self) -> bool:
        return np.any(self.mask > 0)

    def mask_bbox(self) -> tuple[int, int, int, int] | None:
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
        d["type"] = "lama"
        d["mask_file"] = f"layers/{file_key}_mask.npy"
        d["source_file"] = f"layers/{file_key}_source.npy" if self.source_patch is not None else None
        d["patch_x"] = self.patch_x
        d["patch_y"] = self.patch_y
        d["patch_w"] = self.patch_w
        d["patch_h"] = self.patch_h
        return d

    def save_images_to_zip(self, zf: zipfile.ZipFile, path: str):
        super().save_images_to_zip(zf, path)
        file_key = path.replace("/", "_")
        _save_array_to_zip(zf, f"layers/{file_key}_mask.npy", self.mask)
        if self.source_patch is not None:
            _save_array_to_zip(zf, f"layers/{file_key}_source.npy",
                               np.array(self.source_patch))

    @classmethod
    def from_dict(cls, d: dict, zf: zipfile.ZipFile, tile_size: int = 256) -> "LamaLayer":
        arr = _load_array_from_zip(zf, d["image_file"], mode="RGBA")
        h, w = arr.shape[:2]

        mask_arr = _load_array_from_zip(zf, d["mask_file"], mode="L")

        source_patch = None
        if d.get("source_file") and d["source_file"] in zf.namelist():
            source_patch = _load_pil_from_zip(zf, d["source_file"], mode="RGB")

        layer = cls.__new__(cls)
        layer.name = d["name"]
        layer.visible = d["visible"]
        layer.opacity = d["opacity"]
        layer.content = DenseTileGrid.from_array(arr, tile_size=tile_size)
        layer.image = layer.content.array
        layer.children = []
        layer.parent = None
        layer.mask = np.ascontiguousarray(mask_arr)
        layer.source_patch = source_patch
        layer.patch_x = d["patch_x"]
        layer.patch_y = d["patch_y"]
        layer.patch_w = d["patch_w"]
        layer.patch_h = d["patch_h"]
        for child_dict in d.get("children", []):
            child = _layer_from_dict(child_dict, zf, tile_size=tile_size)
            child.parent = layer
            layer.children.append(child)
        return layer


class InstructLayer(Layer):
    def __init__(self, name: str, width: int, height: int,
                 source_patch: Image.Image | None,
                 patch_x: int, patch_y: int, patch_w: int, patch_h: int,
                 instruction: str = "",
                 image_guidance_scale: float = 1.5,
                 guidance_scale: float = 7.0,
                 steps: int = 20,
                 seed: int = -1,
                 tile_size: int = 256):
        super().__init__(name, width, height, tile_size=tile_size)
        self.source_patch = source_patch
        self.patch_x = patch_x
        self.patch_y = patch_y
        self.patch_w = patch_w
        self.patch_h = patch_h
        self.instruction = instruction
        self.image_guidance_scale = image_guidance_scale
        self.guidance_scale = guidance_scale
        self.steps = steps
        self.seed = seed
        self.mask = np.zeros((height, width), dtype=np.uint8)
        self.manual_patch_rect = None  # (x0, y0, x1, y1) or None

    def clear_mask(self):
        self.mask[:] = 0

    def has_mask(self) -> bool:
        return np.any(self.mask > 0)

    def mask_bbox(self) -> tuple[int, int, int, int] | None:
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
        d["type"] = "instruct"
        d["mask_file"] = f"layers/{file_key}_mask.npy"
        d["source_file"] = f"layers/{file_key}_source.npy" if self.source_patch is not None else None
        d["patch_x"] = self.patch_x
        d["patch_y"] = self.patch_y
        d["patch_w"] = self.patch_w
        d["patch_h"] = self.patch_h
        d["instruction"] = self.instruction
        d["image_guidance_scale"] = self.image_guidance_scale
        d["guidance_scale"] = self.guidance_scale
        d["steps"] = self.steps
        d["seed"] = self.seed
        d["manual_patch_rect"] = list(self.manual_patch_rect) if self.manual_patch_rect else None
        return d

    def save_images_to_zip(self, zf: zipfile.ZipFile, path: str):
        super().save_images_to_zip(zf, path)
        file_key = path.replace("/", "_")
        _save_array_to_zip(zf, f"layers/{file_key}_mask.npy", self.mask)
        if self.source_patch is not None:
            _save_array_to_zip(zf, f"layers/{file_key}_source.npy",
                               np.array(self.source_patch))

    @classmethod
    def from_dict(cls, d: dict, zf: zipfile.ZipFile, tile_size: int = 256) -> "InstructLayer":
        arr = _load_array_from_zip(zf, d["image_file"], mode="RGBA")
        h, w = arr.shape[:2]

        mask_arr = np.zeros((h, w), dtype=np.uint8)
        if d.get("mask_file") and d["mask_file"] in zf.namelist():
            mask_arr = _load_array_from_zip(zf, d["mask_file"], mode="L")

        source_patch = None
        if d.get("source_file") and d["source_file"] in zf.namelist():
            source_patch = _load_pil_from_zip(zf, d["source_file"], mode="RGB")

        layer = cls.__new__(cls)
        layer.name = d["name"]
        layer.visible = d["visible"]
        layer.opacity = d["opacity"]
        layer.content = DenseTileGrid.from_array(arr, tile_size=tile_size)
        layer.image = layer.content.array
        layer.children = []
        layer.parent = None
        layer.mask = np.ascontiguousarray(mask_arr)
        layer.source_patch = source_patch
        layer.patch_x = d["patch_x"]
        layer.patch_y = d["patch_y"]
        layer.patch_w = d["patch_w"]
        layer.patch_h = d["patch_h"]
        layer.instruction = d.get("instruction", "")
        layer.image_guidance_scale = d.get("image_guidance_scale", 1.5)
        layer.guidance_scale = d.get("guidance_scale", 7.0)
        layer.steps = d.get("steps", 20)
        layer.seed = d.get("seed", -1)
        mpr = d.get("manual_patch_rect")
        layer.manual_patch_rect = tuple(mpr) if mpr else None
        for child_dict in d.get("children", []):
            child = _layer_from_dict(child_dict, zf, tile_size=tile_size)
            child.parent = layer
            layer.children.append(child)
        return layer


def _layer_from_dict(d: dict, zf: zipfile.ZipFile, tile_size: int = 256) -> Layer:
    """Dispatch layer deserialization by type."""
    if d.get("type") == "diffusion":
        return DiffusionLayer.from_dict(d, zf, tile_size=tile_size)
    if d.get("type") == "lama":
        return LamaLayer.from_dict(d, zf, tile_size=tile_size)
    if d.get("type") == "instruct":
        return InstructLayer.from_dict(d, zf, tile_size=tile_size)
    return Layer.from_dict(d, zf, tile_size=tile_size)

