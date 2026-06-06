import logging
import uuid
import zipfile

import numpy as np

from .archive_serialization import load_array_from_zip, save_array_to_zip
from .tiles import DenseTileGrid
from .mask import Mask
from .tool import Tool
from .tool_serialization import (
    load_tool,
    save_tool_assets,
    serialize_tool,
)

logger = logging.getLogger(__name__)


def new_layer_id() -> str:
    return f"layer_{uuid.uuid4().hex}"


class Layer:
    def __init__(self, name: str, width: int, height: int,
                 image: np.ndarray = None, tile_size: int = 256,
                 x: int = 0, y: int = 0, layer_id: str | None = None):
        self.id = layer_id or new_layer_id()
        self.name = name
        self.visible = True
        self.opacity = 1.0
        self.x = int(x)
        self.y = int(y)
        self.children: list['Layer'] = []
        self.parent: 'Layer | None' = None
        self.patch_rect: tuple[int, int, int, int] | None = None
        if image is not None:
            arr = np.ascontiguousarray(image.astype(np.uint8))
        else:
            arr = np.zeros((height, width, 4), dtype=np.uint8)
        self.content = DenseTileGrid.from_array(arr, tile_size=tile_size)
        self.image = self.content.array
        self.mask = Mask.zeros(height, width)
        self.tool: Tool | None = None

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

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.x + self.width, self.y + self.height)

    def local_rect_to_canvas(self, rect: tuple[int, int, int, int]
                             ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = rect
        return (x0 + self.x, y0 + self.y, x1 + self.x, y1 + self.y)

    def canvas_rect_to_local(self, rect: tuple[int, int, int, int]
                             ) -> tuple[int, int, int, int]:
        x0, y0, x1, y1 = rect
        return (x0 - self.x, y0 - self.y, x1 - self.x, y1 - self.y)

    def clear_mask(self):
        self.mask.clear()

    def has_mask(self) -> bool:
        return not self.mask.is_empty

    def mask_bbox(self) -> tuple[int, int, int, int] | None:
        bbox = self.mask.bbox()
        if bbox is None:
            return None
        return self.local_rect_to_canvas(bbox)

    def mask_center(self) -> tuple[int, int] | None:
        center = self.mask.center()
        if center is None:
            return None
        return (center[0] + self.x, center[1] + self.y)

    def to_dict(self, path: str) -> dict:
        file_key = path.replace("/", "_")
        d = {
            "path": path,
            "type": "layer",
            "id": self.id,
            "name": self.name,
            "visible": self.visible,
            "opacity": self.opacity,
            "x": self.x,
            "y": self.y,
            "patch_rect": list(self.patch_rect) if self.patch_rect else None,
            "image_file": f"layers/{file_key}_image.npy",
            "children": [],
        }
        if not self.mask.is_empty:
            d["mask_file"] = f"layers/{file_key}_mask.npy"
        if self.tool is not None:
            d["tool"] = serialize_tool(self.tool, file_key)
        for i, child in enumerate(self.children):
            d["children"].append(child.to_dict(f"{path}/{i}"))
        return d

    def save_images_to_zip(self, zf: zipfile.ZipFile, path: str):
        file_key = path.replace("/", "_")
        save_array_to_zip(zf, f"layers/{file_key}_image.npy", self.image)
        if not self.mask.is_empty:
            save_array_to_zip(
                zf,
                f"layers/{file_key}_mask.npy",
                self.mask.data.astype(np.float32),
            )
        if self.tool is not None:
            save_tool_assets(self.tool, zf, file_key)
        for i, child in enumerate(self.children):
            child.save_images_to_zip(zf, f"{path}/{i}")

    @classmethod
    def _from_dict_base(cls, d: dict, zf: zipfile.ZipFile,
                        tile_size: int = 256) -> 'Layer':
        """Shared base deserialization for all layer types."""
        arr = load_array_from_zip(zf, d["image_file"], mode="RGBA")
        layer = cls.__new__(cls)
        layer.id = d.get("id") or new_layer_id()
        layer.name = d["name"]
        layer.visible = d["visible"]
        layer.opacity = d["opacity"]
        layer.x = int(d.get("x", 0))
        layer.y = int(d.get("y", 0))
        patch_rect = d.get("patch_rect")
        layer.patch_rect = tuple(patch_rect) if patch_rect else None
        layer.content = DenseTileGrid.from_array(arr, tile_size=tile_size)
        layer.image = layer.content.array
        layer.children = []
        layer.parent = None

        mask_file = d.get("mask_file")
        if mask_file is None and isinstance(d.get("tool"), dict):
            # Projects saved by the first tool-based format kept mask_file
            # inside the tool dict. Migrate it to the layer-owned mask model.
            mask_file = d["tool"].get("mask_file")
        if mask_file and mask_file in zf.namelist():
            mask_arr = load_array_from_zip(zf, mask_file, mode="L")
            if mask_arr.dtype == np.uint8:
                layer.mask = Mask.from_uint8(mask_arr)
            else:
                layer.mask = Mask(mask_arr)
        else:
            layer.mask = Mask.zeros(layer.height, layer.width)

        layer.tool = None
        if "tool" in d:
            result = load_tool(d["tool"], zf)
            layer.tool = result.tool
            if layer.patch_rect is None and result.legacy_patch_rect is not None:
                layer.patch_rect = result.legacy_patch_rect

        for child_dict in d.get("children", []):
            child = _layer_from_dict(child_dict, zf, tile_size=tile_size)
            child.parent = layer
            layer.children.append(child)
        return layer

    @classmethod
    def from_dict(cls, d: dict, zf: zipfile.ZipFile, tile_size: int = 256) -> "Layer":
        return cls._from_dict_base(d, zf, tile_size=tile_size)


def _layer_from_dict(d: dict, zf: zipfile.ZipFile, tile_size: int = 256) -> Layer:
    """Dispatch layer deserialization by type.

    Supports both the new unified format (type=layer + optional tool sub-dict)
    and the legacy format (type=diffusion/lama/instruct with tool data inline).
    """
    layer_type = d.get("type", "layer")

    if layer_type in ("diffusion", "lama", "instruct"):
        # Legacy format: tool data is inline in the layer dict.
        # Load as a plain Layer, then attach the tool from the same dict.
        layer = Layer._from_dict_base(d, zf, tile_size=tile_size)
        result = load_tool(d, zf)
        layer.tool = result.tool
        if layer.patch_rect is None and result.legacy_patch_rect is not None:
            layer.patch_rect = result.legacy_patch_rect
        return layer

    return Layer._from_dict_base(d, zf, tile_size=tile_size)
