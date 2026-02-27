import json
import re
import zipfile

import numpy as np

from .layer import Layer, _layer_from_dict
from .layer_renderer import LayerRenderer


class LayerStack:
    def __init__(self, tile_size: int = 256):
        self._layers: list[Layer] = []  # root-level layers
        self._active_layer: Layer | None = None
        self._width = 0
        self._height = 0
        self._tile_size = tile_size
        self.on_changed: callable = None
        self._renderer = LayerRenderer(self)

    # --- Tree traversal ---

    def _all_layers_flat(self) -> list[Layer]:
        """All layers in depth-first order (index 0 = topmost)."""
        result = []
        for layer in self._layers:
            result.append(layer)
            result.extend(layer.all_descendants())
        return result

    def next_name(self, prefix: str) -> str:
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
    def tile_size(self):
        return self._tile_size

    @property
    def tiles_x(self) -> int:
        if self._width == 0:
            return 0
        return (self._width + self._tile_size - 1) // self._tile_size

    @property
    def tiles_y(self) -> int:
        if self._height == 0:
            return 0
        return (self._height + self._tile_size - 1) // self._tile_size

    def tile_bounds(self, tx: int, ty: int) -> tuple[int, int, int, int]:
        x0 = tx * self._tile_size
        y0 = ty * self._tile_size
        x1 = min(self._width, x0 + self._tile_size)
        y1 = min(self._height, y0 + self._tile_size)
        return x0, y0, x1, y1

    def tile_shape(self, tx: int, ty: int) -> tuple[int, int]:
        x0, y0, x1, y1 = self.tile_bounds(tx, ty)
        return (y1 - y0, x1 - x0)

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
            if self.on_changed:
                self.on_changed()

    def init_from_image(self, image: np.ndarray):
        self._layers.clear()
        h, w = image.shape[:2]
        self._width = w
        self._height = h
        layer = Layer("Background", w, h, image, tile_size=self._tile_size)
        self._layers.append(layer)
        self._active_layer = layer
        self._rebuild_caches()
        if self.on_changed:
            self.on_changed()

    def _insert_near_active(self, layer: Layer):
        """Insert layer as a sibling above the active layer."""
        self._apply_tile_size(layer)
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

    def _apply_tile_size(self, layer: Layer):
        if hasattr(layer, "content"):
            layer.content.tile_size = self._tile_size
        for child in layer.children:
            self._apply_tile_size(child)

    def add_layer(self, name: str, image: np.ndarray = None):
        if self._width == 0 or self._height == 0:
            return
        layer = Layer(name, self._width, self._height, image, tile_size=self._tile_size)
        self._insert_near_active(layer)
        self._rebuild_caches()
        if self.on_changed:
            self.on_changed()

    def insert_layer(self, layer: Layer):
        if self._width == 0 or self._height == 0:
            return
        self._insert_near_active(layer)
        self._rebuild_caches()
        if self.on_changed:
            self.on_changed()

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
        self._rebuild_caches()
        if self.on_changed:
            self.on_changed()

    def move_layer(self, layer: Layer, new_parent: Layer | None, index: int):
        """Move layer to new_parent at index (or root if new_parent is None)."""
        if layer.parent is not None:
            layer.parent.remove_child(layer)
        elif layer in self._layers:
            self._layers.remove(layer)
        if new_parent is not None:
            new_parent.add_child(layer, index)
        else:
            self._layers.insert(index, layer)
        self._active_layer = layer
        self._rebuild_caches()
        if self.on_changed:
            self.on_changed()

    def set_visibility(self, layer: Layer, visible: bool):
        layer.visible = visible
        self.mark_layer_dirty(layer)
        if self.on_changed:
            self.on_changed()

    def set_opacity(self, layer: Layer, opacity: float):
        """Set layer opacity with prefix invalidation."""
        layer.opacity = opacity
        self.mark_layer_dirty(layer)
        if self.on_changed:
            self.on_changed()

    def flatten(self):
        result = self.composite()
        self._layers.clear()
        layer = Layer("Background", self._width, self._height, result, tile_size=self._tile_size)
        self._layers.append(layer)
        self._active_layer = layer
        self._rebuild_caches()
        if self.on_changed:
            self.on_changed()

    # --- Prefix cache management ---

    def _rebuild_caches(self):
        """Reset all renderer caches (after structural changes)."""
        self._renderer.reset_cache()

    def _siblings_of(self, layer: Layer) -> list[Layer]:
        """Return the siblings list containing layer (root list or parent.children)."""
        if layer.parent is not None:
            return layer.parent.children
        return self._layers

    def _comp_order_siblings(self, layer: Layer) -> list[Layer]:
        """Siblings in compositing order (bottom to top = reversed list order)."""
        return list(reversed(self._siblings_of(layer)))

    def comp_order_siblings(self, layer: Layer) -> list[Layer]:
        """Public wrapper used by renderer."""
        return self._comp_order_siblings(layer)

    def _invalidate(self, layer: Layer):
        """Mark a single layer as dirty and clear its cache."""
        self._renderer.invalidate_tiles({layer}, tiles=None)

    def mark_layer_dirty(self, layer: Layer, rect: tuple[int, int, int, int] | None = None):
        """Public: call when a layer's content/visibility/opacity changed."""
        affected = self._collect_affected_layers(layer)
        if not affected:
            self._rebuild_caches()
            return
        tiles = self._tiles_for_rect(rect)
        self._renderer.invalidate_tiles(affected, tiles)

    def _tiles_for_rect(self, rect: tuple[int, int, int, int] | None
                        ) -> set[tuple[int, int]] | None:
        if rect is None:
            return None
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:
            return set()
        tx0 = max(0, x0 // self._tile_size)
        ty0 = max(0, y0 // self._tile_size)
        tx1 = min(self.tiles_x - 1, (x1 - 1) // self._tile_size)
        ty1 = min(self.tiles_y - 1, (y1 - 1) // self._tile_size)
        tiles: set[tuple[int, int]] = set()
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tiles.add((tx, ty))
        return tiles

    def _collect_affected_layers(self, layer: Layer) -> set[Layer]:
        affected: set[Layer] = set()
        try:
            siblings = self._comp_order_siblings(layer)
            idx = siblings.index(layer)
        except ValueError:
            return affected
        for i in range(idx, len(siblings)):
            affected.add(siblings[i])
        cur = layer.parent
        while cur is not None:
            parent_siblings = self._comp_order_siblings(cur)
            try:
                pidx = parent_siblings.index(cur)
            except ValueError:
                break
            for i in range(pidx, len(parent_siblings)):
                affected.add(parent_siblings[i])
            cur = cur.parent
        if layer.parent is not None:
            root = layer
            while root.parent is not None:
                root = root.parent
            root_siblings = self._comp_order_siblings(root)
            try:
                ridx = root_siblings.index(root)
            except ValueError:
                return affected
            for i in range(ridx, len(root_siblings)):
                affected.add(root_siblings[i])
        return affected

    # --- Compositing ---

    def composite(self, exclude_layer: Layer | None = None) -> np.ndarray:
        """Composite visible layers.

        If exclude_layer is set, returns prefix of that layer (everything below it).
        """
        if not self._layers or self._width == 0:
            return np.zeros((1, 1, 4), dtype=np.uint8)

        if exclude_layer is not None:
            return self.get_prefix_below(exclude_layer).copy()

        return self._renderer.composite_full()

    def get_prefix_below(self, layer: Layer) -> np.ndarray:
        return self._renderer.prefix_full(layer)

    def get_prefix_below_rect(self, layer: Layer, x0: int, y0: int,
                              x1: int, y1: int) -> np.ndarray:
        """Return prefix buffer for a rect (uint8 RGBA)."""
        if self._width == 0 or self._height == 0:
            return np.zeros((1, 1, 4), dtype=np.uint8)
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self._width, x1)
        y1 = min(self._height, y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 4), dtype=np.uint8)

        out = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.uint8)
        tiles = self._tiles_for_rect((x0, y0, x1, y1))
        if not tiles:
            return out
        for tx, ty in tiles:
            tile = self._renderer.full_prefix_tile(layer, tx, ty)
            if tile is None:
                continue
            bx0, by0, bx1, by1 = self.tile_bounds(tx, ty)
            ox0 = max(x0, bx0)
            oy0 = max(y0, by0)
            ox1 = min(x1, bx1)
            oy1 = min(y1, by1)
            if ox1 <= ox0 or oy1 <= oy0:
                continue
            src_x0 = ox0 - bx0
            src_y0 = oy0 - by0
            src_x1 = src_x0 + (ox1 - ox0)
            src_y1 = src_y0 + (oy1 - oy0)
            dst_x0 = ox0 - x0
            dst_y0 = oy0 - y0
            dst_x1 = dst_x0 + (ox1 - ox0)
            dst_y1 = dst_y0 + (oy1 - oy0)
            out[dst_y0:dst_y1, dst_x0:dst_x1] = tile[src_y0:src_y1, src_x0:src_x1]
        return out

    # --- Serialization ---

    FORMAT_VERSION = 4

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
            "tile_size": self._tile_size,
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

            self._tile_size = manifest.get("tile_size", self._tile_size)
            new_layers = []
            for layer_dict in manifest["layers"]:
                layer = _layer_from_dict(layer_dict, zf, tile_size=self._tile_size)
                new_layers.append(layer)

        self._layers.clear()
        self._layers.extend(new_layers)
        for layer in self._layers:
            self._apply_tile_size(layer)
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
        self._rebuild_caches()
        if self.on_changed:
            self.on_changed()
