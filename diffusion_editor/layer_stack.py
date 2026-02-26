import json
import re
import zipfile

import numpy as np

from .layer import Layer, _layer_from_dict


class LayerStack:
    def __init__(self):
        self._layers: list[Layer] = []  # root-level layers
        self._active_layer: Layer | None = None
        self._width = 0
        self._height = 0
        self.on_changed: callable = None
        # Per-layer prefix cache (uint8 RGBA), lazy.
        # prefix(L) = Прошлый + Вложенный = всё что ниже L, без L.
        self._prefix: dict[Layer, np.ndarray | None] = {}
        self._nested: dict[Layer, np.ndarray | None] = {}
        self._dirty: set[Layer] = set()
        self._nested_dirty: set[Layer] = set()

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
        layer = Layer("Background", w, h, image)
        self._layers.append(layer)
        self._active_layer = layer
        self._rebuild_caches()
        if self.on_changed:
            self.on_changed()

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
        layer = Layer("Background", self._width, self._height, result)
        self._layers.append(layer)
        self._active_layer = layer
        self._rebuild_caches()
        if self.on_changed:
            self.on_changed()

    # --- Prefix cache management ---

    def _rebuild_caches(self):
        """Reset all prefix caches (after structural changes)."""
        self._prefix.clear()
        self._nested.clear()
        self._dirty.clear()
        self._nested_dirty.clear()
        for layer in self._all_layers_flat():
            self._prefix[layer] = None
            self._nested[layer] = None
            self._dirty.add(layer)
            self._nested_dirty.add(layer)

    def _siblings_of(self, layer: Layer) -> list[Layer]:
        """Return the siblings list containing layer (root list or parent.children)."""
        if layer.parent is not None:
            return layer.parent.children
        return self._layers

    def _comp_order_siblings(self, layer: Layer) -> list[Layer]:
        """Siblings in compositing order (bottom to top = reversed list order)."""
        return list(reversed(self._siblings_of(layer)))

    def _invalidate(self, layer: Layer):
        """Mark a single layer as dirty and clear its cache."""
        self._dirty.add(layer)
        self._nested_dirty.add(layer)
        self._prefix[layer] = None
        self._nested[layer] = None

    def _ensure_nested(self, layer: Layer):
        """Compute nested(L) = composite(top child), cached separately."""
        if layer not in self._nested_dirty:
            return
        if layer.children:
            top_child = layer.children[0]  # children[0] = topmost
            nested = self._composite_of(top_child)
        else:
            nested = None
        self._nested[layer] = nested
        self._nested_dirty.discard(layer)

    def mark_layer_dirty(self, layer: Layer):
        """Public: call when a layer's content/visibility/opacity changed."""
        if layer not in self._prefix:
            self._rebuild_caches()
            return
        # Invalidate layer and all siblings above it
        siblings = self._comp_order_siblings(layer)
        try:
            idx = siblings.index(layer)
        except ValueError:
            self._rebuild_caches()
            return
        for i in range(idx, len(siblings)):
            self._invalidate(siblings[i])
        # Propagate up to parent
        cur = layer.parent
        while cur is not None:
            parent_siblings = self._comp_order_siblings(cur)
            try:
                pidx = parent_siblings.index(cur)
            except ValueError:
                break
            for i in range(pidx, len(parent_siblings)):
                self._invalidate(parent_siblings[i])
            cur = cur.parent
        # Invalidate root level if layer is nested
        if layer.parent is not None:
            root = layer
            while root.parent is not None:
                root = root.parent
            root_siblings = self._comp_order_siblings(root)
            try:
                ridx = root_siblings.index(root)
            except ValueError:
                return
            for i in range(ridx, len(root_siblings)):
                self._invalidate(root_siblings[i])

    def _ensure_prefix(self, layer: Layer):
        """Вычислить prefix(L) = Прошлый + Вложенный.

        Первый sibling получает пустой Прошлый (без наследования от родителя,
        чтобы избежать циклической зависимости).
        """
        if layer not in self._dirty:
            return

        siblings = self._comp_order_siblings(layer)
        idx = siblings.index(layer)

        # Прошлый: composite предыдущего sibling (или пусто для первого)
        if idx > 0:
            previous = self._composite_of(siblings[idx - 1])
        else:
            previous = None

        # Вложенный: composite верхнего ребёнка (или пусто если нет детей)
        self._ensure_nested(layer)
        nested = self._nested.get(layer)

        # prefix = Прошлый + Вложенный
        result = np.zeros((self._height, self._width, 4), dtype=np.float32)
        if previous is not None:
            result = previous.astype(np.float32)
        if nested is not None:
            self._blend_buffer(nested, 1.0, result)

        self._prefix[layer] = np.clip(result, 0, 255).astype(np.uint8)
        self._dirty.discard(layer)

    def _composite_of(self, layer: Layer) -> np.ndarray | None:
        """composite(L) = Прошлый + blend(Вложенный + L.image, L.opacity).

        Opacity применяется к собственному изображению и вложенным как к единому целому.
        Возвращает uint8. Не включает внешний контекст от родителя.
        """
        self._ensure_prefix(layer)

        if not layer.visible or layer.opacity <= 0:
            # Скрыт/opacity=0: собственное и вложенные не рисуются.
            # Возвращаем только Прошлый (composite предыдущего sibling).
            siblings = self._comp_order_siblings(layer)
            idx = siblings.index(layer)
            if idx > 0:
                return self._composite_of(siblings[idx - 1])
            return None

        prefix = self._prefix.get(layer)

        if layer.opacity >= 1.0:
            # Быстрый путь: prefix + image.
            # При opacity=1.0: Прошлый + blend(Вложенный + image, 1.0)
            #                 = Прошлый + Вложенный + image = prefix + image
            result = np.zeros((self._height, self._width, 4), dtype=np.float32)
            if prefix is not None:
                result = prefix.astype(np.float32)
            self._blend_image(layer.image, 1.0, result)
            return np.clip(result, 0, 255).astype(np.uint8)

        # Дробный opacity: нужны Прошлый и Вложенный отдельно.
        siblings = self._comp_order_siblings(layer)
        idx = siblings.index(layer)
        previous = self._composite_of(siblings[idx - 1]) if idx > 0 else None
        self._ensure_nested(layer)
        nested = self._nested.get(layer)

        # subtree = Вложенный + own_image
        subtree = np.zeros((self._height, self._width, 4), dtype=np.float32)
        if nested is not None:
            subtree = nested.astype(np.float32)
        self._blend_image(layer.image, 1.0, subtree)
        subtree_u8 = np.clip(subtree, 0, 255).astype(np.uint8)

        # composite = Прошлый + blend(subtree, opacity)
        result = np.zeros((self._height, self._width, 4), dtype=np.float32)
        if previous is not None:
            result = previous.astype(np.float32)
        self._blend_buffer(subtree_u8, layer.opacity, result)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _external_context(self, layer: Layer) -> np.ndarray | None:
        """External context = everything from outside the sibling list.

        For root layers: None.
        For children of P: external_context(P) + Прошлый(P).
        """
        if layer.parent is None:
            return None
        parent = layer.parent
        parent_ext = self._external_context(parent)
        # Прошлый of parent = composite of siblings before parent
        siblings = self._comp_order_siblings(parent)
        idx = siblings.index(parent)
        if idx > 0:
            prev_composite = self._composite_of(siblings[idx - 1])
        else:
            prev_composite = None

        if parent_ext is None and prev_composite is None:
            return None
        result = np.zeros((self._height, self._width, 4), dtype=np.float32)
        if parent_ext is not None:
            result = parent_ext.astype(np.float32)
        if prev_composite is not None:
            self._blend_buffer(prev_composite, 1.0, result)
        return np.clip(result, 0, 255).astype(np.uint8)

    def _full_prefix(self, layer: Layer) -> np.ndarray | None:
        """Полный prefix = external_context + prefix (для вложенных слоёв)."""
        self._ensure_prefix(layer)
        local = self._prefix.get(layer)
        ext = self._external_context(layer)

        if ext is None:
            return local
        if local is None:
            return ext

        result = ext.astype(np.float32)
        self._blend_buffer(local, 1.0, result)
        return np.clip(result, 0, 255).astype(np.uint8)

    def get_prefix_below(self, layer: Layer) -> np.ndarray | None:
        """Return full prefix: everything below layer's own image (uint8)."""
        if layer not in self._prefix:
            return None
        result = self._full_prefix(layer)
        if result is None:
            return np.zeros((self._height, self._width, 4), dtype=np.uint8)
        return result

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

    def composite(self, exclude_layer: Layer | None = None) -> np.ndarray:
        """Composite visible layers.

        If exclude_layer is set, returns prefix of that layer (everything below it).
        """
        if not self._layers or self._width == 0:
            return np.zeros((1, 1, 4), dtype=np.uint8)

        if exclude_layer is not None:
            result = self.get_prefix_below(exclude_layer)
            if result is not None:
                return result.copy()
            return np.zeros((self._height, self._width, 4), dtype=np.uint8)

        # Полный composite = composite(top_root).
        # Для корневых слоёв внешний контекст отсутствует.
        top_root = self._layers[0]  # _layers[0] = topmost
        result = self._composite_of(top_root)
        if result is not None:
            return result
        return np.zeros((self._height, self._width, 4), dtype=np.uint8)

    # --- Serialization ---

    FORMAT_VERSION = 3

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
        self._rebuild_caches()
        if self.on_changed:
            self.on_changed()
