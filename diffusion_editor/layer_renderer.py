from __future__ import annotations

import numpy as np

from .layer import Layer


class LayerRenderer:
    def __init__(self, layer_stack: "LayerStack"):
        self._stack = layer_stack
        # Caches per (layer, tx, ty)
        self._prefix_cache: dict[tuple[Layer, int, int], np.ndarray | None] = {}
        self._nested_cache: dict[tuple[Layer, int, int], np.ndarray | None] = {}
        self._composite_cache: dict[tuple[Layer, int, int], np.ndarray | None] = {}

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def reset_cache(self) -> None:
        self._prefix_cache.clear()
        self._nested_cache.clear()
        self._composite_cache.clear()

    def cache_memory_bytes(self) -> int:
        """Estimated memory held by tile caches."""
        total = 0
        for cache in (self._prefix_cache, self._nested_cache,
                      self._composite_cache):
            for v in cache.values():
                if v is not None:
                    total += v.nbytes
        return total

    def invalidate_tiles(self, layers: set[Layer], tiles: set[tuple[int, int]] | None) -> None:
        for layer in layers:
            self._invalidate_layer(layer, tiles)

    def _invalidate_layer(self, layer: Layer, tiles: set[tuple[int, int]] | None) -> None:
        if tiles is None:
            self._drop_layer_from_cache(self._prefix_cache, layer)
            self._drop_layer_from_cache(self._nested_cache, layer)
            self._drop_layer_from_cache(self._composite_cache, layer)
            return
        for tx, ty in tiles:
            self._prefix_cache.pop((layer, tx, ty), None)
            self._nested_cache.pop((layer, tx, ty), None)
            self._composite_cache.pop((layer, tx, ty), None)

    @staticmethod
    def _drop_layer_from_cache(cache: dict, layer: Layer) -> None:
        keys = [k for k in cache.keys() if k[0] is layer]
        for k in keys:
            cache.pop(k, None)

    # ------------------------------------------------------------------
    # Tile compositing
    # ------------------------------------------------------------------

    def composite_full(self) -> np.ndarray:
        h, w = self._stack.height, self._stack.width
        if h == 0 or w == 0:
            return np.zeros((1, 1, 4), dtype=np.uint8)
        result = np.zeros((h, w, 4), dtype=np.uint8)
        for ty in range(self._stack.tiles_y):
            for tx in range(self._stack.tiles_x):
                tile = self.composite_tile(tx, ty)
                if tile is None:
                    continue
                x0, y0, x1, y1 = self._stack.tile_bounds(tx, ty)
                result[y0:y1, x0:x1] = tile
        return result

    def _composite_siblings_direct(self, siblings: list[Layer],
                                   h: int, w: int) -> np.ndarray:
        """Composite sibling list on full images (fast, no tiles)."""
        result = np.zeros((h, w, 4), dtype=np.float32)
        # Scratch buffers reused across layers to avoid repeated allocation
        tmp = np.empty((h, w, 4), dtype=np.float32)
        for layer in reversed(siblings):  # bottom to top
            if not layer.visible or layer.opacity <= 0:
                continue
            if layer.opacity >= 1.0 and not layer.children:
                self._blend_image(layer.image, 1.0, result)
            else:
                subtree = np.zeros((h, w, 4), dtype=np.float32)
                if layer.children:
                    child_comp = self._composite_siblings_direct(
                        layer.children, h, w)
                    subtree[:] = child_comp
                self._blend_image(layer.image, 1.0, subtree)
                np.clip(subtree, 0, 255, out=tmp)
                subtree_u8 = tmp.astype(np.uint8)
                self._blend_buffer(subtree_u8, layer.opacity, result)
        np.clip(result, 0, 255, out=result)
        return result.astype(np.uint8)

    def prefix_full(self, layer: Layer) -> np.ndarray:
        h, w = self._stack.height, self._stack.width
        result = np.zeros((h, w, 4), dtype=np.uint8)
        for ty in range(self._stack.tiles_y):
            for tx in range(self._stack.tiles_x):
                tile = self.full_prefix_tile(layer, tx, ty)
                if tile is None:
                    continue
                x0, y0, x1, y1 = self._stack.tile_bounds(tx, ty)
                result[y0:y1, x0:x1] = tile
        return result

    def composite_tile(self, tx: int, ty: int) -> np.ndarray | None:
        if not self._stack.layers:
            return None
        top_root = self._stack.layers[0]
        return self._composite_of(top_root, tx, ty)

    def full_prefix_tile(self, layer: Layer, tx: int, ty: int) -> np.ndarray | None:
        local = self._prefix_of(layer, tx, ty)
        ext = self._external_context(layer, tx, ty)
        if ext is None:
            return local
        if local is None:
            return ext
        result = ext.astype(np.float32)
        self._blend_buffer(local, 1.0, result)
        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Internal cache helpers
    # ------------------------------------------------------------------

    def _prefix_of(self, layer: Layer, tx: int, ty: int) -> np.ndarray | None:
        key = (layer, tx, ty)
        if key in self._prefix_cache:
            return self._prefix_cache[key]

        siblings = self._stack.comp_order_siblings(layer)
        idx = siblings.index(layer)
        previous = self._composite_of(siblings[idx - 1], tx, ty) if idx > 0 else None

        nested = self._nested_of(layer, tx, ty)

        if previous is None and nested is None:
            self._prefix_cache[key] = None
            return None

        result = self._blank_float(tx, ty)
        if previous is not None:
            result[:, :, :] = previous.astype(np.float32)
        if nested is not None:
            self._blend_buffer(nested, 1.0, result)

        out = np.clip(result, 0, 255).astype(np.uint8)
        self._prefix_cache[key] = out
        return out

    def _nested_of(self, layer: Layer, tx: int, ty: int) -> np.ndarray | None:
        key = (layer, tx, ty)
        if key in self._nested_cache:
            return self._nested_cache[key]
        if layer.children:
            top_child = layer.children[0]
            nested = self._composite_of(top_child, tx, ty)
        else:
            nested = None
        self._nested_cache[key] = nested
        return nested

    def _composite_of(self, layer: Layer, tx: int, ty: int) -> np.ndarray | None:
        key = (layer, tx, ty)
        if key in self._composite_cache:
            return self._composite_cache[key]

        if not layer.visible or layer.opacity <= 0:
            siblings = self._stack.comp_order_siblings(layer)
            idx = siblings.index(layer)
            if idx > 0:
                result = self._composite_of(siblings[idx - 1], tx, ty)
            else:
                result = None
            self._composite_cache[key] = result
            return result

        prefix = self._prefix_of(layer, tx, ty)
        own = layer.content.get_tile(tx, ty)
        if own is not None and own.ndim == 2:
            own = own[:, :, None]
        if layer.opacity >= 1.0:
            if prefix is None and (own is None or not np.any(own[:, :, 3])):
                self._composite_cache[key] = None
                return None
            result = self._blank_float(tx, ty)
            if prefix is not None:
                result[:, :, :] = prefix.astype(np.float32)
            if own is not None:
                self._blend_image(own, 1.0, result)
            out = np.clip(result, 0, 255).astype(np.uint8)
            self._composite_cache[key] = out
            return out

        siblings = self._stack.comp_order_siblings(layer)
        idx = siblings.index(layer)
        previous = self._composite_of(siblings[idx - 1], tx, ty) if idx > 0 else None
        nested = self._nested_of(layer, tx, ty)

        if nested is None and (own is None or not np.any(own[:, :, 3])):
            subtree = None
        else:
            subtree_f = self._blank_float(tx, ty)
            if nested is not None:
                subtree_f[:, :, :] = nested.astype(np.float32)
            if own is not None:
                self._blend_image(own, 1.0, subtree_f)
            subtree = np.clip(subtree_f, 0, 255).astype(np.uint8)

        if previous is None and subtree is None:
            self._composite_cache[key] = None
            return None

        result = self._blank_float(tx, ty)
        if previous is not None:
            result[:, :, :] = previous.astype(np.float32)
        if subtree is not None:
            self._blend_buffer(subtree, layer.opacity, result)
        out = np.clip(result, 0, 255).astype(np.uint8)
        self._composite_cache[key] = out
        return out

    def _external_context(self, layer: Layer, tx: int, ty: int) -> np.ndarray | None:
        if layer.parent is None:
            return None
        parent = layer.parent
        parent_ext = self._external_context(parent, tx, ty)
        siblings = self._stack.comp_order_siblings(parent)
        idx = siblings.index(parent)
        if idx > 0:
            prev_composite = self._composite_of(siblings[idx - 1], tx, ty)
        else:
            prev_composite = None

        if parent_ext is None and prev_composite is None:
            return None

        result = self._blank_float(tx, ty)
        if parent_ext is not None:
            result[:, :, :] = parent_ext.astype(np.float32)
        if prev_composite is not None:
            self._blend_buffer(prev_composite, 1.0, result)
        return np.clip(result, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------
    # Blending helpers
    # ------------------------------------------------------------------

    def _blank_float(self, tx: int, ty: int) -> np.ndarray:
        h, w = self._stack.tile_shape(tx, ty)
        return np.zeros((h, w, 4), dtype=np.float32)

    @staticmethod
    def _blend_image(image: np.ndarray, opacity: float, result: np.ndarray) -> None:
        """Blend straight-alpha image onto float32 result buffer."""
        alpha = image[:, :, 3:4].astype(np.float32) * (opacity / 255.0)
        inv_alpha = 1.0 - alpha
        src_rgb = image[:, :, :3].astype(np.float32)
        result[:, :, :3] = src_rgb * alpha + result[:, :, :3] * inv_alpha
        result[:, :, 3:4] = alpha * 255.0 + result[:, :, 3:4] * inv_alpha

    @staticmethod
    def _blend_buffer(src_buf: np.ndarray, opacity: float, result: np.ndarray) -> None:
        """Blend uint8 composited buffer onto float32 result buffer."""
        alpha = src_buf[:, :, 3:4].astype(np.float32) * (opacity / 255.0)
        inv_alpha = 1.0 - alpha
        if opacity != 1.0:
            src_rgb = src_buf[:, :, :3].astype(np.float32) * opacity
        else:
            src_rgb = src_buf[:, :, :3].astype(np.float32)
        result[:, :, :3] = src_rgb + result[:, :, :3] * inv_alpha
        result[:, :, 3:4] = alpha * 255.0 + result[:, :, 3:4] * inv_alpha
