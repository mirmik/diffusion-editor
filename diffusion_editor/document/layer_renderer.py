"""Tile renderer with an explicitly premultiplied internal representation.

Layer images are stored as straight RGBA8.  Cached renderer tiles are
premultiplied float32 in the 0..255 range so Porter-Duff ``source over`` can
be accumulated without repeatedly quantizing translucent pixels.
Premultiplied buffers must not cross the public document boundary;
:class:`LayerStack` converts them back to straight RGBA8 for display, export
and flattening.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .layer import Layer


def premultiplied_to_straight_rgba(
        premultiplied: np.ndarray) -> np.ndarray:
    """Convert a premultiplied 0..255 buffer to straight RGBA8.

    Transparent RGB is normalized to zero.  Rounding to the nearest integer is
    important here: it makes a public composite safe to feed back into a layer
    (for example by ``flatten``) without repeatedly darkening translucent
    pixels.
    """
    if not np.issubdtype(premultiplied.dtype, np.number):
        raise TypeError("premultiplied RGBA buffer must be numeric")
    if premultiplied.ndim != 3 or premultiplied.shape[2] != 4:
        raise ValueError("premultiplied RGBA buffer must have shape (h, w, 4)")

    straight = np.zeros(premultiplied.shape, dtype=np.uint8)
    alpha = np.clip(
        premultiplied[:, :, 3:4].astype(np.float32), 0.0, 255.0)
    straight_alpha = np.rint(alpha).astype(np.uint8)
    straight[:, :, 3:4] = straight_alpha

    # A sub-half alpha becomes zero at the RGBA8 boundary; normalize its RGB
    # to zero as well instead of emitting hidden colour into a transparent
    # pixel.
    nonzero = straight_alpha[:, :, 0] != 0
    if np.any(nonzero):
        src_rgb = premultiplied[:, :, :3][nonzero].astype(np.float32)
        src_alpha = alpha[:, :, 0][nonzero][:, None]
        rgb = np.rint(src_rgb * (255.0 / src_alpha))
        straight[:, :, :3][nonzero] = np.clip(rgb, 0, 255).astype(np.uint8)
    return straight


class LayerRenderer:
    DEFAULT_CACHE_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024
    DEFAULT_CACHE_ENTRY_LIMIT = 65_536

    def __init__(self, layer_stack: "LayerStack"):
        self._stack = layer_stack
        # Caches per (layer, tx, ty)
        self._prefix_cache: dict[tuple[Layer, int, int], np.ndarray | None] = {}
        self._nested_cache: dict[tuple[Layer, int, int], np.ndarray | None] = {}
        self._composite_cache: dict[tuple[Layer, int, int], np.ndarray | None] = {}
        self._cache_order: OrderedDict[
            tuple[str, tuple[Layer, int, int]], None
        ] = OrderedDict()
        self._cache_array_refs: dict[int, tuple[np.ndarray, int]] = {}
        self._cache_memory_bytes = 0
        self._cache_memory_limit_bytes = (
            self.DEFAULT_CACHE_MEMORY_LIMIT_BYTES
        )
        self._cache_entry_limit = self.DEFAULT_CACHE_ENTRY_LIMIT
        self._cache_revision = 0

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def reset_cache(self) -> None:
        if not any((
                self._prefix_cache,
                self._nested_cache,
                self._composite_cache,
        )):
            return
        self._prefix_cache.clear()
        self._nested_cache.clear()
        self._composite_cache.clear()
        self._cache_order.clear()
        self._cache_array_refs.clear()
        self._cache_memory_bytes = 0
        self._cache_revision += 1

    @property
    def cache_revision(self) -> int:
        return self._cache_revision

    def cache_memory_bytes(self) -> int:
        """Memory held by unique cached tile arrays (aliases count once)."""
        return self._cache_memory_bytes

    @property
    def cache_memory_limit_bytes(self) -> int:
        return self._cache_memory_limit_bytes

    def set_cache_memory_limit_bytes(self, limit_bytes: int) -> None:
        if isinstance(limit_bytes, bool) or not isinstance(
                limit_bytes, (int, np.integer)):
            raise ValueError("render cache memory limit must be an integer")
        if limit_bytes < 0:
            raise ValueError("render cache memory limit cannot be negative")
        self._cache_memory_limit_bytes = int(limit_bytes)
        self._evict_to_limits()

    def invalidate_tiles(self, layers: set[Layer], tiles: set[tuple[int, int]] | None) -> None:
        for layer in layers:
            self._invalidate_layer(layer, tiles)

    def _invalidate_layer(self, layer: Layer, tiles: set[tuple[int, int]] | None) -> None:
        if tiles is None:
            changed = False
            changed |= self._drop_layer_from_cache(
                self._prefix_cache, layer)
            changed |= self._drop_layer_from_cache(
                self._nested_cache, layer)
            changed |= self._drop_layer_from_cache(
                self._composite_cache, layer)
            if changed:
                self._cache_revision += 1
            return
        for tx, ty in tiles:
            key = (layer, tx, ty)
            changed = self._cache_pop(self._prefix_cache, key)
            changed |= self._cache_pop(self._nested_cache, key)
            changed |= self._cache_pop(self._composite_cache, key)
            if changed:
                self._cache_revision += 1

    def _drop_layer_from_cache(self, cache: dict, layer: Layer) -> bool:
        keys = [k for k in cache.keys() if k[0] is layer]
        for k in keys:
            self._cache_pop(cache, k)
        return bool(keys)

    def _cache_pop(
            self,
            cache: dict,
            key: tuple[Layer, int, int]) -> bool:
        if key not in cache:
            return False
        value = cache.pop(key)
        token = (self._cache_name(cache), key)
        self._cache_order.pop(token, None)
        self._release_cache_value(value)
        return True

    def _cache_set(
            self,
            cache: dict,
            key: tuple[Layer, int, int],
            value: np.ndarray | None) -> None:
        if key in cache:
            self._cache_pop(cache, key)
        cache[key] = value
        token = (self._cache_name(cache), key)
        self._cache_order[token] = None
        self._retain_cache_value(value)
        self._cache_revision += 1
        self._evict_to_limits()

    def _cache_get(
            self,
            cache: dict,
            key: tuple[Layer, int, int],
    ) -> tuple[bool, np.ndarray | None]:
        if key not in cache:
            return False, None
        token = (self._cache_name(cache), key)
        self._cache_order.move_to_end(token)
        return True, cache[key]

    def _cache_name(self, cache: dict) -> str:
        if cache is self._prefix_cache:
            return "prefix"
        if cache is self._nested_cache:
            return "nested"
        if cache is self._composite_cache:
            return "composite"
        raise RuntimeError("unknown renderer cache")

    def _cache_by_name(self, name: str) -> dict:
        if name == "prefix":
            return self._prefix_cache
        if name == "nested":
            return self._nested_cache
        if name == "composite":
            return self._composite_cache
        raise RuntimeError("unknown renderer cache")

    def _retain_cache_value(self, value: np.ndarray | None) -> None:
        if value is None:
            return
        identity = id(value)
        retained = self._cache_array_refs.get(identity)
        if retained is None:
            self._cache_array_refs[identity] = (value, 1)
            self._cache_memory_bytes += int(value.nbytes)
            return
        self._cache_array_refs[identity] = (retained[0], retained[1] + 1)

    def _release_cache_value(self, value: np.ndarray | None) -> None:
        if value is None:
            return
        identity = id(value)
        retained = self._cache_array_refs.get(identity)
        if retained is None:
            raise RuntimeError("renderer cache accounting is inconsistent")
        if retained[1] == 1:
            self._cache_array_refs.pop(identity)
            self._cache_memory_bytes -= int(retained[0].nbytes)
            return
        self._cache_array_refs[identity] = (retained[0], retained[1] - 1)

    def _evict_to_limits(self) -> None:
        evicted = False
        while self._cache_order and (
                self._cache_memory_bytes > self._cache_memory_limit_bytes
                or len(self._cache_order) > self._cache_entry_limit):
            (name, key), _ = self._cache_order.popitem(last=False)
            cache = self._cache_by_name(name)
            if key not in cache:
                continue
            value = cache.pop(key)
            self._release_cache_value(value)
            evicted = True
        if evicted:
            self._cache_revision += 1

    # ------------------------------------------------------------------
    # Tile compositing
    # ------------------------------------------------------------------

    def composite_full_premultiplied(self) -> np.ndarray:
        """Return the full canvas as internal premultiplied float32."""
        h, w = self._stack.height, self._stack.width
        if h == 0 or w == 0:
            return np.zeros((1, 1, 4), dtype=np.float32)
        result = np.zeros((h, w, 4), dtype=np.float32)
        for ty in range(self._stack.tiles_y):
            for tx in range(self._stack.tiles_x):
                tile = self.composite_tile(tx, ty)
                if tile is None:
                    continue
                x0, y0, x1, y1 = self._stack.tile_bounds(tx, ty)
                result[y0:y1, x0:x1] = tile
        return result

    def composite_full_straight_rgba(self) -> np.ndarray:
        """Assemble public RGBA8 tile-by-tile without a full float buffer."""
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
                result[y0:y1, x0:x1] = (
                    premultiplied_to_straight_rgba(tile)
                )
        return result

    def composite_rect_premultiplied(
            self,
            x0: int,
            y0: int,
            x1: int,
            y1: int) -> np.ndarray:
        """Return a clipped canvas rect as internal premultiplied float32."""
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self._stack.width, x1)
        y1 = min(self._stack.height, y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 4), dtype=np.float32)

        result = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.float32)
        tile_size = self._stack.tile_size
        tx0 = x0 // tile_size
        ty0 = y0 // tile_size
        tx1 = (x1 - 1) // tile_size
        ty1 = (y1 - 1) // tile_size
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                tile = self.composite_tile(tx, ty)
                if tile is None:
                    continue
                bx0, by0, bx1, by1 = self._stack.tile_bounds(tx, ty)
                ox0 = max(x0, bx0)
                oy0 = max(y0, by0)
                ox1 = min(x1, bx1)
                oy1 = min(y1, by1)
                src_x0 = ox0 - bx0
                src_y0 = oy0 - by0
                src_x1 = src_x0 + (ox1 - ox0)
                src_y1 = src_y0 + (oy1 - oy0)
                dst_x0 = ox0 - x0
                dst_y0 = oy0 - y0
                dst_x1 = dst_x0 + (ox1 - ox0)
                dst_y1 = dst_y0 + (oy1 - oy0)
                result[dst_y0:dst_y1, dst_x0:dst_x1] = (
                    tile[src_y0:src_y1, src_x0:src_x1]
                )
        return result

    def _composite_siblings_direct(self, siblings: list[Layer],
                                   h: int, w: int) -> np.ndarray:
        """Composite sibling list on full images (fast, no tiles)."""
        result = np.zeros((h, w, 4), dtype=np.float32)
        for layer in reversed(siblings):  # bottom to top
            if not self._stack.is_layer_tree_visible_for_composition(layer):
                continue
            own_visible = self._stack.is_layer_visible_for_composition(layer)
            if own_visible and layer.opacity <= 0:
                continue
            if layer.opacity >= 1.0 and not layer.children:
                if own_visible:
                    self._blend_image(layer.image, 1.0, result)
            else:
                subtree = np.zeros((h, w, 4), dtype=np.float32)
                if layer.children:
                    child_comp = self._composite_siblings_direct(
                        layer.children, h, w)
                    subtree[:] = child_comp
                if own_visible:
                    self._blend_image(layer.image, 1.0, subtree)
                    self._blend_buffer(subtree, layer.opacity, result)
                else:
                    self._blend_buffer(subtree, 1.0, result)
        np.clip(result, 0, 255, out=result)
        return result

    def prefix_full_premultiplied(self, layer: Layer) -> np.ndarray:
        """Return a layer prefix as internal premultiplied float32."""
        h, w = self._stack.height, self._stack.width
        result = np.zeros((h, w, 4), dtype=np.float32)
        for ty in range(self._stack.tiles_y):
            for tx in range(self._stack.tiles_x):
                tile = self.full_prefix_tile(layer, tx, ty)
                if tile is None:
                    continue
                x0, y0, x1, y1 = self._stack.tile_bounds(tx, ty)
                result[y0:y1, x0:x1] = tile
        return result

    def prefix_full_straight_rgba(self, layer: Layer) -> np.ndarray:
        """Assemble a public prefix without allocating canvas-sized float32."""
        h, w = self._stack.height, self._stack.width
        result = np.zeros((h, w, 4), dtype=np.uint8)
        for ty in range(self._stack.tiles_y):
            for tx in range(self._stack.tiles_x):
                tile = self.full_prefix_tile(layer, tx, ty)
                if tile is None:
                    continue
                x0, y0, x1, y1 = self._stack.tile_bounds(tx, ty)
                result[y0:y1, x0:x1] = (
                    premultiplied_to_straight_rgba(tile)
                )
        return result

    def composite_tile(self, tx: int, ty: int) -> np.ndarray | None:
        """Return a cached premultiplied float32 tile."""
        if not self._stack.layers:
            return None
        top_root = self._stack.layers[0]
        return self._composite_of(top_root, tx, ty)

    def full_prefix_tile(self, layer: Layer, tx: int, ty: int) -> np.ndarray | None:
        """Return the canonical premultiplied prefix below ``layer``.

        The prefix is a partial tree composition, not merely the target's
        lower siblings plus an external buffer.  Group opacity on every
        ancestor must be applied to the partial subtree before it is blended
        over lower root/sibling content.
        """
        result, found = self._compose_siblings_until(
            self._stack.layers,
            layer,
            tx,
            ty,
        )
        if not found:
            raise ValueError("prefix target does not belong to the layer tree")
        return result

    def _compose_siblings_until(
            self,
            siblings,
            target: Layer,
            tx: int,
            ty: int,
    ) -> tuple[np.ndarray | None, bool]:
        result: np.ndarray | None = None
        for layer in reversed(siblings):
            contains_target = (
                layer is target
                or self._stack._layer_contains(layer, target)
            )
            if contains_target:
                partial: np.ndarray | None = None
                tree_visible = (
                    self._stack.is_layer_tree_visible_for_composition(layer)
                )
                own_visible = (
                    self._stack.is_layer_visible_for_composition(layer)
                )
                if tree_visible and not (
                        own_visible and layer.opacity <= 0):
                    if layer is target:
                        partial = self._compose_siblings_isolated(
                            layer.children, tx, ty)
                    else:
                        partial, found = self._compose_siblings_until(
                            layer.children,
                            target,
                            tx,
                            ty,
                        )
                        if not found:
                            raise ValueError(
                                "layer parent/child references are inconsistent")
                    if (
                            partial is not None
                            and own_visible
                            and layer.opacity < 1.0):
                        partial = partial.copy()
                        partial *= float(layer.opacity)
                result = self._blend_optional(partial, result, tx, ty)
                return result, True

            subtree = self._isolated_subtree(layer, tx, ty)
            result = self._blend_optional(subtree, result, tx, ty)
        return result, False

    def _compose_siblings_isolated(
            self,
            siblings,
            tx: int,
            ty: int,
    ) -> np.ndarray | None:
        result: np.ndarray | None = None
        for layer in reversed(siblings):
            subtree = self._isolated_subtree(layer, tx, ty)
            result = self._blend_optional(subtree, result, tx, ty)
        return result

    def _isolated_subtree(
            self,
            layer: Layer,
            tx: int,
            ty: int,
    ) -> np.ndarray | None:
        if not self._stack.is_layer_tree_visible_for_composition(layer):
            return None
        own_visible = self._stack.is_layer_visible_for_composition(layer)
        if own_visible and layer.opacity <= 0:
            return None

        nested = self._compose_siblings_isolated(layer.children, tx, ty)
        own = self._layer_canvas_tile(layer, tx, ty) if own_visible else None
        subtree = self._blend_optional(own, nested, tx, ty, straight_source=True)
        if (
                subtree is not None
                and own_visible
                and layer.opacity < 1.0):
            subtree = subtree.copy()
            subtree *= float(layer.opacity)
        return subtree

    def _blend_optional(
            self,
            source: np.ndarray | None,
            destination: np.ndarray | None,
            tx: int,
            ty: int,
            *,
            straight_source: bool = False,
    ) -> np.ndarray | None:
        if source is None:
            return destination
        result = self._blank_float(tx, ty)
        if destination is not None:
            result[:] = destination
        if straight_source:
            self._blend_image(source, 1.0, result)
        else:
            self._blend_buffer(source, 1.0, result)
        np.clip(result, 0, 255, out=result)
        return result

    # ------------------------------------------------------------------
    # Internal cache helpers
    # ------------------------------------------------------------------

    def _prefix_of(self, layer: Layer, tx: int, ty: int) -> np.ndarray | None:
        key = (layer, tx, ty)
        cached, value = self._cache_get(self._prefix_cache, key)
        if cached:
            return value

        siblings = self._stack.comp_order_siblings(layer)
        idx = siblings.index(layer)
        previous = self._composite_of(siblings[idx - 1], tx, ty) if idx > 0 else None

        nested = self._nested_of(layer, tx, ty)

        if previous is None and nested is None:
            self._cache_set(self._prefix_cache, key, None)
            return None
        if nested is None:
            # A flat layer prefix is exactly the previous sibling composite.
            # Keep an alias instead of a second full-size float tile.
            self._cache_set(self._prefix_cache, key, previous)
            return previous

        result = self._blank_float(tx, ty)
        if previous is not None:
            result[:, :, :] = previous
        if nested is not None:
            self._blend_buffer(nested, 1.0, result)

        np.clip(result, 0, 255, out=result)
        out = result
        self._cache_set(self._prefix_cache, key, out)
        return out

    def _nested_of(self, layer: Layer, tx: int, ty: int) -> np.ndarray | None:
        key = (layer, tx, ty)
        cached, value = self._cache_get(self._nested_cache, key)
        if cached:
            return value
        if layer.children:
            top_child = layer.children[0]
            nested = self._composite_of(top_child, tx, ty)
        else:
            nested = None
        self._cache_set(self._nested_cache, key, nested)
        return nested

    def _composite_of(self, layer: Layer, tx: int, ty: int) -> np.ndarray | None:
        key = (layer, tx, ty)
        cached, value = self._cache_get(self._composite_cache, key)
        if cached:
            return value

        tree_visible = self._stack.is_layer_tree_visible_for_composition(layer)
        own_visible = self._stack.is_layer_visible_for_composition(layer)
        if not tree_visible:
            siblings = self._stack.comp_order_siblings(layer)
            idx = siblings.index(layer)
            if idx > 0:
                result = self._composite_of(siblings[idx - 1], tx, ty)
            else:
                result = None
            self._cache_set(self._composite_cache, key, result)
            return result

        if own_visible and layer.opacity <= 0:
            siblings = self._stack.comp_order_siblings(layer)
            idx = siblings.index(layer)
            result = self._composite_of(siblings[idx - 1], tx, ty) if idx > 0 else None
            self._cache_set(self._composite_cache, key, result)
            return result

        own = self._layer_canvas_tile(layer, tx, ty)
        if not own_visible:
            # ``prefix`` already consists of previous siblings plus the
            # visible nested subtree.  Blending ``nested`` a second time made
            # a translucent soloed descendant darker and more opaque.
            prefix = self._prefix_of(layer, tx, ty)
            self._cache_set(self._composite_cache, key, prefix)
            return prefix

        if layer.opacity >= 1.0:
            prefix = self._prefix_of(layer, tx, ty)
            if prefix is None and (own is None or not np.any(own[:, :, 3])):
                self._cache_set(self._composite_cache, key, None)
                return None
            result = self._blank_float(tx, ty)
            if prefix is not None:
                result[:, :, :] = prefix
            if own is not None:
                self._blend_image(own, 1.0, result)
            np.clip(result, 0, 255, out=result)
            out = result
            self._cache_set(self._composite_cache, key, out)
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
                subtree_f[:, :, :] = nested
            if own is not None:
                self._blend_image(own, 1.0, subtree_f)
            np.clip(subtree_f, 0, 255, out=subtree_f)
            subtree = subtree_f

        if previous is None and subtree is None:
            self._cache_set(self._composite_cache, key, None)
            return None

        result = self._blank_float(tx, ty)
        if previous is not None:
            result[:, :, :] = previous
        if subtree is not None:
            self._blend_buffer(subtree, layer.opacity, result)
        np.clip(result, 0, 255, out=result)
        out = result
        self._cache_set(self._composite_cache, key, out)
        return out

    def _layer_canvas_tile(self, layer: Layer, tx: int, ty: int) -> np.ndarray | None:
        """Return layer pixels positioned in a canvas tile.

        Layer image coordinates are local to the layer; renderer coordinates
        are canvas-global. The returned tile always has the canvas tile shape,
        with transparent pixels outside the layer bounds.
        """
        bx0, by0, bx1, by1 = self._stack.tile_bounds(tx, ty)
        lx0, ly0, lx1, ly1 = layer.bounds
        ox0 = max(bx0, lx0)
        oy0 = max(by0, ly0)
        ox1 = min(bx1, lx1)
        oy1 = min(by1, ly1)
        if ox1 <= ox0 or oy1 <= oy0:
            return None

        src_x0 = ox0 - layer.x
        src_y0 = oy0 - layer.y
        src_x1 = src_x0 + (ox1 - ox0)
        src_y1 = src_y0 + (oy1 - oy0)
        src = layer.image[src_y0:src_y1, src_x0:src_x1]
        if src.ndim == 2:
            src = src[:, :, None]
        if src.shape[2] < 4 or not np.any(src[:, :, 3]):
            return None

        tile = np.zeros((*self._stack.tile_shape(tx, ty), 4), dtype=np.uint8)
        dst_x0 = ox0 - bx0
        dst_y0 = oy0 - by0
        dst_x1 = dst_x0 + (ox1 - ox0)
        dst_y1 = dst_y0 + (oy1 - oy0)
        tile[dst_y0:dst_y1, dst_x0:dst_x1] = src
        return tile

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
            result[:, :, :] = parent_ext
        if prev_composite is not None:
            self._blend_buffer(prev_composite, 1.0, result)
        np.clip(result, 0, 255, out=result)
        return result

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
        """Blend a premultiplied 0..255 buffer onto a premultiplied buffer."""
        alpha = src_buf[:, :, 3:4].astype(np.float32) * (opacity / 255.0)
        inv_alpha = 1.0 - alpha
        if opacity != 1.0:
            src_rgb = src_buf[:, :, :3].astype(np.float32) * opacity
        else:
            src_rgb = src_buf[:, :, :3].astype(np.float32)
        result[:, :, :3] = src_rgb + result[:, :, :3] * inv_alpha
        result[:, :, 3:4] = alpha * 255.0 + result[:, :, 3:4] * inv_alpha
