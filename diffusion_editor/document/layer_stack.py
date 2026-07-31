import json
import io
import logging
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
from typing import Callable
import zipfile

import numpy as np

from .archive_serialization import load_array_from_zip, save_array_to_zip
from .change_event import (
    DocumentChangeCallback,
    DocumentChangeEvent,
    DocumentChangeKind,
    DocumentChangeSubscription,
)
from .layer import Layer, _layer_from_dict
from .layer_renderer import LayerRenderer, premultiplied_to_straight_rgba
from .mask import Selection

logger = logging.getLogger(__name__)


class LayerStack:
    MAX_PROJECT_CANVAS_DIMENSION = 65_536
    MAX_PROJECT_PIXELS = 268_435_456
    MAX_PROJECT_LAYERS = 10_000
    MAX_PROJECT_ARCHIVE_ENTRIES = 50_000
    MAX_PROJECT_UNCOMPRESSED_BYTES = 8 * 1024 * 1024 * 1024
    MAX_PROJECT_MANIFEST_BYTES = 16 * 1024 * 1024

    def __init__(self, tile_size: int = 256):
        self._layers: list[Layer] = []  # root-level layers
        self._active_layer: Layer | None = None
        self.solo_layer_id: str | None = None
        self._width = 0
        self._height = 0
        self._tile_size = tile_size
        self._change_revision = 0
        self._change_lock = RLock()
        self._change_subscribers: dict[int, DocumentChangeCallback] = {}
        self._next_change_token = 1
        # Temporary source-compatibility bridge for third-party callers.  The
        # editor itself uses independent subscription handles.
        self._legacy_on_changed: Callable[[], None] | None = None
        self._renderer = LayerRenderer(self)
        self.selection = Selection()

    @property
    def revision(self) -> int:
        with self._change_lock:
            return self._change_revision

    @property
    def on_changed(self) -> Callable[[], None] | None:
        """Deprecated compatibility callback; prefer ``subscribe``."""
        return self._legacy_on_changed

    @on_changed.setter
    def on_changed(self, callback: Callable[[], None] | None) -> None:
        self._legacy_on_changed = callback

    def subscribe(
            self,
            callback: DocumentChangeCallback) -> DocumentChangeSubscription:
        if not callable(callback):
            raise TypeError("document change subscriber must be callable")
        with self._change_lock:
            token = self._next_change_token
            self._next_change_token += 1
            self._change_subscribers[token] = callback
        return DocumentChangeSubscription(self, token)

    def _unsubscribe_change(self, token: int) -> None:
        with self._change_lock:
            self._change_subscribers.pop(token, None)

    def publish_change(
            self,
            kind: DocumentChangeKind,
            *,
            layers: tuple[Layer, ...] = (),
            layer_ids: tuple[str, ...] = (),
            dirty_rect: tuple[int, int, int, int] | None = None,
            operation: str | None = None) -> DocumentChangeEvent:
        """Publish one committed mutation and advance its semantic revision."""
        if layers and layer_ids:
            raise ValueError("pass layers or layer_ids, not both")
        ids = layer_ids or tuple(layer.id for layer in layers)
        if kind == DocumentChangeKind.PIXELS and dirty_rect is not None:
            if len(layers) != 1:
                raise ValueError("a pixel dirty rect requires exactly one layer")
            dirty_rect = self._clip_rect(
                dirty_rect, layers[0].width, layers[0].height)
        elif kind == DocumentChangeKind.TRANSFORM and dirty_rect is not None:
            dirty_rect = self._clip_rect(
                dirty_rect, self._width, self._height)
        with self._change_lock:
            self._change_revision += 1
            event = DocumentChangeEvent(
                kind=kind,
                revision=self._change_revision,
                layer_ids=ids,
                dirty_rect=dirty_rect,
            )
            subscribers = tuple(self._change_subscribers.values())
            legacy = self._legacy_on_changed
        for callback in subscribers:
            try:
                callback(event)
            except Exception:
                logger.exception(
                    "LayerStack subscriber failed after successful %s",
                    operation or kind.value,
                )
        if legacy is not None:
            try:
                legacy()
            except Exception:
                logger.exception(
                    "LayerStack legacy observer failed after successful %s",
                    operation or kind.value,
                )
        return event

    @staticmethod
    def _clip_rect(
            rect: tuple[int, int, int, int],
            width: int,
            height: int) -> tuple[int, int, int, int] | None:
        x0, y0, x1, y1 = rect
        x0 = max(0, min(int(x0), width))
        y0 = max(0, min(int(y0), height))
        x1 = max(0, min(int(x1), width))
        y1 = max(0, min(int(y1), height))
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1

    # --- Tree traversal ---

    def _all_layers_flat(self) -> list[Layer]:
        """All layers in depth-first order (index 0 = topmost)."""
        result = []
        for layer in self._layers:
            result.append(layer)
            result.extend(layer.all_descendants())
        return result

    def all_layers(self) -> list[Layer]:
        """All layers in depth-first order (index 0 = topmost)."""
        return self._all_layers_flat()

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

    def cache_memory_bytes(self) -> int:
        return self._renderer.cache_memory_bytes()

    @property
    def cache_memory_limit_bytes(self) -> int:
        return self._renderer.cache_memory_limit_bytes

    def set_cache_memory_limit_bytes(self, limit_bytes: int) -> None:
        self._renderer.set_cache_memory_limit_bytes(limit_bytes)

    @property
    def cache_revision(self) -> int:
        return self._renderer.cache_revision

    @property
    def layers(self):
        """Read-only root-layer view."""
        return tuple(self._layers)

    @property
    def active_layer(self) -> Layer | None:
        return self._active_layer

    @active_layer.setter
    def active_layer(self, layer: Layer | None):
        if layer is not None:
            self._require_member(layer)
        if layer is not self._active_layer:
            self._active_layer = layer
            self.publish_change(
                DocumentChangeKind.ACTIVE,
                layers=(layer,) if layer is not None else (),
            )

    def init_from_image(self, image: np.ndarray):
        if not isinstance(image, np.ndarray) or image.ndim < 2:
            raise ValueError("canvas image must be a uint8 RGBA array")
        h, w = image.shape[:2]
        layer = Layer("Background", w, h, image, tile_size=self._tile_size)
        selection = Selection(height=h, width=w)
        for existing in self._layers:
            self._set_subtree_owner(existing, None)
        self._layers.clear()
        self.solo_layer_id = None
        self._width = w
        self._height = h
        self.selection = selection
        self._set_subtree_owner(layer, self)
        self._layers.append(layer)
        self._active_layer = layer
        self._rebuild_caches()
        self.publish_change(
            DocumentChangeKind.STRUCTURE,
            layers=(layer,),
            operation="initialize document",
        )

    def _insert_near_active(self, layer: Layer):
        """Insert layer as a sibling above the active layer."""
        self._validate_detached_subtree(layer)
        self._apply_tile_size(layer)
        if self._active_layer is not None and self._active_layer.parent is not None:
            parent = self._active_layer.parent
            idx = parent.children.index(self._active_layer)
            layer.parent = parent
            parent._children.insert(idx, layer)
        elif self._active_layer is not None and self._active_layer in self._layers:
            idx = self._layers.index(self._active_layer)
            self._layers.insert(idx, layer)
        else:
            self._layers.insert(0, layer)
        self._set_subtree_owner(layer, self)
        self._active_layer = layer

    def _apply_tile_size(self, layer: Layer):
        layer.content.tile_size = self._tile_size
        for child in layer.children:
            self._apply_tile_size(child)

    def add_layer(self, name: str, image: np.ndarray = None):
        if self._width == 0 or self._height == 0:
            return
        if image is not None:
            h, w = image.shape[:2]
            layer = Layer(name, w, h, image, tile_size=self._tile_size)
        else:
            layer = Layer(name, self._width, self._height,
                          tile_size=self._tile_size)
        self._insert_near_active(layer)
        self._rebuild_caches()
        self.publish_change(DocumentChangeKind.STRUCTURE, layers=(layer,))

    def insert_image_layer(self, name: str, image: np.ndarray,
                           x: int = 0, y: int = 0):
        if self._width == 0 or self._height == 0:
            return
        h, w = image.shape[:2]
        layer = Layer(name, w, h, image, tile_size=self._tile_size, x=x, y=y)
        self._insert_near_active(layer)
        self._rebuild_caches()
        self.publish_change(DocumentChangeKind.STRUCTURE, layers=(layer,))

    def insert_layer(self, layer: Layer):
        if self._width == 0 or self._height == 0:
            return
        self._insert_near_active(layer)
        self._rebuild_caches()
        self.publish_change(DocumentChangeKind.STRUCTURE, layers=(layer,))

    def remove_layer(self, layer: Layer):
        """Remove layer and its entire subtree."""
        all_layers = self._all_layers_flat()
        self._require_member(layer)
        removed_count = 1 + len(layer.all_descendants())
        if len(all_layers) <= removed_count:
            raise ValueError("cannot remove the last root layer subtree")
        removed_ids = {layer.id}
        removed_ids.update(child.id for child in layer.all_descendants())
        if layer.parent is not None:
            parent = layer.parent
            idx = parent.children.index(layer)
            parent._children.remove(layer)
            layer.parent = None
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
        self._set_subtree_owner(layer, None)
        if self.solo_layer_id in removed_ids:
            self.solo_layer_id = None
        self._rebuild_caches()
        self.publish_change(
            DocumentChangeKind.STRUCTURE,
            layer_ids=tuple(sorted(removed_ids)),
        )

    def move_layer(self, layer: Layer, new_parent: Layer | None, index: int):
        """Move layer to new_parent at index (or root if new_parent is None)."""
        self._require_member(layer)
        if new_parent is not None:
            self._require_member(new_parent)
            if new_parent is layer or new_parent in layer.all_descendants():
                raise ValueError("cannot move a layer into itself or a descendant")
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError("layer index must be an integer")
        source = (
            layer.parent._children
            if layer.parent is not None else self._layers
        )
        destination = (
            new_parent._children
            if new_parent is not None else self._layers
        )
        if index < 0 or index > len(destination):
            raise IndexError("layer index is out of range")

        if layer.parent is not None:
            layer.parent._children.remove(layer)
            layer.parent = None
        else:
            self._layers.remove(layer)
        index = min(index, len(destination))
        if new_parent is not None:
            layer.parent = new_parent
            new_parent._children.insert(index, layer)
        else:
            self._layers.insert(index, layer)
        self._active_layer = layer
        self._rebuild_caches()
        self.publish_change(DocumentChangeKind.STRUCTURE, layers=(layer,))

    def set_visibility(self, layer: Layer, visible: bool):
        self._require_member(layer)
        if layer.visible == visible:
            return
        layer.visible = visible
        self.mark_layer_dirty(layer, pixels_changed=False)
        self.publish_change(DocumentChangeKind.VISIBILITY, layers=(layer,))

    def set_opacity(self, layer: Layer, opacity: float):
        """Set layer opacity with prefix invalidation."""
        self._require_member(layer)
        if layer.opacity == opacity:
            return
        layer.opacity = opacity
        self.mark_layer_dirty(layer, pixels_changed=False)
        self.publish_change(DocumentChangeKind.OPACITY, layers=(layer,))

    def set_layer_name(self, layer: Layer, name: str):
        self._require_member(layer)
        if layer.name == name:
            return
        layer.name = name
        self.publish_change(DocumentChangeKind.METADATA, layers=(layer,))

    def find_layer_by_id(self, layer_id: str) -> Layer | None:
        if not layer_id:
            return None
        for layer in self._all_layers_flat():
            if layer.id == layer_id:
                return layer
        return None

    def solo_layer(self) -> Layer | None:
        if self.solo_layer_id is None:
            return None
        layer = self.find_layer_by_id(self.solo_layer_id)
        if layer is None:
            self.solo_layer_id = None
        return layer

    def set_solo_layer(self, layer: Layer | None) -> None:
        if layer is not None:
            self._require_member(layer)
        next_id = layer.id if layer is not None else None
        if self.solo_layer_id == next_id:
            return
        self.solo_layer_id = next_id
        self._rebuild_caches()
        self.publish_change(
            DocumentChangeKind.VISIBILITY,
            layers=(layer,) if layer is not None else (),
        )

    def toggle_solo_layer(self, layer: Layer) -> None:
        if self.solo_layer_id == layer.id:
            self.set_solo_layer(None)
        else:
            self.set_solo_layer(layer)

    def is_layer_visible_for_composition(self, layer: Layer) -> bool:
        solo = self.solo_layer()
        if solo is not None:
            return layer is solo
        return layer.visible

    def is_layer_tree_visible_for_composition(self, layer: Layer) -> bool:
        solo = self.solo_layer()
        if solo is None:
            return layer.visible
        if layer is solo:
            return True
        return self._layer_contains(layer, solo)

    def _layer_contains(self, root: Layer, target: Layer) -> bool:
        for child in root.children:
            if child is target or self._layer_contains(child, target):
                return True
        return False

    def set_layer_offset(self, layer: Layer, x: int, y: int,
                         old_bounds: tuple[int, int, int, int] | None = None,
                         *,
                         notify: bool = True):
        self._require_member(layer)
        if old_bounds is None:
            old_bounds = layer.bounds
        if (layer.x, layer.y) == (int(x), int(y)):
            return
        layer.x = int(x)
        layer.y = int(y)
        new_bounds = layer.bounds
        dirty = self._union_rect(old_bounds, new_bounds)
        self.mark_layer_dirty(layer, dirty, pixels_changed=False)
        if notify:
            self.publish_change(
                DocumentChangeKind.TRANSFORM,
                layers=(layer,),
                dirty_rect=dirty,
            )

    def flatten(self):
        if not self._layers:
            raise ValueError("cannot flatten an empty document")
        result = self.composite()
        for existing in self._layers:
            self._set_subtree_owner(existing, None)
        self._layers.clear()
        layer = Layer("Background", self._width, self._height, result, tile_size=self._tile_size)
        self._set_subtree_owner(layer, self)
        self._layers.append(layer)
        self._active_layer = layer
        self.solo_layer_id = None
        self._rebuild_caches()
        self.publish_change(DocumentChangeKind.STRUCTURE, layers=(layer,))

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

    def mark_layer_dirty(
            self,
            layer: Layer,
            rect: tuple[int, int, int, int] | None = None,
            *,
            pixels_changed: bool = True):
        """Public: call when a layer's content/visibility/opacity changed."""
        if pixels_changed:
            layer.mark_pixels_changed()
        affected = self._collect_affected_layers(layer)
        if not affected:
            self._rebuild_caches()
            return
        tiles = self._tiles_for_rect(rect)
        self._renderer.invalidate_tiles(affected, tiles)

    @staticmethod
    def _union_rect(a: tuple[int, int, int, int] | None,
                    b: tuple[int, int, int, int] | None
                    ) -> tuple[int, int, int, int] | None:
        if a is None:
            return b
        if b is None:
            return a
        return (min(a[0], b[0]), min(a[1], b[1]),
                max(a[2], b[2]), max(a[3], b[3]))

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
        """Composite visible layers and return straight RGBA8.

        If exclude_layer is set, returns prefix of that layer (everything below it).
        Premultiplied renderer caches never escape this public boundary.
        """
        if not self._layers or self._width == 0:
            return np.zeros((1, 1, 4), dtype=np.uint8)

        if exclude_layer is not None:
            return self.get_prefix_below(exclude_layer).copy()

        return self._renderer.composite_full_straight_rgba()

    def composite_rect(
            self,
            x0: int,
            y0: int,
            x1: int,
            y1: int) -> np.ndarray:
        """Composite a clipped canvas rect and return straight RGBA8."""
        premultiplied = self._renderer.composite_rect_premultiplied(
            x0, y0, x1, y1)
        return premultiplied_to_straight_rgba(premultiplied)

    def get_prefix_below(self, layer: Layer) -> np.ndarray:
        self._require_member(layer)
        return self._renderer.prefix_full_straight_rgba(layer)

    def get_prefix_below_rect(self, layer: Layer, x0: int, y0: int,
                              x1: int, y1: int) -> np.ndarray:
        """Return prefix buffer for a rect (uint8 RGBA)."""
        self._require_member(layer)
        if self._width == 0 or self._height == 0:
            return np.zeros((1, 1, 4), dtype=np.uint8)
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(self._width, x1)
        y1 = min(self._height, y1)
        if x1 <= x0 or y1 <= y0:
            return np.zeros((0, 0, 4), dtype=np.uint8)

        out = np.zeros((y1 - y0, x1 - x0, 4), dtype=np.float32)
        tiles = self._tiles_for_rect((x0, y0, x1, y1))
        if not tiles:
            return premultiplied_to_straight_rgba(out)
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
        return premultiplied_to_straight_rgba(out)

    # --- Serialization ---

    FORMAT_VERSION = 8

    def _serialize_manifest_and_layers(self, zf: zipfile.ZipFile):
        self._validate_serializable_state()
        manifest = {
            "format_version": self.FORMAT_VERSION,
            "canvas_width": self._width,
            "canvas_height": self._height,
            "tile_size": self._tile_size,
            "active_layer_path": self._find_layer_path(self._active_layer),
            "solo_layer_id": self.solo_layer_id,
            "selection_file": "selection.npy" if not self.selection.is_empty else None,
            "layers": [],
        }
        for i, layer in enumerate(self._layers):
            layer_path = str(i)
            manifest["layers"].append(layer.to_dict(layer_path))
            layer.save_images_to_zip(zf, layer_path)
        if not self.selection.is_empty:
            save_array_to_zip(zf, "selection.npy", self.selection.data)
        zf.writestr("manifest.json",
                    json.dumps(manifest, indent=2, ensure_ascii=False))

    def _load_from_zip(self, zf: zipfile.ZipFile):
        """Load and validate into a detached aggregate, then commit once."""
        manifest = self._read_validated_manifest(zf)
        candidate = LayerStack(tile_size=self._tile_size)
        candidate._load_from_zip_in_place(zf, manifest)
        candidate._validate_loaded_state()

        for existing in self._layers:
            self._set_subtree_owner(existing, None)
        self._tile_size = candidate._tile_size
        self._layers = candidate._layers
        for layer in self._layers:
            self._set_subtree_owner(layer, self)
        self._active_layer = candidate._active_layer
        self.solo_layer_id = candidate.solo_layer_id
        self._width = candidate._width
        self._height = candidate._height
        self.selection = candidate.selection
        self._rebuild_caches()
        self.publish_change(
            DocumentChangeKind.SNAPSHOT_RESTORE,
            layers=tuple(self._all_layers_flat()),
            operation="load project",
        )

    def _load_from_zip_in_place(
            self,
            zf: zipfile.ZipFile,
            manifest: dict) -> None:
        version = manifest.get("format_version", 0)

        self._tile_size = manifest["tile_size"]
        new_layers = []
        for layer_dict in manifest["layers"]:
            layer = _layer_from_dict(layer_dict, zf, tile_size=self._tile_size)
            new_layers.append(layer)

        self._layers.clear()
        self._layers.extend(new_layers)
        for layer in self._layers:
            self._set_subtree_owner(layer, self)
        if version < self.FORMAT_VERSION:
            self._ensure_unique_layer_ids()
        if version < 8:
            self._migrate_canvas_patch_rects_to_layer_local()
        for layer in self._layers:
            self._apply_tile_size(layer)
        self._width = manifest["canvas_width"]
        self._height = manifest["canvas_height"]

        # Restore selection (v6+) or initialize empty
        selection_file = manifest.get("selection_file")
        if selection_file and selection_file in zf.namelist():
            sel_arr = load_array_from_zip(
                zf,
                selection_file,
                mode="L",
                expected_shape=(self._height, self._width),
            )
            if sel_arr.dtype == np.uint8:
                sel_arr = sel_arr.astype(np.float32) / 255.0
            self.selection = Selection(sel_arr)
        else:
            self.selection = Selection(height=self._height, width=self._width)

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
        self.solo_layer_id = manifest.get("solo_layer_id")
        self.solo_layer()
        self._rebuild_caches()

    def _read_validated_manifest(self, zf: zipfile.ZipFile) -> dict:
        infos = zf.infolist()
        if len(infos) > self.MAX_PROJECT_ARCHIVE_ENTRIES:
            raise ValueError("project archive contains too many entries")
        names = [info.filename for info in infos]
        if len(names) != len(set(names)):
            raise ValueError("project archive contains duplicate entry names")
        total_size = sum(max(0, info.file_size) for info in infos)
        if total_size > self.MAX_PROJECT_UNCOMPRESSED_BYTES:
            raise ValueError("project archive exceeds the uncompressed size limit")
        try:
            manifest_info = zf.getinfo("manifest.json")
        except KeyError as exc:
            raise ValueError("project archive is missing manifest.json") from exc
        if manifest_info.file_size > self.MAX_PROJECT_MANIFEST_BYTES:
            raise ValueError("project manifest is too large")
        try:
            manifest = json.loads(zf.read(manifest_info))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("project manifest is invalid JSON") from exc
        if not isinstance(manifest, dict):
            raise ValueError("project manifest must be an object")

        version = self._required_int(
            manifest.get("format_version", 0),
            "format_version",
            minimum=0,
        )
        if version > self.FORMAT_VERSION:
            raise ValueError(
                f"Project version {version} is newer than "
                f"supported version {self.FORMAT_VERSION}")
        width = self._required_int(
            manifest.get("canvas_width"),
            "canvas_width",
            minimum=1,
            maximum=self.MAX_PROJECT_CANVAS_DIMENSION,
        )
        height = self._required_int(
            manifest.get("canvas_height"),
            "canvas_height",
            minimum=1,
            maximum=self.MAX_PROJECT_CANVAS_DIMENSION,
        )
        if width * height > self.MAX_PROJECT_PIXELS:
            raise ValueError("project canvas exceeds the pixel budget")
        tile_size = self._required_int(
            manifest.get("tile_size", self._tile_size),
            "tile_size",
            minimum=1,
            maximum=8192,
        )
        layers = manifest.get("layers")
        if not isinstance(layers, list) or not layers:
            raise ValueError("project manifest must contain at least one layer")
        available = set(names)
        layer_count = 0
        current_ids: set[str] = set()
        stack = list(layers)
        while stack:
            layer_dict = stack.pop()
            layer_count += 1
            if layer_count > self.MAX_PROJECT_LAYERS:
                raise ValueError("project contains too many layers")
            if not isinstance(layer_dict, dict):
                raise ValueError("project layer entry must be an object")
            if version >= self.FORMAT_VERSION:
                layer_id = layer_dict.get("id")
                if not isinstance(layer_id, str) or not layer_id:
                    raise ValueError(
                        "current project layer ID must be a non-empty string")
                if layer_id in current_ids:
                    raise ValueError("current project layer IDs must be unique")
                current_ids.add(layer_id)
            image_file = layer_dict.get("image_file")
            if not isinstance(image_file, str) or image_file not in available:
                raise ValueError("project layer image entry is missing")
            mask_file = layer_dict.get("mask_file")
            if mask_file is not None and (
                    not isinstance(mask_file, str)
                    or mask_file not in available):
                raise ValueError("project layer mask entry is missing")
            tool_dict = layer_dict.get("tool")
            if tool_dict is not None:
                if not isinstance(tool_dict, dict):
                    raise ValueError("project layer tool must be an object")
                tool_type = tool_dict.get("tool_type") or tool_dict.get("type")
                if (
                        version >= self.FORMAT_VERSION
                        and tool_type not in {
                            "diffusion", "lama", "instruct"}):
                    raise ValueError(
                        "current project layer tool type is unsupported")
                source_file = tool_dict.get("source_file")
                if source_file is not None and (
                        not isinstance(source_file, str)
                        or source_file not in available):
                    raise ValueError(
                        "project tool source image entry is missing")
            children = layer_dict.get("children", [])
            if not isinstance(children, list):
                raise ValueError("project layer children must be an array")
            stack.extend(children)

        selection_file = manifest.get("selection_file")
        if selection_file is not None and (
                not isinstance(selection_file, str)
                or selection_file not in available):
            raise ValueError("project selection entry is missing")
        if version >= self.FORMAT_VERSION:
            active_path = manifest.get("active_layer_path")
            if (
                    not isinstance(active_path, str)
                    or not re.fullmatch(r"\d+(?:/\d+)*", active_path)
                    or not self._manifest_path_exists(layers, active_path)):
                raise ValueError(
                    "current project active_layer_path is invalid")
            solo_layer_id = manifest.get("solo_layer_id")
            if (
                    solo_layer_id is not None
                    and (
                        not isinstance(solo_layer_id, str)
                        or not solo_layer_id
                        or solo_layer_id not in current_ids)):
                raise ValueError(
                    "current project solo_layer_id is invalid")
        manifest["format_version"] = version
        manifest["canvas_width"] = width
        manifest["canvas_height"] = height
        manifest["tile_size"] = tile_size
        return manifest

    @staticmethod
    def _manifest_path_exists(layers: list, path: str) -> bool:
        current = layers
        for raw_index in path.split("/"):
            index = int(raw_index)
            if index < 0 or index >= len(current):
                return False
            entry = current[index]
            if not isinstance(entry, dict):
                return False
            children = entry.get("children", [])
            if not isinstance(children, list):
                return False
            current = children
        return True

    def _validate_loaded_state(self) -> None:
        self._validate_serializable_state()

    def _validate_serializable_state(self) -> None:
        """Reject state that this version could not safely load again."""
        self.validate_invariants()
        if not self._layers:
            raise ValueError("project must contain at least one root layer")
        self._required_int(
            self._width,
            "canvas_width",
            minimum=1,
            maximum=self.MAX_PROJECT_CANVAS_DIMENSION,
        )
        self._required_int(
            self._height,
            "canvas_height",
            minimum=1,
            maximum=self.MAX_PROJECT_CANVAS_DIMENSION,
        )
        if self._width * self._height > self.MAX_PROJECT_PIXELS:
            raise ValueError("project canvas exceeds the pixel budget")
        self._required_int(
            self._tile_size,
            "tile_size",
            minimum=1,
            maximum=8192,
        )
        if self.selection.data.shape != (self._height, self._width):
            raise ValueError(
                "project selection dimensions do not match the canvas"
            )
        if self.selection.data.dtype != np.float32:
            raise ValueError("project selection must use float32 values")
        if not np.isfinite(self.selection.data).all():
            raise ValueError("project selection contains non-finite values")
        if (
                self.selection.data.size
                and (
                    self.selection.data.min() < 0.0
                    or self.selection.data.max() > 1.0)):
            raise ValueError("project selection values must be between 0 and 1")
        seen_ids: set[str] = set()
        layer_count = 0
        payload_bytes = int(self.selection.data.nbytes)
        for layer in self._all_layers_flat():
            layer_count += 1
            if not layer.id or layer.id in seen_ids:
                raise ValueError("project contains duplicate layer IDs")
            seen_ids.add(layer.id)
            if (
                    layer.image.dtype != np.uint8
                    or layer.image.ndim != 3
                    or layer.image.shape
                    != (layer.height, layer.width, 4)):
                raise ValueError("project layer image must be uint8 RGBA")
            if layer.width < 1 or layer.height < 1:
                raise ValueError("project layer dimensions must be positive")
            if layer.width * layer.height > self.MAX_PROJECT_PIXELS:
                raise ValueError("project layer exceeds the pixel budget")
            if layer.mask.data.shape != (layer.height, layer.width):
                raise ValueError(
                    "project layer mask dimensions do not match its image"
                )
            if layer.mask.data.dtype != np.float32:
                raise ValueError("project layer mask must use float32 values")
            if not np.isfinite(layer.mask.data).all():
                raise ValueError("project layer mask contains non-finite values")
            if (
                    layer.mask.data.size
                    and (
                        layer.mask.data.min() < 0.0
                        or layer.mask.data.max() > 1.0)):
                raise ValueError(
                    "project layer mask values must be between 0 and 1")
            if not isinstance(layer.visible, bool):
                raise ValueError("project layer visibility must be boolean")
            if (
                    isinstance(layer.opacity, bool)
                    or not isinstance(layer.opacity, (int, float))
                    or not np.isfinite(layer.opacity)
                    or not 0.0 <= layer.opacity <= 1.0):
                raise ValueError("project layer opacity must be between 0 and 1")
            if not isinstance(layer.name, str):
                raise ValueError("project layer name must be a string")
            self._validate_rect(layer.patch_rect, "layer patch_rect")
            payload_bytes += self._validate_tool(layer.tool)
            payload_bytes += int(layer.image.nbytes)
            payload_bytes += int(layer.mask.data.nbytes)
        if layer_count > self.MAX_PROJECT_LAYERS:
            raise ValueError("project contains too many layers")
        if payload_bytes > self.MAX_PROJECT_UNCOMPRESSED_BYTES:
            raise ValueError("project payload exceeds the uncompressed size limit")
        if self._active_layer is None:
            raise ValueError("project has no active layer")

    def _validate_tool(self, tool) -> int:
        if tool is None:
            return 0
        from .tool import DiffusionTool, InstructTool, LamaTool
        from .tool_serialization import serialize_tool

        if not isinstance(tool, (DiffusionTool, InstructTool, LamaTool)):
            raise ValueError("project contains an unsupported layer tool")
        patch_x = self._required_int(
            tool.patch_x,
            "tool patch_x",
            minimum=-self.MAX_PROJECT_CANVAS_DIMENSION,
            maximum=self.MAX_PROJECT_CANVAS_DIMENSION,
        )
        patch_y = self._required_int(
            tool.patch_y,
            "tool patch_y",
            minimum=-self.MAX_PROJECT_CANVAS_DIMENSION,
            maximum=self.MAX_PROJECT_CANVAS_DIMENSION,
        )
        patch_w = self._required_int(
            tool.patch_w,
            "tool patch_w",
            minimum=1,
            maximum=self.MAX_PROJECT_CANVAS_DIMENSION,
        )
        patch_h = self._required_int(
            tool.patch_h,
            "tool patch_h",
            minimum=1,
            maximum=self.MAX_PROJECT_CANVAS_DIMENSION,
        )
        if patch_w * patch_h > self.MAX_PROJECT_PIXELS:
            raise ValueError("tool patch exceeds the pixel budget")
        # Normalize validation only; the aggregate is not mutated.
        del patch_x, patch_y

        if isinstance(tool, DiffusionTool):
            self._validate_string_fields(
                tool,
                (
                    "prompt",
                    "negative_prompt",
                    "model_path",
                    "prediction_type",
                    "mode",
                    "masked_content",
                    "ip_adapter_layer_name_hint",
                ),
            )
            if (
                    tool.ip_adapter_layer_id is not None
                    and not isinstance(tool.ip_adapter_layer_id, str)):
                raise ValueError(
                    "tool ip_adapter_layer_id must be a string or null")
            self._validate_finite_fields(
                tool,
                ("strength", "guidance_scale", "ip_adapter_scale"),
            )
            self._required_int(
                tool.steps, "tool steps", minimum=1, maximum=1_000_000)
            self._required_int(
                tool.seed, "tool seed", minimum=-1, maximum=2**32 - 1)
            if not isinstance(tool.resize_to_model_resolution, bool):
                raise ValueError(
                    "tool resize_to_model_resolution must be boolean")
        elif isinstance(tool, InstructTool):
            self._validate_string_fields(tool, ("instruction",))
            self._validate_finite_fields(
                tool,
                ("image_guidance_scale", "guidance_scale"),
            )
            self._required_int(
                tool.steps, "tool steps", minimum=1, maximum=1_000_000)
            self._required_int(
                tool.seed, "tool seed", minimum=-1, maximum=2**32 - 1)

        payload_bytes = 0
        if tool.source_patch is not None:
            source = np.asarray(tool.source_patch)
            if (
                    source.dtype != np.uint8
                    or source.ndim not in (2, 3)
                    or (
                        source.ndim == 3
                        and source.shape[2] not in (1, 3, 4))
                    or source.shape[0] < 1
                    or source.shape[1] < 1):
                raise ValueError(
                    "tool source_patch must be a non-empty uint8 image "
                    "with 1, 3, or 4 channels")
            if source.shape[0] * source.shape[1] > self.MAX_PROJECT_PIXELS:
                raise ValueError("tool source_patch exceeds the pixel budget")
            payload_bytes = int(source.nbytes)
        try:
            json.dumps(
                serialize_tool(tool, "validation"),
                allow_nan=False,
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "project layer tool is not serializable") from exc
        return payload_bytes

    @staticmethod
    def _validate_string_fields(value, names: tuple[str, ...]) -> None:
        for name in names:
            if not isinstance(getattr(value, name), str):
                raise ValueError(f"tool {name} must be a string")

    @staticmethod
    def _validate_finite_fields(value, names: tuple[str, ...]) -> None:
        for name in names:
            number = getattr(value, name)
            if (
                    isinstance(number, bool)
                    or not isinstance(number, (int, float, np.number))
                    or not np.isfinite(number)):
                raise ValueError(f"tool {name} must be finite")

    @staticmethod
    def _validate_rect(rect, name: str) -> None:
        if rect is None:
            return
        if not isinstance(rect, (tuple, list)) or len(rect) != 4:
            raise ValueError(f"{name} must contain four integers")
        values = []
        for value in rect:
            if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, np.integer))):
                raise ValueError(f"{name} must contain four integers")
            values.append(int(value))
        x0, y0, x1, y1 = values
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"{name} must have positive width and height")

    def validate_invariants(self) -> bool:
        """Validate tree ownership, IDs and active/solo references."""
        if self._width < 0 or self._height < 0:
            raise ValueError("canvas dimensions cannot be negative")
        if (self._width > 0 or self._height > 0) and not self._layers:
            raise ValueError("a non-empty canvas must have at least one root layer")
        if not self._layers:
            if self._active_layer is not None:
                raise ValueError("an empty document cannot have an active layer")
            if self.solo_layer_id is not None:
                raise ValueError("an empty document cannot have a solo layer")
            return True

        seen_objects: set[Layer] = set()
        seen_ids: set[str] = set()
        pending = [
            (layer, None, frozenset())
            for layer in reversed(self._layers)
        ]
        while pending:
            layer, expected_parent, ancestors = pending.pop()
            if layer in ancestors:
                raise ValueError("layer tree contains a cycle")
            if layer in seen_objects:
                raise ValueError("layer appears more than once in the tree")
            if layer.parent is not expected_parent:
                raise ValueError("layer parent reference is inconsistent")
            if layer._owner is not self:
                raise ValueError("layer belongs to a different document")
            if not isinstance(layer.id, str) or not layer.id:
                raise ValueError("layer ID must be a non-empty string")
            if layer.id in seen_ids:
                raise ValueError("layer IDs must be unique")
            seen_objects.add(layer)
            seen_ids.add(layer.id)
            next_ancestors = ancestors | {layer}
            for child in reversed(layer.children):
                if not isinstance(child, Layer):
                    raise ValueError("layer tree contains a non-Layer child")
                pending.append((child, layer, next_ancestors))

        if self._active_layer not in seen_objects:
            raise ValueError("active layer does not belong to the document")
        if (
                self.solo_layer_id is not None
                and self.solo_layer_id not in seen_ids):
            raise ValueError("solo layer does not belong to the document")
        return True

    def _require_member(self, layer: Layer) -> None:
        if (
                not isinstance(layer, Layer)
                or self.find_layer_by_id(layer.id) is not layer):
            raise ValueError("layer does not belong to this document")

    def _validate_detached_subtree(self, root: Layer) -> None:
        if not isinstance(root, Layer):
            raise TypeError("layer must be a Layer")
        if root.parent is not None:
            raise ValueError("inserted layer root must be detached")
        existing_ids = {layer.id for layer in self._all_layers_flat()}
        seen_objects: set[Layer] = set()
        seen_ids: set[str] = set()
        pending = [(root, None, frozenset())]
        while pending:
            layer, expected_parent, ancestors = pending.pop()
            if layer in ancestors:
                raise ValueError("inserted layer subtree contains a cycle")
            if layer in seen_objects:
                raise ValueError("inserted layer appears more than once")
            if layer.parent is not expected_parent:
                raise ValueError("inserted layer parent reference is inconsistent")
            if layer._owner is not None:
                raise ValueError("inserted layer subtree already belongs to a document")
            if not isinstance(layer.id, str) or not layer.id:
                raise ValueError("inserted layer ID must be a non-empty string")
            if layer.id in existing_ids or layer.id in seen_ids:
                raise ValueError("inserted layer ID already exists")
            seen_objects.add(layer)
            seen_ids.add(layer.id)
            next_ancestors = ancestors | {layer}
            for child in reversed(layer.children):
                pending.append((child, layer, next_ancestors))

    @staticmethod
    def _set_subtree_owner(root: Layer, owner: "LayerStack | None") -> None:
        pending = [root]
        while pending:
            layer = pending.pop()
            layer._owner = owner
            pending.extend(layer.children)

    @staticmethod
    def _required_int(
            value,
            name: str,
            *,
            minimum: int,
            maximum: int | None = None) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"project {name} must be an integer")
        if value < minimum:
            raise ValueError(f"project {name} must be >= {minimum}")
        if maximum is not None and value > maximum:
            raise ValueError(f"project {name} must be <= {maximum}")
        return value

    def serialize_state(self) -> bytes:
        """Fast snapshot for undo: ZIP_STORED, no compression."""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as zf:
            self._serialize_manifest_and_layers(zf)
        return buf.getvalue()

    def load_state(self, state: bytes):
        with zipfile.ZipFile(io.BytesIO(state), "r") as zf:
            self._load_from_zip(zf)

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

    def get_layer_path(self, target: Layer | None) -> str | None:
        return self._find_layer_path(target)

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

    def get_layer_by_path(self, path: str) -> Layer | None:
        return self._find_layer_by_path(path)

    def _ensure_unique_layer_ids(self) -> None:
        from .layer import new_layer_id

        seen: set[str] = set()
        for layer in self._all_layers_flat():
            if not layer.id or layer.id in seen:
                layer.id = new_layer_id()
            seen.add(layer.id)

    def _migrate_canvas_patch_rects_to_layer_local(self) -> None:
        for layer in self._all_layers_flat():
            if layer.patch_rect is not None:
                layer.patch_rect = layer.canvas_rect_to_local(layer.patch_rect)

    def save_project(self, path: str):
        destination = Path(path)
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=str(destination.parent),
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(file_descriptor, "w+b") as output:
                file_descriptor = -1
                with zipfile.ZipFile(
                        output, "w", zipfile.ZIP_DEFLATED) as zf:
                    self._serialize_manifest_and_layers(zf)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, destination)
            try:
                self._fsync_directory(destination.parent)
            except OSError:
                # The destination has already been atomically replaced. Some
                # platforms/filesystems cannot fsync directories; reporting a
                # generic save failure here would falsely imply no commit.
                logger.warning(
                    "Project was saved, but its directory could not be fsynced",
                    exc_info=True,
                )
        except BaseException:
            if file_descriptor >= 0:
                os.close(file_descriptor)
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass
            raise

    def load_project(self, path: str):
        with zipfile.ZipFile(path, "r") as zf:
            self._load_from_zip(zf)

    @staticmethod
    def _fsync_directory(directory: Path) -> None:
        flags = os.O_RDONLY
        if hasattr(os, "O_DIRECTORY"):
            flags |= os.O_DIRECTORY
        directory_fd = os.open(directory, flags)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
