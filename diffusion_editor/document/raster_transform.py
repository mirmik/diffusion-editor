"""Raster transform sessions: immutable sources, linear premultiplied resampling."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

from ..color import srgb_to_linear
from .change_event import DocumentChangeKind
from .commands import CommandDelta
from .layer import Layer
from .layer_renderer import LayerRenderer, premultiplied_to_straight_rgba
from .mask import Mask, Selection
from .tiles import DenseTileGrid

Rect = tuple[int, int, int, int]
MAX_TRANSFORM_PIXELS = 32 * 1024 * 1024
MAX_TRANSFORM_DIMENSION = 16384


def validate_size(width: int, height: int) -> None:
    if (width < 1 or height < 1 or max(width, height) > MAX_TRANSFORM_DIMENSION
            or width * height > MAX_TRANSFORM_PIXELS):
        raise ValueError("Transform is too large (maximum 16384 px / 32 MP)")


def _resize(data: np.ndarray, width: int, height: int) -> np.ndarray:
    if data.shape[:2] == (height, width):
        return data.copy()
    if data.ndim == 2:
        return np.asarray(Image.fromarray(data).resize(
            (width, height), Image.Resampling.LANCZOS), dtype=np.float32)
    return np.stack([_resize(data[:, :, i], width, height)
                     for i in range(data.shape[2])], axis=2)


def _linear(image: np.ndarray) -> np.ndarray:
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    return np.concatenate((srgb_to_linear(image[:, :, :3].astype(np.float32) / 255.0) * alpha,
                           alpha), axis=2).astype(np.float32)


def _region_views(destination, destination_rect, source, source_rect):
    """Views of the overlapping samples in two canvas-positioned arrays."""
    x0 = max(destination_rect[0], source_rect[0])
    y0 = max(destination_rect[1], source_rect[1])
    x1 = min(destination_rect[2], source_rect[2])
    y1 = min(destination_rect[3], source_rect[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (destination[y0-destination_rect[1]:y1-destination_rect[1],
                        x0-destination_rect[0]:x1-destination_rect[0]],
            source[y0-source_rect[1]:y1-source_rect[1],
                   x0-source_rect[0]:x1-source_rect[0]])


@dataclass
class RasterState:
    image: np.ndarray
    mask: np.ndarray
    x: int
    y: int
    patch_rect: Rect | None

    @classmethod
    def capture(cls, layer: Layer) -> RasterState:
        return cls(layer.image.copy(), layer.mask.data.copy(), layer.x,
                   layer.y, layer.patch_rect)

    def preview_layer(self, source: Layer) -> Layer:
        h, w = self.image.shape[:2]
        return Layer(source.name, w, h, self.image, x=self.x, y=self.y)


class RasterTransform:
    """One raster or cut selection. Destination uses canvas coordinates."""

    def __init__(self, stack, target: str = "auto"):
        layer = stack.active_layer
        if (layer is None or not layer.accepts_pixel_edits
                or not stack.is_layer_visible_for_composition(layer)):
            raise ValueError("Select a visible raster layer")
        if target not in {"auto", "selection", "layer"}:
            raise ValueError("Unknown transform target")
        validate_size(layer.width, layer.height)
        selected = not stack.selection.is_empty
        self.target = ("selection" if selected else "layer") if target == "auto" else target
        self.layer = layer
        self.revision = stack.revision
        self.before = RasterState.capture(layer)
        self.selection_before = stack.selection.data.copy()
        self.selection_patch = None
        if self.target == "selection":
            bbox = stack.selection.bbox()
            if bbox is None:
                raise ValueError("Nothing selected")
            x0, y0 = max(bbox[0], layer.x), max(bbox[1], layer.y)
            x1, y1 = min(bbox[2], layer.bounds[2]), min(bbox[3], layer.bounds[3])
            if x1 <= x0 or y1 <= y0:
                raise ValueError("Selection does not intersect the active layer")
            self.selection_patch = self.selection_before[y0:y1, x0:x1].copy()
            ys, xs = np.nonzero(self.selection_patch)
            if not len(xs):
                raise ValueError("Selection does not intersect the active layer")
            # Crop to actual selected samples, not just the global selection bbox.
            ax, ay, bx, by = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
            self.selection_patch = self.selection_patch[ay:by, ax:bx]
            x0, y0, x1, y1 = x0 + ax, y0 + ay, x0 + bx, y0 + by
        else:
            x0, y0, x1, y1 = layer.bounds
        self.source_rect = (x0, y0, x1, y1)
        self.rect = self.source_rect
        self.original = self.before.image[y0-layer.y:y1-layer.y,
                                          x0-layer.x:x1-layer.x].copy()
        self.fragment = _linear(self.original)
        if self.selection_patch is not None:
            self.fragment *= self.selection_patch[:, :, None]
        self._scaled_size = None
        self._scaled_fragment = None
        self._scaled_rgba = None
        self._cut_original = self.original.copy()
        if self.selection_patch is not None:
            self._cut_original[:, :, 3] = np.rint(
                self.original[:, :, 3] * (1-self.selection_patch)).astype(np.uint8)

    def scaled_fragment(self) -> np.ndarray:
        size = (self.rect[2]-self.rect[0], self.rect[3]-self.rect[1])
        if size != self._scaled_size:
            source = _resize(self.fragment, *size)
            source[:, :, 3] = np.clip(source[:, :, 3], 0, 1)
            source[:, :, :3] = np.clip(source[:, :, :3], 0, source[:, :, 3:4])
            self._scaled_size = size
            self._scaled_fragment = source
            self._scaled_rgba = None
        return self._scaled_fragment

    def scaled_rgba(self) -> np.ndarray:
        source = self.scaled_fragment()
        if self._scaled_rgba is None:
            self._scaled_rgba = (
                self.original if self.target == "layer" and source.shape == self.original.shape
                else premultiplied_to_straight_rgba(source * 255.0))
        return self._scaled_rgba

    def preview_region(self, rect: Rect) -> np.ndarray:
        """Render only requested raster samples; never build a layer or masks."""
        x0, y0, x1, y1 = rect
        image = np.zeros((y1-y0, x1-x0, 4), dtype=np.uint8)
        b = self.before
        if self.target == "selection" or not self.changed:
            overlap = _region_views(image, rect, b.image,
                (b.x, b.y, b.x+b.image.shape[1], b.y+b.image.shape[0]))
            if overlap is not None:
                overlap[0][:] = overlap[1]
        if not self.changed:
            return image
        if self.target == "selection":
            overlap = _region_views(image, rect, self._cut_original, self.source_rect)
            if overlap is not None:
                overlap[0][:] = overlap[1]
            overlap = _region_views(image, rect, self.scaled_fragment(), self.rect)
            if overlap is not None:
                dst, src = overlap
                if np.all(src[:, :, 3] == 1):
                    # Opaque source-over is a copy. Cache its sRGB conversion
                    # across translations instead of gamma-converting every tile.
                    dst[:] = _region_views(image, rect, self.scaled_rgba(), self.rect)[1]
                else:
                    dst[:] = premultiplied_to_straight_rgba(
                        (src + _linear(dst)*(1-src[:, :, 3:4])) * 255.0)
        else:
            overlap = _region_views(image, rect, self.scaled_rgba(), self.rect)
            if overlap is not None:
                overlap[0][:] = overlap[1]
        return image

    @property
    def changed(self) -> bool:
        return self.rect != self.source_rect

    def set_rect(self, rect: Rect) -> None:
        x0, y0, x1, y1 = map(int, rect)
        validate_size(x1 - x0, y1 - y0)
        if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 65536:
            raise ValueError("Transform position is outside the supported range")
        if self.target == "selection":
            b = self.before
            validate_size(max(x1, b.x + b.image.shape[1]) - min(x0, b.x),
                          max(y1, b.y + b.image.shape[0]) - min(y0, b.y))
        self.rect = (x0, y0, x1, y1)

    def result(self) -> tuple[RasterState, np.ndarray]:
        if not self.changed:
            return self.before, self.selection_before
        x0, y0, x1, y1 = self.rect
        w, h = x1 - x0, y1 - y0
        source = self.scaled_fragment()
        b = self.before
        if self.target == "layer":
            # Preserve hidden RGB on integer translations and exact-size moves.
            image = (self.original.copy() if (h, w) == self.original.shape[:2]
                     else premultiplied_to_straight_rgba(source * 255.0))
            mask = np.clip(_resize(b.mask, w, h), 0, 1)
            patch = b.patch_rect
            if patch is not None:
                sx, sy = w / b.image.shape[1], h / b.image.shape[0]
                patch = (int(np.floor(patch[0]*sx)), int(np.floor(patch[1]*sy)),
                         int(np.ceil(patch[2]*sx)), int(np.ceil(patch[3]*sy)))
            return RasterState(image, mask, x0, y0, patch), self.selection_before

        bx, by = min(b.x, x0), min(b.y, y0)
        bw = max(b.x + b.image.shape[1], x1) - bx
        bh = max(b.y + b.image.shape[0], y1) - by
        image = np.zeros((bh, bw, 4), dtype=np.uint8)
        ox, oy = b.x - bx, b.y - by
        image[oy:oy+b.image.shape[0], ox:ox+b.image.shape[1]] = b.image
        sx0, sy0, sx1, sy1 = self.source_rect
        cut = image[sy0-by:sy1-by, sx0-bx:sx1-bx]
        cut[:, :, 3] = np.rint(cut[:, :, 3] * (1 - self.selection_patch)).astype(np.uint8)
        dst = image[y0-by:y1-by, x0-bx:x1-bx]
        combined = source + _linear(dst) * (1 - source[:, :, 3:4])
        dst[:] = premultiplied_to_straight_rgba(combined * 255.0)
        mask = np.zeros((bh, bw), dtype=np.float32)
        mask[oy:oy+b.mask.shape[0], ox:ox+b.mask.shape[1]] = b.mask
        patch = None if b.patch_rect is None else (
            b.patch_rect[0]+ox, b.patch_rect[1]+oy,
            b.patch_rect[2]+ox, b.patch_rect[3]+oy)
        selection = np.zeros_like(self.selection_before)
        dx0, dy0 = max(0, x0), max(0, y0)
        dx1, dy1 = min(selection.shape[1], x1), min(selection.shape[0], y1)
        if dx1 > dx0 and dy1 > dy0:
            moved_mask = np.clip(_resize(self.selection_patch, w, h), 0, 1)
            selection[dy0:dy1, dx0:dx1] = moved_mask[dy0-y0:dy1-y0, dx0-x0:dx1-x0]
        return RasterState(image, mask, bx, by, patch), selection


class TransformPreviewRenderer(LayerRenderer):
    """Reuse canonical layer ordering/opacity without mutating the document."""

    def __init__(self, stack, source: Layer, replacement: Layer | RasterTransform):
        super().__init__(stack)
        self.source = source
        self.replacement = replacement

    def set_replacement(self, replacement: Layer | RasterTransform, dirty_rect: Rect) -> None:
        self.replacement = replacement
        self.invalidate_tiles(self._stack._collect_affected_layers(self.source),
                              self._stack._tiles_for_rect(dirty_rect))

    def composite_preview_rect(self, rect: Rect) -> np.ndarray:
        # A sole layer at opacity 1 has nothing to composite with. An opaque
        # region of the top root likewise covers all lower layers/descendants.
        if (isinstance(self.replacement, RasterTransform)
                and self._stack.layers[0] is self.source
                and self.source.opacity == 1
                and self._stack.is_layer_visible_for_composition(self.source)):
            image = self.replacement.preview_region(rect)
            if len(self._stack.layers) == 1 and not self.source.children:
                image[image[:, :, 3] == 0] = 0
                return image
            if np.all(image[:, :, 3] == 255):
                return image
        return premultiplied_to_straight_rgba(
            self.composite_rect_premultiplied(*rect))

    def _layer_canvas_tile(self, layer, tx, ty):
        if layer is self.source and isinstance(self.replacement, RasterTransform):
            return self.replacement.preview_region(self._stack.tile_bounds(tx, ty))
        return super()._layer_canvas_tile(
            self.replacement if layer is self.source else layer, tx, ty)


class ApplyRasterTransformCommand:
    label = "Transform Selection / Layer"

    def __init__(self, session: RasterTransform):
        self.session = session
        self.label = "Transform Selection" if session.target == "selection" else "Transform Layer"

    def apply_with_history(self, stack):
        session = self.session
        layer = session.layer
        if stack.find_layer_by_id(layer.id) is not layer or stack.revision != session.revision:
            raise ValueError("Document changed during transform")
        if not session.changed:
            return None
        after, selection_after = session.result()
        # Restore the actual arrays, not copies: older pixel/mask/selection
        # history entries can hold references to those arrays. Each timeline
        # side is restored by subsequent undo operations before we switch it.
        before_storage = (layer.content, layer.mask, layer.x, layer.y,
                          layer.patch_rect, stack.selection)
        after_storage = (
            DenseTileGrid.from_array(after.image, tile_size=layer.content.tile_size),
            Mask(after.mask), after.x, after.y, after.patch_rect,
            Selection(selection_after) if session.target == "selection" else stack.selection,
        )

        def assign(storage):
            (layer.content, layer.mask, layer.x, layer.y,
             layer.patch_rect, stack.selection) = storage
            layer.image = layer.content.array
            layer.mark_pixels_changed()
            stack._rebuild_caches()
            stack.publish_change(DocumentChangeKind.STRUCTURE, layers=(layer,))

        try:
            assign(after_storage)
        except BaseException:
            assign(before_storage)
            raise
        return CommandDelta(
            lambda: assign(before_storage),
            lambda: assign(after_storage),
            sum(a.nbytes for storage in (before_storage, after_storage)
                for a in (storage[0].array, storage[1].data, storage[5].data)),
        )
