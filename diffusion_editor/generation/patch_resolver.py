"""Source patch preparation shared by generation tools."""

from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image

from .patch_image import extract_patch
from .types import GenerationError, PatchSource, Rect
from ..document.layer import Layer
from ..document.tool import DiffusionTool, InstructTool, LamaTool


def clip_rect_to_array(rect: Rect, image: np.ndarray) -> Rect:
    x0, y0, x1, y1 = rect
    h, w = image.shape[:2]
    x0 = max(0, min(int(x0), w))
    x1 = max(0, min(int(x1), w))
    y0 = max(0, min(int(y0), h))
    y1 = max(0, min(int(y1), h))
    return x0, y0, x1, y1


def source_patch_from_layer_patch(
        layer: Layer,
        composite: np.ndarray) -> PatchSource | GenerationError | None:
    if layer.patch_rect is None:
        return None

    raw_rect = layer.local_rect_to_canvas(layer.patch_rect)
    rect = clip_rect_to_array(raw_rect, composite)
    x0, y0, x1, y1 = rect
    if x1 - x0 < 1 or y1 - y0 < 1:
        return GenerationError(
            message="Patch is outside the image",
            log_message=(
                f"Layer patch rect is outside composite for layer {layer.name}: "
                f"local={layer.patch_rect} canvas={raw_rect}"
            ),
        )
    image = Image.fromarray(composite[y0:y1, x0:x1]).convert("RGB")
    return PatchSource(image=image, canvas_rect=rect, source="patch")


def source_patch_from_mask(
        layer: Layer,
        composite: np.ndarray,
        *,
        min_patch_size: int = 512,
        padding_scale: float = 1.25) -> PatchSource | None:
    if not layer.has_mask():
        return None
    bbox = layer.mask_bbox()
    center = layer.mask_center()
    if bbox is None or center is None:
        return None
    bx0, by0, bx1, by1 = bbox
    patch_size = max(bx1 - bx0, by1 - by0)
    patch_size = max(int(patch_size * padding_scale), min_patch_size)
    cx, cy = center
    patch, x0, y0, width, height = extract_patch(
        composite, cx, cy, patch_size=patch_size)
    return PatchSource(
        image=patch,
        canvas_rect=(x0, y0, x0 + width, y0 + height),
        source="mask_bbox",
    )


def source_patch_at_center(
        composite: np.ndarray,
        center_x: int,
        center_y: int,
        *,
        patch_size: int | None = None,
        source: Literal["patch", "mask_bbox", "existing"] = "existing"
        ) -> PatchSource:
    if patch_size is None:
        patch, x0, y0, width, height = extract_patch(
            composite,
            center_x,
            center_y,
        )
    else:
        patch, x0, y0, width, height = extract_patch(
            composite,
            center_x,
            center_y,
            patch_size=patch_size,
        )
    return PatchSource(
        image=patch,
        canvas_rect=(x0, y0, x0 + width, y0 + height),
        source=source,
    )


def source_patch_from_existing_rect(
        composite: np.ndarray,
        rect: Rect) -> PatchSource | GenerationError:
    x0, y0, x1, y1 = rect
    width = x1 - x0
    height = y1 - y0
    if width <= 0 or height <= 0:
        return GenerationError(
            message="Existing patch is empty",
            log_message=f"Existing source patch rect is empty: {rect}",
        )
    center_x = x0 + width // 2
    center_y = y0 + height // 2
    return source_patch_at_center(
        composite,
        center_x,
        center_y,
        patch_size=max(width, height),
        source="existing",
    )


def resolve_source_patch(
        layer: Layer,
        composite: np.ndarray,
        *,
        fallback_canvas_rect: Rect | None = None
        ) -> PatchSource | GenerationError | None:
    patch = source_patch_from_layer_patch(layer, composite)
    if patch is not None:
        return patch
    patch = source_patch_from_mask(layer, composite)
    if patch is not None:
        return patch
    if fallback_canvas_rect is not None:
        return source_patch_from_existing_rect(composite, fallback_canvas_rect)
    return None


def apply_patch_source_to_tool(
        tool: DiffusionTool | InstructTool | LamaTool,
        patch: PatchSource) -> None:
    x0, y0, x1, y1 = patch.canvas_rect
    tool.source_patch = patch.image
    tool.patch_x = x0
    tool.patch_y = y0
    tool.patch_w = x1 - x0
    tool.patch_h = y1 - y0


def extract_layer_mask_patch(layer: Layer, canvas_rect: Rect) -> Image.Image:
    x0, y0, x1, y1 = canvas_rect
    width = max(0, x1 - x0)
    height = max(0, y1 - y0)
    mask_crop = np.zeros((height, width), dtype=np.float32)

    lx0 = x0 - layer.x
    ly0 = y0 - layer.y
    lx1 = lx0 + width
    ly1 = ly0 + height

    src_x0 = max(0, lx0)
    src_y0 = max(0, ly0)
    src_x1 = min(layer.width, lx1)
    src_y1 = min(layer.height, ly1)
    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - lx0
        dst_y0 = src_y0 - ly0
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        mask_crop[dst_y0:dst_y1, dst_x0:dst_x1] = (
            layer.mask.data[src_y0:src_y1, src_x0:src_x1]
        )

    return Image.fromarray((mask_crop * 255).astype(np.uint8), "L")
