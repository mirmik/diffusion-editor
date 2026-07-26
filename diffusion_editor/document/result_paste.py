"""Pixel operations for applying generated images to document layers."""

from __future__ import annotations

import operator

import numpy as np
from PIL import Image


def paste_result(layer_image: np.ndarray, result_pil: Image.Image,
                 paste_x: int, paste_y: int, patch_w: int, patch_h: int,
                 mask: np.ndarray = None) -> bool:
    """Paste a generated patch back onto a layer image.

    Coordinates are layer-local.  The source and destination are clipped as a
    pair, including when the patch starts above or to the left of the layer.
    ``mask``, when present, is expected in full layer coordinates.

    Returns whether the patch intersects the layer.  All validation and result
    preparation happen before the destination array is mutated.
    """
    if not isinstance(layer_image, np.ndarray):
        raise TypeError("layer_image must be a numpy array")
    if layer_image.ndim != 3 or layer_image.shape[2] != 4:
        raise ValueError("layer_image must have shape (height, width, 4)")
    if layer_image.dtype != np.uint8:
        raise ValueError("layer_image must use uint8 RGBA pixels")

    if isinstance(paste_x, bool) or isinstance(paste_y, bool):
        raise ValueError("paste coordinates must be integers")
    try:
        paste_x = operator.index(paste_x)
        paste_y = operator.index(paste_y)
    except TypeError as exc:
        raise ValueError("paste coordinates must be integers") from exc
    if isinstance(patch_w, bool) or isinstance(patch_h, bool):
        raise ValueError("patch dimensions must be positive integers")
    try:
        patch_w = operator.index(patch_w)
        patch_h = operator.index(patch_h)
    except TypeError as exc:
        raise ValueError(
            "patch dimensions must be positive integers") from exc
    if patch_w < 1:
        raise ValueError("patch_w must be a positive integer")
    if patch_h < 1:
        raise ValueError("patch_h must be a positive integer")

    mask_arr = None
    if mask is not None:
        mask_arr = np.asarray(mask)
        if mask_arr.ndim != 2:
            raise ValueError("mask must be a two-dimensional array")
        if mask_arr.shape != layer_image.shape[:2]:
            raise ValueError(
                "mask shape must match the layer image dimensions")

    result = result_pil.resize((patch_w, patch_h), Image.LANCZOS)
    result_arr = np.array(result.convert("RGBA"), dtype=np.uint8)

    h, w = layer_image.shape[:2]
    rh, rw = result_arr.shape[:2]

    dst_x0 = max(0, paste_x)
    dst_y0 = max(0, paste_y)
    dst_x1 = min(w, paste_x + rw)
    dst_y1 = min(h, paste_y + rh)

    if dst_x0 >= dst_x1 or dst_y0 >= dst_y1:
        return False

    src_x0 = dst_x0 - paste_x
    src_y0 = dst_y0 - paste_y
    src_x1 = src_x0 + (dst_x1 - dst_x0)
    src_y1 = src_y0 + (dst_y1 - dst_y0)
    result_slice = result_arr[src_y0:src_y1, src_x0:src_x1].copy()

    if mask_arr is not None:
        mask_slice = mask_arr[dst_y0:dst_y1, dst_x0:dst_x1]
        if np.issubdtype(mask_slice.dtype, np.bool_):
            mask_slice = mask_slice.astype(np.uint8) * 255
        elif np.issubdtype(mask_slice.dtype, np.floating):
            mask_slice = np.clip(mask_slice, 0.0, 1.0)
            mask_slice = np.rint(mask_slice * 255).astype(np.uint8)
        else:
            mask_slice = np.clip(mask_slice, 0, 255).astype(np.uint8)
        result_slice[:, :, 3] = mask_slice
    else:
        result_slice[:, :, 3] = 255

    layer_image[dst_y0:dst_y1, dst_x0:dst_x1] = result_slice
    return True
