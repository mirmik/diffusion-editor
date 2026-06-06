"""Pixel operations for applying generated images to document layers."""

from __future__ import annotations

import numpy as np
from PIL import Image


def paste_result(layer_image: np.ndarray, result_pil: Image.Image,
                 paste_x: int, paste_y: int, patch_w: int, patch_h: int,
                 mask: np.ndarray = None):
    """Paste a generated patch back onto a layer image."""
    result = result_pil.resize((patch_w, patch_h), Image.LANCZOS)
    result_arr = np.array(result.convert("RGBA"), dtype=np.uint8)

    h, w = layer_image.shape[:2]
    rh, rw = result_arr.shape[:2]

    ex = min(paste_x + rw, w)
    ey = min(paste_y + rh, h)
    rw_clamp = ex - paste_x
    rh_clamp = ey - paste_y

    if rw_clamp <= 0 or rh_clamp <= 0:
        return

    result_slice = result_arr[:rh_clamp, :rw_clamp]

    if mask is not None:
        mask_slice = mask[paste_y:ey, paste_x:ex]
        if mask_slice.dtype == np.float32 or mask_slice.dtype == np.float64:
            mask_slice = (mask_slice * 255).astype(np.uint8)
        result_slice[:, :, 3] = mask_slice
    else:
        result_slice[:, :, 3] = 255

    layer_image[paste_y:ey, paste_x:ex] = result_slice
