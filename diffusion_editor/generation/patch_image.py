"""Image patch extraction helpers used by generation workflows."""

from __future__ import annotations

import numpy as np
from PIL import Image

PATCH_SIZE = 512


def extract_patch(composite: np.ndarray, center_x: int, center_y: int,
                  patch_size: int = PATCH_SIZE):
    """Extract a square region from a composite image."""
    h, w = composite.shape[:2]
    patch_size = max(patch_size, 8)
    half = patch_size // 2

    x0 = max(0, center_x - half)
    y0 = max(0, center_y - half)
    x1 = min(w, x0 + patch_size)
    y1 = min(h, y0 + patch_size)

    x0 = max(0, x1 - patch_size)
    y0 = max(0, y1 - patch_size)

    patch_arr = composite[y0:y1, x0:x1]
    patch_pil = Image.fromarray(patch_arr).convert("RGB")

    return patch_pil, x0, y0, x1 - x0, y1 - y0


def extract_mask_patch(mask: np.ndarray, patch_x: int, patch_y: int,
                       patch_w: int, patch_h: int) -> Image.Image:
    """Extract the mask region corresponding to a patch."""
    mask_crop = mask[patch_y:patch_y + patch_h, patch_x:patch_x + patch_w]
    if mask_crop.dtype == np.float32 or mask_crop.dtype == np.float64:
        mask_crop = (mask_crop * 255).astype(np.uint8)
    return Image.fromarray(mask_crop, "L")
