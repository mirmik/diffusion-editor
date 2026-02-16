import numpy as np
from PIL import Image

PATCH_SIZE = 512


def extract_patch(composite: np.ndarray, center_x: int, center_y: int,
                  patch_size: int = PATCH_SIZE):
    """Extract a patch_size x patch_size region from the composite.

    Returns:
        patch_pil: PIL Image (RGB)
        paste_x, paste_y: top-left corner in image coords
        patch_w, patch_h: actual extracted size
    """
    h, w = composite.shape[:2]
    patch_size = max(patch_size, 8)
    half = patch_size // 2

    x0 = max(0, center_x - half)
    y0 = max(0, center_y - half)
    x1 = min(w, x0 + patch_size)
    y1 = min(h, y0 + patch_size)

    # re-adjust if clamped on right/bottom
    x0 = max(0, x1 - patch_size)
    y0 = max(0, y1 - patch_size)

    patch_arr = composite[y0:y1, x0:x1]
    patch_pil = Image.fromarray(patch_arr).convert("RGB")

    actual_w = x1 - x0
    actual_h = y1 - y0

    return patch_pil, x0, y0, actual_w, actual_h


def extract_mask_patch(mask: np.ndarray, patch_x: int, patch_y: int,
                       patch_w: int, patch_h: int) -> Image.Image:
    """Extract the mask region corresponding to a patch. Returns PIL Image 'L'."""
    mask_crop = mask[patch_y:patch_y + patch_h, patch_x:patch_x + patch_w]
    return Image.fromarray(mask_crop, "L")


def paste_result(layer_image: np.ndarray, result_pil: Image.Image,
                 paste_x: int, paste_y: int, patch_w: int, patch_h: int,
                 mask: np.ndarray = None):
    """Paste diffusion result back onto layer_image.

    result_pil is 512x512 RGB from the pipeline.
    Resizes to (patch_w, patch_h) and overwrites the region.

    If mask is provided (2D uint8, full layer size), mask values become
    the alpha channel of the pasted result (0 = transparent, 255 = opaque).
    """
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
        result_slice[:, :, 3] = mask_slice
    else:
        result_slice[:, :, 3] = 255

    layer_image[paste_y:ey, paste_x:ex] = result_slice
