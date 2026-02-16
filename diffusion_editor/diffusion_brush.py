import numpy as np
from PIL import Image

PATCH_SIZE = 512


def extract_patch(composite: np.ndarray, center_x: int, center_y: int):
    """Extract a PATCH_SIZE x PATCH_SIZE region from the composite.

    Returns:
        patch_pil: PIL Image (RGB, resized to 512x512 if smaller)
        paste_x, paste_y: top-left corner in image coords
        patch_w, patch_h: actual extracted size before resize
    """
    h, w = composite.shape[:2]
    half = PATCH_SIZE // 2

    x0 = max(0, center_x - half)
    y0 = max(0, center_y - half)
    x1 = min(w, x0 + PATCH_SIZE)
    y1 = min(h, y0 + PATCH_SIZE)

    # re-adjust if clamped on right/bottom
    x0 = max(0, x1 - PATCH_SIZE)
    y0 = max(0, y1 - PATCH_SIZE)

    patch_arr = composite[y0:y1, x0:x1]
    patch_pil = Image.fromarray(patch_arr).convert("RGB")

    actual_w = x1 - x0
    actual_h = y1 - y0

    return patch_pil, x0, y0, actual_w, actual_h


def paste_result(layer_image: np.ndarray, result_pil: Image.Image,
                 paste_x: int, paste_y: int, patch_w: int, patch_h: int):
    """Paste diffusion result back onto layer_image.

    result_pil is 512x512 RGB from the pipeline.
    Resizes to (patch_w, patch_h) and overwrites the region.
    """
    result = result_pil.resize((patch_w, patch_h), Image.LANCZOS)
    result_arr = np.array(result.convert("RGBA"), dtype=np.uint8)
    result_arr[:, :, 3] = 255

    h, w = layer_image.shape[:2]
    rh, rw = result_arr.shape[:2]

    ex = min(paste_x + rw, w)
    ey = min(paste_y + rh, h)
    rw_clamp = ex - paste_x
    rh_clamp = ey - paste_y

    if rw_clamp <= 0 or rh_clamp <= 0:
        return

    layer_image[paste_y:ey, paste_x:ex] = result_arr[:rh_clamp, :rw_clamp]
