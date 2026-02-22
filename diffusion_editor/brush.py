import numpy as np


class Brush:
    def __init__(self):
        self.size = 20
        self.color = (255, 255, 255, 255)
        self.hardness = 0.4  # 0.0 = fully soft, 1.0 = hard edge
        self._alpha_stamp = None  # 2D (d, d) float32, 0.0-1.0
        self._rebuild_stamp()

    def set_size(self, size):
        self.size = max(1, min(size, 500))
        self._rebuild_stamp()

    def set_color(self, r, g, b, a=255):
        self.color = (r, g, b, a)

    def set_hardness(self, hardness):
        self.hardness = max(0.0, min(hardness, 1.0))
        self._rebuild_stamp()

    def _rebuild_stamp(self):
        d = self.size
        if d < 1:
            self._alpha_stamp = np.zeros((1, 1), dtype=np.float32)
            return
        y, x = np.ogrid[-d / 2:d / 2, -d / 2:d / 2]
        dist = np.sqrt(x * x + y * y)
        radius = d / 2

        if self.hardness >= 1.0:
            self._alpha_stamp = (dist <= radius).astype(np.float32)
        else:
            inner = radius * self.hardness
            self._alpha_stamp = np.clip(
                (radius - dist) / max(radius - inner, 0.001), 0, 1
            ).astype(np.float32)

    def dab_to_mask(self, mask: np.ndarray, cx: int, cy: int):
        """Apply brush dab to 2D uint8 mask using MAX blending (no buildup)."""
        stamp = self._alpha_stamp
        sh, sw = stamp.shape[:2]
        ih, iw = mask.shape[:2]

        x0 = cx - sw // 2
        y0 = cy - sh // 2
        sx0 = max(0, -x0)
        sy0 = max(0, -y0)
        sx1 = min(sw, iw - x0)
        sy1 = min(sh, ih - y0)
        dx0 = max(0, x0)
        dy0 = max(0, y0)
        dx1 = dx0 + (sx1 - sx0)
        dy1 = dy0 + (sy1 - sy0)

        if dx0 >= dx1 or dy0 >= dy1:
            return

        stamp_u8 = (stamp[sy0:sy1, sx0:sx1] * self.color[3]).astype(np.uint8)
        mask[dy0:dy1, dx0:dx1] = np.maximum(
            mask[dy0:dy1, dx0:dx1], stamp_u8)

    def stroke_to_mask(self, mask: np.ndarray,
                       x0: int, y0: int, x1: int, y1: int):
        """Draw interpolated stroke into mask using MAX blending."""
        dx = x1 - x0
        dy = y1 - y0
        dist = max(abs(dx), abs(dy), 1)
        spacing = max(1, self.size // 4)
        steps = max(1, int(dist / spacing))
        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = int(x0 + dx * t)
            y = int(y0 + dy * t)
            self.dab_to_mask(mask, x, y)


def composite_stroke(layer_image: np.ndarray, stroke_mask: np.ndarray,
                     color: tuple):
    """Composite a finished stroke onto layer using straight-alpha source-over.

    layer_image: (H, W, 4) uint8 RGBA (straight alpha)
    stroke_mask: (H, W) uint8 — stroke opacity per pixel
    color: (r, g, b, a) brush color
    """
    where = stroke_mask > 0
    if not np.any(where):
        return

    r, g, b, _a = color
    sa = stroke_mask[where].astype(np.float32) / 255.0
    da = layer_image[where, 3].astype(np.float32) / 255.0

    out_a = sa + da * (1.0 - sa)
    safe_a = np.maximum(out_a, 1.0 / 255.0)
    inv_sa = 1.0 - sa

    for c, src_val in enumerate((r, g, b)):
        dst_c = layer_image[where, c].astype(np.float32)
        layer_image[where, c] = np.clip(
            (src_val * sa + dst_c * da * inv_sa) / safe_a,
            0, 255).astype(np.uint8)

    layer_image[where, 3] = np.clip(out_a * 255.0, 0, 255).astype(np.uint8)


def erase_stroke(layer_image: np.ndarray, stroke_mask: np.ndarray):
    """Erase pixels from layer by reducing alpha according to stroke_mask.

    layer_image: (H, W, 4) uint8 RGBA (straight alpha)
    stroke_mask: (H, W) uint8 — erase strength per pixel (255 = fully erase)
    """
    where = stroke_mask > 0
    if not np.any(where):
        return
    keep = 1.0 - stroke_mask[where].astype(np.float32) / 255.0
    layer_image[where, 3] = np.clip(
        layer_image[where, 3].astype(np.float32) * keep, 0, 255).astype(np.uint8)
