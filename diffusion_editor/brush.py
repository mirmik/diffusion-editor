import numpy as np


class Brush:
    def __init__(self):
        self.size = 20
        self.color = (255, 255, 255, 255)
        self.hardness = 0.4  # 0.0 = fully soft, 1.0 = hard edge
        self.flow = 1.0  # 0.0 = no effect, 1.0 = full strength
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

    def set_flow(self, flow):
        self.flow = max(0.0, min(flow, 1.0))

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
        """Apply brush dab to 2D uint8 mask using MAX blending (no buildup).

        Returns dirty rect (dx0, dy0, dx1, dy1) or None if nothing changed.
        """
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
            return None

        stamp_u8 = (stamp[sy0:sy1, sx0:sx1] * self.color[3] * self.flow).astype(np.uint8)
        mask[dy0:dy1, dx0:dx1] = np.maximum(
            mask[dy0:dy1, dx0:dx1], stamp_u8)
        return (dx0, dy0, dx1, dy1)

    def stroke_to_mask(self, mask: np.ndarray,
                       x0: int, y0: int, x1: int, y1: int):
        """Draw smooth stroke segment (capsule shape) using distance-to-segment.

        Returns dirty rect (dx0, dy0, dx1, dy1) or None.
        """
        ih, iw = mask.shape[:2]
        radius = self.size / 2.0

        # Bounding box of capsule
        bx0 = max(0, int(min(x0, x1) - radius))
        by0 = max(0, int(min(y0, y1) - radius))
        bx1 = min(iw, int(max(x0, x1) + radius) + 1)
        by1 = min(ih, int(max(y0, y1) + radius) + 1)
        if bx0 >= bx1 or by0 >= by1:
            return None

        # Pixel grid inside bounding box
        yy, xx = np.mgrid[by0:by1, bx0:bx1]
        xx = xx.astype(np.float32)
        yy = yy.astype(np.float32)

        # Distance from each pixel to the line segment
        sdx = float(x1 - x0)
        sdy = float(y1 - y0)
        seg_len_sq = sdx * sdx + sdy * sdy

        if seg_len_sq < 0.5:
            # Degenerate segment — single dab
            return self.dab_to_mask(mask, x0, y0)

        # Project onto segment, clamp t to [0,1]
        t = ((xx - x0) * sdx + (yy - y0) * sdy) / seg_len_sq
        np.clip(t, 0.0, 1.0, out=t)

        # Closest point on segment
        cx = x0 + t * sdx
        cy = y0 + t * sdy
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

        # Brush profile
        if self.hardness >= 1.0:
            alpha = (dist <= radius).astype(np.float32)
        else:
            inner = radius * self.hardness
            alpha = np.clip(
                (radius - dist) / max(radius - inner, 0.001), 0.0, 1.0)

        stamp_u8 = (alpha * self.color[3] * self.flow).astype(np.uint8)
        mask[by0:by1, bx0:bx1] = np.maximum(
            mask[by0:by1, bx0:bx1], stamp_u8)
        return (bx0, by0, bx1, by1)


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
