import numpy as np


class Brush:
    def __init__(self):
        self.size = 20
        self.color = (255, 255, 255, 255)
        self.hardness = 0.8  # 0.0 = fully soft, 1.0 = hard edge
        self._stamp = None
        self._rebuild_stamp()

    def set_size(self, size):
        self.size = max(1, min(size, 500))
        self._rebuild_stamp()

    def set_color(self, r, g, b, a=255):
        self.color = (r, g, b, a)
        self._rebuild_stamp()

    def set_hardness(self, hardness):
        self.hardness = max(0.0, min(hardness, 1.0))
        self._rebuild_stamp()

    def _rebuild_stamp(self):
        d = self.size
        if d < 1:
            self._stamp = np.zeros((1, 1, 4), dtype=np.uint8)
            return
        y, x = np.ogrid[-d/2:d/2, -d/2:d/2]
        dist = np.sqrt(x*x + y*y)
        radius = d / 2

        if self.hardness >= 1.0:
            alpha_mask = (dist <= radius).astype(np.float32)
        else:
            inner = radius * self.hardness
            alpha_mask = np.clip((radius - dist) / max(radius - inner, 0.001), 0, 1)

        stamp = np.zeros((d, d, 4), dtype=np.uint8)
        stamp[:, :, 0] = self.color[0]
        stamp[:, :, 1] = self.color[1]
        stamp[:, :, 2] = self.color[2]
        stamp[:, :, 3] = (alpha_mask * self.color[3]).astype(np.uint8)
        self._stamp = stamp

    def dab(self, layer_image: np.ndarray, cx: int, cy: int):
        stamp = self._stamp
        sh, sw = stamp.shape[:2]
        ih, iw = layer_image.shape[:2]

        x0 = cx - sw // 2
        y0 = cy - sh // 2

        # clip to image bounds
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

        src = stamp[sy0:sy1, sx0:sx1].astype(np.float32)
        dst = layer_image[dy0:dy1, dx0:dx1].astype(np.float32)

        alpha = src[:, :, 3:4] / 255.0
        inv_alpha = 1.0 - alpha

        dst[:, :, :3] = src[:, :, :3] * alpha + dst[:, :, :3] * inv_alpha
        dst[:, :, 3:4] = np.clip(src[:, :, 3:4] + dst[:, :, 3:4] * inv_alpha, 0, 255)

        layer_image[dy0:dy1, dx0:dx1] = dst.astype(np.uint8)

    def stroke(self, layer_image: np.ndarray, x0: int, y0: int, x1: int, y1: int):
        dx = x1 - x0
        dy = y1 - y0
        dist = max(abs(dx), abs(dy), 1)
        spacing = max(1, self.size // 4)
        steps = max(1, int(dist / spacing))

        for i in range(steps + 1):
            t = i / max(steps, 1)
            x = int(x0 + dx * t)
            y = int(y0 + dy * t)
            self.dab(layer_image, x, y)
