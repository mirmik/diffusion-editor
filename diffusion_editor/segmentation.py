import numpy as np
from threading import Thread
from PIL import Image


class SegmentationEngine:
    def __init__(self):
        self._session = None
        self._busy = False
        self._result = None    # np.ndarray (H, W) uint8
        self._error = None
        self._thread = None

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def is_busy(self) -> bool:
        return self._busy

    def _ensure_loaded(self):
        """Lazy-load сессии при первом вызове."""
        if self._session is None:
            from rembg import new_session
            self._session = new_session("u2net")

    def submit(self, image: np.ndarray, invert: bool = True):
        """Запускает сегментацию в фоновом потоке.
        image: RGBA или RGB np.ndarray (H, W, 3/4).
        invert=True → маска фона, False → маска объекта.
        """
        if self._busy:
            return False
        self._busy = True
        self._result = None
        self._error = None
        self._thread = Thread(target=self._run, args=(image, invert), daemon=True)
        self._thread.start()
        return True

    def _run(self, image_arr, invert):
        try:
            from rembg import remove
            print("[Segmentation] loading model...")
            self._ensure_loaded()
            print("[Segmentation] model loaded, running inference...")
            pil_img = Image.fromarray(image_arr[:, :, :3])
            fg_mask_pil = remove(pil_img, session=self._session, only_mask=True)
            fg_mask = np.array(fg_mask_pil, dtype=np.uint8)
            print(f"[Segmentation] fg_mask shape={fg_mask.shape}, min={fg_mask.min()}, max={fg_mask.max()}")
            if invert:
                mask = 255 - fg_mask  # background mask
            else:
                mask = fg_mask
            self._result = mask.astype(np.uint8)
            print(f"[Segmentation] done, result shape={self._result.shape}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._error = str(e)
            print(f"[Segmentation] error: {e}")
        self._busy = False

    def poll(self) -> tuple[np.ndarray | None, str | None]:
        if self._busy:
            return None, None
        result, error = self._result, self._error
        if result is None and error is None:
            return None, None
        self._result = None
        self._error = None
        return result, error
