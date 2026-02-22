import numpy as np
from threading import Thread
from PIL import Image


class LamaEngine:
    def __init__(self):
        self._model = None
        self._busy = False
        self._result = None   # PIL.Image or None
        self._error = None
        self._thread = None

    @property
    def is_busy(self) -> bool:
        return self._busy

    def _ensure_loaded(self):
        if self._model is None:
            from simple_lama_inpainting import SimpleLama
            self._model = SimpleLama()

    def submit(self, image: Image.Image, mask: Image.Image):
        if self._busy:
            return False
        self._busy = True
        self._result = None
        self._error = None
        self._thread = Thread(target=self._run, args=(image, mask), daemon=True)
        self._thread.start()
        return True

    def _run(self, image: Image.Image, mask: Image.Image):
        try:
            print("[LamaEngine] loading model...")
            self._ensure_loaded()
            print("[LamaEngine] running inference...")
            image = image.convert("RGB")
            mask = mask.convert("L")
            result = self._model(image, mask)
            self._result = result
            print(f"[LamaEngine] done, result size: {result.size}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._error = str(e)
            print(f"[LamaEngine] error: {e}")
        self._busy = False

    def poll(self) -> tuple[Image.Image | None, str | None]:
        if self._busy:
            return None, None
        result, error = self._result, self._error
        if result is None and error is None:
            return None, None
        self._result = None
        self._error = None
        return result, error
