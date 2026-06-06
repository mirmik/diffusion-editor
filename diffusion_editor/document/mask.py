"""Mask and Selection classes - float32 [0..1] HxW arrays."""

import numpy as np


def coerce_mask_data(data: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 [0..1] 2D mask array."""
    arr = np.asarray(data)
    if arr.ndim != 2:
        raise ValueError(f"mask must be a 2D array, got shape {arr.shape}")
    arr = arr.astype(np.float32, copy=False)
    if arr.size and arr.max() > 1.0:
        arr = arr / 255.0
    return np.ascontiguousarray(np.clip(arr, 0.0, 1.0))


class Mask:
    """Float32 [0..1] HxW mask on a single layer.

    Zero = no mask, 1.0 = fully masked.
    Used by diffusion/lama/instruct engines as the processing area.
    """

    def __init__(self, data: np.ndarray | None = None,
                 height: int = 0, width: int = 0):
        if data is not None:
            self._data = coerce_mask_data(data)
        else:
            self._data = np.zeros((height, width), dtype=np.float32)

    @property
    def data(self) -> np.ndarray:
        return self._data

    def copy(self) -> 'Mask':
        return Mask(self._data.copy())

    def clear(self) -> None:
        self._data[:] = 0.0

    @property
    def is_empty(self) -> bool:
        return not np.any(self._data > 0.0)

    def bbox(self) -> tuple[int, int, int, int] | None:
        """Return (x0, y0, x1, y1) tight bounding box or None."""
        mask = self._data > 0.0
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows):
            return None
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        return int(x0), int(y0), int(x1) + 1, int(y1) + 1

    def center(self) -> tuple[int, int] | None:
        b = self.bbox()
        if b is None:
            return None
        return (b[0] + b[2]) // 2, (b[1] + b[3]) // 2

    @staticmethod
    def zeros(height: int, width: int) -> 'Mask':
        return Mask(np.zeros((height, width), dtype=np.float32))

    def to_uint8(self) -> np.ndarray:
        """Convert to uint8 [0..255] for PIL / external consumers."""
        return (np.clip(self._data, 0.0, 1.0) * 255.0).astype(np.uint8)

    @staticmethod
    def from_uint8(arr: np.ndarray) -> 'Mask':
        """Create from uint8 [0..255] array (e.g. old project format)."""
        return Mask(arr.astype(np.float32) / 255.0)


class Selection:
    """Float32 [0..1] HxW document-level selection.

    One per LayerStack. Represents "where editing is allowed."
    Same data format as Mask, but semantically distinct.
    """

    def __init__(self, data: np.ndarray | None = None,
                 height: int = 0, width: int = 0):
        if data is not None:
            self._data = coerce_mask_data(data)
        else:
            self._data = np.zeros((height, width), dtype=np.float32)

    @property
    def data(self) -> np.ndarray:
        return self._data

    def copy(self) -> 'Selection':
        return Selection(self._data.copy())

    def clear(self) -> None:
        self._data[:] = 0.0

    @property
    def is_empty(self) -> bool:
        return not np.any(self._data > 0.0)

    def bbox(self) -> tuple[int, int, int, int] | None:
        mask = self._data > 0.0
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not np.any(rows):
            return None
        y0, y1 = np.where(rows)[0][[0, -1]]
        x0, x1 = np.where(cols)[0][[0, -1]]
        return int(x0), int(y0), int(x1) + 1, int(y1) + 1

    def center(self) -> tuple[int, int] | None:
        b = self.bbox()
        if b is None:
            return None
        return (b[0] + b[2]) // 2, (b[1] + b[3]) // 2

    def to_mask(self) -> Mask:
        return Mask(self._data.copy())
