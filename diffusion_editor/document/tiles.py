from __future__ import annotations

from typing import Iterable

import numpy as np


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


class BaseTileGrid:
    def __init__(self, width: int, height: int, tile_size: int = 256,
                 channels: int = 4, dtype=np.uint8):
        self.width = int(width)
        self.height = int(height)
        self.tile_size = int(tile_size)
        self.channels = int(channels)
        self.dtype = dtype

    @property
    def tiles_x(self) -> int:
        return _ceil_div(self.width, self.tile_size)

    @property
    def tiles_y(self) -> int:
        return _ceil_div(self.height, self.tile_size)

    def tile_bounds(self, tx: int, ty: int) -> tuple[int, int, int, int]:
        x0 = tx * self.tile_size
        y0 = ty * self.tile_size
        x1 = min(self.width, x0 + self.tile_size)
        y1 = min(self.height, y0 + self.tile_size)
        return x0, y0, x1, y1

    def tile_shape(self, tx: int, ty: int) -> tuple[int, int, int]:
        x0, y0, x1, y1 = self.tile_bounds(tx, ty)
        return (y1 - y0, x1 - x0, self.channels)

    def iter_tiles_for_rect(self, x0: int, y0: int, x1: int, y1: int
                            ) -> Iterable[tuple[int, int, int, int, int, int]]:
        if x1 <= x0 or y1 <= y0:
            return []
        tx0 = max(0, x0 // self.tile_size)
        ty0 = max(0, y0 // self.tile_size)
        tx1 = min(self.tiles_x - 1, (x1 - 1) // self.tile_size)
        ty1 = min(self.tiles_y - 1, (y1 - 1) // self.tile_size)
        result = []
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                bx0, by0, bx1, by1 = self.tile_bounds(tx, ty)
                result.append((tx, ty, bx0, by0, bx1, by1))
        return result

    def get_tile(self, tx: int, ty: int) -> np.ndarray | None:
        raise NotImplementedError

    def set_tile(self, tx: int, ty: int, tile: np.ndarray) -> None:
        raise NotImplementedError

    def clear_tile(self, tx: int, ty: int) -> None:
        raise NotImplementedError

    def to_full_array(self) -> np.ndarray:
        h, w = self.height, self.width
        full = np.zeros((h, w, self.channels), dtype=self.dtype)
        for ty in range(self.tiles_y):
            for tx in range(self.tiles_x):
                tile = self.get_tile(tx, ty)
                if tile is None:
                    continue
                x0, y0, x1, y1 = self.tile_bounds(tx, ty)
                full[y0:y1, x0:x1] = tile
        return full


class DenseTileGrid(BaseTileGrid):
    def __init__(self, width: int, height: int, tile_size: int = 256,
                 channels: int = 4, dtype=np.uint8, array: np.ndarray | None = None):
        super().__init__(width, height, tile_size, channels, dtype)
        if array is None:
            self.array = np.zeros((height, width, channels), dtype=dtype)
        else:
            if array.ndim == 2:
                array = array[:, :, None]
            if array.shape[0] != height or array.shape[1] != width:
                raise ValueError("DenseTileGrid array size mismatch")
            if array.shape[2] != channels:
                raise ValueError("DenseTileGrid channel mismatch")
            self.array = np.ascontiguousarray(array)

    @classmethod
    def from_array(cls, array: np.ndarray, tile_size: int = 256):
        if array.ndim == 2:
            h, w = array.shape
            channels = 1
        else:
            h, w, channels = array.shape
        return cls(w, h, tile_size=tile_size, channels=channels, array=array)

    def get_tile(self, tx: int, ty: int) -> np.ndarray | None:
        x0, y0, x1, y1 = self.tile_bounds(tx, ty)
        return self.array[y0:y1, x0:x1]

    def set_tile(self, tx: int, ty: int, tile: np.ndarray) -> None:
        x0, y0, x1, y1 = self.tile_bounds(tx, ty)
        self.array[y0:y1, x0:x1] = tile

    def clear_tile(self, tx: int, ty: int) -> None:
        x0, y0, x1, y1 = self.tile_bounds(tx, ty)
        self.array[y0:y1, x0:x1] = 0


class SparseTileGrid(BaseTileGrid):
    def __init__(self, width: int, height: int, tile_size: int = 256,
                 channels: int = 4, dtype=np.uint8):
        super().__init__(width, height, tile_size, channels, dtype)
        self._tiles: dict[tuple[int, int], np.ndarray] = {}

    def get_tile(self, tx: int, ty: int) -> np.ndarray | None:
        return self._tiles.get((tx, ty))

    def ensure_tile(self, tx: int, ty: int) -> np.ndarray:
        key = (tx, ty)
        tile = self._tiles.get(key)
        if tile is not None:
            return tile
        h, w, c = self.tile_shape(tx, ty)
        tile = np.zeros((h, w, c), dtype=self.dtype)
        self._tiles[key] = tile
        return tile

    def set_tile(self, tx: int, ty: int, tile: np.ndarray) -> None:
        self._tiles[(tx, ty)] = np.ascontiguousarray(tile)

    def clear_tile(self, tx: int, ty: int) -> None:
        self._tiles.pop((tx, ty), None)

    def iter_tiles(self) -> Iterable[tuple[int, int, np.ndarray]]:
        for (tx, ty), tile in self._tiles.items():
            yield tx, ty, tile
