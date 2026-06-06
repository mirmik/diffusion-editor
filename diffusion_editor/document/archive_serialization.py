"""Low-level zip archive serialization helpers."""

from __future__ import annotations

import io
import zipfile

import numpy as np
from PIL import Image


def save_array_to_zip(zf: zipfile.ZipFile, path: str, arr: np.ndarray):
    buf = io.BytesIO()
    np.save(buf, arr)
    zf.writestr(path, buf.getvalue())


def load_array_from_zip(
        zf: zipfile.ZipFile,
        path: str,
        mode: str | None = None) -> np.ndarray:
    data = zf.read(path)
    if path.endswith(".npy"):
        return np.load(io.BytesIO(data))
    img = Image.open(io.BytesIO(data))
    if mode:
        img = img.convert(mode)
    return np.array(img, dtype=np.uint8)


def load_pil_from_zip(
        zf: zipfile.ZipFile,
        path: str,
        mode: str = "RGB") -> Image.Image:
    data = zf.read(path)
    if path.endswith(".npy"):
        arr = np.load(io.BytesIO(data))
        return Image.fromarray(arr).convert(mode)
    return Image.open(io.BytesIO(data)).convert(mode)
