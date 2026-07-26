"""Low-level zip archive serialization helpers."""

from __future__ import annotations

import io
import math
import zipfile

import numpy as np
from PIL import Image

MAX_NPY_HEADER_BYTES = 10_000
MAX_ARRAY_BYTES = 1024 * 1024 * 1024


def save_array_to_zip(zf: zipfile.ZipFile, path: str, arr: np.ndarray):
    buf = io.BytesIO()
    np.save(buf, arr)
    zf.writestr(path, buf.getvalue())


def _inspect_npy_entry(
        zf: zipfile.ZipFile,
        path: str,
        *,
        max_array_bytes: int = MAX_ARRAY_BYTES,
) -> tuple[tuple[int, ...], np.dtype, int]:
    """Validate an NPY header and payload size before NumPy can allocate."""
    try:
        info = zf.getinfo(path)
    except KeyError as exc:
        raise ValueError(f"array entry is missing: {path}") from exc
    try:
        with zf.open(info, "r") as stream:
            version = np.lib.format.read_magic(stream)
            if version == (1, 0):
                shape, _fortran_order, dtype = (
                    np.lib.format.read_array_header_1_0(
                        stream,
                        max_header_size=MAX_NPY_HEADER_BYTES,
                    )
                )
            elif version == (2, 0):
                shape, _fortran_order, dtype = (
                    np.lib.format.read_array_header_2_0(
                        stream,
                        max_header_size=MAX_NPY_HEADER_BYTES,
                    )
                )
            else:
                raise ValueError(
                    f"unsupported NPY format version {version!r}")
            data_offset = stream.tell()
    except (EOFError, OSError, ValueError) as exc:
        raise ValueError(f"invalid NPY entry: {path}") from exc

    dtype = np.dtype(dtype)
    if dtype.hasobject:
        raise ValueError(f"object arrays are not allowed: {path}")
    if dtype.itemsize < 1:
        raise ValueError(f"zero-size array dtypes are not allowed: {path}")
    if not isinstance(shape, tuple) or any(
            isinstance(size, bool)
            or not isinstance(size, (int, np.integer))
            or int(size) < 0
            for size in shape):
        raise ValueError(f"invalid NPY shape: {path}")
    element_count = math.prod(int(size) for size in shape)
    payload_bytes = element_count * int(dtype.itemsize)
    if payload_bytes > max_array_bytes:
        raise ValueError(f"array entry exceeds the memory budget: {path}")
    available_payload = int(info.file_size) - data_offset
    if payload_bytes > available_payload:
        raise ValueError(
            f"declared array payload exceeds archive entry size: {path}")
    return tuple(int(size) for size in shape), dtype, payload_bytes


def load_array_from_zip(
        zf: zipfile.ZipFile,
        path: str,
        mode: str | None = None,
        *,
        expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
    if path.endswith(".npy"):
        shape, dtype, _payload_bytes = _inspect_npy_entry(zf, path)
        if expected_shape is not None and shape != expected_shape:
            raise ValueError(
                f"array entry has unexpected dimensions: {path}")
        if mode == "RGBA" and (
                dtype != np.dtype(np.uint8)
                or len(shape) != 3
                or shape[2] != 4):
            raise ValueError(f"RGBA array entry must be uint8 (h, w, 4): {path}")
        if mode == "L" and (
                dtype not in (np.dtype(np.uint8), np.dtype(np.float32))
                or len(shape) != 2):
            raise ValueError(
                f"mask array entry must be uint8/float32 (h, w): {path}")
        with zf.open(path, "r") as stream:
            return np.load(stream, allow_pickle=False)
    data = zf.read(path)
    img = Image.open(io.BytesIO(data))
    if mode:
        img = img.convert(mode)
    return np.array(img, dtype=np.uint8)


def load_pil_from_zip(
        zf: zipfile.ZipFile,
        path: str,
        mode: str = "RGB") -> Image.Image:
    if path.endswith(".npy"):
        shape, dtype, _payload_bytes = _inspect_npy_entry(zf, path)
        if (
                dtype != np.dtype(np.uint8)
                or len(shape) not in (2, 3)
                or (len(shape) == 3 and shape[2] not in (1, 3, 4))):
            raise ValueError(
                f"source image array must be uint8 image-like data: {path}")
        with zf.open(path, "r") as stream:
            arr = np.load(stream, allow_pickle=False)
        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return Image.fromarray(arr).convert(mode)
    data = zf.read(path)
    return Image.open(io.BytesIO(data)).convert(mode)
