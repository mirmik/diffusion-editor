"""Colour-transfer helpers for the editor's explicit pixel contract.

Document RGB channels are stored as sRGB-encoded uint8 values.  Compositors
operate on normalized linear-light floats.  Alpha and masks are linear values
and must never pass through either RGB transfer function.
"""

from __future__ import annotations

import numpy as np


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Decode normalized sRGB values to normalized linear-light float32."""
    values = np.clip(np.asarray(srgb, dtype=np.float32), 0.0, 1.0)
    return np.where(
        values <= 0.04045,
        values / 12.92,
        np.power((values + 0.055) / 1.055, 2.4),
    ).astype(np.float32, copy=False)


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Encode normalized linear-light values as normalized sRGB float32."""
    values = np.clip(np.asarray(linear, dtype=np.float32), 0.0, 1.0)
    return np.where(
        values <= 0.0031308,
        values * 12.92,
        1.055 * np.power(values, 1.0 / 2.4) - 0.055,
    ).astype(np.float32, copy=False)


def linear_rgba_to_srgb8(linear_rgba: np.ndarray) -> np.ndarray:
    """Encode straight normalized linear RGBA as straight sRGB RGBA8.

    Only RGB receives the transfer function. Alpha is copied numerically from
    the normalized linear input and quantized to uint8.
    """
    values = np.clip(np.asarray(linear_rgba, dtype=np.float32), 0.0, 1.0)
    if values.ndim < 1 or values.shape[-1] != 4:
        raise ValueError("linear RGBA buffer must have a final dimension of 4")
    encoded = np.empty_like(values)
    encoded[..., :3] = linear_to_srgb(values[..., :3])
    encoded[..., 3] = values[..., 3]
    return np.rint(encoded * 255.0).astype(np.uint8)
