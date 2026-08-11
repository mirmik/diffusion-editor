"""Texture-transfer helpers for local high-resolution 3D refinement."""

from __future__ import annotations

import numpy as np


def local_blend_weights(
        points: np.ndarray,
        target_bounds: np.ndarray,
        *,
        inner_radius: float = 0.72,
        outer_radius: float = 1.08,
) -> np.ndarray:
    """Return smooth local-source weights across an ellipsoidal collar."""
    values = np.asarray(points, dtype=np.float64)
    bounds = np.asarray(target_bounds, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if bounds.shape != (2, 3) or np.any(bounds[1] <= bounds[0]):
        raise ValueError("target_bounds must have positive shape [2, 3]")
    if not 0.0 <= inner_radius < outer_radius:
        raise ValueError("blend radii must satisfy 0 <= inner < outer")
    center = bounds.mean(axis=0)
    radii = (bounds[1] - bounds[0]) * 0.5
    distance = np.linalg.norm((values - center) / radii, axis=1)
    transition = np.clip(
        (distance - inner_radius) / (outer_radius - inner_radius),
        0.0,
        1.0,
    )
    smooth = transition * transition * (3.0 - 2.0 * transition)
    return (1.0 - smooth).astype(np.float32)


def sample_texture_bilinear(
        image: np.ndarray,
        uv: np.ndarray,
) -> np.ndarray:
    """Sample an image using glTF UV wrapping and bottom-left V origin."""
    pixels = np.asarray(image)
    coords = np.asarray(uv, dtype=np.float64)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("image must have shape [H, W, 3 or 4]")
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("uv must have shape [N, 2]")
    height, width = pixels.shape[:2]
    if height == 0 or width == 0:
        raise ValueError("image dimensions must be positive")

    x = np.mod(coords[:, 0] * (width - 1), width)
    y = np.mod((1.0 - coords[:, 1]) * (height - 1), height)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = (x0 + 1) % width
    y1 = (y0 + 1) % height
    wx = (x - x0).astype(np.float32)[:, None]
    wy = (y - y0).astype(np.float32)[:, None]
    floating = pixels.astype(np.float32)
    return (
        floating[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + floating[y0, x1] * wx * (1.0 - wy)
        + floating[y1, x0] * (1.0 - wx) * wy
        + floating[y1, x1] * wx * wy
    )


def srgb_to_linear(values: np.ndarray) -> np.ndarray:
    """Convert normalized sRGB components to linear light."""
    color = np.asarray(values, dtype=np.float32)
    return np.where(
        color <= 0.04045,
        color / 12.92,
        ((color + 0.055) / 1.055) ** 2.4,
    ).astype(np.float32)


def linear_to_srgb(values: np.ndarray) -> np.ndarray:
    """Convert normalized linear-light components to sRGB."""
    color = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    return np.where(
        color <= 0.0031308,
        color * 12.92,
        1.055 * color ** (1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def blend_base_color(
        base_rgba: np.ndarray,
        local_rgba: np.ndarray,
        local_weights: np.ndarray,
) -> np.ndarray:
    """Blend uint8/float RGBA sources in linear light and return uint8 RGBA."""
    base = np.asarray(base_rgba, dtype=np.float32)
    local = np.asarray(local_rgba, dtype=np.float32)
    weights = np.asarray(local_weights, dtype=np.float32).reshape(-1, 1)
    if base.shape != local.shape or base.ndim != 2 or base.shape[1] != 4:
        raise ValueError("base_rgba and local_rgba must share shape [N, 4]")
    if len(weights) != len(base):
        raise ValueError("local_weights must match the color arrays")
    if max(float(base.max(initial=0)), float(local.max(initial=0))) > 1.0:
        base /= 255.0
        local /= 255.0
    rgb = linear_to_srgb(
        srgb_to_linear(base[:, :3]) * (1.0 - weights)
        + srgb_to_linear(local[:, :3]) * weights
    )
    alpha = base[:, 3:] * (1.0 - weights) + local[:, 3:] * weights
    return np.rint(
        np.clip(np.concatenate((rgb, alpha), axis=1), 0.0, 1.0) * 255.0
    ).astype(np.uint8)
