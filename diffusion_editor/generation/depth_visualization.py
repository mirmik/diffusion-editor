"""Human-readable preview rendering for generated depth maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEPTH_PREVIEW_LOW_PERCENTILE = 2.0
DEPTH_PREVIEW_HIGH_PERCENTILE = 98.0

# Compact samples of the Turbo palette, from cold/distant to warm/near.
_TURBO_ANCHORS = np.array([
    (48, 18, 59),
    (70, 107, 227),
    (40, 188, 235),
    (31, 234, 179),
    (164, 252, 60),
    (238, 208, 58),
    (251, 126, 33),
    (188, 44, 35),
    (122, 4, 3),
], dtype=np.float32)


@dataclass(frozen=True)
class DepthPreview:
    rgb: np.ndarray
    low: float
    high: float


def _turbo_rgb(normalized: np.ndarray) -> np.ndarray:
    anchor_positions = np.linspace(
        0.0, 1.0, len(_TURBO_ANCHORS), dtype=np.float32)
    rgb = np.stack([
        np.interp(normalized, anchor_positions, _TURBO_ANCHORS[:, channel])
        for channel in range(3)
    ], axis=-1)
    return np.ascontiguousarray(
        np.clip(np.rint(rgb), 0, 255).astype(np.uint8))


def colorize_confidence(
        confidence: np.ndarray,
        *,
        low_percentile: float = DEPTH_PREVIEW_LOW_PERCENTILE,
        high_percentile: float = DEPTH_PREVIEW_HIGH_PERCENTILE,
) -> DepthPreview:
    """Color a confidence vector/map from cold-low to warm-high.

    Percentiles affect display contrast only.  The returned range is retained
    so the UI can state exactly which raw scores map to the palette endpoints.
    """

    values = np.asarray(confidence)
    if values.ndim not in (1, 2) or values.size == 0:
        raise ValueError("confidence must be a non-empty 1D or 2D array")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("confidence must contain numeric values")
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("confidence must contain only finite values")
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("confidence percentiles are invalid")

    low, high = np.percentile(values, (low_percentile, high_percentile))
    low = float(low)
    high = float(high)
    if high - low <= np.finfo(np.float32).eps:
        normalized = np.full(values.shape, 0.5, dtype=np.float32)
    else:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    return DepthPreview(rgb=_turbo_rgb(normalized), low=low, high=high)


def colorize_depth(
        depth: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        near_is_high: bool = True,
        low_percentile: float = DEPTH_PREVIEW_LOW_PERCENTILE,
        high_percentile: float = DEPTH_PREVIEW_HIGH_PERCENTILE,
) -> DepthPreview:
    """Apply robust contrast and a cold-far/warm-near color palette."""

    values = np.asarray(depth)
    if values.ndim != 2 or values.size == 0:
        raise ValueError("depth map must be a non-empty 2D array")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError("depth map must contain numeric values")
    values = values.astype(np.float32, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("depth map must contain only finite values")
    if not 0.0 <= low_percentile < high_percentile <= 100.0:
        raise ValueError("depth preview percentiles are invalid")

    valid_values = values.reshape(-1)
    if mask is not None:
        mask_values = np.asarray(mask)
        if mask_values.shape != values.shape:
            raise ValueError("depth preview mask shape does not match depth map")
        try:
            mask_values = mask_values.astype(np.float32, copy=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("depth preview mask is not numeric") from exc
        if not np.isfinite(mask_values).all():
            raise ValueError("depth preview mask is not finite")
        valid = mask_values > 0.0
        if not valid.any():
            raise ValueError("depth preview mask is empty")
        valid_values = values[valid]

    low, high = np.percentile(
        valid_values, (low_percentile, high_percentile))
    low = float(low)
    high = float(high)
    if high - low <= np.finfo(np.float32).eps:
        normalized = np.full(values.shape, 0.5, dtype=np.float32)
    else:
        normalized = np.clip((values - low) / (high - low), 0.0, 1.0)
    if not near_is_high:
        normalized = 1.0 - normalized

    return DepthPreview(
        rgb=_turbo_rgb(normalized),
        low=low,
        high=high,
    )
