"""Deterministic source-pixel checks used by native render smoke tests."""

from __future__ import annotations

import numpy as np


def source_contribution_error(
        source: np.ndarray,
        composite: np.ndarray | None,
        *,
        tolerance: int = 2) -> str | None:
    """Return a diagnostic when an opaque source is not recognizable."""
    if composite is None:
        return "no composite pixels were available"
    if composite.shape != source.shape:
        return (
            f"shape mismatch: source={source.shape}, "
            f"composite={composite.shape}")
    if composite.dtype != np.uint8:
        return f"unexpected composite dtype: {composite.dtype}"

    source_i16 = source.astype(np.int16)
    composite_i16 = composite.astype(np.int16)
    delta = np.abs(source_i16 - composite_i16)
    worst = int(delta.max(initial=0))
    mean = float(delta.mean())
    if worst > tolerance:
        return (
            f"pixel mismatch: max_delta={worst}, mean_delta={mean:.2f}; "
            f"source={rgba_signature(source)}; "
            f"composite={rgba_signature(composite)}")
    return None


def rgba_signature(image: np.ndarray | None) -> str:
    if image is None:
        return "unavailable"
    if image.ndim != 3 or image.shape[2] != 4 or image.size == 0:
        return f"shape={image.shape}, dtype={image.dtype}"
    height, width = image.shape[:2]
    points = (
        (0, 0),
        (height // 2, width // 2),
        (height - 1, width - 1),
    )
    probes = "/".join(
        ",".join(str(int(channel)) for channel in image[y, x])
        for y, x in points
    )
    means = ",".join(
        f"{value:.1f}" for value in image.mean(axis=(0, 1)))
    return (
        f"shape={image.shape}, range={int(image.min())}..{int(image.max())}, "
        f"mean={means}, probes={probes}")
