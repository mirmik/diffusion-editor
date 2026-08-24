"""Small geometry helpers shared by multi-view joint fitting experiments."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SimilarityTransform:
    """A proper 3D similarity transform: ``scale * R @ point + t``."""

    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def apply(self, points: np.ndarray) -> np.ndarray:
        values = np.asarray(points, dtype=np.float64)
        return self.scale * (values @ self.rotation.T) + self.translation


def estimate_similarity(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray | None = None,
) -> SimilarityTransform:
    """Return the weighted least-squares proper similarity source -> target."""

    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("source and target must have matching Nx3 shapes")
    if len(source) < 3:
        raise ValueError("at least three points are required")
    if weights is None:
        normalized = np.full(len(source), 1.0 / len(source))
    else:
        normalized = np.asarray(weights, dtype=np.float64)
        if normalized.shape != (len(source),):
            raise ValueError("weights must have shape N")
        if np.any(normalized < 0.0) or not np.any(normalized > 0.0):
            raise ValueError("weights must be non-negative with a positive sum")
        normalized = normalized / normalized.sum()

    source_center = np.sum(normalized[:, None] * source, axis=0)
    target_center = np.sum(normalized[:, None] * target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = (normalized[:, None] * source_zero).T @ target_zero
    left, singular, right_transpose = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(right_transpose.T @ left.T) < 0.0:
        correction[-1, -1] = -1.0
    rotation = right_transpose.T @ correction @ left.T
    denominator = np.sum(normalized * np.sum(source_zero * source_zero, axis=1))
    if denominator <= np.finfo(np.float64).eps:
        raise ValueError("source points have no usable extent")
    scale = float(np.sum(singular * np.diag(correction)) / denominator)
    translation = target_center - scale * (rotation @ source_center)
    return SimilarityTransform(scale, rotation, translation)


def project_scaled_orthographic(
    points: np.ndarray,
    rotation: np.ndarray,
    scale: float,
    translation: np.ndarray,
) -> np.ndarray:
    """Project world points into the normalized coordinates used by the report."""

    points = np.asarray(points, dtype=np.float64)
    rotation = np.asarray(rotation, dtype=np.float64)
    translation = np.asarray(translation, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape Nx3")
    if rotation.shape != (3, 3) or translation.shape != (2,):
        raise ValueError("rotation must be 3x3 and translation must have shape 2")
    return float(scale) * (points @ rotation[:2].T) + translation


def normalized_to_pixels(
    normalized: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """Convert factorization coordinates to image pixels."""

    normalized = np.asarray(normalized, dtype=np.float64)
    return np.stack((
        normalized[..., 0] * height + width * 0.5,
        height * 0.5 - normalized[..., 1] * height,
    ), axis=-1)
