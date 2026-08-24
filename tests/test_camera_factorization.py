from __future__ import annotations

import numpy as np

from diffusion_editor.generation.camera_factorization import (
    camera_yaw_pitch_roll,
    factor_scaled_orthographic,
    reflect_depth,
)


def _rotation_y(degrees: float) -> np.ndarray:
    angle = np.radians(degrees)
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array((
        (cosine, 0.0, sine),
        (0.0, 1.0, 0.0),
        (-sine, 0.0, cosine),
    ))


def test_factorization_recovers_scaled_orthographic_tracks():
    rng = np.random.default_rng(42)
    points = rng.normal(size=(12, 3))
    points[:, 1] *= 1.8
    angles = (-50.0, -25.0, 0.0, 30.0, 55.0)
    rotations = np.stack([_rotation_y(angle) for angle in angles])
    scales = np.array((0.9, 1.1, 1.0, 1.2, 0.95))
    translations = rng.normal(scale=0.15, size=(len(angles), 2))
    observations = np.stack([
        scales[index] * (rotation[:2] @ points.T).T + translations[index]
        for index, rotation in enumerate(rotations)
    ])

    result = factor_scaled_orthographic(observations, reference_view=2)
    mirrored = reflect_depth(result)
    candidates = (result, mirrored)
    best = min(candidates, key=lambda item: np.sqrt(np.mean(
        (item.project() - observations) ** 2)))

    assert np.sqrt(np.mean((best.project() - observations) ** 2)) < 1.0e-7
    fitted = [camera_yaw_pitch_roll(camera.rotation)[0]
              for camera in best.cameras]
    assert max(
        abs(np.corrcoef(fitted, angles)[0, 1]),
        abs(np.corrcoef([-value for value in fitted], angles)[0, 1]),
    ) > 0.999


def test_factorization_validates_track_shape():
    with np.testing.assert_raises_regex(ValueError, "shape"):
        factor_scaled_orthographic(np.zeros((4, 8)))
