import numpy as np
import pytest

from diffusion_editor.generation.joint_fitting import (
    estimate_similarity,
    normalized_to_pixels,
    project_scaled_orthographic,
)


def test_estimate_similarity_recovers_proper_transform():
    source = np.array([
        [-1.0, 0.0, 0.5],
        [1.0, 0.0, -0.5],
        [0.0, 2.0, 0.25],
        [0.4, -1.0, 1.5],
    ])
    angle = np.deg2rad(31.0)
    rotation = np.array([
        [np.cos(angle), 0.0, np.sin(angle)],
        [0.0, 1.0, 0.0],
        [-np.sin(angle), 0.0, np.cos(angle)],
    ])
    target = 2.75 * (source @ rotation.T) + np.array([0.3, -1.2, 4.1])

    fitted = estimate_similarity(source, target, np.array([1.0, 2.0, 1.0, 3.0]))

    assert fitted.scale == pytest.approx(2.75)
    assert np.linalg.det(fitted.rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(fitted.apply(source), target, atol=1.0e-10)


def test_scaled_orthographic_projection_and_pixel_conversion():
    points = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]])
    normalized = project_scaled_orthographic(
        points, np.eye(3), 0.25, np.array([0.1, -0.2]))

    np.testing.assert_allclose(
        normalized, [[0.35, 0.3], [-0.15, -0.075]])
    np.testing.assert_allclose(
        normalized_to_pixels(normalized, 1000, 800),
        [[780.0, 160.0], [380.0, 460.0]],
    )
