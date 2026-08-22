import numpy as np
import pytest

from diffusion_editor.generation.depth_point_cloud import (
    project_depth_point_cloud,
)
from diffusion_editor.generation.types import DepthValueKind


def _intrinsics(width: int, height: int, focal: float = 2.0) -> np.ndarray:
    return np.array([
        [focal, 0.0, width * 0.5],
        [0.0, focal, height * 0.5],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def test_project_depth_point_cloud_unprojects_known_plane_exactly():
    image = np.full((2, 3, 3), 128, dtype=np.uint8)
    depth = np.full((2, 3), 2.0, dtype=np.float32)

    cloud = project_depth_point_cloud(
        image, depth, intrinsics=_intrinsics(3, 2))

    assert cloud.point_count == 6
    assert cloud.approximate_camera is False
    assert cloud.positions.dtype == np.float64
    np.testing.assert_allclose(cloud.positions[0], (-1.5, -2.0, 1.0))
    np.testing.assert_allclose(
        cloud.positions[1] - cloud.positions[0],
        (1.0, 0.0, 0.0),
    )
    np.testing.assert_allclose(
        cloud.positions[3] - cloud.positions[0],
        (0.0, 0.0, -1.0),
    )
    np.testing.assert_allclose(cloud.positions[:, 1], -2.0)


def test_project_depth_point_cloud_preserves_raw_direct_depth_ratios():
    image = np.full((1, 2, 3), 255, dtype=np.uint8)
    depth = np.array([[2.0, 4.0]], dtype=np.float32)
    intrinsics = np.eye(3, dtype=np.float32)

    cloud = project_depth_point_cloud(image, depth, intrinsics=intrinsics)

    np.testing.assert_allclose(cloud.positions[0], (0.0, -2.0, 0.0))
    np.testing.assert_allclose(cloud.positions[1], (4.0, -4.0, 0.0))
    assert cloud.depth_min == pytest.approx(2.0)
    assert cloud.depth_max == pytest.approx(4.0)


def test_project_depth_point_cloud_preserves_image_orientation_and_colors():
    image = np.zeros((3, 3, 4), dtype=np.uint8)
    image[:, :, 3] = 255
    image[0, 0, :3] = (255, 0, 0)
    image[2, 2, :3] = (0, 0, 255)
    depth = np.full((3, 3), 2.0, dtype=np.float32)

    cloud = project_depth_point_cloud(
        image, depth, intrinsics=_intrinsics(3, 3))

    assert cloud.positions[0, 0] < cloud.positions[-1, 0]
    assert cloud.positions[0, 2] > cloud.positions[-1, 2]
    np.testing.assert_array_equal(cloud.colors[0], (1.0, 0.0, 0.0))
    np.testing.assert_array_equal(cloud.colors[-1], (0.0, 0.0, 1.0))


def test_project_depth_point_cloud_uses_only_masked_subject_statistics():
    image = np.full((2, 4, 3), 128, dtype=np.uint8)
    depth = np.array([
        [1.0, 10.0, 15.0, 100.0],
        [1.0, 10.0, 15.0, 100.0],
    ], dtype=np.float32)
    mask = np.array([[0, 255, 255, 0], [0, 255, 255, 0]], dtype=np.uint8)

    cloud = project_depth_point_cloud(
        image, depth, mask=mask, intrinsics=_intrinsics(4, 2))

    assert cloud.point_count == 4
    assert cloud.depth_min == pytest.approx(10.0)
    assert cloud.depth_max == pytest.approx(15.0)


def test_project_depth_point_cloud_discards_low_alpha_fringe_at_half():
    image = np.zeros((2, 4, 3), dtype=np.uint8)
    image[:, 2, 0] = 255
    image[:, 3, 1] = 255
    depth = np.array([
        [1.0, 25.0, 10.0, 15.0],
        [1.0, 25.0, 10.0, 15.0],
    ], dtype=np.float32)
    mask = np.array([
        [1, 127, 128, 255],
        [1, 127, 128, 255],
    ], dtype=np.uint8)

    cloud = project_depth_point_cloud(
        image, depth, mask=mask, intrinsics=_intrinsics(4, 2))

    assert cloud.point_count == 4
    assert cloud.depth_min == pytest.approx(10.0)
    assert cloud.depth_max == pytest.approx(15.0)
    np.testing.assert_array_equal(
        cloud.colors,
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32),
    )


def test_project_depth_point_cloud_keeps_confidence_aligned_with_points():
    image = np.full((2, 3, 3), 128, dtype=np.uint8)
    depth = np.full((2, 3), 2.0, dtype=np.float32)
    confidence = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ], dtype=np.float32)
    mask = np.array([[255, 0, 255], [0, 255, 0]], dtype=np.uint8)

    cloud = project_depth_point_cloud(
        image,
        depth,
        mask=mask,
        confidence=confidence,
        intrinsics=_intrinsics(3, 2),
    )

    assert cloud.confidence is not None
    assert cloud.confidence.dtype == np.float32
    np.testing.assert_array_equal(cloud.confidence, (1.0, 3.0, 5.0))


def test_project_depth_point_cloud_rejects_misaligned_confidence():
    with pytest.raises(ValueError, match="confidence shape"):
        project_depth_point_cloud(
            np.ones((2, 3, 3), dtype=np.uint8),
            np.ones((2, 3), dtype=np.float32),
            confidence=np.ones((3, 2), dtype=np.float32),
            intrinsics=_intrinsics(3, 2),
        )


def test_inverse_relative_fallback_preserves_reciprocal_arbitrary_scale():
    image = np.full((1, 2, 3), 128, dtype=np.uint8)
    inverse_depth = np.array([[1.0, 2.0]], dtype=np.float32)

    cloud = project_depth_point_cloud(
        image,
        inverse_depth,
        value_kind=DepthValueKind.INVERSE_RELATIVE,
        fallback_fov_y_degrees=60.0,
    )

    assert cloud.approximate_camera is True
    reconstructed_depth = -cloud.positions[:, 1]
    np.testing.assert_allclose(reconstructed_depth, (1.0, 0.5))


def test_project_depth_point_cloud_caps_points_with_deterministic_stride():
    image = np.full((100, 120, 3), 128, dtype=np.uint8)
    depth = np.full((100, 120), 2.0, dtype=np.float32)
    intrinsics = _intrinsics(120, 100, focal=100.0)

    first = project_depth_point_cloud(
        image, depth, intrinsics=intrinsics, max_points=1000)
    second = project_depth_point_cloud(
        image, depth, intrinsics=intrinsics, max_points=1000)

    assert first.point_count <= 1000
    assert first.stride > 1
    np.testing.assert_array_equal(first.positions, second.positions)


def test_project_depth_point_cloud_requires_calibration_for_direct_depth():
    with pytest.raises(ValueError, match="requires camera intrinsics"):
        project_depth_point_cloud(
            np.ones((3, 4, 3), dtype=np.uint8),
            np.ones((3, 4), dtype=np.float32),
        )


def test_project_depth_point_cloud_rejects_empty_foreground():
    with pytest.raises(ValueError, match="no foreground"):
        project_depth_point_cloud(
            np.zeros((3, 4, 4), dtype=np.uint8),
            np.ones((3, 4), dtype=np.float32),
            intrinsics=_intrinsics(4, 3),
        )
