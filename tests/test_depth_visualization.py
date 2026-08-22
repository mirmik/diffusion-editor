import numpy as np
import pytest

from diffusion_editor.generation.depth_visualization import (
    colorize_confidence,
    colorize_depth,
)


def test_colorize_depth_uses_percentiles_instead_of_extreme_outliers():
    depth = np.array(
        [-1000.0, *([100.25] * 50), *([150.75] * 50), 5000.0],
        dtype=np.float32,
    ).reshape((6, 17))
    original = depth.copy()

    preview = colorize_depth(depth)

    assert preview.low == pytest.approx(100.25)
    assert preview.high == pytest.approx(150.75)
    np.testing.assert_array_equal(preview.rgb[0, 0], (48, 18, 59))
    np.testing.assert_array_equal(preview.rgb[-1, -1], (122, 4, 3))
    assert preview.rgb.shape == (6, 17, 3)
    assert preview.rgb.dtype == np.uint8
    np.testing.assert_array_equal(depth, original)


def test_colorize_constant_depth_uses_palette_midpoint():
    preview = colorize_depth(np.full((3, 4), 127.125, dtype=np.float32))

    assert preview.low == preview.high == 127.125
    assert np.all(preview.rgb == (164, 252, 60))


def test_colorize_confidence_maps_low_to_cold_and_high_to_warm():
    confidence = np.array([1.0, 2.0, 4.0], dtype=np.float32)
    original = confidence.copy()

    preview = colorize_confidence(
        confidence, low_percentile=0.0, high_percentile=100.0)

    assert preview.low == 1.0
    assert preview.high == 4.0
    np.testing.assert_array_equal(preview.rgb[0], (48, 18, 59))
    np.testing.assert_array_equal(preview.rgb[-1], (122, 4, 3))
    np.testing.assert_array_equal(confidence, original)


def test_colorize_depth_excludes_transparent_background_from_contrast():
    depth = np.array([[0.0, 100.25, 150.75, 255.0]], dtype=np.float32)
    mask = np.array([[0, 255, 255, 0]], dtype=np.uint8)

    preview = colorize_depth(
        depth, mask=mask, low_percentile=0.0, high_percentile=100.0)

    assert preview.low == 100.25
    assert preview.high == 150.75
    np.testing.assert_array_equal(preview.rgb[0, 1], (48, 18, 59))
    np.testing.assert_array_equal(preview.rgb[0, 2], (122, 4, 3))


def test_direct_depth_preview_maps_near_low_values_to_warm_colors():
    depth = np.array([[2.0, 8.0]], dtype=np.float32)

    preview = colorize_depth(
        depth,
        near_is_high=False,
        low_percentile=0.0,
        high_percentile=100.0,
    )

    np.testing.assert_array_equal(preview.rgb[0, 0], (122, 4, 3))
    np.testing.assert_array_equal(preview.rgb[0, 1], (48, 18, 59))


@pytest.mark.parametrize("depth", (
    np.zeros((0, 4), dtype=np.float32),
    np.zeros((2, 3, 1), dtype=np.float32),
    np.array([[np.nan]], dtype=np.float32),
))
def test_colorize_depth_rejects_invalid_maps(depth):
    with pytest.raises(ValueError):
        colorize_depth(depth)
