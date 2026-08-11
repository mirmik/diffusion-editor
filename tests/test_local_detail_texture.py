import numpy as np
import pytest

from diffusion_editor.generation.local_detail_texture import (
    blend_base_color,
    linear_to_srgb,
    local_blend_weights,
    sample_texture_bilinear,
    srgb_to_linear,
)


def test_local_blend_weights_have_smooth_ellipsoidal_collar():
    bounds = np.asarray(((-2.0, -1.0, -1.0), (2.0, 1.0, 1.0)))
    points = np.asarray(((0.0, 0.0, 0.0), (1.8, 0.0, 0.0), (3.0, 0.0, 0.0)))

    weights = local_blend_weights(
        points, bounds, inner_radius=0.5, outer_radius=1.0
    )

    assert weights[0] == pytest.approx(1.0)
    assert 0.0 < weights[1] < 1.0
    assert weights[2] == pytest.approx(0.0)


def test_texture_sampling_uses_bottom_left_uv_origin_and_wraps():
    image = np.asarray((
        ((255, 0, 0), (0, 255, 0)),
        ((0, 0, 255), (255, 255, 255)),
    ), dtype=np.uint8)

    sampled = sample_texture_bilinear(
        image, np.asarray(((0.0, 1.0), (1.5, 0.0)))
    )

    np.testing.assert_allclose(sampled[0], (255, 0, 0))
    np.testing.assert_allclose(sampled[1], (127.5, 127.5, 255.0))


def test_srgb_round_trip():
    values = np.asarray((0.0, 0.04, 0.5, 1.0), dtype=np.float32)
    np.testing.assert_allclose(
        linear_to_srgb(srgb_to_linear(values)), values, atol=1e-6
    )


def test_base_color_blend_uses_linear_light_and_preserves_endpoints():
    base = np.asarray(((0, 0, 0, 255),) * 3, dtype=np.uint8)
    local = np.asarray(((255, 255, 255, 255),) * 3, dtype=np.uint8)

    result = blend_base_color(base, local, np.asarray((0.0, 0.5, 1.0)))

    np.testing.assert_array_equal(result[0], base[0])
    np.testing.assert_array_equal(result[2], local[2])
    assert 180 <= int(result[1, 0]) <= 190
