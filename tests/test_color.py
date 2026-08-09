import numpy as np
import pytest

from diffusion_editor.color import (
    linear_rgba_to_srgb8,
    linear_to_srgb,
    srgb_to_linear,
)


def test_srgb_transfer_reference_values_and_roundtrip():
    encoded = np.array([0.0, 0.04045, 0.5, 1.0], dtype=np.float32)

    linear = srgb_to_linear(encoded)

    np.testing.assert_allclose(
        linear,
        [0.0, 0.0031308, 0.21404114, 1.0],
        atol=1e-7,
    )
    np.testing.assert_allclose(linear_to_srgb(linear), encoded, atol=1e-6)


def test_linear_rgba_encoding_does_not_transfer_alpha():
    linear = np.array([[[0.0, 0.0031308, 0.5, 0.25]]], dtype=np.float32)

    encoded = linear_rgba_to_srgb8(linear)

    np.testing.assert_array_equal(encoded, [[[0, 10, 188, 64]]])


def test_linear_rgba_encoding_rejects_non_rgba_input():
    with pytest.raises(ValueError, match="final dimension of 4"):
        linear_rgba_to_srgb8(np.zeros((2, 3), dtype=np.float32))
