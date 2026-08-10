from __future__ import annotations

import math

import numpy as np
import pytest

from diffusion_editor.app.native_reconstruction_viewport import (
    _OrbitCamera,
    _decode_texture,
    _draw_constants,
)


def test_fit_resets_camera_to_pixal3d_front() -> None:
    camera = _OrbitCamera()
    camera.orbit(30.0, -20.0)

    camera.fit(
        np.asarray((-1.0, -0.5, 0.0), dtype=np.float32),
        np.asarray((1.0, 0.5, 2.0), dtype=np.float32),
    )

    assert camera._camera.azimuth == pytest.approx(math.pi)
    assert camera._camera.elevation == pytest.approx(math.radians(12.0))
    assert camera.direction_from_target() == pytest.approx((
        0.0,
        math.cos(math.radians(12.0)),
        math.sin(math.radians(12.0)),
    ), abs=1e-6)


def test_draw_constants_include_configurable_light_direction() -> None:
    packed = _draw_constants(
        np.eye(4, dtype=np.float32),
        (0.1, 0.2, 0.3, 1.0),
        (0.0, 1.0, 0.25),
    ).view(np.float32)

    assert packed.shape == (24,)
    assert packed[16:20] == pytest.approx((0.1, 0.2, 0.3, 1.0))
    assert packed[20:24] == pytest.approx((0.0, 1.0, 0.25, 0.0))


def test_decode_texture_preserves_top_to_bottom_pixel_order() -> None:
    from io import BytesIO
    from PIL import Image

    source = Image.new("RGB", (1, 2))
    source.putdata([(255, 0, 0), (0, 0, 255)])
    encoded = BytesIO()
    source.save(encoded, format="PNG")

    width, height, pixels = _decode_texture(encoded.getvalue())

    assert (width, height) == (1, 2)
    assert pixels.flags.writeable
    assert pixels.reshape(2, 1, 4).tolist() == [
        [[255, 0, 0, 255]],
        [[0, 0, 255, 255]],
    ]
