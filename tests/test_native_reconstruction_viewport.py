from __future__ import annotations

import math

import numpy as np
import pytest

from diffusion_editor.app.native_reconstruction_viewport import (
    _allocate_resource_namespace,
    RECONSTRUCTION_SHADING_MODES,
    _FRAGMENT_SHADER,
    _OrbitCamera,
    _SMOOTH_TEXTURED_VERTEX_SHADER,
    _SMOOTH_VERTEX_SHADER,
    _TEXTURED_FRAGMENT_SHADER,
    _decode_texture,
    _draw_constants,
    _mesh_resource_id,
    _should_fit_camera,
    _wireframe_indices,
)


def test_reconstruction_viewports_get_distinct_mesh_resource_namespaces() -> None:
    first = _allocate_resource_namespace("primary")
    second = _allocate_resource_namespace("refine")

    assert first != second
    assert _mesh_resource_id(first, 0) != _mesh_resource_id(second, 0)


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


def test_comparison_loads_fit_only_the_first_model_by_default() -> None:
    assert _should_fit_camera(False, None)
    assert not _should_fit_camera(True, None)
    assert _should_fit_camera(True, True)
    assert not _should_fit_camera(False, False)


def test_draw_constants_include_configurable_light_direction() -> None:
    packed = _draw_constants(
        np.eye(4, dtype=np.float32),
        (0.1, 0.2, 0.3, 1.0),
        (0.0, 1.0, 0.25),
    ).view(np.float32)

    assert packed.shape == (24,)
    assert packed[16:20] == pytest.approx((0.1, 0.2, 0.3, 1.0))
    assert packed[20:24] == pytest.approx((0.0, 1.0, 0.25, 0.0))


def test_draw_constants_encode_display_mode_with_the_light() -> None:
    packed = _draw_constants(
        np.eye(4, dtype=np.float32),
        (0.1, 0.2, 0.3, 1.0),
        (0.0, 1.0, 0.25),
        3.0,
    ).view(np.float32)

    assert packed[20:24] == pytest.approx((0.0, 1.0, 0.25, 3.0))


def test_shading_shaders_cover_flat_smooth_unlit_and_normals() -> None:
    assert RECONSTRUCTION_SHADING_MODES == (
        "flat", "smooth", "unlit", "normals", "wireframe"
    )
    assert "layout(location=1) in vec3 a_normal" in _SMOOTH_VERTEX_SHADER
    assert "layout(location=1) in vec3 a_normal" in (
        _SMOOTH_TEXTURED_VERTEX_SHADER
    )
    assert "display_mode == 2" in _FRAGMENT_SHADER
    assert "display_mode == 3" in _FRAGMENT_SHADER
    assert "display_mode == 2" in _TEXTURED_FRAGMENT_SHADER
    assert "display_mode == 3" in _TEXTURED_FRAGMENT_SHADER


def test_wireframe_indices_expand_every_triangle_edge() -> None:
    indices = _wireframe_indices(np.asarray([
        (0, 1, 2),
        (2, 1, 3),
    ], dtype=np.uint32))

    assert indices.dtype == np.uint32
    assert indices.flags.c_contiguous
    assert indices.tolist() == [
        0, 1, 1, 2, 2, 0,
        2, 1, 1, 3, 3, 2,
    ]


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
