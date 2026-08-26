from __future__ import annotations

import math

import numpy as np
import pytest
from tcbase import Action, MouseButton

from diffusion_editor.app.native_reconstruction_viewport import (
    _allocate_resource_namespace,
    RECONSTRUCTION_SHADING_MODES,
    _FRAGMENT_SHADER,
    _OrbitCamera,
    _ViewportSurface,
    _SMOOTH_TEXTURED_VERTEX_SHADER,
    _SMOOTH_VERTEX_SHADER,
    _TEXTURED_FRAGMENT_SHADER,
    _decode_texture,
    _draw_constants,
    _mesh_resource_id,
    _should_fit_camera,
    _wireframe_indices,
)


def test_viewport_surface_routes_keys_to_focused_handler() -> None:
    calls = []
    surface = _ViewportSurface(_OrbitCamera(), lambda: None)
    surface.set_key_handler(
        lambda key, scancode, action, modifiers:
        calls.append((key, scancode, action, modifiers)) or True
    )

    assert surface.dispatch_key(262, 17, int(Action.PRESS), 0)
    assert calls == [(262, 17, int(Action.PRESS), 0)]

    surface.set_key_handler(None)
    assert not surface.dispatch_key(262, 17, int(Action.PRESS), 0)
    surface.close()
    assert not surface.dispatch_key(262, 17, int(Action.PRESS), 0)


def test_viewport_surface_pans_with_native_screen_space_gesture() -> None:
    camera = _OrbitCamera()
    surface = _ViewportSurface(camera, lambda: None)
    surface.resize(800, 600)
    target_before = tuple(camera._camera.target)

    assert surface.dispatch_pointer_button(
        400.0, 300.0,
        int(MouseButton.RIGHT), int(Action.PRESS),
        0, 1,
    )
    assert surface.dispatch_pointer_move(450.0, 330.0)
    assert tuple(camera._camera.target) != pytest.approx(target_before)
    assert surface.dispatch_pointer_button(
        450.0, 330.0,
        int(MouseButton.RIGHT), int(Action.RELEASE),
        0, 1,
    )
    surface.close()


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


def test_fit_keeps_native_zoom_clipping_matched_to_cloud_bounds() -> None:
    camera = _OrbitCamera()
    minimum = np.asarray((-1.0, -16.0, -2.0), dtype=np.float32)
    maximum = np.asarray((1.0, -7.0, 2.0), dtype=np.float32)
    expected_radius = float(np.linalg.norm(maximum - minimum)) * 0.65

    camera.fit(minimum, maximum)
    assert camera._camera.fitted_radius == pytest.approx(expected_radius)

    # Native OrbitCamera recalculates near/far during zoom.  The whole fitted
    # bounding sphere must remain between them after that recalculation.
    camera.zoom(1.0)
    assert camera._camera.near <= camera._camera.distance - expected_radius
    assert camera._camera.far >= camera._camera.distance + expected_radius


def test_camera_mvp_converts_native_mat44_to_numpy() -> None:
    camera = _OrbitCamera()

    mvp = camera.mvp(1280, 800)

    assert mvp.shape == (4, 4)
    assert mvp.dtype == np.float32
    assert np.isfinite(mvp).all()


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
