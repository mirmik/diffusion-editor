from __future__ import annotations

import math

import numpy as np
import pytest
from tcbase import Action, MouseButton

from diffusion_editor.app.native_reconstruction_viewport import (
    NativeReconstructionViewport,
    _MeshRenderItem,
    _allocate_resource_namespace,
    _build_index_subset_mesh,
    _build_smooth_mesh,
    _build_weighted_mask_mesh,
    _faces_under_screen_brush,
    _select_cube_triangles,
    RECONSTRUCTION_SHADING_MODES,
    _FRAGMENT_SHADER,
    _OrbitCamera,
    _ViewportSurface,
    _SMOOTH_TEXTURED_VERTEX_SHADER,
    _SMOOTH_TEXTURED_FRAGMENT_SHADER,
    _SMOOTH_FRAGMENT_SHADER,
    _SMOOTH_VERTEX_SHADER,
    _TEXTURED_FRAGMENT_SHADER,
    _decode_texture,
    _compute_vertex_normals,
    _draw_constants,
    _mesh_resource_id,
    _nearest_ray_triangle_hit,
    _point_segment_distance,
    _ray_plane_intersection,
    _REFINE_GIZMO_AXES,
    _REFINE_SIZE_DIRECTION,
    _should_fit_camera,
    _wireframe_indices,
    _vertex_weights_under_screen_brush,
)


def test_screen_brush_selects_coincident_front_and_back_faces() -> None:
    positions = np.asarray([
        (-0.4, -0.4, -0.5), (0.4, -0.4, -0.5), (0.0, 0.4, -0.5),
        (-0.4, -0.4, 0.5), (0.4, -0.4, 0.5), (0.0, 0.4, 0.5),
        (0.8, 0.8, 0.0), (0.9, 0.8, 0.0), (0.85, 0.9, 0.0),
    ], dtype=np.float32)
    triangles = np.asarray([
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
    ], dtype=np.uint32)

    selected = _faces_under_screen_brush(
        positions,
        triangles,
        np.eye(4, dtype=np.float32),
        200,
        200,
        (100.0, 100.0),
        12.0,
    )

    assert selected.tolist() == [0, 1]


def test_screen_brush_y_matches_native_orbit_camera_projection() -> None:
    camera = _OrbitCamera(front_azimuth=0.0)
    width, height = 800, 600
    center = np.asarray((0.0, 0.0, 0.5), dtype=np.float64)
    screen = camera.project_world_to_screen(center, width, height)
    positions = np.asarray([
        (-0.08, 0.0, 0.45),
        (0.08, 0.0, 0.45),
        (0.0, 0.0, 0.58),
    ], dtype=np.float32)
    triangles = np.asarray(((0, 1, 2),), dtype=np.uint32)

    selected = _faces_under_screen_brush(
        positions, triangles, camera.mvp(width, height),
        width, height, tuple(screen), 18.0,
    )
    vertically_flipped = _faces_under_screen_brush(
        positions, triangles, camera.mvp(width, height),
        width, height, (float(screen[0]), height - float(screen[1])), 18.0,
    )

    assert selected.tolist() == [0]
    assert vertically_flipped.tolist() == []


def test_vertex_brush_has_smooth_center_to_edge_falloff() -> None:
    positions = np.asarray([
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.9, 0.0, 0.0),
        (0.0, 0.0, 0.5),
    ], dtype=np.float32)

    indexes, weights = _vertex_weights_under_screen_brush(
        positions,
        np.eye(4, dtype=np.float32),
        200,
        200,
        (100.0, 100.0),
        100.0,
    )

    assert indexes.tolist() == [0, 1, 2, 3]
    assert weights == pytest.approx((1.0, 0.5, 0.028, 1.0), abs=1e-3)


def test_legacy_face_mask_promotes_its_vertices_to_full_weight() -> None:
    viewport = object.__new__(NativeReconstructionViewport)
    item = _MeshRenderItem(
        mesh=None,
        positions=np.zeros((4, 3), dtype=np.float32),
        triangles=np.asarray(((0, 1, 2), (1, 3, 2)), dtype=np.uint32),
        source_face_indices=np.asarray((5, 9), dtype=np.uint32),
    )
    viewport._mesh_items = [item]
    viewport._rebuild_mask_mesh = lambda _item: None
    viewport._closed = False
    viewport._dirty = False
    viewport._request_repaint = lambda: None

    viewport.set_refine_vertex_mask((), legacy_mesh_faces=((9,),))

    assert item.mask_vertex_weights == pytest.approx((0.0, 1.0, 1.0, 1.0))


def test_cube_subset_arrays_preserve_fringe_and_remap_vertex_weights() -> None:
    viewport = object.__new__(NativeReconstructionViewport)
    viewport._mesh_items = [_MeshRenderItem(
        mesh=None,
        positions=np.asarray((
            (-0.2, 0.0, 0.0),
            (0.8, 0.0, 0.0),
            (0.0, 0.8, 0.0),
            (2.0, 2.0, 2.0),
        ), dtype=np.float32),
        triangles=np.asarray(((0, 1, 2),), dtype=np.uint32),
        source_mesh_index=0,
    )]

    vertices, faces, weights = viewport.cube_mesh_subset_arrays(
        (0.0, 0.0, 0.0),
        1.0,
        mesh_vertex_weights=(((0, 0.25), (2, 0.75)),),
    )

    np.testing.assert_allclose(vertices, [
        [-0.2, 0.0, 0.0],
        [0.8, 0.0, 0.0],
        [0.0, 0.8, 0.0],
    ])
    assert faces.tolist() == [[0, 1, 2]]
    assert weights.tolist() == pytest.approx([0.25, 0.0, 0.75])


def test_vertex_brush_paints_maximum_and_softly_erases_existing_weights() -> None:
    class Camera:
        @staticmethod
        def mvp(_width, _height):
            return np.eye(4, dtype=np.float32)

    positions = np.asarray([
        (0.0, 0.0, 0.0),
        (0.5, 0.0, 0.0),
        (0.9, 0.0, 0.0),
    ], dtype=np.float32)
    item = _MeshRenderItem(
        mesh=None,
        positions=positions,
        triangles=np.asarray(((0, 1, 2),), dtype=np.uint32),
        mask_vertex_weights=np.zeros(3, dtype=np.float32),
    )
    viewport = object.__new__(NativeReconstructionViewport)
    viewport._camera = Camera()
    viewport.surface = type("Surface", (), {"size": (200, 200)})()
    viewport._mesh_items = [item]
    viewport._model_transform = np.eye(4, dtype=np.float32)
    viewport._refine_mask_brush_radius = 100.0
    viewport._refine_mask_paint = True
    viewport._rebuild_mask_mesh = lambda _item: None
    viewport._closed = False
    viewport._dirty = False
    viewport._request_repaint = lambda: None

    viewport._apply_refine_mask_brush(100.0, 100.0)
    assert item.mask_vertex_weights == pytest.approx(
        (1.0, 0.5, 0.028), abs=1e-3
    )

    item.mask_vertex_weights[:] = 1.0
    viewport._refine_mask_paint = False
    viewport._apply_refine_mask_brush(100.0, 100.0)
    assert item.mask_vertex_weights == pytest.approx(
        (0.0, 0.5, 0.972), abs=1e-3
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


def test_viewport_surface_gives_pick_handler_first_left_click() -> None:
    camera = _OrbitCamera()
    surface = _ViewportSurface(camera, lambda: None)
    calls = []
    surface.set_pointer_press_handler(
        lambda x, y, button, modifiers:
        calls.append((x, y, button, modifiers)) or True
    )

    assert surface.dispatch_pointer_button(
        25.0, 40.0, int(MouseButton.LEFT), int(Action.PRESS), 0, 1
    )
    assert surface.dispatch_pointer_move(40.0, 50.0)
    assert calls == [(25.0, 40.0, int(MouseButton.LEFT), 0)]
    surface.close()


def test_viewport_surface_routes_active_gizmo_drag_before_orbit() -> None:
    camera = _OrbitCamera()
    surface = _ViewportSurface(camera, lambda: None)
    moves = []
    surface.set_pointer_move_handler(
        lambda x, y: moves.append((x, y)) or True
    )

    assert surface.dispatch_pointer_move(30.0, 45.0)
    assert moves == [(30.0, 45.0)]
    surface.close()


def test_refine_gizmo_axes_represent_gltf_coordinates_in_z_up_viewport() -> None:
    assert _REFINE_GIZMO_AXES["x"][0] == pytest.approx((1.0, 0.0, 0.0))
    assert _REFINE_GIZMO_AXES["y"][0] == pytest.approx((0.0, 0.0, 1.0))
    assert _REFINE_GIZMO_AXES["z"][0] == pytest.approx((0.0, -1.0, 0.0))
    assert _REFINE_SIZE_DIRECTION == pytest.approx(
        np.asarray((1.0, -1.0, 1.0)) / math.sqrt(3.0)
    )


def test_gizmo_screen_projection_and_segment_pick_distance() -> None:
    camera = _OrbitCamera(front_azimuth=0.0)
    camera.fit(
        np.asarray((-1.0, -1.0, -1.0)),
        np.asarray((1.0, 1.0, 1.0)),
    )
    center = camera.project_world_to_screen(
        np.asarray((0.0, 0.0, 0.0)), 800, 600
    )
    end = camera.project_world_to_screen(
        np.asarray((1.0, 0.0, 0.0)), 800, 600
    )
    elevated = camera.project_world_to_screen(
        np.asarray((0.0, 0.0, 1.0)), 800, 600
    )

    assert center == pytest.approx((400.0, 300.0))
    assert end[0] > center[0]
    # Regression: the Vulkan projection already owns screen-Y orientation.
    # A second manual flip made visible handles unpickable.
    assert elevated[1] < center[1]
    midpoint = (center + end) * 0.5
    distance, parameter = _point_segment_distance(
        midpoint + np.asarray((0.0, 6.0)), center, end
    )
    assert distance == pytest.approx(6.0)
    assert parameter == pytest.approx(0.5)


def test_ray_plane_intersection_supports_gizmo_constraint() -> None:
    hit = _ray_plane_intersection(
        np.asarray((0.0, 0.0, 5.0)),
        np.asarray((0.0, 0.0, -1.0)),
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
    )

    assert hit == pytest.approx((0.0, 0.0, 0.0))


def test_refine_gizmo_x_arrow_drag_publishes_viewport_center() -> None:
    class Ray:
        def __init__(self, origin, direction):
            self.origin = origin
            self.direction = direction

    class Camera:
        def mvp(self, width, height):
            del width, height
            return np.eye(4, dtype=np.float32)

        def direction_from_target(self):
            return np.asarray((0.0, 0.0, 1.0), dtype=np.float32)

        def project_world_to_screen(self, point, width, height):
            return np.asarray((
                (float(point[0]) * 0.5 + 0.5) * width,
                (0.5 - float(point[1]) * 0.5) * height,
            ))

        def screen_ray(self, position, width, height):
            x = position[0] / width * 2.0 - 1.0
            y = 1.0 - position[1] / height * 2.0
            return Ray((x, y, 5.0), (0.0, 0.0, -1.0))

    viewport = object.__new__(NativeReconstructionViewport)
    viewport._camera = Camera()
    viewport.surface = type("Surface", (), {"size": (800, 600)})()
    viewport._refine_cube_center = (0.0, 0.0, 0.0)
    viewport._refine_cube_side = 1.0
    viewport._refine_cube_confirmed = True
    viewport._model_bounds_min = np.asarray((-1.0, -1.0, -1.0))
    viewport._model_bounds_max = np.asarray((1.0, 1.0, 1.0))
    viewport._refine_cube_edit_enabled = True
    viewport._refine_gizmo_drag = None
    viewport._closed = False
    viewport._dirty = False
    viewport._request_repaint = lambda: None
    changes = []
    viewport._refine_cube_edit_handler = (
        lambda center, side: changes.append((center, side))
    )

    assert viewport._begin_refine_gizmo_drag(600.0, 300.0)
    assert viewport._refine_gizmo_drag.kind == "x"
    assert viewport._handle_gizmo_pointer_move(640.0, 300.0)

    assert changes[-1][0] == pytest.approx((0.1, 0.0, 0.0))
    assert changes[-1][1] == pytest.approx(1.0)
    assert not viewport._refine_cube_confirmed


def test_nearest_ray_triangle_hit_selects_frontmost_triangle() -> None:
    positions = np.asarray([
        (-1.0, -1.0, 1.0), (1.0, -1.0, 1.0), (0.0, 1.0, 1.0),
        (-1.0, -1.0, 2.0), (1.0, -1.0, 2.0), (0.0, 1.0, 2.0),
    ], dtype=np.float32)
    triangles = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.uint32)

    hit = _nearest_ray_triangle_hit(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((0.0, 0.0, 1.0)),
        positions,
        triangles,
    )

    np.testing.assert_allclose(hit, (0.0, 0.0, 1.0))


def test_refine_cube_selects_original_faces_without_splitting_vertices() -> None:
    positions = np.asarray([
        (-2.0, -2.0, 0.0),
        (2.0, -2.0, 0.0),
        (0.0, 2.0, 0.0),
        (3.0, 3.0, 3.0),
        (4.0, 3.0, 3.0),
        (3.0, 4.0, 3.0),
    ], dtype=np.float32)
    triangles = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.uint32)

    selected_faces = _select_cube_triangles(
        positions, triangles, (0.0, 0.0, 0.0), 1.0
    )

    np.testing.assert_array_equal(selected_faces, triangles[:1])


def test_index_subset_preserves_source_vertex_buffer_and_layout() -> None:
    from tmesh import TcMesh, TcVertexLayout

    layout = TcVertexLayout.pos_normal_uv()
    vertices = np.ascontiguousarray([
        (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0),
        (1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0),
    ], dtype=np.float32)
    source = TcMesh.from_interleaved(
        vertices,
        len(vertices),
        np.ascontiguousarray((0, 1, 2, 1, 3, 2), dtype=np.uint32),
        layout,
        uuid="refine-subset-source",
    )

    subset = _build_index_subset_mesh(
        source, np.asarray(((1, 3, 2),), dtype=np.uint32)
    )

    assert subset.vertex_count == source.vertex_count
    np.testing.assert_array_equal(
        subset.get_vertices_buffer(), source.get_vertices_buffer()
    )
    np.testing.assert_array_equal(subset.get_indices_buffer(), (1, 3, 2))
    assert subset.mesh.layout.find("normal") == source.mesh.layout.find("normal")
    assert subset.mesh.layout.find("uv") == source.mesh.layout.find("uv")
    np.testing.assert_allclose(subset.vertex_normals, source.vertex_normals)


def test_weighted_mask_mesh_uploads_interpolated_vertex_attribute() -> None:
    from tmesh import TcAttribType, TcMesh, TcVertexLayout

    positions = np.asarray([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ], dtype=np.float32)
    normals = np.tile((0.0, 0.0, 1.0), (3, 1)).astype(np.float32)
    layout = TcVertexLayout()
    assert layout.add("position", 3, TcAttribType.FLOAT32, 0)
    assert layout.add("normal", 3, TcAttribType.FLOAT32, 1)
    source = TcMesh.from_interleaved(
        np.ascontiguousarray(np.concatenate((positions, normals), axis=1)),
        3,
        np.asarray((0, 1, 2), dtype=np.uint32),
        layout,
        uuid="weighted-mask-source",
    )
    weights = np.asarray((1.0, 0.5, 0.0), dtype=np.float32)

    overlay = _build_weighted_mask_mesh(
        source, positions, np.asarray(((0, 1, 2),), dtype=np.uint32), weights
    )

    assert overlay.mesh.layout.find("mask_weight") is not None
    uploaded = np.asarray(overlay.get_vertices_buffer()).reshape(3, 7)
    np.testing.assert_allclose(uploaded[:, 6], weights)


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


def test_fit_can_reset_camera_to_opposite_trellis_front() -> None:
    camera = _OrbitCamera(front_azimuth=0.0)
    camera.orbit(30.0, -20.0)

    camera.fit(
        np.asarray((-1.0, -0.5, 0.0), dtype=np.float32),
        np.asarray((1.0, 0.5, 2.0), dtype=np.float32),
    )

    assert camera._camera.azimuth == pytest.approx(0.0)
    assert camera.direction_from_target() == pytest.approx((
        0.0,
        -math.cos(math.radians(12.0)),
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
    assert "abs(dot(normal, light_direction))" in _SMOOTH_FRAGMENT_SHADER
    assert "abs(dot(normal, light_direction))" in (
        _SMOOTH_TEXTURED_FRAGMENT_SHADER
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


def test_generated_vertex_normals_smooth_shared_vertices() -> None:
    positions = np.asarray([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    ], dtype=np.float32)
    faces = np.asarray([(0, 1, 2), (0, 3, 1)], dtype=np.uint32)

    normals = _compute_vertex_normals(positions, faces)

    np.testing.assert_allclose(normals, np.asarray((
        (0.0, 2**-0.5, 2**-0.5),
        (0.0, 2**-0.5, 2**-0.5),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
    ), dtype=np.float32), atol=1e-6)


def test_position_only_mesh_gets_smooth_render_variant() -> None:
    from tmesh import TcAttribType, TcMesh, TcVertexLayout

    positions = np.ascontiguousarray([
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
    ], dtype=np.float32)
    indices = np.ascontiguousarray((0, 1, 2), dtype=np.uint32)
    layout = TcVertexLayout()
    assert layout.add("position", 3, TcAttribType.FLOAT32, 0)
    source = TcMesh.from_interleaved(
        positions, 3, indices, layout, uuid="smooth-fallback-source"
    )

    smooth = _build_smooth_mesh(source)

    assert smooth is not None
    assert source.vertex_normals is None
    assert smooth.vertex_normals is not None
    np.testing.assert_allclose(
        smooth.vertex_normals,
        np.tile((0.0, 0.0, 1.0), (3, 1)),
        atol=1e-6,
    )


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
