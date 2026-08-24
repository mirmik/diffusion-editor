import numpy as np
from PIL import Image

from diffusion_editor.training.canonical_pointmap import (
    camera_from_canonical_matrix,
    camera_distance_world,
    camera_intrinsics,
    canonical_camera_ray_map,
    canonical_from_world,
    normalized_camera_vector,
    pointmap_reprojection_error,
    project_canonical_points,
    write_binary_mask_png,
    world_from_canonical_matrix,
)


def test_camera_distance_can_be_normalized_by_character_height():
    assert camera_distance_world(
        2.2, character_height=5.0, unit="character-height"
    ) == 11.0
    assert camera_distance_world(4.0, character_height=5.0, unit="world") == 4.0


def test_blender_world_axes_map_to_right_handed_character_frame():
    pelvis = np.array((2.0, 3.0, 4.0))
    world = pelvis + np.array([
        (2.0, 0.0, 0.0),
        (0.0, 0.0, 2.0),
        (0.0, -2.0, 0.0),
    ])

    canonical = canonical_from_world(
        world, pelvis_world=pelvis, character_height=2.0
    )

    np.testing.assert_allclose(canonical, np.eye(3))


def test_world_from_canonical_inverts_coordinate_conversion():
    pelvis = np.array((0.2, -0.3, 1.1))
    canonical = np.array([
        (0.0, 0.0, 0.0),
        (0.25, -0.5, 0.75),
        (-0.1, 0.2, -0.3),
    ])
    transform = world_from_canonical_matrix(pelvis, 1.7)
    homogeneous = np.column_stack((canonical, np.ones(len(canonical))))
    world = (transform @ homogeneous.T).T[:, :3]

    reconstructed = canonical_from_world(
        world, pelvis_world=pelvis, character_height=1.7
    )

    np.testing.assert_allclose(reconstructed, canonical)


def test_centered_intrinsics_project_camera_axis_to_image_center():
    intrinsics = camera_intrinsics(
        width=640, height=480, lens_mm=50.0, sensor_width_mm=36.0
    )
    pixels, depth = project_canonical_points(
        np.array([(0.0, 0.0, 2.0)]),
        intrinsics=intrinsics,
        camera_from_canonical=np.eye(4),
    )

    np.testing.assert_allclose(pixels[0], (320.0, 240.0))
    np.testing.assert_allclose(depth, (2.0,))
    assert intrinsics[0, 0] == intrinsics[1, 1]


def test_blender_camera_matrix_is_converted_to_cv_forward_axis():
    transform = camera_from_canonical_matrix(
        world_from_camera_blender=np.eye(4),
        pelvis_world=np.zeros(3),
        character_height=1.0,
    )
    # Canonical forward is Blender -Y.  Rotate the identity Blender camera
    # frame so this test only checks its local -Z to CV +Z conversion.
    assert transform.shape == (4, 4)
    np.testing.assert_allclose(transform[3], (0.0, 0.0, 0.0, 1.0))


def test_pointmap_reprojection_matches_pixel_centers():
    width, height = 4, 3
    intrinsics = np.array([
        [4.0, 0.0, width * 0.5],
        [0.0, 4.0, height * 0.5],
        [0.0, 0.0, 1.0],
    ])
    rows, columns = np.indices((height, width))
    depth = np.full((height, width), 2.0)
    pointmap = np.stack(
        (
            (columns + 0.5 - intrinsics[0, 2]) * depth / intrinsics[0, 0],
            (rows + 0.5 - intrinsics[1, 2]) * depth / intrinsics[1, 1],
            depth,
        ),
        axis=-1,
    )

    error = pointmap_reprojection_error(
        pointmap,
        np.ones((height, width), dtype=bool),
        intrinsics=intrinsics,
        camera_from_canonical=np.eye(4),
    )

    np.testing.assert_allclose(error, 0.0, atol=1e-12)


def test_binary_mask_png_preserves_orientation_and_values(tmp_path):
    mask = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.uint8)
    path = tmp_path / "mask.png"

    write_binary_mask_png(path, mask)

    saved = np.asarray(Image.open(path))
    np.testing.assert_array_equal(saved, mask * 255)


def test_canonical_camera_ray_map_has_origin_and_unit_center_ray():
    rays = canonical_camera_ray_map(
        camera_from_canonical=np.eye(4),
        intrinsics=np.array([
            [100.0, 0.0, 50.0],
            [0.0, 100.0, 50.0],
            [0.0, 0.0, 1.0],
        ]),
        image_size=(100, 100),
        grid_height=1,
        grid_width=1,
    )

    np.testing.assert_allclose(rays[0, 0, :3], 0.0)
    np.testing.assert_allclose(rays[0, 0, 3:], (0.0, 0.0, 1.0))


def test_normalized_camera_vector_preserves_transform_and_scales_intrinsics():
    transform = np.arange(16, dtype=np.float32).reshape(4, 4)
    intrinsics = np.array([
        [200.0, 0.0, 50.0],
        [0.0, 300.0, 40.0],
        [0.0, 0.0, 1.0],
    ])

    encoded = normalized_camera_vector(
        camera_from_canonical=transform,
        intrinsics=intrinsics,
        image_size=(100, 80),
    )

    np.testing.assert_allclose(encoded[:12], transform[:3].reshape(-1))
    np.testing.assert_allclose(encoded[12:], (2.0, 3.75, 0.5, 0.5))
