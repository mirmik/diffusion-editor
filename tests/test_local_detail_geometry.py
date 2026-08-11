import numpy as np
import pytest

from diffusion_editor.generation.local_detail_geometry import (
    LocalDetailTransform,
    fit_local_detail_transform,
    local_roi_bounds,
    overlap_face_masks,
    project_pixal_points,
    sample_mask_bilinear,
)


def test_pixal_projection_places_origin_at_image_center():
    pixels, depth, valid = project_pixal_points(
        np.zeros((1, 3)),
        camera_angle_x=np.radians(45.0),
        distance=2.0,
        image_size=(512, 512),
    )

    np.testing.assert_allclose(pixels[0], (256.0, 256.0))
    assert depth[0] == pytest.approx(2.0)
    assert valid.tolist() == [True]


def test_mask_sampling_uses_pixel_centers_and_rejects_outside_points():
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 255

    sampled = sample_mask_bilinear(
        mask,
        np.asarray(((1.5, 1.5), (1.0, 1.5), (-2.0, -2.0))),
    )

    np.testing.assert_allclose(sampled, (1.0, 0.5, 0.0))


def test_roi_bounds_rejects_empty_selection_and_pads_selected_geometry():
    points = np.asarray((
        (-0.1, 0.2, -0.05),
        (0.1, 0.2, 0.05),
        (-0.1, 0.4, 0.05),
        (0.1, 0.4, -0.05),
    ))

    bounds = local_roi_bounds(
        points, np.ones(4), quantile=0.0, padding=0.25
    )

    np.testing.assert_allclose(bounds.mean(axis=0), (0.0, 0.3, 0.0))
    np.testing.assert_allclose(bounds[1] - bounds[0], (0.3, 0.3, 0.15))
    with pytest.raises(ValueError, match="too little"):
        local_roi_bounds(points, np.zeros(4))


def test_local_transform_is_uniform_and_invertible():
    local = np.asarray((
        (-0.5, -0.25, -0.1),
        (0.5, 0.25, 0.1),
        (-0.5, 0.25, 0.1),
    ))
    target = np.asarray(((-0.2, 0.3, -0.1), (0.2, 0.7, 0.3)))

    transform = fit_local_detail_transform(local, target, quantile=0.0)
    transformed = transform.apply(local)

    assert transform.scale == pytest.approx(0.4)
    np.testing.assert_allclose(transform.inverse(transformed), local)
    np.testing.assert_allclose(
        np.quantile(transformed, 0.5, axis=0)[1], 0.5, atol=0.1
    )


def test_overlap_masks_leave_a_shared_collar():
    vertices = np.asarray((
        (-1.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.8, 0.1, 0.0),
    ))
    faces = np.asarray(((0, 1, 3), (1, 2, 3)))
    bounds = np.asarray(((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)))

    keep_base, keep_local = overlap_face_masks(
        vertices, faces, vertices, faces, bounds,
        base_inner_radius=0.3, local_outer_radius=0.7,
    )

    assert keep_base.tolist() == [False, True]
    assert keep_local.tolist() == [True, True]
    assert bool((keep_base & keep_local).any())


def test_transform_methods_preserve_float32_arrays():
    transform = LocalDetailTransform(0.5, (1.0, 2.0, 3.0))
    points = np.zeros((2, 3), dtype=np.float32)

    assert transform.apply(points).dtype == np.float32
