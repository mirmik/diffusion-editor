"""Numerical contracts independent of Blender; run with the project venv."""
import unittest
import numpy as np
from anatomy import head_frame, head_to_world
from sdf import ellipsoid, interpolate, smooth_min


class FieldContracts(unittest.TestCase):
    def test_sphere_distance_including_center(self):
        p = (np.array([0., 1., 2.]), np.zeros(3), np.zeros(3))
        np.testing.assert_allclose(ellipsoid(p, (0, 0, 0), (1, 1, 1)), [-1, 0, 1])

    def test_union_blend_depth_and_symmetry(self):
        a, b = np.array([0., .5, -1]), np.array([0., -.4, 1])
        np.testing.assert_allclose(smooth_min(a, b, .04), [-.01, -.4, -1])
        np.testing.assert_allclose(smooth_min(a, b, .04), smooth_min(b, a, .04))

    def test_profile_hits_keys_without_overshoot(self):
        x, y = np.array([0., .7, 2., 3.]), np.array([.5, 2., 1., .1])
        np.testing.assert_allclose(interpolate(x, x, y), y)
        for i in range(3):
            values = interpolate(np.linspace(x[i], x[i+1], 100), x, y)
            self.assertGreaterEqual(values.min(), min(y[i:i+2]) - 1e-10)
            self.assertLessEqual(values.max(), max(y[i:i+2]) + 1e-10)

    def test_eye_placement_uses_same_frame_as_head_field(self):
        for angle in (-35, 0, 25, 35):
            params = dict(head_yaw_degrees=angle, neck_extension=.025)
            point = (.032, -.057, .613)
            np.testing.assert_allclose(head_frame(head_to_world(point, params), params), point, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
