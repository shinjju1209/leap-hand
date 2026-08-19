import unittest

import numpy as np

from hand_angles import calculate_leap_control_angles, joint_bend_angle_deg


class Landmark:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


class HandAngleTests(unittest.TestCase):
    def test_segment_bend_angles(self):
        self.assertAlmostEqual(
            joint_bend_angle_deg(np.array([1, 0, 0]), np.array([1, 0, 0])),
            0.0,
        )
        self.assertAlmostEqual(
            joint_bend_angle_deg(np.array([1, 0, 0]), np.array([0, 1, 0])),
            90.0,
        )

    def test_straight_synthetic_hand_returns_finite_vector(self):
        points = np.zeros((21, 3), dtype=float)
        points[0] = [0.0, 0.0, 0.0]

        rays = {
            5: np.array([0.75, 1.0, 0.0]),
            9: np.array([0.25, 1.1, 0.0]),
            13: np.array([-0.35, 1.0, 0.0]),
            17: np.array([-0.8, 0.85, 0.0]),
        }
        for base, ray in rays.items():
            direction = ray / np.linalg.norm(ray)
            points[base] = ray
            points[base + 1] = ray + 0.35 * direction
            points[base + 2] = ray + 0.65 * direction
            points[base + 3] = ray + 0.9 * direction

        points[1] = [0.35, 0.2, 0.0]
        points[2] = [0.65, 0.35, 0.0]
        points[3] = [0.9, 0.5, 0.0]
        points[4] = [1.1, 0.62, 0.0]

        landmarks = [Landmark(*point) for point in points]
        angles = calculate_leap_control_angles(landmarks)

        self.assertEqual(angles.shape, (16,))
        self.assertTrue(np.all(np.isfinite(angles)))
        np.testing.assert_allclose(angles[1:12:4], 0.0, atol=1e-5)
        np.testing.assert_allclose(angles[2:12:4], 0.0, atol=1e-5)
        np.testing.assert_allclose(angles[3:12:4], 0.0, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
