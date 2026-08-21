import unittest

import numpy as np

from hand_angles import ANGLE_NAMES
from mujoco_finger_test import FINGER_TARGETS_DEGREES, make_pose


class MujocoFingerTestTests(unittest.TestCase):
    def test_each_pose_moves_only_its_named_finger(self):
        expected_nonzero = {
            "Index": {1, 2, 3},
            "Middle": {5, 6, 7},
            "Ring": {9, 10, 11},
            "Thumb": {13, 14, 15},
        }
        for finger_name, changes in FINGER_TARGETS_DEGREES.items():
            pose = make_pose(changes)
            actual_nonzero = set(np.flatnonzero(pose).tolist())
            self.assertEqual(actual_nonzero, expected_nonzero[finger_name])

    def test_target_names_exist_in_angle_interface(self):
        for changes in FINGER_TARGETS_DEGREES.values():
            for angle_name in changes:
                self.assertIn(angle_name, ANGLE_NAMES)


if __name__ == "__main__":
    unittest.main()
