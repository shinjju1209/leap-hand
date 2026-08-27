import unittest
from pathlib import Path
import numpy as np

from hand_angles import ANGLE_NAMES
from leap_hand_hardware_finger_test import (
    FINGER_TARGETS_DEGREES,
    make_pose,
    parse_args,
)


class LeapHandHardwareFingerTestTests(unittest.TestCase):
    def test_each_pose_moves_only_its_named_finger(self):
        expected_nonzero = {
            "Index (검지)": {1, 2, 3},
            "Middle (중지)": {5, 6, 7},
            "Ring (약지)": {9, 10, 11},
            "Thumb (엄지)": {13, 14, 15},
        }
        for finger_name, changes in FINGER_TARGETS_DEGREES.items():
            pose = make_pose(changes)
            actual_nonzero = set(np.flatnonzero(pose).tolist())
            self.assertEqual(actual_nonzero, expected_nonzero[finger_name])

    def test_target_names_exist_in_angle_interface(self):
        for changes in FINGER_TARGETS_DEGREES.values():
            for angle_name in changes:
                self.assertIn(angle_name, ANGLE_NAMES)

    def test_cli_parsing_defaults(self):
        args = parse_args([])
        self.assertEqual(args.mode, "hardware")
        self.assertEqual(args.port, "/dev/ttyUSB0")
        self.assertEqual(args.current_limit, 300)
        self.assertEqual(args.cycles, 1)
        self.assertFalse(args.loop)
        self.assertEqual(args.transition_seconds, 1.2)


if __name__ == "__main__":
    unittest.main()
