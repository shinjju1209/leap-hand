import unittest

from webcam_hand_tracking import DEFAULT_MUJOCO_MODEL_PATH, parse_args


class WebcamCliTests(unittest.TestCase):
    def test_teleoperation_entry_point_can_enable_mujoco_by_default(self):
        args = parse_args([], default_mujoco=True)
        self.assertTrue(args.mujoco)
        self.assertEqual(args.mujoco_model, DEFAULT_MUJOCO_MODEL_PATH)

    def test_mujoco_default_can_be_overridden(self):
        args = parse_args(["--no-mujoco"], default_mujoco=True)
        self.assertFalse(args.mujoco)

    def test_regular_webcam_mode_keeps_mujoco_disabled(self):
        args = parse_args([])
        self.assertFalse(args.mujoco)

    def test_flexion_scale_can_be_tuned(self):
        args = parse_args(["--flexion-scale", "1.1"])
        self.assertEqual(args.flexion_scale, 1.1)

    def test_collision_avoidance_is_enabled_by_default(self):
        self.assertTrue(parse_args([]).collision_avoidance)
        self.assertFalse(
            parse_args(["--no-collision-avoidance"]).collision_avoidance
        )


if __name__ == "__main__":
    unittest.main()
