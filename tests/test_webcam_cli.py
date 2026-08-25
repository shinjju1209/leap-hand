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

    def test_teleoperation_entry_point_can_enable_hardware_by_default(self):
        args = parse_args([], default_hardware=True)
        self.assertTrue(args.hardware)
        self.assertEqual(args.port, "/dev/ttyUSB0")
        self.assertEqual(args.baudrate, 4_000_000)
        self.assertEqual(args.current_limit, 300)
        self.assertEqual(args.max_joint_speed, 120.0)
        self.assertEqual(args.max_tracking_error, 25.0)
        self.assertEqual(args.max_temperature, 50.0)
        self.assertEqual(args.tracking_loss_hold_seconds, 0.2)
        self.assertEqual(args.tracking_loss_disarm_seconds, 0.5)

    def test_hardware_default_can_be_overridden(self):
        args = parse_args(["--no-hardware"], default_hardware=True)
        self.assertFalse(args.hardware)

    def test_hardware_custom_arguments(self):
        args = parse_args([
            "--hardware",
            "--port", "/dev/ttyUSB1",
            "--current-limit", "250",
            "--max-joint-speed", "60.0",
            "--max-tracking-error", "20.0",
            "--max-temperature", "45.0",
            "--tracking-loss-hold-seconds", "0.3",
            "--tracking-loss-disarm-seconds", "0.8",
        ])
        self.assertTrue(args.hardware)
        self.assertEqual(args.port, "/dev/ttyUSB1")
        self.assertEqual(args.current_limit, 250)
        self.assertEqual(args.max_joint_speed, 60.0)
        self.assertEqual(args.max_tracking_error, 20.0)
        self.assertEqual(args.max_temperature, 45.0)
        self.assertEqual(args.tracking_loss_hold_seconds, 0.3)
        self.assertEqual(args.tracking_loss_disarm_seconds, 0.8)


if __name__ == "__main__":
    unittest.main()
