import unittest
import numpy as np
import cv2

from webcam_hand_tracking import draw_hardware_status, main, parse_args


class WebcamHardwareTeleopTests(unittest.TestCase):
    def test_invalid_current_limit_raises_value_error(self):
        with self.assertRaises(ValueError):
            main(["--hardware", "--current-limit", "0"])
        with self.assertRaises(ValueError):
            main(["--hardware", "--current-limit", "600"])

    def test_invalid_max_joint_speed_raises_value_error(self):
        with self.assertRaises(ValueError):
            main(["--hardware", "--max-joint-speed", "0"])

    def test_invalid_max_tracking_error_raises_value_error(self):
        with self.assertRaises(ValueError):
            main(["--hardware", "--max-tracking-error", "0"])

    def test_invalid_max_temperature_raises_value_error(self):
        with self.assertRaises(ValueError):
            main(["--hardware", "--max-temperature", "0"])

    def test_invalid_tracking_loss_timing_raises_value_error(self):
        with self.assertRaises(ValueError):
            main([
                "--hardware",
                "--tracking-loss-hold-seconds", "0.5",
                "--tracking-loss-disarm-seconds", "0.5",
            ])

    def test_draw_hardware_status_renders_all_states(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Disarmed
        draw_hardware_status(
            frame,
            is_armed=False,
            error_msg=None,
            max_temp=38.5,
            worst_error=0.0,
        )
        # Armed
        draw_hardware_status(
            frame,
            is_armed=True,
            error_msg=None,
            max_temp=41.2,
            worst_error=2.4,
        )
        # Tracking loss hold
        draw_hardware_status(
            frame,
            is_armed=True,
            error_msg=None,
            max_temp=41.2,
            worst_error=2.4,
            loss_hold=True,
        )
        # Error message
        draw_hardware_status(
            frame,
            is_armed=False,
            error_msg="TRACKING LOSS TIMEOUT",
            max_temp=42.0,
            worst_error=5.1,
        )


if __name__ == "__main__":
    unittest.main()
