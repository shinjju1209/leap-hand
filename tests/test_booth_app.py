"""Unit tests for LEAP Hand Interactive Booth Kiosk Application."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import cv2
import numpy as np

from booth_app import (
    AppScreen,
    BoothKioskApp,
    Button,
    RpsScoreboard,
    RpsState,
    parse_args,
)


class BoothAppTests(unittest.TestCase):
    """Test suite verifying booth app state transitions, scoring, and UI components."""

    def setUp(self) -> None:
        tracker_patcher = patch("booth_app.MediaPipeTracker")
        self.mock_tracker_class = tracker_patcher.start()
        self.addCleanup(tracker_patcher.stop)
        self.button = Button(
            id="test_btn",
            label="Test Button",
            rect=(100, 100, 200, 50),
            shortcut="T",
        )

    def test_button_hit_detection_and_drawing(self) -> None:
        # Inside
        self.assertTrue(self.button.is_inside(150, 120))
        # Outside
        self.assertFalse(self.button.is_inside(50, 50))
        self.assertFalse(self.button.is_inside(350, 120))

        # Canvas drawing test
        canvas = np.zeros((400, 400, 3), dtype=np.uint8)
        self.button.draw(canvas, (150, 120))
        # Check non-zero pixels drawn inside the button rect
        self.assertGreater(np.count_nonzero(canvas[100:150, 100:300]), 0)

    def test_rps_scoreboard_logic(self) -> None:
        sb = RpsScoreboard()
        self.assertEqual(sb.total_rounds, 0)

        # Human rock vs Robot scissors -> human win
        res1 = sb.record_round("rock", "scissors")
        self.assertEqual(res1, "win")
        self.assertEqual(sb.human_wins, 1)
        self.assertEqual(sb.robot_wins, 0)
        self.assertEqual(sb.ties, 0)

        # Human rock vs Robot paper -> human loss
        res2 = sb.record_round("rock", "paper")
        self.assertEqual(res2, "loss")
        self.assertEqual(sb.robot_wins, 1)

        # Tie
        res3 = sb.record_round("rock", "rock")
        self.assertEqual(res3, "tie")
        self.assertEqual(sb.ties, 1)
        self.assertEqual(sb.total_rounds, 3)

        # Reset
        sb.reset()
        self.assertEqual(sb.total_rounds, 0)
        self.assertEqual(sb.human_wins, 0)
        self.assertEqual(sb.robot_wins, 0)
        self.assertEqual(sb.ties, 0)

    @patch("booth_app.MujocoHandController")
    @patch("booth_app.LeapHandHardwareController")
    def test_navigation_and_screen_transitions(
        self,
        mock_hw: MagicMock,
        mock_mj: MagicMock,
    ) -> None:
        app = BoothKioskApp(
            enable_mujoco=False,
            enable_hardware=False,
        )
        self.assertEqual(app.current_screen, AppScreen.HOME)

        # Go to Teleop
        app.handle_action("goto_teleop")
        self.assertEqual(app.current_screen, AppScreen.TELEOP)

        # Return Home
        app.handle_action("back_home_teleop")
        self.assertEqual(app.current_screen, AppScreen.HOME)

        # Go to RPS
        app.handle_action("goto_rps")
        self.assertEqual(app.current_screen, AppScreen.RPS)

        # Return Home
        app.handle_action("back_home_rps")
        self.assertEqual(app.current_screen, AppScreen.HOME)

        # Go to Showcase
        app.handle_action("goto_showcase")
        self.assertEqual(app.current_screen, AppScreen.SHOWCASE)

    @patch("booth_app.MujocoHandController")
    @patch("booth_app.LeapHandHardwareController")
    def test_teleop_calibration_and_arming(
        self,
        mock_hw: MagicMock,
        mock_mj: MagicMock,
    ) -> None:
        app = BoothKioskApp(
            enable_mujoco=False,
            enable_hardware=False,
        )
        app.current_screen = AppScreen.TELEOP

        # Arm & Disarm
        self.assertFalse(app.armed)
        app.handle_action("toggle_arm")
        self.assertTrue(app.armed)
        app.handle_action("disarm")
        self.assertFalse(app.armed)

        # Open Hand Calib
        app.handle_action("calib_open")
        self.assertTrue(app.calib_open_in_progress)

        # Populate a mock neutral calibration profile
        app.calibrator._profiles[app.profile] = {"Right": np.zeros(16)}

        # Fist Calib now succeeds
        app.handle_action("calib_fist")
        self.assertTrue(app.calib_fist_in_progress)
        self.assertFalse(app.calib_open_in_progress)

    @patch("booth_app.MujocoHandController")
    @patch("booth_app.LeapHandHardwareController")
    def test_tracking_loss_heartbeats_then_auto_disarms(
        self,
        mock_hw_class: MagicMock,
        mock_mj: MagicMock,
    ) -> None:
        app = BoothKioskApp(
            enable_mujoco=False,
            enable_hardware=False,
            tracking_loss_hold_seconds=0.2,
            tracking_loss_disarm_seconds=0.5,
        )
        hardware = mock_hw_class.return_value
        hardware.torque_enabled = True
        app.hardware_controller = hardware
        app.armed = True
        app.tracking_loss_started_at = 10.0

        app._handle_teleop_tracking_loss(10.3)
        hardware.heartbeat.assert_called_once_with()
        self.assertTrue(app.armed)
        self.assertIn("holding last pose", app.status_message)

        app._handle_teleop_tracking_loss(10.6)
        hardware.emergency_stop.assert_called_once_with()
        self.assertFalse(app.armed)
        self.assertIn("automatically DISARMED", app.status_message)

    @patch("booth_app.MujocoHandController")
    @patch("booth_app.LeapHandHardwareController")
    def test_heartbeat_failure_immediately_disarms(
        self,
        mock_hw_class: MagicMock,
        mock_mj: MagicMock,
    ) -> None:
        app = BoothKioskApp(enable_mujoco=False, enable_hardware=False)
        hardware = mock_hw_class.return_value
        hardware.torque_enabled = True
        hardware.heartbeat.side_effect = OSError("serial link lost")
        app.hardware_controller = hardware
        app.armed = True

        app._handle_teleop_tracking_loss(20.0)

        hardware.emergency_stop.assert_called_once_with()
        self.assertFalse(app.armed)
        self.assertIn("heartbeat failed", app.status_message)

    @patch("booth_app.MujocoHandController")
    @patch("booth_app.LeapHandHardwareController")
    def test_arm_failure_does_not_leave_false_armed_state(
        self,
        mock_hw_class: MagicMock,
        mock_mj: MagicMock,
    ) -> None:
        app = BoothKioskApp(enable_mujoco=False, enable_hardware=False)
        hardware = mock_hw_class.return_value
        hardware.enable_torque.side_effect = OSError("watchdog locked")
        app.hardware_controller = hardware

        app.arm_robot()

        self.assertFalse(app.armed)
        hardware.emergency_stop.assert_called_once_with()
        self.assertIn("ARM failed", app.status_message)

    @patch("booth_app.MujocoHandController")
    @patch("booth_app.LeapHandHardwareController")
    def test_shutdown_uses_close_and_releases_all_resources(
        self,
        mock_hw_class: MagicMock,
        mock_mj_class: MagicMock,
    ) -> None:
        app = BoothKioskApp(enable_mujoco=False, enable_hardware=False)
        hardware = mock_hw_class.return_value
        mujoco = mock_mj_class.return_value
        tracker = app.tracker
        capture = MagicMock()
        capture.isOpened.return_value = True
        app.hardware_controller = hardware
        app.mujoco_controller = mujoco

        with patch("booth_app.cv2.destroyAllWindows"):
            app.shutdown(capture)

        hardware.close.assert_called_once_with()
        mujoco.close.assert_called_once_with()
        tracker.close.assert_called_once_with()
        capture.release.assert_called_once_with()

    @patch("booth_app.MujocoHandController")
    @patch("booth_app.LeapHandHardwareController")
    def test_ctrl_c_exits_without_propagating_keyboard_interrupt(
        self,
        mock_hw: MagicMock,
        mock_mj: MagicMock,
    ) -> None:
        app = BoothKioskApp(enable_mujoco=False, enable_hardware=False)
        app.tracker.process_frame.side_effect = KeyboardInterrupt
        capture = MagicMock()
        capture.isOpened.return_value = True
        capture.read.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))

        with (
            patch("booth_app.cv2.VideoCapture", return_value=capture),
            patch("booth_app.cv2.namedWindow"),
            patch("booth_app.cv2.setMouseCallback"),
            patch("booth_app.cv2.destroyAllWindows"),
        ):
            self.assertEqual(app.run(), 0)

        app.tracker.close.assert_called_once_with()
        capture.release.assert_called_once_with()

    @patch("booth_app.MujocoHandController")
    @patch("booth_app.LeapHandHardwareController")
    def test_rps_game_lifecycle(
        self,
        mock_hw: MagicMock,
        mock_mj: MagicMock,
    ) -> None:
        app = BoothKioskApp(
            enable_mujoco=False,
            enable_hardware=False,
        )
        app.current_screen = AppScreen.RPS
        self.assertEqual(app.rps_state, RpsState.IDLE)

        # Start game -> Countdown
        app.handle_action("rps_start")
        self.assertEqual(app.rps_state, RpsState.COUNTDOWN)

        # Auto-play toggle
        self.assertFalse(app.auto_play)
        app.handle_action("rps_auto_toggle")
        self.assertTrue(app.auto_play)
        app.handle_action("rps_auto_toggle")
        self.assertFalse(app.auto_play)

    @patch("booth_app.MujocoHandController")
    @patch("booth_app.LeapHandHardwareController")
    def test_render_all_screens(
        self,
        mock_hw: MagicMock,
        mock_mj: MagicMock,
    ) -> None:
        app = BoothKioskApp(
            enable_mujoco=False,
            enable_hardware=False,
        )
        fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for screen in (AppScreen.HOME, AppScreen.TELEOP, AppScreen.RPS, AppScreen.SHOWCASE):
            app.current_screen = screen
            canvas = app.render(fake_frame, [])
            self.assertEqual(canvas.shape, (720, 1280, 3))
            self.assertGreater(np.count_nonzero(canvas), 0)

    @patch("booth_app.MujocoHandController")
    @patch("booth_app.LeapHandHardwareController")
    def test_showcase_postures_and_animations(
        self,
        mock_hw: MagicMock,
        mock_mj: MagicMock,
    ) -> None:
        app = BoothKioskApp(
            enable_mujoco=False,
            enable_hardware=False,
        )
        app.current_screen = AppScreen.SHOWCASE

        # Test static postures
        for action_id in (
            "showcase_rock",
            "showcase_paper",
            "showcase_scissors",
            "showcase_neutral",
            "showcase_middle",
            "showcase_thumbs_up",
            "showcase_ok",
            "showcase_pointing",
            "showcase_rock_on",
        ):
            app.handle_action(action_id)
            self.assertEqual(len(app.target_joint_angles), 16)
            self.assertIsNone(app.active_animation)

        # Test dynamic animations
        app.handle_action("showcase_finger_wave")
        self.assertEqual(app.active_animation, "finger_wave")
        app.step_smooth_control(0.02)
        self.assertEqual(len(app.target_joint_angles), 16)

        app.handle_action("showcase_wave_hello")
        self.assertEqual(app.active_animation, "wave_hello")
        app.step_smooth_control(0.02)
        self.assertEqual(len(app.target_joint_angles), 16)

    def test_cli_argument_parsing(self) -> None:
        args = parse_args(["--mode", "both", "--port", "/dev/ttyUSB1", "--profile", "hamin"])
        self.assertEqual(args.mode, "both")
        self.assertEqual(args.port, "/dev/ttyUSB1")
        self.assertEqual(args.profile, "hamin")
        self.assertEqual(args.tracking_loss_hold_seconds, 0.2)
        self.assertEqual(args.tracking_loss_disarm_seconds, 0.5)

    def test_tracking_loss_timeouts_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            BoothKioskApp(
                enable_mujoco=False,
                enable_hardware=False,
                tracking_loss_hold_seconds=0.5,
                tracking_loss_disarm_seconds=0.5,
            )

    @patch("booth_app.MujocoHandController")
    @patch("booth_app.LeapHandHardwareController")
    def test_quit_button_requests_clean_loop_exit(
        self,
        mock_hw: MagicMock,
        mock_mj: MagicMock,
    ) -> None:
        app = BoothKioskApp(enable_mujoco=False, enable_hardware=False)
        self.assertFalse(app.exit_requested)
        app.handle_action("quit_app")
        self.assertTrue(app.exit_requested)


if __name__ == "__main__":
    unittest.main()
