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

    def test_cli_argument_parsing(self) -> None:
        args = parse_args(["--mode", "both", "--port", "/dev/ttyUSB1", "--profile", "hamin"])
        self.assertEqual(args.mode, "both")
        self.assertEqual(args.port, "/dev/ttyUSB1")
        self.assertEqual(args.profile, "hamin")


if __name__ == "__main__":
    unittest.main()
