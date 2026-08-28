"""Unit tests for Unbeatable Rock-Paper-Scissors Game (rps_unbeatable_game.py)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from rps_unbeatable_game import (
    GameMode,
    MatchState,
    UnbeatableRpsApp,
    UnbeatableScoreboard,
    WINNING_COUNTER,
    LOSING_COUNTER,
    parse_args,
)


class UnbeatableRpsTests(unittest.TestCase):
    def test_scoreboard_lifecycle(self) -> None:
        sb = UnbeatableScoreboard()
        self.assertEqual(sb.ai_wins, 0)
        self.assertEqual(sb.human_wins, 0)
        self.assertEqual(sb.current_ai_streak, 0)

        # Record AI wins (human losses)
        sb.record_result("loss")
        self.assertEqual(sb.ai_wins, 1)
        self.assertEqual(sb.current_ai_streak, 1)

        sb.record_result("loss")
        self.assertEqual(sb.ai_wins, 2)
        self.assertEqual(sb.current_ai_streak, 2)
        self.assertEqual(sb.max_ai_streak, 2)

        # Record Human win (breaks AI streak)
        sb.record_result("win")
        self.assertEqual(sb.human_wins, 1)
        self.assertEqual(sb.current_ai_streak, 0)
        self.assertEqual(sb.max_ai_streak, 2)

        # Reset
        sb.reset()
        self.assertEqual(sb.ai_wins, 0)
        self.assertEqual(sb.human_wins, 0)
        self.assertEqual(sb.current_ai_streak, 0)
        self.assertEqual(len(sb.recent_history), 0)

    def test_winning_counter_mappings(self) -> None:
        self.assertEqual(WINNING_COUNTER["rock"], "paper")
        self.assertEqual(WINNING_COUNTER["paper"], "scissors")
        self.assertEqual(WINNING_COUNTER["scissors"], "rock")

    def test_losing_counter_mappings(self) -> None:
        self.assertEqual(LOSING_COUNTER["rock"], "scissors")
        self.assertEqual(LOSING_COUNTER["paper"], "rock")
        self.assertEqual(LOSING_COUNTER["scissors"], "paper")

    @patch("rps_unbeatable_game.MujocoHandController")
    @patch("rps_unbeatable_game.LeapHandHardwareController")
    def test_app_mode_cycling(
        self,
        mock_hw: MagicMock,
        mock_mj: MagicMock,
    ) -> None:
        app = UnbeatableRpsApp(enable_hardware=False, enable_mujoco=False)
        self.assertEqual(app.game_mode, GameMode.GOD_MODE)

        app.cycle_mode()
        self.assertEqual(app.game_mode, GameMode.FAIR)

        app.cycle_mode()
        self.assertEqual(app.game_mode, GameMode.TROLL_LOSE)

        app.cycle_mode()
        self.assertEqual(app.game_mode, GameMode.MIRROR)

        app.cycle_mode()
        self.assertEqual(app.game_mode, GameMode.GOD_MODE)

    @patch("rps_unbeatable_game.MujocoHandController")
    @patch("rps_unbeatable_game.LeapHandHardwareController")
    def test_app_render_canvas(
        self,
        mock_hw: MagicMock,
        mock_mj: MagicMock,
    ) -> None:
        app = UnbeatableRpsApp(enable_hardware=False, enable_mujoco=False)
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        for state in (MatchState.IDLE, MatchState.COUNTDOWN, MatchState.RESULT):
            app.match_state = state
            canvas = app.render(dummy_frame, [])
            self.assertEqual(canvas.shape, (720, 1280, 3))
            self.assertGreater(np.count_nonzero(canvas), 0)

    def test_cli_parsing(self) -> None:
        args = parse_args(["--mode", "both", "--port", "/dev/ttyUSB2", "--camera-id", "1"])
        self.assertEqual(args.mode, "both")
        self.assertEqual(args.port, "/dev/ttyUSB2")
        self.assertEqual(args.camera_id, 1)


if __name__ == "__main__":
    unittest.main()
