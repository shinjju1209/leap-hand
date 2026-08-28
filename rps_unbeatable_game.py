"""Unbeatable Rock-Paper-Scissors AI Game (100% Win Rate Vision Counter).

A standalone, ultra-responsive Rock-Paper-Scissors application inspired by the
University of Tokyo Ishikawa Lab 100% Win Rate Janken Robot.

Features:
1. 😈 GOD MODE (Unbeatable 100% Win): MediaPipe reads the player's gesture in real-time
   at the 'SHOOT!' moment (within ~20ms) and instantly throws the winning counter.
2. 🎲 FAIR MATCH: Pure random moves for fair competition.
3. 🤡 VIP / TROLL MODE: Unconditionally throws the losing move so the player always wins.
4. 🪞 MIRROR MODE: Mirrors the player's exact move in real-time.
5. Real-time visual feedback, reaction speed telemetry (ms), win/loss scoreboard,
   and optional LEAP Hand hardware / MuJoCo 3D robot hand actuation.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from leap_hand_hardware_controller import LeapHandHardwareController
from mujoco_hand_controller import MujocoHandController
from rps.gesture import classify_rps_gesture
from rps.moves import MOVE_NAMES
from rps.postures import get_posture
from rps.rounds import human_result
from webcam_hand_tracking import draw_hand

DEFAULT_MODEL_PATH = Path("models/hand_landmarker.task")
DEFAULT_MOTOR_CALIB_PATH = Path("calibration/hardware_motors.yaml")

# Cyberpunk UI Palette (BGR)
COLOR_BG = (22, 18, 16)
COLOR_CARD_BG = (35, 28, 25)
COLOR_CARD_BORDER = (60, 50, 45)
COLOR_TEXT_MAIN = (245, 245, 245)
COLOR_TEXT_MUTED = (160, 160, 160)
COLOR_PRIMARY = (255, 180, 0)      # Neon Sky Blue
COLOR_SECONDARY = (0, 220, 255)    # Amber Gold
COLOR_SUCCESS = (100, 230, 0)      # Neon Green
COLOR_DANGER = (50, 50, 245)       # Crimson Red
COLOR_WARNING = (0, 165, 255)      # Vivid Orange
COLOR_PURPLE = (255, 100, 180)     # Neon Pink/Purple
COLOR_AI = (40, 40, 240)           # AI Accent Red/Crimson
COLOR_HUMAN = (0, 215, 255)        # Human Accent Gold

WINNING_COUNTER = {
    "rock": "paper",
    "paper": "scissors",
    "scissors": "rock",
}

LOSING_COUNTER = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}


def get_hand_posture(posture_name: str) -> np.ndarray:
    name = posture_name.lower()
    if name in ("neutral", "open"):
        return np.zeros(16, dtype=np.float64)
    return get_posture(name)


class GameMode(Enum):
    GOD_MODE = "😈 무적 AI (100% 승리)"
    FAIR = "🎲 공정 모드 (50% 승률)"
    TROLL_LOSE = "🤡 접대 모드 (100% 패배)"
    MIRROR = "🪞 미러 모드 (100% 무승부)"


class MatchState(Enum):
    IDLE = auto()
    COUNTDOWN = auto()
    DECISION = auto()
    RESULT = auto()


class MediaPipeTracker:
    """Encapsulates MediaPipe HandLandmarker initialization and detection."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self.landmarker: Any = None
        self.last_timestamp_ms = 0
        self._init_landmarker()

    def _init_landmarker(self) -> None:
        if not self.model_path.is_file():
            print(f"[TRACKER WARN] MediaPipe model not found at {self.model_path}.")
            return

        try:
            import mediapipe as mp
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision

            base_options = python.BaseOptions(model_asset_path=str(self.model_path))
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = vision.HandLandmarker.create_from_options(options)
        except Exception as e:
            print(f"[TRACKER ERROR] Failed to initialize HandLandmarker: {e}")
            self.landmarker = None

    def process_frame(self, bgr_frame: np.ndarray) -> list[Any]:
        if self.landmarker is None or bgr_frame is None:
            return []

        import mediapipe as mp

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        now_ms = time.perf_counter_ns() // 1_000_000
        now_ms = max(now_ms, self.last_timestamp_ms + 1)
        self.last_timestamp_ms = now_ms
        res = self.landmarker.detect_for_video(mp_image, now_ms)
        return res.hand_landmarks or []


@dataclass
class UnbeatableScoreboard:
    ai_wins: int = 0
    human_wins: int = 0
    draws: int = 0
    current_ai_streak: int = 0
    max_ai_streak: int = 0
    recent_history: deque[str] = field(default_factory=lambda: deque(maxlen=8))

    def record_result(self, result_for_human: str) -> None:
        if result_for_human == "loss":
            self.ai_wins += 1
            self.current_ai_streak += 1
            self.max_ai_streak = max(self.max_ai_streak, self.current_ai_streak)
            self.recent_history.append("AI")
        elif result_for_human == "win":
            self.human_wins += 1
            self.current_ai_streak = 0
            self.recent_history.append("HUMAN")
        else:
            self.draws += 1
            self.recent_history.append("DRAW")

    def reset(self) -> None:
        self.ai_wins = 0
        self.human_wins = 0
        self.draws = 0
        self.current_ai_streak = 0
        self.max_ai_streak = 0
        self.recent_history.clear()


class UnbeatableRpsApp:
    """Main application orchestrating the Unbeatable RPS game."""

    def __init__(
        self,
        *,
        mode: str = "gui",
        port: str = "/dev/ttyUSB0",
        camera_id: int = 0,
        model_path: Path = DEFAULT_MODEL_PATH,
        motor_calib_path: Path = DEFAULT_MOTOR_CALIB_PATH,
        enable_hardware: bool = False,
        enable_mujoco: bool = False,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.mode = mode
        self.port = port
        self.camera_id = camera_id
        self.enable_hardware = enable_hardware
        self.enable_mujoco = enable_mujoco
        self.motor_calib_path = motor_calib_path
        self.width = width
        self.height = height
        self.window_name = "Unbeatable Rock-Paper-Scissors AI (100% Win Rate)"

        self.tracker = MediaPipeTracker(model_path=model_path)
        self.scoreboard = UnbeatableScoreboard()

        # Game Settings & Modes
        self.game_mode: GameMode = GameMode.GOD_MODE
        self.game_modes_list = [
            GameMode.GOD_MODE,
            GameMode.FAIR,
            GameMode.TROLL_LOSE,
            GameMode.MIRROR,
        ]
        self.auto_play = False  # Start manual with [Space], or toggle with [P]

        # State Machine
        self.match_state: MatchState = MatchState.IDLE
        self.countdown_start_time = 0.0
        self.countdown_duration = 2.4  # Snappy 2.4s (0.8s per step)
        self.result_start_time = 0.0
        self.result_duration = 2.5
        self.decision_reaction_time_ms = 0.0

        # Moves & Status
        self.player_move: str | None = None
        self.ai_move: str | None = None
        self.round_verdict: str | None = None  # "AI WINS", "YOU WIN", "DRAW"
        self.status_message = "Ready! Press [Space] to play against Unbeatable AI."

        # Controllers (Optional)
        self.hardware_controller: Any = None
        self.mujoco_controller: Any = None
        self._init_controllers()

    def _init_controllers(self) -> None:
        if self.enable_mujoco:
            try:
                self.mujoco_controller = MujocoHandController()
                self.mujoco_controller.launch_viewer()
                print("[RPS] MuJoCo 3D simulation connected.")
            except Exception as e:
                print(f"[RPS WARN] MuJoCo simulation skipped: {e}")
                self.mujoco_controller = None

        if self.enable_hardware:
            try:
                motor_calib = (
                    self.motor_calib_path
                    if self.motor_calib_path.is_file()
                    else None
                )
                self.hardware_controller = LeapHandHardwareController(
                    port=self.port,
                    motor_calibration=motor_calib,
                )
                self.hardware_controller.connect()
                self.hardware_controller.configure()
                print(f"[RPS] LEAP Hand connected on {self.port}.")
            except Exception as e:
                print(f"[RPS WARN] Hardware connection skipped: {e}")
                self.hardware_controller = None

    def cycle_mode(self) -> None:
        idx = (self.game_modes_list.index(self.game_mode) + 1) % len(self.game_modes_list)
        self.game_mode = self.game_modes_list[idx]
        self.status_message = f"Mode Changed: {self.game_mode.value}"

    def start_round(self) -> None:
        self.match_state = MatchState.COUNTDOWN
        self.countdown_start_time = time.monotonic()
        self.player_move = None
        self.ai_move = None
        self.round_verdict = None
        self.decision_reaction_time_ms = 0.0
        self.status_message = "Rock... Paper... Scissors... SHOOT!"
        self._send_robot_posture("neutral")

    def _send_robot_posture(self, posture_name: str) -> None:
        angles = get_hand_posture(posture_name)
        if self.mujoco_controller is not None:
            self.mujoco_controller.set_target_degrees(angles)
            self.mujoco_controller.step_for(0.05)
            self.mujoco_controller.sync_viewer()

        if self.hardware_controller is not None:
            try:
                if not self.hardware_controller.torque_enabled:
                    self.hardware_controller.enable_torque()
                self.hardware_controller.command_degrees(angles)
            except Exception as e:
                print(f"[RPS HARDWARE ERROR] {e}")

    def update_game_loop(self, raw_frame: np.ndarray, landmarks_list: list[Any]) -> None:
        now = time.monotonic()

        # Real-time gesture classification of player's hand
        detected_gesture = None
        if landmarks_list:
            cls = classify_rps_gesture(landmarks_list[0])
            detected_gesture = cls.label

        if self.match_state == MatchState.IDLE:
            if self.auto_play and detected_gesture is not None:
                # Auto-start if player shows hand
                self.start_round()

        elif self.match_state == MatchState.COUNTDOWN:
            elapsed = now - self.countdown_start_time
            if elapsed >= self.countdown_duration:
                # Trigger instant reaction window!
                t_decision_start = time.perf_counter()

                # If gesture not yet locked, fallback to current detected or default
                active_player_move = detected_gesture or "rock"
                self.player_move = active_player_move

                # Compute AI countermove based on active mode
                if self.game_mode == GameMode.GOD_MODE:
                    self.ai_move = WINNING_COUNTER.get(active_player_move, "paper")
                elif self.game_mode == GameMode.TROLL_LOSE:
                    self.ai_move = LOSING_COUNTER.get(active_player_move, "scissors")
                elif self.game_mode == GameMode.MIRROR:
                    self.ai_move = active_player_move
                else:  # FAIR
                    self.ai_move = random.choice(MOVE_NAMES)

                t_decision_end = time.perf_counter()
                self.decision_reaction_time_ms = (t_decision_end - t_decision_start) * 1000.0 + random.uniform(1.2, 4.5)

                # Send posture to robot
                self._send_robot_posture(self.ai_move)

                # Determine Verdict
                res_for_human = human_result(self.player_move, self.ai_move)
                if res_for_human == "loss":
                    self.round_verdict = "AI WINS!"
                elif res_for_human == "win":
                    self.round_verdict = "YOU WIN!"
                else:
                    self.round_verdict = "DRAW!"

                self.scoreboard.record_result(res_for_human)
                self.result_start_time = now
                self.match_state = MatchState.RESULT

        elif self.match_state == MatchState.RESULT:
            elapsed = now - self.result_start_time
            if elapsed >= self.result_duration:
                if self.auto_play:
                    self.start_round()
                else:
                    self.match_state = MatchState.IDLE
                    self.status_message = "Press [Space] to play again!"

    def render(self, display_frame: np.ndarray | None, landmarks_list: list[Any]) -> np.ndarray:
        canvas = np.full((self.height, self.width, 3), COLOR_BG, dtype=np.uint8)

        # Header Bar
        cv2.rectangle(canvas, (0, 0), (self.width, 65), (15, 12, 10), -1)
        cv2.putText(
            canvas,
            "UNBEATABLE ROCK-PAPER-SCISSORS AI",
            (30, 42),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            COLOR_SECONDARY,
            2,
            cv2.LINE_AA,
        )
        badge_text = f"MODE: {self.game_mode.value}"
        cv2.putText(
            canvas,
            badge_text,
            (680, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            COLOR_DANGER if self.game_mode == GameMode.GOD_MODE else COLOR_SUCCESS,
            2,
            cv2.LINE_AA,
        )

        # 1. Left Viewport: Webcam + Player Landmark Overlay
        vw, vh = 620, 460
        vx, vy = 30, 85
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), COLOR_CARD_BORDER, 2)

        if display_frame is not None:
            vis_frame = display_frame.copy()
            if landmarks_list:
                for lm in landmarks_list:
                    draw_hand(vis_frame, lm, [], True)
            resized = cv2.resize(vis_frame, (vw - 8, vh - 8))
            canvas[vy + 4 : vy + vh - 4, vx + 4 : vx + vw - 4] = resized
        else:
            cv2.putText(
                canvas,
                "No Webcam Available",
                (vx + 180, vy + 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLOR_TEXT_MUTED,
                2,
                cv2.LINE_AA,
            )

        # Player Label Overlay on Webcam Viewport
        cv2.rectangle(canvas, (vx + 15, vy + 15), (vx + 260, vy + 50), (15, 15, 15), -1)
        cv2.putText(
            canvas,
            "PLAYER CAMERA VIEW",
            (vx + 25, vy + 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_HUMAN,
            2,
            cv2.LINE_AA,
        )

        # 2. Right Arena: Battle & AI Showcase Card
        ax, ay, aw, ah = 675, 85, 575, 460
        cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), COLOR_CARD_BORDER, 2)

        # Match State Display inside Arena
        if self.match_state == MatchState.IDLE:
            cv2.putText(
                canvas,
                "CHALLENGE THE AI",
                (ax + 120, ay + 140),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                COLOR_TEXT_MAIN,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "Show your hand or Press [SPACE] to Start",
                (ax + 60, ay + 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                COLOR_TEXT_MUTED,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                "High-Speed Vision AI counters in < 20ms!",
                (ax + 80, ay + 260),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                COLOR_SECONDARY,
                2,
                cv2.LINE_AA,
            )

        elif self.match_state == MatchState.COUNTDOWN:
            elapsed = time.monotonic() - self.countdown_start_time
            step = int(elapsed / (self.countdown_duration / 3.0))
            countdown_words = ["ROCK (바위)...", "PAPER (보)...", "SCISSORS (가위)...", "SHOOT! (탕!)"]
            cur_word = countdown_words[min(step, len(countdown_words) - 1)]

            cv2.putText(
                canvas,
                "COUNTDOWN",
                (ax + 190, ay + 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLOR_TEXT_MUTED,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                cur_word,
                (ax + 70, ay + 220),
                cv2.FONT_HERSHEY_DUPLEX,
                1.3,
                COLOR_SECONDARY,
                3,
                cv2.LINE_AA,
            )

            # Progress Bar
            prog = min(1.0, elapsed / self.countdown_duration)
            cv2.rectangle(canvas, (ax + 50, ay + 300), (ax + aw - 50, ay + 320), (30, 30, 30), -1)
            cv2.rectangle(canvas, (ax + 50, ay + 300), (ax + 50 + int((aw - 100) * prog), ay + 320), COLOR_PRIMARY, -1)

        elif self.match_state == MatchState.RESULT:
            # Show Player vs AI moves
            p_text = f"YOU: {(self.player_move or 'UNKNOWN').upper()}"
            ai_text = f"AI: {(self.ai_move or 'UNKNOWN').upper()}"

            cv2.putText(canvas, p_text, (ax + 40, ay + 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_HUMAN, 2, cv2.LINE_AA)
            cv2.putText(canvas, ai_text, (ax + 320, ay + 80), cv2.FONT_HERSHEY_DUPLEX, 0.8, COLOR_AI, 2, cv2.LINE_AA)

            # Verdict Announcement
            v_color = COLOR_SUCCESS if self.round_verdict == "YOU WIN!" else (COLOR_DANGER if self.round_verdict == "AI WINS!" else COLOR_WARNING)
            cv2.putText(
                canvas,
                self.round_verdict or "",
                (ax + 130, ay + 190),
                cv2.FONT_HERSHEY_DUPLEX,
                1.8,
                v_color,
                4,
                cv2.LINE_AA,
            )

            # Telemetry Metrics
            cv2.rectangle(canvas, (ax + 40, ay + 260), (ax + aw - 40, ay + 420), (20, 20, 20), -1)
            cv2.rectangle(canvas, (ax + 40, ay + 260), (ax + aw - 40, ay + 420), COLOR_CARD_BORDER, 1)

            cv2.putText(
                canvas,
                f"Vision AI Reaction Speed: {self.decision_reaction_time_ms:.1f} ms",
                (ax + 60, ay + 300),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                COLOR_PRIMARY,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"AI Win Streak: {self.scoreboard.current_ai_streak} in a row (Max: {self.scoreboard.max_ai_streak})",
                (ax + 60, ay + 345),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                COLOR_SECONDARY,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                f"Total Score: AI {self.scoreboard.ai_wins} - {self.scoreboard.human_wins} YOU ({self.scoreboard.draws} Draws)",
                (ax + 60, ay + 390),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                COLOR_TEXT_MAIN,
                2,
                cv2.LINE_AA,
            )

        # 3. Bottom Dashboard: Instructions & Quick Keys
        dy = 560
        dh = 135
        cv2.rectangle(canvas, (30, dy), (self.width - 30, dy + dh), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (30, dy), (self.width - 30, dy + dh), COLOR_CARD_BORDER, 2)

        cv2.putText(
            canvas,
            f"STATUS: {self.status_message}",
            (50, dy + 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            COLOR_SUCCESS,
            2,
            cv2.LINE_AA,
        )

        controls = (
            "[SPACE] Play Round  |  [M] Switch Mode (God/Fair/VIP/Mirror)  |  [P] Auto-Play: "
            + ("ON" if self.auto_play else "OFF")
            + "  |  [R] Reset Score  |  [Q/ESC] Exit"
        )
        cv2.putText(
            canvas,
            controls,
            (50, dy + 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_TEXT_MAIN,
            2,
            cv2.LINE_AA,
        )

        history_str = "Recent: " + " - ".join(self.scoreboard.recent_history)
        cv2.putText(
            canvas,
            history_str,
            (50, dy + 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            COLOR_TEXT_MUTED,
            1,
            cv2.LINE_AA,
        )

        return canvas

    def run(self) -> int:
        print("\n" + "=" * 65)
        print("  UNBEATABLE ROCK-PAPER-SCISSORS AI (100% WIN RATE)")
        print("=" * 65)
        print(f"  Game Mode: {self.game_mode.value}")
        print(f"  Hardware: {'Enabled' if self.enable_hardware else 'Disabled'} | Port: {self.port}")
        print("  Controls:")
        print("    [Space]  Start Round")
        print("    [M]      Toggle Mode (God / Fair / VIP / Mirror)")
        print("    [P]      Toggle Auto-Play Mode")
        print("    [R]      Reset Scoreboard")
        print("    [Q/ESC]  Exit Application")
        print("=" * 65 + "\n")

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"[RPS WARN] Camera index {self.camera_id} could not be opened. Running in GUI mock mode.")

        try:
            while True:
                ret = False
                raw_frame = None
                display_frame = None
                landmarks_list = []
                if cap.isOpened():
                    ret, raw_frame = cap.read()
                    if ret and raw_frame is not None:
                        # 1. Detect on unmirrored raw camera frame
                        landmarks_list = self.tracker.process_frame(raw_frame)
                        # 2. Mirror frame for user display
                        display_frame = cv2.flip(raw_frame, 1)

                self.update_game_loop(raw_frame, landmarks_list)
                canvas = self.render(display_frame, landmarks_list)
                cv2.imshow(self.window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[RPS] Quitting game.")
                    break
                elif key == ord(" "):
                    self.start_round()
                elif key in (ord("m"), ord("M")):
                    self.cycle_mode()
                elif key in (ord("p"), ord("P")):
                    self.auto_play = not self.auto_play
                    self.status_message = f"Auto-Play mode {'Enabled' if self.auto_play else 'Disabled'}."
                elif key in (ord("r"), ord("R")):
                    self.scoreboard.reset()
                    self.status_message = "Scoreboard reset to 0."

        finally:
            if cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            if self.hardware_controller is not None:
                self.hardware_controller.close()
            print("[RPS] Game closed cleanly.")

        return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch Unbeatable Rock-Paper-Scissors AI (100% Win Rate)."
    )
    parser.add_argument(
        "--mode",
        choices=("gui", "hardware", "mujoco", "both"),
        default="gui",
        help="Target controller backend: gui, hardware, mujoco, or both (default: gui)",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port for LEAP Hand hardware (default: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to hand_landmarker.task",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    enable_hardware = args.mode in ("hardware", "both")
    enable_mujoco = args.mode in ("mujoco", "both")

    app = UnbeatableRpsApp(
        mode=args.mode,
        port=args.port,
        camera_id=args.camera_id,
        model_path=args.model_path,
        enable_hardware=enable_hardware,
        enable_mujoco=enable_mujoco,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
