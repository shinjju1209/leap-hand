"""LEAP Hand Interactive Booth Kiosk Application.

A touchscreen & keyboard-friendly exhibition kiosk UI supporting:
1. Real-time Teleoperation (with C / F calibration and A / D arming).
2. Interactive Rock-Paper-Scissors Game (with 3-2-1 countdown, robot random move, gesture detection, and live scoreboard).
3. Gesture & Showcase Demo.
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
from typing import Any, Callable

import cv2
import numpy as np

from hand_angles import (
    ANGLE_NAMES,
    calculate_leap_control_angles,
)
from leap_hand_hardware_controller import LeapHandHardwareController
from mujoco_hand_controller import MujocoHandController
from neutral_calibration import NeutralCalibration
from one_euro_filter import OneEuroFilter
from rps.gesture import classify_rps_gesture
from rps.moves import MOVE_NAMES
from rps.postures import get_posture
from rps.rounds import human_result
from webcam_hand_tracking import (
    HAND_CONNECTIONS,
    draw_hand,
)

DEFAULT_CALIB_PATH = Path("calibration/neutral_calibration.json")
DEFAULT_MODEL_PATH = Path("models/hand_landmarker.task")

# Colors in BGR format (Modern Dark Cyberpunk theme)
COLOR_BG = (22, 18, 16)           # Deep charcoal navy
COLOR_CARD_BG = (35, 28, 25)      # Slightly lighter card background
COLOR_CARD_BORDER = (60, 50, 45)  # Card outline
COLOR_TEXT_MAIN = (245, 245, 245) # Bright white
COLOR_TEXT_MUTED = (160, 160, 160)# Slate gray
COLOR_PRIMARY = (255, 180, 0)     # Neon Cyan / Sky Blue in BGR
COLOR_SECONDARY = (0, 220, 255)   # Amber Gold in BGR
COLOR_SUCCESS = (100, 230, 0)     # Emerald Green in BGR
COLOR_DANGER = (50, 50, 245)      # Coral Red in BGR
COLOR_WARNING = (0, 165, 255)     # Orange in BGR
COLOR_ROBOT = (255, 100, 150)     # Purple/Magenta in BGR
COLOR_HUMAN = (0, 200, 255)       # Yellow/Gold in BGR


def get_hand_posture(name: str) -> np.ndarray:
    """Return 16-element joint posture degrees."""
    n = name.lower()
    if n in ("neutral", "paper"):
        return get_posture("paper").copy()
    if n == "rock":
        return get_posture("rock").copy()
    if n == "scissors":
        return get_posture("scissors").copy()
    if n in ("thumbs_up", "thumb_up"):
        # Fist with thumb pointing up
        deg = get_posture("rock").copy()
        deg[12] = 0.0  # thumb_cmc_side
        deg[13] = 0.0  # thumb_cmc_flex
        deg[14] = 0.0  # thumb_mcp_flex
        deg[15] = 0.0  # thumb_ip_flex
        return deg
    if n in ("ok_sign", "ok"):
        # Index & Thumb pinch, Middle/Ring extended
        deg = np.zeros(16, dtype=np.float64)
        deg[0] = 0.0   # index_mcp_side
        deg[1] = 45.0  # index_mcp_flex
        deg[2] = 65.0  # index_pip_flex
        deg[3] = 45.0  # index_dip_flex
        deg[12] = 20.0 # thumb_cmc_side
        deg[13] = 45.0 # thumb_cmc_flex
        deg[14] = 40.0 # thumb_mcp_flex
        deg[15] = 20.0 # thumb_ip_flex
        return deg
    if n == "pointing":
        # Index extended, others closed
        deg = get_posture("rock").copy()
        deg[0] = 0.0
        deg[1] = 0.0
        deg[2] = 0.0
        deg[3] = 0.0
        return deg
    if n in ("rock_on", "rockon", "love"):
        # Index & Ring & Thumb extended, Middle closed
        deg = np.zeros(16, dtype=np.float64)
        deg[4] = 0.0   # middle_mcp_side
        deg[5] = 75.0  # middle_mcp_flex
        deg[6] = 85.0  # middle_pip_flex
        deg[7] = 65.0  # middle_dip_flex
        return deg
    return get_posture(name).copy()


class MediaPipeTracker:
    """Wrapper around MediaPipe HandLandmarker for video frame inference."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        self.model_path = model_path
        self.landmarker = None
        self.last_timestamp_ms = 0
        if model_path.is_file():
            try:
                import mediapipe as mp

                options = mp.tasks.vision.HandLandmarkerOptions(
                    base_options=mp.tasks.BaseOptions(
                        model_asset_path=str(model_path)
                    ),
                    running_mode=mp.tasks.vision.RunningMode.VIDEO,
                    num_hands=1,
                    min_hand_detection_confidence=0.5,
                    min_hand_presence_confidence=0.5,
                    min_tracking_confidence=0.5,
                )
                self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
                    options
                )
            except Exception as e:
                print(f"[BOOTH WARN] Failed to load MediaPipe HandLandmarker: {e}")

    def process_frame(self, bgr_frame: np.ndarray | None) -> list[Any]:
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


class AppScreen(Enum):
    HOME = auto()
    TELEOP = auto()
    RPS = auto()
    SHOWCASE = auto()


class RpsState(Enum):
    IDLE = auto()
    COUNTDOWN = auto()
    SHOOT = auto()
    RESULT = auto()


@dataclass
class Button:
    id: str
    label: str
    rect: tuple[int, int, int, int]  # (x, y, w, h)
    shortcut: str = ""
    icon: str = ""
    bg_color: tuple[int, int, int] = COLOR_CARD_BG
    hover_color: tuple[int, int, int] = (60, 48, 40)
    text_color: tuple[int, int, int] = COLOR_TEXT_MAIN
    border_color: tuple[int, int, int] = COLOR_CARD_BORDER
    enabled: bool = True
    active: bool = False

    def is_inside(self, px: int, py: int) -> bool:
        x, y, w, h = self.rect
        return x <= px <= x + w and y <= py <= y + h

    def draw(self, canvas: np.ndarray, mouse_pos: tuple[int, int]) -> None:
        x, y, w, h = self.rect
        hovered = self.is_inside(*mouse_pos) and self.enabled

        fill_color = (
            self.hover_color
            if hovered
            else (self.bg_color if not self.active else COLOR_PRIMARY)
        )
        if not self.enabled:
            fill_color = (25, 20, 20)

        # Rectangle background
        cv2.rectangle(canvas, (x, y), (x + w, y + h), fill_color, -1)
        border_col = (
            COLOR_PRIMARY
            if (hovered or self.active)
            else (self.border_color if self.enabled else (40, 35, 35))
        )
        cv2.rectangle(canvas, (x, y), (x + w, y + h), border_col, 2)

        # Text and shortcut
        text = self.label
        if self.shortcut:
            text = f"[{self.shortcut}] {text}"
        txt_col = self.text_color if self.enabled else (90, 85, 85)
        if self.active:
            txt_col = (10, 10, 10)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        tx = x + max(10, (w - tw) // 2)
        ty = y + (h + th) // 2
        cv2.putText(
            canvas,
            text,
            (tx, ty),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            txt_col,
            2,
            cv2.LINE_AA,
        )


@dataclass
class RpsScoreboard:
    total_rounds: int = 0
    human_wins: int = 0
    robot_wins: int = 0
    ties: int = 0
    history: deque = field(default_factory=lambda: deque(maxlen=8))

    def record_round(self, human_move: str, robot_move: str) -> str:
        verdict = human_result(human_move, robot_move)
        self.total_rounds += 1
        if verdict == "win":
            self.human_wins += 1
        elif verdict == "loss":
            self.robot_wins += 1
        else:
            self.ties += 1

        self.history.append((self.total_rounds, human_move, robot_move, verdict))
        return verdict

    def reset(self) -> None:
        self.total_rounds = 0
        self.human_wins = 0
        self.robot_wins = 0
        self.ties = 0
        self.history.clear()


class BoothKioskApp:
    """Main interactive booth application controller and renderer."""

    def __init__(
        self,
        *,
        mode: str = "hardware",
        port: str = "/dev/ttyUSB0",
        profile: str = "jiwoo",
        calib_path: Path = DEFAULT_CALIB_PATH,
        model_path: Path = DEFAULT_MODEL_PATH,
        camera_id: int = 0,
        current_limit: int = 350,
        max_joint_speed: float = 350.0,
        max_tracking_error: float = 50.0,
        enable_mujoco: bool = True,
        enable_hardware: bool = False,
        width: int = 1280,
        height: int = 720,
    ) -> None:
        self.mode = mode
        self.port = port
        self.profile = profile
        self.camera_id = camera_id
        self.current_limit = current_limit
        self.max_joint_speed = max_joint_speed
        self.max_tracking_error = max_tracking_error
        self.enable_mujoco = enable_mujoco
        self.enable_hardware = enable_hardware
        self.width = width
        self.height = height

        self.current_screen = AppScreen.HOME
        self.mouse_pos = (0, 0)
        self.window_name = "LEAP Hand Interactive Booth Kiosk"

        # Teleop Controllers & Calibration
        self.tracker = MediaPipeTracker(model_path=model_path)
        self.calibrator = NeutralCalibration(path=calib_path, profile=profile)
        self.filter = OneEuroFilter()
        self.hardware_controller: LeapHandHardwareController | None = None
        self.mujoco_controller: MujocoHandController | None = None

        # Teleop State
        self.armed = False
        self.calib_open_in_progress = False
        self.calib_fist_in_progress = False

        # RPS Game State
        self.rps_state = RpsState.IDLE
        self.rps_scoreboard = RpsScoreboard()
        self.countdown_start_time = 0.0
        self.countdown_duration = 3.0  # 3 seconds countdown
        self.shoot_time = 0.0
        self.shoot_duration = 2.5      # Hold verdict for 2.5 seconds
        self.auto_play = False
        self.robot_chosen_move: str | None = None
        self.human_detected_move: str | None = None
        self.last_round_verdict: str | None = None
        self.consecutive_detected_gesture: str | None = None
        self.gesture_streak_count = 0

        # Showcase Dynamic Animation State
        self.active_animation: str | None = None
        self.animation_start_time: float = 0.0

        # Status & Telemetry
        self.fps = 0.0
        self.last_frame_time = time.monotonic()
        self.current_joint_angles = np.zeros(16, dtype=np.float64)
        self.target_joint_angles = np.zeros(16, dtype=np.float64)
        self.status_message = "Ready. Welcome to the LEAP Hand Booth!"
        self.status_color = COLOR_SUCCESS

        # Initialize buttons
        self.buttons: dict[AppScreen, list[Button]] = {
            AppScreen.HOME: [],
            AppScreen.TELEOP: [],
            AppScreen.RPS: [],
            AppScreen.SHOWCASE: [],
        }
        self._init_buttons()

        # Connect hardware / MuJoCo if configured
        self._init_controllers()

    def _init_buttons(self) -> None:
        # HOME SCREEN BUTTONS
        card_w, card_h = 360, 380
        gap = 40
        start_x = (self.width - (3 * card_w + 2 * gap)) // 2
        y_pos = 180

        self.buttons[AppScreen.HOME] = [
            Button(
                id="goto_teleop",
                label="Teleoperation (1:1 Tracking)",
                shortcut="1",
                rect=(start_x, y_pos, card_w, card_h),
                bg_color=(45, 35, 30),
                hover_color=(75, 55, 40),
            ),
            Button(
                id="goto_rps",
                label="Rock-Paper-Scissors Game",
                shortcut="2",
                rect=(start_x + card_w + gap, y_pos, card_w, card_h),
                bg_color=(45, 35, 30),
                hover_color=(75, 55, 40),
            ),
            Button(
                id="goto_showcase",
                label="Gesture Showcase",
                shortcut="3",
                rect=(start_x + 2 * (card_w + gap), y_pos, card_w, card_h),
                bg_color=(45, 35, 30),
                hover_color=(75, 55, 40),
            ),
            Button(
                id="quit_app",
                label="Exit (Quit)",
                shortcut="Q",
                rect=(self.width - 160, 20, 130, 35),
                bg_color=(50, 20, 20),
                hover_color=(80, 30, 30),
                text_color=COLOR_DANGER,
            ),
        ]

        # TELEOP SCREEN BUTTONS
        bx, by, bw, bh = 880, 100, 370, 48
        b_gap = 12
        self.buttons[AppScreen.TELEOP] = [
            Button(
                id="calib_open",
                label="Open Hand Calibration",
                shortcut="C",
                rect=(bx, by, bw, bh),
                bg_color=COLOR_CARD_BG,
                hover_color=(60, 50, 45),
            ),
            Button(
                id="calib_fist",
                label="Closed Fist Calibration",
                shortcut="F",
                rect=(bx, by + (bh + b_gap), bw, bh),
                bg_color=COLOR_CARD_BG,
                hover_color=(60, 50, 45),
            ),
            Button(
                id="toggle_arm",
                label="ARM / Follow Hand",
                shortcut="A",
                rect=(bx, by + 2 * (bh + b_gap), bw, bh),
                bg_color=(20, 60, 20),
                hover_color=(30, 90, 30),
                text_color=COLOR_SUCCESS,
            ),
            Button(
                id="disarm",
                label="DISARM / Pause Torque",
                shortcut="D",
                rect=(bx, by + 3 * (bh + b_gap), bw, bh),
                bg_color=(60, 20, 20),
                hover_color=(90, 30, 30),
                text_color=COLOR_DANGER,
            ),
            Button(
                id="reset_calib",
                label="Reset Calibration Profile",
                shortcut="R",
                rect=(bx, by + 4 * (bh + b_gap), bw, bh),
                bg_color=COLOR_CARD_BG,
                hover_color=(60, 50, 45),
            ),
            Button(
                id="back_home_teleop",
                label="Return to Main Menu",
                shortcut="H",
                rect=(bx, by + 5 * (bh + b_gap) + 15, bw, bh),
                bg_color=(40, 30, 25),
                hover_color=(70, 50, 40),
            ),
        ]

        # RPS SCREEN BUTTONS
        rx, ry, rw, rh = 880, 100, 370, 52
        self.buttons[AppScreen.RPS] = [
            Button(
                id="rps_start",
                label="START RPS MATCH",
                shortcut="Space",
                rect=(rx, ry, rw, rh),
                bg_color=(20, 65, 30),
                hover_color=(30, 95, 45),
                text_color=COLOR_SUCCESS,
            ),
            Button(
                id="rps_auto_toggle",
                label="Auto-Play Mode: OFF",
                shortcut="P",
                rect=(rx, ry + rh + 12, rw, rh),
                bg_color=COLOR_CARD_BG,
                hover_color=(60, 50, 45),
            ),
            Button(
                id="rps_reset_score",
                label="Reset Scoreboard",
                shortcut="R",
                rect=(rx, ry + 2 * (rh + 12), rw, rh),
                bg_color=COLOR_CARD_BG,
                hover_color=(60, 50, 45),
            ),
            Button(
                id="back_home_rps",
                label="Return to Main Menu",
                shortcut="H",
                rect=(rx, ry + 3 * (rh + 12) + 50, rw, rh),
                bg_color=(40, 30, 25),
                hover_color=(70, 50, 40),
            ),
        ]

        # SHOWCASE SCREEN BUTTONS (2-Column Grid + Dynamic Animations)
        c1_x, c2_x = 815, 1035
        bw, bh = 210, 42
        gap_y = 8
        sy = 90

        self.buttons[AppScreen.SHOWCASE] = [
            # Column 1: Classic & Basic
            Button(
                id="showcase_rock",
                label="Rock (바위)",
                shortcut="1",
                rect=(c1_x, sy, bw, bh),
            ),
            Button(
                id="showcase_paper",
                label="Paper (보)",
                shortcut="2",
                rect=(c1_x, sy + (bh + gap_y), bw, bh),
            ),
            Button(
                id="showcase_scissors",
                label="Scissors (가위)",
                shortcut="3",
                rect=(c1_x, sy + 2 * (bh + gap_y), bw, bh),
            ),
            Button(
                id="showcase_neutral",
                label="Neutral (편 손)",
                shortcut="4",
                rect=(c1_x, sy + 3 * (bh + gap_y), bw, bh),
            ),
            Button(
                id="showcase_middle",
                label="Middle (중지)",
                shortcut="5",
                rect=(c1_x, sy + 4 * (bh + gap_y), bw, bh),
                text_color=COLOR_WARNING,
            ),
            # Column 2: Expressive Gestures & Wave
            Button(
                id="showcase_thumbs_up",
                label="Thumbs Up (엄지척)",
                shortcut="6",
                rect=(c2_x, sy, bw, bh),
                text_color=COLOR_SUCCESS,
            ),
            Button(
                id="showcase_ok",
                label="OK Sign (OK사인)",
                shortcut="7",
                rect=(c2_x, sy + (bh + gap_y), bw, bh),
                text_color=COLOR_PRIMARY,
            ),
            Button(
                id="showcase_pointing",
                label="Pointing (가리키기)",
                shortcut="8",
                rect=(c2_x, sy + 2 * (bh + gap_y), bw, bh),
                text_color=COLOR_SECONDARY,
            ),
            Button(
                id="showcase_rock_on",
                label="Rock On (락앤롤)",
                shortcut="9",
                rect=(c2_x, sy + 3 * (bh + gap_y), bw, bh),
                text_color=COLOR_ROBOT,
            ),
            Button(
                id="showcase_finger_wave",
                label="Wave (웨이브)",
                shortcut="W",
                rect=(c2_x, sy + 4 * (bh + gap_y), bw, bh),
                bg_color=(35, 45, 55),
                hover_color=(50, 65, 80),
                text_color=COLOR_PRIMARY,
            ),
            # Dynamic Animation: Wave Hello
            Button(
                id="showcase_wave_hello",
                label="Wave Hello (손 인사 애니메이션)",
                shortcut="V",
                rect=(c1_x, sy + 5 * (bh + gap_y) + 4, 430, bh),
                bg_color=(30, 50, 35),
                hover_color=(45, 75, 50),
                text_color=COLOR_SUCCESS,
            ),
            # Return Home
            Button(
                id="back_home_showcase",
                label="Return to Main Menu (메인 메뉴)",
                shortcut="H",
                rect=(c1_x, sy + 6 * (bh + gap_y) + 20, 430, 46),
                bg_color=(40, 30, 25),
                hover_color=(70, 50, 40),
            ),
        ]

    def _init_controllers(self) -> None:
        """Initialize Hardware and MuJoCo simulation backend controllers."""
        if self.enable_mujoco:
            try:
                self.mujoco_controller = MujocoHandController()
                self.mujoco_controller.launch_viewer()
                self.status_message = "MuJoCo 3D simulation connected."
            except Exception as e:
                print(f"[BOOTH WARN] MuJoCo simulation launch skipped: {e}")
                self.mujoco_controller = None

        if self.enable_hardware:
            try:
                self.hardware_controller = LeapHandHardwareController(
                    port=self.port,
                    current_limit_milliamps=self.current_limit,
                )
                self.hardware_controller.connect()
                self.hardware_controller.configure()
                self.status_message = f"LEAP Hand connected on {self.port}."
            except Exception as e:
                print(f"[BOOTH WARN] Hardware connection skipped: {e}")
                self.hardware_controller = None

    def send_robot_posture_degrees(self, posture_degrees: Sequence[float]) -> None:
        """Set target angles for smooth interpolation towards the target posture."""
        angles = np.asarray(posture_degrees, dtype=np.float64)
        self.target_joint_angles = angles.copy()

    def step_smooth_control(self, dt: float = 0.02) -> None:
        """Smoothly interpolate current joint angles towards target angles with speed limiting."""
        if self.current_screen == AppScreen.TELEOP and self.armed:
            self.current_joint_angles = self.target_joint_angles.copy()
        else:
            # Dynamic Animation Update
            if self.current_screen == AppScreen.SHOWCASE and self.active_animation:
                t = time.monotonic() - self.animation_start_time
                if self.active_animation == "finger_wave":
                    wave_pose = np.zeros(16, dtype=np.float64)
                    # Index wave
                    wave_pose[1] = 40.0 + 35.0 * math.sin(4.0 * t)
                    wave_pose[2] = 45.0 + 40.0 * math.sin(4.0 * t)
                    wave_pose[3] = 35.0 + 30.0 * math.sin(4.0 * t)
                    # Middle wave (phase offset -1.0)
                    wave_pose[5] = 40.0 + 35.0 * math.sin(4.0 * t - 1.0)
                    wave_pose[6] = 45.0 + 40.0 * math.sin(4.0 * t - 1.0)
                    wave_pose[7] = 35.0 + 30.0 * math.sin(4.0 * t - 1.0)
                    # Ring wave (phase offset -2.0)
                    wave_pose[9] = 40.0 + 35.0 * math.sin(4.0 * t - 2.0)
                    wave_pose[10] = 45.0 + 40.0 * math.sin(4.0 * t - 2.0)
                    wave_pose[11] = 35.0 + 30.0 * math.sin(4.0 * t - 2.0)
                    # Thumb wave (phase offset -3.0)
                    wave_pose[13] = 30.0 + 25.0 * math.sin(4.0 * t - 3.0)
                    wave_pose[14] = 35.0 + 30.0 * math.sin(4.0 * t - 3.0)
                    wave_pose[15] = 25.0 + 20.0 * math.sin(4.0 * t - 3.0)
                    self.target_joint_angles = wave_pose
                elif self.active_animation == "wave_hello":
                    hello_pose = np.zeros(16, dtype=np.float64)
                    side_val = 18.0 * math.sin(5.0 * t)
                    hello_pose[0] = side_val
                    hello_pose[4] = side_val
                    hello_pose[8] = side_val
                    hello_pose[12] = 10.0 * math.sin(5.0 * t)
                    flex_val = 20.0 + 15.0 * math.sin(3.0 * t)
                    hello_pose[1] = flex_val
                    hello_pose[2] = flex_val
                    hello_pose[5] = flex_val
                    hello_pose[6] = flex_val
                    hello_pose[9] = flex_val
                    hello_pose[10] = flex_val
                    self.target_joint_angles = hello_pose

            # Smooth trajectory interpolation for Showcases and RPS transitions (~350 deg/s)
            max_step = self.max_joint_speed * max(0.001, min(dt, 0.05))
            diff = self.target_joint_angles - self.current_joint_angles
            step = np.clip(diff, -max_step, max_step)
            self.current_joint_angles += step

        if self.mujoco_controller is not None:
            self.mujoco_controller.set_target_degrees(self.current_joint_angles)
            self.mujoco_controller.step_for(dt)
            self.mujoco_controller.sync_viewer()

        if self.hardware_controller is not None and self.hardware_controller.torque_enabled:
            try:
                self.hardware_controller.command_degrees(self.current_joint_angles)
            except Exception as e:
                print(f"[HARDWARE ERROR] Command failed: {e}")

    def on_mouse_event(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Handle mouse movement and clicks."""
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self._handle_click(x, y)

    def _handle_click(self, x: int, y: int) -> None:
        for btn in self.buttons.get(self.current_screen, []):
            if btn.is_inside(x, y) and btn.enabled:
                self.handle_action(btn.id)
                break

    def handle_action(self, action_id: str) -> None:
        """Execute logic triggered by button click or keyboard shortcut."""
        # Navigation
        if action_id == "goto_teleop":
            self.current_screen = AppScreen.TELEOP
            self.status_message = "Teleoperation: [C] Open Calib, [F] Fist Calib, [A] Arm"
            self.status_color = COLOR_PRIMARY
        elif action_id == "goto_rps":
            self.current_screen = AppScreen.RPS
            self.rps_state = RpsState.IDLE
            self.status_message = "RPS Match: Press [Space] to start countdown!"
            self.status_color = COLOR_SECONDARY
        elif action_id == "goto_showcase":
            self.current_screen = AppScreen.SHOWCASE
            self.status_message = "Showcase: Click buttons to test postures."
            self.status_color = COLOR_SUCCESS
        elif action_id in ("back_home_teleop", "back_home_rps", "back_home_showcase"):
            self.current_screen = AppScreen.HOME
            self.disarm_robot()
            self.status_message = "Main Menu. Select a mode to begin."
            self.status_color = COLOR_TEXT_MAIN

        # Teleop Actions
        elif action_id == "calib_open":
            self.calib_open_in_progress = True
            self.calib_fist_in_progress = False
            self.calibrator.start("Right", time.perf_counter(), "neutral")
            self.status_message = "Capturing Open Hand pose for 3 seconds..."
            self.status_color = COLOR_WARNING
        elif action_id == "calib_fist":
            self.calib_fist_in_progress = True
            self.calib_open_in_progress = False
            try:
                self.calibrator.start("Right", time.perf_counter(), "closed")
                self.status_message = "Capturing Closed Fist pose for 3 seconds..."
                self.status_color = COLOR_WARNING
            except ValueError as e:
                self.status_message = f"Error: {e}"
                self.status_color = COLOR_DANGER
                self.calib_fist_in_progress = False
        elif action_id == "toggle_arm":
            self.arm_robot()
        elif action_id == "disarm":
            self.disarm_robot()
        elif action_id == "reset_calib":
            self.calibrator.reset("Right")
            self.status_message = "Calibration profile reset to default."
            self.status_color = COLOR_SUCCESS

        # RPS Game Actions
        elif action_id == "rps_start":
            self.start_rps_countdown()
        elif action_id == "rps_auto_toggle":
            self.auto_play = not self.auto_play
            for btn in self.buttons[AppScreen.RPS]:
                if btn.id == "rps_auto_toggle":
                    btn.label = f"Auto-Play Mode: {'ON' if self.auto_play else 'OFF'}"
                    btn.active = self.auto_play
            self.status_message = f"Auto-play mode {'enabled' if self.auto_play else 'disabled'}."
        elif action_id == "rps_reset_score":
            self.rps_scoreboard.reset()
            self.status_message = "Scoreboard reset to 0."

        # Showcase Actions
        elif action_id == "showcase_rock":
            self.play_showcase_posture("rock", "Rock Gesture (바위)")
        elif action_id == "showcase_paper":
            self.play_showcase_posture("paper", "Paper Gesture (보)")
        elif action_id == "showcase_scissors":
            self.play_showcase_posture("scissors", "Scissors Gesture (가위)")
        elif action_id == "showcase_neutral":
            self.play_showcase_posture("neutral", "Open Neutral (편 손)")
        elif action_id == "showcase_middle":
            self.play_showcase_middle_finger()
        elif action_id == "showcase_thumbs_up":
            self.play_showcase_posture("thumbs_up", "Thumbs Up (엄지 척)")
        elif action_id == "showcase_ok":
            self.play_showcase_posture("ok_sign", "OK Sign (OK 사인)")
        elif action_id == "showcase_pointing":
            self.play_showcase_posture("pointing", "Pointing (가리키기)")
        elif action_id == "showcase_rock_on":
            self.play_showcase_posture("rock_on", "Rock On (락앤롤)")
        elif action_id == "showcase_finger_wave":
            self.play_showcase_animation("finger_wave", "Finger Wave (파도타기 애니메이션)")
        elif action_id == "showcase_wave_hello":
            self.play_showcase_animation("wave_hello", "Wave Hello (손 인사 애니메이션)")

    def arm_robot(self) -> None:
        """Enable hardware torque and activate teleoperation tracking."""
        self.armed = True
        if self.hardware_controller is not None:
            try:
                self.hardware_controller.enable_torque()
            except Exception as e:
                print(f"[HARDWARE ERROR] Failed to enable torque: {e}")
        for btn in self.buttons[AppScreen.TELEOP]:
            if btn.id == "toggle_arm":
                btn.active = True
                btn.label = "ARMED (Tracking Active)"
        self.status_message = "Robot ARMED! Hand movements will now be followed."
        self.status_color = COLOR_SUCCESS

    def disarm_robot(self) -> None:
        """Disable torque and safely park robot hand."""
        self.armed = False
        if self.hardware_controller is not None:
            try:
                self.hardware_controller.emergency_stop()
            except Exception as e:
                print(f"[HARDWARE ERROR] Emergency stop: {e}")
        for btn in self.buttons[AppScreen.TELEOP]:
            if btn.id == "toggle_arm":
                btn.active = False
                btn.label = "ARM / Follow Hand"
        self.status_message = "Robot DISARMED (Holding / Torque Paused)."
        self.status_color = COLOR_WARNING

    def start_rps_countdown(self) -> None:
        """Begin a 3-2-1 Rock-Paper-Scissors match."""
        if self.hardware_controller is not None and not self.hardware_controller.torque_enabled:
            try:
                self.hardware_controller.enable_torque()
            except Exception:
                pass

        self.send_robot_posture_degrees(get_hand_posture("neutral"))
        self.rps_state = RpsState.COUNTDOWN
        self.countdown_start_time = time.monotonic()
        self.robot_chosen_move = None
        self.human_detected_move = None
        self.last_round_verdict = None
        self.status_message = "Rock... Paper... Scissors... Shoot!"
        self.status_color = COLOR_SECONDARY

    def play_showcase_posture(self, posture_name: str, display_name: str) -> None:
        self.active_animation = None
        if self.hardware_controller is not None and not self.hardware_controller.torque_enabled:
            self.hardware_controller.enable_torque()
        deg = get_hand_posture(posture_name)
        self.send_robot_posture_degrees(deg)
        self.status_message = f"Postured: {display_name}"
        self.status_color = COLOR_SUCCESS

    def play_showcase_middle_finger(self) -> None:
        self.active_animation = None
        if self.hardware_controller is not None and not self.hardware_controller.torque_enabled:
            self.hardware_controller.enable_torque()
        deg = get_hand_posture("rock").copy()
        deg[4] = 0.0  # mf_mcp_side
        deg[5] = 0.0  # mf_mcp_flex
        deg[6] = 0.0  # mf_pip_flex
        deg[7] = 0.0  # mf_dip_flex
        self.send_robot_posture_degrees(deg)
        self.status_message = "Postured: Middle Finger Extension (중지)"
        self.status_color = COLOR_WARNING

    def play_showcase_animation(self, anim_name: str, display_name: str) -> None:
        if self.hardware_controller is not None and not self.hardware_controller.torque_enabled:
            self.hardware_controller.enable_torque()
        self.active_animation = anim_name
        self.animation_start_time = time.monotonic()
        self.status_message = f"Animation Running: {display_name}"
        self.status_color = COLOR_PRIMARY

    def update_teleop_frame(self, frame: np.ndarray, landmarks_list: list[Any]) -> None:
        """Process teleoperation hand tracking, calibration, and joint commanding."""
        if not landmarks_list:
            return

        landmarks = landmarks_list[0]
        raw_angles = calculate_leap_control_angles(landmarks)

        # Calibration collection
        if self.calib_open_in_progress or self.calib_fist_in_progress:
            now_perf = time.perf_counter()
            completed = self.calibrator.add_sample("Right", raw_angles, now_perf)
            if completed:
                if self.calib_open_in_progress:
                    self.calib_open_in_progress = False
                    self.status_message = "Open Hand Calibration Complete!"
                else:
                    self.calib_fist_in_progress = False
                    self.status_message = "Closed Fist Calibration Complete!"
                self.status_color = COLOR_SUCCESS

        # Joint angle calculation & calibration
        calibrated_angles = self.calibrator.apply("Right", raw_angles)
        smoothed_angles = self.filter.filter(calibrated_angles, time.monotonic())

        self.current_joint_angles = smoothed_angles.copy()

        if self.armed:
            self.send_robot_posture_degrees(smoothed_angles)

    def update_rps_frame(self, frame: np.ndarray, landmarks_list: list[Any]) -> None:
        """Process Rock-Paper-Scissors game loop, gesture detection, and robot moves."""
        now = time.monotonic()

        current_gesture = None
        if landmarks_list:
            classification = classify_rps_gesture(landmarks_list[0])
            current_gesture = classification.label

        if current_gesture == self.consecutive_detected_gesture:
            self.gesture_streak_count += 1
        else:
            self.consecutive_detected_gesture = current_gesture
            self.gesture_streak_count = 1

        if self.gesture_streak_count >= 3:
            self.human_detected_move = current_gesture

        # State Machine
        if self.rps_state == RpsState.COUNTDOWN:
            elapsed = now - self.countdown_start_time
            shake_amp = math.sin(elapsed * 12.0) * 15.0
            shake_pose = get_hand_posture("neutral").copy()
            shake_pose[1] += shake_amp
            shake_pose[5] += shake_amp
            shake_pose[9] += shake_amp
            self.send_robot_posture_degrees(shake_pose)

            if elapsed >= self.countdown_duration:
                self.robot_chosen_move = random.choice(list(MOVE_NAMES))
                robot_posture = get_hand_posture(self.robot_chosen_move)
                self.send_robot_posture_degrees(robot_posture)

                self.rps_state = RpsState.SHOOT
                self.shoot_time = now

        elif self.rps_state == RpsState.SHOOT:
            if now - self.shoot_time >= 0.5:
                if self.human_detected_move in MOVE_NAMES and self.robot_chosen_move is not None:
                    verdict = self.rps_scoreboard.record_round(
                        self.human_detected_move,
                        self.robot_chosen_move,
                    )
                    self.last_round_verdict = verdict
                    self.rps_state = RpsState.RESULT
                    self.shoot_time = now
                elif now - self.shoot_time >= self.shoot_duration:
                    self.last_round_verdict = "no_hand"
                    self.rps_state = RpsState.RESULT
                    self.shoot_time = now

        elif self.rps_state == RpsState.RESULT:
            if now - self.shoot_time >= self.shoot_duration:
                if self.auto_play:
                    self.start_rps_countdown()
                else:
                    self.rps_state = RpsState.IDLE
                    self.send_robot_posture_degrees(get_hand_posture("neutral"))

    def render(self, camera_frame: np.ndarray | None, landmarks_list: list[Any]) -> np.ndarray:
        """Render the complete Kiosk UI canvas."""
        canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        canvas[:] = COLOR_BG

        now = time.monotonic()
        dt = now - self.last_frame_time
        if dt > 0:
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)
        self.last_frame_time = now

        self._draw_header(canvas)

        if self.current_screen == AppScreen.HOME:
            self._render_home_screen(canvas)
        elif self.current_screen == AppScreen.TELEOP:
            self._render_teleop_screen(canvas, camera_frame, landmarks_list)
        elif self.current_screen == AppScreen.RPS:
            self._render_rps_screen(canvas, camera_frame, landmarks_list)
        elif self.current_screen == AppScreen.SHOWCASE:
            self._render_showcase_screen(canvas, camera_frame)

        for btn in self.buttons.get(self.current_screen, []):
            btn.draw(canvas, self.mouse_pos)

        self._draw_footer(canvas)
        return canvas

    def _draw_header(self, canvas: np.ndarray) -> None:
        cv2.rectangle(canvas, (0, 0), (self.width, 65), (28, 22, 20), -1)
        cv2.line(canvas, (0, 65), (self.width, 65), COLOR_CARD_BORDER, 2)

        cv2.putText(
            canvas,
            "LEAP HAND INTERACTIVE BOOTH",
            (25, 42),
            cv2.FONT_HERSHEY_DUPLEX,
            0.8,
            COLOR_TEXT_MAIN,
            2,
            cv2.LINE_AA,
        )

        mode_str = f"MODE: {self.mode.upper()}"
        cv2.rectangle(canvas, (480, 16), (620, 48), (45, 35, 30), -1)
        cv2.putText(canvas, mode_str, (490, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_PRIMARY, 1, cv2.LINE_AA)

        profile_str = f"PROFILE: {self.profile}"
        cv2.rectangle(canvas, (630, 16), (790, 48), (45, 35, 30), -1)
        cv2.putText(canvas, profile_str, (640, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_SECONDARY, 1, cv2.LINE_AA)

        fps_str = f"FPS: {self.fps:.1f}"
        cv2.rectangle(canvas, (800, 16), (900, 48), (45, 35, 30), -1)
        cv2.putText(canvas, fps_str, (810, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_footer(self, canvas: np.ndarray) -> None:
        y = self.height - 45
        cv2.rectangle(canvas, (0, y), (self.width, self.height), (25, 20, 18), -1)
        cv2.line(canvas, (0, y), (self.width, y), COLOR_CARD_BORDER, 1)

        cv2.circle(canvas, (25, y + 22), 6, self.status_color, -1)
        cv2.putText(
            canvas,
            self.status_message,
            (40, y + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_TEXT_MAIN,
            1,
            cv2.LINE_AA,
        )

    def _render_home_screen(self, canvas: np.ndarray) -> None:
        cv2.putText(
            canvas,
            "Select an interactive mode below to begin:",
            (self.width // 2 - 210, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            COLOR_TEXT_MUTED,
            2,
            cv2.LINE_AA,
        )

        card_w, card_h = 360, 380
        gap = 40
        start_x = (self.width - (3 * card_w + 2 * gap)) // 2
        y_pos = 180

        # Card 1
        cv2.putText(canvas, "1:1 Live Hand Tracking", (start_x + 30, y_pos + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_PRIMARY, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Replicate user hand motion", (start_x + 30, y_pos + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "in real time across 16 joints.", (start_x + 30, y_pos + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "- Open / Fist Calibration", (start_x + 30, y_pos + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, "- OneEuro Jitter Filtering", (start_x + 30, y_pos + 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, "- Real-time Arm / Disarm", (start_x + 30, y_pos + 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Card 2
        cx2 = start_x + card_w + gap
        cv2.putText(canvas, "Rock-Paper-Scissors", (cx2 + 30, y_pos + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_SECONDARY, 2, cv2.LINE_AA)
        cv2.putText(canvas, "Battle the robot in real time!", (cx2 + 30, y_pos + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "3-2-1 countdown & camera vision.", (cx2 + 30, y_pos + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "- Auto Gesture Recognition", (cx2 + 30, y_pos + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, "- Real-time Winner Verdict", (cx2 + 30, y_pos + 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, "- Live Booth Scoreboard", (cx2 + 30, y_pos + 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        # Card 3
        cx3 = start_x + 2 * (card_w + gap)
        cv2.putText(canvas, "Showcase & Demos", (cx3 + 30, y_pos + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_SUCCESS, 2, cv2.LINE_AA)
        cv2.putText(canvas, "One-click gesture demonstrations", (cx3 + 30, y_pos + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "and posture sanity checks.", (cx3 + 30, y_pos + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(canvas, "- Rock / Paper / Scissors", (cx3 + 30, y_pos + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, "- Middle Finger Extension", (cx3 + 30, y_pos + 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
        cv2.putText(canvas, "- Hardware Limit Check", (cx3 + 30, y_pos + 270), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    def _render_teleop_screen(
        self,
        canvas: np.ndarray,
        camera_frame: np.ndarray | None,
        landmarks_list: list[Any],
    ) -> None:
        vw, vh = 800, 530
        vx, vy = 40, 85
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), COLOR_CARD_BORDER, 2)

        if camera_frame is not None:
            vis_frame = camera_frame.copy()
            if landmarks_list:
                for lm in landmarks_list:
                    draw_hand(vis_frame, lm, [], False)
            resized = cv2.resize(vis_frame, (vw - 8, vh - 8))
            canvas[vy + 4 : vy + vh - 4, vx + 4 : vx + vw - 4] = resized
        else:
            cv2.putText(
                canvas,
                "No Camera Stream Available",
                (vx + 260, vy + 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLOR_TEXT_MUTED,
                2,
                cv2.LINE_AA,
            )

        arm_text = "ARMED (Tracking Active)" if self.armed else "DISARMED (Paused)"
        arm_color = COLOR_SUCCESS if self.armed else COLOR_WARNING
        cv2.rectangle(canvas, (vx + 20, vy + 20), (vx + 340, vy + 55), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            arm_text,
            (vx + 30, vy + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            arm_color,
            2,
            cv2.LINE_AA,
        )

        if self.calib_open_in_progress or self.calib_fist_in_progress:
            progress = self.calibrator.progress(time.perf_counter())
            calib_title = (
                "Capturing Open Neutral Pose..."
                if self.calib_open_in_progress
                else "Capturing Closed Fist Pose..."
            )

            pw, ph = 460, 90
            px = vx + (vw - pw) // 2
            py = vy + vh - ph - 25
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), (15, 15, 15), -1)
            cv2.rectangle(canvas, (px, py), (px + pw, py + ph), COLOR_PRIMARY, 2)
            cv2.putText(canvas, calib_title, (px + 20, py + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_MAIN, 2, cv2.LINE_AA)

            bar_w = pw - 40
            cv2.rectangle(canvas, (px + 20, py + 48), (px + 20 + bar_w, py + 72), (40, 40, 40), -1)
            cv2.rectangle(canvas, (px + 20, py + 48), (px + 20 + int(bar_w * progress), py + 72), COLOR_SUCCESS, -1)
            cv2.putText(canvas, f"{int(progress * 100)}%", (px + pw - 80, py + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT_MAIN, 2, cv2.LINE_AA)

    def _render_rps_screen(
        self,
        canvas: np.ndarray,
        camera_frame: np.ndarray | None,
        landmarks_list: list[Any],
    ) -> None:
        vw, vh = 460, 360
        vx, vy = 40, 85
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), COLOR_CARD_BORDER, 2)

        if camera_frame is not None:
            vis_frame = camera_frame.copy()
            if landmarks_list:
                for lm in landmarks_list:
                    draw_hand(vis_frame, lm, [], False)
            resized = cv2.resize(vis_frame, (vw - 8, vh - 8))
            canvas[vy + 4 : vy + vh - 4, vx + 4 : vx + vw - 4] = resized

        gw, gh = 460, 85
        gx, gy = vx, vy + vh + 15
        cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), COLOR_CARD_BORDER, 2)
        cv2.putText(canvas, "Player Gesture Detected:", (gx + 15, gy + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

        gesture_labels = {
            "rock": "ROCK (Rock Move)",
            "paper": "PAPER (Paper Move)",
            "scissors": "SCISSORS (Scissors Move)",
            None: "Show hand to camera...",
        }
        g_label = gesture_labels.get(self.human_detected_move, "Detecting...")
        g_col = COLOR_HUMAN if self.human_detected_move else COLOR_TEXT_MUTED
        cv2.putText(canvas, g_label, (gx + 20, gy + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, g_col, 2, cv2.LINE_AA)

        ax, ay, aw, ah = 520, 85, 340, 460
        cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (ax, ay), (ax + aw, ay + ah), COLOR_CARD_BORDER, 2)

        cv2.putText(canvas, "BATTLE ARENA", (ax + 85, ay + 38), cv2.FONT_HERSHEY_DUPLEX, 0.7, COLOR_PRIMARY, 2, cv2.LINE_AA)
        cv2.line(canvas, (ax + 20, ay + 50), (ax + aw - 20, ay + 50), COLOR_CARD_BORDER, 1)

        now = time.monotonic()
        if self.rps_state == RpsState.IDLE:
            cv2.putText(canvas, "READY TO PLAY!", (ax + 85, ay + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.75, COLOR_TEXT_MAIN, 2, cv2.LINE_AA)
            cv2.putText(canvas, "Press [Space] key or", (ax + 75, ay + 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
            cv2.putText(canvas, "click [START] button", (ax + 70, ay + 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_PRIMARY, 2, cv2.LINE_AA)

        elif self.rps_state == RpsState.COUNTDOWN:
            elapsed = now - self.countdown_start_time
            count_val = max(1, 3 - int(elapsed))
            cx, cy = ax + aw // 2, ay + 210
            radius = 60 + int(math.sin(elapsed * 10.0) * 8.0)
            cv2.circle(canvas, (cx, cy), radius, COLOR_SECONDARY, 4)
            cv2.putText(canvas, str(count_val), (cx - 20, cy + 22), cv2.FONT_HERSHEY_DUPLEX, 2.2, COLOR_TEXT_MAIN, 4, cv2.LINE_AA)
            cv2.putText(canvas, "Rock... Paper... Scissors...", (ax + 40, ay + 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_SECONDARY, 2, cv2.LINE_AA)

        elif self.rps_state in (RpsState.SHOOT, RpsState.RESULT):
            cv2.putText(canvas, "ROBOT HAND:", (ax + 25, ay + 95), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_ROBOT, 2, cv2.LINE_AA)
            r_move_text = f"[{self.robot_chosen_move.upper()}]" if self.robot_chosen_move else "..."
            cv2.putText(canvas, r_move_text, (ax + 55, ay + 145), cv2.FONT_HERSHEY_DUPLEX, 1.1, COLOR_ROBOT, 3, cv2.LINE_AA)

            cv2.putText(canvas, "YOU (PLAYER):", (ax + 25, ay + 215), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_HUMAN, 2, cv2.LINE_AA)
            h_move_text = f"[{self.human_detected_move.upper()}]" if self.human_detected_move else "NO HAND"
            cv2.putText(canvas, h_move_text, (ax + 55, ay + 265), cv2.FONT_HERSHEY_DUPLEX, 1.1, COLOR_HUMAN, 3, cv2.LINE_AA)

            if self.last_round_verdict:
                v_text = ""
                v_color = COLOR_TEXT_MAIN
                if self.last_round_verdict == "win":
                    v_text = "YOU WIN!"
                    v_color = COLOR_SUCCESS
                elif self.last_round_verdict == "loss":
                    v_text = "ROBOT WINS!"
                    v_color = COLOR_DANGER
                elif self.last_round_verdict == "tie":
                    v_text = "DRAW (TIE)!"
                    v_color = COLOR_WARNING
                else:
                    v_text = "No Hand Detected"
                    v_color = COLOR_TEXT_MUTED

                cv2.rectangle(canvas, (ax + 15, ay + 330), (ax + aw - 15, ay + 410), (20, 20, 20), -1)
                cv2.rectangle(canvas, (ax + 15, ay + 330), (ax + aw - 15, ay + 410), v_color, 2)
                (vtw, _), _ = cv2.getTextSize(v_text, cv2.FONT_HERSHEY_DUPLEX, 0.75, 2)
                cv2.putText(canvas, v_text, (ax + (aw - vtw) // 2, ay + 378), cv2.FONT_HERSHEY_DUPLEX, 0.75, v_color, 2, cv2.LINE_AA)

        sx, sy, sw, sh = 880, 390, 370, 200
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (sx, sy), (sx + sw, sy + sh), COLOR_CARD_BORDER, 2)
        cv2.putText(canvas, "SCOREBOARD", (sx + 20, sy + 32), cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_TEXT_MAIN, 2, cv2.LINE_AA)
        cv2.line(canvas, (sx + 20, sy + 44), (sx + sw - 20, sy + 44), COLOR_CARD_BORDER, 1)

        sb = self.rps_scoreboard
        win_rate = (sb.human_wins / max(1, sb.total_rounds)) * 100.0
        cv2.putText(canvas, f"Total Rounds: {sb.total_rounds}", (sx + 25, sy + 75), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Player Wins:  {sb.human_wins} Wins", (sx + 25, sy + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_SUCCESS, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Robot Wins:   {sb.robot_wins} Wins", (sx + 25, sy + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_DANGER, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Ties / Draws: {sb.ties}  |  Win Rate: {win_rate:.1f}%", (sx + 25, sy + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WARNING, 1, cv2.LINE_AA)

    def _render_showcase_screen(self, canvas: np.ndarray, camera_frame: np.ndarray | None) -> None:
        vw, vh = 800, 530
        vx, vy = 40, 85
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), COLOR_CARD_BG, -1)
        cv2.rectangle(canvas, (vx, vy), (vx + vw, vy + vh), COLOR_CARD_BORDER, 2)

        # In Showcase mode, render MuJoCo 3D digital twin hand directly into the viewport
        sim_frame = None
        if self.mujoco_controller is not None:
            try:
                sim_frame = self.mujoco_controller.render_bgr(480, 640)
            except Exception:
                sim_frame = None

        display_frame = sim_frame if sim_frame is not None else camera_frame

        if display_frame is not None:
            resized = cv2.resize(display_frame, (vw - 8, vh - 8))
            canvas[vy + 4 : vy + vh - 4, vx + 4 : vx + vw - 4] = resized
        else:
            cv2.putText(
                canvas,
                "No 3D Simulation Stream Available",
                (vx + 180, vy + 270),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLOR_TEXT_MUTED,
                2,
                cv2.LINE_AA,
            )

        badge_text = (
            "3D Robot Simulation (Digital Twin)"
            if sim_frame is not None
            else "Camera View"
        )
        cv2.rectangle(canvas, (vx + 20, vy + 20), (vx + 380, vy + 55), (20, 20, 20), -1)
        cv2.putText(
            canvas,
            badge_text,
            (vx + 30, vy + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            COLOR_PRIMARY if sim_frame is not None else COLOR_SUCCESS,
            2,
            cv2.LINE_AA,
        )

    def run(self) -> int:
        """Main application loop."""
        print("\n" + "=" * 65)
        print("  LEAP HAND INTERACTIVE BOOTH KIOSK APPLICATION")
        print("=" * 65)
        print(f"  Mode: {self.mode.upper()} | Port: {self.port} | Profile: {self.profile}")
        print("  Controls:")
        print("    [1] Teleoperation | [2] RPS Game | [3] Showcase | [H] Home")
        print("    [C] Open Calib    | [F] Fist Calib | [A] Arm     | [D] Disarm")
        print("    [Space] RPS Start | [P] Auto Play  | [Q/ESC] Quit")
        print("=" * 65 + "\n")

        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self.on_mouse_event)

        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"[BOOTH WARN] Camera index {self.camera_id} could not be opened. Running in GUI-only mode.")

        try:
            while True:
                ret = False
                frame = None
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frame = cv2.flip(frame, 1)

                landmarks_list = []
                if ret and frame is not None:
                    landmarks_list = self.tracker.process_frame(frame)

                if self.current_screen == AppScreen.TELEOP:
                    self.update_teleop_frame(frame, landmarks_list)
                elif self.current_screen == AppScreen.RPS:
                    self.update_rps_frame(frame, landmarks_list)

                # Smooth joint interpolation and physics stepping
                self.step_smooth_control(0.02)

                canvas = self.render(frame, landmarks_list)
                cv2.imshow(self.window_name, canvas)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("[BOOTH] Exit requested by user.")
                    break

                # Global Navigation Keys
                if key == ord("1"):
                    self.handle_action("goto_teleop")
                elif key == ord("2"):
                    self.handle_action("goto_rps")
                elif key == ord("3"):
                    self.handle_action("goto_showcase")
                elif key in (ord("h"), ord("H")):
                    self.handle_action("back_home_teleop")

                # Teleop Keys
                elif self.current_screen == AppScreen.TELEOP:
                    if key in (ord("c"), ord("C")):
                        self.handle_action("calib_open")
                    elif key in (ord("f"), ord("F")):
                        self.handle_action("calib_fist")
                    elif key in (ord("a"), ord("A")):
                        self.handle_action("toggle_arm")
                    elif key in (ord("d"), ord("D"), ord(" ")):
                        self.handle_action("disarm")
                    elif key in (ord("r"), ord("R")):
                        self.handle_action("reset_calib")

                # RPS Keys
                elif self.current_screen == AppScreen.RPS:
                    if key == ord(" "):
                        self.handle_action("rps_start")
                    elif key in (ord("p"), ord("P")):
                        self.handle_action("rps_auto_toggle")
                    elif key in (ord("r"), ord("R")):
                        self.handle_action("rps_reset_score")

                # Showcase Keys
                elif self.current_screen == AppScreen.SHOWCASE:
                    if key == ord("1"):
                        self.handle_action("showcase_rock")
                    elif key == ord("2"):
                        self.handle_action("showcase_paper")
                    elif key == ord("3"):
                        self.handle_action("showcase_scissors")
                    elif key == ord("4"):
                        self.handle_action("showcase_neutral")
                    elif key == ord("5"):
                        self.handle_action("showcase_middle")
                    elif key == ord("6"):
                        self.handle_action("showcase_thumbs_up")
                    elif key == ord("7"):
                        self.handle_action("showcase_ok")
                    elif key == ord("8"):
                        self.handle_action("showcase_pointing")
                    elif key == ord("9"):
                        self.handle_action("showcase_rock_on")
                    elif key in (ord("w"), ord("W")):
                        self.handle_action("showcase_finger_wave")
                    elif key in (ord("v"), ord("V")):
                        self.handle_action("showcase_wave_hello")

        finally:
            if cap.isOpened():
                cap.release()
            cv2.destroyAllWindows()
            self.disarm_robot()
            if self.hardware_controller is not None:
                self.hardware_controller.disconnect()
            print("[BOOTH] Kiosk shutdown complete.")

        return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch the LEAP Hand Interactive Booth Kiosk Application."
    )
    parser.add_argument(
        "--mode",
        choices=("hardware", "mujoco", "both"),
        default="hardware",
        help="Target controller backend: hardware, mujoco, or both (default: hardware)",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port for LEAP Hand hardware (default: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--profile",
        default="jiwoo",
        help="User neutral calibration profile name (default: jiwoo)",
    )
    parser.add_argument(
        "--calib-path",
        type=Path,
        default=DEFAULT_CALIB_PATH,
        help="Path to neutral calibration JSON file (default: calibration/neutral_calibration.json)",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--current-limit",
        type=int,
        default=350,
        help="Motor current limit in mA (default: 350)",
    )
    parser.add_argument(
        "--no-mujoco",
        action="store_true",
        help="Disable MuJoCo simulation viewer",
    )
    parser.add_argument(
        "--no-hardware",
        action="store_true",
        help="Disable physical hardware controller",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    enable_mujoco = not args.no_mujoco and args.mode in ("mujoco", "both")
    enable_hardware = not args.no_hardware and args.mode in ("hardware", "both")

    app = BoothKioskApp(
        mode=args.mode,
        port=args.port,
        profile=args.profile,
        calib_path=args.calib_path,
        camera_id=args.camera_id,
        current_limit=args.current_limit,
        enable_mujoco=enable_mujoco,
        enable_hardware=enable_hardware,
    )
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
