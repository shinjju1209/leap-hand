"""Live webcam hand-landmark visualization with MediaPipe Tasks."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from deadband_filter import AngleDeadband
from hand_angles import DISPLAY_NAMES, calculate_leap_control_angles
from neutral_calibration import (
    DEFAULT_FLEXION_TARGETS_DEGREES,
    NeutralCalibration,
)
from one_euro_filter import OneEuroFilter
from rps.gesture import GestureClassification, GestureStabilizer, classify_rps_gesture
from rps.moves import MOVE_NAMES
from rps.rounds import CsvRoundRecorder, RpsRoundSession


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

FINGERTIP_IDS = {4, 8, 12, 16, 20}
DEFAULT_MUJOCO_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "mujoco"
    / "leap_hand"
    / "scene_right.xml"
)


def parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_mujoco: bool = False,
    default_hardware: bool = False,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Track hand landmarks from a laptop webcam."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/hand_landmarker.task"),
        help="Path to the MediaPipe Hand Landmarker model",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-hands", type=int, default=1)
    parser.add_argument(
        "--min-cutoff",
        type=float,
        default=0.5,
        help="One Euro smoothing strength while nearly stationary",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.08,
        help="One Euro responsiveness to fast motion",
    )
    parser.add_argument(
        "--derivative-cutoff",
        type=float,
        default=1.0,
        help="One Euro derivative low-pass cutoff",
    )
    parser.add_argument(
        "--deadband",
        type=float,
        default=1.2,
        help="Minimum filtered angle change to emit, in degrees; 0 disables it",
    )
    parser.add_argument(
        "--profile",
        default="default",
        help="Person name used to select a neutral-calibration profile",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=Path("calibration/neutral_angles.json"),
        help="JSON file containing per-person neutral offsets",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=float,
        default=1.5,
        help="Seconds of pose samples collected after pressing C or F",
    )
    parser.add_argument(
        "--flexion-scale",
        type=float,
        default=1.0,
        help="Scale all range-calibrated MuJoCo flexion targets",
    )
    parser.add_argument(
        "--neutral-calibration",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply the selected person's saved neutral calibration",
    )
    parser.add_argument(
        "--filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable the One Euro Filter",
    )
    parser.add_argument(
        "--mirror",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Mirror the webcam image like a selfie camera",
    )
    parser.add_argument(
        "--gesture-stable-frames",
        type=int,
        default=4,
        help="Consecutive matching frames required to confirm an RPS gesture",
    )
    parser.add_argument(
        "--gesture-extended-max",
        type=float,
        default=55.0,
        help="Largest PIP+DIP bend classified as an extended finger (degrees)",
    )
    parser.add_argument(
        "--gesture-curled-min",
        type=float,
        default=100.0,
        help="Smallest PIP+DIP bend classified as a curled finger (degrees)",
    )
    parser.add_argument(
        "--gesture-thumb-extended-max",
        type=float,
        default=70.0,
        help="Largest thumb bend accepted for the index+thumb scissors pose",
    )
    parser.add_argument(
        "--gesture-thumb-extended-min-span",
        type=float,
        default=0.75,
        help="Minimum thumb-tip span for the index+thumb scissors pose",
    )
    parser.add_argument(
        "--gesture-thumb-curled-max-span",
        type=float,
        default=0.55,
        help="Maximum thumb-tip span classified as a folded thumb",
    )
    parser.add_argument(
        "--robot-move",
        choices=MOVE_NAMES,
        help="Known robot move for the first round; keys 1/2/3 start later rounds",
    )
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("rps_results.csv"),
        help="CSV file used to append completed round results",
    )
    parser.add_argument(
        "--mujoco",
        action=argparse.BooleanOptionalAction,
        default=default_mujoco,
        help="Open the MuJoCo viewer and drive its right LEAP Hand",
    )
    parser.add_argument(
        "--mujoco-model",
        type=Path,
        default=DEFAULT_MUJOCO_MODEL_PATH,
        help="Path to the right LEAP Hand MuJoCo scene",
    )
    parser.add_argument(
        "--collision-avoidance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Predict and limit MuJoCo targets that cause hand self-contact",
    )
    parser.add_argument(
        "--hardware",
        action=argparse.BooleanOptionalAction,
        default=default_hardware,
        help="Connect and command physical LEAP Hand v1 hardware via DYNAMIXEL",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="DYNAMIXEL serial port (default: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=4_000_000,
        help="DYNAMIXEL baudrate (default: 4000000)",
    )
    parser.add_argument(
        "--current-limit",
        type=int,
        default=300,
        help="Hardware goal current limit in mA (default: 300)",
    )
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=120.0,
        help="Maximum hardware joint speed in deg/s (default: 120.0)",
    )
    parser.add_argument(
        "--max-tracking-error",
        type=float,
        default=25.0,
        help="Maximum allowable tracking error in deg before emergency stop",
    )
    parser.add_argument(
        "--max-temperature",
        type=float,
        default=50.0,
        help="Maximum motor temperature in C before emergency stop",
    )
    parser.add_argument(
        "--motor-calibration-file",
        type=Path,
        default=Path("calibration/hardware_motors.yaml"),
        help="Hardware motor calibration YAML file",
    )
    parser.add_argument(
        "--tracking-loss-hold-seconds",
        type=float,
        default=0.2,
        help="Seconds to hold pose during brief vision tracking loss",
    )
    parser.add_argument(
        "--tracking-loss-disarm-seconds",
        type=float,
        default=0.5,
        help="Seconds of tracking loss before automatic hardware disarm",
    )
    return parser.parse_args(argv)


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # DirectShow for Windows, V4L2 for Linux, generic fallback otherwise
    if sys.platform.startswith("win"):
        capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    elif sys.platform.startswith("linux"):
        capture = cv2.VideoCapture(index, cv2.CAP_V4L2)
    else:
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)

    if not capture.isOpened():
        raise RuntimeError(
            f"카메라 {index}번을 열 수 없습니다. 다른 앱이 카메라를 사용 중인지 "
            "확인하거나 --camera 1을 시도하세요."
        )

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def draw_hand(frame, landmarks, handedness, mirror_display: bool) -> None:
    height, width = frame.shape[:2]
    points = [
        (
            int((1.0 - landmark.x if mirror_display else landmark.x) * width),
            int(landmark.y * height),
        )
        for landmark in landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(frame, points[start], points[end], (70, 220, 70), 2, cv2.LINE_AA)

    for index, point in enumerate(points):
        radius = 6 if index in FINGERTIP_IDS else 4
        color = (0, 190, 255) if index in FINGERTIP_IDS else (255, 100, 40)
        cv2.circle(frame, point, radius, color, -1, cv2.LINE_AA)

    label = "Hand"
    score = 0.0
    if handedness:
        category = handedness[0]
        label = category.category_name or label
        score = category.score

    label_x = max(10, min(point[0] for point in points))
    label_y = max(30, min(point[1] for point in points) - 12)
    cv2.putText(
        frame,
        f"{label} {score:.2f}",
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_angle_panel(
    frame,
    raw_angles,
    output_angles,
    hand_label: str,
    panel_index: int,
    processing_label: str | None,
) -> None:
    """Draw raw and processed human-hand angles in two columns."""
    panel_width = 610
    panel_height = 236
    x0 = max(8, frame.shape[1] - panel_width - 14)
    y0 = 82 + panel_index * (panel_height + 10)
    y1 = min(frame.shape[0] - 8, y0 + panel_height)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_width, y1), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + panel_width, y1), (90, 210, 255), 1)

    cv2.putText(
        frame,
        (
            f"{hand_label} joint angles: raw -> {processing_label} (deg)"
            if processing_label
            else f"{hand_label} joint angles: processing OFF (deg)"
        ),
        (x0 + 12, y0 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (90, 220, 255),
        1,
        cv2.LINE_AA,
    )

    column_width = 296
    for index, (name, raw_value, output_value) in enumerate(
        zip(DISPLAY_NAMES, raw_angles, output_angles)
    ):
        column = index // 8
        row = index % 8
        text_x = x0 + 12 + column * column_width
        text_y = y0 + 50 + row * 22
        cv2.putText(
            frame,
            (
                f"{name:<17} {raw_value:6.1f} > {output_value:6.1f}"
                if processing_label
                else f"{name:<17} {raw_value:6.1f}"
            ),
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )


def draw_gesture_status(
    frame,
    classification: GestureClassification,
    stable_label: str | None,
    panel_index: int,
) -> None:
    """Show the one-frame candidate and temporally stable RPS result."""
    candidate = classification.label.upper() if classification.label else "UNKNOWN"
    confirmed = stable_label.upper() if stable_label else "HOLD STEADY"
    color = (50, 230, 80) if stable_label else (0, 210, 255)
    y = 105 + panel_index * 42
    cv2.putText(
        frame,
        f"RPS: {confirmed} | candidate {candidate}",
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
        cv2.LINE_AA,
    )
    state_text = "  ".join(
        f"{name}:{state[0].upper()} {bend:.0f}"
        for name, state, bend in zip(
            ("T", "I", "M", "R", "P"),
            classification.finger_states,
            classification.bend_degrees,
        )
    )
    state_text += f"  Tspan:{classification.thumb_span_ratio:.2f}"
    cv2.putText(
        frame,
        state_text,
        (18, y + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )


def draw_round_status(frame, session: RpsRoundSession) -> None:
    """Draw the known robot move, last result, and cumulative score."""
    robot_move = (
        session.pending_robot_move.upper()
        if session.pending_robot_move
        else "PRESS 1/2/3"
    )
    cv2.putText(
        frame,
        f"Robot move: {robot_move}  [1 rock, 2 paper, 3 scissors]",
        (18, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 210, 80),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        (
            f"Score: human {session.human_wins} | robot {session.robot_wins} "
            f"| ties {session.ties}"
        ),
        (18, 192),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    if session.last_record is not None:
        record = session.last_record
        cv2.putText(
            frame,
            (
                f"Round {record.round_number}: human {record.human_move.upper()} "
                f"vs robot {record.robot_move.upper()} = {record.human_result.upper()}"
            ),
            (18, 219),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 230, 100) if record.human_result == "win" else (80, 190, 255),
            2,
            cv2.LINE_AA,
        )


def draw_hardware_status(
    frame,
    *,
    is_armed: bool,
    error_msg: str | None,
    max_temp: float,
    worst_error: float,
    loss_hold: bool = False,
) -> None:
    """Draw live hardware connection, arm state, temperature, and errors."""
    y = 250
    if error_msg:
        color = (0, 0, 255)
        text = f"HW: {error_msg} | Press A to Reset/Arm"
    elif is_armed:
        if loss_hold:
            color = (0, 180, 255)
            text = f"HW: ARMED (HOLDING POSE - Tracking Lost) | Temp: {max_temp:.0f}C | [D/Space: Disarm]"
        else:
            color = (80, 255, 120)
            text = f"HW: ARMED (FOLLOWING RIGHT HAND) | Temp: {max_temp:.0f}C | Err: {worst_error:.1f}deg | [D/Space: Disarm]"
    else:
        color = (0, 220, 255)
        text = f"HW: DISARMED (Torque OFF) | Temp: {max_temp:.0f}C | [Press A to Arm]"

    cv2.putText(
        frame,
        text,
        (18, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
        cv2.LINE_AA,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    default_mujoco: bool = False,
    default_hardware: bool = False,
) -> None:
    args = parse_args(
        argv,
        default_mujoco=default_mujoco,
        default_hardware=default_hardware,
    )
    if args.deadband < 0.0:
        raise ValueError("--deadband must be zero or greater")
    if args.calibration_seconds <= 0.0:
        raise ValueError("--calibration-seconds must be greater than zero")
    if args.flexion_scale <= 0.0:
        raise ValueError("--flexion-scale must be greater than zero")
    if args.gesture_stable_frames < 1:
        raise ValueError("--gesture-stable-frames must be at least one")
    if args.gesture_extended_max < 0.0:
        raise ValueError("--gesture-extended-max must be zero or greater")
    if args.gesture_curled_min <= args.gesture_extended_max:
        raise ValueError("--gesture-curled-min must exceed --gesture-extended-max")
    if args.gesture_thumb_extended_max < 0.0:
        raise ValueError("--gesture-thumb-extended-max must be zero or greater")
    if args.gesture_thumb_curled_max_span < 0.0:
        raise ValueError("--gesture-thumb-curled-max-span must be zero or greater")
    if (
        args.gesture_thumb_extended_min_span
        <= args.gesture_thumb_curled_max_span
    ):
        raise ValueError(
            "--gesture-thumb-extended-min-span must exceed "
            "--gesture-thumb-curled-max-span"
        )
    if args.current_limit < 1 or args.current_limit > 550:
        raise ValueError("--current-limit must be between 1 and 550 mA")
    if args.max_joint_speed <= 0.0:
        raise ValueError("--max-joint-speed must be greater than zero")
    if args.max_tracking_error <= 0.0:
        raise ValueError("--max-tracking-error must be greater than zero")
    if args.max_temperature <= 0.0:
        raise ValueError("--max-temperature must be greater than zero")
    if args.tracking_loss_hold_seconds < 0.0:
        raise ValueError("--tracking-loss-hold-seconds must be non-negative")
    if args.tracking_loss_disarm_seconds <= args.tracking_loss_hold_seconds:
        raise ValueError(
            "--tracking-loss-disarm-seconds must exceed "
            "--tracking-loss-hold-seconds"
        )

    base_processing_steps = []
    if args.filter:
        base_processing_steps.append("One Euro")
    if args.deadband > 0.0:
        base_processing_steps.append(f"deadband {args.deadband:.1f}")

    neutral_calibration = NeutralCalibration(
        args.calibration_file,
        profile=args.profile,
        duration_seconds=args.calibration_seconds,
        flexion_targets_degrees=(
            DEFAULT_FLEXION_TARGETS_DEGREES * args.flexion_scale
        ),
    )

    model_path = args.model.resolve()
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MediaPipe 모델을 찾을 수 없습니다: {model_path}\n"
            "README.md의 모델 다운로드 단계를 먼저 실행하세요."
        )

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=args.max_hands,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )

    mujoco_controller = None
    mujoco_viewer = None
    hardware_controller = None
    hardware_armed = False
    hardware_error_msg: str | None = None
    last_right_hand_seen_seconds: float | None = None
    last_hw_command_angles: np.ndarray | None = None
    hw_max_temperature = 0.0
    hw_worst_tracking_error = 0.0
    last_hw_health_check_time = 0.0
    hw_holding_loss = False

    try:
        if args.mujoco:
            from mujoco_hand_controller import MujocoHandController

            mujoco_controller = MujocoHandController(args.mujoco_model)
            mujoco_viewer = mujoco_controller.launch_viewer()

        if args.hardware:
            from leap_hand_hardware_controller import LeapHandHardwareController

            calib_file = (
                args.motor_calibration_file
                if args.motor_calibration_file.is_file()
                else None
            )
            if (
                calib_file is None
                and str(args.motor_calibration_file)
                != "calibration/hardware_motors.yaml"
            ):
                print(
                    f"Warning: Calibration file {args.motor_calibration_file} "
                    "not found; using nominal open-pose calibration.",
                    flush=True,
                )
            hardware_controller = LeapHandHardwareController(
                args.port,
                baudrate=args.baudrate,
                current_limit_milliamps=args.current_limit,
                max_joint_speed_degrees_per_second=args.max_joint_speed,
                motor_calibration=calib_file,
            )
            models = hardware_controller.connect()
            print(
                f"[HARDWARE] Connected on {args.port}. IDs/Models: {models}",
                flush=True,
            )
            hardware_controller.configure()
            print(
                "[HARDWARE] Low-current mode configured. Torque is OFF. "
                "Press 'A' in the GUI window to Arm (enable torque).",
                flush=True,
            )

        capture = open_camera(args.camera, args.width, args.height)
    except Exception:
        if mujoco_controller is not None:
            mujoco_controller.close()
        if hardware_controller is not None:
            hardware_controller.close()
        raise
    if args.hardware and args.mujoco:
        window_name = "MediaPipe + Hardware + MuJoCo Teleoperation"
    elif args.hardware:
        window_name = (
            "MediaPipe + LEAP Hand Hardware - A: Arm | D/Space: Disarm | Q: Quit"
        )
    elif args.mujoco:
        window_name = "MediaPipe + MuJoCo Teleoperation"
    else:
        window_name = "MediaPipe Hand Tracking - C/F calibrate | Q/ESC quit"
    previous_frame_time = time.perf_counter()
    smoothed_fps = 0.0
    last_timestamp_ms = -1
    angle_filters: dict[str, OneEuroFilter] = {}
    angle_deadbands: dict[str, AngleDeadband] = {}
    gesture_stabilizers: dict[str, GestureStabilizer] = {}
    round_session = RpsRoundSession(CsvRoundRecorder(args.results_csv))
    if args.robot_move:
        round_session.start_round(args.robot_move)
    last_hand_seen_seconds: float | None = None
    last_simulation_time = time.perf_counter()

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, frame = capture.read()
                if not ok:
                    print("카메라 프레임을 읽지 못했습니다.")
                    break

                # Always run handedness inference on the original camera image.
                # Mirroring before inference swaps Right/Left classification.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                timestamp_ms = time.perf_counter_ns() // 1_000_000
                timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                timestamp_seconds = timestamp_ms / 1000.0

                if result.hand_landmarks:
                    last_hand_seen_seconds = timestamp_seconds
                elif (
                    last_hand_seen_seconds is not None
                    and timestamp_seconds - last_hand_seen_seconds > 0.5
                ):
                    angle_filters.clear()
                    angle_deadbands.clear()
                    gesture_stabilizers.clear()
                    last_hand_seen_seconds = None

                if args.mirror:
                    frame = cv2.flip(frame, 1)

                visible_hand_labels: list[str] = []
                mujoco_command_sent = False
                mujoco_collision_scale = 1.0
                for index, landmarks in enumerate(result.hand_landmarks):
                    handedness = (
                        result.handedness[index]
                        if index < len(result.handedness)
                        else []
                    )
                    draw_hand(frame, landmarks, handedness, args.mirror)

                    world_landmarks = (
                        result.hand_world_landmarks[index]
                        if index < len(result.hand_world_landmarks)
                        else landmarks
                    )
                    hand_label = (
                        handedness[0].category_name
                        if handedness and handedness[0].category_name
                        else "Hand"
                    )
                    filter_key = hand_label
                    gesture_stabilizer = gesture_stabilizers.get(filter_key)
                    if gesture_stabilizer is None:
                        gesture_stabilizer = GestureStabilizer(
                            args.gesture_stable_frames
                        )
                        gesture_stabilizers[filter_key] = gesture_stabilizer
                    try:
                        gesture = classify_rps_gesture(
                            world_landmarks,
                            extended_max_degrees=args.gesture_extended_max,
                            curled_min_degrees=args.gesture_curled_min,
                            thumb_extended_max_degrees=(
                                args.gesture_thumb_extended_max
                            ),
                            thumb_extended_min_span=(
                                args.gesture_thumb_extended_min_span
                            ),
                            thumb_curled_max_span=(
                                args.gesture_thumb_curled_max_span
                            ),
                        )
                    except ValueError:
                        gesture_stabilizer.reset()
                    else:
                        stable_gesture = gesture_stabilizer.update(gesture.label)
                        draw_gesture_status(frame, gesture, stable_gesture, index)
                        completed_round = (
                            round_session.observe_confirmed_human_move(stable_gesture)
                        )
                        if completed_round is not None:
                            print(
                                f"Round {completed_round.round_number}: human "
                                f"{completed_round.human_move} vs robot "
                                f"{completed_round.robot_move} -> human "
                                f"{completed_round.human_result}",
                                flush=True,
                            )

                    try:
                        angles = calculate_leap_control_angles(world_landmarks)
                    except ValueError as error:
                        cv2.putText(
                            frame,
                            f"ANGLE ERROR: {error}",
                            (18, 96),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 0, 255),
                            1,
                            cv2.LINE_AA,
                        )
                    else:
                        visible_hand_labels.append(hand_label)
                        if args.neutral_calibration:
                            completed_pose = neutral_calibration.active_pose
                            calibration_completed = neutral_calibration.add_sample(
                                hand_label,
                                angles,
                                timestamp_seconds,
                            )
                            if calibration_completed:
                                angle_filters.pop(filter_key, None)
                                angle_deadbands.pop(filter_key, None)
                                print(
                                    f"[{args.profile}] {hand_label} "
                                    f"{completed_pose} "
                                    f"calibration saved to "
                                    f"{neutral_calibration.path.resolve()}"
                                )
                            calibrated_angles = neutral_calibration.apply(
                                hand_label,
                                angles,
                            )
                        else:
                            calibrated_angles = angles

                        if args.filter:
                            angle_filter = angle_filters.get(filter_key)
                            if angle_filter is None:
                                angle_filter = OneEuroFilter(
                                    min_cutoff=args.min_cutoff,
                                    beta=args.beta,
                                    derivative_cutoff=args.derivative_cutoff,
                                )
                                angle_filters[filter_key] = angle_filter
                            filtered_angles = angle_filter.filter(
                                calibrated_angles,
                                timestamp_seconds,
                            )
                        else:
                            filtered_angles = calibrated_angles

                        if args.deadband > 0.0:
                            angle_deadband = angle_deadbands.get(filter_key)
                            if angle_deadband is None:
                                angle_deadband = AngleDeadband(args.deadband)
                                angle_deadbands[filter_key] = angle_deadband
                            command_angles = angle_deadband.filter(filtered_angles)
                        else:
                            command_angles = filtered_angles

                        if hand_label == "Right":
                            if mujoco_controller is not None:
                                if args.collision_avoidance:
                                    command_angles = (
                                        mujoco_controller.set_collision_safe_target_degrees(
                                            command_angles,
                                        )
                                    )
                                    mujoco_collision_scale = (
                                        mujoco_controller.last_collision_scale
                                    )
                                else:
                                    mujoco_controller.set_target_degrees(command_angles)
                                mujoco_command_sent = True

                            if hardware_controller is not None:
                                last_right_hand_seen_seconds = timestamp_seconds
                                hw_holding_loss = False
                                if (
                                    hardware_armed
                                    and hardware_error_msg is None
                                ):
                                    try:
                                        last_hw_command_angles = (
                                            hardware_controller.command_degrees(
                                                command_angles
                                            )
                                        )
                                    except Exception as exc:
                                        hardware_controller.emergency_stop()
                                        hardware_armed = False
                                        hardware_error_msg = f"COMM ERROR: {exc}"
                                        print(
                                            f"[HARDWARE ERROR] {exc}",
                                            flush=True,
                                        )

                        processing_steps = []
                        if (
                            args.neutral_calibration
                            and neutral_calibration.has_offset(hand_label)
                        ):
                            calibration_mode = (
                                "range"
                                if neutral_calibration.has_range(hand_label)
                                else "neutral"
                            )
                            processing_steps.append(
                                f"{calibration_mode}:{args.profile}"
                            )
                        processing_steps.extend(base_processing_steps)
                        hand_processing_label = " + ".join(processing_steps) or None

                        draw_angle_panel(
                            frame,
                            angles,
                            command_angles,
                            hand_label,
                            index,
                            hand_processing_label,
                        )

                now = time.perf_counter()
                if mujoco_controller is not None:
                    if mujoco_viewer is None or not mujoco_viewer.is_running():
                        break
                    simulation_duration = min(
                        max(now - last_simulation_time, mujoco_controller.timestep),
                        0.1,
                    )
                    mujoco_controller.step_for(simulation_duration)
                    mujoco_controller.sync_viewer()
                    last_simulation_time = now

                if hardware_controller is not None:
                    if (
                        hardware_armed
                        and "Right" not in visible_hand_labels
                        and hardware_error_msg is None
                    ):
                        if last_right_hand_seen_seconds is not None:
                            loss_duration = (
                                timestamp_seconds - last_right_hand_seen_seconds
                            )
                            if loss_duration <= args.tracking_loss_hold_seconds:
                                hw_holding_loss = True
                                try:
                                    hardware_controller.heartbeat()
                                except Exception as exc:
                                    hardware_controller.emergency_stop()
                                    hardware_armed = False
                                    hardware_error_msg = f"HEARTBEAT ERROR: {exc}"
                            elif loss_duration > args.tracking_loss_disarm_seconds:
                                hardware_controller.emergency_stop()
                                hardware_armed = False
                                hw_holding_loss = False
                                hardware_error_msg = (
                                    f"TRACKING LOSS TIMEOUT ({loss_duration:.1f}s) "
                                    "- AUTO DISARMED"
                                )
                                print(
                                    f"[HARDWARE SAFETY] Tracking lost for "
                                    f"{loss_duration:.1f}s: Auto-disarmed torque.",
                                    flush=True,
                                )
                            else:
                                hw_holding_loss = True
                        else:
                            hw_holding_loss = False

                    if now - last_hw_health_check_time >= 0.1:
                        last_hw_health_check_time = now
                        if hardware_error_msg is None:
                            try:
                                health = hardware_controller.read_health()
                                hw_max_temperature = float(
                                    np.max(health.temperatures_celsius)
                                )
                                if np.any(health.hardware_errors != 0):
                                    err_ids = np.flatnonzero(
                                        health.hardware_errors
                                    ).tolist()
                                    hardware_controller.emergency_stop()
                                    hardware_armed = False
                                    hardware_error_msg = (
                                        f"HARDWARE ERROR ON MOTOR {err_ids}"
                                    )
                                    print(
                                        f"[HARDWARE ERROR] Motor hardware errors: "
                                        f"{err_ids}",
                                        flush=True,
                                    )
                                elif hw_max_temperature >= args.max_temperature:
                                    hardware_controller.emergency_stop()
                                    hardware_armed = False
                                    hardware_error_msg = (
                                        f"OVERHEAT ({hw_max_temperature:.1f}C >= "
                                        f"{args.max_temperature:.1f}C)"
                                    )
                                    print(
                                        f"[HARDWARE ERROR] Overheat limit reached: "
                                        f"{hw_max_temperature:.1f}C",
                                        flush=True,
                                    )

                                if (
                                    hardware_armed
                                    and last_hw_command_angles is not None
                                ):
                                    fb = hardware_controller.read_feedback()
                                    errs = np.abs(
                                        fb.positions_degrees
                                        - last_hw_command_angles
                                    )
                                    hw_worst_tracking_error = float(
                                        np.max(errs)
                                    )
                                    if (
                                        hw_worst_tracking_error
                                        > args.max_tracking_error
                                    ):
                                        hardware_controller.emergency_stop()
                                        hardware_armed = False
                                        hardware_error_msg = (
                                            f"TRACKING ERROR "
                                            f"({hw_worst_tracking_error:.1f}deg > "
                                            f"{args.max_tracking_error:.1f}deg)"
                                        )
                                        print(
                                            f"[HARDWARE ERROR] Tracking error limit "
                                            f"exceeded: {hw_worst_tracking_error:.1f}deg",
                                            flush=True,
                                        )
                            except Exception as exc:
                                hardware_controller.emergency_stop()
                                hardware_armed = False
                                hardware_error_msg = f"STATUS READ ERROR: {exc}"
                                print(
                                    f"[HARDWARE ERROR] Status read failed: {exc}",
                                    flush=True,
                                )

                instantaneous_fps = 1.0 / max(now - previous_frame_time, 1e-6)
                previous_frame_time = now
                smoothed_fps = (
                    instantaneous_fps
                    if smoothed_fps == 0.0
                    else 0.1 * instantaneous_fps + 0.9 * smoothed_fps
                )

                status = "TRACKING" if result.hand_landmarks else "NO HAND"
                status_color = (60, 220, 60) if result.hand_landmarks else (60, 60, 255)
                cv2.putText(
                    frame,
                    f"{status} | FPS {smoothed_fps:.1f}",
                    (18, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2,
                    cv2.LINE_AA,
                )

                if mujoco_controller is not None:
                    if not mujoco_command_sent:
                        mujoco_text = "MUJOCO: WAITING FOR RIGHT HAND"
                        mujoco_color = (0, 190, 255)
                    elif mujoco_collision_scale < 0.999:
                        mujoco_text = (
                            "MUJOCO: SELF-COLLISION HOLD | "
                            f"{mujoco_collision_scale * 100:.0f}% TARGET"
                        )
                        mujoco_color = (0, 190, 255)
                    else:
                        mujoco_text = "MUJOCO: FOLLOWING RIGHT HAND"
                        mujoco_color = (80, 255, 160)
                    cv2.putText(
                        frame,
                        mujoco_text,
                        (18, 96),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        mujoco_color,
                        1,
                        cv2.LINE_AA,
                    )
                cv2.putText(
                    frame,
                    (
                        f"Profile: {args.profile} | C: open hand | F: closed fist"
                        if args.neutral_calibration
                        else "Neutral calibration OFF"
                    ),
                    (18, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                draw_round_status(frame, round_session)

                if args.hardware:
                    draw_hardware_status(
                        frame,
                        is_armed=hardware_armed,
                        error_msg=hardware_error_msg,
                        max_temp=hw_max_temperature,
                        worst_error=hw_worst_tracking_error,
                        loss_hold=hw_holding_loss,
                    )

                if neutral_calibration.is_collecting:
                    progress = neutral_calibration.progress(timestamp_seconds)
                    pose_instruction = (
                        "KEEP HAND OPEN"
                        if neutral_calibration.active_pose == "neutral"
                        else "KEEP FIST FULLY CLOSED"
                    )
                    calibration_text = (
                        f"CALIBRATING {neutral_calibration.active_hand} "
                        f"{neutral_calibration.active_pose}: "
                        f"{progress * 100:3.0f}% | {pose_instruction} "
                        f"({neutral_calibration.sample_count} samples)"
                    )
                    cv2.putText(
                        frame,
                        calibration_text,
                        (18, frame.shape[0] - 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 220, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                if key in (ord("a"), ord("A")):
                    if hardware_controller is not None:
                        if not hardware_armed:
                            try:
                                hardware_controller.enable_torque()
                                hardware_armed = True
                                hardware_error_msg = None
                                last_right_hand_seen_seconds = timestamp_seconds
                                print(
                                    "[HARDWARE] Torque ENABLED. Following right hand. "
                                    "(Press D or Space to Disarm)",
                                    flush=True,
                                )
                            except Exception as exc:
                                hardware_controller.emergency_stop()
                                hardware_armed = False
                                hardware_error_msg = f"ARM FAILED: {exc}"
                                print(f"[HARDWARE ERROR] Arm failed: {exc}", flush=True)
                if key in (ord("d"), ord("D"), ord(" ")):
                    if hardware_controller is not None and hardware_armed:
                        hardware_controller.emergency_stop()
                        hardware_armed = False
                        print("[HARDWARE] DISARMED. Torque is OFF.", flush=True)
                robot_move_keys = {
                    ord("1"): "rock",
                    ord("2"): "paper",
                    ord("3"): "scissors",
                }
                if key in robot_move_keys:
                    robot_move = robot_move_keys[key]
                    round_session.start_round(robot_move)
                    print(f"Robot move set to {robot_move}; waiting for human.", flush=True)
                if key in (ord("c"), ord("C")):
                    if not args.neutral_calibration:
                        print("Neutral calibration is disabled.")
                    elif not visible_hand_labels:
                        print("손이 보이지 않습니다. 손을 편 상태로 화면에 보여주세요.")
                    else:
                        calibration_hand = visible_hand_labels[0]
                        neutral_calibration.start(
                            calibration_hand,
                            timestamp_seconds,
                            pose="neutral",
                        )
                        angle_filters.pop(calibration_hand, None)
                        angle_deadbands.pop(calibration_hand, None)
                        print(
                            f"[{args.profile}] {calibration_hand} calibration "
                            f"started. Keep the hand open for "
                            f"{args.calibration_seconds:.1f} seconds."
                        )
                if key in (ord("f"), ord("F")):
                    if not args.neutral_calibration:
                        print("Neutral calibration is disabled.")
                    elif not visible_hand_labels:
                        print("손이 보이지 않습니다. 주먹을 쥔 상태로 보여주세요.")
                    else:
                        calibration_hand = visible_hand_labels[0]
                        if not neutral_calibration.has_offset(calibration_hand):
                            print(
                                f"[{args.profile}] {calibration_hand}: "
                                "C 키로 편 손 캘리브레이션을 먼저 하세요."
                            )
                        else:
                            neutral_calibration.start(
                                calibration_hand,
                                timestamp_seconds,
                                pose="closed",
                            )
                            angle_filters.pop(calibration_hand, None)
                            angle_deadbands.pop(calibration_hand, None)
                            print(
                                f"[{args.profile}] {calibration_hand} closed-pose "
                                f"calibration started. Keep a full fist for "
                                f"{args.calibration_seconds:.1f} seconds."
                            )
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if mujoco_controller is not None:
            mujoco_controller.close()
        if hardware_controller is not None:
            shutdown_fails = hardware_controller.close()
            if shutdown_fails:
                print(
                    f"WARNING: Torque-off not acknowledged by IDs: "
                    f"{list(shutdown_fails)}. Cut 5V power!",
                    flush=True,
                )
            else:
                print(
                    "Hardware Torque OFF; serial port closed cleanly.",
                    flush=True,
                )


if __name__ == "__main__":
    main()
