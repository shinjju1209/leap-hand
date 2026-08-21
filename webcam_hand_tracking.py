"""Live webcam hand-landmark visualization with MediaPipe Tasks."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import mediapipe as mp

from deadband_filter import AngleDeadband
from hand_angles import DISPLAY_NAMES, calculate_leap_control_angles
from one_euro_filter import OneEuroFilter
from rps_gesture import GestureClassification, GestureStabilizer, classify_rps_gesture
from rps_moves import MOVE_NAMES
from rps_rounds import CsvRoundRecorder, RpsRoundSession


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)

FINGERTIP_IDS = {4, 8, 12, 16, 20}


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    # DirectShow normally opens integrated cameras faster on Windows.
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
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


def main() -> None:
    args = parse_args()
    if args.deadband < 0.0:
        raise ValueError("--deadband must be zero or greater")
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

    processing_steps = []
    if args.filter:
        processing_steps.append("One Euro")
    if args.deadband > 0.0:
        processing_steps.append(f"deadband {args.deadband:.1f}")
    processing_label = " + ".join(processing_steps) or None

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

    capture = open_camera(args.camera, args.width, args.height)
    window_name = "MediaPipe Hand Tracking - Q/ESC to quit"
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
                                angles,
                                timestamp_seconds,
                            )
                        else:
                            filtered_angles = angles

                        if args.deadband > 0.0:
                            angle_deadband = angle_deadbands.get(filter_key)
                            if angle_deadband is None:
                                angle_deadband = AngleDeadband(args.deadband)
                                angle_deadbands[filter_key] = angle_deadband
                            command_angles = angle_deadband.filter(filtered_angles)
                        else:
                            command_angles = filtered_angles

                        draw_angle_panel(
                            frame,
                            angles,
                            command_angles,
                            hand_label,
                            index,
                            processing_label,
                        )

                now = time.perf_counter()
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
                cv2.putText(
                    frame,
                    (
                        f"Move, open, close, rotate | raw > {processing_label}"
                        if processing_label
                        else "Move, open, close, rotate | processing OFF"
                    ),
                    (18, 66),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )
                draw_round_status(frame, round_session)

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    break
                robot_move_keys = {
                    ord("1"): "rock",
                    ord("2"): "paper",
                    ord("3"): "scissors",
                }
                if key in robot_move_keys:
                    robot_move = robot_move_keys[key]
                    round_session.start_round(robot_move)
                    print(f"Robot move set to {robot_move}; waiting for human.", flush=True)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
