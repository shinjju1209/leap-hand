"""Interactive low-speed, single-joint motion test for LEAP Hand v1."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from hand_angles import ANGLE_NAMES
from leap_hand_hardware_controller import (
    SIM_MAX_RADIANS,
    SIM_MIN_RADIANS,
    LeapHandFeedback,
    LeapHandHardwareController,
    LeapHandHealth,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move exactly one LEAP Hand v1 joint by a small relative angle, "
            "then return to its starting pose."
        )
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM13")
    parser.add_argument("--joint", required=True, choices=ANGLE_NAMES)
    parser.add_argument(
        "--delta",
        type=float,
        default=5.0,
        help="Relative joint motion in degrees; negative reverses direction",
    )
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=30.0,
        help="Maximum joint speed in degrees per second (maximum allowed: 60)",
    )
    parser.add_argument(
        "--current-limit",
        type=int,
        default=300,
        help="Goal current limit in mA (maximum allowed by this test: 300)",
    )
    parser.add_argument("--hold-seconds", type=float, default=1.0)
    parser.add_argument("--command-rate", type=float, default=50.0)
    parser.add_argument(
        "--max-tracking-error",
        type=float,
        default=15.0,
        help="Torque-off threshold for command/feedback error in degrees",
    )
    parser.add_argument(
        "--max-temperature",
        type=float,
        default=50.0,
        help="Refuse or stop the test at or above this motor temperature",
    )
    parser.add_argument(
        "--motor-calibration-file",
        type=Path,
        help="Optional YAML created by leap_hand_motor_calibration.py",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if not np.isfinite(args.delta) or not 0.0 < abs(args.delta) <= 15.0:
        raise ValueError("--delta magnitude must be greater than 0 and at most 15 degrees")
    if (
        not np.isfinite(args.max_joint_speed)
        or not 0.0 < args.max_joint_speed <= 60.0
    ):
        raise ValueError("--max-joint-speed must be greater than 0 and at most 60")
    if not 1 <= args.current_limit <= 300:
        raise ValueError("--current-limit must be between 1 and 300 mA")
    if not np.isfinite(args.hold_seconds) or not 0.0 <= args.hold_seconds <= 5.0:
        raise ValueError("--hold-seconds must be between 0 and 5")
    if not np.isfinite(args.command_rate) or not 10.0 <= args.command_rate <= 100.0:
        raise ValueError("--command-rate must be between 10 and 100 Hz")
    if (
        not np.isfinite(args.max_tracking_error)
        or not 1.0 <= args.max_tracking_error <= 30.0
    ):
        raise ValueError("--max-tracking-error must be between 1 and 30 degrees")
    if not np.isfinite(args.max_temperature) or not 30.0 <= args.max_temperature <= 70.0:
        raise ValueError("--max-temperature must be between 30 and 70 C")


def build_relative_target(
    initial_degrees: Sequence[float],
    joint_name: str,
    delta_degrees: float,
) -> np.ndarray:
    """Return a one-joint target, refusing to move from an unsafe starting pose."""
    initial = np.asarray(initial_degrees, dtype=np.float64)
    if initial.shape != (16,) or not np.all(np.isfinite(initial)):
        raise ValueError("initial_degrees must contain 16 finite values")
    if joint_name not in ANGLE_NAMES:
        raise ValueError(f"Unknown joint: {joint_name}")

    minimum = np.rad2deg(SIM_MIN_RADIANS)
    maximum = np.rad2deg(SIM_MAX_RADIANS)
    outside = np.flatnonzero((initial < minimum) | (initial > maximum))
    if outside.size:
        names = [ANGLE_NAMES[index] for index in outside]
        raise RuntimeError(
            "Current pose is outside the configured safety bounds for: "
            f"{names}. Check horn alignment before enabling torque."
        )

    target = initial.copy()
    joint_index = ANGLE_NAMES.index(joint_name)
    target[joint_index] += float(delta_degrees)
    if target[joint_index] < minimum[joint_index] or target[joint_index] > maximum[joint_index]:
        raise RuntimeError(
            f"Requested target {target[joint_index]:.1f} deg is outside the "
            f"safe range for {joint_name}: {minimum[joint_index]:.1f} to "
            f"{maximum[joint_index]:.1f} deg. Reverse or reduce --delta."
        )
    return target


def assert_health(health: LeapHandHealth, max_temperature: float) -> None:
    error_ids = np.flatnonzero(health.hardware_errors)
    if error_ids.size:
        raise RuntimeError(f"Hardware error reported by motor IDs: {error_ids.tolist()}")
    hot_ids = np.flatnonzero(health.temperatures_celsius >= max_temperature)
    if hot_ids.size:
        temperatures = health.temperatures_celsius[hot_ids].tolist()
        raise RuntimeError(
            f"Motor temperature limit reached at IDs {hot_ids.tolist()}: "
            f"{temperatures} C"
        )


def assert_tracking(
    feedback: LeapHandFeedback,
    commanded_degrees: np.ndarray,
    max_tracking_error: float,
) -> None:
    errors = np.abs(feedback.positions_degrees - commanded_degrees)
    worst_index = int(np.argmax(errors))
    if errors[worst_index] > max_tracking_error:
        raise RuntimeError(
            f"Tracking error {errors[worst_index]:.1f} deg exceeded the limit "
            f"at {ANGLE_NAMES[worst_index]}"
        )


def move_to_target(
    controller: LeapHandHardwareController,
    target_degrees: np.ndarray,
    *,
    command_rate_hz: float,
    max_tracking_error: float,
    timeout_seconds: float,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> LeapHandFeedback:
    """Repeatedly command the target until the internal slew limiter reaches it."""
    period = 1.0 / command_rate_hz
    deadline = clock() + timeout_seconds
    while True:
        applied = controller.command_degrees(target_degrees)
        feedback = controller.read_feedback()
        assert_tracking(feedback, applied, max_tracking_error)
        if np.allclose(applied, target_degrees, atol=0.05, rtol=0.0):
            return feedback
        if clock() >= deadline:
            raise TimeoutError("Timed out while moving the selected joint")
        sleep(period)


def hold_position(
    controller: LeapHandHardwareController,
    duration_seconds: float,
    *,
    command_rate_hz: float,
    max_tracking_error: float,
) -> None:
    deadline = time.monotonic() + duration_seconds
    period = 1.0 / command_rate_hz
    while time.monotonic() < deadline:
        controller.heartbeat()
        feedback = controller.read_feedback()
        assert_tracking(
            feedback,
            controller.last_command_degrees,
            max_tracking_error,
        )
        time.sleep(period)


def _print_feedback(label: str, feedback: LeapHandFeedback) -> None:
    print(f"{label}: {np.round(feedback.positions_degrees, 2).tolist()} deg")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    controller = LeapHandHardwareController(
        args.port,
        current_limit_milliamps=args.current_limit,
        max_joint_speed_degrees_per_second=args.max_joint_speed,
        max_command_interval_seconds=0.1,
        motor_calibration=args.motor_calibration_file,
    )
    shutdown_failures: tuple[int, ...] = ()
    try:
        models = controller.connect()
        print(f"Connected with torque OFF: {args.port}")
        print(f"Motor IDs/model numbers: {models}")
        initial_feedback = controller.read_feedback()
        health = controller.read_health()
        assert_health(health, args.max_temperature)
        initial_degrees = initial_feedback.positions_degrees.copy()
        target_degrees = build_relative_target(
            initial_degrees,
            args.joint,
            args.delta,
        )
        joint_index = ANGLE_NAMES.index(args.joint)
        _print_feedback("Initial positions", initial_feedback)
        print(f"Temperatures: {health.temperatures_celsius.tolist()} C")
        print(f"Input voltages: {health.input_voltages.tolist()} V")
        print(
            f"PLAN: {args.joint} {initial_degrees[joint_index]:.1f} -> "
            f"{target_degrees[joint_index]:.1f} deg at no more than "
            f"{args.max_joint_speed:.1f} deg/s, then return"
        )
        controller.configure()
        print("Low-current control configured; torque remains OFF.")
        confirmation = input(
            "Clear the hand and type MOVE to enable torque (anything else cancels): "
        )
        if confirmation.strip() != "MOVE":
            print("Cancelled. Torque was never enabled.")
            return

        # Re-sample after the operator confirmation so the relative motion and
        # return pose use the position immediately before torque is enabled.
        initial_feedback = controller.read_feedback()
        assert_health(controller.read_health(), args.max_temperature)
        initial_degrees = initial_feedback.positions_degrees.copy()
        target_degrees = build_relative_target(
            initial_degrees,
            args.joint,
            args.delta,
        )
        print(
            f"FINAL PLAN: {args.joint} {initial_degrees[joint_index]:.1f} -> "
            f"{target_degrees[joint_index]:.1f} deg at no more than "
            f"{args.max_joint_speed:.1f} deg/s, then return"
        )
        controller.enable_torque()
        timeout = abs(args.delta) / args.max_joint_speed + 2.0
        print("Torque ON; moving the selected joint only.")
        reached_feedback = move_to_target(
            controller,
            target_degrees,
            command_rate_hz=args.command_rate,
            max_tracking_error=args.max_tracking_error,
            timeout_seconds=timeout,
        )
        _print_feedback("Reached target", reached_feedback)
        hold_position(
            controller,
            args.hold_seconds,
            command_rate_hz=args.command_rate,
            max_tracking_error=args.max_tracking_error,
        )
        assert_health(controller.read_health(), args.max_temperature)

        print("Returning to the exact starting pose.")
        returned_feedback = move_to_target(
            controller,
            initial_degrees,
            command_rate_hz=args.command_rate,
            max_tracking_error=args.max_tracking_error,
            timeout_seconds=timeout,
        )
        _print_feedback("Returned positions", returned_feedback)
        print("Single-joint test completed; disabling torque.")
    finally:
        shutdown_failures = controller.close()
        if shutdown_failures:
            print(
                "WARNING: Torque-off was not acknowledged by IDs "
                f"{list(shutdown_failures)}. Cut 5V power immediately."
            )
        else:
            print("Torque OFF; serial port closed.")


if __name__ == "__main__":
    main()
