"""Torque-off diagnostics and explicit no-motion torque test for LEAP Hand v1."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from leap_hand_hardware_controller import LeapHandHardwareController


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely verify a 16-motor LEAP Hand v1 connection."
    )
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM13")
    parser.add_argument("--baudrate", type=int, default=4_000_000)
    parser.add_argument(
        "--current-limit",
        type=int,
        default=300,
        help="Goal current limit in mA; start at 300 or lower",
    )
    parser.add_argument(
        "--watchdog-ms",
        type=int,
        default=500,
        help="DYNAMIXEL bus watchdog timeout",
    )
    parser.add_argument(
        "--max-joint-speed",
        type=float,
        default=120.0,
        help="Maximum commanded speed for every joint in degrees per second",
    )
    parser.add_argument(
        "--max-command-interval",
        type=float,
        default=0.1,
        help="Largest elapsed time used by the command slew limiter",
    )
    parser.add_argument(
        "--motor-calibration-file",
        type=Path,
        help="Optional YAML created by leap_hand_motor_calibration.py",
    )
    parser.add_argument(
        "--torque-test",
        action="store_true",
        help="Explicitly enable torque while holding each present position",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=2.0,
        help="Duration of the no-motion torque test",
    )
    return parser.parse_args()


def _print_vector(label: str, values: np.ndarray, unit: str) -> None:
    print(f"{label}: {np.round(values, 2).tolist()} {unit}")


def main() -> None:
    args = parse_args()
    if args.hold_seconds <= 0.0:
        raise ValueError("--hold-seconds must be greater than zero")

    controller = LeapHandHardwareController(
        args.port,
        baudrate=args.baudrate,
        current_limit_milliamps=args.current_limit,
        bus_watchdog_milliseconds=args.watchdog_ms,
        max_joint_speed_degrees_per_second=args.max_joint_speed,
        max_command_interval_seconds=args.max_command_interval,
        motor_calibration=args.motor_calibration_file,
    )
    shutdown_failures: tuple[int, ...] = ()
    try:
        models = controller.connect()
        print(f"Connected with torque OFF: {args.port}")
        print(f"Motor IDs/model numbers: {models}")
        feedback = controller.read_feedback()
        health = controller.read_health()
        _print_vector("Positions", feedback.positions_degrees, "deg")
        _print_vector("Temperatures", health.temperatures_celsius, "C")
        _print_vector("Input voltages", health.input_voltages, "V")
        print(f"Hardware errors: {health.hardware_errors.tolist()}")

        if not args.torque_test:
            print("Torque was never enabled. Add --torque-test only after inspection.")
            return

        print("Configuring low-current control with torque OFF...")
        controller.configure()
        print("Enabling torque at the PRESENT positions; no motion command is sent.")
        controller.enable_torque()
        deadline = time.monotonic() + args.hold_seconds
        while time.monotonic() < deadline:
            # Re-send the exact seeded raw goal; do not clip or convert it.
            controller.heartbeat()
            time.sleep(0.05)
        print("No-motion torque test completed; disabling torque.")
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
