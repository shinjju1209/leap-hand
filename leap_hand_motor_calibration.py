"""Record the physical open-hand motor position for every LEAP Hand v1 motor."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from hardware_calibration import HardwareMotorCalibration
from leap_hand_hardware_controller import LeapHandHardwareController


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "With torque OFF, record median raw positions while the LEAP Hand v1 "
            "is held fully open."
        )
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port (default: /dev/ttyUSB0)",
    )
    parser.add_argument("--baudrate", type=int, default=4_000_000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("calibration/hardware_motors.yaml"),
        help="Destination YAML file",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=31,
        help="Number of torque-off raw-position samples to record",
    )
    parser.add_argument(
        "--sample-period",
        type=float,
        default=0.05,
        help="Seconds between raw-position samples",
    )
    parser.add_argument(
        "--max-temperature",
        type=float,
        default=50.0,
        help="Refuse calibration at or above this motor temperature",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow replacing an existing output file",
    )
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.baudrate <= 0:
        raise ValueError("--baudrate must be positive")
    if not 3 <= args.samples <= 301:
        raise ValueError("--samples must be between 3 and 301")
    if not np.isfinite(args.sample_period) or not 0.01 <= args.sample_period <= 1.0:
        raise ValueError("--sample-period must be between 0.01 and 1.0 seconds")
    if not np.isfinite(args.max_temperature) or not 30.0 <= args.max_temperature <= 70.0:
        raise ValueError("--max-temperature must be between 30 and 70 C")


def assert_healthy_for_calibration(controller: LeapHandHardwareController, max_temperature: float) -> None:
    health = controller.read_health()
    error_ids = np.flatnonzero(health.hardware_errors)
    if error_ids.size:
        raise RuntimeError(
            f"Hardware errors are present for motor IDs: {error_ids.tolist()}"
        )
    hot_ids = np.flatnonzero(health.temperatures_celsius >= max_temperature)
    if hot_ids.size:
        raise RuntimeError(
            f"Motors above {max_temperature:.1f} C: {hot_ids.tolist()}"
        )


def collect_open_pose_samples(
    controller: LeapHandHardwareController,
    *,
    samples: int,
    sample_period_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    """Collect raw motor radians while torque is guaranteed off."""
    if controller.torque_enabled:
        raise RuntimeError("Torque must be off before recording calibration")
    readings = np.zeros((samples, 16), dtype=np.float64)
    for index in range(samples):
        readings[index] = controller.read_motor_positions_radians()
        if index + 1 < samples:
            sleep(sample_period_seconds)
    return readings


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"Calibration file already exists: {args.output}. Use --force to replace it."
        )

    controller = LeapHandHardwareController(args.port, baudrate=args.baudrate)
    shutdown_failures: tuple[int, ...] = ()
    try:
        models = controller.connect()
        print(f"Connected with torque OFF: {args.port}")
        print(f"Motor IDs/model numbers: {models}")
        assert_healthy_for_calibration(controller, args.max_temperature)
        print(
            "Place the physical hand fully open in its intended neutral pose. "
            "Torque remains OFF throughout this procedure."
        )
        confirmation = input("Type RECORD to capture the open-pose calibration: ")
        if confirmation.strip() != "RECORD":
            print("Cancelled. No calibration file was written.")
            return

        readings = collect_open_pose_samples(
            controller,
            samples=args.samples,
            sample_period_seconds=args.sample_period,
        )
        assert_healthy_for_calibration(controller, args.max_temperature)
        calibration = HardwareMotorCalibration.from_open_pose_samples(
            controller.motor_ids,
            readings,
        )
        output = calibration.save(args.output)
        spread_degrees = np.rad2deg(np.ptp(readings, axis=0))
        print(f"Saved open-pose calibration: {output.resolve()}")
        print(
            "Per-motor sample spread (deg): "
            f"{np.round(spread_degrees, 3).tolist()}"
        )
        print(
            "All signs are initialized to +1. Confirm every joint direction with "
            "leap_hand_joint_test.py and change only the affected YAML sign to -1."
        )
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
