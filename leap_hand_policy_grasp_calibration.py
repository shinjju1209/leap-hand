"""Virtually anchor an assembled LEAP Hand to an RL policy starting grasp."""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np
import yaml

from hand_angles import ANGLE_NAMES
from hardware_calibration import HardwareMotorCalibration
from leap_hand_hardware_controller import LeapHandHardwareController


DEFAULT_CONFIG_PATH = Path("configs/inhand_cube_rotation.yaml")
DEFAULT_BASE_CALIBRATION_PATH = Path("calibration/hardware_motors_local.yaml")
DEFAULT_OUTPUT_PATH = Path("calibration/hardware_motors_policy.yaml")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "With torque OFF, label the current physical grasp as the RL "
            "policy's configured starting pose and save fixed virtual offsets."
        )
    )
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--baudrate", type=int, default=4_000_000)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="RL config containing default_joint_pose_radians",
    )
    parser.add_argument(
        "--base-calibration-file",
        type=Path,
        default=DEFAULT_BASE_CALIBRATION_PATH,
        help="Existing calibration used for motor IDs and joint directions",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="New calibration YAML; the base file is never modified",
    )
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--sample-period", type=float, default=0.05)
    parser.add_argument("--max-temperature", type=float, default=50.0)
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


def load_policy_reference_pose(config_path: str | Path) -> np.ndarray:
    source = Path(config_path)
    try:
        payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as error:
        raise OSError(f"Could not read RL config: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError("RL config must contain a YAML mapping")
    pose = np.asarray(payload.get("default_joint_pose_radians"), dtype=np.float64)
    if pose.shape != (16,) or not np.all(np.isfinite(pose)):
        raise ValueError("default_joint_pose_radians must contain 16 finite values")
    return pose


def collect_raw_samples(
    controller: LeapHandHardwareController,
    *,
    samples: int,
    sample_period_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> np.ndarray:
    """Read raw motor positions while torque remains disabled."""
    if controller.torque_enabled:
        raise RuntimeError("Torque must be off while anchoring the policy grasp")
    readings = np.zeros((samples, 16), dtype=np.float64)
    for index in range(samples):
        readings[index] = controller.read_motor_positions_radians()
        if index + 1 < samples:
            sleep(sample_period_seconds)
    return readings


def assert_healthy(
    controller: LeapHandHardwareController,
    max_temperature: float,
) -> None:
    health = controller.read_health()
    error_indices = np.flatnonzero(health.hardware_errors)
    if error_indices.size:
        motor_ids = [controller.motor_ids[index] for index in error_indices]
        raise RuntimeError(f"Hardware errors are present for motor IDs: {motor_ids}")
    hot_indices = np.flatnonzero(health.temperatures_celsius >= max_temperature)
    if hot_indices.size:
        motor_ids = [controller.motor_ids[index] for index in hot_indices]
        raise RuntimeError(
            f"Motors at or above {max_temperature:.1f} C: {motor_ids}"
        )


def build_policy_grasp_calibration(
    base: HardwareMotorCalibration,
    raw_samples: Sequence[Sequence[float]],
    reference_pose_radians: Sequence[float],
) -> HardwareMotorCalibration:
    return HardwareMotorCalibration.from_reference_pose_samples(
        base.motor_ids,
        raw_samples,
        reference_pose_radians,
        signs=base.signs,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    validate_args(args)
    if args.output.exists() and not args.force:
        raise FileExistsError(
            f"Output already exists: {args.output}. Use --force to replace it."
        )

    base = HardwareMotorCalibration.load(args.base_calibration_file)
    reference_pose = load_policy_reference_pose(args.config)
    controller = LeapHandHardwareController(
        args.port,
        baudrate=args.baudrate,
        motor_calibration=base,
    )
    try:
        models = controller.connect()
        print(f"Connected with torque OFF: {args.port}")
        print(f"Motor IDs/model numbers: {models}")
        assert_healthy(controller, args.max_temperature)
        print("\nPOLICY GRASP VIRTUAL ALIGNMENT")
        print("Torque will remain OFF and no movement command will be sent.")
        print(
            "Place the cube and manually arrange the fingers in the stable "
            "physical grasp that should represent the RL episode start."
        )
        print(
            "This pose will be labelled as the configured policy pose; it will "
            "not be labelled as sixteen zero-degree joints."
        )
        confirmation = input("Type ANCHOR to record this policy grasp: ")
        if confirmation.strip() != "ANCHOR":
            print("Cancelled. No calibration file was written.")
            return

        readings = collect_raw_samples(
            controller,
            samples=args.samples,
            sample_period_seconds=args.sample_period,
        )
        assert_healthy(controller, args.max_temperature)
        calibration = build_policy_grasp_calibration(
            base,
            readings,
            reference_pose,
        )
        output = calibration.save(args.output)

        median_raw = np.median(readings, axis=0)
        decoded_pose = calibration.motor_to_sim_radians(median_raw)
        residual_degrees = np.rad2deg(decoded_pose - reference_pose)
        offset_change_degrees = np.rad2deg(
            calibration.open_motor_radians - base.open_motor_radians
        )
        sample_spread_degrees = np.rad2deg(np.ptp(readings, axis=0))

        print(f"Saved policy-grasp calibration: {output.resolve()}")
        print(
            "Maximum anchoring residual: "
            f"{float(np.max(np.abs(residual_degrees))):.4f} deg"
        )
        print("Virtual-zero changes relative to the base calibration:")
        for name, change in zip(ANGLE_NAMES, offset_change_degrees):
            print(f"  {name:22s} {change:+8.2f} deg")
        print(
            "Per-motor sample spread (deg): "
            f"{np.round(sample_spread_degrees, 3).tolist()}"
        )
        print(
            "Use this file for hardware checks and RL deployment. The offsets "
            "remain fixed, so later policy motion is preserved relative to this grasp."
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
