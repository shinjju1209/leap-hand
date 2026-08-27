"""Execute Rock-Paper-Scissors posture sequence directly on LEAP Hand hardware without webcam."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from leap_hand_hardware_controller import LeapHandHardwareController
from mujoco_hand_controller import MujocoHandController
from rps.moves import MOVE_NAMES
from rps.postures import get_posture

DEFAULT_MOTOR_CALIB_PATH = Path("calibration/hardware_motors.yaml")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Execute Rock-Paper-Scissors postures sequentially on LEAP Hand (Hardware / MuJoCo / Both)."
    )
    parser.add_argument(
        "moves",
        nargs="*",
        choices=MOVE_NAMES,
        help="Moves to play in order (default: rock paper scissors)",
    )
    parser.add_argument(
        "--mode",
        choices=("hardware", "mujoco", "both"),
        default="hardware",
        help="Target execution mode (default: hardware)",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port for real LEAP Hand hardware",
    )
    parser.add_argument(
        "--motor-calibration-file",
        type=Path,
        default=DEFAULT_MOTOR_CALIB_PATH,
        help="Path to motor zero calibration YAML (default: calibration/hardware_motors.yaml)",
    )
    parser.add_argument(
        "--current-limit",
        type=int,
        default=300,
        help="Motor current limit in mA (default: 300 mA)",
    )
    parser.add_argument(
        "--position-p-gain",
        type=int,
        default=450,
        help="DYNAMIXEL Position P Gain (default: 450, range: 0~16383)",
    )
    parser.add_argument(
        "--position-d-gain",
        type=int,
        default=1200,
        help="DYNAMIXEL Position D Gain for damping/smoothness (default: 1200, range: 0~16383)",
    )
    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=0.8,
        help="Transition time in seconds between gestures (default: 0.8s)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=1.5,
        help="Hold time in seconds for each gesture (default: 1.5s)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of complete RPS cycles to run (default: 1)",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Run continuously in a loop until Ctrl+C is pressed",
    )
    parser.add_argument(
        "--update-hz",
        type=float,
        default=50.0,
        help="Control update rate in Hz (default: 50.0 Hz)",
    )
    args = parser.parse_args(argv)
    if not args.moves:
        args.moves = list(MOVE_NAMES)
    return args


def interpolate_and_send(
    hardware_controller: LeapHandHardwareController | None,
    mujoco_controller: MujocoHandController | None,
    viewer,
    start_pose: np.ndarray,
    target_pose: np.ndarray,
    duration_seconds: float,
    update_hz: float,
) -> bool:
    """Smoothly interpolate from start_pose to target_pose using S-curve."""
    dt = 1.0 / update_hz
    steps = max(1, int(round(duration_seconds * update_hz)))

    for step in range(1, steps + 1):
        t0 = time.perf_counter()
        alpha = step / steps
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        command_deg = start_pose + smooth_alpha * (target_pose - start_pose)

        # Hardware
        if hardware_controller is not None:
            hardware_controller.command_degrees(command_deg)
            if step % 5 == 0:
                health = hardware_controller.read_health()
                if np.any(health.hardware_errors != 0):
                    print(f"\n[ERROR] Hardware error detected: {health.hardware_errors}")
                    return False
                if np.any(health.temperatures_celsius >= 50.0):
                    print(f"\n[ERROR] Overheat detected: {health.temperatures_celsius}")
                    return False

        # MuJoCo
        if mujoco_controller is not None:
            if viewer is not None and not viewer.is_running():
                return False
            mujoco_controller.set_target_degrees(command_deg)
            mujoco_controller.step_for(dt)
            mujoco_controller.sync_viewer()

        elapsed = time.perf_counter() - t0
        remaining = dt - elapsed
        if remaining > 0.0:
            time.sleep(remaining)

    return True


def hold_pose(
    hardware_controller: LeapHandHardwareController | None,
    mujoco_controller: MujocoHandController | None,
    viewer,
    pose: np.ndarray,
    duration_seconds: float,
    update_hz: float,
) -> bool:
    """Hold a target pose for duration_seconds."""
    return interpolate_and_send(
        hardware_controller,
        mujoco_controller,
        viewer,
        pose,
        pose,
        duration_seconds,
        update_hz,
    )


def main() -> int:
    args = parse_args()
    neutral_pose = np.zeros(16, dtype=np.float64)

    hardware_ctrl: LeapHandHardwareController | None = None
    mujoco_ctrl: MujocoHandController | None = None
    viewer = None

    DISPLAY_MAP = {
        "rock": "✊ 바위 (ROCK)",
        "paper": "🖐️ 보 (PAPER)",
        "scissors": "✌️ 가위 (SCISSORS)",
    }

    try:
        # 1. Initialize Hardware
        if args.mode in ("hardware", "both"):
            motor_calib = (
                args.motor_calibration_file
                if args.motor_calibration_file.is_file()
                else None
            )
            print(f"[HARDWARE] Connecting to LEAP Hand on {args.port}...")
            if motor_calib:
                print(f"[HARDWARE] Using motor calibration: {motor_calib}")
            else:
                print("[WARNING] No motor calibration file found! Using uncalibrated offsets.")

            hardware_ctrl = LeapHandHardwareController(
                port=args.port,
                current_limit_milliamps=args.current_limit,
                position_p_gain=args.position_p_gain,
                position_d_gain=args.position_d_gain,
                motor_calibration=motor_calib,
            )
            hardware_ctrl.connect()
            hardware_ctrl.configure()
            print("[HARDWARE] Connected successfully with torque OFF.")

            print("\n" + "=" * 60)
            print("  LEAP Hand Rock-Paper-Scissors Sequence Ready")
            print(f"  Mode: {args.mode.upper()} | Sequence: {' -> '.join(args.moves).upper()}")
            print(f"  Cycles: {'Infinite' if args.loop else args.cycles} | Current limit: {args.current_limit}mA")
            print("=" * 60)
            input("Press [ENTER] to enable Torque and start RPS sequence (or Ctrl+C to cancel)... ")

            hardware_ctrl.enable_torque()
            print("[HARDWARE] Torque ON. Starting sequence...")

        # 2. Initialize MuJoCo
        if args.mode in ("mujoco", "both"):
            print("[MUJOCO] Initializing MuJoCo simulation environment...")
            mujoco_ctrl = MujocoHandController()
            viewer = mujoco_ctrl.launch_viewer()

        # 3. Main Sequence Loop
        current_pose = neutral_pose.copy()
        cycle_count = 0

        while True:
            cycle_count += 1
            if not args.loop and cycle_count > args.cycles:
                break

            print(f"\n--- Cycle {cycle_count}{' (Continuous Loop)' if args.loop else f' / {args.cycles}'} ---")

            for move in args.moves:
                label = DISPLAY_MAP.get(move, move.upper())
                target_pose = get_posture(move)

                print(f"  ▶ {label}...", flush=True)
                if not interpolate_and_send(
                    hardware_ctrl,
                    mujoco_ctrl,
                    viewer,
                    current_pose,
                    target_pose,
                    args.transition_seconds,
                    args.update_hz,
                ):
                    return 1

                current_pose = target_pose
                if args.hold_seconds > 0.0:
                    if not hold_pose(
                        hardware_ctrl,
                        mujoco_ctrl,
                        viewer,
                        current_pose,
                        args.hold_seconds,
                        args.update_hz,
                    ):
                        return 1

            # Smoothly return to neutral open hand at end of cycle if not already
            if not np.array_equal(current_pose, neutral_pose):
                print("  ↩ 편 손 중립 복귀 (Neutral Return)...", flush=True)
                if not interpolate_and_send(
                    hardware_ctrl,
                    mujoco_ctrl,
                    viewer,
                    current_pose,
                    neutral_pose,
                    args.transition_seconds,
                    args.update_hz,
                ):
                    return 1
                current_pose = neutral_pose.copy()

        print("\n[SUCCESS] RPS sequence completed successfully.")
        return 0

    except KeyboardInterrupt:
        print("\n[STOP] Keyboard interrupt received. Stopping safely...")
        return 0
    except Exception as exc:
        print(f"\n[ERROR] Exception during execution: {exc}")
        return 1
    finally:
        if hardware_ctrl is not None:
            print("[HARDWARE] Disabling torque and closing serial port...")
            hardware_ctrl.close()
            print("[HARDWARE] Clean shutdown complete.")
        if mujoco_ctrl is not None:
            mujoco_ctrl.close()


if __name__ == "__main__":
    sys.exit(main())
