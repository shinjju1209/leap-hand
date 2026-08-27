"""Demonstrate middle finger gesture (Index, Ring, Thumb folded; Middle extended) in MuJoCo and Hardware."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from hand_angles import ANGLE_NAMES
from leap_hand_hardware_controller import LeapHandHardwareController
from mujoco_hand_controller import MujocoHandController

DEFAULT_MOTOR_CALIB_PATH = Path("calibration/hardware_motors.yaml")

# Pose definition: Middle finger extended, Index/Ring/Thumb folded
_CLOSED_INDEX = {
    "index_mcp_flex": 75.0,
    "index_pip_flex": 85.0,
    "index_dip_flex": 65.0,
}
_EXTENDED_MIDDLE = {
    "middle_mcp_side": 0.0,
    "middle_mcp_flex": 0.0,
    "middle_pip_flex": 0.0,
    "middle_dip_flex": 0.0,
}
_CLOSED_RING = {
    "ring_mcp_flex": 75.0,
    "ring_pip_flex": 85.0,
    "ring_dip_flex": 65.0,
}
_CLOSED_THUMB = {
    "thumb_cmc_flex": 55.0,
    "thumb_mcp_flex": 65.0,
    "thumb_ip_flex": 50.0,
}


def make_middle_finger_posture() -> np.ndarray:
    """Build the 16-angle target vector for middle finger gesture."""
    posture = np.zeros(len(ANGLE_NAMES), dtype=np.float64)
    target_dict = _CLOSED_INDEX | _EXTENDED_MIDDLE | _CLOSED_RING | _CLOSED_THUMB
    for angle_name, value in target_dict.items():
        posture[ANGLE_NAMES.index(angle_name)] = float(value)
    return posture


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview or execute middle finger gesture on LEAP Hand (MuJoCo / Hardware / Both)."
    )
    parser.add_argument(
        "--mode",
        choices=("mujoco", "hardware", "both"),
        default="mujoco",
        help="Target execution mode (default: mujoco)",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port for real LEAP Hand hardware (default: /dev/ttyUSB0)",
    )
    parser.add_argument(
        "--motor-calibration-file",
        type=Path,
        default=DEFAULT_MOTOR_CALIB_PATH,
        help="Path to motor zero calibration YAML",
    )
    parser.add_argument(
        "--position-p-gain",
        type=int,
        default=450,
        help="DYNAMIXEL Position P Gain (default: 450)",
    )
    parser.add_argument(
        "--position-d-gain",
        type=int,
        default=1200,
        help="DYNAMIXEL Position D Gain (default: 1200)",
    )
    parser.add_argument(
        "--current-limit",
        type=int,
        default=350,
        help="Motor current limit in mA (default: 350 mA)",
    )
    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=0.8,
        help="Transition time in seconds to form gesture (default: 0.8s)",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=2.0,
        help="Hold time in seconds for the gesture (default: 2.0s)",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of gesture cycles to run (default: 1)",
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
    return parser.parse_args(argv)


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
    target_gesture = make_middle_finger_posture()

    hardware_ctrl: LeapHandHardwareController | None = None
    mujoco_ctrl: MujocoHandController | None = None
    viewer = None

    try:
        # 1. Initialize Hardware (if requested)
        if args.mode in ("hardware", "both"):
            motor_calib = (
                args.motor_calibration_file
                if args.motor_calibration_file.is_file()
                else None
            )
            print(f"[HARDWARE] Connecting to LEAP Hand on {args.port}...")
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
            input("Press [ENTER] to enable Torque and start gesture demo... ")
            hardware_ctrl.enable_torque()
            print("[HARDWARE] Torque ON. Starting gesture...")

        # 2. Initialize MuJoCo (if requested)
        if args.mode in ("mujoco", "both"):
            print("[MUJOCO] Initializing MuJoCo simulation environment...")
            mujoco_ctrl = MujocoHandController()
            viewer = mujoco_ctrl.launch_viewer()
            print("[MUJOCO] Viewer ready.")

        # 3. Main Loop
        current_pose = neutral_pose.copy()
        cycle_count = 0

        while True:
            cycle_count += 1
            if not args.loop and cycle_count > args.cycles:
                break

            print(f"\n--- Cycle {cycle_count}{' (Continuous Loop)' if args.loop else f' / {args.cycles}'} ---")
            print("  ▶ 🖕 중지 펴기 (Middle Finger Extension)...", flush=True)

            if not interpolate_and_send(
                hardware_ctrl,
                mujoco_ctrl,
                viewer,
                current_pose,
                target_gesture,
                args.transition_seconds,
                args.update_hz,
            ):
                return 1

            current_pose = target_gesture
            if args.hold_seconds > 0.0:
                print(f"  ⏳ {args.hold_seconds:.1f}초간 자세 유지 (Holding)...", flush=True)
                if not hold_pose(
                    hardware_ctrl,
                    mujoco_ctrl,
                    viewer,
                    current_pose,
                    args.hold_seconds,
                    args.update_hz,
                ):
                    return 1

            print("  ↩ 편 손 복귀 (Return to Open Neutral)...", flush=True)
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

            if args.hold_seconds > 0.0 and (args.loop or cycle_count < args.cycles):
                time.sleep(0.5)

        print("\n[SUCCESS] Middle finger gesture demo finished successfully.")
        return 0

    except KeyboardInterrupt:
        print("\n[STOP] Stopped safely by user.")
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
