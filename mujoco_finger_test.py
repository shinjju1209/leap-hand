"""Move each finger of the right LEAP Hand separately in MuJoCo."""

from __future__ import annotations

import argparse
import time

import numpy as np

from hand_angles import ANGLE_NAMES
from mujoco_hand_controller import MujocoHandController


FINGER_TARGETS_DEGREES = {
    "Index": {
        "index_mcp_flex": 70.0,
        "index_pip_flex": 70.0,
        "index_dip_flex": 50.0,
    },
    "Middle": {
        "middle_mcp_flex": 70.0,
        "middle_pip_flex": 70.0,
        "middle_dip_flex": 50.0,
    },
    "Ring": {
        "ring_mcp_flex": 70.0,
        "ring_pip_flex": 70.0,
        "ring_dip_flex": 50.0,
    },
    "Thumb": {
        "thumb_cmc_flex": 45.0,
        "thumb_mcp_flex": 50.0,
        "thumb_ip_flex": 45.0,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move the right LEAP Hand fingers one at a time in MuJoCo."
    )
    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=1.0,
        help="Time to close or open one finger",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.5,
        help="Time to hold each closed or open pose",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=1,
        help="Number of complete four-finger test cycles",
    )
    parser.add_argument(
        "--update-hz",
        type=float,
        default=60.0,
        help="Target update and viewer refresh rate",
    )
    return parser.parse_args()


def make_pose(changes: dict[str, float]) -> np.ndarray:
    """Create a 16-angle pose from named joint changes."""
    pose = np.zeros(16, dtype=np.float64)
    for angle_name, value in changes.items():
        pose[ANGLE_NAMES.index(angle_name)] = value
    return pose


def _advance_realtime(
    controller: MujocoHandController,
    viewer,
    duration_seconds: float,
    update_hz: float,
    start_pose: np.ndarray,
    target_pose: np.ndarray,
) -> bool:
    """Interpolate a pose while advancing physics at approximately real time."""
    frame_seconds = 1.0 / update_hz
    frame_count = max(1, int(round(duration_seconds * update_hz)))

    for frame_index in range(1, frame_count + 1):
        if not viewer.is_running():
            return False

        frame_started = time.perf_counter()
        alpha = frame_index / frame_count
        smooth_alpha = alpha * alpha * (3.0 - 2.0 * alpha)
        command = start_pose + smooth_alpha * (target_pose - start_pose)
        controller.set_target_degrees(command)
        controller.step_for(frame_seconds)
        controller.sync_viewer()

        remaining = frame_seconds - (time.perf_counter() - frame_started)
        if remaining > 0.0:
            time.sleep(remaining)

    return True


def _hold_pose(
    controller: MujocoHandController,
    viewer,
    pose: np.ndarray,
    duration_seconds: float,
    update_hz: float,
) -> bool:
    """Hold a pose while continuing physics and viewer updates."""
    return _advance_realtime(
        controller,
        viewer,
        duration_seconds,
        update_hz,
        pose,
        pose,
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.transition_seconds <= 0.0:
        raise ValueError("--transition-seconds must be greater than zero")
    if args.hold_seconds < 0.0:
        raise ValueError("--hold-seconds must be zero or greater")
    if args.cycles < 1:
        raise ValueError("--cycles must be at least one")
    if args.update_hz <= 0.0:
        raise ValueError("--update-hz must be greater than zero")


def main() -> None:
    args = parse_args()
    validate_args(args)
    neutral_pose = np.zeros(16, dtype=np.float64)

    with MujocoHandController() as controller:
        viewer = controller.launch_viewer()
        current_pose = neutral_pose.copy()
        print("MuJoCo right LEAP Hand finger test started.", flush=True)

        for cycle in range(1, args.cycles + 1):
            print(f"Cycle {cycle}/{args.cycles}", flush=True)
            for finger_name, changes in FINGER_TARGETS_DEGREES.items():
                target_pose = make_pose(changes)
                print(f"  {finger_name}: close", flush=True)
                if not _advance_realtime(
                    controller,
                    viewer,
                    args.transition_seconds,
                    args.update_hz,
                    current_pose,
                    target_pose,
                ):
                    return
                current_pose = target_pose
                if args.hold_seconds > 0.0 and not _hold_pose(
                    controller,
                    viewer,
                    current_pose,
                    args.hold_seconds,
                    args.update_hz,
                ):
                    return

                print(f"  {finger_name}: open", flush=True)
                if not _advance_realtime(
                    controller,
                    viewer,
                    args.transition_seconds,
                    args.update_hz,
                    current_pose,
                    neutral_pose,
                ):
                    return
                current_pose = neutral_pose.copy()
                if args.hold_seconds > 0.0 and not _hold_pose(
                    controller,
                    viewer,
                    current_pose,
                    args.hold_seconds,
                    args.update_hz,
                ):
                    return

        print("Finger test completed.", flush=True)


if __name__ == "__main__":
    main()
