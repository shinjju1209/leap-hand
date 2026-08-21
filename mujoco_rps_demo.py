"""Preview the LEAP Hand rock-paper-scissors postures in MuJoCo."""

from __future__ import annotations

import argparse

import numpy as np

from mujoco_finger_test import _advance_realtime, _hold_pose
from mujoco_hand_controller import MujocoHandController
from rps_postures import MOVE_NAMES, get_posture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview named rock-paper-scissors postures in MuJoCo."
    )
    parser.add_argument(
        "moves",
        nargs="*",
        choices=MOVE_NAMES,
        default=list(MOVE_NAMES),
        help="Moves to show in order (default: rock paper scissors)",
    )
    parser.add_argument("--transition-seconds", type=float, default=0.8)
    parser.add_argument("--hold-seconds", type=float, default=1.5)
    parser.add_argument("--update-hz", type=float, default=60.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.transition_seconds <= 0.0:
        raise ValueError("--transition-seconds must be greater than zero")
    if args.hold_seconds < 0.0:
        raise ValueError("--hold-seconds must be zero or greater")
    if args.update_hz <= 0.0:
        raise ValueError("--update-hz must be greater than zero")


def main() -> None:
    args = parse_args()
    validate_args(args)
    neutral = np.zeros(16, dtype=np.float64)

    with MujocoHandController() as controller:
        viewer = controller.launch_viewer()
        current = neutral.copy()
        print("MuJoCo RPS posture preview started.", flush=True)

        for move in args.moves:
            target = get_posture(move)
            print(f"  {move.upper()}", flush=True)
            if not _advance_realtime(
                controller,
                viewer,
                args.transition_seconds,
                args.update_hz,
                current,
                target,
            ):
                return
            current = target
            if args.hold_seconds > 0.0 and not _hold_pose(
                controller,
                viewer,
                current,
                args.hold_seconds,
                args.update_hz,
            ):
                return

        if not np.array_equal(current, neutral):
            _advance_realtime(
                controller,
                viewer,
                args.transition_seconds,
                args.update_hz,
                current,
                neutral,
            )
        print("RPS posture preview completed.", flush=True)


if __name__ == "__main__":
    main()
