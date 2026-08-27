"""Sim-to-Real RL Policy Deployment for LEAP Hand v1.

Executes a pre-trained RL policy (TorchScript, ONNX, or dummy) in MuJoCo
simulation, on real LEAP Hand v1 hardware, or both (Digital Twin mode).
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from hand_angles import ANGLE_NAMES
from leap_hand_hardware_controller import LeapHandHardwareController
from mujoco_hand_controller import MujocoHandController
from rl_policy_runner import (
    DEFAULT_MANIPULATION_POSE_RADIANS,
    RLPolicyRunner,
    load_policy,
)

DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "inhand_cube_rotation.yaml"
)


def load_yaml_config(config_path: Path | str | None) -> dict[str, Any]:
    """Load configuration dictionary from YAML file if it exists."""
    if config_path is None:
        return {}
    path = Path(config_path).resolve()
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a pre-trained RL manipulation policy to LEAP Hand (Sim / Real).",
    )
    parser.add_argument(
        "--mode",
        choices=["mujoco", "hardware", "both"],
        default="mujoco",
        help="Deployment target: mujoco (simulation), hardware (real robot), or both (digital twin)",
    )
    parser.add_argument(
        "--policy",
        "--checkpoint",
        dest="policy",
        type=Path,
        default=None,
        help="Path to pre-trained policy (.pt, .pth, .onnx, .npz). If not specified, a dummy oscillation policy is used.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to YAML configuration file with default grasp pose and control parameters",
    )
    parser.add_argument(
        "--port",
        default="/dev/ttyUSB0",
        help="Serial port for real LEAP Hand hardware",
    )
    parser.add_argument(
        "--motor-calibration-file",
        type=Path,
        default=Path("calibration/hardware_motors.yaml"),
        help="Path to hardware motor zero-point calibration YAML",
    )
    parser.add_argument(
        "--control-hz",
        type=float,
        default=None,
        help="Policy control loop frequency in Hz (default: from config or 20.0)",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=None,
        help="Action scale multiplier in radians (default: from config or 0.1)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=None,
        help="Action smoothing factor (0 < alpha <= 1.0, default: from config or 0.8)",
    )
    parser.add_argument(
        "--history-length",
        type=int,
        default=None,
        help="Observation history buffer length (default: from config or 3)",
    )
    parser.add_argument(
        "--target-axis",
        type=float,
        default=1.0,
        help="Rotation command parameter (+1.0 for CCW, -1.0 for CW)",
    )
    parser.add_argument(
        "--current-limit",
        type=int,
        default=350,
        help="Motor current limit in mA for hardware mode (100~550 mA)",
    )
    parser.add_argument(
        "--auto-arm",
        action="store_true",
        help="Automatically arm and start policy immediately on launch",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Optional maximum run duration in seconds",
    )
    parser.add_argument(
        "--scene",
        type=Path,
        default=Path("models/mujoco/leap_hand/scene_right_cube.xml"),
        help="Path to MuJoCo scene XML",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without OpenCV GUI window",
    )
    return parser.parse_args(argv)


def create_dummy_policy() -> Any:
    """Create a lightweight oscillation policy for testing without external model weights."""
    t_start = time.monotonic()

    def dummy_policy(obs: np.ndarray) -> np.ndarray:
        t = time.monotonic() - t_start
        # Subtle gentle wave motion across fingers
        action = np.zeros(16, dtype=np.float64)
        for i in range(4):
            # MCP flex, PIP flex oscillate slightly out of phase
            action[i * 4 + 1] = 0.5 * np.sin(2.0 * np.pi * 0.5 * t + i * 0.5)
            action[i * 4 + 2] = 0.3 * np.cos(2.0 * np.pi * 0.5 * t + i * 0.5)
        return action

    return dummy_policy


def draw_dashboard(
    mode: str,
    armed: bool,
    control_hz: float,
    actual_hz: float,
    step_count: int,
    target_axis: float,
    action_norm: float,
    error_msg: str | None = None,
) -> np.ndarray:
    """Render a clean status dashboard image for OpenCV window."""
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    # Header
    cv2.putText(
        canvas,
        "LEAP Hand v1 | Sim-to-Real Policy Deployer",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Status pills
    mode_color = (0, 200, 255) if mode == "mujoco" else (0, 255, 128) if mode == "hardware" else (255, 128, 0)
    cv2.putText(
        canvas,
        f"MODE: {mode.upper()}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        mode_color,
        2,
        cv2.LINE_AA,
    )

    state_str = "ARMED (POLICY ACTIVE)" if armed else "DISARMED (HOLDING / TORQUE OFF)"
    state_color = (0, 255, 0) if armed else (0, 165, 255)
    cv2.putText(
        canvas,
        f"STATE: {state_str}",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        state_color,
        2,
        cv2.LINE_AA,
    )

    # Metrics
    cv2.putText(canvas, f"Target Frequency: {control_hz:.1f} Hz | Actual: {actual_hz:.1f} Hz", (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Step Count: {step_count} | Action Norm: {action_norm:.3f}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Target Command (Rotation): {target_axis:+.1f}", (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if error_msg:
        cv2.putText(canvas, f"ERROR: {error_msg}", (20, 245), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # Controls guide
    cv2.line(canvas, (20, 260), (620, 260), (70, 70, 70), 1)
    controls = [
        "[A] Arm / Start Policy   | [D / Space] Disarm / E-Stop",
        "[R] Reset to Default Pose | [1 / 2] Rotation Axis (+1 / -1)",
        "[Q / ESC] Quit & Disarm",
    ]
    for idx, text in enumerate(controls):
        cv2.putText(canvas, text, (20, 290 + idx * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)

    return canvas


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = load_yaml_config(args.config)

    # Resolve parameters from config with CLI override
    control_cfg = cfg.get("control", {})
    control_hz = args.control_hz or control_cfg.get("control_hz", 20.0)
    action_scale = args.action_scale or control_cfg.get("action_scale", 0.1)
    ema_alpha = args.ema_alpha or control_cfg.get("ema_alpha", 0.8)
    history_length = args.history_length or control_cfg.get("history_length", 3)

    raw_pose = cfg.get("default_joint_pose_radians", DEFAULT_MANIPULATION_POSE_RADIANS)
    default_joint_pose = np.asarray(raw_pose, dtype=np.float64)

    # Load or create policy
    policy_path = args.policy or cfg.get("policy", {}).get("checkpoint")
    if policy_path is not None and Path(policy_path).is_file():
        print(f"[POLICY] Loading pre-trained policy from {policy_path}...")
        policy_backend = load_policy(policy_path)
    else:
        print("[POLICY] No policy file specified or found. Using test oscillation policy.")
        policy_backend = create_dummy_policy()

    runner = RLPolicyRunner(
        policy=policy_backend,
        default_joint_pose_radians=default_joint_pose,
        control_hz=control_hz,
        action_scale=action_scale,
        ema_alpha=ema_alpha,
        history_length=history_length,
    )

    # Setup controllers according to mode
    mujoco_controller: MujocoHandController | None = None
    hardware_controller: LeapHandHardwareController | None = None

    if args.mode in ("mujoco", "both"):
        print(f"[MUJOCO] Initializing MuJoCo simulation environment ({args.scene})...")
        mujoco_controller = MujocoHandController(model_path=args.scene)
        init_deg = np.rad2deg(default_joint_pose)
        init_rad = mujoco_controller.set_target_degrees(init_deg)
        mujoco_controller.data.qpos[mujoco_controller.qpos_addresses] = init_rad
        mujoco_controller.step_for(0.01)
        mujoco_controller.sync_viewer()
        mujoco_controller.launch_viewer()

    if args.mode in ("hardware", "both"):
        print(f"[HARDWARE] Connecting to LEAP Hand on {args.port}...")
        motor_calib = (
            args.motor_calibration_file
            if args.motor_calibration_file.is_file()
            else None
        )
        hardware_controller = LeapHandHardwareController(
            port=args.port,
            current_limit_milliamps=args.current_limit,
            motor_calibration=motor_calib,
        )
        hardware_controller.connect()
        hardware_controller.configure()
        print("[HARDWARE] Connected with torque OFF.")

    # Control loop state
    default_deg = np.rad2deg(default_joint_pose)
    step_count = 0
    target_axis = float(args.target_axis)
    error_msg: str | None = None
    armed = bool(args.auto_arm)
    if armed:
        print("[DEPLOY] Auto-armed: Starting policy execution immediately.")
        if mujoco_controller is not None:
            mujoco_controller.set_target_degrees(default_deg)
            mujoco_controller.step_for(0.2)
        if hardware_controller is not None:
            hardware_controller.enable_torque()
            hardware_controller.command_degrees(default_deg)

    window_name = "LEAP Hand Sim2Real Deployment"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("\n" + "=" * 60)
    print("  LEAP Hand Sim-to-Real Deployment Ready")
    print(f"  Mode: {args.mode.upper()} | Frequency: {control_hz:.1f} Hz")
    print("  Press 'A' in the GUI window to ARM and start policy execution.")
    print("  Press 'D' or Space to DISARM / Emergency Stop.")
    print("  Press 'Q' or ESC to Quit.")
    print("=" * 60 + "\n")

    dt = 1.0 / control_hz
    start_time = time.monotonic()
    last_loop_time = time.monotonic()
    actual_hz = control_hz

    try:
        while True:
            loop_start = time.monotonic()
            if args.duration is not None and (loop_start - start_time) >= args.duration:
                print(f"[DEPLOY] Target duration ({args.duration:.1f}s) reached. Exiting.")
                break

            if mujoco_controller is not None and mujoco_controller._viewer is not None:
                if not mujoco_controller._viewer.is_running():
                    print("[MUJOCO] Viewer closed by user.")
                    break

            elapsed_since_last = loop_start - last_loop_time
            last_loop_time = loop_start
            if elapsed_since_last > 0:
                actual_hz = 0.9 * actual_hz + 0.1 * (1.0 / elapsed_since_last)

            # 1. Read current joint angles
            if args.mode == "hardware" and hardware_controller is not None:
                try:
                    fb = hardware_controller.read_feedback()
                    current_deg = fb.positions_degrees
                except Exception as e:
                    error_msg = f"Hardware read error: {e}"
                    current_deg = default_deg.copy()
            elif mujoco_controller is not None:
                current_rad = mujoco_controller.data.qpos[:16].copy()
                current_deg = np.rad2deg(current_rad)
            else:
                current_deg = default_deg.copy()

            # 2. Run policy step if ARMED
            action_norm = 0.0
            if armed:
                extra_obs = None
                if mujoco_controller is not None and len(mujoco_controller.data.qpos) >= 23:
                    cube_pos = mujoco_controller.data.qpos[16:19]
                    cube_quat = mujoco_controller.data.qpos[19:23]
                    goal_quat = (
                        np.array([0.7071, 0.7071, 0.0, 0.0])
                        if target_axis >= 0
                        else np.array([0.7071, -0.7071, 0.0, 0.0])
                    )
                    extra_obs = np.concatenate([cube_pos, cube_quat, goal_quat])

                target_deg, target_rad, raw_action = runner.step(
                    current_deg,
                    target_command=target_axis,
                    is_degrees=True,
                    extra_obs=extra_obs,
                )
                action_norm = float(np.linalg.norm(raw_action))
                step_count += 1

                # Send command to targets (Direct joint control for RL policy)
                if mujoco_controller is not None:
                    mujoco_controller.data.ctrl[:16] = target_rad
                    mujoco_controller.step_for(dt)
                    mujoco_controller.sync_viewer()

                if hardware_controller is not None and hardware_controller.torque_enabled:
                    try:
                        hardware_controller.command_degrees(target_deg)
                    except Exception as e:
                        error_msg = f"Hardware command error: {e}"
                        armed = False
                        hardware_controller.emergency_stop()
            else:
                # Holding / Disarmed: Step MuJoCo slowly if active
                if mujoco_controller is not None:
                    mujoco_controller.step_for(dt)
                    mujoco_controller.sync_viewer()
                if hardware_controller is not None and hardware_controller.torque_enabled:
                    hardware_controller.heartbeat()

            # 3. Real-time Terminal Feedback & Auto-Respawn for Fallen Cube
            if mujoco_controller is not None and len(mujoco_controller.data.qpos) >= 23:
                c_pos = mujoco_controller.data.qpos[16:19]
                c_quat = mujoco_controller.data.qpos[19:23]
                status_arm = "ARMED (ACTIVE)" if armed else "DISARMED (WAIT)"
                sys.stdout.write(
                    f"\r[CUBE 3D] Pos: [X:{c_pos[0]:+.3f}, Y:{c_pos[1]:+.3f}, Z:{c_pos[2]:+.3f}] | "
                    f"Quat: [W:{c_quat[0]:+.3f}, X:{c_quat[1]:+.3f}, Y:{c_quat[2]:+.3f}, Z:{c_quat[3]:+.3f}] | "
                    f"{status_arm} | ActionNorm: {action_norm:.2f} | Step: {step_count:04d}   "
                )
                sys.stdout.flush()

                # If cube drops out of hand, auto-respawn into palm cradle
                if c_pos[2] < 0.08:
                    mujoco_controller.data.qpos[16:19] = [0.005, 0.030, 0.155]
                    mujoco_controller.data.qpos[19:23] = [1.0, 0.0, 0.0, 0.0]
                    mujoco_controller.data.qvel[16:22] = 0.0

            # 4. GUI Dashboard & Keyboard Handling
            key = 255
            if not args.headless:
                dashboard = draw_dashboard(
                    mode=args.mode,
                    armed=armed,
                    control_hz=control_hz,
                    actual_hz=actual_hz,
                    step_count=step_count,
                    target_axis=target_axis,
                    action_norm=action_norm,
                    error_msg=error_msg,
                )
                cv2.imshow(window_name, dashboard)
                key = cv2.waitKey(1) & 0xFF

            # Key actions
            if key in (ord("q"), ord("Q"), 27):  # ESC or Q
                print("\n[DEPLOY] Exit requested.")
                break

            if key in (ord("r"), ord("R")):  # RESET Hand & Cube
                print("\n[DEPLOY] Resetting hand and cube position...")
                if mujoco_controller is not None:
                    mujoco_controller.data.qpos[:16] = default_joint_pose
                    if len(mujoco_controller.data.qpos) >= 23:
                        mujoco_controller.data.qpos[16:19] = [0.005, 0.030, 0.155]
                        mujoco_controller.data.qpos[19:23] = [1.0, 0.0, 0.0, 0.0]
                        mujoco_controller.data.qvel[:] = 0.0
                    mujoco_controller.step_for(0.1)
                    mujoco_controller.sync_viewer()
                runner.reset()

            if key in (ord("a"), ord("A")):  # ARM
                if not armed:
                    print("[DEPLOY] ARMING policy execution...")
                    if hardware_controller is not None:
                        try:
                            # Move to default pose first smoothly
                            hardware_controller.enable_torque()
                            hardware_controller.command_degrees(default_deg)
                            time.sleep(args.transition_seconds)
                        except Exception as e:
                            print(f"[HARDWARE ERROR] Failed to enable torque: {e}")
                            hardware_controller.emergency_stop()
                            armed = False
                            continue
                    if mujoco_controller is not None:
                        mujoco_controller.set_target_degrees(default_deg)
                        mujoco_controller.step_for(0.2)
                    runner.reset()
                    armed = True
                    error_msg = None
                    print("[DEPLOY] Policy execution active!")

            if key in (ord("d"), ord("D"), ord(" ")):  # DISARM
                if armed:
                    print("[DEPLOY] DISARMING policy execution.")
                    armed = False
                    if hardware_controller is not None:
                        hardware_controller.emergency_stop()

            if key in (ord("r"), ord("R")):  # RESET to default grasp pose
                print("[DEPLOY] Resetting hand to default grasp pose...")
                runner.reset()
                if mujoco_controller is not None:
                    mujoco_controller.set_target_degrees(default_deg)
                    mujoco_controller.step_for(0.5)
                if hardware_controller is not None and hardware_controller.torque_enabled:
                    hardware_controller.command_degrees(default_deg)

            if key == ord("1"):
                target_axis = 1.0
                print("[DEPLOY] Target axis set to +1.0 (CCW rotation)")
            elif key == ord("2"):
                target_axis = -1.0
                print("[DEPLOY] Target axis set to -1.0 (CW rotation)")

            # Rate sleep
            elapsed = time.monotonic() - loop_start
            sleep_time = max(0.0, dt - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[DEPLOY] Keyboard interrupt received.")
    finally:
        print("[DEPLOY] Shutting down controllers safely...")
        if hardware_controller is not None:
            hardware_controller.close()
        if mujoco_controller is not None:
            mujoco_controller.close()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[DEPLOY] Clean shutdown complete.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
