"""Sim-to-Real RL Policy Deployment for LEAP Hand v1.

Executes a pre-trained RL policy (TorchScript, ONNX, or dummy) in MuJoCo
simulation, on real LEAP Hand v1 hardware, or both (Digital Twin mode).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from hand_angles import ANGLE_NAMES
from hardware_calibration import HardwareMotorCalibration
from leap_hand_hardware_controller import (
    LeapHandHardwareController,
    clip_sim_radians,
)
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
        default=None,
        help="Rotation direction (+1.0 for CCW, -1.0 for experimental CW)",
    )
    parser.add_argument(
        "--transition-seconds",
        type=float,
        default=None,
        help="Maximum calibrated pre-grasp transition time (default: from config)",
    )
    parser.add_argument(
        "--current-limit",
        type=int,
        default=None,
        help="Motor current limit in mA (default: from config)",
    )
    parser.add_argument(
        "--rl-position-p-gain",
        type=int,
        default=None,
        help="RL-only hardware position P gain (default: from config)",
    )
    parser.add_argument(
        "--rl-position-i-gain",
        type=int,
        default=None,
        help="RL-only hardware position I gain (default: from config)",
    )
    parser.add_argument(
        "--rl-position-d-gain",
        type=int,
        default=None,
        help="RL-only hardware position D gain (default: from config)",
    )
    parser.add_argument(
        "--rl-side-gain-scale",
        type=float,
        default=None,
        help="RL-only gain scale for MCP side motors (default: from config)",
    )
    parser.add_argument(
        "--diagnostics-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV recording policy target, real feedback, and MuJoCo "
            "feedback for each joint"
        ),
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


def reset_mujoco_episode(
    controller: MujocoHandController,
    runner: RLPolicyRunner,
    joint_pose_radians: Sequence[float],
    cube_position: Sequence[float],
    cube_quaternion_wxyz: Sequence[float],
) -> None:
    """Atomically restore the complete hand, cube, and recurrent policy state."""
    hand_pose = np.asarray(joint_pose_radians, dtype=np.float64)
    cube_pos = np.asarray(cube_position, dtype=np.float64)
    cube_quat = np.asarray(cube_quaternion_wxyz, dtype=np.float64)
    if hand_pose.shape != (16,):
        raise ValueError("joint_pose_radians must contain 16 values")
    if cube_pos.shape != (3,):
        raise ValueError("cube_position must contain 3 values")
    if cube_quat.shape != (4,) or not np.isclose(
        np.linalg.norm(cube_quat), 1.0, atol=1e-3
    ):
        raise ValueError("cube_quaternion_wxyz must be a unit wxyz quaternion")

    controller.reset()
    target = controller.set_target_degrees(np.rad2deg(hand_pose))
    controller.data.qpos[controller.qpos_addresses] = target
    if controller.model.nq >= 23:
        controller.data.qpos[16:19] = cube_pos
        controller.data.qpos[19:23] = cube_quat / np.linalg.norm(cube_quat)
    controller.data.qvel[:] = 0.0
    controller.forward()
    runner.reset()


def calibrated_motor_targets(
    calibration: HardwareMotorCalibration,
    joint_pose_radians: Sequence[float],
) -> np.ndarray:
    """Convert an RL pose into raw motor coordinates using saved zero points."""
    pose = np.asarray(joint_pose_radians, dtype=np.float64)
    if pose.shape != (16,) or not np.all(np.isfinite(pose)):
        raise ValueError("joint_pose_radians must contain 16 finite values")
    clipped = clip_sim_radians(pose)
    if not np.allclose(pose, clipped, atol=1e-9):
        invalid = [
            ANGLE_NAMES[index]
            for index in np.flatnonzero(~np.isclose(pose, clipped, atol=1e-9))
        ]
        raise ValueError(
            f"RL initial pose exceeds hardware safety limits: {invalid}"
        )
    return calibration.sim_to_motor_radians(pose)


def apply_named_pose_overrides(
    base_pose_radians: Sequence[float],
    overrides: dict[str, float] | None,
) -> np.ndarray:
    """Return a joint pose with validated ANGLE_NAMES-based overrides."""
    pose = np.asarray(base_pose_radians, dtype=np.float64).copy()
    if pose.shape != (16,) or not np.all(np.isfinite(pose)):
        raise ValueError("base_pose_radians must contain 16 finite values")
    for joint_name, value in (overrides or {}).items():
        if joint_name not in ANGLE_NAMES:
            raise ValueError(f"unknown loading-pose joint: {joint_name}")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"loading-pose value for {joint_name} is not finite")
        pose[ANGLE_NAMES.index(joint_name)] = value
    clipped = clip_sim_radians(pose)
    if not np.allclose(pose, clipped, atol=1e-9):
        raise ValueError("hardware loading pose exceeds joint safety limits")
    return pose


def transition_hardware_to_pose(
    controller: LeapHandHardwareController,
    target_degrees: Sequence[float],
    *,
    duration_seconds: float,
    control_hz: float,
    tolerance_degrees: float,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Slew to the calibrated RL grasp and verify physical tracking before inference."""
    target = np.asarray(target_degrees, dtype=np.float64)
    if target.shape != (16,) or not np.all(np.isfinite(target)):
        raise ValueError("target_degrees must contain 16 finite values")
    if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
        raise ValueError("duration_seconds must be finite and positive")
    if not np.isfinite(control_hz) or control_hz <= 0.0:
        raise ValueError("control_hz must be finite and positive")
    if not np.isfinite(tolerance_degrees) or tolerance_degrees <= 0.0:
        raise ValueError("tolerance_degrees must be finite and positive")
    if not controller.torque_enabled:
        raise RuntimeError("Torque must be enabled before the grasp transition")

    interval = 1.0 / control_hz
    steps = max(1, int(math.ceil(duration_seconds * control_hz)))
    for _ in range(steps):
        sleep(interval)
        controller.command_degrees(target)

    tracking_error, _ = verify_hardware_pose(
        controller,
        target,
        tolerance_degrees=tolerance_degrees,
        context="calibrated pre-grasp",
    )
    return tracking_error


def verify_hardware_pose(
    controller: LeapHandHardwareController,
    target_degrees: Sequence[float],
    *,
    tolerance_degrees: float,
    context: str = "hardware pose",
) -> tuple[float, np.ndarray]:
    """Read feedback and require the held pose to remain near its target."""
    target = np.asarray(target_degrees, dtype=np.float64)
    if target.shape != (16,) or not np.all(np.isfinite(target)):
        raise ValueError("target_degrees must contain 16 finite values")
    if not np.isfinite(tolerance_degrees) or tolerance_degrees <= 0.0:
        raise ValueError("tolerance_degrees must be finite and positive")
    feedback_degrees = np.asarray(
        controller.read_feedback().positions_degrees,
        dtype=np.float64,
    )
    errors = feedback_degrees - target
    worst_index = int(np.argmax(np.abs(errors)))
    tracking_error = float(abs(errors[worst_index]))
    if tracking_error > tolerance_degrees:
        raise RuntimeError(
            f"{context} tracking error is "
            f"{tracking_error:.1f} deg at {ANGLE_NAMES[worst_index]} "
            f"(target {target[worst_index]:.1f} deg, "
            f"actual {feedback_degrees[worst_index]:.1f} deg, "
            f"limit {tolerance_degrees:.1f} deg)"
        )
    return tracking_error, feedback_degrees.copy()


def write_joint_diagnostics(
    writer: Any,
    *,
    elapsed_seconds: float,
    step_count: int,
    target_degrees: Sequence[float],
    real_degrees: Sequence[float],
    mujoco_degrees: Sequence[float] | None,
) -> None:
    """Write one aligned 16-joint sim/real tracking snapshot to a CSV writer."""
    target = np.asarray(target_degrees, dtype=np.float64)
    real = np.asarray(real_degrees, dtype=np.float64)
    sim = (
        np.full(16, np.nan, dtype=np.float64)
        if mujoco_degrees is None
        else np.asarray(mujoco_degrees, dtype=np.float64)
    )
    if target.shape != (16,) or real.shape != (16,) or sim.shape != (16,):
        raise ValueError("diagnostic joint vectors must contain exactly 16 values")
    for name, target_value, real_value, sim_value in zip(
        ANGLE_NAMES, target, real, sim
    ):
        writer.writerow(
            [
                f"{elapsed_seconds:.6f}",
                step_count,
                name,
                f"{target_value:.6f}",
                f"{real_value:.6f}",
                "" if np.isnan(sim_value) else f"{sim_value:.6f}",
                f"{real_value - target_value:.6f}",
                "" if np.isnan(sim_value) else f"{real_value - sim_value:.6f}",
            ]
        )


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
    hardware_cfg = cfg.get("hardware", {})
    current_limit = int(
        args.current_limit
        if args.current_limit is not None
        else hardware_cfg.get("current_limit_milliamps", 350)
    )
    rl_position_p_gain = int(
        args.rl_position_p_gain
        if args.rl_position_p_gain is not None
        else hardware_cfg.get("rl_position_p_gain", 800)
    )
    rl_position_i_gain = int(
        args.rl_position_i_gain
        if args.rl_position_i_gain is not None
        else hardware_cfg.get("rl_position_i_gain", 0)
    )
    rl_position_d_gain = int(
        args.rl_position_d_gain
        if args.rl_position_d_gain is not None
        else hardware_cfg.get("rl_position_d_gain", 200)
    )
    rl_side_gain_scale = float(
        args.rl_side_gain_scale
        if args.rl_side_gain_scale is not None
        else hardware_cfg.get("rl_side_gain_scale", 1.0)
    )
    max_joint_speed = float(hardware_cfg.get("max_joint_speed", 120.0))
    max_tracking_error = float(hardware_cfg.get("max_tracking_error", 25.0))
    tracking_warning = float(hardware_cfg.get("tracking_warning", 8.0))
    initial_pose_tolerance = float(
        hardware_cfg.get("initial_pose_tolerance", 12.0)
    )
    transition_seconds = float(
        args.transition_seconds
        if args.transition_seconds is not None
        else hardware_cfg.get("transition_seconds", 2.0)
    )
    simulation_cfg = cfg.get("simulation", {})
    cube_position = np.asarray(
        simulation_cfg.get("initial_cube_position", [-0.020763, 0.041853, 0.170797]),
        dtype=np.float64,
    )
    cube_quaternion = np.asarray(
        simulation_cfg.get(
            "initial_cube_quaternion_wxyz",
            [0.997554, 0.003869, 0.014019, -0.068372],
        ),
        dtype=np.float64,
    )
    cube_drop_height = float(simulation_cfg.get("cube_drop_height", 0.08))

    raw_pose = cfg.get("default_joint_pose_radians", DEFAULT_MANIPULATION_POSE_RADIANS)
    default_joint_pose = np.asarray(raw_pose, dtype=np.float64)
    hardware_loading_pose = apply_named_pose_overrides(
        default_joint_pose,
        hardware_cfg.get("loading_pose_overrides_radians"),
    )

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
        mujoco_controller = MujocoHandController(
            model_path=args.scene,
            position_kp=float(simulation_cfg.get("position_kp", 3.0)),
            velocity_kv=float(simulation_cfg.get("velocity_kv", 0.1)),
        )
        reset_mujoco_episode(
            mujoco_controller,
            runner,
            default_joint_pose,
            cube_position,
            cube_quaternion,
        )
        mujoco_controller.sync_viewer()
        if not args.headless:
            mujoco_controller.launch_viewer()

    if args.mode in ("hardware", "both"):
        print(f"[HARDWARE] Connecting to LEAP Hand on {args.port}...")
        if not args.motor_calibration_file.is_file():
            raise FileNotFoundError(
                "Hardware mode requires the saved motor zero calibration: "
                f"{args.motor_calibration_file}"
            )
        motor_calib = HardwareMotorCalibration.load(args.motor_calibration_file)
        initial_motor_targets = calibrated_motor_targets(
            motor_calib, default_joint_pose
        )
        print(
            f"[HARDWARE] Loaded calibration: {args.motor_calibration_file.resolve()}"
        )
        print(
            "[HARDWARE] Calibrated initial raw motor range: "
            f"{np.min(initial_motor_targets):.3f}..{np.max(initial_motor_targets):.3f} rad"
        )
        hardware_controller = LeapHandHardwareController(
            port=args.port,
            current_limit_milliamps=current_limit,
            position_p_gain=rl_position_p_gain,
            position_i_gain=rl_position_i_gain,
            position_d_gain=rl_position_d_gain,
            side_gain_scale=rl_side_gain_scale,
            max_joint_speed_degrees_per_second=max_joint_speed,
            motor_calibration=motor_calib,
        )
        try:
            hardware_controller.connect()
            hardware_controller.configure()
        except Exception:
            hardware_controller.close()
            raise
        print("[HARDWARE] Connected with torque OFF.")
        print(
            "[HARDWARE] RL-only actuator settings: "
            f"P={rl_position_p_gain}, I={rl_position_i_gain}, "
            f"D={rl_position_d_gain}, side_scale={rl_side_gain_scale:.2f}, "
            f"current={current_limit} mA"
        )

    # Control loop state
    default_deg = np.rad2deg(default_joint_pose)
    loading_deg = np.rad2deg(hardware_loading_pose)
    step_count = 0
    target_axis = float(
        args.target_axis
        if args.target_axis is not None
        else control_cfg.get("target_axis", 1.0)
    )
    error_msg: str | None = None
    armed = False

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
    diagnostics_handle = None
    diagnostics_writer = None
    if args.diagnostics_csv is not None:
        diagnostics_path = args.diagnostics_csv.resolve()
        diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        diagnostics_handle = diagnostics_path.open(
            "w", encoding="utf-8", newline=""
        )
        diagnostics_writer = csv.writer(diagnostics_handle)
        diagnostics_writer.writerow(
            [
                "elapsed_seconds",
                "step",
                "joint",
                "target_degrees",
                "real_degrees",
                "mujoco_degrees",
                "real_minus_target_degrees",
                "real_minus_mujoco_degrees",
            ]
        )
        print(f"[DIAGNOSTICS] Recording joint trace: {diagnostics_path}")

    try:
        hardware_initial_deg = None
        if hardware_controller is not None:
            print(
                "[HARDWARE] Moving the empty hand to the cube-loading pose. "
                "Keep the hand clear."
            )
            hardware_controller.enable_torque()
            error = transition_hardware_to_pose(
                hardware_controller,
                loading_deg,
                duration_seconds=transition_seconds,
                control_hz=control_hz,
                tolerance_degrees=initial_pose_tolerance,
            )
            error, hardware_initial_deg = verify_hardware_pose(
                hardware_controller,
                loading_deg,
                tolerance_degrees=initial_pose_tolerance,
                context="held cube-loading pose",
            )
            print(
                "[HARDWARE] Cube-loading pose ready and holding "
                f"(max error {error:.1f} deg). Place the cube, then press A."
            )

        if args.auto_arm:
            print("[DEPLOY] Auto-arm requested; moving to the exact policy grasp.")
            if hardware_controller is not None:
                transition_hardware_to_pose(
                    hardware_controller,
                    default_deg,
                    duration_seconds=transition_seconds,
                    control_hz=control_hz,
                    tolerance_degrees=initial_pose_tolerance,
                )
                _, hardware_initial_deg = verify_hardware_pose(
                    hardware_controller,
                    default_deg,
                    tolerance_degrees=initial_pose_tolerance,
                    context="exact policy grasp",
                )
            if mujoco_controller is not None:
                reset_mujoco_episode(
                    mujoco_controller,
                    runner,
                    default_joint_pose,
                    cube_position,
                    cube_quaternion,
                )
                if hardware_initial_deg is not None:
                    runner.reset_from_joint_feedback(hardware_initial_deg)
            elif hardware_initial_deg is not None:
                runner.reset_from_joint_feedback(hardware_initial_deg)
            else:
                runner.reset()
            armed = True
            start_time = time.monotonic()
            last_loop_time = start_time
            print("[DEPLOY] Policy execution active!")

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
            mujoco_deg = (
                mujoco_controller.get_joint_degrees()
                if mujoco_controller is not None
                else None
            )
            if hardware_controller is not None:
                try:
                    fb = hardware_controller.read_feedback()
                    current_deg = fb.positions_degrees
                    if armed:
                        tracking_errors = np.abs(
                            current_deg - hardware_controller.last_command_degrees
                        )
                        tracking_error = float(np.max(tracking_errors))
                        if diagnostics_writer is not None:
                            write_joint_diagnostics(
                                diagnostics_writer,
                                elapsed_seconds=loop_start - start_time,
                                step_count=step_count,
                                target_degrees=hardware_controller.last_command_degrees,
                                real_degrees=current_deg,
                                mujoco_degrees=mujoco_deg,
                            )
                        if (
                            tracking_error >= tracking_warning
                            and step_count % max(1, int(round(control_hz))) == 0
                        ):
                            worst_index = int(np.argmax(tracking_errors))
                            sim_delta = ""
                            if mujoco_deg is not None:
                                sim_delta = (
                                    ", real-sim "
                                    f"{current_deg[worst_index] - mujoco_deg[worst_index]:+.1f} deg"
                                )
                            print(
                                "\n[TRACKING WARNING] "
                                f"{ANGLE_NAMES[worst_index]}: target-real "
                                f"{hardware_controller.last_command_degrees[worst_index] - current_deg[worst_index]:+.1f} deg"
                                f"{sim_delta}"
                            )
                        if tracking_error > max_tracking_error:
                            raise RuntimeError(
                                "tracking error "
                                f"{tracking_error:.1f} deg exceeds "
                                f"{max_tracking_error:.1f} deg"
                            )
                except Exception as e:
                    error_msg = f"Hardware read error: {e}"
                    armed = False
                    hardware_controller.emergency_stop()
                    current_deg = default_deg.copy()
            elif mujoco_controller is not None:
                current_rad = mujoco_controller.data.qpos[mujoco_controller.qpos_addresses].copy()
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
                        np.array([0.7071, 0.0, 0.0, 0.7071])
                        if target_axis >= 0
                        else np.array([0.7071, 0.0, 0.0, -0.7071])
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
                    mujoco_controller.set_target_degrees(target_deg)
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
                # Keep the exact stable grasp frozen until policy activation.
                if mujoco_controller is not None:
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
                if c_pos[2] < cube_drop_height:
                    print("\n[MUJOCO] Cube dropped; resetting the complete policy episode.")
                    if hardware_controller is not None and armed:
                        armed = False
                        hardware_controller.emergency_stop()
                        error_msg = "Digital twin cube dropped; hardware torque disabled"
                    reset_mujoco_episode(
                        mujoco_controller,
                        runner,
                        default_joint_pose,
                        cube_position,
                        cube_quaternion,
                    )

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
                    reset_mujoco_episode(
                        mujoco_controller,
                        runner,
                        default_joint_pose,
                        cube_position,
                        cube_quaternion,
                    )
                    mujoco_controller.sync_viewer()
                else:
                    runner.reset()
                if hardware_controller is not None and hardware_controller.torque_enabled:
                    try:
                        error = transition_hardware_to_pose(
                            hardware_controller,
                            default_deg,
                            duration_seconds=transition_seconds,
                            control_hz=control_hz,
                            tolerance_degrees=initial_pose_tolerance,
                        )
                        print(
                            "[HARDWARE] Calibrated grasp reset complete "
                            f"(max error {error:.1f} deg)."
                        )
                        runner.reset_from_joint_feedback(
                            hardware_controller.read_feedback().positions_degrees
                        )
                    except Exception as e:
                        error_msg = f"Hardware reset error: {e}"
                        armed = False
                        hardware_controller.emergency_stop()
                step_count = 0

            if key in (ord("a"), ord("A")):  # ARM
                if not armed:
                    print("[DEPLOY] ARMING policy execution...")
                    hardware_initial_deg = None
                    if hardware_controller is not None:
                        try:
                            if not hardware_controller.torque_enabled:
                                hardware_controller.enable_torque()
                            print(
                                "[HARDWARE] Cube placed; moving from the loading "
                                "pose to the exact MuJoCo policy grasp."
                            )
                            transition_hardware_to_pose(
                                hardware_controller,
                                default_deg,
                                duration_seconds=transition_seconds,
                                control_hz=control_hz,
                                tolerance_degrees=initial_pose_tolerance,
                            )
                            error, hardware_initial_deg = verify_hardware_pose(
                                hardware_controller,
                                default_deg,
                                tolerance_degrees=initial_pose_tolerance,
                                context="held initial pose before policy start",
                            )
                            print(
                                "[HARDWARE] Exact policy grasp verified; starting "
                                f"policy (max error {error:.1f} deg)."
                            )
                        except Exception as e:
                            print(f"[HARDWARE ERROR] Failed to prepare grasp: {e}")
                            hardware_controller.emergency_stop()
                            armed = False
                            continue
                    if mujoco_controller is not None:
                        reset_mujoco_episode(
                            mujoco_controller,
                            runner,
                            default_joint_pose,
                            cube_position,
                            cube_quaternion,
                        )
                        if hardware_initial_deg is not None:
                            runner.reset_from_joint_feedback(hardware_initial_deg)
                    elif hardware_initial_deg is not None:
                        runner.reset_from_joint_feedback(hardware_initial_deg)
                    else:
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
        if diagnostics_handle is not None:
            diagnostics_handle.close()
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
