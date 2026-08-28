"""Unit tests for the Sim-to-Real RL deployment CLI and configuration parser."""

from __future__ import annotations

import csv
import io
import unittest
from types import SimpleNamespace

import numpy as np

from hardware_calibration import HardwareMotorCalibration
from rl_sim2real_deploy import (
    DEFAULT_CONFIG_PATH,
    apply_named_pose_overrides,
    calibrated_motor_targets,
    create_dummy_policy,
    draw_dashboard,
    load_yaml_config,
    parse_args,
    transition_hardware_to_pose,
    verify_hardware_pose,
    write_joint_diagnostics,
)


class RlSim2RealDeployTests(unittest.TestCase):
    def test_default_config_file_exists_and_loads(self):
        self.assertTrue(DEFAULT_CONFIG_PATH.is_file(), "Default config YAML must exist")
        cfg = load_yaml_config(DEFAULT_CONFIG_PATH)
        self.assertIn("control", cfg)
        self.assertIn("simulation", cfg)
        self.assertIn("default_joint_pose_radians", cfg)
        self.assertEqual(len(cfg["default_joint_pose_radians"]), 16)
        self.assertEqual(len(cfg["simulation"]["initial_cube_position"]), 3)
        quaternion = np.asarray(
            cfg["simulation"]["initial_cube_quaternion_wxyz"], dtype=np.float64
        )
        self.assertEqual(quaternion.shape, (4,))
        self.assertAlmostEqual(float(np.linalg.norm(quaternion)), 1.0, places=5)

    def test_cli_argument_defaults_and_overrides(self):
        args = parse_args(["--mode", "mujoco", "--target-axis", "-1.0", "--headless"])
        self.assertEqual(args.mode, "mujoco")
        self.assertEqual(args.target_axis, -1.0)
        self.assertTrue(args.headless)
        self.assertIsNone(args.current_limit)
        self.assertIsNone(args.transition_seconds)
        self.assertIsNone(args.rl_position_p_gain)
        self.assertIsNone(args.rl_position_d_gain)
        self.assertIsNone(args.diagnostics_csv)

    def test_rl_hardware_gains_are_separate_from_shared_controller_defaults(self):
        cfg = load_yaml_config(DEFAULT_CONFIG_PATH)
        hardware = cfg["hardware"]
        self.assertEqual(hardware["rl_position_p_gain"], 800)
        self.assertEqual(hardware["rl_position_i_gain"], 0)
        self.assertEqual(hardware["rl_position_d_gain"], 200)
        self.assertEqual(hardware["rl_side_gain_scale"], 1.0)

    def test_calibrated_rl_pose_uses_saved_per_motor_zero_points(self):
        cfg = load_yaml_config(DEFAULT_CONFIG_PATH)
        pose = np.asarray(cfg["default_joint_pose_radians"], dtype=np.float64)
        calibration = HardwareMotorCalibration.load(
            DEFAULT_CONFIG_PATH.parent.parent / "calibration/hardware_motors.yaml"
        )
        raw_targets = calibrated_motor_targets(calibration, pose)
        np.testing.assert_allclose(
            raw_targets,
            calibration.open_motor_radians + calibration.signs * pose,
        )
        self.assertNotAlmostEqual(
            raw_targets[1],
            float(np.pi + pose[1]),
            places=2,
        )

    def test_hardware_loading_pose_only_relaxes_named_joint(self):
        cfg = load_yaml_config(DEFAULT_CONFIG_PATH)
        policy_pose = np.asarray(cfg["default_joint_pose_radians"], dtype=np.float64)
        loading_pose = apply_named_pose_overrides(
            policy_pose,
            cfg["hardware"]["loading_pose_overrides_radians"],
        )
        expected = policy_pose.copy()
        expected[5] = 0.145
        np.testing.assert_allclose(loading_pose, expected)

    def test_hardware_loading_pose_rejects_unknown_joint(self):
        with self.assertRaisesRegex(ValueError, "unknown loading-pose joint"):
            apply_named_pose_overrides(np.zeros(16), {"not_a_joint": 0.0})

    def test_hardware_transition_repeats_commands_and_checks_feedback(self):
        target = np.linspace(-20.0, 40.0, 16)

        class FakeController:
            torque_enabled = True

            def __init__(self):
                self.commands = []

            def command_degrees(self, command):
                self.commands.append(np.asarray(command).copy())

            def read_feedback(self):
                return SimpleNamespace(positions_degrees=target.copy())

        controller = FakeController()
        error = transition_hardware_to_pose(
            controller,
            target,
            duration_seconds=1.0,
            control_hz=20.0,
            tolerance_degrees=2.0,
            sleep=lambda _: None,
        )
        self.assertEqual(len(controller.commands), 20)
        np.testing.assert_allclose(controller.commands[-1], target)
        self.assertEqual(error, 0.0)

    def test_hardware_transition_rejects_unreached_grasp(self):
        class FakeController:
            torque_enabled = True

            def command_degrees(self, command):
                pass

            def read_feedback(self):
                return SimpleNamespace(positions_degrees=np.zeros(16))

        with self.assertRaisesRegex(
            RuntimeError,
            "tracking error.*index_mcp_side.*target 20.0 deg.*actual 0.0 deg",
        ):
            transition_hardware_to_pose(
                FakeController(),
                np.full(16, 20.0),
                duration_seconds=0.1,
                control_hz=20.0,
                tolerance_degrees=5.0,
                sleep=lambda _: None,
            )

    def test_held_pose_verification_returns_measured_feedback(self):
        measured = np.linspace(-5.0, 5.0, 16)

        class FakeController:
            def read_feedback(self):
                return SimpleNamespace(positions_degrees=measured.copy())

        error, feedback = verify_hardware_pose(
            FakeController(),
            measured + 1.0,
            tolerance_degrees=2.0,
            context="test pose",
        )
        self.assertAlmostEqual(error, 1.0)
        np.testing.assert_allclose(feedback, measured)

    def test_dummy_policy_returns_16_dof(self):
        policy = create_dummy_policy()
        obs = np.zeros(99, dtype=np.float32)
        action = policy(obs)
        self.assertEqual(action.shape, (16,))
        self.assertTrue(np.all(np.isfinite(action)))

    def test_joint_diagnostics_aligns_named_sim_and_real_values(self):
        output = io.StringIO()
        writer = csv.writer(output)
        target = np.arange(16, dtype=np.float64)
        real = target + 0.5
        sim = target - 0.25
        write_joint_diagnostics(
            writer,
            elapsed_seconds=1.25,
            step_count=7,
            target_degrees=target,
            real_degrees=real,
            mujoco_degrees=sim,
        )
        rows = list(csv.reader(io.StringIO(output.getvalue())))
        self.assertEqual(len(rows), 16)
        self.assertEqual(rows[0][2], "index_mcp_side")
        self.assertEqual(rows[-1][2], "thumb_ip_flex")
        self.assertEqual(float(rows[3][6]), 0.5)
        self.assertEqual(float(rows[3][7]), 0.75)

    def test_dashboard_drawing(self):
        canvas = draw_dashboard(
            mode="mujoco",
            armed=True,
            control_hz=20.0,
            actual_hz=19.9,
            step_count=42,
            target_axis=1.0,
            action_norm=0.85,
        )
        self.assertEqual(canvas.shape, (360, 640, 3))


if __name__ == "__main__":
    unittest.main()
