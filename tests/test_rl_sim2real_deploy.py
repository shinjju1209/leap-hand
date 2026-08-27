"""Unit tests for the Sim-to-Real RL deployment CLI and configuration parser."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from rl_sim2real_deploy import (
    DEFAULT_CONFIG_PATH,
    create_dummy_policy,
    draw_dashboard,
    load_yaml_config,
    parse_args,
)


class RlSim2RealDeployTests(unittest.TestCase):
    def test_default_config_file_exists_and_loads(self):
        self.assertTrue(DEFAULT_CONFIG_PATH.is_file(), "Default config YAML must exist")
        cfg = load_yaml_config(DEFAULT_CONFIG_PATH)
        self.assertIn("control", cfg)
        self.assertIn("default_joint_pose_radians", cfg)
        self.assertEqual(len(cfg["default_joint_pose_radians"]), 16)

    def test_cli_argument_defaults_and_overrides(self):
        args = parse_args(["--mode", "mujoco", "--target-axis", "-1.0", "--headless"])
        self.assertEqual(args.mode, "mujoco")
        self.assertEqual(args.target_axis, -1.0)
        self.assertTrue(args.headless)

    def test_dummy_policy_returns_16_dof(self):
        policy = create_dummy_policy()
        obs = np.zeros(99, dtype=np.float32)
        action = policy(obs)
        self.assertEqual(action.shape, (16,))
        self.assertTrue(np.all(np.isfinite(action)))

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
