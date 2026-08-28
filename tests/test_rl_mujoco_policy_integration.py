"""Deterministic MuJoCo smoke test for the official in-hand policy."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from mujoco_hand_controller import MujocoHandController
from rl_policy_runner import (
    OFFICIAL_DOF_LOWER,
    OFFICIAL_DOF_UPPER,
    RLPolicyRunner,
    load_policy,
)
from rl_sim2real_deploy import DEFAULT_CONFIG_PATH, load_yaml_config, reset_mujoco_episode


ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "models/mujoco/leap_hand/scene_right_cube.xml"
CHECKPOINT = ROOT / "models/LeapHand.pth"


class OfficialPolicyMujocoIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(CHECKPOINT.is_file(), "official checkpoint is not installed")
    def test_cube_stays_grasped_and_rotates_about_z(self):
        cfg = load_yaml_config(DEFAULT_CONFIG_PATH)
        pose = np.asarray(cfg["default_joint_pose_radians"], dtype=np.float64)
        sim_cfg = cfg["simulation"]
        cube_position = np.asarray(sim_cfg["initial_cube_position"], dtype=np.float64)
        cube_quaternion = np.asarray(
            sim_cfg["initial_cube_quaternion_wxyz"], dtype=np.float64
        )

        controller = MujocoHandController(
            SCENE,
            position_kp=float(sim_cfg["position_kp"]),
            velocity_kv=float(sim_cfg["velocity_kv"]),
        )
        try:
            runner = RLPolicyRunner(
                policy=load_policy(CHECKPOINT),
                default_joint_pose_radians=pose,
                control_hz=20.0,
                action_scale=float(cfg["control"]["action_scale"]),
            )
            reset_mujoco_episode(
                controller, runner, pose, cube_position, cube_quaternion
            )

            min_cube_height = float("inf")
            integrated_z_rotation = 0.0
            for _ in range(200):  # ten simulated seconds at 20 Hz
                current_degrees = controller.get_joint_degrees()
                target_degrees, _, _ = runner.step(
                    current_degrees, target_command=1.0, is_degrees=True
                )
                controller.set_target_degrees(target_degrees)
                controller.step_for(0.05)
                min_cube_height = min(
                    min_cube_height, float(controller.data.qpos[18])
                )
                integrated_z_rotation += float(controller.data.qvel[21]) * 0.05

            self.assertGreater(min_cube_height, 0.14)
            self.assertGreater(integrated_z_rotation, 3.00)
            target = runner.policy.sim_target
            saturated = np.isclose(target, OFFICIAL_DOF_LOWER, atol=1e-5) | np.isclose(
                target, OFFICIAL_DOF_UPPER, atol=1e-5
            )
            self.assertLessEqual(int(np.count_nonzero(saturated)), 4)
        finally:
            controller.close()


if __name__ == "__main__":
    unittest.main()
