"""Unit tests for the Sim-to-Real RL policy runner and observation pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl_policy_runner import (
    DEFAULT_MANIPULATION_POSE_RADIANS,
    ActionProcessor,
    CallablePolicyBackend,
    ObservationManager,
    RLPolicyRunner,
    TorchScriptPolicyBackend,
    load_policy,
)


class ObservationManagerTests(unittest.TestCase):
    def test_dimensions_and_step_obs_construction(self):
        default_pose = np.zeros(16, dtype=np.float64)
        obs_mgr = ObservationManager(
            default_joint_pose_radians=default_pose,
            history_length=3,
            include_command=True,
        )
        self.assertEqual(obs_mgr.step_dim, 33)
        self.assertEqual(obs_mgr.total_dim, 99)

        current_rad = np.ones(16, dtype=np.float64) * 0.5
        last_action = np.ones(16, dtype=np.float64) * -0.2
        step_obs = obs_mgr.build_step_obs(
            current_rad,
            last_action=last_action,
            target_command=1.0,
        )
        self.assertEqual(step_obs.shape, (33,))
        np.testing.assert_allclose(step_obs[:16], 0.5)
        np.testing.assert_allclose(step_obs[16:32], -0.2)
        self.assertEqual(step_obs[32], 1.0)

    def test_history_buffer_stacking_and_reset(self):
        default_pose = np.zeros(16, dtype=np.float64)
        obs_mgr = ObservationManager(
            default_joint_pose_radians=default_pose,
            history_length=3,
            include_command=False,
        )
        self.assertEqual(obs_mgr.step_dim, 32)
        self.assertEqual(obs_mgr.total_dim, 96)

        # First step should prefill all 3 history slots with identical state
        obs1 = obs_mgr.update_and_get_observation(np.ones(16) * 0.1)
        self.assertEqual(obs1.shape, (96,))
        np.testing.assert_allclose(obs1[:32], obs1[32:64])
        np.testing.assert_allclose(obs1[32:64], obs1[64:96])

        # Second step should shift history
        obs2 = obs_mgr.update_and_get_observation(np.ones(16) * 0.2)
        np.testing.assert_allclose(obs2[64:80], 0.2)
        np.testing.assert_allclose(obs2[32:48], 0.1)

        # Reset clears buffer
        obs_mgr.reset()
        obs_fresh = obs_mgr.update_and_get_observation(np.ones(16) * 0.5)
        np.testing.assert_allclose(obs_fresh[:16], 0.5)
        np.testing.assert_allclose(obs_fresh[64:80], 0.5)

    def test_normalization_and_clipping(self):
        default_pose = np.zeros(16)
        mean = np.ones(99) * 0.5
        std = np.ones(99) * 2.0
        obs_mgr = ObservationManager(
            default_joint_pose_radians=default_pose,
            history_length=3,
            obs_mean=mean,
            obs_std=std,
            clip_obs=2.0,
            include_command=True,
        )
        obs = obs_mgr.update_and_get_observation(np.ones(16) * 10.0)
        self.assertTrue(np.all(obs <= 2.0))
        self.assertTrue(np.all(obs >= -2.0))


class ActionProcessorTests(unittest.TestCase):
    def test_scaling_and_clamping(self):
        default_pose = np.zeros(16)
        processor = ActionProcessor(
            default_joint_pose_radians=default_pose,
            action_scale=0.2,
            ema_alpha=1.0,  # No smoothing
            min_joint_bounds_radians=np.ones(16) * -0.5,
            max_joint_bounds_radians=np.ones(16) * 0.5,
        )

        raw_action = np.ones(16) * 1.0  # -> 0.2 rad
        target_rad, target_deg, smoothed = processor.process(raw_action)
        np.testing.assert_allclose(target_rad, 0.2)
        np.testing.assert_allclose(target_deg, np.rad2deg(0.2))

        # Test extreme action being clamped
        extreme_action = np.ones(16) * 10.0  # -> 2.0 rad, clamped to 0.5
        target_rad_clamped, _, _ = processor.process(extreme_action)
        np.testing.assert_allclose(target_rad_clamped, 0.5)

    def test_ema_smoothing(self):
        default_pose = np.zeros(16)
        processor = ActionProcessor(
            default_joint_pose_radians=default_pose,
            action_scale=1.0,
            ema_alpha=0.5,
        )
        # Step 1: action = 1.0 -> smoothed = 1.0
        _, _, smoothed1 = processor.process(np.ones(16) * 1.0)
        np.testing.assert_allclose(smoothed1, 1.0)

        # Step 2: action = 0.0 -> smoothed = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        _, _, smoothed2 = processor.process(np.zeros(16))
        np.testing.assert_allclose(smoothed2, 0.5)


class RLPolicyRunnerTests(unittest.TestCase):
    def test_runner_with_callable_policy(self):
        def mock_policy(obs: np.ndarray) -> np.ndarray:
            return np.ones(16) * 0.5

        runner = RLPolicyRunner(
            policy=mock_policy,
            default_joint_pose_radians=DEFAULT_MANIPULATION_POSE_RADIANS,
            control_hz=20.0,
            action_scale=0.1,
            ema_alpha=1.0,
            history_length=2,
        )

        current_deg = np.rad2deg(DEFAULT_MANIPULATION_POSE_RADIANS)
        target_deg, target_rad, raw_action = runner.step(
            current_deg,
            target_command=1.0,
            is_degrees=True,
        )

        expected_rad = DEFAULT_MANIPULATION_POSE_RADIANS + 0.1 * 0.5
        np.testing.assert_allclose(target_rad, expected_rad, rtol=1e-4)
        np.testing.assert_allclose(target_deg, np.rad2deg(expected_rad), rtol=1e-4)
        np.testing.assert_allclose(raw_action, 0.5)

    def test_torchscript_backend_save_and_load(self):
        import torch

        # Create a simple TorchScript model for testing
        class SimplePolicy(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.ones((x.shape[0], 16), dtype=torch.float32) * 0.25

        model = SimplePolicy()
        scripted = torch.jit.script(model)

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "test_policy.pt"
            scripted.save(str(model_path))

            backend = load_policy(model_path)
            self.assertIsInstance(backend, TorchScriptPolicyBackend)

            obs = np.zeros(99, dtype=np.float32)
            action = backend(obs)
            self.assertEqual(action.shape, (16,))
            np.testing.assert_allclose(action, 0.25)

    def test_playground_jax_backend_npz_and_pt(self):
        from rl_policy_runner import PlaygroundJaxPolicyBackend

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. Test .npz export
            npz_path = Path(tmpdir) / "test_jax_policy.npz"
            # 2-layer MLP: 32 -> 32 -> 16
            k0 = np.ones((32, 32), dtype=np.float32) * 0.01
            b0 = np.zeros((32,), dtype=np.float32)
            k1 = np.ones((32, 16), dtype=np.float32) * 0.01
            b1 = np.ones((16,), dtype=np.float32) * 0.5

            np.savez(
                npz_path,
                **{
                    "params/hidden_0/kernel": k0,
                    "params/hidden_0/bias": b0,
                    "params/hidden_1/kernel": k1,
                    "params/hidden_1/bias": b1,
                }
            )

            backend_npz = load_policy(npz_path)
            self.assertIsInstance(backend_npz, PlaygroundJaxPolicyBackend)
            obs = np.ones(32, dtype=np.float32)
            action_npz = backend_npz(obs)
            self.assertEqual(action_npz.shape, (16,))
            self.assertTrue(np.all(np.isfinite(action_npz)))

            # 2. Test .pt format
            import torch
            pt_path = Path(tmpdir) / "test_jax_policy.pt"
            torch.save(
                {
                    "framework": "mujoco_playground",
                    "weights": {
                        "hidden_0/kernel": k0,
                        "hidden_0/bias": b0,
                        "hidden_1/kernel": k1,
                        "hidden_1/bias": b1,
                    },
                },
                pt_path,
            )
            backend_pt = load_policy(pt_path)
            self.assertIsInstance(backend_pt, PlaygroundJaxPolicyBackend)
            action_pt = backend_pt(obs)
            np.testing.assert_allclose(action_pt, action_npz)


if __name__ == "__main__":
    unittest.main()
