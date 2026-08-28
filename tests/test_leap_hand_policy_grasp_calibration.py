"""Tests for torque-off RL starting-grasp virtual alignment."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from hardware_calibration import HardwareMotorCalibration
from leap_hand_policy_grasp_calibration import (
    build_policy_grasp_calibration,
    collect_raw_samples,
    load_policy_reference_pose,
    parse_args,
    validate_args,
)


class PolicyGraspCalibrationTests(unittest.TestCase):
    def test_cli_uses_a_separate_output_file(self):
        args = parse_args(["--port", "/dev/test"])
        validate_args(args)
        self.assertEqual(args.port, "/dev/test")
        self.assertEqual(
            args.output,
            Path("calibration/hardware_motors_policy.yaml"),
        )
        self.assertNotEqual(args.output, args.base_calibration_file)

    def test_loads_policy_reference_pose_from_config(self):
        expected = np.linspace(-0.5, 1.0, 16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.yaml"
            path.write_text(
                yaml.safe_dump(
                    {"default_joint_pose_radians": expected.tolist()}
                ),
                encoding="utf-8",
            )
            actual = load_policy_reference_pose(path)
        np.testing.assert_allclose(actual, expected)

    def test_build_preserves_ids_and_signs_and_anchors_pose(self):
        signs = np.asarray([1.0, -1.0] * 8)
        base = HardwareMotorCalibration(
            tuple(reversed(range(16))),
            np.linspace(2.8, 3.2, 16),
            signs,
        )
        reference = np.linspace(-0.2, 1.2, 16)
        desired_zero = np.linspace(2.5, 3.5, 16)
        raw = desired_zero + signs * reference
        samples = np.tile(raw, (7, 1))

        calibration = build_policy_grasp_calibration(base, samples, reference)

        self.assertEqual(calibration.motor_ids, base.motor_ids)
        np.testing.assert_array_equal(calibration.signs, base.signs)
        np.testing.assert_allclose(calibration.open_motor_radians, desired_zero)
        np.testing.assert_allclose(
            calibration.motor_to_sim_radians(raw),
            reference,
        )

    def test_collection_refuses_torque_on(self):
        controller = SimpleNamespace(torque_enabled=True)
        with self.assertRaisesRegex(RuntimeError, "Torque must be off"):
            collect_raw_samples(
                controller,
                samples=3,
                sample_period_seconds=0.01,
                sleep=lambda _: None,
            )

    def test_collection_reads_requested_number_of_raw_samples(self):
        class FakeController:
            torque_enabled = False

            def __init__(self):
                self.read_count = 0

            def read_motor_positions_radians(self):
                self.read_count += 1
                return np.full(16, self.read_count, dtype=np.float64)

        controller = FakeController()
        samples = collect_raw_samples(
            controller,
            samples=3,
            sample_period_seconds=0.01,
            sleep=lambda _: None,
        )
        self.assertEqual(controller.read_count, 3)
        np.testing.assert_allclose(samples[:, 0], [1.0, 2.0, 3.0])


if __name__ == "__main__":
    unittest.main()
