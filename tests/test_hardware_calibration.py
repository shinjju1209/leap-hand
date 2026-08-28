import tempfile
import unittest
from pathlib import Path

import numpy as np

from hardware_calibration import HardwareMotorCalibration


class HardwareMotorCalibrationTests(unittest.TestCase):
    def test_nominal_calibration_preserves_the_pi_open_convention(self):
        calibration = HardwareMotorCalibration.nominal()
        sim = np.linspace(-0.4, 0.8, 16)
        np.testing.assert_allclose(
            calibration.sim_to_motor_radians(sim),
            sim + np.pi,
        )
        np.testing.assert_allclose(
            calibration.motor_to_sim_radians(sim + np.pi),
            sim,
            atol=1e-12,
        )

    def test_per_motor_offset_and_sign_round_trip(self):
        open_positions = np.linspace(2.8, 3.5, 16)
        signs = np.ones(16)
        signs[[2, 13]] = -1.0
        calibration = HardwareMotorCalibration(tuple(range(16)), open_positions, signs)
        sim = np.linspace(-0.3, 0.5, 16)
        motor = calibration.sim_to_motor_radians(sim)
        np.testing.assert_allclose(
            motor,
            open_positions + signs * sim,
        )
        np.testing.assert_allclose(calibration.motor_to_sim_radians(motor), sim)

    def test_yaml_round_trip_preserves_joint_mapping(self):
        calibration = HardwareMotorCalibration(
            tuple(reversed(range(16))),
            np.linspace(2.9, 3.4, 16),
            np.asarray([1.0, -1.0] * 8),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hardware.yaml"
            calibration.save(path)
            loaded = HardwareMotorCalibration.load(path)
        self.assertEqual(loaded.motor_ids, calibration.motor_ids)
        np.testing.assert_allclose(loaded.open_motor_radians, calibration.open_motor_radians)
        np.testing.assert_array_equal(loaded.signs, calibration.signs)

    def test_repository_example_is_a_valid_nominal_calibration(self):
        example_path = Path(__file__).resolve().parents[1] / "hardware_calibration.example.yaml"
        calibration = HardwareMotorCalibration.load(example_path)
        np.testing.assert_allclose(
            calibration.open_motor_radians,
            np.full(16, np.pi),
            atol=1e-9,
        )
        np.testing.assert_array_equal(calibration.signs, np.ones(16))

    def test_samples_use_the_median_open_pose(self):
        samples = np.tile(np.linspace(3.0, 3.3, 16), (5, 1))
        samples[0] += 0.5
        samples[-1] -= 0.5
        calibration = HardwareMotorCalibration.from_open_pose_samples(
            tuple(range(16)),
            samples,
        )
        np.testing.assert_allclose(
            calibration.open_motor_radians,
            np.linspace(3.0, 3.3, 16),
        )

    def test_reference_pose_samples_decode_to_the_reference_pose(self):
        reference = np.linspace(-0.4, 1.1, 16)
        virtual_zero = np.linspace(2.6, 3.5, 16)
        signs = np.asarray([1.0, -1.0] * 8)
        raw_reference = virtual_zero + signs * reference
        samples = np.tile(raw_reference, (5, 1))
        samples[0] += 0.2
        samples[-1] -= 0.2

        calibration = HardwareMotorCalibration.from_reference_pose_samples(
            tuple(range(16)),
            samples,
            reference,
            signs=signs,
        )

        np.testing.assert_allclose(calibration.open_motor_radians, virtual_zero)
        np.testing.assert_allclose(
            calibration.motor_to_sim_radians(raw_reference),
            reference,
        )
        np.testing.assert_allclose(
            calibration.sim_to_motor_radians(reference),
            raw_reference,
        )


if __name__ == "__main__":
    unittest.main()
