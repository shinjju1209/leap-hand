import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hand_angles import ANGLE_NAMES
from neutral_calibration import (
    DEFAULT_FLEXION_TARGETS_DEGREES,
    FLEX_ANGLE_INDICES,
    NeutralCalibration,
)


class NeutralCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name) / "neutral.json"

    def test_calibration_uses_median_and_saves_profile(self):
        calibration = NeutralCalibration(
            self.path,
            profile="jiwoo",
            duration_seconds=1.0,
            min_samples=3,
        )
        baseline = np.arange(16, dtype=np.float64)
        calibration.start("Right", 10.0)

        self.assertFalse(calibration.add_sample("Right", baseline - 1.0, 10.0))
        self.assertFalse(calibration.add_sample("Right", baseline, 10.5))
        outlier = baseline.copy()
        outlier[5] = 999.0
        self.assertTrue(calibration.add_sample("Right", outlier, 11.0))

        expected = baseline.copy()
        expected[:5] -= 0.0
        expected[5] = baseline[5]
        expected[6:] -= 0.0
        np.testing.assert_array_equal(calibration.offset_for("Right"), expected)
        self.assertTrue(self.path.is_file())
        self.assertFalse(calibration.is_collecting)

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(payload["version"], 2)
        self.assertEqual(
            payload["profiles"]["jiwoo"]["Right"]["neutral_sample_count"],
            3,
        )

    def test_saved_profiles_and_hands_are_loaded_independently(self):
        right = np.full(16, 2.0)
        left = np.full(16, 7.0)

        first = NeutralCalibration(
            self.path,
            profile="person-a",
            duration_seconds=0.1,
            min_samples=1,
        )
        first.start("Right", 0.0)
        first.add_sample("Right", right, 0.1)

        second = NeutralCalibration(
            self.path,
            profile="person-b",
            duration_seconds=0.1,
            min_samples=1,
        )
        second.start("Left", 1.0)
        second.add_sample("Left", left, 1.1)

        loaded_a = NeutralCalibration(self.path, profile="person-a")
        loaded_b = NeutralCalibration(self.path, profile="person-b")
        np.testing.assert_array_equal(loaded_a.offset_for("Right"), right)
        self.assertIsNone(loaded_a.offset_for("Left"))
        np.testing.assert_array_equal(loaded_b.offset_for("Left"), left)
        self.assertIsNone(loaded_b.offset_for("Right"))

    def test_apply_subtracts_offset_and_clamps_only_flexion(self):
        offset = np.full(16, 10.0)
        calibration = NeutralCalibration(
            self.path,
            duration_seconds=0.1,
            min_samples=1,
        )
        calibration.start("Right", 0.0)
        calibration.add_sample("Right", offset, 0.1)

        sample = np.full(16, 8.0)
        actual = calibration.apply("Right", sample)
        flex_indices = set(FLEX_ANGLE_INDICES.tolist())
        for index, name in enumerate(ANGLE_NAMES):
            with self.subTest(angle=name):
                expected = 0.0 if index in flex_indices else -2.0
                self.assertEqual(actual[index], expected)

    def test_apply_without_saved_offset_returns_input_copy(self):
        calibration = NeutralCalibration(self.path)
        sample = np.arange(16, dtype=np.float64)
        actual = calibration.apply("Right", sample)
        np.testing.assert_array_equal(actual, sample)
        self.assertIsNot(actual, sample)

    def test_closed_pose_maps_each_flexion_joint_to_robot_range(self):
        calibration = NeutralCalibration(
            self.path,
            duration_seconds=0.1,
            min_samples=1,
        )
        neutral = np.full(16, 10.0)
        closed = np.full(16, 50.0)
        calibration.start("Right", 0.0, pose="neutral")
        calibration.add_sample("Right", neutral, 0.1)
        calibration.start("Right", 1.0, pose="closed")
        calibration.add_sample("Right", closed, 1.1)

        halfway = np.full(16, 30.0)
        actual = calibration.apply("Right", halfway)
        np.testing.assert_allclose(
            actual[FLEX_ANGLE_INDICES],
            DEFAULT_FLEXION_TARGETS_DEGREES[FLEX_ANGLE_INDICES] * 0.5,
        )
        side_indices = np.setdiff1d(np.arange(16), FLEX_ANGLE_INDICES)
        np.testing.assert_allclose(actual[side_indices], 20.0)

    def test_bad_dip_span_borrows_ratio_from_same_finger(self):
        calibration = NeutralCalibration(
            self.path,
            duration_seconds=0.1,
            min_samples=1,
        )
        neutral = np.zeros(16)
        closed = np.full(16, 50.0)
        closed[3] = 5.0
        calibration.start("Right", 0.0)
        calibration.add_sample("Right", neutral, 0.1)
        calibration.start("Right", 1.0, pose="closed")
        calibration.add_sample("Right", closed, 1.1)

        halfway = closed * 0.5
        actual = calibration.apply("Right", halfway)
        self.assertEqual(actual[1], 45.0)
        self.assertEqual(actual[2], 50.0)
        self.assertEqual(actual[3], 40.0)

    def test_new_neutral_pose_invalidates_old_closed_range(self):
        calibration = NeutralCalibration(
            self.path,
            duration_seconds=0.1,
            min_samples=1,
        )
        calibration.start("Right", 0.0)
        calibration.add_sample("Right", np.zeros(16), 0.1)
        calibration.start("Right", 1.0, pose="closed")
        calibration.add_sample("Right", np.full(16, 50.0), 1.1)
        self.assertTrue(calibration.has_range("Right"))

        calibration.start("Right", 2.0)
        calibration.add_sample("Right", np.ones(16), 2.1)
        self.assertFalse(calibration.has_range("Right"))

    def test_closed_pose_requires_neutral_pose_first(self):
        calibration = NeutralCalibration(self.path)
        with self.assertRaises(ValueError):
            calibration.start("Right", 0.0, pose="closed")

    def test_version_one_neutral_file_is_migrated_on_next_save(self):
        payload = {
            "version": 1,
            "angle_names": list(ANGLE_NAMES),
            "profiles": {
                "legacy": {
                    "Right": {
                        "neutral_offsets_degrees": np.arange(16).tolist(),
                        "sample_count": 12,
                    }
                }
            },
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")
        calibration = NeutralCalibration(
            self.path,
            profile="legacy",
            duration_seconds=0.1,
            min_samples=1,
        )
        np.testing.assert_array_equal(
            calibration.offset_for("Right"),
            np.arange(16),
        )
        calibration.start("Right", 1.0, pose="closed")
        calibration.add_sample("Right", np.arange(16) + 30.0, 1.1)
        migrated = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(migrated["version"], 2)
        self.assertIn(
            "closed_angles_degrees",
            migrated["profiles"]["legacy"]["Right"],
        )

    def test_wrong_hand_is_ignored_while_collecting(self):
        calibration = NeutralCalibration(
            self.path,
            duration_seconds=0.1,
            min_samples=1,
        )
        calibration.start("Right", 0.0)
        self.assertFalse(
            calibration.add_sample("Left", np.zeros(16), 1.0)
        )
        self.assertTrue(calibration.is_collecting)
        self.assertEqual(calibration.sample_count, 0)

    def test_invalid_angle_vector_is_rejected(self):
        calibration = NeutralCalibration(self.path)
        with self.assertRaises(ValueError):
            calibration.apply("Right", np.zeros(15))
        with self.assertRaises(ValueError):
            calibration.apply("Right", np.full(16, np.nan))


if __name__ == "__main__":
    unittest.main()
