import unittest

import numpy as np

from deadband_filter import AngleDeadband


class AngleDeadbandTests(unittest.TestCase):
    def test_first_sample_passes_through(self):
        deadband = AngleDeadband(threshold=0.8)
        sample = np.array([10.0, 20.0])
        np.testing.assert_array_equal(deadband.filter(sample), sample)

    def test_small_changes_are_held(self):
        deadband = AngleDeadband(threshold=0.8)
        deadband.filter(np.array([10.0, 20.0]))
        actual = deadband.filter(np.array([10.7, 19.3]))
        np.testing.assert_array_equal(actual, np.array([10.0, 20.0]))

    def test_change_accumulates_against_last_output(self):
        deadband = AngleDeadband(threshold=0.8)
        deadband.filter(np.array([10.0]))
        np.testing.assert_array_equal(
            deadband.filter(np.array([10.4])),
            np.array([10.0]),
        )
        np.testing.assert_array_equal(
            deadband.filter(np.array([10.9])),
            np.array([10.9]),
        )

    def test_joints_update_independently(self):
        deadband = AngleDeadband(threshold=0.8)
        deadband.filter(np.array([10.0, 20.0]))
        actual = deadband.filter(np.array([11.0, 20.5]))
        np.testing.assert_array_equal(actual, np.array([11.0, 20.0]))

    def test_negative_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            AngleDeadband(threshold=-0.1)


if __name__ == "__main__":
    unittest.main()
