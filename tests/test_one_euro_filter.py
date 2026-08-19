import unittest

import numpy as np

from one_euro_filter import OneEuroFilter


class OneEuroFilterTests(unittest.TestCase):
    def test_first_sample_passes_through(self):
        one_euro = OneEuroFilter()
        sample = np.array([1.0, 2.0, 3.0])
        np.testing.assert_array_equal(one_euro.filter(sample, 10.0), sample)

    def test_constant_signal_remains_constant(self):
        one_euro = OneEuroFilter()
        expected = np.full(16, 25.0)
        for frame in range(30):
            actual = one_euro.filter(expected, frame / 30.0)
        np.testing.assert_allclose(actual, expected)

    def test_beta_increases_fast_motion_response(self):
        fixed = OneEuroFilter(min_cutoff=1.0, beta=0.0)
        adaptive = OneEuroFilter(min_cutoff=1.0, beta=0.1)
        zero = np.array([0.0])
        step = np.array([90.0])

        fixed.filter(zero, 0.0)
        adaptive.filter(zero, 0.0)
        fixed_step = fixed.filter(step, 1.0 / 30.0)
        adaptive_step = adaptive.filter(step, 1.0 / 30.0)

        self.assertGreater(adaptive_step[0], fixed_step[0])
        self.assertLess(adaptive_step[0], step[0])

    def test_reset_forgets_previous_state(self):
        one_euro = OneEuroFilter()
        one_euro.filter(np.array([0.0]), 0.0)
        one_euro.filter(np.array([90.0]), 1.0 / 30.0)
        one_euro.reset()
        np.testing.assert_array_equal(
            one_euro.filter(np.array([20.0]), 1.0),
            np.array([20.0]),
        )

    def test_timestamp_must_increase(self):
        one_euro = OneEuroFilter()
        one_euro.filter(np.array([0.0]), 1.0)
        with self.assertRaises(ValueError):
            one_euro.filter(np.array([1.0]), 1.0)


if __name__ == "__main__":
    unittest.main()

