import unittest
from types import SimpleNamespace

import numpy as np

from hand_angles import ANGLE_NAMES
from leap_hand_hardware_controller import LeapHandFeedback
from leap_hand_joint_test import (
    build_relative_target,
    move_to_target,
    validate_args,
)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def sleep(self, seconds):
        self.value += seconds


class FakeSlewController:
    def __init__(self):
        self.applied = np.zeros(16, dtype=np.float64)

    def command_degrees(self, target):
        self.applied += np.clip(np.asarray(target) - self.applied, -1.0, 1.0)
        return self.applied.copy()

    def read_feedback(self):
        return LeapHandFeedback(
            positions_degrees=self.applied.copy(),
            velocities_degrees_per_second=np.zeros(16),
            currents_milliamps=np.zeros(16),
        )


class LeapHandJointTestTests(unittest.TestCase):
    def test_build_target_changes_only_selected_joint(self):
        initial = np.zeros(16)
        target = build_relative_target(initial, "index_pip_flex", 5.0)
        expected = initial.copy()
        expected[ANGLE_NAMES.index("index_pip_flex")] = 5.0
        np.testing.assert_allclose(target, expected)

    def test_build_target_rejects_out_of_range_start(self):
        initial = np.zeros(16)
        initial[ANGLE_NAMES.index("index_pip_flex")] = -40.0
        with self.assertRaises(RuntimeError):
            build_relative_target(initial, "index_mcp_flex", 5.0)

    def test_validation_enforces_bringup_limits(self):
        valid = SimpleNamespace(
            delta=5.0,
            max_joint_speed=30.0,
            current_limit=300,
            hold_seconds=1.0,
            command_rate=50.0,
            max_tracking_error=15.0,
            max_temperature=50.0,
        )
        validate_args(valid)
        valid.delta = 20.0
        with self.assertRaises(ValueError):
            validate_args(valid)

    def test_move_repeats_until_slew_limiter_reaches_target(self):
        controller = FakeSlewController()
        clock = FakeClock()
        target = np.full(16, 3.0)
        feedback = move_to_target(
            controller,
            target,
            command_rate_hz=50.0,
            max_tracking_error=1.0,
            timeout_seconds=1.0,
            clock=clock,
            sleep=clock.sleep,
        )
        np.testing.assert_allclose(feedback.positions_degrees, target)


if __name__ == "__main__":
    unittest.main()
