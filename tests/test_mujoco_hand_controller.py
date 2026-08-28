import unittest

import numpy as np

from mujoco_hand_controller import INPUT_TO_ACTUATOR, MujocoHandController
from neutral_calibration import (
    DEFAULT_FLEXION_TARGETS_DEGREES,
    FLEX_ANGLE_INDICES,
)


class MujocoHandControllerTests(unittest.TestCase):
    def setUp(self):
        self.controller = MujocoHandController()

    def tearDown(self):
        self.controller.close()

    def test_model_has_expected_16_dof_interface(self):
        self.assertEqual(self.controller.model.nq, 16)
        self.assertEqual(self.controller.model.nu, 16)
        self.assertEqual(len(INPUT_TO_ACTUATOR), 16)
        self.assertEqual(len(set(self.controller.actuator_ids.tolist())), 16)

    def test_rl_gain_override_does_not_change_default_controller(self):
        default_ids = self.controller.actuator_ids
        np.testing.assert_allclose(
            self.controller.model.actuator_gainprm[default_ids, 0], 12.0
        )
        rl_controller = MujocoHandController(position_kp=3.0, velocity_kv=0.1)
        try:
            rl_ids = rl_controller.actuator_ids
            np.testing.assert_allclose(
                rl_controller.model.actuator_gainprm[rl_ids, 0], 3.0
            )
            np.testing.assert_allclose(
                rl_controller.model.actuator_biasprm[rl_ids, 1], -3.0
            )
            np.testing.assert_allclose(
                rl_controller.model.actuator_biasprm[rl_ids, 2], -0.1
            )
        finally:
            rl_controller.close()

    def test_human_order_is_written_to_named_actuators(self):
        angles = np.array(
            [5.0, 10.0, 15.0, 20.0] * 3 + [5.0, 10.0, 15.0, 20.0]
        )
        expected = self.controller.set_target_degrees(angles)
        actual = self.controller.data.ctrl[self.controller.actuator_ids]
        np.testing.assert_allclose(actual, expected)

        # Side and flex are intentionally swapped in the model's actuator IDs.
        self.assertNotEqual(
            self.controller.actuator_ids[0],
            self.controller.actuator_ids[1],
        )
        self.assertEqual(INPUT_TO_ACTUATOR[:2], ("if_rot_act", "if_mcp_act"))

    def test_targets_are_clipped_to_actuator_ranges(self):
        target = self.controller.set_target_degrees(np.full(16, 10_000.0))
        ranges = self.controller.model.actuator_ctrlrange[
            self.controller.actuator_ids
        ]
        np.testing.assert_array_less(target, ranges[:, 1] + 1e-12)
        np.testing.assert_array_less(ranges[:, 0] - 1e-12, target)

    def test_step_for_advances_simulation(self):
        initial_time = self.controller.data.time
        steps = self.controller.step_for(1.0 / 30.0)
        self.assertGreater(steps, 0)
        self.assertGreater(self.controller.data.time, initial_time)

    def test_sign_and_offset_are_applied_and_inverted(self):
        signs = np.ones(16)
        signs[0] = -1.0
        offsets = np.zeros(16)
        offsets[0] = 5.0
        controller = MujocoHandController(signs=signs, offsets_degrees=offsets)
        try:
            controller.set_target_degrees(np.zeros(16))
            controller.data.qpos[controller.qpos_addresses] = controller.target_radians
            measured = controller.get_joint_degrees()
            np.testing.assert_allclose(measured, np.zeros(16), atol=1e-12)
        finally:
            controller.close()

    def test_invalid_command_shape_is_rejected(self):
        with self.assertRaises(ValueError):
            self.controller.set_target_degrees(np.zeros(15))

    def test_collision_safe_target_leaves_open_pose_unchanged(self):
        requested = np.zeros(16)
        requested[1] = 20.0
        applied = self.controller.set_collision_safe_target_degrees(requested)
        np.testing.assert_array_equal(applied, requested)
        self.assertEqual(self.controller.last_collision_scale, 1.0)
        self.assertEqual(self.controller.last_predicted_self_contacts, 0)

    def test_collision_safe_target_backs_full_fist_off(self):
        applied = self.controller.set_collision_safe_target_degrees(
            DEFAULT_FLEXION_TARGETS_DEGREES
        )
        self.assertGreater(self.controller.last_predicted_self_contacts, 0)
        self.assertGreater(self.controller.last_collision_scale, 0.0)
        self.assertLess(self.controller.last_collision_scale, 1.0)
        self.assertEqual(
            self.controller._predicted_self_contact_count(applied),
            0,
        )
        self.controller.step_for(1.0)
        self.assertEqual(self.controller.data.ncon, 0)

    def test_collision_limiter_arguments_are_validated(self):
        with self.assertRaises(ValueError):
            self.controller.set_collision_safe_target_degrees(
                np.zeros(16),
                search_iterations=0,
            )
        with self.assertRaises(ValueError):
            self.controller.set_collision_safe_target_degrees(
                np.zeros(16),
                backoff_ratio=0.0,
            )

    def test_new_collision_does_not_snap_fingers_back_toward_zero(self):
        first = self.controller.set_collision_safe_target_degrees(
            DEFAULT_FLEXION_TARGETS_DEGREES
        )
        changed = DEFAULT_FLEXION_TARGETS_DEGREES.copy()
        changed[0] = 30.0
        second = self.controller.set_collision_safe_target_degrees(changed)

        np.testing.assert_array_less(
            first[FLEX_ANGLE_INDICES] - 1e-9,
            second[FLEX_ANGLE_INDICES],
        )
        self.assertEqual(
            self.controller._predicted_self_contact_count(second),
            0,
        )


if __name__ == "__main__":
    unittest.main()
