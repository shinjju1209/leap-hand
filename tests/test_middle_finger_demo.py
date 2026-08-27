import unittest
import numpy as np

from hand_angles import ANGLE_NAMES
from middle_finger_demo import make_middle_finger_posture, parse_args


class MiddleFingerDemoTests(unittest.TestCase):
    def test_posture_vector_shape_and_active_joints(self):
        posture = make_middle_finger_posture()
        self.assertEqual(posture.shape, (16,))
        self.assertTrue(np.all(np.isfinite(posture)))

        # Middle finger joints should remain at 0 (fully extended)
        for joint in ("middle_mcp_side", "middle_mcp_flex", "middle_pip_flex", "middle_dip_flex"):
            self.assertEqual(posture[ANGLE_NAMES.index(joint)], 0.0)

        # Index, Ring, and Thumb flexion joints should be folded (>0)
        for joint in (
            "index_mcp_flex", "index_pip_flex", "index_dip_flex",
            "ring_mcp_flex", "ring_pip_flex", "ring_dip_flex",
            "thumb_cmc_flex", "thumb_mcp_flex", "thumb_ip_flex",
        ):
            self.assertGreater(posture[ANGLE_NAMES.index(joint)], 0.0)

    def test_cli_parsing_defaults(self):
        args = parse_args([])
        self.assertEqual(args.mode, "mujoco")
        self.assertEqual(args.transition_seconds, 0.8)
        self.assertEqual(args.hold_seconds, 2.0)
        self.assertEqual(args.cycles, 1)
        self.assertFalse(args.loop)


if __name__ == "__main__":
    unittest.main()
