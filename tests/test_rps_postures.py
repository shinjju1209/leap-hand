import unittest

import numpy as np

from hand_angles import ANGLE_NAMES
from rps.postures import MOVE_NAMES, RPS_POSTURES, get_posture, make_posture


class RpsPostureTests(unittest.TestCase):
    def test_all_moves_have_finite_16_angle_postures(self):
        self.assertEqual(set(RPS_POSTURES), set(MOVE_NAMES))
        for posture in RPS_POSTURES.values():
            self.assertEqual(posture.shape, (16,))
            self.assertTrue(np.all(np.isfinite(posture)))

    def test_scissors_extends_index_and_thumb_only(self):
        scissors = get_posture("scissors")
        for finger in ("index", "thumb"):
            for joint in ("mcp_flex", "pip_flex", "dip_flex", "cmc_flex", "ip_flex"):
                joint_name = f"{finger}_{joint}"
                if joint_name in ANGLE_NAMES:
                    self.assertEqual(scissors[ANGLE_NAMES.index(joint_name)], 0.0)
        for finger in ("middle", "ring"):
            flex_indices = [
                index
                for index, name in enumerate(ANGLE_NAMES)
                if name.startswith(f"{finger}_") and name.endswith("_flex")
            ]
            self.assertTrue(all(scissors[index] > 0.0 for index in flex_indices))

    def test_rock_closes_every_finger_and_paper_is_open(self):
        rock = get_posture("rock")
        paper = get_posture("paper")
        flex_indices = [
            index for index, name in enumerate(ANGLE_NAMES) if name.endswith("_flex")
        ]
        self.assertTrue(all(rock[index] > 0.0 for index in flex_indices))
        np.testing.assert_array_equal(paper, np.zeros(16))

    def test_returned_posture_is_a_copy(self):
        first = get_posture("rock")
        first[0] = 999.0
        self.assertNotEqual(get_posture("rock")[0], 999.0)

    def test_unknown_names_and_moves_are_rejected(self):
        with self.assertRaises(ValueError):
            make_posture({"not_a_joint": 1.0})
        with self.assertRaises(ValueError):
            get_posture("lizard")


if __name__ == "__main__":
    unittest.main()
