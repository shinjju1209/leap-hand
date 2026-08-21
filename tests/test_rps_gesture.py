import unittest

import numpy as np

from rps.gesture import GestureStabilizer, classify_rps_gesture


class Landmark:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def make_hand(states, *, thumb_state="extended"):
    """Build simple 3D landmarks with straight or two-joint 90-degree fingers."""
    points = [[0.0, 0.0, 0.0] for _ in range(21)]
    points[0] = [0.0, -1.0, 0.0]
    for finger_index, (base_id, state) in enumerate(
        zip((5, 9, 13, 17), states)
    ):
        x = 0.6 - finger_index * 0.4
        points[base_id] = [x, 0.0, 0.0]
        points[base_id + 1] = [x, 1.0, 0.0]
        if state == "extended":
            points[base_id + 2] = [x, 2.0, 0.0]
            points[base_id + 3] = [x, 3.0, 0.0]
        elif state == "curled":
            points[base_id + 2] = [x + 1.0, 1.0, 0.0]
            points[base_id + 3] = [x + 1.0, 0.0, 0.0]
        else:
            raise ValueError(state)

    if thumb_state == "extended":
        points[1:5] = [
            [-0.4, -0.4, 0.0],
            [-0.7, -0.1, 0.0],
            [-0.9, 0.1, 0.0],
            [-1.0, 0.3, 0.0],
        ]
    elif thumb_state == "folded":
        points[1:5] = [
            [-0.4, -0.4, 0.0],
            [-0.2, -0.2, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0],
        ]
    else:
        raise ValueError(thumb_state)
    return [Landmark(*point) for point in points]


class RpsGestureTests(unittest.TestCase):
    def test_classifies_three_moves(self):
        cases = {
            "rock": ("curled", "curled", "curled", "curled"),
            "paper": ("extended", "extended", "extended", "extended"),
            "scissors": ("extended", "extended", "curled", "curled"),
        }
        for expected, states in cases.items():
            with self.subTest(expected):
                result = classify_rps_gesture(make_hand(states))
                self.assertEqual(result.label, expected)
                self.assertEqual(result.finger_states, ("extended", *states))
                self.assertAlmostEqual(result.confidence, 1.0)

    def test_index_and_thumb_check_sign_is_alternate_scissors(self):
        long_fingers = ("extended", "curled", "curled", "curled")
        check_sign = classify_rps_gesture(
            make_hand(long_fingers, thumb_state="extended")
        )
        pointing = classify_rps_gesture(
            make_hand(long_fingers, thumb_state="folded")
        )

        self.assertEqual(check_sign.label, "scissors")
        self.assertEqual(check_sign.finger_states[0], "extended")
        self.assertIsNone(pointing.label)
        self.assertEqual(pointing.finger_states[0], "curled")

    def test_classification_is_invariant_to_rotation_scale_and_translation(self):
        landmarks = make_hand(("extended", "extended", "curled", "curled"))
        angle = np.deg2rad(57.0)
        rotation = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0.0],
                [np.sin(angle), np.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        transformed = []
        for landmark in landmarks:
            point = 2.7 * rotation @ np.array([landmark.x, landmark.y, landmark.z])
            point += np.array([4.0, -3.0, 1.5])
            transformed.append(Landmark(*point))

        result = classify_rps_gesture(transformed)
        self.assertEqual(result.label, "scissors")

    def test_other_finger_pattern_is_unknown(self):
        result = classify_rps_gesture(
            make_hand(("extended", "curled", "extended", "curled"))
        )
        self.assertIsNone(result.label)
        self.assertEqual(result.confidence, 0.0)

    def test_invalid_thresholds_and_landmark_count_are_rejected(self):
        with self.assertRaises(ValueError):
            classify_rps_gesture(make_hand(("extended",) * 4), curled_min_degrees=50)
        with self.assertRaises(ValueError):
            classify_rps_gesture(
                make_hand(("extended",) * 4), thumb_extended_max_degrees=-1
            )
        with self.assertRaises(ValueError):
            classify_rps_gesture(
                make_hand(("extended",) * 4),
                thumb_extended_min_span=0.5,
                thumb_curled_max_span=0.6,
            )
        with self.assertRaises(ValueError):
            classify_rps_gesture([Landmark(0, 0)] * 20)

    def test_stabilizer_requires_consecutive_frames(self):
        stabilizer = GestureStabilizer(required_frames=3)
        self.assertIsNone(stabilizer.update("rock"))
        self.assertIsNone(stabilizer.update("rock"))
        self.assertIsNone(stabilizer.update(None))
        self.assertIsNone(stabilizer.update("rock"))
        self.assertIsNone(stabilizer.update("rock"))
        self.assertEqual(stabilizer.update("rock"), "rock")
        stabilizer.reset()
        self.assertIsNone(stabilizer.update("rock"))


if __name__ == "__main__":
    unittest.main()
