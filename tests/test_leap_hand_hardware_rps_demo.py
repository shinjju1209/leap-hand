import unittest
from pathlib import Path
from leap_hand_hardware_rps_demo import parse_args


class LeapHandHardwareRpsDemoTests(unittest.TestCase):
    def test_cli_parsing_defaults(self):
        args = parse_args([])
        self.assertEqual(args.moves, ["rock", "paper", "scissors"])
        self.assertEqual(args.mode, "hardware")
        self.assertEqual(args.port, "/dev/ttyUSB0")
        self.assertEqual(args.current_limit, 300)
        self.assertEqual(args.cycles, 1)
        self.assertFalse(args.loop)
        self.assertEqual(args.transition_seconds, 0.8)
        self.assertEqual(args.hold_seconds, 1.5)

    def test_cli_parsing_custom_moves(self):
        args = parse_args(["scissors", "rock"])
        self.assertEqual(args.moves, ["scissors", "rock"])


if __name__ == "__main__":
    unittest.main()
