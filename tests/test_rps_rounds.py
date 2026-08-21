import csv
import tempfile
import unittest
from pathlib import Path

from rps_rounds import CsvRoundRecorder, RpsRoundSession, human_result


class RpsRoundTests(unittest.TestCase):
    def test_all_rule_outcomes(self):
        self.assertEqual(human_result("rock", "scissors"), "win")
        self.assertEqual(human_result("paper", "rock"), "win")
        self.assertEqual(human_result("scissors", "paper"), "win")
        self.assertEqual(human_result("rock", "paper"), "loss")
        self.assertEqual(human_result("paper", "paper"), "tie")

    def test_session_records_only_once_per_robot_move(self):
        session = RpsRoundSession()
        session.start_round("rock")
        first = session.observe_confirmed_human_move("paper")
        duplicate = session.observe_confirmed_human_move("paper")
        self.assertIsNotNone(first)
        self.assertEqual(first.human_result, "win")
        self.assertIsNone(duplicate)
        self.assertEqual(session.human_wins, 1)
        self.assertEqual(session.round_count, 1)

    def test_csv_recorder_writes_header_and_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "rounds.csv"
            session = RpsRoundSession(CsvRoundRecorder(csv_path))
            session.start_round("scissors")
            session.observe_confirmed_human_move("rock")
            session.start_round("rock")
            session.observe_confirmed_human_move("rock")

            with csv_path.open(newline="", encoding="utf-8") as input_file:
                rows = list(csv.DictReader(input_file))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["human_result"], "win")
            self.assertEqual(rows[1]["human_result"], "tie")

            resumed = RpsRoundSession(CsvRoundRecorder(csv_path))
            self.assertEqual(resumed.round_count, 2)
            self.assertEqual(resumed.human_wins, 1)
            self.assertEqual(resumed.ties, 1)
            resumed.start_round("paper")
            third = resumed.observe_confirmed_human_move("scissors")
            self.assertEqual(third.round_number, 3)


if __name__ == "__main__":
    unittest.main()
