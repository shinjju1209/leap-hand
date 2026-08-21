"""Rock-paper-scissors rules and CSV round recording."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .moves import MOVE_NAMES


BEATS = {
    "rock": "scissors",
    "paper": "rock",
    "scissors": "paper",
}


def human_result(human_move: str, robot_move: str) -> str:
    """Return ``win``, ``loss``, or ``tie`` from the human's perspective."""
    human_move = human_move.lower()
    robot_move = robot_move.lower()
    if human_move not in MOVE_NAMES or robot_move not in MOVE_NAMES:
        raise ValueError(f"Moves must be one of {', '.join(MOVE_NAMES)}")
    if human_move == robot_move:
        return "tie"
    return "win" if BEATS[human_move] == robot_move else "loss"


@dataclass(frozen=True)
class RoundRecord:
    round_number: int
    recorded_at_utc: str
    robot_move: str
    human_move: str
    human_result: str


class CsvRoundRecorder:
    """Append completed rounds to a CSV file with a stable schema."""

    FIELD_NAMES = (
        "round_number",
        "recorded_at_utc",
        "robot_move",
        "human_move",
        "human_result",
    )

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def read_all(self) -> list[RoundRecord]:
        """Load existing rounds so a restarted session can continue its score."""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        with self.path.open(newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            if tuple(reader.fieldnames or ()) != self.FIELD_NAMES:
                raise ValueError(
                    f"Unexpected CSV columns in {self.path}; expected "
                    f"{', '.join(self.FIELD_NAMES)}"
                )
            return [
                RoundRecord(
                    round_number=int(row["round_number"]),
                    recorded_at_utc=row["recorded_at_utc"],
                    robot_move=row["robot_move"],
                    human_move=row["human_move"],
                    human_result=row["human_result"],
                )
                for row in reader
            ]

    def append(self, record: RoundRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=self.FIELD_NAMES)
            if needs_header:
                writer.writeheader()
            writer.writerow(
                {
                    "round_number": record.round_number,
                    "recorded_at_utc": record.recorded_at_utc,
                    "robot_move": record.robot_move,
                    "human_move": record.human_move,
                    "human_result": record.human_result,
                }
            )


class RpsRoundSession:
    """Match confirmed human gestures with moves supplied by robot code."""

    def __init__(self, recorder: CsvRoundRecorder | None = None) -> None:
        self.recorder = recorder
        self.pending_robot_move: str | None = None
        existing_records = recorder.read_all() if recorder is not None else []
        self.last_record: RoundRecord | None = (
            existing_records[-1] if existing_records else None
        )
        self.round_count = max(
            (record.round_number for record in existing_records),
            default=0,
        )
        self.human_wins = sum(
            record.human_result == "win" for record in existing_records
        )
        self.robot_wins = sum(
            record.human_result == "loss" for record in existing_records
        )
        self.ties = sum(record.human_result == "tie" for record in existing_records)

    def start_round(self, robot_move: str) -> None:
        robot_move = robot_move.lower()
        if robot_move not in MOVE_NAMES:
            raise ValueError(f"Robot move must be one of {', '.join(MOVE_NAMES)}")
        self.pending_robot_move = robot_move

    def observe_confirmed_human_move(
        self,
        human_move: str | None,
    ) -> RoundRecord | None:
        """Record at most one confirmed gesture for the pending robot move."""
        if human_move is None or self.pending_robot_move is None:
            return None

        human_move = human_move.lower()
        result = human_result(human_move, self.pending_robot_move)
        self.round_count += 1
        record = RoundRecord(
            round_number=self.round_count,
            recorded_at_utc=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            robot_move=self.pending_robot_move,
            human_move=human_move,
            human_result=result,
        )
        self.pending_robot_move = None
        self.last_record = record

        if result == "win":
            self.human_wins += 1
        elif result == "loss":
            self.robot_wins += 1
        else:
            self.ties += 1

        if self.recorder is not None:
            self.recorder.append(record)
        return record


__all__ = [
    "BEATS",
    "CsvRoundRecorder",
    "RoundRecord",
    "RpsRoundSession",
    "human_result",
]
