"""Per-person open/closed range calibration for 16 hand joint angles."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from hand_angles import ANGLE_NAMES


FLEX_ANGLE_INDICES = np.asarray(
    [index for index, name in enumerate(ANGLE_NAMES) if name.endswith("_flex")],
    dtype=np.intp,
)

# A fully closed human pose is mapped to these safe, fist-like robot targets.
# Side angles remain signed neutral offsets and are not range-mapped.
DEFAULT_FLEXION_TARGETS_DEGREES = np.asarray(
    [
        0.0, 90.0, 100.0, 80.0,
        0.0, 90.0, 100.0, 80.0,
        0.0, 90.0, 100.0, 80.0,
        0.0, 60.0, 75.0, 70.0,
    ],
    dtype=np.float64,
)

CALIBRATION_POSES = ("neutral", "closed")
FLEXION_GROUPS = (
    np.asarray((1, 2, 3), dtype=np.intp),
    np.asarray((5, 6, 7), dtype=np.intp),
    np.asarray((9, 10, 11), dtype=np.intp),
    np.asarray((13, 14, 15), dtype=np.intp),
)


class NeutralCalibration:
    """Collect, persist, and apply open/closed hand-angle calibration.

    Calibrations are stored per profile and per handedness. During collection,
    the median of all accepted samples is used so that an occasional landmark
    spike does not shift the neutral pose.
    """

    FILE_VERSION = 2

    def __init__(
        self,
        path: str | Path,
        profile: str = "default",
        duration_seconds: float = 1.5,
        min_samples: int = 10,
        clamp_flexion: bool = True,
        flexion_targets_degrees: np.ndarray = DEFAULT_FLEXION_TARGETS_DEGREES,
        minimum_flexion_span_degrees: float = 10.0,
    ) -> None:
        if not profile.strip():
            raise ValueError("Calibration profile must not be empty")
        if not math.isfinite(duration_seconds) or duration_seconds <= 0.0:
            raise ValueError("Calibration duration must be a positive finite number")
        if min_samples <= 0:
            raise ValueError("Calibration minimum sample count must be positive")
        if (
            not math.isfinite(minimum_flexion_span_degrees)
            or minimum_flexion_span_degrees <= 0.0
        ):
            raise ValueError("Minimum flexion span must be positive and finite")

        self.path = Path(path)
        self.profile = profile.strip()
        self.duration_seconds = float(duration_seconds)
        self.min_samples = int(min_samples)
        self.clamp_flexion = bool(clamp_flexion)
        self.flexion_targets_degrees = self._validate_angles(
            flexion_targets_degrees
        ).copy()
        if np.any(self.flexion_targets_degrees[FLEX_ANGLE_INDICES] <= 0.0):
            raise ValueError("Flexion targets must be greater than zero")
        self.minimum_flexion_span_degrees = float(
            minimum_flexion_span_degrees
        )

        self._profiles: dict[str, dict[str, np.ndarray]] = {}
        self._sample_counts: dict[str, dict[str, int]] = {}
        self._closed_profiles: dict[str, dict[str, np.ndarray]] = {}
        self._closed_sample_counts: dict[str, dict[str, int]] = {}
        self._active_hand: str | None = None
        self._active_pose: str | None = None
        self._started_at: float | None = None
        self._samples: list[np.ndarray] = []
        self._load()

    @property
    def is_collecting(self) -> bool:
        return self._active_hand is not None

    @property
    def active_hand(self) -> str | None:
        return self._active_hand

    @property
    def active_pose(self) -> str | None:
        return self._active_pose

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def has_offset(self, hand_label: str) -> bool:
        return hand_label in self._profiles.get(self.profile, {})

    def has_range(self, hand_label: str) -> bool:
        return self.has_offset(hand_label) and (
            hand_label in self._closed_profiles.get(self.profile, {})
        )

    def offset_for(self, hand_label: str) -> np.ndarray | None:
        offset = self._profiles.get(self.profile, {}).get(hand_label)
        return None if offset is None else offset.copy()

    def closed_for(self, hand_label: str) -> np.ndarray | None:
        closed = self._closed_profiles.get(self.profile, {}).get(hand_label)
        return None if closed is None else closed.copy()

    def start(
        self,
        hand_label: str,
        timestamp_seconds: float,
        pose: str = "neutral",
    ) -> None:
        """Start replacing an open-neutral or fully-closed calibration pose."""
        hand_label = self._validate_hand_label(hand_label)
        timestamp_seconds = self._validate_timestamp(timestamp_seconds)
        if pose not in CALIBRATION_POSES:
            raise ValueError(f"Calibration pose must be one of {CALIBRATION_POSES}")
        if pose == "closed" and not self.has_offset(hand_label):
            raise ValueError("Neutral calibration must be completed before closed pose")
        self._active_hand = hand_label
        self._active_pose = pose
        self._started_at = timestamp_seconds
        self._samples = []

    def cancel(self) -> None:
        self._active_hand = None
        self._active_pose = None
        self._started_at = None
        self._samples = []

    def progress(self, timestamp_seconds: float) -> float:
        if not self.is_collecting or self._started_at is None:
            return 0.0
        timestamp_seconds = self._validate_timestamp(timestamp_seconds)
        elapsed = max(0.0, timestamp_seconds - self._started_at)
        return min(1.0, elapsed / self.duration_seconds)

    def add_sample(
        self,
        hand_label: str,
        angles_degrees: np.ndarray,
        timestamp_seconds: float,
    ) -> bool:
        """Add a raw angle vector and return True when calibration completes."""
        if not self.is_collecting or hand_label != self._active_hand:
            return False

        sample = self._validate_angles(angles_degrees)
        timestamp_seconds = self._validate_timestamp(timestamp_seconds)
        if self._started_at is None or timestamp_seconds < self._started_at:
            raise ValueError("Calibration timestamp cannot move backwards")

        self._samples.append(sample.copy())
        elapsed = timestamp_seconds - self._started_at
        if elapsed < self.duration_seconds or len(self._samples) < self.min_samples:
            return False

        pose_angles = np.median(np.stack(self._samples), axis=0)
        if self._active_pose == "neutral":
            profile_offsets = self._profiles.setdefault(self.profile, {})
            profile_counts = self._sample_counts.setdefault(self.profile, {})
            profile_offsets[hand_label] = pose_angles
            profile_counts[hand_label] = len(self._samples)
            # A new open pose invalidates the old open-to-closed range.
            self._closed_profiles.get(self.profile, {}).pop(hand_label, None)
            self._closed_sample_counts.get(self.profile, {}).pop(hand_label, None)
        elif self._active_pose == "closed":
            closed_offsets = self._closed_profiles.setdefault(self.profile, {})
            closed_counts = self._closed_sample_counts.setdefault(self.profile, {})
            closed_offsets[hand_label] = pose_angles
            closed_counts[hand_label] = len(self._samples)
        else:
            raise RuntimeError("Calibration pose state is missing")
        self._save()
        self.cancel()
        return True

    def apply(self, hand_label: str, angles_degrees: np.ndarray) -> np.ndarray:
        """Subtract the saved neutral offset from a raw 16-angle vector."""
        angles = self._validate_angles(angles_degrees)
        offset = self._profiles.get(self.profile, {}).get(hand_label)
        if offset is None:
            return angles.copy()

        calibrated = angles - offset
        closed = self._closed_profiles.get(self.profile, {}).get(hand_label)
        if closed is not None:
            span = closed - offset
            for group_indices in FLEXION_GROUPS:
                usable = span[group_indices] >= self.minimum_flexion_span_degrees
                usable_indices = group_indices[usable]
                if usable_indices.size == 0:
                    continue

                usable_ratios = np.clip(
                    calibrated[usable_indices] / span[usable_indices],
                    0.0,
                    1.0,
                )
                calibrated[usable_indices] = (
                    usable_ratios * self.flexion_targets_degrees[usable_indices]
                )

                # When a PIP/DIP landmark barely moves or becomes occluded,
                # borrow the median bend ratio from the other joints in the
                # same finger rather than leaving that robot joint under-bent.
                fallback_indices = group_indices[~usable]
                if fallback_indices.size:
                    group_ratio = float(np.median(usable_ratios))
                    calibrated[fallback_indices] = (
                        group_ratio
                        * self.flexion_targets_degrees[fallback_indices]
                    )
        if self.clamp_flexion:
            calibrated[FLEX_ANGLE_INDICES] = np.maximum(
                calibrated[FLEX_ANGLE_INDICES],
                0.0,
            )
        return calibrated

    @staticmethod
    def _validate_hand_label(hand_label: str) -> str:
        if not isinstance(hand_label, str) or not hand_label.strip():
            raise ValueError("Hand label must not be empty")
        return hand_label.strip()

    @staticmethod
    def _validate_timestamp(timestamp_seconds: float) -> float:
        timestamp = float(timestamp_seconds)
        if not math.isfinite(timestamp):
            raise ValueError("Timestamp must be finite")
        return timestamp

    @staticmethod
    def _validate_angles(angles_degrees: np.ndarray) -> np.ndarray:
        angles = np.asarray(angles_degrees, dtype=np.float64)
        if angles.shape != (len(ANGLE_NAMES),):
            raise ValueError(
                f"Expected {len(ANGLE_NAMES)} joint angles, received shape "
                f"{angles.shape}"
            )
        if not np.all(np.isfinite(angles)):
            raise ValueError("Joint angles contain NaN or infinity")
        return angles

    def _load(self) -> None:
        if not self.path.is_file():
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Failed to read calibration file {self.path}: {error}"
            ) from error

        file_version = payload.get("version")
        if file_version not in (1, self.FILE_VERSION):
            raise ValueError(f"Unsupported calibration file version: {self.path}")
        if payload.get("angle_names") != list(ANGLE_NAMES):
            raise ValueError(
                "Calibration angle order does not match the current application"
            )

        profiles = payload.get("profiles")
        if not isinstance(profiles, dict):
            raise ValueError("Calibration file is missing the profiles object")

        for profile_name, hands in profiles.items():
            if not isinstance(profile_name, str) or not isinstance(hands, dict):
                raise ValueError("Calibration profile data is invalid")
            loaded_offsets: dict[str, np.ndarray] = {}
            loaded_counts: dict[str, int] = {}
            loaded_closed: dict[str, np.ndarray] = {}
            loaded_closed_counts: dict[str, int] = {}
            for hand_label, entry in hands.items():
                if not isinstance(hand_label, str) or not isinstance(entry, dict):
                    raise ValueError("Calibration hand data is invalid")
                loaded_offsets[hand_label] = self._validate_angles(
                    entry.get("neutral_offsets_degrees")
                ).copy()
                sample_count = entry.get(
                    "neutral_sample_count",
                    entry.get("sample_count", 0),
                )
                if not isinstance(sample_count, int) or sample_count < 0:
                    raise ValueError("Calibration sample count is invalid")
                loaded_counts[hand_label] = sample_count
                closed_values = entry.get("closed_angles_degrees")
                if closed_values is not None:
                    loaded_closed[hand_label] = self._validate_angles(
                        closed_values
                    ).copy()
                    closed_count = entry.get("closed_sample_count", 0)
                    if not isinstance(closed_count, int) or closed_count < 0:
                        raise ValueError("Closed calibration sample count is invalid")
                    loaded_closed_counts[hand_label] = closed_count
            self._profiles[profile_name] = loaded_offsets
            self._sample_counts[profile_name] = loaded_counts
            self._closed_profiles[profile_name] = loaded_closed
            self._closed_sample_counts[profile_name] = loaded_closed_counts

    def _save(self) -> None:
        profiles: dict[str, dict[str, dict[str, object]]] = {}
        for profile_name, hands in self._profiles.items():
            profiles[profile_name] = {}
            for hand_label, offset in hands.items():
                entry: dict[str, object] = {
                    "neutral_offsets_degrees": offset.tolist(),
                    "neutral_sample_count": self._sample_counts.get(
                        profile_name,
                        {},
                    ).get(
                        hand_label,
                        0,
                    ),
                }
                closed = self._closed_profiles.get(profile_name, {}).get(hand_label)
                if closed is not None:
                    entry["closed_angles_degrees"] = closed.tolist()
                    entry["closed_sample_count"] = self._closed_sample_counts.get(
                        profile_name,
                        {},
                    ).get(hand_label, 0)
                profiles[profile_name][hand_label] = entry

        payload = {
            "version": self.FILE_VERSION,
            "angle_names": list(ANGLE_NAMES),
            "profiles": profiles,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(self.path)
