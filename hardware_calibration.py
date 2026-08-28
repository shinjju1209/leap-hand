"""Per-motor open-pose calibration for the 16-motor LEAP Hand v1."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from hand_angles import ANGLE_NAMES


CALIBRATION_VERSION = 1
NOMINAL_OPEN_MOTOR_RADIANS = float(np.pi)


def _vector16(values: Sequence[float], name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64)
    if vector.shape != (16,):
        raise ValueError(f"{name} must contain exactly 16 values, got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} contains NaN or infinity")
    return vector


@dataclass(frozen=True)
class HardwareMotorCalibration:
    """Map project joint angles to individually aligned motor coordinates.

    Entries always follow ``hand_angles.ANGLE_NAMES`` order. ``open_motor_radians``
    is the raw motor position measured while the physical hand is fully open.
    A positive ``sign`` means positive project flexion moves the motor in its
    positive raw-position direction; ``-1`` reverses that motor.
    """

    motor_ids: tuple[int, ...]
    open_motor_radians: np.ndarray
    signs: np.ndarray

    def __post_init__(self) -> None:
        ids = tuple(int(motor_id) for motor_id in self.motor_ids)
        if len(ids) != 16 or len(set(ids)) != 16:
            raise ValueError("motor_ids must contain 16 unique IDs")
        open_positions = _vector16(self.open_motor_radians, "open_motor_radians")
        signs = _vector16(self.signs, "signs")
        if not np.all(np.isin(signs, (-1.0, 1.0))):
            raise ValueError("signs must contain only -1 or 1")
        object.__setattr__(self, "motor_ids", ids)
        object.__setattr__(self, "open_motor_radians", open_positions.copy())
        object.__setattr__(self, "signs", signs.copy())

    @classmethod
    def nominal(cls, motor_ids: Sequence[int] = tuple(range(16))) -> "HardwareMotorCalibration":
        """Return the factory convention: open hand is pi radians for every motor."""
        return cls(
            tuple(motor_ids),
            np.full(16, NOMINAL_OPEN_MOTOR_RADIANS, dtype=np.float64),
            np.ones(16, dtype=np.float64),
        )

    @classmethod
    def from_open_pose_samples(
        cls,
        motor_ids: Sequence[int],
        samples_motor_radians: Sequence[Sequence[float]],
        *,
        signs: Sequence[float] | None = None,
    ) -> "HardwareMotorCalibration":
        samples = np.asarray(samples_motor_radians, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] < 3 or samples.shape[1] != 16:
            raise ValueError("samples_motor_radians must have shape (at least 3, 16)")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples_motor_radians contains NaN or infinity")
        return cls(
            tuple(motor_ids),
            np.median(samples, axis=0),
            np.ones(16, dtype=np.float64) if signs is None else signs,
        )

    @classmethod
    def from_reference_pose_samples(
        cls,
        motor_ids: Sequence[int],
        samples_motor_radians: Sequence[Sequence[float]],
        reference_sim_radians: Sequence[float],
        *,
        signs: Sequence[float] | None = None,
    ) -> "HardwareMotorCalibration":
        """Anchor measured raw positions to a known 16-joint simulation pose.

        A reference pose does not become all-zero. Instead, the returned zero
        offsets make the median measured raw pose decode to
        ``reference_sim_radians``. Commands after that pose use the same fixed
        offsets, preserving all policy motion relative to the anchor.
        """
        samples = np.asarray(samples_motor_radians, dtype=np.float64)
        if samples.ndim != 2 or samples.shape[0] < 3 or samples.shape[1] != 16:
            raise ValueError("samples_motor_radians must have shape (at least 3, 16)")
        if not np.all(np.isfinite(samples)):
            raise ValueError("samples_motor_radians contains NaN or infinity")
        reference = _vector16(reference_sim_radians, "reference_sim_radians")
        direction = _vector16(
            np.ones(16, dtype=np.float64) if signs is None else signs,
            "signs",
        )
        if not np.all(np.isin(direction, (-1.0, 1.0))):
            raise ValueError("signs must contain only -1 or 1")
        median_raw = np.median(samples, axis=0)
        return cls(
            tuple(motor_ids),
            median_raw - direction * reference,
            direction,
        )

    @classmethod
    def load(cls, path: str | Path) -> "HardwareMotorCalibration":
        source = Path(path)
        try:
            data = yaml.safe_load(source.read_text(encoding="utf-8"))
        except OSError as error:
            raise OSError(f"Could not read hardware calibration: {source}") from error
        if not isinstance(data, dict):
            raise ValueError("Hardware calibration YAML must contain a mapping")
        if data.get("version") != CALIBRATION_VERSION:
            raise ValueError(
                f"Unsupported hardware calibration version: {data.get('version')!r}"
            )
        joints = data.get("joints")
        if not isinstance(joints, dict) or set(joints) != set(ANGLE_NAMES):
            raise ValueError("Hardware calibration must define exactly the 16 known joints")

        motor_ids: list[int] = []
        open_positions: list[float] = []
        signs: list[float] = []
        for joint_name in ANGLE_NAMES:
            entry = joints[joint_name]
            if not isinstance(entry, dict):
                raise ValueError(f"Calibration entry for {joint_name} must be a mapping")
            try:
                motor_ids.append(int(entry["motor_id"]))
                open_positions.append(float(entry["open_motor_radians"]))
                signs.append(float(entry["sign"]))
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Calibration entry for {joint_name} needs motor_id, "
                    "open_motor_radians, and sign"
                ) from error
        return cls(tuple(motor_ids), np.asarray(open_positions), np.asarray(signs))

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": CALIBRATION_VERSION,
            "description": (
                "LEAP Hand v1 per-motor open-pose calibration. "
                "Values are in hand_angles.ANGLE_NAMES order."
            ),
            "joints": {
                joint_name: {
                    "motor_id": motor_id,
                    "open_motor_radians": round(float(open_position), 9),
                    "sign": int(sign),
                }
                for joint_name, motor_id, open_position, sign in zip(
                    ANGLE_NAMES,
                    self.motor_ids,
                    self.open_motor_radians,
                    self.signs,
                )
            },
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            yaml.safe_dump(self.to_dict(), allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return destination

    def sim_to_motor_radians(self, sim_radians: Sequence[float]) -> np.ndarray:
        """Map project radians to raw motor radians using saved offsets/signs."""
        return self.open_motor_radians + self.signs * _vector16(
            sim_radians,
            "sim_radians",
        )

    def motor_to_sim_radians(self, motor_radians: Sequence[float]) -> np.ndarray:
        """Map raw motor radians to project radians using saved offsets/signs."""
        return (_vector16(motor_radians, "motor_radians") - self.open_motor_radians) * self.signs


__all__ = [
    "CALIBRATION_VERSION",
    "HardwareMotorCalibration",
    "NOMINAL_OPEN_MOTOR_RADIANS",
]
