"""Joint ordering and safe deployment limits for the LEAP hand.

The safe limits are the intersection of the policy's MJX training limits and
the canonical hardware URDF limits.  This keeps commands both inside the
training distribution and inside the physical hand's limits.
"""

from __future__ import annotations

from math import pi as PI

import numpy as np


PLAYGROUND_JOINT_NAMES = [
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
]
PLAYGROUND_TO_MOTOR = [
    1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15
]

MJX_LOWER = np.array(
    [-0.314, -1.047, -0.506, -0.366] * 3
    + [-0.349, -0.349, -0.470, -1.340],
    dtype=np.float64,
)
MJX_UPPER = np.array(
    [2.230, 1.047, 1.885, 2.042] * 3
    + [2.094, 2.094, 2.443, 1.880],
    dtype=np.float64,
)
URDF_LOWER_BY_MOTOR = np.array(
    [-1.047, -0.314, -0.506, -0.366] * 3
    + [-0.349, -0.470, -1.200, -1.340],
    dtype=np.float64,
)
URDF_UPPER_BY_MOTOR = np.array(
    [1.047, 2.230, 1.885, 2.042] * 3
    + [2.094, 2.443, 1.900, 1.880],
    dtype=np.float64,
)

SAFE_LOWER = np.maximum(MJX_LOWER, URDF_LOWER_BY_MOTOR[PLAYGROUND_TO_MOTOR])
SAFE_UPPER = np.minimum(MJX_UPPER, URDF_UPPER_BY_MOTOR[PLAYGROUND_TO_MOTOR])


def _vector16(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (16,):
        raise ValueError(f"expected shape (16,), got {vector.shape}")
    return vector


def playground_to_motor(v16: np.ndarray) -> np.ndarray:
    """Reorder a Playground vector into LEAP motor-ID order."""
    value = _vector16(v16)
    result = np.empty(16, dtype=np.float64)
    result[PLAYGROUND_TO_MOTOR] = value
    return result


def motor_to_playground(v16: np.ndarray) -> np.ndarray:
    """Reorder a LEAP motor-ID vector into Playground joint order."""
    return _vector16(v16)[PLAYGROUND_TO_MOTOR].copy()


def sim_to_motor_radians(v16: np.ndarray) -> np.ndarray:
    """Clip safe simulated radians and apply the LEAP motor offset."""
    return np.clip(_vector16(v16), SAFE_LOWER, SAFE_UPPER) + PI


def motor_to_sim_radians(v16: np.ndarray) -> np.ndarray:
    """Remove the LEAP motor offset from radians."""
    return _vector16(v16) - PI
