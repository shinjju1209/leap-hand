"""Named rock-paper-scissors postures for the 16-DoF LEAP Hand."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from hand_angles import ANGLE_NAMES
from .moves import MOVE_NAMES


def make_posture(changes: Mapping[str, float]) -> np.ndarray:
    """Build a posture in ``ANGLE_NAMES`` order from named joint values."""
    posture = np.zeros(len(ANGLE_NAMES), dtype=np.float64)
    unknown_names = set(changes) - set(ANGLE_NAMES)
    if unknown_names:
        raise ValueError(f"Unknown LEAP Hand angle names: {sorted(unknown_names)}")

    for angle_name, value in changes.items():
        posture[ANGLE_NAMES.index(angle_name)] = float(value)
    return posture


# These are conservative starting values in degrees for the MuJoCo right hand.
# The side joints remain centered so the silhouettes stay easy to recognize.
_CLOSED_INDEX = {
    "index_mcp_flex": 75.0,
    "index_pip_flex": 85.0,
    "index_dip_flex": 65.0,
}
_CLOSED_MIDDLE = {
    "middle_mcp_flex": 75.0,
    "middle_pip_flex": 85.0,
    "middle_dip_flex": 65.0,
}
_CLOSED_RING = {
    "ring_mcp_flex": 75.0,
    "ring_pip_flex": 85.0,
    "ring_dip_flex": 65.0,
}
_CLOSED_THUMB = {
    "thumb_cmc_flex": 55.0,
    "thumb_mcp_flex": 65.0,
    "thumb_ip_flex": 50.0,
}

RPS_POSTURES = {
    "rock": make_posture(
        _CLOSED_INDEX | _CLOSED_MIDDLE | _CLOSED_RING | _CLOSED_THUMB
    ),
    "paper": make_posture({}),
    "scissors": make_posture(_CLOSED_MIDDLE | _CLOSED_RING),
}


def get_posture(move: str) -> np.ndarray:
    """Return a copy of a named posture so callers cannot mutate constants."""
    try:
        return RPS_POSTURES[move.lower()].copy()
    except KeyError as error:
        raise ValueError(
            f"Unknown move {move!r}; expected one of {', '.join(MOVE_NAMES)}"
        ) from error


__all__ = ["MOVE_NAMES", "RPS_POSTURES", "get_posture", "make_posture"]
