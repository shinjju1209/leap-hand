"""Recognize human rock-paper-scissors gestures from MediaPipe landmarks."""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from hand_angles import joint_bend_angle_deg


FINGER_CHAINS = {
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}

LONG_FINGER_PATTERNS = {
    "rock": ("curled", "curled", "curled", "curled"),
    "paper": ("extended", "extended", "extended", "extended"),
    "scissors": ("extended", "extended", "curled", "curled"),
}

THUMB_CHAIN = (1, 2, 3, 4)


@dataclass(frozen=True)
class GestureClassification:
    """One-frame gesture result and the measurements behind it."""

    label: str | None
    confidence: float
    # Order: thumb, index, middle, ring, pinky.
    finger_states: tuple[str, str, str, str, str]
    bend_degrees: tuple[float, float, float, float, float]
    thumb_span_ratio: float


def _landmark_array(landmarks: Sequence[object]) -> np.ndarray:
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 hand landmarks, received {len(landmarks)}")
    points = np.asarray(
        [[point.x, point.y, point.z] for point in landmarks],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(points)):
        raise ValueError("Hand landmarks contain NaN or infinity")
    return points


def classify_rps_gesture(
    landmarks: Sequence[object],
    *,
    extended_max_degrees: float = 55.0,
    curled_min_degrees: float = 100.0,
    thumb_extended_max_degrees: float = 70.0,
    thumb_extended_min_span: float = 0.75,
    thumb_curled_max_span: float = 0.55,
) -> GestureClassification:
    """Classify a hand using the PIP+DIP bend of its four long fingers.

    Rock, paper, and the index+middle scissors pose use the four long fingers,
    so natural thumb variation does not affect them. The alternate index+thumb
    scissors pose additionally requires a straight thumb held away from the
    palm, which distinguishes it from an ordinary pointing gesture.
    """
    if extended_max_degrees < 0.0:
        raise ValueError("extended_max_degrees must be zero or greater")
    if curled_min_degrees <= extended_max_degrees:
        raise ValueError("curled_min_degrees must exceed extended_max_degrees")
    if thumb_extended_max_degrees < 0.0:
        raise ValueError("thumb_extended_max_degrees must be zero or greater")
    if thumb_curled_max_span < 0.0:
        raise ValueError("thumb_curled_max_span must be zero or greater")
    if thumb_extended_min_span <= thumb_curled_max_span:
        raise ValueError("thumb_extended_min_span must exceed thumb_curled_max_span")

    points = _landmark_array(landmarks)
    bends: list[float] = []
    states: list[str] = []
    state_confidences: list[float] = []

    for mcp_id, pip_id, dip_id, tip_id in FINGER_CHAINS.values():
        proximal = points[pip_id] - points[mcp_id]
        middle = points[dip_id] - points[pip_id]
        distal = points[tip_id] - points[dip_id]
        bend = joint_bend_angle_deg(proximal, middle) + joint_bend_angle_deg(
            middle,
            distal,
        )
        bends.append(bend)

        if bend <= extended_max_degrees:
            states.append("extended")
            state_confidences.append(
                float(np.clip(1.0 - bend / max(extended_max_degrees, 1.0), 0.0, 1.0))
            )
        elif bend >= curled_min_degrees:
            states.append("curled")
            state_confidences.append(
                float(
                    np.clip(
                        (bend - curled_min_degrees)
                        / max(180.0 - curled_min_degrees, 1.0),
                        0.0,
                        1.0,
                    )
                )
            )
        else:
            states.append("ambiguous")
            state_confidences.append(0.0)

    long_state_tuple = tuple(states)

    thumb_cmc_id, thumb_mcp_id, thumb_ip_id, thumb_tip_id = THUMB_CHAIN
    thumb_proximal = points[thumb_mcp_id] - points[thumb_cmc_id]
    thumb_middle = points[thumb_ip_id] - points[thumb_mcp_id]
    thumb_distal = points[thumb_tip_id] - points[thumb_ip_id]
    thumb_bend = joint_bend_angle_deg(
        thumb_proximal,
        thumb_middle,
    ) + joint_bend_angle_deg(thumb_middle, thumb_distal)

    palm_width = float(np.linalg.norm(points[5] - points[17]))
    if palm_width < 1e-8:
        raise ValueError("Cannot classify a hand with zero palm width")
    thumb_span_ratio = float(
        np.linalg.norm(points[thumb_tip_id] - points[9]) / palm_width
    )
    if (
        thumb_bend <= thumb_extended_max_degrees
        and thumb_span_ratio >= thumb_extended_min_span
    ):
        thumb_state = "extended"
        thumb_confidence = min(
            float(
                np.clip(
                    1.0 - thumb_bend / max(thumb_extended_max_degrees, 1.0),
                    0.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    (thumb_span_ratio - thumb_extended_min_span)
                    / max(1.25 - thumb_extended_min_span, 1e-8),
                    0.0,
                    1.0,
                )
            ),
        )
    elif (
        thumb_bend >= curled_min_degrees
        or thumb_span_ratio <= thumb_curled_max_span
    ):
        thumb_state = "curled"
        thumb_confidence = max(
            float(
                np.clip(
                    (thumb_bend - curled_min_degrees)
                    / max(180.0 - curled_min_degrees, 1.0),
                    0.0,
                    1.0,
                )
            ),
            float(
                np.clip(
                    (thumb_curled_max_span - thumb_span_ratio)
                    / max(thumb_curled_max_span, 1e-8),
                    0.0,
                    1.0,
                )
            ),
        )
    else:
        thumb_state = "ambiguous"
        thumb_confidence = 0.0

    label = next(
        (
            gesture
            for gesture, expected_states in LONG_FINGER_PATTERNS.items()
            if long_state_tuple == expected_states
        ),
        None,
    )
    uses_thumb_scissors = (
        long_state_tuple == ("extended", "curled", "curled", "curled")
        and thumb_state == "extended"
    )
    if uses_thumb_scissors:
        label = "scissors"
        confidence = min(*state_confidences, thumb_confidence)
    else:
        confidence = min(state_confidences) if label is not None else 0.0

    return GestureClassification(
        label=label,
        confidence=confidence,
        finger_states=(thumb_state, *long_state_tuple),
        bend_degrees=(thumb_bend, *bends),
        thumb_span_ratio=thumb_span_ratio,
    )


class GestureStabilizer:
    """Emit a gesture only after it agrees for several consecutive frames."""

    def __init__(self, required_frames: int = 4) -> None:
        if not isinstance(required_frames, int) or isinstance(required_frames, bool):
            raise TypeError("required_frames must be an integer")
        if required_frames < 1:
            raise ValueError("required_frames must be at least one")
        self.required_frames = required_frames
        self._history: deque[str | None] = deque(maxlen=required_frames)

    def update(self, label: str | None) -> str | None:
        self._history.append(label)
        if (
            label is not None
            and len(self._history) == self.required_frames
            and all(value == label for value in self._history)
        ):
            return label
        return None

    def reset(self) -> None:
        self._history.clear()


__all__ = [
    "FINGER_CHAINS",
    "LONG_FINGER_PATTERNS",
    "THUMB_CHAIN",
    "GestureClassification",
    "GestureStabilizer",
    "classify_rps_gesture",
]
