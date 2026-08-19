"""Convert MediaPipe hand landmarks into 16 LEAP-oriented control angles."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


ANGLE_NAMES = (
    "index_mcp_side",
    "index_mcp_flex",
    "index_pip_flex",
    "index_dip_flex",
    "middle_mcp_side",
    "middle_mcp_flex",
    "middle_pip_flex",
    "middle_dip_flex",
    "ring_mcp_side",
    "ring_mcp_flex",
    "ring_pip_flex",
    "ring_dip_flex",
    "thumb_cmc_side",
    "thumb_cmc_flex",
    "thumb_mcp_flex",
    "thumb_ip_flex",
)

DISPLAY_NAMES = (
    "Index MCP side",
    "Index MCP flex",
    "Index PIP flex",
    "Index DIP flex",
    "Middle MCP side",
    "Middle MCP flex",
    "Middle PIP flex",
    "Middle DIP flex",
    "Ring MCP side",
    "Ring MCP flex",
    "Ring PIP flex",
    "Ring DIP flex",
    "Thumb CMC side",
    "Thumb CMC flex",
    "Thumb MCP flex",
    "Thumb IP flex",
)

_EPSILON = 1e-8


def _as_array(landmarks: Sequence[object]) -> np.ndarray:
    """Return 21 MediaPipe landmarks as a float array with shape (21, 3)."""
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 hand landmarks, received {len(landmarks)}")

    points = np.array(
        [[point.x, point.y, point.z] for point in landmarks],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(points)):
        raise ValueError("Hand landmarks contain NaN or infinity")
    return points


def _normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < _EPSILON:
        raise ValueError("Cannot calculate an angle from coincident landmarks")
    return vector / length


def joint_bend_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    """Return 0 degrees for aligned segments and 180 for opposite segments."""
    first_unit = _normalize(first)
    second_unit = _normalize(second)
    cosine = float(np.clip(np.dot(first_unit, second_unit), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _project_to_plane(vector: np.ndarray, normal: np.ndarray) -> np.ndarray:
    return vector - np.dot(vector, normal) * normal


def _signed_angle_deg(
    first: np.ndarray,
    second: np.ndarray,
    normal: np.ndarray,
) -> float:
    first_unit = _normalize(first)
    second_unit = _normalize(second)
    sine = float(np.dot(normal, np.cross(first_unit, second_unit)))
    cosine = float(np.clip(np.dot(first_unit, second_unit), -1.0, 1.0))
    return float(np.degrees(np.arctan2(sine, cosine)))


def _out_of_plane_angle_deg(vector: np.ndarray, normal: np.ndarray) -> float:
    vector_unit = _normalize(vector)
    normal_component = float(np.clip(np.dot(vector_unit, normal), -1.0, 1.0))
    # Magnitude is used for now. Direction offsets are handled during calibration.
    return abs(float(np.degrees(np.arcsin(normal_component))))


def _finger_angles(
    points: np.ndarray,
    wrist_id: int,
    mcp_id: int,
    pip_id: int,
    dip_id: int,
    tip_id: int,
    palm_normal: np.ndarray,
) -> tuple[float, float, float, float]:
    palm_ray = points[mcp_id] - points[wrist_id]
    proximal = points[pip_id] - points[mcp_id]
    middle = points[dip_id] - points[pip_id]
    distal = points[tip_id] - points[dip_id]

    palm_ray_projected = _project_to_plane(palm_ray, palm_normal)
    proximal_projected = _project_to_plane(proximal, palm_normal)

    side = _signed_angle_deg(
        palm_ray_projected,
        proximal_projected,
        palm_normal,
    )
    mcp_flex = _out_of_plane_angle_deg(proximal, palm_normal)
    pip_flex = joint_bend_angle_deg(proximal, middle)
    dip_flex = joint_bend_angle_deg(middle, distal)
    return side, mcp_flex, pip_flex, dip_flex


def calculate_leap_control_angles(landmarks: Sequence[object]) -> np.ndarray:
    """Calculate 16 human-hand angles corresponding to LEAP Hand controls.

    MediaPipe world landmarks are preferred because their 3D scale and orientation
    are less dependent on image size. Returned values are degrees for visual
    inspection; motor retargeting will later convert calibrated values to radians.
    """
    points = _as_array(landmarks)

    wrist = points[0]
    across_palm = points[5] - points[17]  # pinky side -> index side
    along_palm = points[9] - wrist        # wrist -> middle MCP
    palm_normal = _normalize(np.cross(across_palm, along_palm))

    values: list[float] = []
    for chain in ((0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16)):
        values.extend(_finger_angles(points, *chain, palm_normal))

    thumb_proximal = points[2] - points[1]
    thumb_middle = points[3] - points[2]
    thumb_distal = points[4] - points[3]

    thumb_reference = _project_to_plane(across_palm, palm_normal)
    thumb_projected = _project_to_plane(thumb_proximal, palm_normal)
    thumb_cmc_side = _signed_angle_deg(
        thumb_reference,
        thumb_projected,
        palm_normal,
    )
    thumb_cmc_flex = _out_of_plane_angle_deg(thumb_proximal, palm_normal)
    thumb_mcp_flex = joint_bend_angle_deg(thumb_proximal, thumb_middle)
    thumb_ip_flex = joint_bend_angle_deg(thumb_middle, thumb_distal)
    values.extend(
        (thumb_cmc_side, thumb_cmc_flex, thumb_mcp_flex, thumb_ip_flex)
    )

    result = np.asarray(values, dtype=np.float64)
    if result.shape != (16,) or not np.all(np.isfinite(result)):
        raise ValueError("Failed to calculate a finite 16-angle hand vector")
    return result

