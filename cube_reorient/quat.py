"""Quaternion operations using MuJoCo's ``wxyz`` convention."""

from __future__ import annotations

import numpy as np


def normalize_with_norm(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Return a unit vector and its norm, matching MJX near zero."""
    value = np.asarray(x, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm < 1e-12:
        return value.copy(), 0.0
    return value / norm, norm


def quat_mul(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Multiply two ``wxyz`` quaternions."""
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return np.array(
        [
            u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
            u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
            u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
            u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
        ],
        dtype=np.float64,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Return the conjugate of a unit ``wxyz`` quaternion."""
    return np.asarray(q, dtype=np.float64) * np.array(
        [1.0, -1.0, -1.0, -1.0], dtype=np.float64
    )


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Convert a unit ``wxyz`` quaternion to a 3-by-3 rotation matrix."""
    q = np.outer(
        np.asarray(q, dtype=np.float64), np.asarray(q, dtype=np.float64)
    )
    return np.array(
        [
            [q[0, 0] + q[1, 1] - q[2, 2] - q[3, 3], 2 * (q[1, 2] - q[0, 3]), 2 * (q[1, 3] + q[0, 2])],
            [2 * (q[1, 2] + q[0, 3]), q[0, 0] - q[1, 1] + q[2, 2] - q[3, 3], 2 * (q[2, 3] - q[0, 1])],
            [2 * (q[1, 3] - q[0, 2]), 2 * (q[2, 3] + q[0, 1]), q[0, 0] - q[1, 1] - q[2, 2] + q[3, 3]],
        ],
        dtype=np.float64,
    )


def quat_integrate(q: np.ndarray, v: np.ndarray, dt: float) -> np.ndarray:
    """Integrate angular velocity exactly as MuJoCo MJX does."""
    unit_v, norm = normalize_with_norm(v)
    angle = float(dt) * norm
    q_res = np.concatenate(
        (
            np.array([np.cos(angle / 2.0)], dtype=np.float64),
            unit_v * np.sin(angle / 2.0),
        )
    )
    result, _ = normalize_with_norm(quat_mul(q, q_res))
    return result


def _quat_angle_rad(a: np.ndarray, b: np.ndarray) -> float:
    relative = quat_mul(a, quat_inv(b))
    vector_norm = float(np.linalg.norm(relative[1:]))
    return float(2.0 * np.arctan2(vector_norm, abs(float(relative[0]))))


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Return the shortest rotation angle between two quaternions in degrees."""
    return float(np.rad2deg(_quat_angle_rad(a, b)))

