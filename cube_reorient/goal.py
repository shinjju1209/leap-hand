"""Goal-orientation dynamics matching Playground's reorientation task."""

from __future__ import annotations

import numpy as np

from .quat import normalize_with_norm, quat_integrate, quat_inv, quat_mul


def ori_error(cube_quat: np.ndarray, goal_quat: np.ndarray) -> float:
    """Return the shortest cube-to-goal orientation error in radians."""
    quat_diff = quat_mul(cube_quat, quat_inv(goal_quat))
    quat_diff, norm = normalize_with_norm(quat_diff)
    if norm == 0.0:
        raise ValueError("cube_quat and goal_quat must define valid rotations")
    vector_norm = np.clip(np.linalg.norm(quat_diff[1:]), a_min=None, a_max=1.0)
    return float(2.0 * np.arcsin(vector_norm))


class GoalScheduler:
    """Advance goals after evaluating the latest post-physics cube pose.

    Each call to :meth:`step` first evaluates success using the cube pose after
    the action and physics step, then updates angular velocity, and finally
    integrates the next goal.  This ordering matches ``reorient.py``.
    """

    def __init__(
        self,
        dt: float = 0.05,
        success_threshold: float = 0.1,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.dt = float(dt)
        self.success_threshold = float(success_threshold)
        self.rng = np.random.default_rng() if rng is None else rng
        self.goal_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.dquat = np.zeros(3, dtype=np.float64)
        self._success = False
        self.reset()

    def _random_quat(self) -> np.ndarray:
        u, v, w = self.rng.uniform(size=3)
        return np.array(
            [
                np.sqrt(1.0 - u) * np.sin(2.0 * np.pi * v),
                np.sqrt(1.0 - u) * np.cos(2.0 * np.pi * v),
                np.sqrt(u) * np.sin(2.0 * np.pi * w),
                np.sqrt(u) * np.cos(2.0 * np.pi * w),
            ],
            dtype=np.float64,
        )

    def reset(
        self,
        goal_quat: np.ndarray | None = None,
        dquat: np.ndarray | None = None,
    ) -> np.ndarray:
        """Reset and return the goal, sampling omitted state values."""
        if goal_quat is None:
            self.goal_quat = self._random_quat()
        else:
            normalized, norm = normalize_with_norm(goal_quat)
            if norm == 0.0:
                raise ValueError("goal_quat must have nonzero norm")
            self.goal_quat = normalized
        self.dquat = (
            np.zeros(3, dtype=np.float64)
            if dquat is None
            else np.asarray(dquat, dtype=np.float64).copy()
        )
        if self.dquat.shape != (3,):
            raise ValueError(f"expected dquat shape (3,), got {self.dquat.shape}")
        self._success = False
        return self.goal_quat.copy()

    def step(self, cube_quat: np.ndarray) -> np.ndarray:
        """Evaluate success and integrate the next goal orientation."""
        self._success = ori_error(cube_quat, self.goal_quat) < self.success_threshold
        if self._success:
            self.dquat = 3.0 + self.rng.uniform(-2.0, 2.0, size=3)
        else:
            self.dquat = self.dquat * 0.8
        self.goal_quat = quat_integrate(self.goal_quat, self.dquat, 2.0 * self.dt)
        return self.goal_quat.copy()

    @property
    def success(self) -> bool:
        return self._success
