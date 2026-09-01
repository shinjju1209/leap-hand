"""Stateful assembly of the policy's 57-dimensional observation.

The palm and cube position sensors, as well as the cube quaternion, are all in
world axes in MuJoCo (the ``framepos`` sensors have no ``reftype``).  Therefore
both the position difference and relative-rotation matrix slices are assembled
directly from world-frame values; neither is transformed into palm-local axes.
"""

from __future__ import annotations

import numpy as np

from .mapping import SAFE_LOWER, SAFE_UPPER
from .quat import quat_inv, quat_mul, quat_to_mat


class ObservationBuilder:
    """Own previous command/action state and assemble policy observations."""

    def __init__(
        self,
        action_scale: float = 0.5,
        ema_alpha: float = 1.0,
        lower: np.ndarray | None = None,
        upper: np.ndarray | None = None,
    ) -> None:
        """Build observations and accumulate incremental commands.

        ``lower``/``upper`` bound the commanded targets.  They default to the
        hardware-safe intersection limits, which is what a real deployment
        wants.  Sim-parity checks must pass the MJX training limits instead
        (``mapping.MJX_LOWER``/``MJX_UPPER``): ``reorient.py`` clips against
        the model's ctrl range, and for the thumb that range is wider than the
        physical hand allows, so the two disagree unless stated explicitly.
        """
        self.action_scale = float(action_scale)
        self.ema_alpha = float(ema_alpha)
        self.lower = SAFE_LOWER.copy() if lower is None else np.asarray(lower, dtype=np.float64)
        self.upper = SAFE_UPPER.copy() if upper is None else np.asarray(upper, dtype=np.float64)
        if self.lower.shape != (16,) or self.upper.shape != (16,):
            raise ValueError("lower/upper must both have shape (16,)")
        if np.any(self.lower > self.upper):
            raise ValueError("lower must not exceed upper")
        self.motor_targets = np.zeros(16, dtype=np.float64)
        self.last_act = np.zeros(16, dtype=np.float64)
        # The action the *observation* reports, which lags last_act by one step.
        # See apply_action() for why.
        self.obs_act = np.zeros(16, dtype=np.float64)

    @staticmethod
    def _array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
        result = np.asarray(value, dtype=np.float64)
        if result.shape != shape:
            raise ValueError(f"expected {name} shape {shape}, got {result.shape}")
        return result

    def reset(self, initial_motor_targets: np.ndarray) -> None:
        """Reset command history from the initial commanded joint targets."""
        self.motor_targets = self._array(
            initial_motor_targets, (16,), "initial_motor_targets"
        ).copy()
        self.last_act = np.zeros(16, dtype=np.float64)
        self.obs_act = np.zeros(16, dtype=np.float64)

    def build(
        self,
        joint_angles: np.ndarray,
        palm_pos: np.ndarray,
        cube_pos: np.ndarray,
        cube_quat: np.ndarray,
        goal_quat: np.ndarray,
    ) -> np.ndarray:
        """Build the raw, deliberately unnormalized 57-value observation."""
        joint_angles = self._array(joint_angles, (16,), "joint_angles")
        palm_pos = self._array(palm_pos, (3,), "palm_pos")
        cube_pos = self._array(cube_pos, (3,), "cube_pos")
        cube_quat = self._array(cube_quat, (4,), "cube_quat")
        goal_quat = self._array(goal_quat, (4,), "goal_quat")
        xmat_diff = quat_to_mat(quat_mul(cube_quat, quat_inv(goal_quat)))
        return np.concatenate(
            (
                joint_angles,
                joint_angles - self.motor_targets,
                palm_pos - cube_pos,
                xmat_diff.ravel()[3:],
                self.obs_act,
            ),
            dtype=np.float64,
        )

    def apply_action(self, action: np.ndarray) -> np.ndarray:
        """Accumulate an incremental action from the last commanded targets.

        The action reported back in the observation lags one step behind the
        action just applied.  This reproduces ``reorient.py``, which calls
        ``_get_obs`` (line 232) *before* it records ``info["last_act"] = action``
        (line 259) -- so the environment's observation always carries the action
        from one step further back.  The policy was trained against that lag, so
        removing it would put the observation outside the training distribution.
        Verified by the replay golden test: without this the ``last_act`` slice
        differs by up to 1.5 while every other slice matches to ~1e-7.
        """
        action = self._array(action, (16,), "action")
        proposed = self.motor_targets + action * self.action_scale
        clipped = np.clip(proposed, self.lower, self.upper)
        self.motor_targets = (
            self.ema_alpha * clipped
            + (1.0 - self.ema_alpha) * self.motor_targets
        )
        self.obs_act = self.last_act
        self.last_act = action.copy()
        return self.motor_targets.copy()
