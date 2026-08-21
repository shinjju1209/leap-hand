"""Per-joint deadband for stable robot angle commands."""

from __future__ import annotations

import numpy as np


class AngleDeadband:
    """Hold each output until its input moves by at least ``threshold``.

    The comparison is made against the last emitted value, not the previous
    input sample. Slow intentional motion therefore accumulates until it is
    large enough to cross the deadband.
    """

    def __init__(self, threshold: float = 1.2) -> None:
        if threshold < 0.0:
            raise ValueError("deadband threshold must be non-negative")
        self.threshold = float(threshold)
        self.reset()

    def reset(self) -> None:
        """Forget the previously emitted command."""
        self._output: np.ndarray | None = None

    def filter(self, value) -> np.ndarray:
        """Apply the deadband independently to every angle."""
        sample = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(sample)):
            raise ValueError("deadband input contains NaN or infinity")

        if self._output is None:
            self._output = sample.copy()
            return sample.copy()

        if sample.shape != self._output.shape:
            raise ValueError(
                f"Input shape changed from {self._output.shape} to {sample.shape}"
            )

        update_mask = np.abs(sample - self._output) >= self.threshold
        self._output[update_mask] = sample[update_mask]
        return self._output.copy()
