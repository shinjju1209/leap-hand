"""Vectorized One Euro Filter for real-time hand joint angles."""

from __future__ import annotations

import numpy as np


class OneEuroFilter:
    """Adaptive low-pass filter for scalar or NumPy-array signals.

    Slow motion is smoothed using ``min_cutoff``. As signal speed increases,
    ``beta`` raises the cutoff frequency so the output remains responsive.
    Timestamps must be supplied in seconds and strictly increase.
    """

    def __init__(
        self,
        min_cutoff: float = 0.5,
        beta: float = 0.08,
        derivative_cutoff: float = 1.0,
    ) -> None:
        if min_cutoff <= 0.0:
            raise ValueError("min_cutoff must be greater than zero")
        if beta < 0.0:
            raise ValueError("beta must be non-negative")
        if derivative_cutoff <= 0.0:
            raise ValueError("derivative_cutoff must be greater than zero")

        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.reset()

    def reset(self) -> None:
        """Forget all previous samples."""
        self._previous_time: float | None = None
        self._previous_raw: np.ndarray | None = None
        self._filtered_value: np.ndarray | None = None
        self._filtered_derivative: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: float | np.ndarray, dt: float) -> float | np.ndarray:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, value, timestamp: float) -> np.ndarray:
        """Filter one sample and return a copy of the filtered value."""
        sample = np.asarray(value, dtype=np.float64)
        if not np.all(np.isfinite(sample)):
            raise ValueError("One Euro input contains NaN or infinity")

        if self._previous_time is None:
            self._previous_time = float(timestamp)
            self._previous_raw = sample.copy()
            self._filtered_value = sample.copy()
            self._filtered_derivative = np.zeros_like(sample)
            return sample.copy()

        if sample.shape != self._previous_raw.shape:
            raise ValueError(
                f"Input shape changed from {self._previous_raw.shape} to {sample.shape}"
            )

        dt = float(timestamp) - self._previous_time
        if dt <= 0.0:
            raise ValueError("One Euro timestamps must strictly increase")

        derivative = (sample - self._previous_raw) / dt
        derivative_alpha = self._alpha(self.derivative_cutoff, dt)
        self._filtered_derivative = (
            derivative_alpha * derivative
            + (1.0 - derivative_alpha) * self._filtered_derivative
        )

        adaptive_cutoff = (
            self.min_cutoff + self.beta * np.abs(self._filtered_derivative)
        )
        value_alpha = self._alpha(adaptive_cutoff, dt)
        self._filtered_value = (
            value_alpha * sample
            + (1.0 - value_alpha) * self._filtered_value
        )

        self._previous_time = float(timestamp)
        self._previous_raw = sample.copy()
        return self._filtered_value.copy()

