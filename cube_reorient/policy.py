"""Pure-numpy inference for an exported Brax policy."""

from __future__ import annotations

from os import PathLike

import numpy as np


_OBS_DIM = 57
_ACT_DIM = 16
_LAYER_SIZES = (512, 256, 128, 32)
_PARAMETER_SHAPES = {
    "mean": (_OBS_DIM,),
    "std": (_OBS_DIM,),
    "W0": (_OBS_DIM, _LAYER_SIZES[0]),
    "b0": (_LAYER_SIZES[0],),
    "W1": (_LAYER_SIZES[0], _LAYER_SIZES[1]),
    "b1": (_LAYER_SIZES[1],),
    "W2": (_LAYER_SIZES[1], _LAYER_SIZES[2]),
    "b2": (_LAYER_SIZES[2],),
    "W3": (_LAYER_SIZES[2], _LAYER_SIZES[3]),
    "b3": (_LAYER_SIZES[3],),
}


def _scalar(value: np.ndarray, name: str) -> object:
    if value.shape != ():
        raise ValueError(f"expected metadata {name!r} to be scalar, got shape {value.shape}")
    return value.item()


class NumpyPolicy:
    """A deterministic policy whose forward pass depends only on numpy."""

    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        weights: tuple[np.ndarray, ...],
        biases: tuple[np.ndarray, ...],
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float64)
        self.std = np.asarray(std, dtype=np.float64)
        self.weights = tuple(np.asarray(value, dtype=np.float64) for value in weights)
        self.biases = tuple(np.asarray(value, dtype=np.float64) for value in biases)

    @classmethod
    def load(cls, npz_path: str | PathLike[str]) -> NumpyPolicy:
        """Load and validate an archive produced by ``tools/export_policy.py``."""
        with np.load(npz_path, allow_pickle=False) as archive:
            missing = set(_PARAMETER_SHAPES) - set(archive.files)
            metadata = {"obs_dim", "act_dim", "layer_sizes", "activation"}
            missing.update(metadata - set(archive.files))
            if missing:
                names = ", ".join(sorted(missing))
                raise ValueError(f"policy archive is missing required keys: {names}")

            arrays = {
                name: np.asarray(archive[name], dtype=np.float64).copy()
                for name in _PARAMETER_SHAPES
            }
            for name, expected_shape in _PARAMETER_SHAPES.items():
                if arrays[name].shape != expected_shape:
                    raise ValueError(
                        f"expected {name} shape {expected_shape}, got {arrays[name].shape}"
                    )

            obs_dim = int(_scalar(archive["obs_dim"], "obs_dim"))
            act_dim = int(_scalar(archive["act_dim"], "act_dim"))
            layer_sizes = tuple(int(value) for value in archive["layer_sizes"].tolist())
            activation = str(_scalar(archive["activation"], "activation"))

        if obs_dim != _OBS_DIM:
            raise ValueError(f"expected obs_dim {_OBS_DIM}, got {obs_dim}")
        if act_dim != _ACT_DIM:
            raise ValueError(f"expected act_dim {_ACT_DIM}, got {act_dim}")
        if layer_sizes != _LAYER_SIZES:
            raise ValueError(f"expected layer_sizes {_LAYER_SIZES}, got {layer_sizes}")
        if activation != "swish":
            raise ValueError(f"expected activation 'swish', got {activation!r}")
        if not np.all(np.isfinite(arrays["std"])) or np.any(arrays["std"] == 0.0):
            raise ValueError("std must contain only finite, non-zero values")

        return cls(
            mean=arrays["mean"],
            std=arrays["std"],
            weights=tuple(arrays[f"W{index}"] for index in range(4)),
            biases=tuple(arrays[f"b{index}"] for index in range(4)),
        )

    @staticmethod
    def _swish(value: np.ndarray) -> np.ndarray:
        sigmoid = np.empty_like(value)
        positive = value >= 0.0
        sigmoid[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
        exp_value = np.exp(value[~positive])
        sigmoid[~positive] = exp_value / (1.0 + exp_value)
        return value * sigmoid

    def __call__(self, obs57: np.ndarray) -> np.ndarray:
        """Return the 16-dimensional deterministic tanh action."""
        observation = np.asarray(obs57, dtype=np.float64)
        if observation.shape != (self.obs_dim,):
            raise ValueError(
                f"expected obs57 shape ({self.obs_dim},), got {observation.shape}"
            )

        hidden = (observation - self.mean) / self.std
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            hidden = self._swish(hidden @ weight + bias)
        logits = hidden @ self.weights[-1] + self.biases[-1]
        return np.tanh(logits[: self.act_dim])

    @property
    def obs_dim(self) -> int:
        return _OBS_DIM

    @property
    def act_dim(self) -> int:
        return _ACT_DIM
