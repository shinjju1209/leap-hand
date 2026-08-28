"""Sim-to-Real Reinforcement Learning (RL) Policy Runner for LEAP Hand v1.

Provides policy loaders (TorchScript, ONNX, and Callable mock backends),
an observation history manager, and an action processor with EMA smoothing and
joint limit clamping for in-hand manipulation and cube rotation policies.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from hand_angles import ANGLE_NAMES
from leap_hand_hardware_controller import (
    SIM_MAX_RADIANS,
    SIM_MIN_RADIANS,
    clip_sim_radians,
)

# Standard default grasp pose for In-Hand Cube Rotation (in radians, zero-open frame)
DEFAULT_MANIPULATION_POSE_RADIANS = np.array(
    [
        0.0, 0.8, 0.8, 0.4,  # index (mcp_side, mcp_flex, pip_flex, dip_flex)
        0.0, 0.8, 0.8, 0.4,  # middle
        0.0, 0.8, 0.8, 0.4,  # ring
        1.2, 0.5, 0.5, 0.5,  # thumb (cmc_side, cmc_flex, mcp_flex, ip_flex)
    ],
    dtype=np.float64,
)


def _vector16(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (16,):
        raise ValueError(f"{name} must contain exactly 16 values, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


class PolicyBackend(ABC):
    """Abstract interface for policy inference."""

    @abstractmethod
    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Run forward inference on a 1D or 2D observation array.

        Args:
            obs: 1D (obs_dim,) or 2D (1, obs_dim) observation array.

        Returns:
            1D numpy array of actions with shape (16,).
        """
        ...


class CallablePolicyBackend(PolicyBackend):
    """Wrap any Python callable or function as a policy backend."""

    def __init__(self, fn: Callable[[np.ndarray], np.ndarray]) -> None:
        if not callable(fn):
            raise TypeError("fn must be callable")
        self.fn = fn

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        action = self.fn(np.asarray(obs, dtype=np.float32))
        return np.asarray(action, dtype=np.float64).reshape(16)


class TorchScriptPolicyBackend(PolicyBackend):
    """Execute a PyTorch / TorchScript policy (.pt or .pth)."""

    def __init__(self, model_path: str | Path, device: str = "cpu") -> None:
        import torch

        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"TorchScript model not found: {self.model_path}")

        self.device = torch.device(device)
        try:
            self.model = torch.jit.load(str(self.model_path), map_location=self.device)
            self.model.eval()
        except Exception as exc:
            # Fallback to standard torch.load if it's an nn.Module
            try:
                loaded = torch.load(str(self.model_path), map_location=self.device)
                if hasattr(loaded, "eval"):
                    loaded.eval()
                    self.model = loaded
                else:
                    raise exc
            except Exception:
                raise RuntimeError(
                    f"Failed to load TorchScript policy from {self.model_path}: {exc}"
                ) from exc

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        import torch

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            action_tensor = self.model(obs_tensor)
            if isinstance(action_tensor, (tuple, list)):
                action_tensor = action_tensor[0]
            elif isinstance(action_tensor, dict):
                action_tensor = action_tensor.get("action", next(iter(action_tensor.values())))
            elif hasattr(action_tensor, "loc"):
                action_tensor = action_tensor.loc

        action_np = action_tensor.squeeze(0).detach().cpu().numpy()
        return np.asarray(action_np, dtype=np.float64).reshape(16)


# Official LEAP_Hand_Sim joint permutations and canonical limits
# Isaac Gym order: Index(0~3), Thumb(4~7), Middle(8~11), Ring(12~15)
# REAL / ANGLE_NAMES order: Index(0~3), Middle(4~7), Ring(8~11), Thumb(12~15)
OFFICIAL_REAL_TO_SIM = np.array([0, 1, 2, 3, 12, 13, 14, 15, 4, 5, 6, 7, 8, 9, 10, 11], dtype=np.int64)
OFFICIAL_SIM_TO_REAL = np.array([0, 1, 2, 3, 8, 9, 10, 11, 12, 13, 14, 15, 4, 5, 6, 7], dtype=np.int64)

OFFICIAL_CANONICAL_POSE_SIM = np.array([
    0.2078, -0.9765, 1.7760, -0.1653,
    1.0993,  1.3305, 0.8575,  1.0603,
    -0.1046, -0.0162, 1.5904,  0.1451,
    0.1976,  0.8854, 1.7806,  0.0184,
], dtype=np.float64)
OFFICIAL_CANONICAL_POSE_REAL = OFFICIAL_CANONICAL_POSE_SIM[OFFICIAL_SIM_TO_REAL]

OFFICIAL_DOF_LOWER = np.array([
    -1.5716, -0.4416, -1.2216, -1.3416,
    -0.519205, -0.57159, -0.25159, -1.3416,
    -1.5716, -0.4416, -1.2216, -1.3416,
    -1.5716, -0.4416, -1.2216, -1.3416,
], dtype=np.float64)

OFFICIAL_DOF_UPPER = np.array([
    1.5584, 1.8584, 1.8584, 1.8584,
    1.7408, 1.96841, 1.8584, 1.8584,
    1.5584, 1.8584, 1.8584, 1.8584,
    1.5584, 1.8584, 1.8584, 1.8584,
], dtype=np.float64)


class LeapHandOfficialPolicyBackend(PolicyBackend):
    """Execute the official LEAP_Hand_Sim LeapHand.pth actor-critic policy with GRU memory."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        *,
        action_scale: float = 1.0 / 24.0,
        phase_period: float = 2.0,
    ) -> None:
        import torch
        import torch.nn as nn

        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Official model checkpoint not found: {self.model_path}")

        self.device = torch.device(device)
        self.action_scale = float(action_scale)
        self.phase_period = float(phase_period)

        data = torch.load(str(self.model_path), map_location=self.device, weights_only=False)
        model_dict = data["model"] if isinstance(data, dict) and "model" in data else data

        class OfficialActor(nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer(
                    "running_mean",
                    model_dict["running_mean_std.running_mean"].to(torch.float32),
                )
                self.register_buffer(
                    "running_var",
                    model_dict["running_mean_std.running_var"].to(torch.float32),
                )

                self.rnn = nn.GRU(102, 256, batch_first=True)
                self.rnn.weight_ih_l0.data.copy_(model_dict["a2c_network.rnn.rnn.weight_ih_l0"])
                self.rnn.weight_hh_l0.data.copy_(model_dict["a2c_network.rnn.rnn.weight_hh_l0"])
                self.rnn.bias_ih_l0.data.copy_(model_dict["a2c_network.rnn.rnn.bias_ih_l0"])
                self.rnn.bias_hh_l0.data.copy_(model_dict["a2c_network.rnn.rnn.bias_hh_l0"])

                self.layer_norm = nn.LayerNorm(256)
                self.layer_norm.weight.data.copy_(model_dict["a2c_network.layer_norm.weight"])
                self.layer_norm.bias.data.copy_(model_dict["a2c_network.layer_norm.bias"])

                self.mlp_0 = nn.Linear(256, 512)
                self.mlp_0.weight.data.copy_(model_dict["a2c_network.actor_mlp.0.weight"])
                self.mlp_0.bias.data.copy_(model_dict["a2c_network.actor_mlp.0.bias"])

                self.mlp_2 = nn.Linear(512, 256)
                self.mlp_2.weight.data.copy_(model_dict["a2c_network.actor_mlp.2.weight"])
                self.mlp_2.bias.data.copy_(model_dict["a2c_network.actor_mlp.2.bias"])

                self.mlp_4 = nn.Linear(256, 128)
                self.mlp_4.weight.data.copy_(model_dict["a2c_network.actor_mlp.4.weight"])
                self.mlp_4.bias.data.copy_(model_dict["a2c_network.actor_mlp.4.bias"])

                self.mu = nn.Linear(128, 16)
                self.mu.weight.data.copy_(model_dict["a2c_network.mu.weight"])
                self.mu.bias.data.copy_(model_dict["a2c_network.mu.bias"])

                self.elu = nn.ELU()
                self.hidden_state = None

            def reset_hidden(self):
                self.hidden_state = None

            def forward(self, obs: torch.Tensor) -> torch.Tensor:
                if obs.ndim == 1:
                    obs = obs.unsqueeze(0)
                obs_f32 = obs.to(torch.float32)
                std = torch.sqrt(self.running_var + 1e-8)
                norm_obs = torch.clamp((obs_f32 - self.running_mean) / std, -5.0, 5.0)
                if norm_obs.ndim == 2:
                    norm_obs = norm_obs.unsqueeze(1)
                rnn_out, self.hidden_state = self.rnn(norm_obs, self.hidden_state)
                x = rnn_out.squeeze(1)
                x = self.layer_norm(x)
                x = self.elu(self.mlp_0(x))
                x = self.elu(self.mlp_2(x))
                x = self.elu(self.mlp_4(x))
                return torch.clamp(self.mu(x), -1.0, 1.0)

        self.model = OfficialActor().to(self.device)
        self.model.eval()

        self.sim_target = OFFICIAL_CANONICAL_POSE_SIM.copy()
        self.step_counter = 0

    def reset(self) -> None:
        self.model.reset_hidden()
        self.sim_target = OFFICIAL_CANONICAL_POSE_SIM.copy()
        self.step_counter = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        import torch

        obs_arr = np.asarray(obs, dtype=np.float32)
        if obs_arr.shape[-1] < 102:
            # Build official 102-dim observation from current state
            t_sec = self.step_counter * 0.05
            phase = np.array([
                np.sin(2.0 * np.pi * t_sec / self.phase_period),
                np.cos(2.0 * np.pi * t_sec / self.phase_period),
            ], dtype=np.float32)
            if len(obs_arr) >= 16:
                cur_sim = obs_arr[:16][OFFICIAL_REAL_TO_SIM].astype(np.float64)
            else:
                cur_sim = self.sim_target.copy()
            unscaled = (2.0 * cur_sim - OFFICIAL_DOF_UPPER - OFFICIAL_DOF_LOWER) / (
                OFFICIAL_DOF_UPPER - OFFICIAL_DOF_LOWER
            )
            step_chunk = np.concatenate([unscaled, self.sim_target, phase]).astype(np.float32)
            obs_arr = np.concatenate([step_chunk, step_chunk, step_chunk])

        self.step_counter += 1
        obs_tensor = torch.as_tensor(obs_arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            action_tensor = self.model(obs_tensor)
        action_sim = action_tensor.squeeze(0).detach().cpu().numpy().astype(np.float64)

        # Integrate target in SIM frame, then permute to REAL frame
        self.sim_target = np.clip(
            self.sim_target + self.action_scale * action_sim,
            OFFICIAL_DOF_LOWER,
            OFFICIAL_DOF_UPPER,
        )
        action_real = action_sim[OFFICIAL_SIM_TO_REAL]
        return action_real


class OnnxPolicyBackend(PolicyBackend):
    """Execute an ONNX exported policy (.onnx)."""

    def __init__(self, model_path: str | Path) -> None:
        import onnxruntime as ort

        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs_array = np.asarray(obs, dtype=np.float32)
        if obs_array.ndim == 1:
            obs_array = np.expand_dims(obs_array, axis=0)

        outputs = self.session.run(
            [self.output_name],
            {self.input_name: obs_array},
        )
        action_np = outputs[0]
        return np.asarray(action_np, dtype=np.float64).reshape(16)


class PlaygroundJaxPolicyBackend(PolicyBackend):
    """Execute a MuJoCo Playground (JAX / Brax PPO) trained policy (.npz or .pt)."""

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"JAX model not found: {self.model_path}")

        self.weights: dict[str, np.ndarray] = {}
        if self.model_path.suffix.lower() == ".npz":
            data = np.load(str(self.model_path))
            self.weights = {k: data[k] for k in data.files}
        else:
            import torch
            loaded = torch.load(str(self.model_path), map_location="cpu", weights_only=False)
            if isinstance(loaded, dict) and "weights" in loaded:
                self.weights = {
                    k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v))
                    for k, v in loaded["weights"].items()
                }
            elif isinstance(loaded, dict):
                self.weights = {
                    k: (v.numpy() if hasattr(v, "numpy") else np.asarray(v))
                    for k, v in loaded.items()
                }

        # Normalization parameters
        self.obs_mean = self.weights.get("0/mean/state", self.weights.get("mean/state", self.weights.get("obs_mean")))
        self.obs_std = self.weights.get("0/std/state", self.weights.get("std/state", self.weights.get("obs_std")))

        # Filter policy layers (exclude critic/value network which is in 2/params or value/)
        policy_keys = [
            k for k in self.weights
            if ("hidden_" in k) and not (k.startswith("2/") or "value" in k.lower())
        ]

        self.layers: list[tuple[np.ndarray, np.ndarray]] = []
        layer_indices = sorted(
            set(
                int(k.split("hidden_")[1].split("/")[0])
                for k in policy_keys
                if k.split("hidden_")[1].split("/")[0].isdigit()
            )
        )
        for idx in layer_indices:
            k_candidates = [k for k in policy_keys if f"hidden_{idx}/kernel" in k]
            b_candidates = [k for k in policy_keys if f"hidden_{idx}/bias" in k]
            if k_candidates and b_candidates:
                self.layers.append((self.weights[k_candidates[0]], self.weights[b_candidates[0]]))

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, dtype=np.float32)
        if x.ndim == 2:
            x = x[0]

        if not self.layers:
            return np.zeros(16, dtype=np.float64)

        # Adapt observation dimension if needed
        expected_dim = self.layers[0][0].shape[0]
        if x.shape[0] != expected_dim:
            if x.shape[0] > expected_dim:
                x = x[:expected_dim]
            else:
                x = np.pad(x, (0, expected_dim - x.shape[0]))

        # Apply observation normalizer if present
        if self.obs_mean is not None and self.obs_std is not None:
            if x.shape == self.obs_mean.shape:
                x = (x - self.obs_mean) / (self.obs_std + 1e-8)

        for i, (kernel, bias) in enumerate(self.layers):
            x = np.matmul(x, kernel) + bias
            if i < len(self.layers) - 1:
                # SiLU / Swish activation: x * sigmoid(x)
                x = x / (1.0 + np.exp(-np.clip(x, -20.0, 20.0)))

        # Output is 16 action means
        action = x[:16]
        return np.asarray(action, dtype=np.float64).reshape(16)


def load_policy(
    source: str | Path | Callable[[np.ndarray], np.ndarray] | PolicyBackend,
    device: str = "cpu",
) -> PolicyBackend:
    """Auto-detect and instantiate a PolicyBackend from path or callable."""
    if isinstance(source, PolicyBackend):
        return source
    if callable(source) and not isinstance(source, (str, Path)):
        return CallablePolicyBackend(source)

    path = Path(source).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Policy file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".npz":
        return PlaygroundJaxPolicyBackend(path)

    if suffix in (".pt", ".pth"):
        # Check if it's a Playground JAX export
        try:
            import torch
            data = torch.load(str(path), map_location=device, weights_only=False)
            if isinstance(data, dict) and data.get("framework") == "mujoco_playground":
                return PlaygroundJaxPolicyBackend(path)
            if isinstance(data, dict) and "model" in data:
                if any("a2c_network" in k for k in data["model"].keys()):
                    return LeapHandOfficialPolicyBackend(path, device=device)
        except Exception:
            pass
        return TorchScriptPolicyBackend(path, device=device)

    if suffix == ".onnx":
        return OnnxPolicyBackend(path)

    # Default fallback to TorchScript
    try:
        return TorchScriptPolicyBackend(path, device=device)
    except Exception:
        return OnnxPolicyBackend(path)


class ObservationManager:
    """Constructs, normalizes, and stacks RL observation history vectors.

    Standard observation format per step:
        [q_t - q_default (16), a_{t-1} (16), command (optional)]
    Full observation vector stacks `history_length` consecutive steps.
    """

    def __init__(
        self,
        default_joint_pose_radians: Sequence[float] = DEFAULT_MANIPULATION_POSE_RADIANS,
        history_length: int = 3,
        *,
        obs_mean: Sequence[float] | None = None,
        obs_std: Sequence[float] | None = None,
        clip_obs: float | None = None,
        include_command: bool = True,
    ) -> None:
        self.default_joint_pose = _vector16(
            default_joint_pose_radians,
            "default_joint_pose_radians",
        )
        if not isinstance(history_length, int) or history_length < 1:
            raise ValueError("history_length must be a positive integer")

        self.history_length = history_length
        self.include_command = include_command
        self.clip_obs = float(clip_obs) if clip_obs is not None else None
        self.obs_mean = (
            np.asarray(obs_mean, dtype=np.float64) if obs_mean is not None else None
        )
        self.obs_std = (
            np.asarray(obs_std, dtype=np.float64) if obs_std is not None else None
        )

        self._history: deque[np.ndarray] = deque(maxlen=self.history_length)
        self._last_action = np.zeros(16, dtype=np.float64)

    @property
    def step_dim(self) -> int:
        return 32 + (1 if self.include_command else 0)

    @property
    def total_dim(self) -> int:
        return self.step_dim * self.history_length

    def build_step_obs(
        self,
        current_joint_radians: Sequence[float],
        last_action: Sequence[float] | None = None,
        target_command: float = 1.0,
    ) -> np.ndarray:
        """Create a single-step observation array."""
        current_rad = _vector16(current_joint_radians, "current_joint_radians")
        pos_error = current_rad - self.default_joint_pose

        action = (
            self._last_action
            if last_action is None
            else _vector16(last_action, "last_action")
        )
        self._last_action = action.copy()

        parts = [pos_error, action]
        if self.include_command:
            parts.append(np.array([float(target_command)], dtype=np.float64))

        return np.concatenate(parts).astype(np.float64)

    def update_and_get_observation(
        self,
        current_joint_radians: Sequence[float],
        last_action: Sequence[float] | None = None,
        target_command: float = 1.0,
    ) -> np.ndarray:
        """Append latest step observation to history and return the stacked 1D vector."""
        step_obs = self.build_step_obs(
            current_joint_radians,
            last_action=last_action,
            target_command=target_command,
        )

        # Pre-fill buffer with initial step if history is not full
        if len(self._history) == 0:
            for _ in range(self.history_length):
                self._history.append(step_obs.copy())
        else:
            self._history.append(step_obs)

        stacked = np.concatenate(list(self._history)).astype(np.float64)

        if self.obs_mean is not None and self.obs_std is not None:
            stacked = (stacked - self.obs_mean) / (self.obs_std + 1e-8)

        if self.clip_obs is not None and self.clip_obs > 0.0:
            stacked = np.clip(stacked, -self.clip_obs, self.clip_obs)

        return stacked

    def reset(self) -> None:
        """Clear observation history and reset last action state."""
        self._history.clear()
        self._last_action = np.zeros(16, dtype=np.float64)


class ActionProcessor:
    """Scales, smooths (EMA), and clamps policy actions to safe LEAP Hand targets."""

    def __init__(
        self,
        default_joint_pose_radians: Sequence[float] = DEFAULT_MANIPULATION_POSE_RADIANS,
        *,
        action_scale: float = 0.1,
        ema_alpha: float = 0.8,
        min_joint_bounds_radians: Sequence[float] = SIM_MIN_RADIANS,
        max_joint_bounds_radians: Sequence[float] = SIM_MAX_RADIANS,
    ) -> None:
        self.default_joint_pose = _vector16(
            default_joint_pose_radians,
            "default_joint_pose_radians",
        )
        if not np.isfinite(action_scale) or action_scale <= 0.0:
            raise ValueError("action_scale must be finite and positive")
        if not np.isfinite(ema_alpha) or not 0.0 < ema_alpha <= 1.0:
            raise ValueError("ema_alpha must be in the interval (0, 1]")

        self.action_scale = float(action_scale)
        self.ema_alpha = float(ema_alpha)
        self.min_bounds = _vector16(min_joint_bounds_radians, "min_joint_bounds_radians")
        self.max_bounds = _vector16(max_joint_bounds_radians, "max_joint_bounds_radians")

        self._last_smoothed_action: np.ndarray | None = None

    def process(
        self,
        raw_action: Sequence[float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Process raw action output from policy.

        Returns:
            (target_radians, target_degrees, smoothed_action)
        """
        action = _vector16(raw_action, "raw_action")

        # Exponential Moving Average (EMA) action filter
        if self._last_smoothed_action is None:
            smoothed = action.copy()
        else:
            smoothed = self.ema_alpha * action + (1.0 - self.ema_alpha) * self._last_smoothed_action
        self._last_smoothed_action = smoothed.copy()

        # q_target = q_default + action_scale * a
        target_radians = self.default_joint_pose + self.action_scale * smoothed

        # Clamp within LEAPsim bounds
        target_radians = np.clip(target_radians, self.min_bounds, self.max_bounds)
        target_degrees = np.rad2deg(target_radians)

        return target_radians, target_degrees, smoothed.copy()

    def reset(self) -> None:
        """Reset internal EMA action smoothing state."""
        self._last_smoothed_action = None


class RLPolicyRunner:
    """Coordinates policy execution, observation management, and action processing."""

    def __init__(
        self,
        policy: str | Path | Callable[[np.ndarray], np.ndarray] | PolicyBackend,
        default_joint_pose_radians: Sequence[float] = DEFAULT_MANIPULATION_POSE_RADIANS,
        *,
        control_hz: float = 20.0,
        action_scale: float = 0.1,
        ema_alpha: float = 0.8,
        history_length: int = 3,
        include_command: bool = True,
        obs_mean: Sequence[float] | None = None,
        obs_std: Sequence[float] | None = None,
        clip_obs: float | None = None,
        device: str = "cpu",
    ) -> None:
        self.policy = load_policy(policy, device=device)
        self.control_hz = float(control_hz)
        if self.control_hz <= 0.0:
            raise ValueError("control_hz must be positive")
        self.control_dt = 1.0 / self.control_hz

        self.default_joint_pose = _vector16(
            default_joint_pose_radians,
            "default_joint_pose_radians",
        )
        self.obs_manager = ObservationManager(
            default_joint_pose_radians=self.default_joint_pose,
            history_length=history_length,
            obs_mean=obs_mean,
            obs_std=obs_std,
            clip_obs=clip_obs,
            include_command=include_command,
        )
        self.action_processor = ActionProcessor(
            default_joint_pose_radians=self.default_joint_pose,
            action_scale=action_scale,
            ema_alpha=ema_alpha,
        )
        self._last_action = np.zeros(16, dtype=np.float64)

    @property
    def last_action(self) -> np.ndarray:
        return self._last_action.copy()

    def step(
        self,
        current_joint_angles: Sequence[float],
        *,
        target_command: float = 1.0,
        is_degrees: bool = True,
        extra_obs: Sequence[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Perform one step of RL policy execution.

        Args:
            current_joint_angles: 16 joint angles (degrees if is_degrees=True, else radians).
            target_command: Command parameter (e.g., +1.0 for CCW rotation, -1.0 for CW).
            is_degrees: Whether input angles are in degrees.
            extra_obs: Optional additional state observations (e.g., cube position & quaternion).

        Returns:
            (target_degrees, target_radians, raw_action)
        """
        angles = _vector16(current_joint_angles, "current_joint_angles")
        current_rad = np.deg2rad(angles) if is_degrees else angles

        if extra_obs is not None:
            extra_arr = np.asarray(extra_obs, dtype=np.float64).ravel()
            obs = np.concatenate([current_rad, self._last_action, extra_arr]).astype(np.float64)
        else:
            obs = self.obs_manager.update_and_get_observation(
                current_rad,
                last_action=self._last_action,
                target_command=target_command,
            )

        raw_action = self.policy(obs)
        self._last_action = raw_action.copy()

        target_rad, target_deg, _ = self.action_processor.process(raw_action)
        return target_deg, target_rad, raw_action

    def reset(self) -> None:
        """Reset runner states (history buffer and action filter)."""
        self.obs_manager.reset()
        self.action_processor.reset()
        self._last_action = np.zeros(16, dtype=np.float64)


__all__ = [
    "DEFAULT_MANIPULATION_POSE_RADIANS",
    "ActionProcessor",
    "CallablePolicyBackend",
    "ObservationManager",
    "OnnxPolicyBackend",
    "PolicyBackend",
    "RLPolicyRunner",
    "TorchScriptPolicyBackend",
    "load_policy",
]
