"""MuJoCo backend for commanding the 16-DoF right LEAP Hand model."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import mujoco
import numpy as np

from hand_angles import ANGLE_NAMES


DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "mujoco"
    / "leap_hand"
    / "scene_right.xml"
)

# ANGLE_NAMES uses side-before-flex for each finger. The MuJoCo model uses
# flex-before-side (if_mcp_act, if_rot_act, ...), so actuator names are resolved
# explicitly instead of assuming that both 16-element arrays share an order.
INPUT_TO_ACTUATOR = (
    "if_rot_act",  # index_mcp_side
    "if_mcp_act",  # index_mcp_flex
    "if_pip_act",  # index_pip_flex
    "if_dip_act",  # index_dip_flex
    "mf_rot_act",  # middle_mcp_side
    "mf_mcp_act",  # middle_mcp_flex
    "mf_pip_act",  # middle_pip_flex
    "mf_dip_act",  # middle_dip_flex
    "rf_rot_act",  # ring_mcp_side
    "rf_mcp_act",  # ring_mcp_flex
    "rf_pip_act",  # ring_pip_flex
    "rf_dip_act",  # ring_dip_flex
    "th_cmc_act",  # thumb_cmc_side; verify direction during visual calibration
    "th_axl_act",  # thumb_cmc_flex; verify direction during visual calibration
    "th_mcp_act",  # thumb_mcp_flex
    "th_ipl_act",  # thumb_ip_flex
)


def _vector16(values: Sequence[float], name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (16,):
        raise ValueError(f"{name} must contain exactly 16 values, got {result.shape}")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains NaN or infinity")
    return result


class MujocoHandController:
    """Control the right LEAP Hand MJCF with human-hand angle commands.

    Input commands follow ``hand_angles.ANGLE_NAMES`` and use degrees. The
    MuJoCo position actuators receive radians after order, sign, offset, and
    control-range conversion. The optional collision-safe setter predicts
    hand self-contact in scratch simulation data before applying a target.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        *,
        signs: Sequence[float] | None = None,
        offsets_degrees: Sequence[float] | None = None,
        position_kp: float | None = None,
        velocity_kv: float | None = None,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo LEAP Hand model not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self.signs = _vector16(
            np.ones(16) if signs is None else signs,
            "signs",
        )
        if np.any(self.signs == 0.0):
            raise ValueError("signs cannot contain zero")
        self.offsets_degrees = _vector16(
            np.zeros(16) if offsets_degrees is None else offsets_degrees,
            "offsets_degrees",
        )

        self.actuator_ids = np.array(
            [
                mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    actuator_name,
                )
                for actuator_name in INPUT_TO_ACTUATOR
            ],
            dtype=np.int32,
        )
        missing = [
            name
            for name, actuator_id in zip(INPUT_TO_ACTUATOR, self.actuator_ids)
            if actuator_id < 0
        ]
        if missing:
            raise ValueError(f"MuJoCo model is missing actuators: {missing}")

        if position_kp is not None:
            if not np.isfinite(position_kp) or position_kp <= 0.0:
                raise ValueError("position_kp must be finite and positive")
            self.model.actuator_gainprm[self.actuator_ids, 0] = position_kp
            self.model.actuator_biasprm[self.actuator_ids, 1] = -position_kp
        if velocity_kv is not None:
            if not np.isfinite(velocity_kv) or velocity_kv < 0.0:
                raise ValueError("velocity_kv must be finite and non-negative")
            self.model.actuator_biasprm[self.actuator_ids, 2] = -velocity_kv

        joint_ids = self.model.actuator_trnid[self.actuator_ids, 0]
        self.qpos_addresses = self.model.jnt_qposadr[joint_ids]
        palm_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "palm",
        )
        if palm_body_id < 0:
            raise ValueError("MuJoCo model is missing the palm body")
        self._hand_body_ids = {
            body_id
            for body_id in range(1, self.model.nbody)
            if self._is_descendant_body(body_id, palm_body_id)
        }
        self._collision_data = mujoco.MjData(self.model)
        self._viewer = None
        self._target_radians = np.zeros(16, dtype=np.float64)
        self._last_safe_command_degrees = np.zeros(16, dtype=np.float64)
        self._last_collision_scale = 1.0
        self._last_predicted_self_contacts = 0
        self.reset()

    @property
    def timestep(self) -> float:
        """Return the MuJoCo physics timestep in seconds."""
        return float(self.model.opt.timestep)

    @property
    def target_radians(self) -> np.ndarray:
        """Return the last clipped actuator targets in human-angle order."""
        return self._target_radians.copy()

    @property
    def last_collision_scale(self) -> float:
        """Return the fraction of the latest requested pose that was applied."""
        return self._last_collision_scale

    @property
    def last_predicted_self_contacts(self) -> int:
        """Return self-contacts found in the latest unmodified target pose."""
        return self._last_predicted_self_contacts

    def _is_descendant_body(self, body_id: int, ancestor_id: int) -> bool:
        current_id = body_id
        while current_id > 0:
            if current_id == ancestor_id:
                return True
            current_id = int(self.model.body_parentid[current_id])
        return False

    def _target_from_degrees(self, angles_degrees: Sequence[float]) -> np.ndarray:
        human_degrees = _vector16(angles_degrees, "angles_degrees")
        model_degrees = human_degrees * self.signs + self.offsets_degrees
        target_radians = np.deg2rad(model_degrees)
        control_ranges = self.model.actuator_ctrlrange[self.actuator_ids]
        return np.clip(
            target_radians,
            control_ranges[:, 0],
            control_ranges[:, 1],
        )

    def set_target_degrees(self, angles_degrees: Sequence[float]) -> np.ndarray:
        """Set 16 position targets and return their clipped radian values."""
        target_radians = self._target_from_degrees(angles_degrees)
        self.data.ctrl[self.actuator_ids] = target_radians
        self._target_radians = target_radians
        return target_radians.copy()

    def _predicted_self_contact_count(
        self,
        angles_degrees: Sequence[float],
    ) -> int:
        """Count hand-vs-hand contacts at a candidate kinematic target pose."""
        target_radians = self._target_from_degrees(angles_degrees)
        mujoco.mj_resetData(self.model, self._collision_data)
        self._collision_data.qpos[self.qpos_addresses] = target_radians
        mujoco.mj_forward(self.model, self._collision_data)

        contact_count = 0
        for contact in self._collision_data.contact:
            body1 = int(self.model.geom_bodyid[contact.geom1])
            body2 = int(self.model.geom_bodyid[contact.geom2])
            if body1 in self._hand_body_ids and body2 in self._hand_body_ids:
                contact_count += 1
        return contact_count

    def set_collision_safe_target_degrees(
        self,
        angles_degrees: Sequence[float],
        *,
        search_iterations: int = 12,
        backoff_ratio: float = 0.98,
    ) -> np.ndarray:
        """Apply the largest contact-free fraction of a requested hand pose.

        The candidate is checked kinematically in scratch ``MjData`` before it
        reaches the live simulation. If self-contact is predicted, a binary
        search advances only from the previously safe pose toward the new
        request. This prevents the whole hand from snapping open toward zero.
        The returned vector is the applied command in human-angle degrees.
        """
        requested = _vector16(angles_degrees, "angles_degrees")
        if not isinstance(search_iterations, int) or search_iterations < 1:
            raise ValueError("search_iterations must be a positive integer")
        if not np.isfinite(backoff_ratio) or not 0.0 < backoff_ratio <= 1.0:
            raise ValueError("backoff_ratio must be in the interval (0, 1]")

        predicted_contacts = self._predicted_self_contact_count(requested)
        self._last_predicted_self_contacts = predicted_contacts
        if predicted_contacts == 0:
            self._last_collision_scale = 1.0
            self._last_safe_command_degrees = requested.copy()
            self.set_target_degrees(requested)
            return requested.copy()

        safe_origin = self._last_safe_command_degrees.copy()
        if self._predicted_self_contact_count(safe_origin) > 0:
            safe_origin = np.zeros(16, dtype=np.float64)
        if self._predicted_self_contact_count(safe_origin) > 0:
            raise RuntimeError("Neutral MuJoCo hand pose contains self-contact")

        safe_motion_fraction = 0.0
        colliding_motion_fraction = 1.0
        for _ in range(search_iterations):
            candidate_fraction = 0.5 * (
                safe_motion_fraction + colliding_motion_fraction
            )
            candidate = safe_origin + candidate_fraction * (
                requested - safe_origin
            )
            if self._predicted_self_contact_count(candidate) == 0:
                safe_motion_fraction = candidate_fraction
            else:
                colliding_motion_fraction = candidate_fraction

        applied_motion_fraction = safe_motion_fraction * backoff_ratio
        applied = safe_origin + applied_motion_fraction * (
            requested - safe_origin
        )
        requested_magnitude = float(np.linalg.norm(requested))
        if requested_magnitude > 1e-9:
            self._last_collision_scale = min(
                1.0,
                float(np.linalg.norm(applied)) / requested_magnitude,
            )
        else:
            self._last_collision_scale = 1.0
        self._last_safe_command_degrees = applied.copy()
        self.set_target_degrees(applied)
        return applied.copy()

    def step(self, steps: int = 1) -> None:
        """Advance the physics simulation by an integer number of steps."""
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("steps must be a positive integer")
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

    def forward(self) -> None:
        """Recompute kinematics and contacts after directly resetting state."""
        mujoco.mj_forward(self.model, self.data)

    def step_for(self, duration_seconds: float) -> int:
        """Advance by approximately ``duration_seconds`` and return step count."""
        if not np.isfinite(duration_seconds) or duration_seconds <= 0.0:
            raise ValueError("duration_seconds must be finite and greater than zero")
        steps = max(1, int(round(duration_seconds / self.timestep)))
        self.step(steps)
        return steps

    def get_joint_degrees(self) -> np.ndarray:
        """Read simulated joints back in human-angle order and degrees."""
        model_degrees = np.rad2deg(self.data.qpos[self.qpos_addresses])
        return (model_degrees - self.offsets_degrees) / self.signs

    def reset(self) -> None:
        """Reset simulation state and command a neutral zero-degree pose."""
        mujoco.mj_resetData(self.model, self.data)
        self.set_target_degrees(np.zeros(16, dtype=np.float64))
        self._last_safe_command_degrees = np.zeros(16, dtype=np.float64)
        self._last_collision_scale = 1.0
        self._last_predicted_self_contacts = 0
        mujoco.mj_forward(self.model, self.data)
        self.sync_viewer()

    def launch_viewer(self):
        """Open a passive MuJoCo viewer and return its handle."""
        if self._viewer is None:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
        return self._viewer

    def render_bgr(
        self,
        height: int = 480,
        width: int = 640,
        camera_id: int = -1,
    ) -> np.ndarray:
        """Render the current 3D hand scene to a BGR numpy image."""
        if not hasattr(self, "_renderer") or self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=height, width=width)
        self._renderer.update_scene(self.data, camera=camera_id)
        rgb = self._renderer.render()
        import cv2

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def sync_viewer(self) -> None:
        """Refresh the viewer if it has been launched and remains open."""
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self) -> None:
        """Close the optional viewer."""
        if self._viewer is not None:
            try:
                if self._viewer.is_running():
                    self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    def __enter__(self) -> "MujocoHandController":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


__all__ = [
    "ANGLE_NAMES",
    "DEFAULT_MODEL_PATH",
    "INPUT_TO_ACTUATOR",
    "MujocoHandController",
]
