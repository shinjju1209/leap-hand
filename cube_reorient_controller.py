"""In-hand cube reorientation, driven by the trained RL policy in simulation.

This is to the reorientation demo what MujocoHandController is to the showcase:
the booth owns the screen and the buttons, this owns the physics. It steps the
MuJoCo Playground LeapCubeReorient scene at the policy's own 20 Hz, renders the
scene to a BGR image the kiosk can blit, and holds the goal the visitor sets.

Simulation only, by construction. Nothing here imports or reaches a hardware
controller, and the booth must not route this screen's output to the hand: the
policy was trained against a hand mounted at a specific angle and commands
joint targets far faster than the teleop path does, so sending them to a real
LEAP Hand is not something to do by accident. Watching it in simulation is the
point of the demo.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from cube_reorient import (
    GoalScheduler,
    MujocoSimBackend,
    NumpyPolicy,
    ObservationBuilder,
    ori_error,
)

DEFAULT_POLICY_PATH = Path("models/cube_reorient_policy.npz")

# The scene defines a camera that frames the hand on the right and the goal
# cube on the left, which is exactly the demo's story: make one match the other.
DEFAULT_CAMERA = "side"

IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0])


class CubeReorientController:
    """Run the reorientation policy in simulation and render it."""

    def __init__(
        self,
        policy_path: str | Path = DEFAULT_POLICY_PATH,
        *,
        playground_root: str | Path | None = None,
        seed: int = 0,
        tilt_degrees: float = 0.0,
        auto_goal: bool = False,
    ) -> None:
        path = Path(policy_path)
        if not path.is_file():
            raise FileNotFoundError(f"Cube reorientation policy not found: {path}")

        self.policy = NumpyPolicy.load(str(path))
        self.backend = MujocoSimBackend(playground_root=playground_root)
        self.builder = ObservationBuilder()
        self.seed = int(seed)

        if tilt_degrees:
            # The model mounts the hand with the fingers about 20 degrees below
            # horizontal, and that tilt is functional: gravity cradles the cube
            # against the palm. Tilting gravity asks what a hand mounted at a
            # different angle would do, without touching the model's geometry.
            theta = math.radians(tilt_degrees)
            self.backend.model.opt.gravity[:] = [
                -9.81 * math.sin(theta),
                0.0,
                -9.81 * math.cos(theta),
            ]
        self.tilt_degrees = float(tilt_degrees)

        self._rng = np.random.default_rng(self.seed)
        self._goal = IDENTITY_QUAT.copy()
        self._scheduler = GoalScheduler(rng=np.random.default_rng(self.seed))
        self.auto_goal = bool(auto_goal)

        self.paused = False
        self.steps = 0
        self.successes = 0
        self.drops = 0
        self._last_step_time: float | None = None
        self._renderer: Any | None = None

        self.reset()

    # -- simulation ---------------------------------------------------------

    def reset(self) -> None:
        """Return the cube and hand to the scene's home keyframe."""
        self.backend.reset(seed=self.seed + self.drops)
        self.builder.reset(self.backend.read_joint_angles())
        self._last_step_time = None

    @property
    def dt(self) -> float:
        return self.backend.dt

    def step_if_due(self, now: float) -> bool:
        """Advance one policy tick if a control period has passed.

        The kiosk redraws as fast as the display allows; the policy has to run
        at the rate it was trained at, so the two are decoupled here rather
        than by slowing the whole loop down.
        """
        if self._last_step_time is not None and now - self._last_step_time < self.dt:
            return False
        self._last_step_time = now
        self.step()
        return True

    def step(self) -> None:
        """One control period: observe, act, advance the physics."""
        joints = self.backend.read_joint_angles()
        cube_position, cube_quaternion = self.backend.cube_pose()
        goal = self._scheduler.goal_quat if self.auto_goal else self._goal

        if not self.paused:
            observation = self.builder.build(
                joints,
                self.backend.palm_position(),
                cube_position,
                cube_quaternion,
                goal,
            )
            targets = self.builder.apply_action(self.policy(observation))
            self.backend.write_motor_targets(targets)
            if self.auto_goal:
                self._scheduler.step(cube_quaternion)
                self.successes += int(self._scheduler.success)
            self.steps += 1

        if self.backend.cube_dropped():
            self.drops += 1
            self.reset()
            return

        # The goal body is a mocap; posing it is what draws the target cube.
        self.backend.data.mocap_quat[0] = goal
        mujoco.mj_forward(self.backend.model, self.backend.data)

    # -- goal ---------------------------------------------------------------

    @property
    def goal(self) -> np.ndarray:
        return (self._scheduler.goal_quat if self.auto_goal else self._goal).copy()

    @property
    def goal_error_degrees(self) -> float:
        _, cube_quaternion = self.backend.cube_pose()
        return float(math.degrees(ori_error(cube_quaternion, self.goal)))

    def _set_goal(self, quaternion: np.ndarray) -> None:
        self._goal = np.asarray(quaternion, dtype=np.float64)
        self._goal = self._goal / np.linalg.norm(self._goal)

    def rotate_goal(self, axis: int, degrees: float) -> None:
        """Turn the goal by a quarter turn about a world axis."""
        from cube_reorient.quat import quat_mul

        half = math.radians(degrees) / 2.0
        delta = np.zeros(4)
        delta[0] = math.cos(half)
        delta[axis + 1] = math.sin(half)
        self._set_goal(quat_mul(delta, self.goal))

    def randomize_goal(self) -> None:
        quaternion = self._rng.normal(size=4)
        self._set_goal(quaternion / np.linalg.norm(quaternion))

    def reset_goal(self) -> None:
        self._set_goal(IDENTITY_QUAT)

    def adopt_cube_goal(self) -> None:
        """Take the cube's current orientation as the goal -- a zero-error start."""
        _, cube_quaternion = self.backend.cube_pose()
        self._set_goal(cube_quaternion)

    def toggle_auto_goal(self) -> None:
        """Switch between a goal the visitor sets and the training schedule.

        The training goal spins away on every success and decays back toward
        the cube; a goal held still is that schedule's limit case.
        """
        self.auto_goal = not self.auto_goal

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def reset_statistics(self) -> None:
        self.steps = 0
        self.successes = 0
        self.drops = 0

    # -- rendering ----------------------------------------------------------

    def render_bgr(
        self,
        height: int = 480,
        width: int = 640,
        camera: str | int = DEFAULT_CAMERA,
    ) -> np.ndarray:
        """Render the current scene to a BGR image, as the showcase does."""
        import cv2

        if self._renderer is not None and (
            self._renderer.height != height or self._renderer.width != width
        ):
            # A renderer is fixed to the size it was built at, so a different
            # one asked for later came back silently at the old size.
            self._renderer.close()
            self._renderer = None

        if self._renderer is None:
            model = self.backend.model
            # The offscreen buffer is sized in the XML and defaults to 640x480;
            # asking the renderer for more than it without raising these is an
            # error, not a resize.
            model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
            model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
            self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._renderer.update_scene(self.backend.data, camera=camera)
        return cv2.cvtColor(self._renderer.render(), cv2.COLOR_RGB2BGR)

    def close(self) -> None:
        if self._renderer is not None:
            try:
                self._renderer.close()
            except Exception:
                pass
            self._renderer = None

    def __enter__(self) -> "CubeReorientController":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
