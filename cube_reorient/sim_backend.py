"""CPU MuJoCo implementation of the deployment hardware interface."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np

from .mapping import PLAYGROUND_JOINT_NAMES


DEFAULT_SIM_DT = 0.01
DEFAULT_CTRL_DT = 0.05
DEFAULT_DROP_HEIGHT = -0.05  # matches reorient.py: fall when cube_position[2] < -0.05


DEFAULT_PLAYGROUND_ROOT = Path.home() / "Projects" / "mujoco_playground"

# The scene and the meshes it needs, copied into this repo so the booth runs
# from a clone alone. It is 8 MB: the Playground checkout's own leap_hand
# assets minus the left-hand meshes, which this scene never references.
BUNDLED_SCENE_DIR = Path(__file__).resolve().parent.parent / "assets" / "reorient_scene"


def _bundled_assets() -> tuple[Path, dict[str, bytes]] | None:
    """The scene XML and its assets from this repo, if they were shipped."""
    scene = BUNDLED_SCENE_DIR / "scene_mjx_cube.xml"
    if not scene.is_file():
        return None
    assets = {
        path.name: path.read_bytes()
        for path in BUNDLED_SCENE_DIR.iterdir()
        if path.is_file()
    }
    return scene, assets


def _playground_package_dir(playground_root: str | Path | None = None) -> Path:
    """Locate the mujoco_playground package, by import or else on disk.

    The booth venv deliberately does not have mujoco_playground installed: it
    pulls in jax and ml_collections, and nothing on this path needs either. So
    an installed package is the first thing tried here, not the only one.
    """
    spec = importlib.util.find_spec("mujoco_playground")
    locations = None if spec is None else spec.submodule_search_locations
    if locations:
        return Path(next(iter(locations)))

    root = DEFAULT_PLAYGROUND_ROOT if playground_root is None else Path(playground_root)
    package_dir = root / "mujoco_playground"
    if package_dir.is_dir():
        return package_dir
    raise ModuleNotFoundError(
        "mujoco_playground is neither importable nor checked out at "
        f"{package_dir}; pass playground_root to say where it lives"
    )


def _default_model_path(playground_root: str | Path | None = None) -> Path:
    model_path = (
        _playground_package_dir(playground_root)
        / "_src"
        / "manipulation"
        / "leap_hand"
        / "xmls"
        / "scene_mjx_cube.xml"
    )
    if not model_path.is_file():
        raise FileNotFoundError(
            f"MuJoCo Playground LEAP cube model was not found at {model_path}"
        )
    return model_path


def _collect_assets(package_dir: Path) -> dict[str, bytes]:
    """Every file the scene XML may ask for, keyed by basename.

    Playground's own get_assets() builds the same dict, but importing it costs
    jax. The model built from this one is the model training used: it is the
    same XML, and MuJoCo resolves assets by basename alone. Reading them here
    also sidesteps the scene's relative path to the menagerie meshes, which
    assumes a checkout layout that from_xml_path cannot resolve.
    """
    xml_dir = package_dir / "_src" / "manipulation" / "leap_hand" / "xmls"
    assets: dict[str, bytes] = {}

    def add(directory: Path, pattern: str = "*") -> None:
        if directory.is_dir():
            for path in directory.glob(pattern):
                if path.is_file():
                    assets[path.name] = path.read_bytes()

    add(package_dir / "external_deps" / "mujoco_menagerie" / "leap_hand" / "assets")
    add(xml_dir, "*.xml")
    add(xml_dir / "reorientation_cube_textures")
    add(xml_dir / "meshes")
    return assets


class MujocoSimBackend:
    """Run the MuJoCo Playground LEAP cube scene using CPU MuJoCo."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        playground_root: str | Path | None = None,
        sim_dt: float = DEFAULT_SIM_DT,
        ctrl_dt: float = DEFAULT_CTRL_DT,
        drop_height: float = DEFAULT_DROP_HEIGHT,
    ) -> None:
        if sim_dt <= 0.0 or ctrl_dt <= 0.0:
            raise ValueError("sim_dt and ctrl_dt must be positive")
        step_ratio = ctrl_dt / sim_dt
        steps_per_control = round(step_ratio)
        if steps_per_control < 1 or not np.isclose(step_ratio, steps_per_control):
            raise ValueError("ctrl_dt must be an integer multiple of sim_dt")

        try:
            import mujoco
        except ImportError as exc:
            raise ModuleNotFoundError(
                "mujoco is required to construct MujocoSimBackend"
            ) from exc

        bundled = _bundled_assets() if model_path is None else None
        if bundled is not None:
            path, assets = bundled
        else:
            path = (
                _default_model_path(playground_root)
                if model_path is None
                else Path(model_path)
            )
            if not path.is_file():
                raise FileNotFoundError(f"MuJoCo model does not exist: {path}")
            assets = _collect_assets(_playground_package_dir(playground_root))

        self._mujoco: Any = mujoco
        # The scene XML reaches its meshes through relative paths that assume a
        # layout neither checkout has, so from_xml_path cannot resolve them.
        # Playground loads the model from a string with an explicit asset dict
        # keyed by basename; both routes above build the same dict, so the model
        # is the trained one without importing jax to get it.
        self._model = mujoco.MjModel.from_xml_string(path.read_text(), assets=assets)
        self._model.opt.timestep = float(sim_dt)
        self._data = mujoco.MjData(self._model)
        self._dt = float(ctrl_dt)
        self._steps_per_control = int(steps_per_control)
        self._drop_height = float(drop_height)
        self._rng = np.random.default_rng()

        if self._model.nu < 16:
            raise ValueError(f"expected at least 16 actuators, model has {self._model.nu}")
        self._joint_qpos_indices = np.asarray(
            [self._joint_qpos_address(name) for name in PLAYGROUND_JOINT_NAMES],
            dtype=np.int64,
        )
        self._cube_position_slice = self._sensor_slice("cube_position", 3)
        self._cube_orientation_slice = self._sensor_slice("cube_orientation", 4)
        self._palm_position_slice = self._sensor_slice("palm_position", 3)
        self._home_key = self._name_id(mujoco.mjtObj.mjOBJ_KEY, "home")
        self._cube_joint = self._name_id(
            mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint"
        )
        self.reset()

    def _name_id(self, object_type: Any, name: str) -> int:
        object_id = int(self._mujoco.mj_name2id(self._model, object_type, name))
        if object_id < 0:
            raise ValueError(f"MuJoCo model has no {object_type!s} named {name!r}")
        return object_id

    def _joint_qpos_address(self, name: str) -> int:
        joint_id = self._name_id(self._mujoco.mjtObj.mjOBJ_JOINT, name)
        return int(self._model.jnt_qposadr[joint_id])

    def _sensor_slice(self, name: str, expected_dim: int) -> slice:
        sensor_id = self._name_id(self._mujoco.mjtObj.mjOBJ_SENSOR, name)
        dimension = int(self._model.sensor_dim[sensor_id])
        if dimension != expected_dim:
            raise ValueError(
                f"sensor {name!r} has dimension {dimension}, expected {expected_dim}"
            )
        address = int(self._model.sensor_adr[sensor_id])
        return slice(address, address + dimension)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def model(self) -> Any:
        """The loaded MjModel, for rendering and for posing the goal mocap."""
        return self._model

    @property
    def data(self) -> Any:
        """The live MjData. Read it; step it only through this class."""
        return self._data

    def reset(self, seed: int | None = None) -> None:
        """Restore the named ``home`` keyframe deterministically."""
        self._rng = np.random.default_rng(seed)
        self._mujoco.mj_resetDataKeyframe(self._model, self._data, self._home_key)
        self._mujoco.mj_forward(self._model, self._data)

    def read_joint_angles(self) -> np.ndarray:
        return self._data.qpos[self._joint_qpos_indices].copy()

    def write_motor_targets(self, targets: np.ndarray) -> None:
        values = np.asarray(targets, dtype=np.float64)
        if values.shape != (16,):
            raise ValueError(f"expected targets shape (16,), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError("targets contain NaN or infinity")
        self._data.ctrl[:16] = values
        for _ in range(self._steps_per_control):
            self._mujoco.mj_step(self._model, self._data)

    def hold(self) -> None:
        """Advance one control period while retaining the existing controls."""
        for _ in range(self._steps_per_control):
            self._mujoco.mj_step(self._model, self._data)

    def emergency_stop(self) -> None:
        """Freeze targets at the currently measured joint positions."""
        self._data.ctrl[:16] = self.read_joint_angles()

    def close(self) -> None:
        """MuJoCo's Python-owned model and data need no explicit teardown."""

    def cube_pose(self) -> tuple[np.ndarray, np.ndarray]:
        position = self._data.sensordata[self._cube_position_slice].copy()
        quaternion = self._data.sensordata[self._cube_orientation_slice].copy()
        return position, quaternion

    def palm_position(self) -> np.ndarray:
        return self._data.sensordata[self._palm_position_slice].copy()

    def cube_dropped(self) -> bool:
        return bool(self.cube_pose()[0][2] < self._drop_height)


__all__ = [
    "DEFAULT_CTRL_DT",
    "DEFAULT_DROP_HEIGHT",
    "DEFAULT_SIM_DT",
    "MujocoSimBackend",
]
