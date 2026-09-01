"""Cube reorientation policy: the pieces the booth needs to run it.

Vendored from the leap_cube_pose project (src/leap_cube_pose/deploy), which is
where these modules are developed and tested. They are copied rather than
imported so the booth stays runnable from this checkout alone -- it is the
machine that has to work at an exhibition, not a machine with two repos on its
PYTHONPATH. Only sim_backend differs from its source, in two ways: it reads the
Playground scene's assets off disk instead of importing mujoco_playground,
which would drag in jax and ml_collections for nothing, and it exposes model
and data as properties so the booth can render the scene and pose the goal.

Everything here is numpy-only apart from sim_backend, which needs mujoco.
"""

from .goal import GoalScheduler, ori_error
from .observation import ObservationBuilder
from .policy import NumpyPolicy
from .sim_backend import MujocoSimBackend

__all__ = [
    "GoalScheduler",
    "MujocoSimBackend",
    "NumpyPolicy",
    "ObservationBuilder",
    "ori_error",
]
