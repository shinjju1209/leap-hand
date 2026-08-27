# -*- coding: utf-8 -*-
"""Google Colab 1-Click Training Script for LEAP Hand Cube Reorient in MuJoCo Playground (JAX/MJX)

Run this in Google Colab (Runtime -> Change runtime type -> GPU: T4 / A100 / L4).
Training completes in ~20 to 30 minutes on GPU.
"""

# Cell 1: Install Dependencies
# !pip install -q git+https://github.com/google-deepmind/mujoco_playground.git
# !pip install -q brax flax orbax-checkpoint torch

import os
import time
import functools
import numpy as np
import jax
import jax.numpy as jnp
import flax

print(f"JAX Devices: {jax.devices()}")
print(f"JAX Default Backend: {jax.default_backend()}")

# Cell 2: Import MuJoCo Playground & Brax
from mujoco_playground import registry, wrapper
from mujoco_playground.config import manipulation_params
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks

# Cell 3: Load LeapCubeReorient Environment Config & RL Hyperparameters
env_name = "LeapCubeReorient"
env_cfg = registry.get_default_config(env_name)
env = registry.load(env_name, config=env_cfg)
print(f"Loaded environment: {env_name}")

# Get official tuned PPO training hyperparameters and prepare network_factory
rl_config = manipulation_params.brax_ppo_config(env_name)
ppo_training_params = dict(rl_config)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in rl_config:
    del ppo_training_params["network_factory"]
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks, **rl_config.network_factory
    )
print("Loaded official PPO RL hyperparameters and configured network factory.")

# Cell 4: Train Policy
t0 = time.time()

def progress_fn(num_steps, metrics):
    elapsed = time.time() - t0
    reward = metrics.get('eval/episode_reward', metrics.get('training/reward', 0.0))
    print(f"Steps: {num_steps:10d} | Elapsed: {elapsed/60:.1f}m | Eval Reward: {reward:.2f}")

print("Starting JAX/MJX PPO training on GPU...")
make_inference_fn, params, metrics = ppo.train(
    environment=env,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    progress_fn=progress_fn,
    network_factory=network_factory,
    **ppo_training_params,
)

print(f"\nTraining completed in {(time.time() - t0)/60:.2f} minutes!")

# Cell 5: Export Policy Weights to NumPy / PyTorch Portable Format
def extract_weights(flax_params):
    """Recursively extract JAX/Flax parameter PyTree to plain numpy dict."""
    flat = flax.traverse_util.flatten_dict(flax_params, sep="/")
    return {k: np.array(v) for k, v in flat.items()}

numpy_weights = extract_weights(params[0] if isinstance(params, (tuple, list)) else params)

# Save as .npz
npz_path = "leap_cube_reorient_jax.npz"
np.savez(npz_path, **numpy_weights)
print(f"Saved NumPy weights to: {npz_path} (Size: {os.path.getsize(npz_path)/1024:.1f} KB)")

# Save as PyTorch state_dict (.pt)
try:
    import torch
    torch_weights = {k: torch.from_numpy(v) for k, v in numpy_weights.items()}
    pt_path = "leap_cube_reorient_jax.pt"
    torch.save({"weights": torch_weights, "env": env_name, "framework": "mujoco_playground"}, pt_path)
    print(f"Saved PyTorch weights to: {pt_path}")
except ImportError:
    pass

# Cell 6: Download files in Colab
try:
    from google.colab import files
    print("Triggering browser download...")
    files.download(npz_path)
    if os.path.exists("leap_cube_reorient_jax.pt"):
        files.download("leap_cube_reorient_jax.pt")
except Exception as e:
    print(f"Manual download: look for {npz_path} in the Colab file manager on the left panel.")
