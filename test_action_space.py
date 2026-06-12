import gym
import sys
import os
sys.path.insert(0, os.getcwd())

# Import root utils
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("root_utils", os.path.join(os.getcwd(), "utils.py"))
root_utils = module_from_spec(spec)
spec.loader.exec_module(root_utils)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

from wrappers.pcgrl_env import make_pcgrl_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

rm = root_utils.ResourceMonitor(use_gpu=False)

# Test Sokoban narrow
print("=== Sokoban Narrow ===")
env = make_pcgrl_env(
    resource_monitor=rm,
    game='sokoban',
    representation='narrow',
)
print(f'Action space: {env.action_space}')
print(f'Action space type: {type(env.action_space)}')
if hasattr(env.action_space, 'n'):
    print(f'Action space n: {env.action_space.n}')
if hasattr(env.action_space, 'shape'):
    print(f'Action space shape: {env.action_space.shape}')
env.close()

# Test Sokoban wide
print("\n=== Sokoban Wide ===")
env = make_pcgrl_env(
    resource_monitor=rm,
    game='sokoban',
    representation='wide',
)
print(f'Action space: {env.action_space}')
print(f'Action space type: {type(env.action_space)}')
if hasattr(env.action_space, 'n'):
    print(f'Action space n: {env.action_space.n}')
if hasattr(env.action_space, 'shape'):
    print(f'Action space shape: {env.action_space.shape}')
env.close()

# Test Sokoban turtle
print("\n=== Sokoban Turtle ===")
env = make_pcgrl_env(
    resource_monitor=rm,
    game='sokoban',
    representation='turtle',
)
print(f'Action space: {env.action_space}')
print(f'Action space type: {type(env.action_space)}')
if hasattr(env.action_space, 'n'):
    print(f'Action space n: {env.action_space.n}')
if hasattr(env.action_space, 'shape'):
    print(f'Action space shape: {env.action_space.shape}')
env.close()
