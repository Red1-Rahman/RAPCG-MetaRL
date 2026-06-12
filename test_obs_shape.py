import gym
import sys
import os

sys.path.insert(0, os.getcwd())

# Import root utils
from importlib.util import spec_from_file_location, module_from_spec

spec = spec_from_file_location("root_utils", os.path.join(os.getcwd(), "utils.py"))
root_utils = module_from_spec(spec)
spec.loader.exec_module(root_utils)

from wrappers.pcgrl_env import make_pcgrl_env
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

rm = root_utils.ResourceMonitor(use_gpu=True)

# Test 1: Single wrapped environment
print("=== Single wrapped environment ===")
single_env = make_pcgrl_env(
    resource_monitor=rm,
    game="sokoban",
    representation="narrow",
)
print(f"Single env observation_space: {single_env.observation_space}")
print(f"Single env observation_space.shape: {single_env.observation_space.shape}")

if isinstance(single_env.observation_space, gym.spaces.Dict):
    from maml_trainer import DictFlattenWrapper

    single_env = DictFlattenWrapper(single_env)
    print(f"After DictFlattenWrapper: {single_env.observation_space}")
    print(f"After DictFlattenWrapper shape: {single_env.observation_space.shape}")
    print(f"Product of shape: {np.prod(single_env.observation_space.shape)}")

obs = single_env.reset()
print(f"Single env reset observation shape: {obs.shape}")

single_env.close()

# Test 2: VectorizedEnv
print("\n=== Vectorized environment (DummyVecEnv) ===")


def make_env():
    env = make_pcgrl_env(
        resource_monitor=rm,
        game="sokoban",
        representation="narrow",
    )
    if isinstance(env.observation_space, gym.spaces.Dict):
        from maml_trainer import DictFlattenWrapper

        env = DictFlattenWrapper(env)
    return env


vec_env = DummyVecEnv([make_env])
print(f"DummyVecEnv observation_space.shape: {vec_env.observation_space.shape}")
print(f"Product of shape: {np.prod(vec_env.observation_space.shape)}")

obs = vec_env.reset()
print(f"Vectorized env reset observation shape: {obs.shape}")
vec_env.close()
