# rlhf_trainer.py
"""
RLHF (Reinforcement Learning from Human Feedback) for RAPCG-MetaRL

Pipeline:
    Phase 1 — Generate levels with current policy
    Phase 2 — Collect human preferences (pairwise comparisons)
    Phase 3 — Train a Bradley-Terry reward model on preferences
    Phase 4 — Fine-tune the PCG agent with PPO against the learned reward
"""

import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import argparse
import time
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "gym-pcgrl"))

from utils import ResourceMonitor, TrainingLogger, create_checkpoint_dir
from wrappers.helper import calculate_content_metrics
from wrappers.pcgrl_env import make_pcgrl_env

import gym
from gym import spaces

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    PPO = None
    DummyVecEnv = None


class DictFlattenWrapper(gym.Wrapper):
    """Flatten a Dict observation space into a 1-D Box (old-gym compatible)."""
    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.observation_space, spaces.Dict):
            self.subspace = env.observation_space.spaces["map"]
        else:
            self.subspace = env.observation_space
        self.flat_dim = np.prod(self.subspace.shape)
        self.observation_space = spaces.Box(
            low=0, high=20, shape=(self.flat_dim,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._fl(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._fl(obs), reward, done, info

    def _fl(self, obs):
        if isinstance(obs, dict):
            return obs["map"].flatten().astype(np.float32)
        return obs.flatten().astype(np.float32)


class PreferenceRewardModel(nn.Module):
    """Standard Bradley-Terry MLP reward projection network architecture."""
    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super(PreferenceRewardModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, level_features: torch.Tensor) -> torch.Tensor:
        return self.net(level_features)


class RLHFRewardWrapper(gym.Wrapper):
    """
    Blends environmental rewards with empirical neural feedback calculations.
    Safely supports both raw environments and vectorized environments wrappers.
    """
    def __init__(self, env, reward_model: nn.Module, rlhf_weight: float = 0.4, input_dim: int = 225):
        super().__init__(env)
        self.reward_model = reward_model
        self.rlhf_weight = rlhf_weight
        self.input_dim = input_dim
        self.device = next(reward_model.parameters()).device

    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)

        is_vectorized = isinstance(info, (list, tuple))
        target_info = info[0] if is_vectorized else info

        level_flat = self._to_flat(obs, target_info)

        with torch.no_grad():
            t = torch.FloatTensor(level_flat).unsqueeze(0).to(self.device)
            human_reward = self.reward_model(t).item()

        blended = (1.0 - self.rlhf_weight) * env_reward + self.rlhf_weight * human_reward

        if is_vectorized:
            info[0]["env_reward"] = env_reward
            info[0]["human_reward"] = human_reward
            info[0]["blended_reward"] = blended
        else:
            info["env_reward"] = env_reward
            info["human_reward"] = human_reward
            info["blended_reward"] = blended

        return obs, blended, done, info

    def _to_flat(self, obs, info: dict) -> np.ndarray:
        if isinstance(info, dict) and "map" in info:
            arr = np.array(info["map"]).flatten()
        elif isinstance(obs, np.ndarray):
            arr = obs.flatten()
        else:
            arr = np.zeros(self.input_dim)

        if len(arr) >= self.input_dim:
            return arr[:self.input_dim].astype(np.float32)
            
        return np.pad(arr, (0, self.input_dim - len(arr)), mode='constant', constant_values=0).astype(np.float32)


class RLHFTrainer:
    def __init__(
        self,
        game: str = "zelda",
        representation: str = "narrow",
        base_model_path: Optional[str] = None,
        rlhf_weight: float = 0.4,
        reward_model_epochs: int = 50,
        ppo_timesteps: int = 20000,
        device: str = "auto",
        experiment_name: Optional[str] = None,
    ):
        self.game = game
        self.representation = representation
        self.base_model_path = base_model_path
        self.rlhf_weight = rlhf_weight
        self.reward_model_epochs = reward_model_epochs
        self.ppo_timesteps = ppo_timesteps

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.resource_monitor = ResourceMonitor()

        # Safely pass monitor token to detect dimensions
        test_env = make_pcgrl_env(self.resource_monitor, game, representation)
        if isinstance(test_env.observation_space, spaces.Dict):
            self.input_dim = np.prod(test_env.observation_space.spaces["map"].shape)
        else:
            self.input_dim = np.prod(test_env.observation_space.shape)
        test_env.close()

        self.reward_model = PreferenceRewardModel(self.input_dim).to(self.device)
        self.reward_optimizer = optim.Adam(self.reward_model.parameters(), lr=0.0005)

        self.exp_tag = experiment_name if experiment_name else f"RLHF_{game}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.checkpoint_dir = create_checkpoint_dir(f"checkpoints/{self.exp_tag}")
        self.logger = TrainingLogger(log_dir="logs", experiment_name=self.exp_tag)

    def generate_levels_for_feedback(self, n_levels: int = 10) -> List[np.ndarray]:
        print(f"Generating {n_levels} reference layouts for alignment pooling...")
        env = DictFlattenWrapper(make_pcgrl_env(self.resource_monitor, self.game, self.representation))
        levels = []
        
        for _ in range(n_levels):
            obs = env.reset()
            done = False
            step_count = 0
            while not done and step_count < 200:
                action = env.action_space.sample()
                obs, r, done, info = env.step(action)
                step_count += 1
            levels.append(obs.copy())
            
        env.close()
        return levels

    def collect_preferences(self, levels: List[np.ndarray], interactive: bool = False) -> List[Tuple[np.ndarray, np.ndarray, int]]:
        preferences = []
        n = len(levels)
        if n < 2:
            return preferences

        print(f"Compiling comparison maps pairs from dataset entries...")
        for i in range(0, n - 1, 2):
            lA = levels[i]
            lB = levels[i+1]
            
            if interactive:
                print(f"\n--- Layout A (Sum Elements: {np.sum(lA)}) vs Layout B (Sum Elements: {np.sum(lB)}) ---")
                choice = input("Select preferred topology profile (A/B): ").strip().lower()
                winner = 0 if choice == 'a' else 1
            else:
                winner = 0 if np.sum(lA) >= np.sum(lB) else 1
                
            preferences.append((lA, lB, winner))
        return preferences

    def train_reward_model(self, preferences: List[Tuple[np.ndarray, np.ndarray, int]]):
        if not preferences:
            print("[ERROR] Preferred data matrix contains zero items. Skipping optimization.")
            return

        print(f"Optimizing alignment utility parameters across model iterations...")
        self.reward_model.train()
        
        for epoch in range(self.reward_model_epochs):
            total_loss = 0.0
            for lA, lB, winner in preferences:
                self.reward_optimizer.zero_grad()
                
                tA = torch.FloatTensor(lA).unsqueeze(0).to(self.device)
                tB = torch.FloatTensor(lB).unsqueeze(0).to(self.device)
                
                rA = self.reward_model(tA)
                rB = self.reward_model(tB)
                
                if winner == 0:
                    loss = -torch.log(torch.sigmoid(rA - rB) + 1e-8)
                else:
                    loss = -torch.log(torch.sigmoid(rB - rA) + 1e-8)
                    
                loss.backward()
                self.reward_optimizer.step()
                total_loss += loss.item()

            if epoch % 10 == 0 or epoch == self.reward_model_epochs - 1:
                print(f" Reward Training Epoch {epoch:02d}/{self.reward_model_epochs} | Cumulative Loss: {total_loss:.4f}")

        model_out = os.path.join(self.checkpoint_dir, "learned_reward_model.pt")
        torch.save(self.reward_model.state_dict(), model_out)
        print(f"Alignment optimization routine completed. Weights saved to: {model_out}")

    def fine_tune_agent(self):
        """Builds environment wrappers and explicitly processes the Stable-Baselines3 PPO fine-tuning loop."""
        if PPO is None or DummyVecEnv is None:
            print("[ERROR] Stable-Baselines3 is missing. Cannot execute fine-tuning loop.")
            return

        print("Hooking alignment reward wrappers into environment factory pipeline...")
        
        def make_env():
            raw_env = make_pcgrl_env(self.resource_monitor, self.game, self.representation)
            wrapped_reward = RLHFRewardWrapper(
                raw_env, 
                self.reward_model, 
                rlhf_weight=self.rlhf_weight, 
                input_dim=self.input_dim
            )
            return DictFlattenWrapper(wrapped_reward)

        env = DummyVecEnv([make_env])
        
        print("Initializing Stable-Baselines3 PPO policy model network...")
        if self.base_model_path and os.path.exists(self.base_model_path):
            print(f"Loading pretrained weights checkpoint asset from: {self.base_model_path}")
            model = PPO.load(self.base_model_path, env=env, device=args.device)
        else:
            print("No pretrained checkpoint asset provided. Initializing baseline policy network architecture.")
            model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0003, device=args.device)

        print(f"Beginning PPO reinforcement tracking execution path across {self.ppo_timesteps} steps...")
        model.learn(total_timesteps=self.ppo_timesteps)
        
        # Save structural checkpoint to disk
        policy_out = os.path.join(self.checkpoint_dir, "final_rlhf_ppo_agent.zip")
        model.save(policy_out)
        print(f"[SUCCESS] Agent optimization routine complete. Training checkpoint saved to: {policy_out}")
        
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", default="zelda")
    parser.add_argument("--representation", default="narrow")
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--rlhf-weight", type=float, default=0.4)
    parser.add_argument("--n-levels", type=int, default=10)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--timesteps", type=int, default=2000)
    parser.add_argument("--reward-epochs", type=int, default=20)
    parser.add_argument("--reward-model-only", action="store_true")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--experiment-name", type=str, default=None)

    args = parser.parse_args()

    trainer = RLHFTrainer(
        game=args.game,
        representation=args.representation,
        base_model_path=args.base_model,
        rlhf_weight=args.rlhf_weight,
        reward_model_epochs=args.reward_epochs,
        ppo_timesteps=args.timesteps,
        device=args.device,
        experiment_name=args.experiment_name
    )

    ref_levels = trainer.generate_levels_for_feedback(args.n_levels)
    prefs = trainer.collect_preferences(ref_levels, interactive=args.interactive)
    trainer.train_reward_model(prefs)
    
    if not args.reward_model_only:
        trainer.fine_tune_agent()