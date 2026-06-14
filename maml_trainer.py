# maml_trainer.py
"""
MAML (Model-Agnostic Meta-Learning) for RAPCG-MetaRL
Enables fast adaptation to new PCG tasks with few gradient steps.

Algorithm (Finn et al., 2017):
    1. Sample batch of tasks T_i from task distribution
    2. For each task T_i:
       a. Inner loop: theta'_i = theta - alpha * grad_theta L_{T_i}(theta)   (adapt)
       b. Compute loss L_{T_i}(theta'_i) on adapted params
    3. Outer loop: theta = theta - beta * grad_theta * Sum_i L_{T_i}(theta'_i)    (meta-update)
"""

import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import OrderedDict
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import argparse

# Project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "gym-pcgrl"))

from utils import ResourceMonitor, TrainingLogger, create_checkpoint_dir
from wrappers.pcgrl_env import make_pcgrl_env

import gym
from gym import spaces

try:
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    DummyVecEnv = None


class MAMLPolicy(nn.Module):
    """
    A simple MLP Policy (Actor-Critic) that supports functional forward passes
    by taking an external dictionary of parameters (weights/biases).
    """

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int = 64):
        super(MAMLPolicy, self).__init__()
        self.input_dim = input_dim
        self.action_dim = action_dim

        # Structure matching standard PyTorch / Stable-Baselines3 MLP layouts
        self.actor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
        )

        self.critic = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # Guard flag for printing key structure verification once
        self._keys_verified = False

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Standard forward pass using internal module weights."""
        action_logits = self.actor(obs)
        value = self.critic(obs)
        return action_logits, value

    def functional_forward(
        self, obs: torch.Tensor, params: dict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes forward pass via explicit structural grouping of keys.
        Extracts indices dynamically, sorting them to maintain proper execution order,
        and automatically skips the final layer projection's activation function.
        """
        # Run an engineering sanity check on the first pass to ensure checkpoint key format matching
        if not self._keys_verified:
            sample_keys = list(params.keys())
            print(f"\n[SANITY CHECK] Verifying state dictionary key formatting...")
            print(f"[SANITY CHECK] First 5 keys in parameter map: {sample_keys[:5]}")

            # Verify the delimiter structure splits safely into an expected layer integer
            try:
                test_key = [k for k in sample_keys if "actor." in k and ".weight" in k][
                    0
                ]
                parts = test_key.split(".")
                _ = int(parts[1])  # Ensure position [1] holds a valid layer digit
                print(
                    f"[SANITY CHECK] Success: Key format matches standard dot-notation index sequence ('{test_key}').\n"
                )
            except Exception as e:
                print(
                    f"[WARNING] Parameter dict key layout did not match expected 'actor.[index].weight' format."
                )
                print(
                    f"[WARNING] Found keys format: {sample_keys[:3]}. Exception details: {str(e)}"
                )

            self._keys_verified = True

        def _forward_network(x: torch.Tensor, prefix: str) -> torch.Tensor:
            # Step 1: Isolate and sort numeric structural layer positions
            layer_ids = sorted(
                list(
                    set(
                        int(k.split(".")[1])
                        for k in params.keys()
                        if k.startswith(prefix) and (".weight" in k or ".bias" in k)
                    )
                )
            )

            # Step 2: Iterate and process through the layers in structural order
            for idx, layer_id in enumerate(layer_ids):
                w = params[f"{prefix}.{layer_id}.weight"]
                b = params[f"{prefix}.{layer_id}.bias"]

                x = torch.matmul(x, w.t()) + b

                # Apply Tanh to intermediate layers only; leave output layer un-activated
                if idx < len(layer_ids) - 1:
                    x = torch.tanh(x)
            return x

        action_logits = _forward_network(obs, "actor")
        value = _forward_network(obs, "critic")
        return action_logits, value


class MAMLTrainer:
    def __init__(
        self,
        games: List[str],
        representations: List[str],
        meta_lr: float = 0.001,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        meta_batch: int = 4,
        iterations: int = 500,
        n_trajectories: int = 128,
        device: str = "auto",
        second_order: bool = False,
        experiment_name: Optional[str] = None,
    ):
        self.games = games
        self.representations = representations
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_batch = meta_batch
        self.iterations = iterations
        self.n_trajectories = n_trajectories
        self.second_order = second_order

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"MAML Framework Initialized on device: {self.device}")

        # Construct prototypical proxy environment to discover feature bounds
        self.proxy_env = make_pcgrl_env(games[0], representations[0])
        # Accommodate both standard and Dict spaces cleanly
        if isinstance(self.proxy_env.observation_space, spaces.Dict):
            self.obs_dim = np.prod(self.proxy_env.observation_space.spaces["map"].shape)
        else:
            self.obs_dim = np.prod(self.proxy_env.observation_space.shape)

        self.action_dim = self.proxy_env.action_space.n
        self.proxy_env.close()

        # Allocate Master Meta-Policy Policy parameters
        self.meta_policy = MAMLPolicy(self.obs_dim, self.action_dim).to(self.device)
        self.meta_optimizer = optim.Adam(self.meta_policy.parameters(), lr=meta_lr)

        # Telemetry & Experiment Management Structures
        self.resource_monitor = ResourceMonitor()
        exp_tag = (
            experiment_name
            if experiment_name
            else f"MAML_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        self.checkpoint_dir = create_checkpoint_dir(f"checkpoints/{exp_tag}")
        self.logger = TrainingLogger(log_dir="logs", experiment_name=exp_tag)

    def sample_trajectory(
        self, env, policy, params: dict
    ) -> List[Tuple[np.ndarray, int, float]]:
        """Collects trajectories using parameter weights mapped via functional evaluation passes."""
        trajectory = []
        obs = env.reset()
        if isinstance(obs, dict):
            obs = obs["map"]

        for _ in range(self.n_trajectories):
            obs_t = torch.FloatTensor(obs.flatten()).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits, _ = policy.functional_forward(obs_t, params)
                action_probs = torch.softmax(logits, dim=-1)
                action = torch.multinomial(action_probs, num_samples=1).item()

            next_obs, reward, done, info = env.step(action)
            trajectory.append((obs.flatten(), action, float(reward)))

            if isinstance(next_obs, dict):
                obs = next_obs["map"]
            else:
                obs = next_obs

            if done:
                obs = env.reset()
                if isinstance(obs, dict):
                    obs = obs["map"]
        return trajectory

    def compute_loss(
        self, policy, params: dict, trajectory: List[Tuple[np.ndarray, int, float]]
    ) -> torch.Tensor:
        """Computes basic reinforcing negative policy log-likelihoods over collected targets."""
        loss = torch.tensor(0.0, device=self.device)
        if not trajectory:
            return loss

        states, actions, rewards = zip(*trajectory)
        states_t = torch.FloatTensor(np.array(states)).to(self.device)
        actions_t = torch.LongTensor(actions).to(self.device)
        rewards_t = torch.FloatTensor(rewards).to(self.device)

        # Standardized return normalization baseline
        discounted_rewards = torch.zeros_like(rewards_t)
        running_add = 0.0
        for t in reversed(range(len(rewards_t))):
            running_add = running_add * 0.99 + rewards_t[t]
            discounted_rewards[t] = running_add
        if len(discounted_rewards) > 1:
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                discounted_rewards.std() + 1e-8
            )

        logits, _ = policy.functional_forward(states_t, params)
        log_probs = torch.log_softmax(logits, dim=-1)
        action_log_probs = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        loss = -(action_log_probs * discounted_rewards).mean()
        return loss

    def train(self):
        """Main Meta-Optimization loop execution block."""
        print(
            f"Beginning Meta-Learning routine across tasks ({self.games} x {self.representations})."
        )
        start_time = time.time()

        for iteration in range(self.iterations):
            self.meta_optimizer.zero_grad()
            meta_outer_loss = torch.tensor(0.0, device=self.device)

            # Pack current base parameters map values
            meta_weights = OrderedDict(self.meta_policy.state_dict())
            iteration_rewards = []

            for task_idx in range(self.meta_batch):
                game = np.random.choice(self.games)
                rep = np.random.choice(self.representations)

                env = make_pcgrl_env(game, rep)

                # Copy baseline structural weight graph to local fast-adaptation context
                task_weights = copy.deepcopy(meta_weights)

                # --- Step 1: Inner Adaptation Steps Loop (Fast Weight Adjustments) ---
                for step in range(self.inner_steps):
                    traj = self.sample_trajectory(env, self.meta_policy, task_weights)
                    if traj:
                        iteration_rewards.append(np.sum([t[2] for t in traj]))
                    inner_loss = self.compute_loss(self.meta_policy, task_weights, traj)

                    if inner_loss.item() != 0:
                        grads = torch.autograd.grad(
                            inner_loss,
                            task_weights.values(),
                            create_graph=self.second_order,
                            allow_unused=True,
                        )
                        # Perform explicit gradient step adjustments across our parameter maps
                        updated_weights = OrderedDict()
                        for (k, v), g in zip(task_weights.items(), grads):
                            if g is not None:
                                updated_weights[k] = v - self.inner_lr * g
                            else:
                                updated_weights[k] = v
                        task_weights = updated_weights

                # --- Step 2: Meta-Update Evaluation (Evaluate updated weights on a new trajectory) ---
                post_adapt_traj = self.sample_trajectory(
                    env, self.meta_policy, task_weights
                )
                outer_loss = self.compute_loss(
                    self.meta_policy, task_weights, post_adapt_traj
                )
                meta_outer_loss += outer_loss

                env.close()

            # --- Step 3: Global Meta-Update Parameter Modification ---
            meta_outer_loss = meta_outer_loss / self.meta_batch
            if meta_outer_loss.item() != 0:
                meta_outer_loss.backward()
                self.meta_optimizer.step()

            # Track real-time compute diagnostics to avoid crashing under hardware limits
            res_stats = self.resource_monitor.get_resources()
            mean_rewards = np.mean(iteration_rewards) if iteration_rewards else 0.0

            # Format logs to save tracking metrics
            log_metrics = {
                "step": iteration,
                "loss": meta_outer_loss.item(),
                "reward": mean_rewards,
                "cpu_percent": res_stats["cpu_percent"],
                "ram_percent": res_stats["ram_percent"],
                "gpu_percent": res_stats["gpu_percent"],
                "fps": float(iteration) / (time.time() - start_time + 1e-5),
            }
            self.logger.log_step(log_metrics)

            if iteration % 20 == 0 or iteration == self.iterations - 1:
                print(
                    f"Iteration {iteration:03d}/{self.iterations} | "
                    f"Outer Loss: {meta_outer_loss.item():.4f} | "
                    f"Avg Reward: {mean_rewards:.2f} | "
                    f"RAM: {res_stats['ram_percent']}% | "
                    f"GPU: {res_stats['gpu_percent']}%"
                )

                # Persist stable parameters checkpoints
                torch.save(
                    self.meta_policy.state_dict(),
                    os.path.join(self.checkpoint_dir, "best_meta_model.pt"),
                )

        print(
            f"Meta-Training completed. Weights saved inside folder: '{self.checkpoint_dir}'"
        )


if __name__ == "__main__":
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--games", nargs="+", default=["zelda", "sokoban"])
    parser.add_argument("--representations", nargs="+", default=["narrow", "turtle"])
    parser.add_argument("--meta-lr", type=float, default=0.001)
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--inner-steps", type=int, default=3)
    parser.add_argument("--meta-batch", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--n-trajectories", type=int, default=64)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--second-order", action="store_true")
    parser.add_argument("--experiment-name", type=str, default=None)

    args = parser.parse_args()

    trainer = MAMLTrainer(
        games=args.games,
        representations=args.representations,
        meta_lr=args.meta_lr,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_batch=args.meta_batch,
        iterations=args.iterations,
        n_trajectories=args.n_trajectories,
        device=args.device,
        second_order=args.second_order,
        experiment_name=args.experiment_name,
    )
    trainer.train()
