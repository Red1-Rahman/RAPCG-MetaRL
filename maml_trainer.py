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

Patches applied vs original:
    [P1] functional_forward: replaced fragile string-index iteration with
         _forward_network() that sorts layer IDs numerically from param keys.
    [P2] MAMLTrainer.train(): meta_weights now cloned from named_parameters()
         with requires_grad_(True) so inner-loop gradients are never None.
    [P3] ResourceMonitor key: gpu_percent -> gpu_mem_percent (matches utils.py).
    [P4] TrainingLogger.log_step(): called with correct positional signature.
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

# Project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "gym-pcgrl"))

from utils import ResourceMonitor, TrainingLogger, create_checkpoint_dir
from wrappers.pcgrl_env import make_pcgrl_env

import gym

try:
    from sokoban_utils import check_sokoban_deadlock, compute_dead_squares

    _SOKOBAN_UTILS_AVAILABLE = True
except ImportError:
    _SOKOBAN_UTILS_AVAILABLE = False

    def check_sokoban_deadlock(level, crate_pos, dead_squares=None):  # type: ignore[misc]
        """Fallback: corner-only check when sokoban_utils is unavailable."""
        y, x = crate_pos
        h, w = level.shape
        is_wall_up    = (y == 0 or level[y - 1, x] == 1)
        is_wall_down  = (y == h - 1 or level[y + 1, x] == 1)
        is_wall_left  = (x == 0 or level[y, x - 1] == 1)
        is_wall_right = (x == w - 1 or level[y, x + 1] == 1)
        return (
            (is_wall_up and is_wall_left)
            or (is_wall_up and is_wall_right)
            or (is_wall_down and is_wall_left)
            or (is_wall_down and is_wall_right)
        )

    def compute_dead_squares(level, target_positions):  # type: ignore[misc]
        return set()

from gym import spaces

try:
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print(
        "Error: stable-baselines3 not installed. Install with: pip install stable-baselines3"
    )
    sys.exit(1)


class DictFlattenWrapper(gym.Wrapper):
    """Flatten a Dict observation space into a 1-D Box (old-gym compatible)."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, gym.spaces.Dict)
        self._keys = sorted(env.observation_space.spaces.keys())
        low_parts, high_parts = [], []
        for k in self._keys:
            sp = env.observation_space.spaces[k]
            low_parts.append(np.asarray(sp.low).flatten())
            high_parts.append(np.asarray(sp.high).flatten())
        low = np.concatenate(low_parts).astype(np.float32)
        high = np.concatenate(high_parts).astype(np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _flatten(self, obs):
        parts = [np.asarray(obs[k], dtype=np.float32).flatten() for k in self._keys]
        return np.concatenate(parts)

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self._flatten(obs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self._flatten(obs), reward, done, info


class SokobanDeadlockGuardrail(gym.Wrapper):
    """
    Per-step deadlock penalty for Sokoban box placements.

    Scans the live tile map on every env step and subtracts `deadlock_penalty`
    for every crate that occupies a geometrically un-pushable position:
      - Corner deadlock  : two orthogonal wall/boundary edges adjacent to the crate
      - 3-wall pocket    : three or more adjacent walls (via check_sokoban_deadlock)

    The penalty is intentionally dense (applied every step, not just at episode end)
    so that gradients point *away* from bad geometry even when the A*/BFS solver
    returns a path length of 0 (no gradient signal from the sparse solvability reward).

    A crate sitting exactly on a target tile is never penalised — that is the
    solved configuration.

    NOTE: Inherits from gym.Wrapper (not gym.RewardWrapper) and overrides step()
    directly with the old 4-tuple API to stay compatible with gym-pcgrl's env layer,
    which pre-dates the Gymnasium 5-tuple (obs, reward, terminated, truncated, info).

    Injection point: inside TaskDistribution.create_env()._make(), after
    ResourceAwarePCGRLWrapper and before DictFlattenWrapper, for Sokoban tasks only.
    """

    def __init__(self, env, deadlock_penalty: float = 1.5):
        """
        Args:
            env: Wrapped gym environment (after ResourceAwarePCGRLWrapper).
            deadlock_penalty: Penalty magnitude subtracted per deadlocked crate
                              per step.  Default 1.5 is intentionally larger than
                              a typical tile-count dense reward step to create a
                              clear gradient direction.
        """
        super().__init__(env)
        self._deadlock_penalty = deadlock_penalty

    def _get_grid(self) -> np.ndarray:
        """Walk the wrapper chain to reach the innermost env and read its tile map."""
        inner = self.env
        while hasattr(inner, "env"):
            inner = inner.env
        # gym-pcgrl stores the current board in _rep._map
        if hasattr(inner, "_rep") and hasattr(inner._rep, "_map"):
            return np.array(inner._rep._map, dtype=int)
        return None

    def _deadlock_penalty_for_grid(self, grid) -> float:
        """Compute total deadlock penalty for all crates in the current grid."""
        if grid is None:
            return 0.0

        h, w = grid.shape
        penalty = 0.0

        # Precompute dead squares (reverse-BFS from all targets) for richer detection.
        # Falls back to an empty set when no targets are placed yet.
        target_positions = (
            [(int(y), int(x)) for y, x in zip(*np.where(grid == 4))]
            if np.any(grid == 4)
            else []
        )
        dead_squares = (
            compute_dead_squares(grid, target_positions)
            if target_positions
            else set()
        )

        for y in range(h):
            for x in range(w):
                # Tile 3 == crate not yet on a target.
                # (gym-pcgrl may use tile 5 for crate-on-target; tile 3 is safe.)
                if grid[y, x] == 3:
                    if check_sokoban_deadlock(grid, (y, x), dead_squares):
                        penalty -= self._deadlock_penalty

        return penalty

    def step(self, action):
        """Old 4-tuple step, compatible with gym-pcgrl's pre-Gymnasium API."""
        obs, reward, done, info = self.env.step(action)
        grid = self._get_grid()
        shaped_reward = reward + self._deadlock_penalty_for_grid(grid)
        return obs, shaped_reward, done, info


# ===========================================================================
# Task Distribution
# ===========================================================================


class TaskDistribution:
    """
    Generates diverse PCG tasks for meta-learning.
    Each task is a unique (game, representation, reward_weights) combination
    that acts as a separate MDP for MAML's inner loop.
    """

    def __init__(self, games: List[str] = None, representations: List[str] = None):
        self.games = games or ["zelda", "sokoban", "binary"]
        self.representations = representations or ["narrow", "wide", "turtle"]

        # Task-specific reward weight variations for diversity
        self._reward_variations = {
            "zelda": [
                {"path-length": 2.0, "regions": 1.0, "nearest-enemy": 1.0},
                {"path-length": 1.0, "regions": 2.0, "nearest-enemy": 1.0},
                {"path-length": 1.0, "regions": 1.0, "nearest-enemy": 3.0},
            ],
            "sokoban": [
                {"dist-win": 2.0, "sol-length": 1.0, "ratio": 1.0},
                {"dist-win": 1.0, "sol-length": 2.0, "ratio": 1.0},
                {"dist-win": 3.0, "sol-length": 1.0, "ratio": 2.0},
            ],
            "binary": [
                {"regions": 5.0, "path-length": 1.0},
                {"regions": 1.0, "path-length": 5.0},
                {"regions": 3.0, "path-length": 3.0},
            ],
        }

    def sample_tasks(
        self, n_tasks: int, fixed_game: Optional[str] = None
    ) -> List[Dict]:
        """
        Sample n_tasks from the task distribution.

        Args:
            n_tasks: Number of tasks to sample
            fixed_game: If set, only sample tasks for this game

        Returns:
            List of task configuration dicts
        """
        tasks = []
        for _ in range(n_tasks):
            game = fixed_game or np.random.choice(self.games)

            # Filter representations by game (wide action space incompatible with current policy)
            valid_reps = self.representations
            if game == "sokoban":
                valid_reps = [r for r in self.representations if r != "wide"]

            representation = np.random.choice(valid_reps)

            # Sample reward weight variation
            variations = self._reward_variations.get(game, [{}])
            rewards = dict(variations[np.random.randint(len(variations))])

            # Add noise for continuous task distribution
            for key in rewards:
                rewards[key] *= np.random.uniform(0.8, 1.2)

            # Vary change_percentage for difficulty diversity
            change_percentage = np.random.uniform(0.2, 0.8)

            tasks.append(
                {
                    "game": game,
                    "representation": representation,
                    "reward_weights": rewards,
                    "change_percentage": change_percentage,
                }
            )
        return tasks

    def create_env(self, task: Dict, resource_monitor: ResourceMonitor) -> DummyVecEnv:
        """Create a vectorized environment for a specific task."""

        def _make():
            env = make_pcgrl_env(
                resource_monitor=resource_monitor,
                game=task["game"],
                representation=task["representation"],
                ram_penalty_weight=0.2,
                cpu_penalty_weight=0.1,
                gpu_penalty_weight=0.1,
            )
            # Apply task-specific reward weights
            if task["reward_weights"] and hasattr(env, "unwrapped"):
                try:
                    if hasattr(env.unwrapped, "_prob"):
                        env.unwrapped._prob.adjust_param(
                            change_percentage=task["change_percentage"],
                            rewards=task["reward_weights"],
                        )
                except Exception:
                    pass  # Some envs may not support all params
            # Strategy A guardrail: penalise deadlocked crate positions per-step so
            # the inner-loop gradient has a dense signal pointing away from corner
            # placements, breaking the count-matching local minimum.
            if task["game"] == "sokoban":
                env = SokobanDeadlockGuardrail(env, deadlock_penalty=1.5)

            # Flatten Dict observation space to a 1-D vector
            if isinstance(env.observation_space, gym.spaces.Dict):
                env = DictFlattenWrapper(env)
            return env

        return DummyVecEnv([_make])


# ===========================================================================
# MAML Policy Network
# ===========================================================================


class MAMLPolicy(nn.Module):
    """
    Policy network compatible with MAML's inner-loop gradient updates.
    Uses a simple actor-critic MLP that supports functional forward passes
    with arbitrary parameter dicts (required for differentiable inner loop).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int] = None):
        super().__init__()
        hidden_sizes = hidden_sizes or [64, 64]

        # Build actor (policy) network
        actor_layers = []
        prev = obs_dim
        for h in hidden_sizes:
            actor_layers.append(nn.Linear(prev, h))
            actor_layers.append(nn.Tanh())
            prev = h
        actor_layers.append(nn.Linear(prev, action_dim))
        self.actor = nn.Sequential(*actor_layers)

        # Build critic (value) network
        critic_layers = []
        prev = obs_dim
        for h in hidden_sizes:
            critic_layers.append(nn.Linear(prev, h))
            critic_layers.append(nn.Tanh())
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

        # Guard flag — print key sanity check once per session
        self._keys_verified = False

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning action logits and value estimate."""
        return self.actor(obs), self.critic(obs)

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> int:
        """Sample an action from the policy."""
        with torch.no_grad():
            logits, _ = self.forward(obs)
            if deterministic:
                return logits.argmax(dim=-1).item()
            probs = torch.softmax(logits, dim=-1)
            return torch.distributions.Categorical(probs).sample().item()

    def functional_forward(
        self, obs: torch.Tensor, params: OrderedDict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        [PATCH P1] Forward pass using external parameters (for MAML inner loop).

        Replaces the original fragile per-key string iteration with a robust
        _forward_network() helper that:
          1. Scans params once to extract numeric layer IDs per prefix.
          2. Iterates layers in sorted structural order (0, 2, 4 ... for nn.Sequential
             with interleaved Tanh modules).
          3. Applies Tanh to all hidden layers; skips activation on the final projection.

        This is independent of OrderedDict iteration order and handles any number
        of hidden layers automatically.
        """
        # One-time sanity check to confirm key format matches expectations
        if not self._keys_verified:
            sample_keys = list(params.keys())
            print(f"\n[MAML SANITY CHECK] First 5 param keys: {sample_keys[:5]}")
            try:
                test_key = next(
                    k for k in sample_keys if "actor." in k and ".weight" in k
                )
                _ = int(test_key.split(".")[1])
                print(f"[MAML SANITY CHECK] Key format OK: '{test_key}'\n")
            except Exception as e:
                print(
                    f"[MAML WARNING] Unexpected key format: {sample_keys[:3]} — {e}\n"
                )
            self._keys_verified = True

        def _forward_network(x: torch.Tensor, prefix: str) -> torch.Tensor:
            # Extract and sort numeric layer indices from param keys for this prefix.
            # nn.Sequential with Linear+Tanh pairs produces keys like:
            #   actor.0.weight, actor.0.bias  (Linear)
            #   actor.2.weight, actor.2.bias  (Linear, Tanh is index 1 but has no params)
            #   actor.4.weight, actor.4.bias  (final Linear)
            layer_ids = sorted(
                set(
                    int(k.split(".")[1])
                    for k in params.keys()
                    if k.startswith(prefix + ".") and (".weight" in k or ".bias" in k)
                )
            )

            for idx, layer_id in enumerate(layer_ids):
                w = params[f"{prefix}.{layer_id}.weight"]
                b = params[f"{prefix}.{layer_id}.bias"]
                x = torch.matmul(x, w.t()) + b
                # Apply Tanh to all hidden layers; skip on final output projection
                if idx < len(layer_ids) - 1:
                    x = torch.tanh(x)
            return x

        action_logits = _forward_network(obs, "actor")
        value = _forward_network(obs, "critic")
        return action_logits, value


# ===========================================================================
# Trajectory Collection & Policy Loss
# ===========================================================================


def collect_trajectories(
    env,
    policy: MAMLPolicy,
    n_steps: int = 128,
    device: str = "cpu",
    params: Optional[OrderedDict] = None,
) -> Dict:
    """
    Collect trajectory rollouts from environment.

    Args:
        env: Vectorized environment (DummyVecEnv)
        policy: MAMLPolicy instance
        n_steps: Number of environment steps to collect
        device: Torch device
        params: Optional external params for functional forward

    Returns:
        Dictionary with stacked tensors for observations, actions,
        rewards, values, log_probs, dones.
    """
    observations, actions, rewards = [], [], []
    values, log_probs, dones = [], [], []

    obs = env.reset()

    for step_idx in range(n_steps):
        obs_t = torch.FloatTensor(obs).to(device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        obs_flat = obs_t.reshape(obs_t.shape[0], -1)

        # Debug: Check observation shape on first iteration
        if step_idx == 0 and n_steps > 0 and params is not None:
            for name, p in params.items():
                if "actor" in name and "weight" in name:
                    expected_dim = p.shape[-1]
                    if obs_flat.shape[-1] != expected_dim:
                        print(f"[WARNING] Observation dimension mismatch!")
                        print(f"  Observation shape: {obs_flat.shape}")
                        print(f"  Expected input dim: {expected_dim}")
                        if obs_flat.shape[-1] < expected_dim:
                            padding = expected_dim - obs_flat.shape[-1]
                            obs_flat = torch.cat(
                                [
                                    obs_flat,
                                    torch.zeros(
                                        obs_flat.shape[0], padding, device=device
                                    ),
                                ],
                                dim=-1,
                            )
                            print(f"  -> Padded observation to {obs_flat.shape}")
                    break

        with torch.no_grad():
            if params is not None:
                logits, val = policy.functional_forward(obs_flat, params)
            else:
                logits, val = policy(obs_flat)

            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            act = dist.sample()
            lp = dist.log_prob(act)

        next_obs, rew, done, info = env.step(act.cpu().numpy())

        observations.append(obs_flat)
        actions.append(act)
        rewards.append(torch.FloatTensor(rew).to(device))
        values.append(val.squeeze(-1))
        log_probs.append(lp)
        dones.append(torch.FloatTensor(done.astype(float)).to(device))

        obs = next_obs

    return {
        "observations": torch.stack(observations),
        "actions": torch.stack(actions),
        "rewards": torch.stack(rewards),
        "values": torch.stack(values),
        "log_probs": torch.stack(log_probs),
        "dones": torch.stack(dones),
    }


def compute_policy_loss(
    trajectory: Dict,
    policy: MAMLPolicy,
    params: Optional[OrderedDict] = None,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> torch.Tensor:
    """
    Compute policy gradient loss with Generalized Advantage Estimation (GAE).

    Args:
        trajectory: Rollout data from collect_trajectories
        policy: MAMLPolicy instance
        params: Optional external params for functional forward
        gamma: Discount factor
        gae_lambda: GAE lambda

    Returns:
        Scalar loss tensor (differentiable)
    """
    rewards = trajectory["rewards"]
    values = trajectory["values"]
    dones = trajectory["dones"]
    observations = trajectory["observations"]
    actions = trajectory["actions"]

    T = len(rewards)

    # --- GAE advantages ---
    advantages = torch.zeros(T, device=rewards.device)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_val = values[t + 1].detach() if t < T - 1 else 0.0
        delta = rewards[t] + gamma * next_val * (1 - dones[t]) - values[t].detach()
        advantages[t] = last_gae = (
            delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        )

    # --- Recompute log-probs with gradient tracking ---
    obs_all = observations.squeeze(1)
    act_all = actions.squeeze(1) if actions.dim() > 1 else actions

    if params is not None:
        logits, new_values = policy.functional_forward(obs_all, params)
    else:
        logits, new_values = policy(obs_all)

    probs = torch.softmax(logits, dim=-1)
    dist = torch.distributions.Categorical(probs)
    new_lp = dist.log_prob(act_all)

    # Policy loss (REINFORCE with baseline)
    policy_loss = -(new_lp * advantages.detach()).mean()

    # Value loss
    returns = advantages + values.detach()
    value_loss = ((new_values.squeeze(-1) - returns) ** 2).mean()

    # Entropy bonus for exploration
    entropy = dist.entropy().mean()

    return policy_loss + 0.5 * value_loss - 0.01 * entropy


# ===========================================================================
# MAML Trainer
# ===========================================================================


class MAMLTrainer:
    """
    MAML trainer for resource-aware procedural content generation.

    Supports both first-order MAML (FOMAML) for computational efficiency
    and full second-order MAML.

    Integration points:
        - ResourceMonitor  → hardware-aware penalty in every task env
        - make_pcgrl_env   → creates task-specific environments
        - TrainingLogger   → tracks meta-training metrics
    """

    def __init__(
        self,
        games: List[str] = None,
        representations: List[str] = None,
        meta_lr: float = 1e-3,
        inner_lr: float = 0.01,
        inner_steps: int = 5,
        meta_batch_size: int = 4,
        n_trajectories: int = 128,
        total_meta_iterations: int = 500,
        first_order: bool = True,
        device: str = "auto",
        experiment_name: str = None,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
    ):
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_batch_size = meta_batch_size
        self.n_trajectories = n_trajectories
        self.total_meta_iterations = total_meta_iterations
        self.first_order = first_order

        # Device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Task distribution
        self.task_distribution = TaskDistribution(games, representations)

        # Resource monitoring
        self.resource_monitor = ResourceMonitor(use_gpu=(self.device == "cuda"))

        # Experiment tracking
        if experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"MAML_{timestamp}"
        self.experiment_name = experiment_name
        self.logger = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name)
        self.checkpoint_dir = create_checkpoint_dir(checkpoint_dir, experiment_name)

        # Will be initialized when we discover obs/action dims from env
        self.policy: Optional[MAMLPolicy] = None
        self.meta_optimizer: Optional[optim.Adam] = None

        print(f"\n{'=' * 60}")
        print(f"MAML Trainer Initialized")
        print(f"{'=' * 60}")
        print(f"  Games          : {games or ['zelda', 'sokoban', 'binary']}")
        print(f"  Meta LR (beta) : {meta_lr}")
        print(f"  Inner LR (alpha): {inner_lr}")
        print(f"  Inner Steps (K): {inner_steps}")
        print(f"  Meta Batch Size: {meta_batch_size}")
        print(f"  First-order    : {first_order}")
        print(f"  Device         : {self.device}")
        print(f"  Checkpoint     : {self.checkpoint_dir}")
        print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    def _init_policy(self, obs_dim: int, action_dim: int):
        """Initialize policy once env dimensions are known."""
        self.policy = MAMLPolicy(obs_dim, action_dim).to(self.device)
        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=self.meta_lr)
        print(f"[OK] Policy initialized: obs_dim={obs_dim}, action_dim={action_dim}")

    # ------------------------------------------------------------------
    def inner_loop(self, task: Dict) -> Tuple[OrderedDict, float]:
        """
        MAML inner loop — adapt policy to a specific task.

        Args:
            task: Task configuration dict

        Returns:
            adapted_params: Parameters after K gradient steps
            final_loss: Loss value after adaptation
        """
        env = self.task_distribution.create_env(task, self.resource_monitor)

        # Lazy init policy from env shapes
        if self.policy is None:
            obs_space = env.observation_space
            act_space = env.action_space
            obs_dim = int(np.prod(obs_space.shape))
            action_dim = (
                act_space.n
                if hasattr(act_space, "n")
                else int(np.prod(act_space.shape))
            )
            self._init_policy(obs_dim, action_dim)

        # [PATCH P2] Clone from named_parameters() with requires_grad so that
        # torch.autograd.grad() can actually differentiate through these tensors.
        # The original used state_dict() which returns detached tensors — meaning
        # all inner-loop gradients were silently None and MAML was not learning.
        adapted_params = OrderedDict(
            (name, param.clone().requires_grad_(True))
            for name, param in self.policy.named_parameters()
        )

        # K gradient steps on this task
        final_loss_val = 0.0
        for step_k in range(self.inner_steps):
            traj = collect_trajectories(
                env, self.policy, self.n_trajectories, self.device, adapted_params
            )
            loss = compute_policy_loss(traj, self.policy, adapted_params)

            grads = torch.autograd.grad(
                loss,
                adapted_params.values(),
                create_graph=not self.first_order,
                allow_unused=True,
            )

            adapted_params = OrderedDict(
                (
                    name,
                    param
                    - self.inner_lr * (g if g is not None else torch.zeros_like(param)),
                )
                for (name, param), g in zip(adapted_params.items(), grads)
            )
            final_loss_val = loss.item()

        env.close()
        return adapted_params, final_loss_val

    # ------------------------------------------------------------------
    def meta_update(self, tasks: List[Dict]) -> float:
        """
        MAML outer loop — meta-update across a batch of tasks.

        Args:
            tasks: List of task configuration dicts

        Returns:
            Average meta-loss across tasks
        """
        meta_loss = torch.tensor(0.0, device=self.device, requires_grad=True)

        for task in tasks:
            # Inner loop: adapt (also lazily initializes policy on first call)
            adapted_params, _ = self.inner_loop(task)

            # Evaluate adapted params on fresh trajectories from same task
            env = self.task_distribution.create_env(task, self.resource_monitor)
            eval_traj = collect_trajectories(
                env, self.policy, self.n_trajectories, self.device, adapted_params
            )
            task_loss = compute_policy_loss(eval_traj, self.policy, adapted_params)
            meta_loss = meta_loss + task_loss / len(tasks)
            env.close()

        # Meta gradient step
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
        self.meta_optimizer.step()

        return meta_loss.item()

    # ------------------------------------------------------------------
    def train(self):
        """Run full MAML meta-training loop."""
        print(
            f"\nStarting MAML meta-training "
            f"({self.total_meta_iterations} iterations)...\n"
        )

        best_meta_loss = float("inf")
        meta_loss_history: List[float] = []

        for iteration in range(1, self.total_meta_iterations + 1):
            # Sample task batch
            tasks = self.task_distribution.sample_tasks(self.meta_batch_size)

            # Meta-update
            meta_loss = self.meta_update(tasks)
            meta_loss_history.append(meta_loss)

            # Resource snapshot
            resources = self.resource_monitor.get_resources()

            task_names = [t["game"] for t in tasks]
            print(
                f"Iter {iteration}/{self.total_meta_iterations} | "
                f"Meta-Loss: {meta_loss:.4f} | "
                f"Tasks: {task_names} | "
                f"CPU: {resources['cpu_percent']:.0f}% | "
                f"RAM: {resources['ram_percent']:.0f}%"
            )

            # [PATCH P4] Log step using correct TrainingLogger signature:
            # log_step(reward, resources, content_metrics, action, penalty_info)
            # Meta-loss is passed as a proxy reward for trend tracking.
            self.logger.log_step(
                reward=-meta_loss,  # negative loss as reward proxy
                resources=resources,
                content_metrics={"meta_loss": meta_loss, "iteration": iteration},
            )

            # Save best
            if meta_loss < best_meta_loss:
                best_meta_loss = meta_loss
                self._save_checkpoint("best_meta_model.pt", iteration, meta_loss)

            # Periodic checkpoint
            if iteration % 50 == 0:
                self._save_checkpoint(
                    f"meta_model_iter_{iteration}.pt", iteration, meta_loss
                )

        # Final checkpoint
        self._save_checkpoint(
            "final_meta_model.pt", self.total_meta_iterations, meta_loss
        )
        self.logger.save()
        print(f"\n[OK] MAML training complete. Best meta-loss: {best_meta_loss:.4f}")
        return meta_loss_history

    # ------------------------------------------------------------------
    def adapt_to_new_task(self, task: Dict, adaptation_steps: int = None) -> MAMLPolicy:
        """
        Adapt meta-learned policy to a new task with few gradient steps.
        This is the key benefit of MAML — fast adaptation.

        Args:
            task: New task configuration
            adaptation_steps: Override inner_steps for adaptation

        Returns:
            A copy of the policy with adapted parameters
        """
        steps = adaptation_steps or self.inner_steps
        print(f"Adapting to new task: {task['game']} ({steps} steps)...")

        orig = self.inner_steps
        self.inner_steps = steps
        adapted_params, loss = self.inner_loop(task)
        self.inner_steps = orig

        adapted_policy = copy.deepcopy(self.policy)
        with torch.no_grad():
            for name, param in adapted_policy.named_parameters():
                param.copy_(adapted_params[name])

        print(f"[OK] Adapted. Loss after {steps} steps: {loss:.4f}")
        return adapted_policy

    # ------------------------------------------------------------------
    def _save_checkpoint(self, filename: str, iteration: int, loss: float):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "iteration": iteration,
                "policy_state_dict": self.policy.state_dict(),
                "optimizer_state_dict": self.meta_optimizer.state_dict(),
                "meta_loss": loss,
                "config": {
                    "meta_lr": self.meta_lr,
                    "inner_lr": self.inner_lr,
                    "inner_steps": self.inner_steps,
                    "first_order": self.first_order,
                },
            },
            path,
        )

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=self.device)
        if self.policy is not None:
            self.policy.load_state_dict(ckpt["policy_state_dict"])
            self.meta_optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[OK] Loaded checkpoint from {path} (iteration {ckpt['iteration']})")


# ===========================================================================
# CLI Entry Point
# ===========================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MAML Training for RAPCG-MetaRL")
    parser.add_argument(
        "--games",
        nargs="+",
        default=["zelda", "sokoban", "binary"],
        help="Games for the task distribution",
    )
    parser.add_argument(
        "--representations",
        nargs="+",
        default=["narrow", "wide", "turtle"],
        help="Representation types",
    )
    parser.add_argument(
        "--meta-lr",
        type=float,
        default=1e-3,
        help="Meta learning rate (outer loop, beta)",
    )
    parser.add_argument(
        "--inner-lr", type=float, default=0.01, help="Inner loop learning rate (α)"
    )
    parser.add_argument(
        "--inner-steps", type=int, default=5, help="Gradient steps per inner loop (K)"
    )
    parser.add_argument(
        "--meta-batch", type=int, default=4, help="Tasks per meta-update"
    )
    parser.add_argument(
        "--iterations", type=int, default=500, help="Total meta-training iterations"
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=128,
        help="Steps per trajectory collection",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"]
    )
    parser.add_argument(
        "--second-order",
        action="store_true",
        help="Use full second-order MAML (slower)",
    )
    parser.add_argument("--experiment-name", type=str, default=None)

    args = parser.parse_args()

    trainer = MAMLTrainer(
        games=args.games,
        representations=args.representations,
        meta_lr=args.meta_lr,
        inner_lr=args.inner_lr,
        inner_steps=args.inner_steps,
        meta_batch_size=args.meta_batch,
        n_trajectories=args.n_trajectories,
        total_meta_iterations=args.iterations,
        first_order=not args.second_order,
        device=args.device,
        experiment_name=args.experiment_name,
    )

    trainer.train()
