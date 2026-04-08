# rlhf_trainer.py
"""
RLHF (Reinforcement Learning from Human Feedback) for RAPCG-MetaRL

Pipeline:
    Phase 1 — Generate levels with current policy
    Phase 2 — Collect human preferences (pairwise comparisons)
    Phase 3 — Train a Bradley-Terry reward model on preferences
    Phase 4 — Fine-tune the PCG agent with PPO against the learned reward

The learned reward model is blended with the existing resource-aware
environment reward so the agent optimizes BOTH hardware efficiency
and human-preferred level quality.
"""
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from pathlib import Path

# Project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, 'gym-pcgrl'))

from utils import ResourceMonitor, TrainingLogger, create_checkpoint_dir
from wrappers.pcgrl_env import make_pcgrl_env
from wrappers.helper import calculate_content_metrics

import gym
from gym import spaces


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

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("Error: stable-baselines3 not installed. "
          "Install with: pip install stable-baselines3")
    sys.exit(1)


# ===========================================================================
# Phase 1 — Level Generation for Feedback
# ===========================================================================

def generate_levels(game: str = 'zelda',
                    representation: str = 'narrow',
                    n_levels: int = 50,
                    model_path: Optional[str] = None,
                    device: str = 'cpu') -> List[np.ndarray]:
    """
    Generate levels using an existing policy (or random actions).

    Args:
        game: Game environment name
        representation: Representation type
        n_levels: Number of levels to generate
        model_path: Path to a pre-trained PPO model (.zip). None → random.
        device: Torch device

    Returns:
        List of level arrays (numpy)
    """
    resource_monitor = ResourceMonitor(use_gpu=(device == 'cuda'))
    env = make_pcgrl_env(game=game, representation=representation,
                         resource_monitor=resource_monitor)
    # Flatten Dict observation space to 1-D vector
    if isinstance(env.observation_space, gym.spaces.Dict):
        env = DictFlattenWrapper(env)

    model = PPO.load(model_path, device=device) if model_path else None

    levels: List[np.ndarray] = []
    for i in range(n_levels):
        obs = env.reset()
        done = False
        steps = 0
        while not done and steps < 500:
            if model:
                action, _ = model.predict(obs, deterministic=False)
            else:
                action = env.action_space.sample()
            obs, _, done, info = env.step(action)
            steps += 1

        # Extract level representation
        if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '_rep'):
            level = env.unwrapped._rep._map.copy()
        elif isinstance(obs, np.ndarray) and obs.ndim >= 2:
            level = obs[:, :, 0] if obs.ndim == 3 else obs
        else:
            side = max(1, int(np.sqrt(obs.size)))
            level = obs.flatten()[:side * side].reshape(side, side)

        levels.append(level)
        if (i + 1) % 10 == 0:
            print(f"  Generated {i + 1}/{n_levels} levels")

    env.close()
    print(f"[OK] Generated {len(levels)} levels")
    return levels


# ===========================================================================
# Phase 2 — Preference Collection
# ===========================================================================

class PreferenceCollector:
    """
    Collects and persists human preference labels.
    Each entry: (level_A, level_B, preference)
        preference ∈ {0.0: A preferred, 1.0: B preferred, 0.5: tie}
    """

    def __init__(self, save_path: str = 'data/preferences'):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.preferences: List[Dict] = []
        self._load_existing()

    # ----- persistence -----
    def _load_existing(self):
        pref_file = os.path.join(self.save_path, 'preferences.json')
        if os.path.exists(pref_file):
            with open(pref_file, 'r') as f:
                self.preferences = json.load(f)
            print(f"[OK] Loaded {len(self.preferences)} existing preferences")

    def save(self):
        pref_file = os.path.join(self.save_path, 'preferences.json')
        with open(pref_file, 'w') as f:
            json.dump(self.preferences, f, indent=2, default=str)
        print(f"[OK] Saved {len(self.preferences)} preferences to {pref_file}")

    # ----- add data -----
    def add_preference(self, level_a: np.ndarray, level_b: np.ndarray,
                       preference: float, metadata: Optional[Dict] = None):
        """
        Record a single pairwise preference.

        Args:
            level_a / level_b: Level arrays
            preference: 0.0 = A wins, 1.0 = B wins, 0.5 = tie
            metadata: Optional info (annotator id, game, …)
        """
        self.preferences.append({
            'level_a': level_a.tolist(),
            'level_b': level_b.tolist(),
            'preference': preference,
            'metrics_a': calculate_content_metrics(level_a),
            'metrics_b': calculate_content_metrics(level_b),
            'metadata': metadata or {},
            'timestamp': datetime.now().isoformat(),
        })

    # ----- interactive collection -----
    def collect_interactive(self, levels: List[np.ndarray],
                            game: str = 'zelda',
                            n_comparisons: int = 20):
        """
        CLI-based interactive preference collection.
        Displays pairs of levels and asks the annotator to choose.
        """
        try:
            import matplotlib.pyplot as plt
            _has_plt = True
        except ImportError:
            _has_plt = False

        print(f"\n{'=' * 60}")
        print(f"Human Preference Collection ({game})")
        print(f"{'=' * 60}")
        print(f"You will see {n_comparisons} pairs of levels.")
        print(f"Choose: [1] Left  [2] Right  [3] Tie  [q] Quit\n")

        for i in range(n_comparisons):
            idx_a, idx_b = np.random.choice(len(levels), 2, replace=False)
            level_a, level_b = levels[idx_a], levels[idx_b]

            metrics_a = calculate_content_metrics(level_a)
            metrics_b = calculate_content_metrics(level_b)

            # Visualise if matplotlib available
            if _has_plt:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                ax1.imshow(level_a, cmap='tab20')
                ax1.set_title(f'Level A (idx {idx_a})')
                ax1.axis('off')
                ax2.imshow(level_b, cmap='tab20')
                ax2.set_title(f'Level B (idx {idx_b})')
                ax2.axis('off')
                fig.suptitle(
                    f'Comparison {i + 1}/{n_comparisons}  |  '
                    f'A: div={metrics_a["diversity"]:.2f} cmplx={metrics_a["complexity"]:.2f}  |  '
                    f'B: div={metrics_b["diversity"]:.2f} cmplx={metrics_b["complexity"]:.2f}'
                )
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1)
            else:
                print(f"\n--- Comparison {i + 1}/{n_comparisons} ---")
                print(f"  A (idx {idx_a}): "
                      f"div={metrics_a['diversity']:.2f}  "
                      f"cmplx={metrics_a['complexity']:.2f}")
                print(f"  B (idx {idx_b}): "
                      f"div={metrics_b['diversity']:.2f}  "
                      f"cmplx={metrics_b['complexity']:.2f}")

            while True:
                choice = input(f"  [{i+1}/{n_comparisons}] "
                               f"[1] A  [2] B  [3] Tie  [q] Quit: ").strip()
                if choice == '1':
                    self.add_preference(level_a, level_b, 0.0,
                                        {'game': game, 'type': 'interactive'})
                    break
                elif choice == '2':
                    self.add_preference(level_a, level_b, 1.0,
                                        {'game': game, 'type': 'interactive'})
                    break
                elif choice == '3':
                    self.add_preference(level_a, level_b, 0.5,
                                        {'game': game, 'type': 'interactive'})
                    break
                elif choice.lower() == 'q':
                    if _has_plt:
                        plt.close('all')
                    self.save()
                    return
                else:
                    print("  Invalid. Enter 1, 2, 3, or q.")

            if _has_plt:
                plt.close('all')

        self.save()
        print(f"\n[OK] Collected {n_comparisons} preferences "
              f"(total: {len(self.preferences)})")

    # ----- synthetic (for testing / bootstrapping) -----
    def generate_synthetic_preferences(self, levels: List[np.ndarray],
                                        n_comparisons: int = 100,
                                        game: str = 'zelda'):
        """
        Generate synthetic preferences based on content metrics.
        Simulates a human who prefers higher diversity and moderate complexity.
        Useful for prototyping before collecting real annotations.
        """
        print(f"Generating {n_comparisons} synthetic preferences...")

        for _ in range(n_comparisons):
            idx_a, idx_b = np.random.choice(len(levels), 2, replace=False)
            level_a, level_b = levels[idx_a], levels[idx_b]

            m_a = calculate_content_metrics(level_a)
            m_b = calculate_content_metrics(level_b)

            score_a = m_a['diversity'] * 0.6 + m_a['complexity'] * 0.4
            score_b = m_b['diversity'] * 0.6 + m_b['complexity'] * 0.4

            # Noise to simulate human inconsistency
            score_a += np.random.normal(0, 0.05)
            score_b += np.random.normal(0, 0.05)

            if score_a > score_b + 0.02:
                pref = 0.0
            elif score_b > score_a + 0.02:
                pref = 1.0
            else:
                pref = 0.5

            self.add_preference(level_a, level_b, pref,
                                {'game': game, 'type': 'synthetic'})

        self.save()
        print(f"[OK] Generated {n_comparisons} synthetic preferences")


# ===========================================================================
# Phase 3 — Reward Model (Bradley-Terry)
# ===========================================================================

class RewardModel(nn.Module):
    """
    Learned scalar reward model trained on pairwise human preferences.

    Bradley-Terry model:  P(A ≻ B) = σ(r(A) − r(B))
    """

    def __init__(self, input_dim: int,
                 hidden_sizes: List[int] = None):
        super().__init__()
        hidden_sizes = hidden_sizes or [128, 64, 32]

        layers: List[nn.Module] = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]
            prev = h
        layers.append(nn.Linear(prev, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, level: torch.Tensor) -> torch.Tensor:
        """Predict scalar reward for a level."""
        return self.network(level)

    def predict_preference(self, level_a: torch.Tensor,
                           level_b: torch.Tensor) -> torch.Tensor:
        """P(A ≻ B) via Bradley-Terry."""
        return torch.sigmoid(self.forward(level_a) - self.forward(level_b))


class RewardModelTrainer:
    """Train the reward model on collected preferences."""

    def __init__(self, input_dim: int, device: str = 'cpu',
                 learning_rate: float = 1e-3):
        self.device = device
        self.input_dim = input_dim
        self.reward_model = RewardModel(input_dim).to(device)
        self.optimizer = optim.Adam(self.reward_model.parameters(),
                                    lr=learning_rate)

    # ------------------------------------------------------------------
    def train(self, preferences: List[Dict], epochs: int = 100,
              batch_size: int = 32,
              validation_split: float = 0.1) -> Dict[str, List[float]]:
        """
        Train reward model on preference data.

        Args:
            preferences: List of preference dicts from PreferenceCollector
            epochs: Number of training epochs
            batch_size: Mini-batch size
            validation_split: Fraction held out for validation

        Returns:
            History dict with train_loss, val_loss, val_accuracy lists
        """
        print(f"\nTraining Reward Model on {len(preferences)} preferences...")

        levels_a, levels_b, labels = [], [], []
        for pref in preferences:
            a = self._pad_or_truncate(np.array(pref['level_a']).flatten())
            b = self._pad_or_truncate(np.array(pref['level_b']).flatten())
            levels_a.append(a)
            levels_b.append(b)
            labels.append(pref['preference'])

        levels_a = torch.FloatTensor(np.array(levels_a)).to(self.device)
        levels_b = torch.FloatTensor(np.array(levels_b)).to(self.device)
        labels = torch.FloatTensor(labels).to(self.device)

        # Split
        n = len(labels)
        n_val = max(1, int(n * validation_split))
        perm = torch.randperm(n)
        train_idx, val_idx = perm[n_val:], perm[:n_val]

        history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': [], 'val_accuracy': []
        }

        for epoch in range(epochs):
            # ---- train ----
            self.reward_model.train()
            shuffled = train_idx[torch.randperm(len(train_idx))]
            epoch_loss, n_batches = 0.0, 0

            for i in range(0, len(shuffled), batch_size):
                bi = shuffled[i:i + batch_size]
                pred = self.reward_model.predict_preference(
                    levels_a[bi], levels_b[bi]).squeeze(-1)
                loss = nn.functional.binary_cross_entropy(pred, labels[bi])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            # ---- validate ----
            self.reward_model.eval()
            with torch.no_grad():
                v_pred = self.reward_model.predict_preference(
                    levels_a[val_idx], levels_b[val_idx]).squeeze(-1)
                v_loss = nn.functional.binary_cross_entropy(
                    v_pred, labels[val_idx]).item()

                choices = (v_pred > 0.5).float()
                gt = (labels[val_idx] > 0.5).float()
                ties = (labels[val_idx] == 0.5)
                v_acc = ((choices == gt) | ties).float().mean().item()

            history['train_loss'].append(epoch_loss / max(n_batches, 1))
            history['val_loss'].append(v_loss)
            history['val_accuracy'].append(v_acc)

            if (epoch + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | "
                      f"Train: {epoch_loss / max(n_batches, 1):.4f} | "
                      f"Val: {v_loss:.4f} | "
                      f"Acc: {v_acc:.1%}")

        print(f"[OK] Reward model trained. "
              f"Final val accuracy: {v_acc:.1%}")
        return history

    # ------------------------------------------------------------------
    def _pad_or_truncate(self, arr: np.ndarray) -> np.ndarray:
        if len(arr) >= self.input_dim:
            return arr[:self.input_dim].astype(np.float32)
        return np.pad(arr, (0, self.input_dim - len(arr))).astype(np.float32)

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.reward_model.state_dict(),
            'input_dim': self.input_dim,
        }, path)
        print(f"[OK] Reward model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.reward_model.load_state_dict(ckpt['model_state_dict'])
        print(f"[OK] Reward model loaded from {path}")


# ===========================================================================
# Phase 4 — RLHF Environment Wrapper & PPO Fine-tuning
# ===========================================================================

class RLHFRewardWrapper(gym.Wrapper):
    """
    Blends the original environment reward with the learned human-preference
    reward model.

        r_final = (1 − w) · r_env  +  w · r_human

    where w = rlhf_weight.
    """

    def __init__(self, env, reward_model: RewardModel,
                 rlhf_weight: float = 0.5,
                 device: str = 'cpu',
                 input_dim: int = 121):
        super().__init__(env)
        self.reward_model = reward_model
        self.rlhf_weight = rlhf_weight
        self.device = device
        self.input_dim = input_dim

    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)

        # Flatten current observation for the reward model
        level_flat = self._to_flat(obs, info)

        with torch.no_grad():
            t = torch.FloatTensor(level_flat).unsqueeze(0).to(self.device)
            human_reward = self.reward_model(t).item()

        blended = ((1 - self.rlhf_weight) * env_reward
                    + self.rlhf_weight * human_reward)

        info['env_reward'] = env_reward
        info['human_reward'] = human_reward
        info['blended_reward'] = blended

        return obs, blended, done, info

    def _to_flat(self, obs, info: Dict) -> np.ndarray:
        """Extract and pad/truncate a flat level vector."""
        if isinstance(info, dict) and 'map' in info:
            arr = np.array(info['map']).flatten()
        elif isinstance(obs, np.ndarray):
            arr = obs.flatten()
        else:
            arr = np.zeros(self.input_dim)

        if len(arr) >= self.input_dim:
            return arr[:self.input_dim].astype(np.float32)
        return np.pad(arr, (0, self.input_dim - len(arr))).astype(np.float32)


class RLHFCallback(BaseCallback):
    """
    Callback that logs RLHF-specific metrics (blended, env, human rewards)
    during PPO fine-tuning.
    """

    def __init__(self, resource_monitor: ResourceMonitor,
                 training_logger: TrainingLogger,
                 verbose: int = 1):
        super().__init__(verbose)
        self.resource_monitor = resource_monitor
        self.training_logger = training_logger
        self.episode_count = 0
        self.ep_env_rewards: List[float] = []
        self.ep_human_rewards: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        info = infos[0] if infos else {}

        self.ep_env_rewards.append(info.get('env_reward', 0.0))
        self.ep_human_rewards.append(info.get('human_reward', 0.0))

        done = self.locals.get('dones', [False])[0]
        if done:
            self.episode_count += 1
            if self.verbose and self.episode_count % 20 == 0:
                resources = self.resource_monitor.get_resources()
                mean_env = np.mean(self.ep_env_rewards[-200:])
                mean_hum = np.mean(self.ep_human_rewards[-200:])
                print(f"  RLHF Ep {self.episode_count} | "
                      f"Env: {mean_env:.3f}  Human: {mean_hum:.3f} | "
                      f"CPU: {resources['cpu_percent']:.0f}%  "
                      f"RAM: {resources['ram_percent']:.0f}%")
        return True


# ===========================================================================
# RLHF Trainer (full pipeline)
# ===========================================================================

class RLHFTrainer:
    """
    End-to-end RLHF pipeline for RAPCG-MetaRL.

    Steps:
        1. generate_levels()      — roll out current policy
        2. collect_preferences()  — gather human (or synthetic) labels
        3. train_reward_model()   — fit Bradley-Terry reward model
        4. fine_tune_with_rlhf()  — PPO against blended reward
    """

    def __init__(self,
                 game: str = 'zelda',
                 representation: str = 'narrow',
                 base_model_path: Optional[str] = None,
                 rlhf_weight: float = 0.5,
                 reward_model_lr: float = 1e-3,
                 reward_model_epochs: int = 100,
                 ppo_timesteps: int = 50_000,
                 device: str = 'auto',
                 experiment_name: Optional[str] = None,
                 log_dir: str = 'logs',
                 checkpoint_dir: str = 'checkpoints'):
        """
        Args:
            game: Game environment name
            representation: Level representation
            base_model_path: Pre-trained PPO model path (None → random init)
            rlhf_weight: Blend weight for human reward (0–1)
            reward_model_lr: Reward-model learning rate
            reward_model_epochs: Reward-model training epochs
            ppo_timesteps: PPO fine-tuning timesteps
            device: 'cpu', 'cuda', or 'auto'
            experiment_name: Unique run name
            log_dir / checkpoint_dir: Output directories
        """
        self.game = game
        self.representation = representation
        self.base_model_path = base_model_path
        self.rlhf_weight = rlhf_weight
        self.reward_model_lr = reward_model_lr
        self.reward_model_epochs = reward_model_epochs
        self.ppo_timesteps = ppo_timesteps

        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        if experiment_name is None:
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            experiment_name = f"RLHF_{game}_{ts}"
        self.experiment_name = experiment_name

        self.resource_monitor = ResourceMonitor(
            use_gpu=(self.device == 'cuda'))
        self.logger = TrainingLogger(log_dir=log_dir,
                                     experiment_name=experiment_name)
        self.checkpoint_dir = create_checkpoint_dir(checkpoint_dir,
                                                    experiment_name)

        # Determine input dimension from a temporary env
        tmp_env = make_pcgrl_env(game=game, representation=representation,
                                 resource_monitor=self.resource_monitor)
        if isinstance(tmp_env.observation_space, gym.spaces.Dict):
            tmp_env = DictFlattenWrapper(tmp_env)
        self.input_dim = int(np.prod(tmp_env.observation_space.shape))
        tmp_env.close()

        # Sub-components
        self.preference_collector = PreferenceCollector(
            save_path=os.path.join('data', 'preferences', game))
        self.reward_trainer = RewardModelTrainer(
            self.input_dim, self.device, reward_model_lr)

        print(f"\n{'=' * 60}")
        print(f"RLHF Trainer Initialized")
        print(f"{'=' * 60}")
        print(f"  Game            : {game}")
        print(f"  Representation  : {representation}")
        print(f"  RLHF Weight     : {rlhf_weight}")
        print(f"  Input Dimension : {self.input_dim}")
        print(f"  Device          : {self.device}")
        print(f"  Base Model      : {base_model_path or '(random init)'}")
        print(f"  PPO Timesteps   : {ppo_timesteps}")
        print(f"  Checkpoint      : {self.checkpoint_dir}")
        print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Step 1
    # ------------------------------------------------------------------
    def generate_levels_for_feedback(self,
                                     n_levels: int = 50) -> List[np.ndarray]:
        """Generate levels with the current (or random) policy."""
        return generate_levels(
            game=self.game,
            representation=self.representation,
            n_levels=n_levels,
            model_path=self.base_model_path,
            device=self.device,
        )

    # ------------------------------------------------------------------
    # Step 2
    # ------------------------------------------------------------------
    def collect_preferences(self, levels: List[np.ndarray],
                            n_comparisons: int = 50,
                            use_synthetic: bool = False):
        """Collect human or synthetic preferences."""
        if use_synthetic:
            self.preference_collector.generate_synthetic_preferences(
                levels, n_comparisons, self.game)
        else:
            self.preference_collector.collect_interactive(
                levels, self.game, n_comparisons)

    # ------------------------------------------------------------------
    # Step 3
    # ------------------------------------------------------------------
    def train_reward_model(self) -> RewardModel:
        """Train the Bradley-Terry reward model on collected preferences."""
        if not self.preference_collector.preferences:
            raise ValueError(
                "No preferences collected. "
                "Run collect_preferences() first.")

        self.reward_trainer.train(
            self.preference_collector.preferences,
            epochs=self.reward_model_epochs,
        )

        rm_path = os.path.join(self.checkpoint_dir, 'reward_model.pt')
        self.reward_trainer.save(rm_path)

        return self.reward_trainer.reward_model

    # ------------------------------------------------------------------
    # Step 4
    # ------------------------------------------------------------------
    def fine_tune_with_rlhf(self,
                            reward_model: Optional[RewardModel] = None):
        """
        Fine-tune the PCG policy with PPO using a blended reward
        (environment + learned human preference).
        """
        if reward_model is None:
            reward_model = self.reward_trainer.reward_model

        print(f"\nFine-tuning with RLHF (weight={self.rlhf_weight})...")

        rm = self.resource_monitor  # shorthand

        def _make_rlhf_env():
            base = make_pcgrl_env(
                game=self.game,
                representation=self.representation,
                resource_monitor=rm,
                ram_penalty_weight=0.2,
                cpu_penalty_weight=0.1,
                gpu_penalty_weight=0.1,
            )
            # Flatten Dict observation space
            if isinstance(base.observation_space, gym.spaces.Dict):
                base = DictFlattenWrapper(base)
            return RLHFRewardWrapper(
                base, reward_model,
                rlhf_weight=self.rlhf_weight,
                device=self.device,
                input_dim=self.input_dim,
            )

        env = DummyVecEnv([_make_rlhf_env])

        # Load or create PPO model
        if self.base_model_path:
            model = PPO.load(self.base_model_path, env=env,
                             device=self.device)
            print(f"[OK] Loaded base model from {self.base_model_path}")
        else:
            model = PPO(
                'MlpPolicy', env,
                learning_rate=2.5e-4,
                n_steps=128,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                device=self.device,
            )
            print("[OK] Created new PPO model for RLHF fine-tuning")

        # Callback
        callback = RLHFCallback(self.resource_monitor, self.logger)

        model.learn(total_timesteps=self.ppo_timesteps, callback=callback)

        # Save
        rlhf_path = os.path.join(self.checkpoint_dir, 'rlhf_model.zip')
        model.save(rlhf_path)
        print(f"[OK] RLHF-tuned model saved to {rlhf_path}")

        env.close()
        return model

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run_full_pipeline(self, n_levels: int = 50,
                          n_comparisons: int = 50,
                          use_synthetic: bool = True):
        """
        Run the complete RLHF pipeline end-to-end.

        Args:
            n_levels: Levels to generate for feedback
            n_comparisons: Pairwise comparisons to collect
            use_synthetic: True → synthetic prefs (for testing)
        """
        print(f"\n{'=' * 60}")
        print(f"RLHF Full Pipeline — {self.game}")
        print(f"{'=' * 60}\n")

        levels = self.generate_levels_for_feedback(n_levels)
        self.collect_preferences(levels, n_comparisons, use_synthetic)
        reward_model = self.train_reward_model()
        model = self.fine_tune_with_rlhf(reward_model)

        print(f"\n{'=' * 60}")
        print(f"[OK] RLHF Pipeline Complete")
        print(f"{'=' * 60}")
        print(f"  Reward model : {self.checkpoint_dir}/reward_model.pt")
        print(f"  RLHF model   : {self.checkpoint_dir}/rlhf_model.zip")

        return model


# ===========================================================================
# CLI Entry Point
# ===========================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='RLHF Training for RAPCG-MetaRL')
    parser.add_argument('--game', type=str, default='zelda',
                        choices=['zelda', 'sokoban', 'binary'])
    parser.add_argument('--representation', type=str, default='narrow')
    parser.add_argument('--base-model', type=str, default=None,
                        help='Path to pre-trained PPO .zip model')
    parser.add_argument('--rlhf-weight', type=float, default=0.5,
                        help='Weight of human-preference reward (0–1)')
    parser.add_argument('--n-levels', type=int, default=50,
                        help='Levels to generate for feedback')
    parser.add_argument('--n-comparisons', type=int, default=50,
                        help='Number of pairwise comparisons')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic preferences (for testing)')
    parser.add_argument('--interactive', action='store_true',
                        help='Collect real human preferences')
    parser.add_argument('--timesteps', type=int, default=50_000,
                        help='PPO fine-tuning timesteps')
    parser.add_argument('--reward-epochs', type=int, default=100,
                        help='Reward model training epochs')
    parser.add_argument('--reward-model-only', action='store_true',
                        help='Only train reward model, skip PPO')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'])
    parser.add_argument('--experiment-name', type=str, default=None)

    args = parser.parse_args()

    trainer = RLHFTrainer(
        game=args.game,
        representation=args.representation,
        base_model_path=args.base_model,
        rlhf_weight=args.rlhf_weight,
        reward_model_epochs=args.reward_epochs,
        ppo_timesteps=args.timesteps,
        device=args.device,
        experiment_name=args.experiment_name,
    )

    if args.reward_model_only:
        levels = trainer.generate_levels_for_feedback(args.n_levels)
        trainer.collect_preferences(
            levels, args.n_comparisons,
            use_synthetic=not args.interactive)
        trainer.train_reward_model()
    else:
        trainer.run_full_pipeline(
            n_levels=args.n_levels,
            n_comparisons=args.n_comparisons,
            use_synthetic=not args.interactive,
        )
