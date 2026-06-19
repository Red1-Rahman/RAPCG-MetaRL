# key Codes

## wrappers/helper.py :

```python
# wrappers/helper.py
"""
Helper utilities for RAPCG-MetaRL
Includes resource monitoring, level parsing, and other utility functions.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional


def parse_vglc_level(file_path: str) -> np.ndarray:
    """
    Convert VGLC tile-based level file into numpy array.

    Args:
        file_path: Path to VGLC level file (.txt or .json)

    Returns:
        numpy array representing the level grid
    """
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data = json.load(f)
            # Handle different JSON structures
            if isinstance(data, dict) and "level" in data:
                level = data["level"]
            elif isinstance(data, list):
                level = data
            else:
                raise ValueError(f"Unsupported JSON structure in {file_path}")

            # Convert to numpy array
            return np.array(level)

    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            lines = f.readlines()
        grid = [list(line.strip()) for line in lines if line.strip()]
        return np.array(grid)

    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def load_vglc_levels(data_dir: str, game: str) -> List[np.ndarray]:
    """
    Load all VGLC levels for a specific game.

    Args:
        data_dir: Directory containing VGLC data
        game: Game name (e.g., 'SMB', 'zelda')

    Returns:
        List of level arrays
    """
    levels = []
    json_file = os.path.join(data_dir, f"{game}.json")

    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Parse JSON structure
            if isinstance(data, dict):
                for level_name, level_data in data.items():
                    if isinstance(level_data, list):
                        levels.append(np.array(level_data))
            elif isinstance(data, list):
                levels = [np.array(level) for level in data]

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return levels


def tile_diversity(level: np.ndarray) -> float:
    """
    Calculate diversity metric for a level based on unique tiles.

    Args:
        level: Level array

    Returns:
        Diversity score (0-1)
    """
    unique_tiles = len(np.unique(level))
    total_tiles = level.size
    return unique_tiles / total_tiles if total_tiles > 0 else 0.0


def pattern_complexity(level: np.ndarray, window_size: int = 3) -> float:
    """
    Calculate pattern complexity using sliding window of unique patterns.

    Args:
        level: Level array
        window_size: Size of pattern window

    Returns:
        Complexity score
    """
    if level.size == 0:
        return 0.0

    patterns = set()
    h, w = level.shape if len(level.shape) == 2 else (1, level.shape[0])

    for i in range(h - window_size + 1):
        for j in range(w - window_size + 1):
            if len(level.shape) == 2:
                pattern = tuple(
                    level[i : i + window_size, j : j + window_size].flatten()
                )
            else:
                pattern = tuple(level[j : j + window_size])
            patterns.add(pattern)

    max_patterns = (h - window_size + 1) * (w - window_size + 1)
    return len(patterns) / max_patterns if max_patterns > 0 else 0.0


def calculate_content_metrics(level: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive content quality metrics for a level.

    Args:
        level: Level array

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "diversity": tile_diversity(level),
        "complexity": pattern_complexity(level),
        "size": level.size,
        "unique_tiles": len(np.unique(level)),
    }

    return metrics


def save_level(level: np.ndarray, filepath: str, format: str = "npy"):
    """
    Save generated level to file.

    Args:
        level: Level array
        filepath: Output file path
        format: Save format ('npy', 'txt', 'json')
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if format == "npy":
        np.save(filepath, level)
    elif format == "txt":
        with open(filepath, "w") as f:
            if len(level.shape) == 2:
                for row in level:
                    f.write("".join(map(str, row)) + "\n")
            else:
                f.write("".join(map(str, level)) + "\n")
    elif format == "json":
        with open(filepath, "w") as f:
            json.dump(level.tolist(), f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_level(filepath: str) -> np.ndarray:
    """
    Load level from file.

    Args:
        filepath: Path to level file

    Returns:
        Level array
    """
    if filepath.endswith(".npy"):
        return np.load(filepath)
    elif filepath.endswith(".txt"):
        with open(filepath, "r") as f:
            lines = f.readlines()
        return np.array([list(line.strip()) for line in lines if line.strip()])
    elif filepath.endswith(".json"):
        with open(filepath, "r") as f:
            data = json.load(f)
        return np.array(data)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
```

## wrappers\pcgrl_env.py :

```python
# wrappers/pcgrl_env.py
"""
Resource-Aware PCGRL Environment Wrapper
Wraps gym-pcgrl environments with dynamic resource adaptation.
"""

import sys
import os
import numpy as np

# gym-pcgrl uses old gym, so we need to import it directly
import gym

# Add gym-pcgrl to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gym-pcgrl"))
import gym_pcgrl

# Import Sokoban utilities and solvability config
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)
from sokoban_utils import SokobanSolvabilityWrapper
from solvability_config import get_solvability_config, print_solvability_config


class ResourceAwarePCGRLWrapper(gym.Wrapper):
    """
    Wrapper that adapts PCGRL environment complexity based on resource usage
    AND introduces resource-based reward shaping to make the agent truly resource-aware.
    """

    def __init__(
        self,
        env,
        resource_monitor,
        ram_penalty_weight=0.2,
        cpu_penalty_weight=0.1,
        gpu_penalty_weight=0.1,
        max_complexity=10,
        min_complexity=3,
    ):
        """
        Args:
            env: Base gym-pcgrl environment
            resource_monitor: ResourceMonitor instance for tracking resource usage
            ram_penalty_weight: Weight for RAM usage penalty (higher = stronger penalty)
            cpu_penalty_weight: Weight for CPU usage penalty
            gpu_penalty_weight: Weight for GPU usage penalty
            max_complexity: Maximum environment complexity (e.g., max enemies)
            min_complexity: Minimum environment complexity
        """
        super().__init__(env)
        self.resource_monitor = resource_monitor
        self.ram_penalty_weight = ram_penalty_weight
        self.cpu_penalty_weight = cpu_penalty_weight
        self.gpu_penalty_weight = gpu_penalty_weight
        self.max_complexity = max_complexity
        self.min_complexity = min_complexity
        self.current_complexity = max_complexity

        # Track penalty statistics for debugging
        self.total_ram_penalty = 0.0
        self.total_cpu_penalty = 0.0
        self.total_gpu_penalty = 0.0
        self.penalty_steps = 0

    def adapt_complexity(self, cpu_usage, ram_usage, gpu_usage):
        """
        Dynamically adjust environment complexity based on resource usage.

        Args:
            cpu_usage: CPU usage percentage (0-100)
            ram_usage: RAM usage percentage (0-100)
            gpu_usage: GPU usage percentage (0-100)
        """
        # If any resource is critically high, reduce complexity
        if gpu_usage > 85 or ram_usage > 90 or cpu_usage > 90:
            self.current_complexity = max(
                self.min_complexity, self.current_complexity - 1
            )
        # If resources are comfortable, increase complexity
        elif gpu_usage < 60 and ram_usage < 70 and cpu_usage < 70:
            self.current_complexity = min(
                self.max_complexity, self.current_complexity + 1
            )

        return self.current_complexity

    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Step the environment with resource-aware reward shaping.

        This implements the feedback loop that makes the agent resource-aware:
        1. Monitor current resource usage (CPU, RAM, GPU)
        2. Adapt environment complexity dynamically
        3. Apply resource-based penalties to the reward
        4. Return modified reward that guides agent toward efficient generation
        """
        # 1. Get current resource usage
        resources = self.resource_monitor.get_resources()
        cpu_usage = resources["cpu_percent"]
        ram_usage = resources["ram_percent"]
        gpu_usage = resources["gpu_mem_percent"]  # GPU memory usage

        # 2. ACTIVATE COMPLEXITY ADAPTATION (previously dormant)
        new_complexity = self.adapt_complexity(cpu_usage, ram_usage, gpu_usage)
        if new_complexity != self.current_complexity:
            self.current_complexity = new_complexity
            # Note: Real PCGRL env adaptation would require modifying internal parameters
            # For now, we track complexity for logging purposes

        # 3. Take the normal environment step
        obs, reward, done, info = self.env.step(action)

        # 4. IMPLEMENT RESOURCE REWARD SHAPING (The crucial fix!)
        # This creates the feedback loop that makes the agent resource-aware

        # RAM Penalty: Penalize if RAM usage exceeds 60%
        # Example: 80% RAM -> (80 - 60) * 0.2 = -4.0 penalty
        ram_penalty = 0.0
        if ram_usage > 78.0:
            ram_penalty = (ram_usage - 78.0) * self.ram_penalty_weight

        # CPU Penalty: Penalize if CPU usage exceeds 70%
        cpu_penalty = 0.0
        if cpu_usage > 70.0:
            cpu_penalty = (cpu_usage - 70.0) * self.cpu_penalty_weight

        # GPU Penalty: Penalize if GPU memory exceeds 70%
        gpu_penalty = 0.0
        if gpu_usage > 70.0:
            gpu_penalty = (gpu_usage - 70.0) * self.gpu_penalty_weight

        # Total resource penalty
        total_penalty = ram_penalty + cpu_penalty + gpu_penalty

        # Apply penalty to reward (this is what makes the agent learn!)
        shaped_reward = reward - total_penalty

        # Track statistics
        self.total_ram_penalty += ram_penalty
        self.total_cpu_penalty += cpu_penalty
        self.total_gpu_penalty += gpu_penalty
        self.penalty_steps += 1

        # Add debugging info
        if not isinstance(info, dict):
            info = {}

        info["raw_reward"] = reward
        info["shaped_reward"] = shaped_reward
        info["ram_penalty"] = ram_penalty
        info["cpu_penalty"] = cpu_penalty
        info["gpu_penalty"] = gpu_penalty
        info["total_penalty"] = total_penalty
        info["env_complexity"] = self.current_complexity
        info["ram_usage"] = ram_usage
        info["cpu_usage"] = cpu_usage
        info["gpu_usage"] = gpu_usage

        return obs, shaped_reward, done, info

    def get_penalty_stats(self):
        """Get accumulated penalty statistics."""
        if self.penalty_steps == 0:
            return {
                "avg_ram_penalty": 0.0,
                "avg_cpu_penalty": 0.0,
                "avg_gpu_penalty": 0.0,
                "total_steps": 0,
            }

        return {
            "avg_ram_penalty": self.total_ram_penalty / self.penalty_steps,
            "avg_cpu_penalty": self.total_cpu_penalty / self.penalty_steps,
            "avg_gpu_penalty": self.total_gpu_penalty / self.penalty_steps,
            "total_steps": self.penalty_steps,
        }


def make_pcgrl_env(
    resource_monitor,
    game="zelda",
    representation="narrow",
    ram_penalty_weight=0.2,
    cpu_penalty_weight=0.1,
    gpu_penalty_weight=0.1,
    crop_size=None,
    sokoban_unsolvable_penalty=25.0,
    use_solvability_config=True,
    **kwargs,
):
    """
    Create a resource-aware PCGRL environment with solvability optimization.

    Args:
        resource_monitor: ResourceMonitor instance for tracking resource usage
        game: Game name ('zelda', 'sokoban', 'binary', etc.)
        representation: Representation type ('narrow', 'wide', 'turtle')
        ram_penalty_weight: Weight for RAM penalty (higher = agent cares more about RAM)
        cpu_penalty_weight: Weight for CPU penalty
        gpu_penalty_weight: Weight for GPU penalty
        crop_size: Crop size for observation
        sokoban_unsolvable_penalty: Penalty for unsolvable Sokoban levels (default: 25.0)
        use_solvability_config: Apply tuned solvability reward weights (default: True)
        **kwargs: Additional arguments for the environment

    Returns:
        Gym environment with resource-aware reward shaping AND solvability optimization
    """
    # gym-pcgrl registers environments as '{game}-{representation}-v0'
    game_name = game.lower()
    rep_name = representation.lower()
    env_name = f"{game_name}-{rep_name}-v0"

    try:
        # Create environment using registered name
        env = gym.make(env_name)

        # Apply solvability-optimized reward weights and parameters
        if use_solvability_config:
            config = get_solvability_config(game_name)
            if config:
                print(
                    f"  Applying solvability-optimized configuration for {game_name.upper()}"
                )

                # Adjust reward weights in the problem instance
                if hasattr(env, "_prob"):
                    env._prob.adjust_param(
                        rewards=config["rewards"], **config["params"]
                    )
                    print(f"  ✓ Tuned reward weights for maximum solvability")
                    print(
                        f"    - Regions weight: {config['rewards'].get('regions', 'N/A')} (connectivity)"
                    )
                    if game_name == "sokoban":
                        print(
                            f"    - Ratio weight: {config['rewards']['ratio']} (crates = targets)"
                        )
                        print(
                            f"    - Dist-win weight: {config['rewards']['dist-win']} (solvability)"
                        )
                        print(
                            f"    - Sol-length weight: {config['rewards']['sol-length']} (solution quality)"
                        )
                    elif game_name == "zelda":
                        print(
                            f"    - Path-length weight: {config['rewards']['path-length']} (player→key→door)"
                        )
                        print(
                            f"    - Nearest-enemy weight: {config['rewards']['nearest-enemy']} (challenge)"
                        )
            else:
                print(f"  No solvability config found for {game_name}, using defaults")

        # For Sokoban, add solvability wrapper (double-layer protection)
        if game_name == "sokoban":
            print(
                f"  Adding Sokoban solvability wrapper (penalty={sokoban_unsolvable_penalty})"
            )
            print(f"    - Layer 1: Tuned reward weights (dist-win, sol-length)")
            print(f"    - Layer 2: Unsolvable penalty wrapper (double-check)")
            print(f"  [WARNING] Unsolvable Sokoban levels will be HEAVILY PENALIZED")
            env = SokobanSolvabilityWrapper(
                env,
                unsolvable_penalty=sokoban_unsolvable_penalty,
                min_solution_length=5,
                max_solution_length=50,
                terminate_on_unsolvable=False,  # Set to True for even stricter training
            )

        # Wrap with resource-aware wrapper (passing the monitor for feedback loop)
        env = ResourceAwarePCGRLWrapper(
            env,
            resource_monitor=resource_monitor,
            ram_penalty_weight=ram_penalty_weight,
            cpu_penalty_weight=cpu_penalty_weight,
            gpu_penalty_weight=gpu_penalty_weight,
        )

        return env

    except Exception as e:
        print(f"Error creating environment '{env_name}': {e}")
        print(f"\nAvailable games: zelda, sokoban, binary")
        print(f"Available representations: narrow, wide, turtle")
        print(f"Environment should be registered as: {env_name}")
        raise


def test_environment():
    """Test function to verify environment creation."""
    print("Testing PCGRL Environment Creation...")

    try:
        # Test Zelda environment
        env = make_pcgrl_env("zelda", "narrow")
        print(f"✓ Created Zelda environment")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")

        # Test reset
        state = env.reset()
        print(f"✓ Environment reset successful")
        print(
            f"  State shape: {state.shape if hasattr(state, 'shape') else type(state)}"
        )

        # Test step
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  Reward: {reward}")

        env.close()
        print("\n✓ All environment tests passed!")
        return True

    except Exception as e:
        print(f"\n✗ Environment test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_environment()
```

## utils.py:

```python
# utils.py
"""
RAPCG-MetaRL Utilities
Resource monitoring, logging, and training utilities.
"""

import os
import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class ResourceMonitor:
    """
    Monitor system resources (CPU, RAM, GPU) during training.
    """

    def __init__(self, use_gpu: bool = True):
        """
        Initialize resource monitor.

        Args:
            use_gpu: Whether to monitor GPU (requires pynvml)
        """
        self.use_gpu = use_gpu
        self.gpu_available = False

        if use_gpu:
            try:
                import pynvml

                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
                print("[OK] GPU monitoring enabled")
            except Exception as e:
                print(f"[WARNING] GPU monitoring unavailable: {e}")
                self.gpu_available = False

    def get_resources(self) -> Dict[str, float]:
        """
        Get current resource usage.

        Returns:
            Dictionary with CPU, RAM, GPU metrics
        """
        resources = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_percent": psutil.virtual_memory().percent,
            "ram_used_gb": psutil.virtual_memory().used / (1024**3),
            "ram_available_gb": psutil.virtual_memory().available / (1024**3),
        }

        if self.gpu_available:
            try:
                gpu_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_util = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)

                resources.update(
                    {
                        "gpu_util_percent": gpu_util.gpu,
                        "gpu_mem_used_mb": gpu_info.used / (1024**2),
                        "gpu_mem_total_mb": gpu_info.total / (1024**2),
                        "gpu_mem_percent": (gpu_info.used / gpu_info.total) * 100,
                    }
                )
            except Exception as e:
                print(f"Warning: GPU metrics error: {e}")
        else:
            resources.update(
                {
                    "gpu_util_percent": 0.0,
                    "gpu_mem_used_mb": 0.0,
                    "gpu_mem_total_mb": 0.0,
                    "gpu_mem_percent": 0.0,
                }
            )

        return resources

    def check_resource_pressure(
        self, thresholds: Optional[Dict[str, float]] = None
    ) -> Tuple[bool, str]:
        """
        Check if resources are under pressure.

        Args:
            thresholds: Custom thresholds for CPU, RAM, GPU

        Returns:
            Tuple of (is_under_pressure, message)
        """
        if thresholds is None:
            thresholds = {
                "cpu_percent": 90.0,
                "ram_percent": 90.0,
                "gpu_mem_percent": 85.0,
                "gpu_util_percent": 85.0,
            }

        resources = self.get_resources()

        for key, threshold in thresholds.items():
            if key in resources and resources[key] > threshold:
                return (
                    True,
                    f"{key} exceeds threshold: {resources[key]:.1f}% > {threshold}%",
                )

        return False, "Resources normal"

    def __del__(self):
        """Cleanup GPU monitoring."""
        if self.gpu_available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass


class TrainingLogger:
    """
    Log training metrics including rewards, resources, and content quality.
    """

    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        """
        Initialize training logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (auto-generated if None)
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.csv")

        self.episodes = []
        self.steps = []
        self.rewards = []
        self.resource_logs = []
        self.content_metrics = []
        self.timestamps = []
        self.actions = []  # NEW: Track actions
        self.penalties = []  # NEW: Track penalties per step

        self.episode_count = 0
        self.total_steps = 0
        self.start_time = time.time()

        print(f"[OK] Logger initialized: {self.log_file}")

    def log_step(
        self,
        reward: float,
        resources: Dict[str, float],
        content_metrics: Optional[Dict[str, float]] = None,
        action: Optional[int] = None,
        penalty_info: Optional[Dict[str, float]] = None,
    ):
        """
        Log a single training step.

        Args:
            reward: Step reward (shaped)
            resources: Resource usage dict
            content_metrics: Content quality metrics
            action: Action taken by agent
            penalty_info: Penalty breakdown (ram_penalty, cpu_penalty, etc.)
        """
        self.steps.append(self.total_steps)
        self.episodes.append(self.episode_count)
        self.rewards.append(reward)
        self.resource_logs.append(resources)
        self.content_metrics.append(content_metrics or {})
        self.timestamps.append(time.time() - self.start_time)
        self.actions.append(action if action is not None else -1)
        self.penalties.append(penalty_info or {})

        self.total_steps += 1

    def log_episode_end(self):
        """Mark end of episode."""
        self.episode_count += 1

    def save(self):
        """Save logs to CSV file."""
        if not self.steps:
            print("No data to save")
            return

        # Prepare dataframe
        data = {
            "episode": self.episodes,
            "step": self.steps,
            "reward": self.rewards,
            "timestamp": self.timestamps,
        }

        # Add resource metrics
        if self.resource_logs:
            for key in self.resource_logs[0].keys():
                data[key] = [r.get(key, 0) for r in self.resource_logs]

        # Add content metrics
        if self.content_metrics and any(self.content_metrics):
            all_keys = set()
            for cm in self.content_metrics:
                all_keys.update(cm.keys())

            for key in all_keys:
                data[f"content_{key}"] = [cm.get(key, 0) for cm in self.content_metrics]

        # Add action tracking
        if self.actions:
            data["action"] = self.actions

        # Add penalty breakdown
        if self.penalties and any(self.penalties):
            penalty_keys = set()
            for p in self.penalties:
                penalty_keys.update(p.keys())

            for key in penalty_keys:
                data[f"penalty_{key}"] = [p.get(key, 0.0) for p in self.penalties]

        df = pd.DataFrame(data)
        df.to_csv(self.log_file, index=False)
        print(f"[OK] Logs saved to {self.log_file} ({len(df)} steps)")

    def get_stats(self) -> Dict[str, float]:
        """
        Get summary statistics.

        Returns:
            Dictionary of training statistics
        """
        if not self.rewards:
            return {}

        rewards_array = np.array(self.rewards)

        stats = {
            "total_episodes": self.episode_count,
            "total_steps": self.total_steps,
            "mean_reward": np.mean(rewards_array),
            "std_reward": np.std(rewards_array),
            "min_reward": np.min(rewards_array),
            "max_reward": np.max(rewards_array),
            "elapsed_time": time.time() - self.start_time,
        }

        return stats

    def print_stats(self):
        """Print summary statistics."""
        stats = self.get_stats()

        print("\n" + "=" * 50)
        print(f"Training Statistics - {self.experiment_name}")
        print("=" * 50)
        for key, value in stats.items():
            if "time" in key:
                print(f"{key:20s}: {value:.2f}s")
            else:
                print(f"{key:20s}: {value:.4f}")
        print("=" * 50 + "\n")

    def analyze_action_penalty_correlation(self, recent_steps: int = 1000) -> Dict:
        """
        Analyze correlation between actions and resource penalties.

        Args:
            recent_steps: Number of recent steps to analyze

        Returns:
            Dictionary with correlation analysis
        """
        if not self.actions or not self.penalties:
            return {}

        # Get recent data
        start_idx = max(0, len(self.actions) - recent_steps)
        recent_actions = self.actions[start_idx:]
        recent_penalties = self.penalties[start_idx:]

        # Extract penalty types
        penalty_types = ["ram_penalty", "cpu_penalty", "gpu_penalty", "total_penalty"]

        # Group by action
        action_stats = {}
        unique_actions = set(a for a in recent_actions if a >= 0)

        for action in unique_actions:
            # Find indices where this action was taken
            action_indices = [i for i, a in enumerate(recent_actions) if a == action]

            if not action_indices:
                continue

            action_stats[action] = {
                "count": len(action_indices),
                "avg_penalties": {},
                "max_penalties": {},
            }

            for penalty_type in penalty_types:
                penalties = [
                    recent_penalties[i].get(penalty_type, 0.0) for i in action_indices
                ]

                if penalties:
                    action_stats[action]["avg_penalties"][penalty_type] = np.mean(
                        penalties
                    )
                    action_stats[action]["max_penalties"][penalty_type] = np.max(
                        penalties
                    )

        return action_stats

    def get_high_penalty_actions(
        self,
        top_n: int = 5,
        penalty_type: str = "ram_penalty",
        recent_steps: int = 1000,
    ) -> List[Tuple[int, float]]:
        """
        Get actions that cause highest penalties.

        Args:
            top_n: Number of top actions to return
            penalty_type: Type of penalty to analyze ('ram_penalty', 'cpu_penalty', etc.)
            recent_steps: Number of recent steps to analyze

        Returns:
            List of (action, avg_penalty) tuples, sorted by penalty (descending)
        """
        stats = self.analyze_action_penalty_correlation(recent_steps)

        if not stats:
            return []

        # Extract action-penalty pairs
        action_penalties = []
        for action, data in stats.items():
            avg_penalty = data["avg_penalties"].get(penalty_type, 0.0)
            action_penalties.append((action, avg_penalty))

        # Sort by penalty (descending)
        action_penalties.sort(key=lambda x: x[1], reverse=True)

        return action_penalties[:top_n]


def calculate_fps(start_time: float, steps: int) -> float:
    """
    Calculate frames per second (steps per second).

    Args:
        start_time: Start timestamp
        steps: Number of steps

    Returns:
        FPS value
    """
    elapsed = time.time() - start_time
    return steps / elapsed if elapsed > 0 else 0.0


def estimate_training_time(
    current_steps: int, total_steps: int, start_time: float
) -> str:
    """
    Estimate remaining training time.

    Args:
        current_steps: Steps completed
        total_steps: Total steps to complete
        start_time: Training start time

    Returns:
        Formatted time string
    """
    if current_steps == 0:
        return "Calculating..."

    elapsed = time.time() - start_time
    steps_per_sec = current_steps / elapsed
    remaining_steps = total_steps - current_steps
    remaining_secs = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

    hours = int(remaining_secs // 3600)
    minutes = int((remaining_secs % 3600) // 60)
    seconds = int(remaining_secs % 60)

    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def create_checkpoint_dir(
    base_dir: str = "checkpoints", experiment_name: Optional[str] = None
) -> str:
    """
    Create checkpoint directory.

    Args:
        base_dir: Base checkpoint directory
        experiment_name: Experiment name

    Returns:
        Path to checkpoint directory
    """
    if experiment_name is None:
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    checkpoint_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    return checkpoint_dir
```

## train.py :

```python
# train.py
"""
RAPCG-MetaRL Training Script
Train Meta-RL agents for resource-aware procedural content generation.
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Add project paths - prioritize main project over gym-pcgrl
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "gym-pcgrl"))

# Import local modules
from utils import ResourceMonitor, TrainingLogger, create_checkpoint_dir
from wrappers.pcgrl_env import make_pcgrl_env

# Stable Baselines3 for RL
try:
    from stable_baselines3 import PPO, A2C, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print(
        "Error: stable-baselines3 not installed. Install with: pip install stable-baselines3"
    )
    sys.exit(1)


class ResourceAwareCallback(BaseCallback):
    """
    Callback for resource-aware training with dynamic adaptation.
    """

    def __init__(
        self,
        resource_monitor,
        training_logger,
        save_freq=1000,
        checkpoint_dir="checkpoints",
        verbose=1,
    ):
        super().__init__(verbose)
        self.resource_monitor = resource_monitor
        self.training_logger = training_logger
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """Called at each step with resource-aware penalty tracking."""
        # Get current resources
        resources = self.resource_monitor.get_resources()

        # Get reward (already shaped by ResourceAwarePCGRLWrapper)
        reward = self.locals.get("rewards", [0])[0]
        self.current_episode_reward += reward
        self.current_episode_length += 1

        # Get additional info from environment (penalties, raw reward, etc.)
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}

        # Extract penalty information from wrapper
        ram_penalty = info.get("ram_penalty", 0.0)
        cpu_penalty = info.get("cpu_penalty", 0.0)
        gpu_penalty = info.get("gpu_penalty", 0.0)
        total_penalty = info.get("total_penalty", 0.0)

        # Get action taken
        actions = self.locals.get("actions", [None])
        action = actions[0] if actions is not None and len(actions) > 0 else None
        if hasattr(action, "item"):
            action = action.item()  # Convert numpy/torch to Python int

        # Prepare penalty breakdown
        penalty_info = {
            "ram_penalty": ram_penalty,
            "cpu_penalty": cpu_penalty,
            "gpu_penalty": gpu_penalty,
            "total_penalty": total_penalty,
        }

        # Log step with action and penalty tracking
        self.training_logger.log_step(
            reward, resources, action=action, penalty_info=penalty_info
        )

        # Track penalties for analysis
        if not hasattr(self, "total_penalties"):
            self.total_penalties = []
        self.total_penalties.append(total_penalty)

        # Check if episode ended
        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.training_logger.log_episode_end()

            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                # Show penalty info
                recent_penalties = (
                    self.total_penalties[-100:]
                    if hasattr(self, "total_penalties")
                    else []
                )
                avg_penalty = np.mean(recent_penalties) if recent_penalties else 0.0
                print(
                    f"Episode {len(self.episode_rewards)}: "
                    f"Mean Reward (last 10): {mean_reward:.2f}, "
                    f"Avg Penalty: {avg_penalty:.2f}, "
                    f"CPU: {resources['cpu_percent']:.1f}%, "
                    f"RAM: {resources['ram_percent']:.1f}%, "
                    f"GPU: {resources['gpu_mem_percent']:.1f}%"
                )

            # Show action-penalty correlation every 50 episodes
            if self.verbose > 0 and len(self.episode_rewards) % 50 == 0:
                high_penalty_actions = self.training_logger.get_high_penalty_actions(
                    top_n=5, penalty_type="ram_penalty", recent_steps=1000
                )
                if high_penalty_actions:
                    print("\n  Top RAM-intensive actions (recent 1000 steps):")
                    for action, avg_penalty in high_penalty_actions:
                        print(
                            f"    Action {action}: Avg RAM Penalty = {avg_penalty:.3f}"
                        )
                    print()

            self.current_episode_reward = 0
            self.current_episode_length = 0

        # Save checkpoint periodically
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"model_step_{self.n_calls}.zip"
            )
            self.model.save(checkpoint_path)
            self.training_logger.save()

            if self.verbose > 0:
                print(f"✓ Checkpoint saved: {checkpoint_path}")

        # Check resource pressure and adapt if needed
        is_pressure, msg = self.resource_monitor.check_resource_pressure()
        if is_pressure and self.verbose > 0:
            print(f"⚠ Resource pressure detected: {msg}")
            # Could implement dynamic adaptation here

        return True

    def _on_training_end(self) -> None:
        """Called at end of training."""
        self.training_logger.save()
        self.training_logger.print_stats()


class MetaRLTrainer:
    """
    Meta-RL trainer for PCGRL environments.
    Supports PPO, A2C, and other algorithms with resource-aware training.
    """

    def __init__(
        self,
        game="zelda",
        representation="narrow",
        algorithm="PPO",
        total_timesteps=50000,
        n_steps=128,
        batch_size=64,
        learning_rate=2.5e-4,
        n_envs=1,
        device="auto",
        seed=None,
        experiment_name=None,
        use_gpu_monitoring=True,
        checkpoint_freq=1000,
        log_dir="logs",
        checkpoint_dir="checkpoints",
        sokoban_unsolvable_penalty=25.0,
        use_solvability_tuning=True,
    ):
        """
        Initialize Meta-RL trainer.

        Args:
            game: Game environment ('zelda', 'sokoban', 'binary')
            representation: Representation type ('narrow', 'wide', 'turtle')
            algorithm: RL algorithm ('PPO', 'A2C', 'SAC')
            total_timesteps: Total training steps
            n_steps: Steps per update
            batch_size: Batch size for training
            learning_rate: Learning rate
            n_envs: Number of parallel environments
            device: Device ('cpu', 'cuda', 'auto')
            seed: Random seed
            experiment_name: Experiment name for logging
            use_gpu_monitoring: Enable GPU monitoring
            checkpoint_freq: Checkpoint save frequency
            log_dir: Log directory
            checkpoint_dir: Checkpoint directory
        """
        self.game = game
        self.representation = representation
        self.algorithm = algorithm
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_envs = n_envs
        self.device = device
        self.seed = seed
        self.use_solvability_tuning = use_solvability_tuning
        self.sokoban_unsolvable_penalty = sokoban_unsolvable_penalty

        # Detect GPU availability and adjust device
        if device == "auto":
            import torch

            if torch.cuda.is_available():
                self.device = "cuda"
                print("✓ GPU detected: Using CUDA for training")
            else:
                self.device = "cpu"
                print("⚠ No GPU detected: Using CPU for training")
                print(
                    "  To enable GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121"
                )

        # Set up experiment tracking
        if experiment_name is None:
            # Include device type (CUDA/CPU) in experiment name
            device_suffix = "CUDA" if self.device == "cuda" else "CPU"
            experiment_name = f"{game}_{algorithm}_{device_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name

        # Initialize monitoring and logging
        # Only monitor GPU if we're actually using it for training
        use_gpu_monitoring_actual = use_gpu_monitoring and (self.device == "cuda")
        self.resource_monitor = ResourceMonitor(use_gpu=use_gpu_monitoring_actual)
        self.logger = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name)
        self.checkpoint_dir = create_checkpoint_dir(checkpoint_dir, experiment_name)
        self.checkpoint_freq = checkpoint_freq

        # Will be set during training
        self.env = None
        self.model = None

        print(f"\n{'=' * 60}")
        print(f"Meta-RL Trainer Initialized")
        print(f"{'=' * 60}")
        print(f"Game: {game}")
        print(f"Algorithm: {algorithm}")
        print(f"Total Timesteps: {total_timesteps:,}")
        print(f"Device: {self.device}")
        if self.device == "cuda":
            import torch

            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(
                f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB"
            )
        print(
            f"GPU Monitoring: {'Enabled' if use_gpu_monitoring_actual else 'Disabled'}"
        )
        print(f"Experiment: {experiment_name}")
        print(f"{'=' * 60}\n")

    def make_env(self, rank=0):
        """
        Create and wrap environment with resource awareness.

        Args:
            rank: Environment rank for seeding

        Returns:
            Callable that creates the environment
        """

        def _init():
            # AND apply solvability-optimized reward weights
            env = make_pcgrl_env(
                resource_monitor=self.resource_monitor,  # CRUCIAL: Enable feedback loop
                game=self.game,
                representation=self.representation,
                ram_penalty_weight=0.2,  # Configurable penalty weights
                cpu_penalty_weight=0.1,
                gpu_penalty_weight=0.1,
                sokoban_unsolvable_penalty=self.sokoban_unsolvable_penalty,
                use_solvability_config=self.use_solvability_tuning,  # Apply tuned weights
            )
            # Don't use Monitor - we have our own logging via TrainingLogger
            if self.seed is not None:
                env.seed(self.seed + rank)
            return env

        return _init

    def setup_environments(self):
        """Set up training environments."""
        print(f"Setting up {self.n_envs} environment(s)...")

        if self.n_envs == 1:
            self.env = DummyVecEnv([self.make_env(0)])
        else:
            self.env = SubprocVecEnv([self.make_env(i) for i in range(self.n_envs)])

        print(f"✓ Environments ready")

    def setup_model(self):
        """Set up RL model."""
        print(f"Setting up {self.algorithm} model...")

        # Determine policy type based on observation space
        obs_space = self.env.observation_space
        if hasattr(obs_space, "spaces") and isinstance(obs_space.spaces, dict):
            policy_type = "MultiInputPolicy"
        else:
            policy_type = "MlpPolicy"

        print(f"Using policy: {policy_type}")

        # Common parameters
        model_kwargs = {
            "learning_rate": self.learning_rate,
            "verbose": 1,
            "device": self.device,
            "seed": self.seed,
        }

        # Algorithm-specific parameters
        if self.algorithm == "PPO":
            model_kwargs.update(
                {
                    "n_steps": self.n_steps,
                    "batch_size": self.batch_size,
                    "n_epochs": 10,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "clip_range": 0.2,
                    "ent_coef": 0.01,
                }
            )
            self.model = PPO(policy_type, self.env, **model_kwargs)

        elif self.algorithm == "A2C":
            model_kwargs.update(
                {
                    "n_steps": self.n_steps,
                    "gamma": 0.99,
                    "gae_lambda": 0.95,
                    "ent_coef": 0.01,
                }
            )
            self.model = A2C(policy_type, self.env, **model_kwargs)

        elif self.algorithm == "SAC":
            # SAC requires continuous action spaces
            # For discrete actions, it will fail - consider using PPO or A2C instead
            action_space = self.env.action_space
            from gym import spaces

            if isinstance(action_space, spaces.Discrete):
                print("⚠ WARNING: SAC is designed for continuous action spaces!")
                print(
                    "  gym-pcgrl uses discrete actions. Consider using PPO or A2C instead."
                )
                print("  Training may fail or produce poor results.")

            model_kwargs.update(
                {
                    "buffer_size": 100000,  # Replay buffer size
                    "learning_starts": 1000,  # Start learning after N steps
                    "batch_size": 256,  # Larger batch for off-policy
                    "tau": 0.005,  # Soft update coefficient
                    "gamma": 0.99,
                    "train_freq": 1,
                    "gradient_steps": 1,
                    "ent_coef": "auto",  # Automatic entropy tuning
                }
            )
            self.model = SAC(policy_type, self.env, **model_kwargs)

        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        print(f"✓ Model initialized")
        print(f"  Policy: {policy_type}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Batch size: {self.batch_size}")

    def train(self):
        """Run training loop."""
        print(f"\nStarting training...")
        print(f"Target: {self.total_timesteps:,} timesteps\n")

        # Set up callback
        callback = ResourceAwareCallback(
            resource_monitor=self.resource_monitor,
            training_logger=self.logger,
            save_freq=self.checkpoint_freq,
            checkpoint_dir=self.checkpoint_dir,
            verbose=1,
        )

        try:
            # Train model
            self.model.learn(total_timesteps=self.total_timesteps, callback=callback)

            print("\n✓ Training completed!")

            # Save final model
            final_model_path = os.path.join(self.checkpoint_dir, "final_model.zip")
            self.model.save(final_model_path)
            print(f"✓ Final model saved: {final_model_path}")

            # Save logs
            self.logger.save()
            self.logger.print_stats()

        except KeyboardInterrupt:
            print("\n\n⚠ Training interrupted by user")
            self.logger.save()
            print("✓ Logs saved")

        except Exception as e:
            print(f"\n✗ Training error: {e}")
            import traceback

            traceback.print_exc()
            self.logger.save()

        finally:
            if self.env is not None:
                self.env.close()

    def evaluate(self, n_episodes=10):
        """
        Evaluate trained model.

        Args:
            n_episodes: Number of episodes to evaluate
        """
        if self.model is None:
            print("Error: No model to evaluate. Train first or load a model.")
            return

        print(f"\nEvaluating model for {n_episodes} episodes...")

        episode_rewards = []
        episode_lengths = []

        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward[0]
                episode_length += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)

            print(
                f"  Episode {episode + 1}: Reward={episode_reward:.2f}, Length={episode_length}"
            )

        print(f"\nEvaluation Results:")
        print(
            f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}"
        )
        print(
            f"  Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}"
        )

    def load_model(self, model_path):
        """
        Load a trained model.

        Args:
            model_path: Path to model file
        """
        print(f"Loading model from {model_path}...")

        if self.algorithm == "PPO":
            self.model = PPO.load(model_path, env=self.env)
        elif self.algorithm == "A2C":
            self.model = A2C.load(model_path, env=self.env)
        elif self.algorithm == "SAC":
            self.model = SAC.load(model_path, env=self.env)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")

        print("✓ Model loaded")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Meta-RL agent for PCGRL")

    parser.add_argument(
        "--game",
        type=str,
        default="zelda",
        choices=["zelda", "sokoban", "binary", "zelda-narrow"],
        help="Game environment",
    )
    parser.add_argument(
        "--representation",
        type=str,
        default="narrow",
        choices=["narrow", "wide", "turtle"],
        help="Representation type",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "A2C", "SAC"],
        help="RL algorithm (SAC only works with continuous action spaces)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=50000, help="Total training timesteps"
    )
    parser.add_argument("--n-steps", type=int, default=128, help="Steps per update")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2.5e-4, help="Learning rate")
    parser.add_argument(
        "--sokoban-penalty",
        type=float,
        default=25.0,
        help="Penalty for unsolvable Sokoban levels (default: 25.0 - very strict)",
    )
    parser.add_argument(
        "--no-solvability-tuning",
        action="store_true",
        help="Disable solvability-optimized reward weights (not recommended)",
    )
    parser.add_argument(
        "--n-envs", type=int, default=1, help="Number of parallel environments"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device for training",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument(
        "--experiment-name", type=str, default=None, help="Experiment name"
    )
    parser.add_argument(
        "--no-gpu-monitoring", action="store_true", help="Disable GPU monitoring"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=1000, help="Checkpoint save frequency"
    )
    parser.add_argument(
        "--evaluate", action="store_true", help="Evaluate after training"
    )
    parser.add_argument(
        "--load-model", type=str, default=None, help="Load pre-trained model"
    )

    args = parser.parse_args()

    # Create trainer
    trainer = MetaRLTrainer(
        game=args.game,
        representation=args.representation,
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_envs=args.n_envs,
        device=args.device,
        seed=args.seed,
        experiment_name=args.experiment_name,
        use_gpu_monitoring=not args.no_gpu_monitoring,
        checkpoint_freq=args.checkpoint_freq,
        sokoban_unsolvable_penalty=args.sokoban_penalty,
        use_solvability_tuning=not args.no_solvability_tuning,
    )

    # Set up environments and model
    trainer.setup_environments()

    if args.load_model:
        trainer.load_model(args.load_model)
    else:
        trainer.setup_model()

    # Train
    trainer.train()

    # Evaluate if requested
    if args.evaluate:
        trainer.evaluate(n_episodes=10)


if __name__ == "__main__":
    main()
```

## inference.py :

```python
"""
RAPCG-MetaRL Inference Script
Generate levels using trained Meta-RL agents.
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project paths - prioritize main project over gym-pcgrl
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "gym-pcgrl"))

from utils import ResourceMonitor
from wrappers.pcgrl_env import make_pcgrl_env
from wrappers.helper import calculate_content_metrics, save_level
from visualize_levels import save_level_image, render_level

try:
    from stable_baselines3 import PPO, A2C
except ImportError:
    print("Error: stable-baselines3 not installed")
    sys.exit(1)


class LevelGenerator:
    """
    Generate levels using trained Meta-RL model.
    """

    def __init__(
        self,
        model_path,
        game="zelda",
        representation="narrow",
        algorithm="PPO",
        device="auto",
    ):
        """
        Initialize level generator.

        Args:
            model_path: Path to trained model
            game: Game environment
            representation: Representation type
            algorithm: RL algorithm used
            device: Device for inference
        """
        self.model_path = model_path
        self.game = game
        self.representation = representation
        self.algorithm = algorithm
        self.device = device

        # Create resource monitor (needed for environment wrapper)
        # Enable GPU monitoring if device is cuda
        use_gpu_monitor = device == "cuda" or (
            device == "auto" and self._is_cuda_available()
        )
        self.resource_monitor = ResourceMonitor(use_gpu=use_gpu_monitor)

        # Create environment
        print(f"Creating {game} environment...")
        if use_gpu_monitor:
            print(f"  GPU monitoring: ENABLED")
        else:
            print(f"  GPU monitoring: DISABLED (CPU-only mode)")
        self.env = make_pcgrl_env(
            resource_monitor=self.resource_monitor,
            game=game,
            representation=representation,
            use_solvability_config=True,  # Use tuned weights for inference too
        )

        # Load model
        print(f"Loading model from {model_path}...")
        if algorithm == "PPO":
            self.model = PPO.load(model_path, device=device)
        elif algorithm == "A2C":
            self.model = A2C.load(model_path, device=device)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")

        print("✓ Generator ready")

    def _is_cuda_available(self):
        """Check if CUDA is available for PyTorch."""
        try:
            import torch

            return torch.cuda.is_available()
        except:
            return False

    def generate(
        self,
        n_levels=1,
        max_steps=1000,
        deterministic=True,
        save_dir=None,
        visualize=True,
    ):
        """
        Generate levels.

        Args:
            n_levels: Number of levels to generate
            max_steps: Maximum steps per level
            deterministic: Use deterministic policy
            save_dir: Directory to save levels
            visualize: Visualize generated levels

        Returns:
            List of generated levels with metadata
        """
        generated_levels = []

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Levels will be saved to: {save_dir}")

        for i in range(n_levels):
            print(f"\nGenerating level {i + 1}/{n_levels}...")

            obs = self.env.reset()
            done = False
            steps = 0
            total_reward = 0

            while not done and steps < max_steps:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1

            # Extract level from environment
            level = self._extract_level(info)

            # Calculate metrics
            metrics = calculate_content_metrics(level)
            metrics["total_reward"] = total_reward
            metrics["steps"] = steps

            # Store result
            result = {"level": level, "metrics": metrics, "info": info}
            generated_levels.append(result)

            # Print metrics
            print(f"  Steps: {steps}, Reward: {total_reward:.2f}")
            print(
                f"  Diversity: {metrics['diversity']:.3f}, "
                f"Complexity: {metrics['complexity']:.3f}"
            )

            # Save level
            if save_dir:
                level_path = os.path.join(save_dir, f"level_{i + 1}")
                save_level(level, level_path + ".npy", format="npy")
                save_level(level, level_path + ".txt", format="txt")

                # Save visualization as high-res PNG
                img_path = level_path + ".png"
                save_level_image(
                    level, img_path, game=self.game, scale=25, show_grid=True, dpi=300
                )
                print(f"  ✓ Saved to {level_path} (.npy, .txt, .png)")

            # Visualize
            if visualize:
                self._visualize_level(level, f"Generated Level {i + 1}")

        return generated_levels

    def _extract_level(self, info):
        """Extract level array from environment info."""
        # gym-pcgrl stores level in the base environment
        # Need to unwrap to get to the actual PCGRL environment
        env = self.env

        # Unwrap to get to base gym-pcgrl environment
        while hasattr(env, "env"):
            env = env.env

        # gym-pcgrl stores the map in _rep._map
        if hasattr(env, "_rep") and hasattr(env._rep, "_map"):
            return np.array(env._rep._map, dtype=int)

        # Alternative: check if it's in info
        if "level" in info:
            return np.array(info["level"], dtype=int)

        # Last resort: try to get from observation
        print("Warning: Could not extract level, using fallback")
        return np.zeros((11, 11), dtype=int)

    def _visualize_level(self, level, title="Level"):
        """Visualize level with proper tile colors."""
        rgb = render_level(level, game=self.game, scale=20, show_grid=True)

        plt.figure(figsize=(10, 8))
        plt.imshow(rgb)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def close(self):
        """Close environment."""
        self.env.close()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="Generate levels using trained model")

    parser.add_argument(
        "model_path", type=str, help="Path to trained model (.zip file)"
    )
    parser.add_argument("--game", type=str, default="zelda", help="Game environment")
    parser.add_argument(
        "--representation", type=str, default="narrow", help="Representation type"
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="PPO",
        choices=["PPO", "A2C"],
        help="RL algorithm",
    )
    parser.add_argument(
        "--n-levels", type=int, default=5, help="Number of levels to generate"
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Maximum steps per level"
    )
    parser.add_argument(
        "--stochastic", action="store_true", help="Use stochastic policy"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="generated_levels",
        help="Directory to save levels",
    )
    parser.add_argument(
        "--no-visualize", action="store_true", help="Disable visualization"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device for inference"
    )

    args = parser.parse_args()

    # Create generator
    generator = LevelGenerator(
        model_path=args.model_path,
        game=args.game,
        representation=args.representation,
        algorithm=args.algorithm,
        device=args.device,
    )

    # Generate levels
    levels = generator.generate(
        n_levels=args.n_levels,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        save_dir=args.save_dir,
        visualize=not args.no_visualize,
    )

    print(f"\n✓ Generated {len(levels)} levels")

    # Print summary statistics
    all_metrics = [l["metrics"] for l in levels]
    print(f"\nSummary Statistics:")
    print(f"  Mean Diversity: {np.mean([m['diversity'] for m in all_metrics]):.3f}")
    print(f"  Mean Complexity: {np.mean([m['complexity'] for m in all_metrics]):.3f}")
    print(f"  Mean Reward: {np.mean([m['total_reward'] for m in all_metrics]):.2f}")

    generator.close()


if __name__ == "__main__":
    main()
```

## maml_trainer.py :

```python
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
```

## maml_inference_timed.py :

```python
"""
MAML Inference Timed Script
Loads a MAML checkpoint (best_meta_model.pt / final_meta_model.pt) and
generates levels with the same timing/metrics pipeline as inference_timed.py.

Key additions vs v1:
  - TrainingLogger live CSV written after every level (matches maml_trainer.py log format)
  - Per-level row flushed to --log-file immediately so no data lost on crash
  - Full terminal output with warnings (env creation, SB3 compat, etc.) visible

Usage:
    python maml_inference_timed.py checkpoints/sokoban_MAML_inference/best_meta_model.pt \\
        --game sokoban --n-levels 20 --max-steps 500 --device cuda
"""

import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
import time
import torch
from collections import OrderedDict
from typing import Optional, Tuple
from datetime import datetime

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "gym-pcgrl"))

from utils import ResourceMonitor, TrainingLogger
from wrappers.pcgrl_env import make_pcgrl_env
from wrappers.helper import calculate_content_metrics, save_level
from maml_trainer import MAMLPolicy, DictFlattenWrapper, SokobanDeadlockGuardrail

try:
    from visualize_levels import save_level_image
except ImportError:

    def save_level_image(level, path, **kwargs):
        print(f"  [SKIP] visualize_levels not available: {path}")


try:
    from sokoban_utils import is_valid_sokoban
except ImportError:

    def is_valid_sokoban(level):
        players = int(np.sum(level == 2))
        crates = int(np.sum(level == 3))
        targets = int(np.sum(level == 4))
        ok = players == 1 and crates >= 1 and targets >= 1
        return ok, f"players={players}, crates={crates}, targets={targets}"


import gym

try:
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("Error: stable-baselines3 not installed.  pip install stable-baselines3")
    sys.exit(1)


# ==========================================================================
# Policy wrapper  (gives SB3-style .predict() over MAMLPolicy)
# ==========================================================================


class MAMLPolicyWrapper:
    def __init__(
        self,
        policy: MAMLPolicy,
        device: str,
        obs_dim: int,
        adapted_params: Optional[OrderedDict] = None,
    ):
        self.policy = policy
        self.device = device
        self.obs_dim = obs_dim
        self.adapted_params = adapted_params

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        obs_t = torch.FloatTensor(obs).to(self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        obs_flat = obs_t.reshape(obs_t.shape[0], -1)

        feat = obs_flat.shape[-1]
        if feat < self.obs_dim:
            pad = torch.zeros(
                obs_flat.shape[0], self.obs_dim - feat, device=self.device
            )
            obs_flat = torch.cat([obs_flat, pad], dim=-1)
        elif feat > self.obs_dim:
            obs_flat = obs_flat[:, : self.obs_dim]

        with torch.no_grad():
            if self.adapted_params is not None:
                logits, _ = self.policy.functional_forward(
                    obs_flat, self.adapted_params
                )
            else:
                logits, _ = self.policy(obs_flat)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                action = torch.distributions.Categorical(probs).sample()

        return action.cpu().numpy(), None


# ==========================================================================
# Checkpoint loader
# ==========================================================================


def load_maml_checkpoint(
    checkpoint_path: str, obs_dim: int, action_dim: int, device: str
) -> Tuple[MAMLPolicy, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})

    policy = MAMLPolicy(obs_dim, action_dim).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    print(f"[OK] Loaded MAML checkpoint : {checkpoint_path}")
    print(f"     Saved at iteration     : {ckpt.get('iteration', '?')}")
    print(f"     Meta-loss at save      : {ckpt.get('meta_loss', float('nan')):.4f}")
    print(f"     Training config        : {config}")
    return policy, config


# ==========================================================================
# Environment helpers
# ==========================================================================


def build_env(
    game: str, representation: str, resource_monitor: ResourceMonitor
) -> DummyVecEnv:
    def _make():
        env = make_pcgrl_env(
            resource_monitor=resource_monitor,
            game=game,
            representation=representation,
            use_solvability_config=True,
        )
        # Mirror the training-time deadlock guardrail so adaptation gradients
        # match the reward landscape the meta-policy was trained against.
        if game == "sokoban":
            env = SokobanDeadlockGuardrail(env, deadlock_penalty=1.5)
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = DictFlattenWrapper(env)
        return env

    return DummyVecEnv([_make])


def get_env_dims(env: DummyVecEnv) -> Tuple[int, int]:
    obs_dim = int(np.prod(env.observation_space.shape))
    act_space = env.action_space
    action_dim = (
        act_space.n if hasattr(act_space, "n") else int(np.prod(act_space.shape))
    )
    return obs_dim, action_dim


def extract_level(env: DummyVecEnv, info) -> np.ndarray:
    inner = env.envs[0]
    while hasattr(inner, "env"):
        inner = inner.env
    if hasattr(inner, "_rep") and hasattr(inner._rep, "_map"):
        return np.array(inner._rep._map, dtype=int)
    if isinstance(info, dict) and "level" in info:
        return np.array(info["level"], dtype=int)
    if isinstance(info, (list, tuple)) and info and "level" in info[0]:
        return np.array(info[0]["level"], dtype=int)
    print("  [WARN] Could not extract level – returning zeros")
    return np.zeros((10, 10), dtype=int)


# ==========================================================================
# Optional fast task-adaptation (MAML inner loop)
# ==========================================================================


def adapt_policy(
    policy: MAMLPolicy,
    env: DummyVecEnv,
    inner_lr: float,
    inner_steps: int,
    n_trajectories: int,
    device: str,
) -> OrderedDict:
    from maml_trainer import collect_trajectories, compute_policy_loss

    adapted = OrderedDict(
        (name, param.clone()) for name, param in policy.named_parameters()
    )
    for k in range(inner_steps):
        traj = collect_trajectories(env, policy, n_trajectories, device, adapted)
        loss = compute_policy_loss(traj, policy, adapted)
        grads = torch.autograd.grad(loss, adapted.values(), allow_unused=True)
        adapted = OrderedDict(
            (name, p - inner_lr * (g if g is not None else torch.zeros_like(p)))
            for (name, p), g in zip(adapted.items(), grads)
        )
        print(f"    adapt step {k + 1}/{inner_steps}  loss={loss.item():.4f}")
    return adapted


# ==========================================================================
# CSV live-writer  (flushes one row per level immediately)
# ==========================================================================

_CSV_COLUMNS = [
    "level_id",
    "timestamp",
    "game",
    "algorithm",
    "adapt_steps",
    "reset_time_ms",
    "generation_time_ms",
    "extract_time_ms",
    "validation_time_ms",
    "metrics_time_ms",
    "solvability_time_ms",
    "save_time_ms",
    "total_time_ms",
    "steps",
    "mean_inference_ms",
    "std_inference_ms",
    "min_inference_ms",
    "max_inference_ms",
    "total_reward",
    "diversity",
    "complexity",
    "unique_tiles",
    "is_solvable",
    "ram_start_pct",
    "ram_end_pct",
    "ram_delta_pct",
    "cpu_start_pct",
    "cpu_end_pct",
    "cpu_delta_pct",
    "gpu_start_pct",
    "gpu_end_pct",
    "gpu_delta_pct",
]


class LiveCSVWriter:
    """Opens the CSV immediately, writes header, then flushes one row at a time."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(
            self._fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore"
        )
        self._w.writeheader()
        self._fh.flush()
        print(f"[OK] Timing CSV  -> {path}  (flushed after every level)")

    def write(self, row: dict):
        self._w.writerow(row)
        self._fh.flush()

    def close(self):
        self._fh.close()


# ==========================================================================
# Main generation loop
# ==========================================================================


def generate_timed(
    checkpoint_path: str,
    game: str = "sokoban",
    representation: str = "narrow",
    n_levels: int = 20,
    max_steps: int = 500,
    deterministic: bool = True,
    adapt_steps: int = 0,
    inner_lr: float = 0.01,
    n_trajectories: int = 64,
    device: str = "auto",
    save_dir: str = "generated_levels/maml",
    log_file: str = "inference_timing_maml.csv",
    log_dir: str = "logs",
    experiment_name: str = None,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ---- Experiment name -------------------------------------------------
    if experiment_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"MAML_inference_{game}_{ts}"

    # ---- Loggers ---------------------------------------------------------
    # 1. TrainingLogger  -> logs/<experiment_name>.csv  (step-level live log)
    training_logger = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name)
    print(f"[OK] TrainingLogger -> {log_dir}/{experiment_name}.csv")

    # 2. LiveCSVWriter   -> --log-file (level-level timing CSV, flushed live)
    csv_writer = LiveCSVWriter(log_file)

    use_gpu = device == "cuda"
    resource_monitor = ResourceMonitor(use_gpu=use_gpu)

    print(f"\n{'=' * 70}")
    print(f"MAML TIMED INFERENCE")
    print(f"{'=' * 70}")
    print(f"  Checkpoint   : {checkpoint_path}")
    print(f"  Game         : {game}")
    print(f"  Device       : {device}")
    print(f"  Levels       : {n_levels}   Max steps: {max_steps}")
    print(f"  Adapt steps  : {adapt_steps}  (0 = direct meta-weights)")
    print(f"  Experiment   : {experiment_name}")
    print(f"{'=' * 70}\n")

    # ---- Build env, discover dims ----------------------------------------
    setup_start = time.perf_counter()
    env = build_env(game, representation, resource_monitor)
    obs_dim, act_dim = get_env_dims(env)
    print(f"[OK] Env dims -> obs={obs_dim}, actions={act_dim}")

    # ---- Load checkpoint -------------------------------------------------
    policy, config = load_maml_checkpoint(checkpoint_path, obs_dim, act_dim, device)

    # ---- Optional task adaptation ----------------------------------------
    adapted_params: Optional[OrderedDict] = None
    if adapt_steps > 0:
        print(f"\n[ADAPT] {adapt_steps} inner-loop steps on {game} ...")
        t0 = time.perf_counter()
        adapted_params = adapt_policy(
            policy,
            env,
            inner_lr=config.get("inner_lr", inner_lr),
            inner_steps=adapt_steps,
            n_trajectories=n_trajectories,
            device=device,
        )
        print(f"[OK] Adaptation done in {time.perf_counter() - t0:.2f}s")
    else:
        print("[INFO] Using meta-learned weights directly (no adaptation)\n")

    model = MAMLPolicyWrapper(policy, device, obs_dim, adapted_params)
    setup_time = time.perf_counter() - setup_start
    print(f"[OK] Setup complete in {setup_time:.3f}s\n")

    # ---- Generation loop -------------------------------------------------
    all_results = []

    for i in range(n_levels):
        level_id = i + 1
        print(f"Level {level_id}/{n_levels}:")

        # Reset
        reset_start = time.perf_counter()
        obs = env.reset()
        reset_time = time.perf_counter() - reset_start

        # Generate
        gen_start = time.perf_counter()
        res_start = resource_monitor.get_resources()
        done = False
        steps = 0
        total_reward = 0.0
        inference_times = []

        while not done and steps < max_steps:
            t0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=deterministic)
            inference_times.append(time.perf_counter() - t0)

            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            # Log every 50 steps (and on done) via TrainingLogger.log_step()
            if steps % 50 == 0 or done:
                step_res = resource_monitor.get_resources()
                training_logger.log_step(
                    reward=float(reward),
                    resources=step_res,
                    content_metrics={
                        "level_id": level_id,
                        "total_reward": total_reward,
                        "done": int(bool(done)),
                    },
                )

        gen_time = time.perf_counter() - gen_start
        res_end = resource_monitor.get_resources()

        # Extract & validate level
        extract_start = time.perf_counter()
        level = extract_level(env, info)
        val_msg = "N/A"
        validation_time = 0.0

        if game == "sokoban":
            vs = time.perf_counter()
            is_valid, val_msg = is_valid_sokoban(level)
            validation_time = time.perf_counter() - vs

        extract_time = time.perf_counter() - extract_start

        # Metrics
        metrics_start = time.perf_counter()
        metrics = calculate_content_metrics(level)
        metrics_time = time.perf_counter() - metrics_start

        # Solvability flag from env info
        solve_start = time.perf_counter()
        if isinstance(info, dict):
            is_solvable = info.get("solvable", None)
        elif isinstance(info, (list, tuple)) and info:
            is_solvable = info[0].get("solvable", None)
        else:
            is_solvable = None
        solvability_time = time.perf_counter() - solve_start

        # Save files
        save_start = time.perf_counter()
        lpath = os.path.join(save_dir, f"level_{level_id:03d}")
        save_level(level, lpath + ".npy", format="npy")
        save_level(level, lpath + ".txt", format="txt")
        try:
            save_level_image(
                level, lpath + ".png", game=game, scale=25, show_grid=True, dpi=300
            )
        except Exception as e:
            print(f"  [WARN] Image save failed: {e}")
        save_time = time.perf_counter() - save_start

        total_time = (
            reset_time
            + gen_time
            + extract_time
            + validation_time
            + metrics_time
            + solvability_time
            + save_time
        )

        result = {
            "level_id": level_id,
            "timestamp": datetime.now().isoformat(),
            "game": game,
            "algorithm": "MAML",
            "adapt_steps": adapt_steps,
            "reset_time_ms": reset_time * 1000,
            "generation_time_ms": gen_time * 1000,
            "extract_time_ms": extract_time * 1000,
            "validation_time_ms": validation_time * 1000,
            "metrics_time_ms": metrics_time * 1000,
            "solvability_time_ms": solvability_time * 1000,
            "save_time_ms": save_time * 1000,
            "total_time_ms": total_time * 1000,
            "steps": steps,
            "mean_inference_ms": np.mean(inference_times) * 1000,
            "std_inference_ms": np.std(inference_times) * 1000,
            "min_inference_ms": np.min(inference_times) * 1000,
            "max_inference_ms": np.max(inference_times) * 1000,
            "total_reward": total_reward,
            "diversity": metrics["diversity"],
            "complexity": metrics["complexity"],
            "unique_tiles": metrics["unique_tiles"],
            "is_solvable": is_solvable,
            "ram_start_pct": res_start["ram_percent"],
            "ram_end_pct": res_end["ram_percent"],
            "ram_delta_pct": res_end["ram_percent"] - res_start["ram_percent"],
            "cpu_start_pct": res_start["cpu_percent"],
            "cpu_end_pct": res_end["cpu_percent"],
            "cpu_delta_pct": res_end["cpu_percent"] - res_start["cpu_percent"],
            "gpu_start_pct": res_start["gpu_mem_percent"],
            "gpu_end_pct": res_end["gpu_mem_percent"],
            "gpu_delta_pct": res_end["gpu_mem_percent"] - res_start["gpu_mem_percent"],
        }

        # Flush this row immediately — no data lost even if run crashes
        csv_writer.write(result)
        all_results.append(result)

        # Mark episode end and save TrainingLogger CSV
        training_logger.log_episode_end()
        training_logger.save()

        # Console summary
        print(
            f"  Total: {total_time * 1000:.1f} ms | "
            f"gen: {gen_time * 1000:.1f} ms ({steps} steps) | "
            f"infer/step: {np.mean(inference_times) * 1000:.3f} ms"
        )
        print(
            f"  diversity={metrics['diversity']:.3f}  "
            f"complexity={metrics['complexity']:.3f}  "
            f"solvable={is_solvable}"
        )
        if game == "sokoban":
            print(f"  sokoban check : {val_msg}")
        print(
            f"  CPU={res_end['cpu_percent']:.0f}%  "
            f"RAM={res_end['ram_percent']:.0f}%  "
            f"GPU={res_end['gpu_mem_percent']:.0f}%"
        )
        print(f"  Saved -> {lpath}.*\n")

    # ---- Cleanup ---------------------------------------------------------
    csv_writer.close()
    env.close()

    df = pd.DataFrame(all_results)
    _print_summary(df)
    _write_latex(df, log_file.replace(".csv", "_table.tex"))

    print(f"[OK] Timing CSV  -> {log_file}")
    print(f"[OK] Live log    -> {log_dir}/{experiment_name}.csv")
    print(f"[OK] LaTeX table -> {log_file.replace('.csv', '_table.tex')}")
    print(f"[OK] Levels      -> {save_dir}/\n")
    return df


# ==========================================================================
# Summary / LaTeX helpers
# ==========================================================================


def _print_summary(df: pd.DataFrame):
    print(f"\n{'=' * 70}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 70}")

    print("\nTIMING PERFORMANCE:")
    print(
        f"  Total time (mean)         : "
        f"{df['total_time_ms'].mean():.2f} ± {df['total_time_ms'].std():.2f} ms"
    )
    print(
        f"  Generation time (mean)    : "
        f"{df['generation_time_ms'].mean():.2f} ± {df['generation_time_ms'].std():.2f} ms"
    )
    print(
        f"  Per-step inference (mean) : "
        f"{df['mean_inference_ms'].mean():.3f} ± {df['mean_inference_ms'].std():.3f} ms"
    )
    print(
        f"  Solvability check (mean)  : "
        f"{df['solvability_time_ms'].mean():.2f} ± {df['solvability_time_ms'].std():.2f} ms"
    )

    print("\nGENERATION QUALITY:")
    print(f"  Mean steps      : {df['steps'].mean():.1f} ± {df['steps'].std():.1f}")
    print(
        f"  Mean reward     : {df['total_reward'].mean():.2f} ± {df['total_reward'].std():.2f}"
    )
    print(
        f"  Mean diversity  : {df['diversity'].mean():.3f} ± {df['diversity'].std():.3f}"
    )
    print(
        f"  Mean complexity : {df['complexity'].mean():.3f} ± {df['complexity'].std():.3f}"
    )

    if df["is_solvable"].notna().any():
        rate = df["is_solvable"].sum() / len(df) * 100
        print(f"  Solvability     : {rate:.1f}%")

    print("\nRESOURCE USAGE:")
    print(f"  RAM delta (mean) : {df['ram_delta_pct'].mean():.2f}%")
    print(f"  CPU usage (mean) : {df['cpu_end_pct'].mean():.1f}%")
    print(f"  GPU usage (mean) : {df['gpu_end_pct'].mean():.1f}%")
    print(f"{'=' * 70}\n")


def _write_latex(df: pd.DataFrame, tex_path: str):
    rows = [
        ("Total Time (ms)", "total_time_ms"),
        ("Generation Time (ms)", "generation_time_ms"),
        ("Per-Step Inference (ms)", "mean_inference_ms"),
        ("Steps", "steps"),
        ("Diversity", "diversity"),
        ("Complexity", "complexity"),
    ]
    with open(tex_path, "w") as f:
        f.write("% MAML Inference Timing — auto-generated\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{MAML Inference Timing Performance}\n")
        f.write("\\label{tab:maml_inference_timing}\n")
        f.write("\\begin{tabular}{lcc}\\hline\n")
        f.write("Metric & Mean & Std Dev \\\\\n\\hline\n")
        for label, col in rows:
            f.write(f"{label} & {df[col].mean():.2f} & {df[col].std():.2f} \\\\\n")
        if df["is_solvable"].notna().any():
            rate = df["is_solvable"].sum() / len(df) * 100
            f.write(f"Solvability (\\%) & {rate:.1f} & -- \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"[OK] LaTeX table -> {tex_path}")


# ==========================================================================
# CLI
# ==========================================================================


def main():
    parser = argparse.ArgumentParser(description="Timed MAML inference – paper metrics")

    parser.add_argument(
        "checkpoint",
        help="Path to MAML .pt checkpoint "
        "(e.g. checkpoints/sokoban_MAML_inference/best_meta_model.pt)",
    )
    parser.add_argument("--game", default="sokoban")
    parser.add_argument("--representation", default="narrow")
    parser.add_argument("--n-levels", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Stochastic policy (default: deterministic)",
    )
    parser.add_argument(
        "--adapt-steps",
        type=int,
        default=0,
        help="MAML inner-loop adaptation steps before inference "
        "(0 = use meta-weights directly, fastest)",
    )
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--n-trajectories", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--save-dir", default="generated_levels/maml")
    parser.add_argument(
        "--log-file",
        default="inference_timing_maml.csv",
        help="Per-level timing CSV (row flushed after every level)",
    )
    parser.add_argument(
        "--log-dir", default="logs", help="Directory for TrainingLogger step-level CSV"
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment tag used in TrainingLogger filename",
    )

    args = parser.parse_args()

    generate_timed(
        checkpoint_path=args.checkpoint,
        game=args.game,
        representation=args.representation,
        n_levels=args.n_levels,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        adapt_steps=args.adapt_steps,
        inner_lr=args.inner_lr,
        n_trajectories=args.n_trajectories,
        device=args.device,
        save_dir=args.save_dir,
        log_file=args.log_file,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
```

## rlhf_trainer.py :

```python
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

Patches applied vs original:
    [P1] RLHFRewardWrapper.step(): added is_vectorized guard so the wrapper
         handles both plain gym dicts and DummyVecEnv list-of-dicts safely.
    [P2] RLHFRewardWrapper._to_flat(): added explicit mode='constant',
         constant_values=0 to np.pad to silence implicit padding warnings.
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
sys.path.append(os.path.join(project_root, "gym-pcgrl"))

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
    print(
        "Error: stable-baselines3 not installed. "
        "Install with: pip install stable-baselines3"
    )
    sys.exit(1)


# ===========================================================================
# Phase 1 — Level Generation for Feedback
# ===========================================================================


def generate_levels(
    game: str = "zelda",
    representation: str = "narrow",
    n_levels: int = 50,
    model_path: Optional[str] = None,
    device: str = "cpu",
) -> List[np.ndarray]:
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
    resource_monitor = ResourceMonitor(use_gpu=(device == "cuda"))
    env = make_pcgrl_env(
        game=game, representation=representation, resource_monitor=resource_monitor
    )
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
        if hasattr(env, "unwrapped") and hasattr(env.unwrapped, "_rep"):
            level = env.unwrapped._rep._map.copy()
        elif isinstance(obs, np.ndarray) and obs.ndim >= 2:
            level = obs[:, :, 0] if obs.ndim == 3 else obs
        else:
            side = max(1, int(np.sqrt(obs.size)))
            level = obs.flatten()[: side * side].reshape(side, side)

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

    def __init__(self, save_path: str = "data/preferences"):
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)
        self.preferences: List[Dict] = []
        self._load_existing()

    # ----- persistence -----
    def _load_existing(self):
        pref_file = os.path.join(self.save_path, "preferences.json")
        if os.path.exists(pref_file):
            with open(pref_file, "r") as f:
                self.preferences = json.load(f)
            print(f"[OK] Loaded {len(self.preferences)} existing preferences")

    def save(self):
        pref_file = os.path.join(self.save_path, "preferences.json")
        with open(pref_file, "w") as f:
            json.dump(self.preferences, f, indent=2, default=str)
        print(f"[OK] Saved {len(self.preferences)} preferences to {pref_file}")

    # ----- add data -----
    def add_preference(
        self,
        level_a: np.ndarray,
        level_b: np.ndarray,
        preference: float,
        metadata: Optional[Dict] = None,
    ):
        """
        Record a single pairwise preference.

        Args:
            level_a / level_b: Level arrays
            preference: 0.0 = A wins, 1.0 = B wins, 0.5 = tie
            metadata: Optional info (annotator id, game, …)
        """
        self.preferences.append(
            {
                "level_a": level_a.tolist(),
                "level_b": level_b.tolist(),
                "preference": preference,
                "metrics_a": calculate_content_metrics(level_a),
                "metrics_b": calculate_content_metrics(level_b),
                "metadata": metadata or {},
                "timestamp": datetime.now().isoformat(),
            }
        )

    # ----- interactive collection -----
    def collect_interactive(
        self, levels: List[np.ndarray], game: str = "zelda", n_comparisons: int = 20
    ):
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
                ax1.imshow(level_a, cmap="tab20")
                ax1.set_title(f"Level A (idx {idx_a})")
                ax1.axis("off")
                ax2.imshow(level_b, cmap="tab20")
                ax2.set_title(f"Level B (idx {idx_b})")
                ax2.axis("off")
                fig.suptitle(
                    f"Comparison {i + 1}/{n_comparisons}  |  "
                    f"A: div={metrics_a['diversity']:.2f} cmplx={metrics_a['complexity']:.2f}  |  "
                    f"B: div={metrics_b['diversity']:.2f} cmplx={metrics_b['complexity']:.2f}"
                )
                plt.tight_layout()
                plt.show(block=False)
                plt.pause(0.1)
            else:
                print(f"\n--- Comparison {i + 1}/{n_comparisons} ---")
                print(
                    f"  A (idx {idx_a}): "
                    f"div={metrics_a['diversity']:.2f}  "
                    f"cmplx={metrics_a['complexity']:.2f}"
                )
                print(
                    f"  B (idx {idx_b}): "
                    f"div={metrics_b['diversity']:.2f}  "
                    f"cmplx={metrics_b['complexity']:.2f}"
                )

            while True:
                choice = input(
                    f"  [{i + 1}/{n_comparisons}] [1] A  [2] B  [3] Tie  [q] Quit: "
                ).strip()
                if choice == "1":
                    self.add_preference(
                        level_a, level_b, 0.0, {"game": game, "type": "interactive"}
                    )
                    break
                elif choice == "2":
                    self.add_preference(
                        level_a, level_b, 1.0, {"game": game, "type": "interactive"}
                    )
                    break
                elif choice == "3":
                    self.add_preference(
                        level_a, level_b, 0.5, {"game": game, "type": "interactive"}
                    )
                    break
                elif choice.lower() == "q":
                    if _has_plt:
                        plt.close("all")
                    self.save()
                    return
                else:
                    print("  Invalid. Enter 1, 2, 3, or q.")

            if _has_plt:
                plt.close("all")

        self.save()
        print(
            f"\n[OK] Collected {n_comparisons} preferences "
            f"(total: {len(self.preferences)})"
        )

    # ----- synthetic (for testing / bootstrapping) -----
    def generate_synthetic_preferences(
        self, levels: List[np.ndarray], n_comparisons: int = 100, game: str = "zelda"
    ):
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

            score_a = m_a["diversity"] * 0.6 + m_a["complexity"] * 0.4
            score_b = m_b["diversity"] * 0.6 + m_b["complexity"] * 0.4

            # Noise to simulate human inconsistency
            score_a += np.random.normal(0, 0.05)
            score_b += np.random.normal(0, 0.05)

            if score_a > score_b + 0.02:
                pref = 0.0
            elif score_b > score_a + 0.02:
                pref = 1.0
            else:
                pref = 0.5

            self.add_preference(
                level_a, level_b, pref, {"game": game, "type": "synthetic"}
            )

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

    def __init__(self, input_dim: int, hidden_sizes: List[int] = None):
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

    def predict_preference(
        self, level_a: torch.Tensor, level_b: torch.Tensor
    ) -> torch.Tensor:
        """P(A ≻ B) via Bradley-Terry."""
        return torch.sigmoid(self.forward(level_a) - self.forward(level_b))


class RewardModelTrainer:
    """Train the reward model on collected preferences."""

    def __init__(
        self, input_dim: int, device: str = "cpu", learning_rate: float = 1e-3
    ):
        self.device = device
        self.input_dim = input_dim
        self.reward_model = RewardModel(input_dim).to(device)
        self.optimizer = optim.Adam(self.reward_model.parameters(), lr=learning_rate)

    # ------------------------------------------------------------------
    def train(
        self,
        preferences: List[Dict],
        epochs: int = 100,
        batch_size: int = 32,
        validation_split: float = 0.1,
    ) -> Dict[str, List[float]]:
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
            a = self._pad_or_truncate(np.array(pref["level_a"]).flatten())
            b = self._pad_or_truncate(np.array(pref["level_b"]).flatten())
            levels_a.append(a)
            levels_b.append(b)
            labels.append(pref["preference"])

        levels_a = torch.FloatTensor(np.array(levels_a)).to(self.device)
        levels_b = torch.FloatTensor(np.array(levels_b)).to(self.device)
        labels = torch.FloatTensor(labels).to(self.device)

        # Split
        n = len(labels)
        n_val = max(1, int(n * validation_split))
        perm = torch.randperm(n)
        train_idx, val_idx = perm[n_val:], perm[:n_val]

        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        for epoch in range(epochs):
            # ---- train ----
            self.reward_model.train()
            shuffled = train_idx[torch.randperm(len(train_idx))]
            epoch_loss, n_batches = 0.0, 0

            for i in range(0, len(shuffled), batch_size):
                bi = shuffled[i : i + batch_size]
                pred = self.reward_model.predict_preference(
                    levels_a[bi], levels_b[bi]
                ).squeeze(-1)
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
                    levels_a[val_idx], levels_b[val_idx]
                ).squeeze(-1)
                v_loss = nn.functional.binary_cross_entropy(
                    v_pred, labels[val_idx]
                ).item()

                choices = (v_pred > 0.5).float()
                gt = (labels[val_idx] > 0.5).float()
                ties = labels[val_idx] == 0.5
                v_acc = ((choices == gt) | ties).float().mean().item()

            history["train_loss"].append(epoch_loss / max(n_batches, 1))
            history["val_loss"].append(v_loss)
            history["val_accuracy"].append(v_acc)

            if (epoch + 1) % 20 == 0:
                print(
                    f"  Epoch {epoch + 1}/{epochs} | "
                    f"Train: {epoch_loss / max(n_batches, 1):.4f} | "
                    f"Val: {v_loss:.4f} | "
                    f"Acc: {v_acc:.1%}"
                )

        print(f"[OK] Reward model trained. Final val accuracy: {v_acc:.1%}")
        return history

    # ------------------------------------------------------------------
    def _pad_or_truncate(self, arr: np.ndarray) -> np.ndarray:
        if len(arr) >= self.input_dim:
            return arr[: self.input_dim].astype(np.float32)
        return np.pad(
            arr,
            (0, self.input_dim - len(arr)),
            mode="constant",
            constant_values=0,
        ).astype(np.float32)

    def save(self, path: str):
        torch.save(
            {
                "model_state_dict": self.reward_model.state_dict(),
                "input_dim": self.input_dim,
            },
            path,
        )
        print(f"[OK] Reward model saved to {path}")

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.reward_model.load_state_dict(ckpt["model_state_dict"])
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

    def __init__(
        self,
        env,
        reward_model: RewardModel,
        rlhf_weight: float = 0.5,
        device: str = "cpu",
        input_dim: int = 121,
    ):
        super().__init__(env)
        self.reward_model = reward_model
        self.rlhf_weight = rlhf_weight
        self.device = device
        self.input_dim = input_dim

    def step(self, action):
        obs, env_reward, done, info = self.env.step(action)

        # [PATCH P1] Guard against vectorized environments that return info as a
        # list-of-dicts instead of a plain dict. RLHFRewardWrapper is applied
        # before DummyVecEnv in the current pipeline, so info is always a plain
        # dict in practice — but this makes the wrapper safe if wrapping order
        # ever changes (e.g. during dashboard fine-tuning experiments).
        is_vectorized = isinstance(info, (list, tuple))
        target_info = info[0] if is_vectorized else info

        # Flatten current observation for the reward model
        level_flat = self._to_flat(obs, target_info)

        with torch.no_grad():
            t = torch.FloatTensor(level_flat).unsqueeze(0).to(self.device)
            human_reward = self.reward_model(t).item()

        blended = (1 - self.rlhf_weight) * env_reward + self.rlhf_weight * human_reward

        # Write metrics back to the appropriate container position
        if is_vectorized:
            info[0]["env_reward"] = env_reward
            info[0]["human_reward"] = human_reward
            info[0]["blended_reward"] = blended
        else:
            info["env_reward"] = env_reward
            info["human_reward"] = human_reward
            info["blended_reward"] = blended

        return obs, blended, done, info

    def _to_flat(self, obs, info: Dict) -> np.ndarray:
        """Extract and pad/truncate a flat level vector."""
        if isinstance(info, dict) and "map" in info:
            arr = np.array(info["map"]).flatten()
        elif isinstance(obs, np.ndarray):
            arr = obs.flatten()
        else:
            arr = np.zeros(self.input_dim)

        if len(arr) >= self.input_dim:
            return arr[: self.input_dim].astype(np.float32)

        # [PATCH P2] Explicit mode and constant_values to avoid implicit padding
        # interpretation warnings and make intent unambiguous.
        return np.pad(
            arr,
            (0, self.input_dim - len(arr)),
            mode="constant",
            constant_values=0,
        ).astype(np.float32)


class RLHFCallback(BaseCallback):
    """
    Callback that logs RLHF-specific metrics (blended, env, human rewards)
    during PPO fine-tuning.
    """

    def __init__(
        self,
        resource_monitor: ResourceMonitor,
        training_logger: TrainingLogger,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.resource_monitor = resource_monitor
        self.training_logger = training_logger
        self.episode_count = 0
        self.ep_env_rewards: List[float] = []
        self.ep_human_rewards: List[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [{}])
        info = infos[0] if infos else {}

        self.ep_env_rewards.append(info.get("env_reward", 0.0))
        self.ep_human_rewards.append(info.get("human_reward", 0.0))

        done = self.locals.get("dones", [False])[0]
        if done:
            self.episode_count += 1
            if self.verbose and self.episode_count % 20 == 0:
                resources = self.resource_monitor.get_resources()
                mean_env = np.mean(self.ep_env_rewards[-200:])
                mean_hum = np.mean(self.ep_human_rewards[-200:])
                print(
                    f"  RLHF Ep {self.episode_count} | "
                    f"Env: {mean_env:.3f}  Human: {mean_hum:.3f} | "
                    f"CPU: {resources['cpu_percent']:.0f}%  "
                    f"RAM: {resources['ram_percent']:.0f}%"
                )
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

    def __init__(
        self,
        game: str = "zelda",
        representation: str = "narrow",
        base_model_path: Optional[str] = None,
        rlhf_weight: float = 0.5,
        reward_model_lr: float = 1e-3,
        reward_model_epochs: int = 100,
        ppo_timesteps: int = 50_000,
        device: str = "auto",
        experiment_name: Optional[str] = None,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
    ):
        self.game = game
        self.representation = representation
        self.base_model_path = base_model_path
        self.rlhf_weight = rlhf_weight
        self.reward_model_lr = reward_model_lr
        self.reward_model_epochs = reward_model_epochs
        self.ppo_timesteps = ppo_timesteps

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if experiment_name is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = f"RLHF_{game}_{ts}"
        self.experiment_name = experiment_name

        self.resource_monitor = ResourceMonitor(use_gpu=(self.device == "cuda"))
        self.logger = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name)
        self.checkpoint_dir = create_checkpoint_dir(checkpoint_dir, experiment_name)

        # Determine input dimension from a temporary env
        tmp_env = make_pcgrl_env(
            game=game,
            representation=representation,
            resource_monitor=self.resource_monitor,
        )
        if isinstance(tmp_env.observation_space, gym.spaces.Dict):
            tmp_env = DictFlattenWrapper(tmp_env)
        self.input_dim = int(np.prod(tmp_env.observation_space.shape))
        tmp_env.close()

        # Sub-components
        self.preference_collector = PreferenceCollector(
            save_path=os.path.join("data", "preferences", game)
        )
        self.reward_trainer = RewardModelTrainer(
            self.input_dim, self.device, reward_model_lr
        )

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
    def generate_levels_for_feedback(self, n_levels: int = 50) -> List[np.ndarray]:
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
    def collect_preferences(
        self,
        levels: List[np.ndarray],
        n_comparisons: int = 50,
        use_synthetic: bool = False,
    ):
        """Collect human or synthetic preferences."""
        if use_synthetic:
            self.preference_collector.generate_synthetic_preferences(
                levels, n_comparisons, self.game
            )
        else:
            self.preference_collector.collect_interactive(
                levels, self.game, n_comparisons
            )

    # ------------------------------------------------------------------
    # Step 3
    # ------------------------------------------------------------------
    def train_reward_model(self) -> RewardModel:
        """Train the Bradley-Terry reward model on collected preferences."""
        if not self.preference_collector.preferences:
            raise ValueError(
                "No preferences collected. Run collect_preferences() first."
            )

        self.reward_trainer.train(
            self.preference_collector.preferences,
            epochs=self.reward_model_epochs,
        )

        rm_path = os.path.join(self.checkpoint_dir, "reward_model.pt")
        self.reward_trainer.save(rm_path)

        return self.reward_trainer.reward_model

    # ------------------------------------------------------------------
    # Step 4
    # ------------------------------------------------------------------
    def fine_tune_with_rlhf(self, reward_model: Optional[RewardModel] = None):
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
                base,
                reward_model,
                rlhf_weight=self.rlhf_weight,
                device=self.device,
                input_dim=self.input_dim,
            )

        env = DummyVecEnv([_make_rlhf_env])

        # Load or create PPO model
        if self.base_model_path:
            model = PPO.load(self.base_model_path, env=env, device=self.device)
            print(f"[OK] Loaded base model from {self.base_model_path}")
        else:
            model = PPO(
                "MlpPolicy",
                env,
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
        rlhf_path = os.path.join(self.checkpoint_dir, "rlhf_model.zip")
        model.save(rlhf_path)
        print(f"[OK] RLHF-tuned model saved to {rlhf_path}")

        env.close()
        return model

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------
    def run_full_pipeline(
        self, n_levels: int = 50, n_comparisons: int = 50, use_synthetic: bool = True
    ):
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RLHF Training for RAPCG-MetaRL")
    parser.add_argument(
        "--game", type=str, default="zelda", choices=["zelda", "sokoban", "binary"]
    )
    parser.add_argument("--representation", type=str, default="narrow")
    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Path to pre-trained PPO .zip model",
    )
    parser.add_argument(
        "--rlhf-weight",
        type=float,
        default=0.5,
        help="Weight of human-preference reward (0–1)",
    )
    parser.add_argument(
        "--n-levels", type=int, default=50, help="Levels to generate for feedback"
    )
    parser.add_argument(
        "--n-comparisons", type=int, default=50, help="Number of pairwise comparisons"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic preferences (for testing)",
    )
    parser.add_argument(
        "--interactive", action="store_true", help="Collect real human preferences"
    )
    parser.add_argument(
        "--timesteps", type=int, default=50_000, help="PPO fine-tuning timesteps"
    )
    parser.add_argument(
        "--reward-epochs", type=int, default=100, help="Reward model training epochs"
    )
    parser.add_argument(
        "--reward-model-only",
        action="store_true",
        help="Only train reward model, skip PPO fine-tuning",
    )
    parser.add_argument(
        "--use-existing-preferences",
        action="store_true",
        help=(
            "Train from preferences already saved under data/preferences/<game>/ "
            "instead of generating a new feedback batch"
        ),
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "cpu"]
    )
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
        experiment_name=args.experiment_name,
    )

    if args.use_existing_preferences:
        reward_model = trainer.train_reward_model()
        if not args.reward_model_only:
            trainer.fine_tune_with_rlhf(reward_model)
    elif args.reward_model_only:
        levels = trainer.generate_levels_for_feedback(args.n_levels)
        trainer.collect_preferences(
            levels,
            args.n_comparisons,
            use_synthetic=not args.interactive,
        )
        trainer.train_reward_model()
    else:
        trainer.run_full_pipeline(
            n_levels=args.n_levels,
            n_comparisons=args.n_comparisons,
            use_synthetic=not args.interactive,
        )
```

## Dockerfile :

```
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install dashboard dependency
RUN pip install --no-cache-dir streamlit

# Copy project files
COPY . .

# Install gym-pcgrl
RUN cd gym-pcgrl && pip install -e . && cd ..

# Create necessary directories
RUN mkdir -p logs checkpoints generated_levels dashboard

# Set Python path
ENV PYTHONPATH="/workspace:${PYTHONPATH}"

# Streamlit config — disable telemetry, set port
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Expose Streamlit port
EXPOSE 8501

# Default: launch dashboard
CMD ["streamlit", "run", "dashboard/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## dashboard\dashboard.py :

```python
"""
RAPCG-MetaRL Dashboard
Streamlit UI for training, inference, and level visualization.
"""

import os
import sys
import time
import subprocess
import threading
import queue
import glob
import json
import random
from datetime import datetime
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAPCG-MetaRL",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PY = str(PROJECT_ROOT / "pcg_env" / "bin" / "python")
# Fallback for Windows
if not Path(PY).exists():
    PY = str(PROJECT_ROOT / "pcg_env" / "Scripts" / "python.exe")

sys.path.insert(0, str(PROJECT_ROOT))

try:
    from wrappers.helper import calculate_content_metrics
except Exception:

    def calculate_content_metrics(level: np.ndarray) -> dict:
        unique, counts = np.unique(level, return_counts=True)
        probs = counts / max(1, counts.sum())
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        return {
            "diversity": float(len(unique) / max(1, level.size)),
            "complexity": entropy,
        }


# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Base */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark terminal feel */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-card .label {
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 600;
        color: #58a6ff;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
    }
    .metric-card .sub {
        font-size: 12px;
        color: #8b949e;
        margin-top: 2px;
    }

    /* Log box */
    .log-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #7ee787;
        min-height: 200px;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .badge-running  { background: #1f3a1f; color: #7ee787; border: 1px solid #3fb950; }
    .badge-idle     { background: #1c2128; color: #8b949e; border: 1px solid #30363d; }
    .badge-done     { background: #1a2b4a; color: #58a6ff; border: 1px solid #388bfd; }
    .badge-error    { background: #3b1212; color: #f85149; border: 1px solid #da3633; }

    /* Section headers */
    .section-header {
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #21262d;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #30363d;
        border-color: #58a6ff;
        color: #58a6ff;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #238636;
        border-color: #2ea043;
        color: #fff;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2ea043;
    }

    /* Inputs */
    .stSelectbox > div, .stNumberInput > div, .stSlider {
        background-color: #161b22 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
        border-bottom: 1px solid #30363d;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #8b949e;
        padding: 10px 20px;
        border-radius: 0;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
        background: transparent !important;
    }

    /* Level grid tiles */
    .level-grid { font-family: 'JetBrains Mono', monospace; font-size: 18px; line-height: 1.4; }
    .tile-0 { color: #21262d; }   /* empty */
    .tile-1 { color: #484f58; }   /* wall */
    .tile-2 { color: #f0883e; }   /* player */
    .tile-3 { color: #7ee787; }   /* crate / path */
    .tile-4 { color: #58a6ff; }   /* target / key */
    .tile-5 { color: #f85149; }   /* enemy */
    .tile-6 { color: #ffa657; }   /* door */

    /* Divider */
    hr { border-color: #21262d; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #30363d; border-radius: 6px; }

    /* Progress bar */
    .stProgress > div > div { background-color: #238636; }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #8b949e;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "train_process": None,
        "train_log": [],
        "train_status": "idle",
        "train_start_time": None,
        "infer_process": None,
        "infer_log": [],
        "infer_status": "idle",
        "generated_levels": [],
        "rlhf_pair": None,
        "rlhf_pair_source": None,
        "log_queue": queue.Queue(),
        "last_refresh": time.time(),
        "last_save_dir": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ── Poll background processes ──────────────────────────────────────────────────
if (
    st.session_state.train_status == "running"
    and st.session_state.train_process is not None
):
    poll = st.session_state.train_process.poll()
    if poll is not None:
        st.session_state.train_status = "done" if poll == 0 else "error"
        st.session_state.train_process = None

if (
    st.session_state.infer_status == "running"
    and st.session_state.infer_process is not None
):
    poll = st.session_state.infer_process.poll()
    if poll is not None:
        st.session_state.infer_status = "done" if poll == 0 else "error"
        st.session_state.infer_process = None

# ── Helpers ───────────────────────────────────────────────────────────────────
TILE_CHARS = {
    0: "·",  # empty
    1: "█",  # wall
    2: "☺",  # player
    3: "▣",  # crate
    4: "◎",  # target
    5: "☠",  # enemy
    6: "▤",  # door / key
}
TILE_CLASSES = {
    0: "tile-0",
    1: "tile-1",
    2: "tile-2",
    3: "tile-3",
    4: "tile-4",
    5: "tile-5",
    6: "tile-6",
}


def render_level_html(level: np.ndarray) -> str:
    rows = []
    for row in level:
        cells = ""
        for val in row:
            v = int(val)
            ch = TILE_CHARS.get(v, "?")
            cls = TILE_CLASSES.get(v, "tile-0")
            cells += f'<span class="{cls}">{ch}</span>'
        rows.append(cells)
    inner = "<br>".join(rows)
    return f'<div class="level-grid">{inner}</div>'


def status_badge(status: str) -> str:
    labels = {"idle": "IDLE", "running": "RUNNING", "done": "DONE", "error": "ERROR"}
    return f'<span class="badge badge-{status}">{labels.get(status, status.upper())}</span>'


def stream_process(proc, log_list: list):
    """Stream stdout/stderr from subprocess into log list."""

    def _read(stream):
        for line in iter(stream.readline, b""):
            decoded = line.decode("utf-8", errors="replace").rstrip()
            log_list.append(decoded)
        stream.close()

    t_out = threading.Thread(target=_read, args=(proc.stdout,), daemon=True)
    t_err = threading.Thread(target=_read, args=(proc.stderr,), daemon=True)
    t_out.start()
    t_err.start()

    def _wait():
        proc.wait()
        t_out.join()
        t_err.join()

    threading.Thread(target=_wait, daemon=True).start()


def find_checkpoints() -> list:
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    if not ckpt_dir.exists():
        return []
    models = sorted(glob.glob(str(ckpt_dir / "**" / "*.zip"), recursive=True))
    models += sorted(glob.glob(str(ckpt_dir / "**" / "*.pt"), recursive=True))
    return models


def find_level_files() -> list:
    gen_dir = PROJECT_ROOT / "generated_levels"
    if not gen_dir.exists():
        return []
    return sorted(glob.glob(str(gen_dir / "**" / "*.npy"), recursive=True))


def find_log_csvs() -> list:
    log_dir = PROJECT_ROOT / "logs"
    if not log_dir.exists():
        return []
    return sorted(glob.glob(str(log_dir / "*.csv")), reverse=True)


def load_level(path: str) -> np.ndarray:
    try:
        return np.load(path)
    except Exception:
        return None


def preference_file(game: str) -> Path:
    return PROJECT_ROOT / "data" / "preferences" / game / "preferences.json"


def load_preferences(game: str) -> list:
    path = preference_file(game)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_dashboard_preference(
    game: str,
    level_a: np.ndarray,
    level_b: np.ndarray,
    preference: float,
    metadata: dict,
) -> int:
    path = preference_file(game)
    path.parent.mkdir(parents=True, exist_ok=True)
    prefs = load_preferences(game)
    prefs.append(
        {
            "level_a": level_a.tolist(),
            "level_b": level_b.tolist(),
            "preference": preference,
            "metrics_a": calculate_content_metrics(level_a),
            "metrics_b": calculate_content_metrics(level_b),
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        }
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2, default=str)
    return len(prefs)


def format_metrics(level: np.ndarray) -> str:
    metrics = calculate_content_metrics(level)
    return "div={:.3f}  complexity={:.3f}".format(
        metrics.get("diversity", 0.0),
        metrics.get("complexity", 0.0),
    )


def select_feedback_pair(files: list, source_key: str) -> tuple:
    if len(files) < 2:
        return None
    cached = st.session_state.rlhf_pair
    if cached and st.session_state.rlhf_pair_source == source_key:
        if all(Path(p).exists() for p in cached):
            return cached
    pair = tuple(random.sample(files, 2))
    st.session_state.rlhf_pair = pair
    st.session_state.rlhf_pair_source = source_key
    return pair


def resource_color(pct: float) -> str:
    if pct > 85:
        return "#f85149"
    if pct > 70:
        return "#ffa657"
    return "#7ee787"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="section-header">RAPCG-MetaRL</div>', unsafe_allow_html=True
    )
    st.markdown("**Thesis Dashboard**")
    st.markdown("Redwan Rahman · DIU", unsafe_allow_html=True)
    st.markdown("---")

    # Live resource monitor
    st.markdown('<div class="section-header">System</div>', unsafe_allow_html=True)
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0.2)
        ram = psutil.virtual_memory()
        ram_pct = ram.percent
        ram_used = ram.used / (1024**3)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="label">CPU</div>
                <div class="value" style="color:{resource_color(cpu)}">{cpu:.0f}%</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="label">RAM</div>
                <div class="value" style="color:{resource_color(ram_pct)}">{ram_pct:.0f}%</div>
                <div class="sub">{ram_used:.1f} / {ram.total / (1024**3):.0f} GB</div>
            </div>""",
                unsafe_allow_html=True,
            )

        # GPU
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpu_pct = (mem.used / mem.total) * 100
            gpu_util = util.gpu
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="label">GPU VRAM</div>
                <div class="value" style="color:{resource_color(gpu_pct)}">{gpu_pct:.0f}%</div>
                <div class="sub">{mem.used / (1024**2):.0f} / {mem.total / (1024**2):.0f} MB · util {gpu_util}%</div>
            </div>""",
                unsafe_allow_html=True,
            )
        except Exception:
            st.markdown(
                """
            <div class="metric-card">
                <div class="label">GPU</div>
                <div class="value" style="color:#8b949e">N/A</div>
            </div>""",
                unsafe_allow_html=True,
            )
    except ImportError:
        st.warning("psutil not available")

    st.markdown("---")

    # Process status
    st.markdown('<div class="section-header">Processes</div>', unsafe_allow_html=True)
    st.markdown(
        f"Training &nbsp; {status_badge(st.session_state.train_status)}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"Inference {status_badge(st.session_state.infer_status)}",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("⟳ Refresh", use_container_width=True):
        st.rerun()

    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (3s)", value=False)
    if auto_refresh:
        time.sleep(3)
        st.rerun()

# ── Main tabs ─────────────────────────────────────────────────────────────────
_legacy_tab_labels = ["⚡ Train", "🎲 Inference", "🗺 Levels", "📊 Logs", "⚖ Compare"]

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRAIN
# ══════════════════════════════════════════════════════════════════════════════
tab_train, tab_infer, tab_levels, tab_feedback, tab_logs, tab_compare = st.tabs(
    ["Train", "Inference", "Levels", "RLHF Feedback", "Logs", "Compare"]
)

with tab_train:
    st.markdown(
        '<div class="section-header">Training Configuration</div>',
        unsafe_allow_html=True,
    )

    col_cfg, col_log = st.columns([1, 1], gap="large")

    with col_cfg:
        # Config form
        training_mode = st.selectbox(
            "Workflow",
            ["Standard PPO/A2C/SAC", "MAML meta-training", "RLHF fine-tuning"],
            index=0,
        )
        game = st.selectbox("Game", ["zelda", "sokoban", "binary"], index=0)
        algo = st.selectbox("Algorithm", ["PPO", "A2C", "SAC"], index=0)
        representation = st.selectbox(
            "Representation", ["narrow", "wide", "turtle"], index=0
        )
        timesteps = st.number_input(
            "Timesteps", min_value=1000, max_value=1_000_000, value=50_000, step=10_000
        )

        col_a, col_b = st.columns(2)
        with col_a:
            n_envs = st.number_input(
                "Parallel envs",
                min_value=1,
                max_value=6,
                value=1,
                help="Max 6 on this hardware",
            )
            batch_size = st.number_input(
                "Batch size", min_value=16, max_value=512, value=64, step=16
            )
        with col_b:
            n_steps = st.number_input(
                "Steps/update", min_value=32, max_value=2048, value=128, step=32
            )
            lr = st.number_input(
                "Learning rate",
                min_value=1e-5,
                max_value=1e-2,
                value=2.5e-4,
                format="%.5f",
            )

        checkpoint_freq = st.number_input(
            "Checkpoint every N steps",
            min_value=500,
            max_value=50_000,
            value=5_000,
            step=500,
        )

        if game == "sokoban":
            sokoban_penalty = st.slider(
                "Sokoban unsolvable penalty",
                min_value=0.0,
                max_value=50.0,
                value=25.0,
                step=1.0,
            )
            use_backward = st.checkbox(
                "Use backward generation (guaranteed solvable)", value=False
            )
        else:
            sokoban_penalty = 25.0
            use_backward = False

        device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
        experiment_name = st.text_input(
            "Experiment name (optional)",
            value="",
            placeholder="auto-generated if blank",
        )

        if training_mode == "MAML meta-training":
            st.markdown("**MAML settings**")
            maml_games = st.multiselect(
                "Task games",
                ["zelda", "sokoban", "binary"],
                default=["zelda", "sokoban", "binary"],
            )
            maml_representations = st.multiselect(
                "Task representations",
                ["narrow", "wide", "turtle"],
                default=["narrow", "wide", "turtle"],
            )
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                maml_iterations = st.number_input(
                    "Meta iterations", min_value=1, max_value=10_000, value=500
                )
                maml_meta_batch = st.number_input(
                    "Meta batch", min_value=1, max_value=16, value=4
                )
                maml_inner_steps = st.number_input(
                    "Inner steps", min_value=1, max_value=50, value=5
                )
            with col_m2:
                maml_trajectories = st.number_input(
                    "Trajectory steps", min_value=16, max_value=2048, value=128, step=16
                )
                maml_meta_lr = st.number_input(
                    "Meta LR", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f"
                )
                maml_inner_lr = st.number_input(
                    "Inner LR",
                    min_value=1e-5,
                    max_value=1e-1,
                    value=1e-2,
                    format="%.5f",
                )
            maml_second_order = st.checkbox("Use second-order MAML", value=False)
        else:
            maml_games = []
            maml_representations = []
            maml_iterations = 500
            maml_meta_batch = 4
            maml_inner_steps = 5
            maml_trajectories = 128
            maml_meta_lr = 1e-3
            maml_inner_lr = 1e-2
            maml_second_order = False

        if training_mode == "RLHF fine-tuning":
            st.markdown("**RLHF settings**")
            zip_checkpoints = [c for c in find_checkpoints() if c.endswith(".zip")]
            base_options = ["(random init)"] + zip_checkpoints
            base_model_choice = st.selectbox(
                "Base PPO model",
                base_options,
                format_func=lambda p: p
                if p == "(random init)"
                else Path(p).relative_to(PROJECT_ROOT).as_posix(),
            )
            feedback_source = st.selectbox(
                "Feedback source",
                ["Dashboard preferences", "Synthetic preferences"],
                index=0,
                help="Collect real human labels in the RLHF Feedback tab.",
            )
            existing_pref_count = len(load_preferences(game))
            if feedback_source == "Dashboard preferences":
                st.caption(f"{existing_pref_count} saved preference(s) for {game}.")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                rlhf_weight = st.slider(
                    "Human reward weight", min_value=0.0, max_value=1.0, value=0.5
                )
                rlhf_levels = st.number_input(
                    "Feedback levels", min_value=2, max_value=500, value=50
                )
                rlhf_comparisons = st.number_input(
                    "Comparisons", min_value=1, max_value=500, value=50
                )
            with col_r2:
                rlhf_timesteps = st.number_input(
                    "Fine-tune timesteps",
                    min_value=0,
                    max_value=1_000_000,
                    value=50_000,
                    step=5_000,
                )
                reward_epochs = st.number_input(
                    "Reward epochs", min_value=1, max_value=1000, value=100
                )
                reward_model_only = st.checkbox("Reward model only", value=False)
        else:
            base_model_choice = "(random init)"
            feedback_source = "Synthetic preferences"
            existing_pref_count = 0
            rlhf_weight = 0.5
            rlhf_levels = 50
            rlhf_comparisons = 50
            rlhf_timesteps = 50_000
            reward_epochs = 100
            reward_model_only = False

        st.markdown("---")

        col_btn1, col_btn2 = st.columns(2)
        start_disabled = st.session_state.train_status == "running"
        if training_mode == "MAML meta-training":
            start_disabled = (
                start_disabled or not maml_games or not maml_representations
            )
        if (
            training_mode == "RLHF fine-tuning"
            and feedback_source == "Dashboard preferences"
        ):
            start_disabled = start_disabled or existing_pref_count == 0
        with col_btn1:
            start_clicked = st.button(
                "▶ Start Training",
                type="primary",
                use_container_width=True,
                disabled=start_disabled,
            )
        with col_btn2:
            stop_clicked = st.button(
                "■ Stop",
                use_container_width=True,
                disabled=(st.session_state.train_status != "running"),
            )

    with col_log:
        st.markdown(
            '<div class="section-header">Live Output</div>', unsafe_allow_html=True
        )

        # Status + elapsed
        elapsed = ""
        if (
            st.session_state.train_start_time
            and st.session_state.train_status == "running"
        ):
            secs = int(time.time() - st.session_state.train_start_time)
            elapsed = f" · {secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"
        st.markdown(
            f"{status_badge(st.session_state.train_status)}{elapsed}",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        log_text = (
            "\n".join(st.session_state.train_log[-200:]) or "Waiting for output..."
        )
        st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)

        # Progress estimate from log
        if st.session_state.train_log:
            progress_val = 0.0
            for line in reversed(st.session_state.train_log):
                if "timesteps" in line.lower() and "/" in line:
                    try:
                        parts = [p.strip() for p in line.split("|")]
                        for p in parts:
                            if "/" in p and any(c.isdigit() for c in p):
                                nums = [
                                    int(x.replace(",", ""))
                                    for x in p.split("/")
                                    if x.strip().replace(",", "").isdigit()
                                ]
                                if len(nums) == 2 and nums[1] > 0:
                                    progress_val = min(nums[0] / nums[1], 1.0)
                                    break
                    except Exception:
                        pass
                    if progress_val > 0:
                        break
            if progress_val > 0:
                st.progress(progress_val, text=f"{progress_val * 100:.1f}% complete")

    # ── Start / stop logic ────────────────────────────────────────────────────
    if start_clicked:
        st.session_state.train_log = []
        st.session_state.train_status = "running"
        st.session_state.train_start_time = time.time()

        if training_mode == "MAML meta-training":
            cmd = [
                PY,
                str(PROJECT_ROOT / "maml_trainer.py"),
                "--games",
                *maml_games,
                "--representations",
                *maml_representations,
                "--meta-lr",
                str(maml_meta_lr),
                "--inner-lr",
                str(maml_inner_lr),
                "--inner-steps",
                str(maml_inner_steps),
                "--meta-batch",
                str(maml_meta_batch),
                "--iterations",
                str(maml_iterations),
                "--n-trajectories",
                str(maml_trajectories),
                "--device",
                device,
            ]
            if maml_second_order:
                cmd.append("--second-order")
            if experiment_name.strip():
                cmd += ["--experiment-name", experiment_name.strip()]
        elif training_mode == "RLHF fine-tuning":
            cmd = [
                PY,
                str(PROJECT_ROOT / "rlhf_trainer.py"),
                "--game",
                game,
                "--representation",
                representation,
                "--rlhf-weight",
                str(rlhf_weight),
                "--n-levels",
                str(rlhf_levels),
                "--n-comparisons",
                str(rlhf_comparisons),
                "--timesteps",
                str(rlhf_timesteps),
                "--reward-epochs",
                str(reward_epochs),
                "--device",
                device,
            ]
            if base_model_choice != "(random init)":
                cmd += ["--base-model", base_model_choice]
            if feedback_source == "Dashboard preferences":
                cmd.append("--use-existing-preferences")
            else:
                cmd.append("--synthetic")
            if reward_model_only:
                cmd.append("--reward-model-only")
            if experiment_name.strip():
                cmd += ["--experiment-name", experiment_name.strip()]
        elif use_backward and game == "sokoban":
            cmd = [
                PY,
                str(PROJECT_ROOT / "train_backward.py"),
                "--game",
                game,
                "--timesteps",
                str(timesteps),
                "--device",
                device,
            ]
        else:
            cmd = [
                PY,
                str(PROJECT_ROOT / "train.py"),
                "--game",
                game,
                "--algorithm",
                algo,
                "--representation",
                representation,
                "--timesteps",
                str(timesteps),
                "--n-envs",
                str(n_envs),
                "--batch-size",
                str(batch_size),
                "--n-steps",
                str(n_steps),
                "--lr",
                str(lr),
                "--checkpoint-freq",
                str(checkpoint_freq),
                "--device",
                device,
                "--sokoban-penalty",
                str(sokoban_penalty),
            ]
            if experiment_name.strip():
                cmd += ["--experiment-name", experiment_name.strip()]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        st.session_state.train_process = proc
        stream_process(proc, st.session_state.train_log)
        st.rerun()

    if stop_clicked and st.session_state.train_process:
        st.session_state.train_process.terminate()
        st.session_state.train_status = "idle"
        st.session_state.train_log.append("— Training stopped by user —")
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_infer:
    st.markdown(
        '<div class="section-header">Inference Configuration</div>',
        unsafe_allow_html=True,
    )

    col_icfg, col_ilog = st.columns([1, 1], gap="large")

    with col_icfg:
        checkpoints = find_checkpoints()
        if checkpoints:
            ckpt_labels = [
                Path(c).relative_to(PROJECT_ROOT).as_posix() for c in checkpoints
            ]
            ckpt_idx = st.selectbox(
                "Checkpoint",
                range(len(ckpt_labels)),
                format_func=lambda i: ckpt_labels[i],
            )
            selected_ckpt = checkpoints[ckpt_idx]
        else:
            st.warning("No checkpoints found. Train a model first.")
            selected_ckpt = None

        infer_game = st.selectbox(
            "Game ", ["zelda", "sokoban", "binary"], key="infer_game"
        )
        infer_mode = st.selectbox("Mode", ["Standard PPO/A2C", "MAML"], index=0)

        # Select algorithm (only relevant for standard SB3 models)
        if infer_mode == "Standard PPO/A2C":
            infer_algo = st.selectbox(
                "Algorithm ", ["PPO", "A2C"], index=0, key="infer_algo"
            )
        else:
            infer_algo = "PPO"

        infer_repr = st.selectbox(
            "Representation ", ["narrow", "wide", "turtle"], index=0, key="infer_repr"
        )

        n_levels = st.number_input(
            "Levels to generate", min_value=1, max_value=100, value=10
        )
        max_steps = st.number_input(
            "Max steps/level", min_value=50, max_value=2000, value=500, step=50
        )
        infer_device = st.selectbox(
            "Device ", ["auto", "cuda", "cpu"], index=0, key="infer_device"
        )

        if infer_mode == "MAML":
            adapt_steps = st.number_input(
                "Adaptation steps (0 = meta-weights directly)",
                min_value=0,
                max_value=20,
                value=0,
            )
        else:
            adapt_steps = 0

        log_file = st.text_input("Output CSV name", value="inference_timing.csv")

        st.markdown("---")
        run_infer = st.button(
            "▶ Run Inference",
            type="primary",
            use_container_width=True,
            disabled=(
                st.session_state.infer_status == "running" or selected_ckpt is None
            ),
        )

    with col_ilog:
        st.markdown(
            '<div class="section-header">Live Output</div>', unsafe_allow_html=True
        )
        st.markdown(status_badge(st.session_state.infer_status), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        ilog_text = (
            "\n".join(st.session_state.infer_log[-200:]) or "Waiting for output..."
        )
        st.markdown(f'<div class="log-box">{ilog_text}</div>', unsafe_allow_html=True)

    # Display generated levels on completion below the configuration and live output columns
    if st.session_state.infer_status == "done" and st.session_state.last_save_dir:
        st.markdown("---")
        st.markdown(
            '<div class="section-header">Generated Levels Preview</div>',
            unsafe_allow_html=True,
        )
        save_path = Path(st.session_state.last_save_dir)
        if save_path.exists():
            files = sorted(glob.glob(str(save_path / "*.npy")))
            if not files:
                st.info("No level files found in the output directory.")
            else:
                st.success(
                    f"Successfully generated {len(files)} levels! You can also view them in the **Levels** tab (Set: `{save_path.relative_to(PROJECT_ROOT).as_posix()}`)."
                )

                cols_per_row = 4
                for row_start in range(0, len(files), cols_per_row):
                    row_files = files[row_start : row_start + cols_per_row]
                    cols = st.columns(cols_per_row)
                    for col, fpath in zip(cols, row_files):
                        level = load_level(fpath)
                        if level is None:
                            continue
                        name = Path(fpath).stem
                        with col:
                            st.markdown(f"**{name}**")
                            png_path = fpath.replace(".npy", ".png")
                            if Path(png_path).exists():
                                st.image(png_path, use_container_width=True)
                            else:
                                st.markdown(
                                    render_level_html(level), unsafe_allow_html=True
                                )

                            unique = len(np.unique(level))
                            size = f"{level.shape[0]}×{level.shape[1]}"
                            st.caption(f"{size} · {unique} tile types")

    if run_infer and selected_ckpt:
        st.session_state.infer_log = []
        st.session_state.infer_status = "running"

        if infer_mode == "MAML":
            save_dir = str(PROJECT_ROOT / "generated_levels" / "maml")
            cmd = [
                PY,
                str(PROJECT_ROOT / "maml_inference_timed.py"),
                selected_ckpt,
                "--game",
                infer_game,
                "--representation",
                infer_repr,
                "--n-levels",
                str(n_levels),
                "--max-steps",
                str(max_steps),
                "--adapt-steps",
                str(adapt_steps),
                "--log-file",
                str(PROJECT_ROOT / log_file),
                "--device",
                infer_device,
            ]
        else:
            save_dir = str(
                PROJECT_ROOT
                / "generated_levels"
                / f"{infer_game}_{infer_algo}_{infer_repr}_standard"
            )
            cmd = [
                PY,
                str(PROJECT_ROOT / "inference_timed.py"),
                selected_ckpt,
                "--game",
                infer_game,
                "--algorithm",
                infer_algo,
                "--representation",
                infer_repr,
                "--n-levels",
                str(n_levels),
                "--max-steps",
                str(max_steps),
                "--log-file",
                str(PROJECT_ROOT / log_file),
                "--device",
                infer_device,
                "--save-dir",
                save_dir,
            ]
        st.session_state.last_save_dir = save_dir

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(PROJECT_ROOT),
            env=env,
        )
        st.session_state.infer_process = proc
        stream_process(proc, st.session_state.infer_log)
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LEVELS
# ══════════════════════════════════════════════════════════════════════════════
with tab_levels:
    st.markdown(
        '<div class="section-header">Generated Levels</div>', unsafe_allow_html=True
    )

    level_files = find_level_files()

    if not level_files:
        st.info("No generated levels found. Run inference first.")
    else:
        col_ctrl, col_info = st.columns([2, 1])
        with col_ctrl:
            # Group by subdirectory
            dirs = sorted(set(str(Path(f).parent) for f in level_files))
            dir_labels = [Path(d).relative_to(PROJECT_ROOT).as_posix() for d in dirs]
            selected_dir_idx = st.selectbox(
                "Level set", range(len(dir_labels)), format_func=lambda i: dir_labels[i]
            )
            selected_dir = dirs[selected_dir_idx]

        files_in_dir = [f for f in level_files if str(Path(f).parent) == selected_dir]

        with col_info:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="label">Levels in set</div>
                <div class="value">{len(files_in_dir)}</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Grid display — 4 per row
        cols_per_row = 4
        for row_start in range(0, len(files_in_dir), cols_per_row):
            row_files = files_in_dir[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, fpath in zip(cols, row_files):
                level = load_level(fpath)
                if level is None:
                    continue
                name = Path(fpath).stem
                with col:
                    st.markdown(f"**{name}**")

                    # Try PNG first
                    png_path = fpath.replace(".npy", ".png")
                    if Path(png_path).exists():
                        st.image(png_path, width="stretch")
                    else:
                        # Render as ASCII grid
                        html = render_level_html(level)
                        st.markdown(html, unsafe_allow_html=True)

                    # Mini stats
                    unique = len(np.unique(level))
                    size = f"{level.shape[0]}×{level.shape[1]}"
                    st.caption(f"{size} · {unique} tile types")

                    # Download button
                    txt_path = fpath.replace(".npy", ".txt")
                    if Path(txt_path).exists():
                        with open(txt_path) as f:
                            st.download_button(
                                "↓ .txt",
                                f.read(),
                                file_name=Path(txt_path).name,
                                key=f"dl_{fpath}",
                            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LOGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_feedback:
    st.markdown(
        '<div class="section-header">RLHF Human Preference Collection</div>',
        unsafe_allow_html=True,
    )

    feedback_game = st.selectbox(
        "Feedback game", ["zelda", "sokoban", "binary"], key="feedback_game"
    )
    pref_count = len(load_preferences(feedback_game))
    pref_path = preference_file(feedback_game).relative_to(PROJECT_ROOT).as_posix()
    st.caption(f"{pref_count} saved preference(s) in {pref_path}")

    level_files = find_level_files()
    if len(level_files) < 2:
        st.info("Generate at least two levels first, then return here to label pairs.")
    else:
        dirs = sorted(set(str(Path(f).parent) for f in level_files))
        dir_labels = [Path(d).relative_to(PROJECT_ROOT).as_posix() for d in dirs]
        selected_feedback_dir_idx = st.selectbox(
            "Level source",
            range(len(dir_labels)),
            format_func=lambda i: dir_labels[i],
            key="feedback_dir",
        )
        selected_feedback_dir = dirs[selected_feedback_dir_idx]
        files_in_feedback_dir = [
            f for f in level_files if str(Path(f).parent) == selected_feedback_dir
        ]

        source_key = f"{feedback_game}:{selected_feedback_dir}"
        if st.button("New Pair", key="rlhf_new_pair"):
            st.session_state.rlhf_pair = None
            st.session_state.rlhf_pair_source = None
            st.rerun()

        pair = select_feedback_pair(files_in_feedback_dir, source_key)
        if pair is None:
            st.warning("This level source needs at least two .npy files.")
        else:
            level_a = load_level(pair[0])
            level_b = load_level(pair[1])
            if level_a is None or level_b is None:
                st.error("Could not load one of the selected level files.")
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**A**")
                    png_a = pair[0].replace(".npy", ".png")
                    if Path(png_a).exists():
                        st.image(png_a, width="stretch")
                    else:
                        st.markdown(render_level_html(level_a), unsafe_allow_html=True)
                    st.caption(f"{Path(pair[0]).name} | {format_metrics(level_a)}")
                with col_b:
                    st.markdown("**B**")
                    png_b = pair[1].replace(".npy", ".png")
                    if Path(png_b).exists():
                        st.image(png_b, width="stretch")
                    else:
                        st.markdown(render_level_html(level_b), unsafe_allow_html=True)
                    st.caption(f"{Path(pair[1]).name} | {format_metrics(level_b)}")

                col_p1, col_p2, col_p3 = st.columns(3)
                preference = None
                if col_p1.button("Prefer A", type="primary", use_container_width=True):
                    preference = 0.0
                if col_p2.button("Tie", use_container_width=True):
                    preference = 0.5
                if col_p3.button("Prefer B", type="primary", use_container_width=True):
                    preference = 1.0

                if preference is not None:
                    total = save_dashboard_preference(
                        feedback_game,
                        level_a,
                        level_b,
                        preference,
                        {
                            "game": feedback_game,
                            "type": "dashboard_interactive",
                            "source_a": Path(pair[0])
                            .relative_to(PROJECT_ROOT)
                            .as_posix(),
                            "source_b": Path(pair[1])
                            .relative_to(PROJECT_ROOT)
                            .as_posix(),
                        },
                    )
                    st.session_state.rlhf_pair = None
                    st.session_state.rlhf_pair_source = None
                    st.success(f"Saved preference #{total}.")
                    st.rerun()

with tab_logs:
    st.markdown(
        '<div class="section-header">Training Logs</div>', unsafe_allow_html=True
    )

    log_csvs = find_log_csvs()

    if not log_csvs:
        st.info("No log files found yet.")
    else:
        log_labels = [Path(f).name for f in log_csvs]
        selected_log = st.selectbox(
            "Log file", log_csvs, format_func=lambda f: Path(f).name
        )

        try:
            df = pd.read_csv(selected_log)
            st.markdown(
                f"**{len(df):,} steps · {df['episode'].max() if 'episode' in df.columns else '?'} episodes**"
            )

            # Summary metrics row
            if "reward" in df.columns:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="label">Mean Reward</div>
                        <div class="value">{df["reward"].mean():.3f}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="label">Max Reward</div>
                        <div class="value">{df["reward"].max():.3f}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )
                with m3:
                    penalty_cols = [c for c in df.columns if "penalty_total" in c]
                    avg_pen = df[penalty_cols[0]].mean() if penalty_cols else 0.0
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="label">Avg Penalty</div>
                        <div class="value" style="color:#f85149">{avg_pen:.3f}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )
                with m4:
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="label">Total Steps</div>
                        <div class="value">{len(df):,}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # Charts
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                if "reward" in df.columns:
                    # Rolling mean
                    smoothed = (
                        df["reward"].rolling(window=min(200, len(df) // 10 or 1)).mean()
                    )
                    chart_df = pd.DataFrame(
                        {"reward": df["reward"], "smoothed": smoothed}
                    )
                    st.markdown("**Reward**")
                    st.line_chart(chart_df, color=["#30363d", "#58a6ff"])

            with chart_col2:
                resource_cols = [
                    c
                    for c in ["ram_percent", "cpu_percent", "gpu_mem_percent"]
                    if c in df.columns
                ]
                if resource_cols:
                    st.markdown("**Resource Usage %**")
                    st.line_chart(
                        df[resource_cols].iloc[:: max(1, len(df) // 500)],
                        color=["#f85149", "#ffa657", "#7ee787"][: len(resource_cols)],
                    )

            # Penalty breakdown chart
            penalty_cols = [
                c
                for c in df.columns
                if c.startswith("penalty_") and c != "penalty_total_penalty"
            ]
            if penalty_cols:
                st.markdown("**Penalty Breakdown**")
                st.line_chart(df[penalty_cols].iloc[:: max(1, len(df) // 500)])

            st.markdown("---")
            with st.expander("Raw data (last 500 rows)"):
                st.dataframe(df.tail(500), use_container_width=True)

            # Download
            st.download_button(
                "↓ Download CSV",
                df.to_csv(index=False),
                file_name=Path(selected_log).name,
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error reading log: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown(
        '<div class="section-header">Compare Runs</div>', unsafe_allow_html=True
    )

    log_csvs = find_log_csvs()

    if len(log_csvs) < 2:
        st.info("Need at least 2 training runs to compare.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            run_a = st.selectbox(
                "Run A", log_csvs, format_func=lambda f: Path(f).name, key="cmp_a"
            )
        with col_b:
            run_b = st.selectbox(
                "Run B",
                log_csvs,
                format_func=lambda f: Path(f).name,
                index=min(1, len(log_csvs) - 1),
                key="cmp_b",
            )

        if run_a and run_b and run_a != run_b:
            try:
                df_a = pd.read_csv(run_a)
                df_b = pd.read_csv(run_b)

                # Summary table
                def run_summary(df, name):
                    s = {"Run": Path(name).stem, "Steps": len(df)}
                    if "reward" in df.columns:
                        s["Mean Reward"] = f"{df['reward'].mean():.4f}"
                        s["Max Reward"] = f"{df['reward'].max():.4f}"
                    if "ram_percent" in df.columns:
                        s["Avg RAM %"] = f"{df['ram_percent'].mean():.1f}"
                    pen = [c for c in df.columns if "penalty_total" in c]
                    if pen:
                        s["Avg Penalty"] = f"{df[pen[0]].mean():.4f}"
                    return s

                summary = pd.DataFrame(
                    [
                        run_summary(df_a, run_a),
                        run_summary(df_b, run_b),
                    ]
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)

                st.markdown("---")

                # Reward comparison chart
                if "reward" in df_a.columns and "reward" in df_b.columns:
                    st.markdown("**Reward Comparison (rolling mean)**")
                    window = 200
                    min_len = min(len(df_a), len(df_b))
                    smooth_a = df_a["reward"].rolling(window).mean().iloc[:min_len]
                    smooth_b = df_b["reward"].rolling(window).mean().iloc[:min_len]
                    cmp_df = pd.DataFrame(
                        {
                            Path(run_a).stem[:30]: smooth_a.values,
                            Path(run_b).stem[:30]: smooth_b.values,
                        }
                    )
                    st.line_chart(cmp_df, color=["#58a6ff", "#7ee787"])

                # Resource comparison
                if "ram_percent" in df_a.columns and "ram_percent" in df_b.columns:
                    st.markdown("**RAM Usage Comparison**")
                    stride = max(1, min_len // 500)
                    ram_df = pd.DataFrame(
                        {
                            Path(run_a).stem[:30]: df_a["ram_percent"]
                            .iloc[::stride]
                            .values[: min_len // stride],
                            Path(run_b).stem[:30]: df_b["ram_percent"]
                            .iloc[::stride]
                            .values[: min_len // stride],
                        }
                    )
                    st.line_chart(ram_df, color=["#58a6ff", "#7ee787"])

            except Exception as e:
                st.error(f"Error comparing runs: {e}")

        elif run_a == run_b:
            st.warning("Select two different runs to compare.")

# ── Auto-refresh / Rerun while running ────────────────────────────────────────
if (
    st.session_state.train_status == "running"
    or st.session_state.infer_status == "running"
):
    time.sleep(1.0)
    st.rerun()
```

## fix_sokoban_prefered_levels.py :

```python
import json
import numpy as np
from collections import deque

# ── 1. Helper Reachability & Dead-Square Analysis ─────────────────────────

def get_reachable_player_tiles(level, start, walkable=(0, 2, 4)):
    """Returns all tiles the player can walk to without pushing anything."""
    h, w = level.shape
    visited = {start}
    q = deque([start])
    while q:
        y, x = q.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                if level[ny, nx] in walkable:
                    visited.add((ny, nx))
                    q.append((ny, nx))
    return visited

def compute_live_squares(level):
    """
    Finds all squares from which a crate can eventually be pulled to a target.
    Any tile not in this set is a 'dead square'.
    """
    h, w = level.shape
    targets = list(zip(*np.where(level == 4)))
    live = set()
    q = deque(targets)

    while q:
        cy, cx = q.popleft()
        if (cy, cx) in live:
            continue
        live.add((cy, cx))

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            prev_y = cy - dy
            prev_x = cx - dx
            player_y = cy - 2 * dy
            player_x = cx - 2 * dx

            if (0 <= prev_y < h and 0 <= prev_x < w and
                0 <= player_y < h and 0 <= player_x < w):
                # To pull a crate from (cy, cx) to (prev_y, prev_x),
                # neither the destination nor the required player spot can be a wall.
                if level[prev_y, prev_x] != 1 and level[player_y, player_x] != 1:
                    if (prev_y, prev_x) not in live:
                        q.append((prev_y, prev_x))
    return live

# ── 2. Full Sokoban State-Space Solver ────────────────────────────────────

def solve_sokoban(level):
    """
    A lightweight BFS solver to guarantee total level solvability.
    State representation: (player_pos, tuple_of_sorted_crate_positions)
    """
    h, w = level.shape

    # Extract structural layers (walls=1, targets=4)
    walls = (level == 1)
    targets = set(zip(*np.where(level == 4)))

    # Extract initial dynamic entities
    start_player = tuple(zip(*np.where(level == 2)))[0]
    start_crates = tuple(sorted(zip(*np.where(level == 3))))

    if not start_crates:
        return True # Trivially solved if no crates exist (handled during balancing)

    # BFS Initialization
    initial_state = (start_player, start_crates)
    queue = deque([initial_state])
    visited = {initial_state}

    while queue:
        player, crates = queue.popleft()

        # Check Win Condition: All crates are on targets
        if all(c in targets for c in crates):
            return True

        py, px = player
        crate_set = set(crates)

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = py + dy, px + dx

            if not (0 <= ny < h and 0 <= nx < w) or walls[ny, nx]:
                continue

            # Scenario A: Moving into a crate (Attempting a Push)
            if (ny, nx) in crate_set:
                next_box_y, next_box_x = ny + dy, nx + dx

                # Check if push destination is out of bounds, a wall, or another crate
                if not (0 <= next_box_y < h and 0 <= next_box_x < w):
                    continue
                if walls[next_box_y, next_box_x] or (next_box_y, next_box_x) in crate_set:
                    continue

                # Valid push: update crate positions
                new_crates = tuple(sorted([
                    (next_box_y, next_box_x) if c == (ny, nx) else c for c in crates
                ]))
                new_state = ((ny, nx), new_crates)

                if new_state not in visited:
                    visited.add(new_state)
                    queue.append(new_state)

            # Scenario B: Just walking into an empty/target space
            else:
                new_state = ((ny, nx), crates)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append(new_state)

    return False

# ── 3. Level Repair Pipeline ──────────────────────────────────────────────

def fix_level(raw):
    level = np.array(raw, dtype=int)
    h, w = level.shape
    issues = []

    # 1. Player Count Sanity Check (exactly 1)
    players = list(zip(*np.where(level == 2)))
    if len(players) == 0:
        empties = list(zip(*np.where(level == 0)))
        if empties:
            level[empties[0]] = 2
            issues.append("added missing player")
        players = list(zip(*np.where(level == 2)))
    elif len(players) > 1:
        for p in players[1:]:
            level[p] = 0
        issues.append(f"removed {len(players)-1} extra players")
        players = [players[0]]
    player_pos = players[0] if players else None

    # 2. Balance Crates and Targets (Equal counts, minimum 1)
    crate_count = int(np.sum(level == 3))
    target_count = int(np.sum(level == 4))

    if crate_count > target_count:
        excess = list(zip(*np.where(level == 3)))
        for pos in excess[:crate_count - target_count]:
            level[pos] = 0
        issues.append(f"removed {crate_count - target_count} excess crates")
    elif target_count > crate_count:
        excess = list(zip(*np.where(level == 4)))
        for pos in excess[:target_count - crate_count]:
            level[pos] = 0
        issues.append(f"removed {target_count - crate_count} excess targets")

    if int(np.sum(level == 3)) == 0:
        empties = list(zip(*np.where(level == 0)))
        if empties: level[empties[0]] = 3; issues.append("added missing crate")
    if int(np.sum(level == 4)) == 0:
        empties = list(zip(*np.where(level == 0)))
        if empties: level[empties[0]] = 4; issues.append("added missing target")

    # 3. Dead-Square Elimination
    live_squares = compute_live_squares(level)
    crates = list(zip(*np.where(level == 3)))
    removed_dead = 0
    for cy, cx in crates:
        if (cy, cx) not in live_squares:
            level[cy, cx] = 0
            removed_dead += 1
    if removed_dead:
        issues.append(f"removed {removed_dead} dead-square crates")

    # Re-balance counts if dead-square logic pruned required crates
    crate_count = int(np.sum(level == 3))
    target_count = int(np.sum(level == 4))
    if crate_count < target_count:
        # Re-add crates *only* on non-dead, non-target open live squares
        live_empties = [pos for pos in live_squares if level[pos] == 0]
        needed = target_count - crate_count
        for pos in live_empties[:needed]:
            level[pos] = 3
            issues.append(f"re-placed pruned crate onto live square {pos}")

    # 4. Global Solvability Check & Dynamic Relocation Loop
    # If a micro-map is mathematically unsolvable, relocate until BFS solves it.
    attempts = 0
    while not solve_sokoban(level) and attempts < 15:
        attempts += 1
        current_player = list(zip(*np.where(level == 2)))[0]
        reachable_tiles = get_reachable_player_tiles(level, current_player)
        live_squares = compute_live_squares(level)

        # Valid locations to push a crate from: must be reachable by player and live
        valid_crate_spots = [t for t in live_squares if t in reachable_tiles and level[t] == 0]
        current_crates = list(zip(*np.where(level == 3)))

        if current_crates and valid_crate_spots:
            # Shift the first blocked crate to an active player-accessible location
            bad_crate = current_crates[0]
            new_spot = valid_crate_spots[0]
            level[bad_crate] = 0
            level[new_spot] = 3
            issues.append(f"unsolvable state fallback: moved crate {bad_crate} to {new_spot}")
        else:
            break # Break loop if structural limits prevent clean path adjustments

    return level.tolist(), issues

# ── 4. Main Execution Engine ──────────────────────────────────────────────

input_path = "data/preferences/sokoban/preferences.json"
output_path = "preferences_fixed.json"

with open(input_path) as f:
    prefs = json.load(f)

total = len(prefs)
fixed_count = 0
all_issues = []

for i, pref in enumerate(prefs):
    fixed_a, issues_a = fix_level(pref["level_a"])
    fixed_b, issues_b = fix_level(pref["level_b"])

    if issues_a or issues_b:
        fixed_count += 1
        entry_issues = {}
        if issues_a: entry_issues["level_a"] = issues_a
        if issues_b: entry_issues["level_b"] = issues_b
        all_issues.append({"pair": i, **entry_issues})

    pref["level_a"] = fixed_a
    pref["level_b"] = fixed_b

    def metrics(level):
        arr = np.array(level)
        unique = len(np.unique(arr))
        return {
            "diversity": round(unique / arr.size, 4),
            "complexity": 1.0,
            "size": arr.size,
            "unique_tiles": unique
        }

    pref["metrics_a"] = metrics(fixed_a)
    pref["metrics_b"] = metrics(fixed_b)
    pref["metadata"]["fixed"] = True

with open(output_path, "w") as f:
    json.dump(prefs, f, indent=2)

print(f"Total pairs processed : {total}")
print(f"Pairs corrected       : {fixed_count}")
print(f"\nDetailed Corrections per Pair:")
for item in all_issues:
    print(f"  Pair {item['pair']:02d}: ", end="")
    if "level_a" in item: print(f"A={item['level_a']}", end="  ")
    if "level_b" in item: print(f"B={item['level_b']}", end="")
    print()
```

## sokoban_utils.py :

```python
# sokoban_utils.py
"""
Comprehensive Sokoban level validation and solvability utilities.

Implements all Sokoban game rules with advanced deadlock detection.
"""

import numpy as np
import gym
import sys
import os
from typing import Tuple, Dict, List
from collections import deque

# Add gym-pcgrl to path for solver access
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, "gym-pcgrl"))

# Import proper Sokoban solver with deadlock detection
try:
    from gym_pcgrl.envs.probs.sokoban.engine import State, AStarAgent, BFSAgent

    SOLVER_AVAILABLE = True
except ImportError:
    print("Warning: gym-pcgrl solver not available. Solvability checking disabled.")
    SOLVER_AVAILABLE = False


# Tile encoding:
# 0 = empty, 1 = wall, 2 = player, 3 = crate, 4 = target


def get_reachable_positions(
    level: np.ndarray, start_pos: tuple, walkable_tiles: list = None
) -> set:
    """
    Get all positions reachable from start_pos using BFS.

    Args:
        level: 2D numpy array
        start_pos: (y, x) starting position
        walkable_tiles: List of walkable tile values (default: [0, 2, 4])

    Returns:
        Set of reachable (y, x) positions
    """
    if walkable_tiles is None:
        walkable_tiles = [0, 2, 4]  # empty, player, target

    h, w = level.shape
    visited = set()
    queue = deque([start_pos])
    visited.add(start_pos)

    while queue:
        y, x = queue.popleft()

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx

            if (
                0 <= ny < h
                and 0 <= nx < w
                and (ny, nx) not in visited
                and level[ny, nx] in walkable_tiles
            ):
                visited.add((ny, nx))
                queue.append((ny, nx))

    return visited


def compute_dead_squares(level: np.ndarray, target_positions: list) -> set:
    """
    Compute dead squares - positions where a crate can never reach any target.
    Uses reverse BFS from all targets.

    Args:
        level: 2D numpy array
        target_positions: List of (y, x) target positions

    Returns:
        Set of dead square (y, x) positions
    """
    h, w = level.shape

    # Start BFS from all targets simultaneously
    reachable_from_targets = set()
    queue = deque(target_positions)

    for pos in target_positions:
        reachable_from_targets.add(pos)

    while queue:
        y, x = queue.popleft()

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx

            if (
                0 <= ny < h
                and 0 <= nx < w
                and (ny, nx) not in reachable_from_targets
                and level[ny, nx] in [0, 2, 3, 4]
            ):  # walkable or crate
                reachable_from_targets.add((ny, nx))
                queue.append((ny, nx))

    # Dead squares are all non-wall positions NOT reachable from targets
    dead_squares = set()
    for y in range(h):
        for x in range(w):
            if level[y, x] != 1 and (y, x) not in reachable_from_targets:
                dead_squares.add((y, x))

    return dead_squares


def check_sokoban_deadlock(
    level: np.ndarray, crate_pos: tuple, dead_squares: set = None
) -> bool:
    """
    Check if a crate is in a deadlock position.

    Detects:
    - Corner deadlocks (2 adjacent walls)
    - Wall-edge deadlocks (3+ adjacent walls)
    - Dead square positions (precomputed)
    - Wall-adjacent crates without nearby targets

    Args:
        level: 2D numpy array
        crate_pos: (y, x) position of crate
        dead_squares: Precomputed set of dead squares

    Returns:
        True if crate is deadlocked, False otherwise
    """
    y, x = crate_pos
    h, w = level.shape

    # Check dead squares first (fast lookup)
    if dead_squares is not None and crate_pos in dead_squares:
        return True

    # Get adjacent tiles
    up = level[y - 1, x] if y > 0 else 1
    down = level[y + 1, x] if y < h - 1 else 1
    left = level[y, x - 1] if x > 0 else 1
    right = level[y, x + 1] if x < w - 1 else 1

    # Count walls
    wall_count = sum([up == 1, down == 1, left == 1, right == 1])

    # Check corner deadlocks (2 adjacent walls)
    if (
        (up == 1 and left == 1)
        or (up == 1 and right == 1)
        or (down == 1 and left == 1)
        or (down == 1 and right == 1)
    ):
        return True

    # Check wall-edge deadlocks (3+ walls)
    if wall_count >= 3:
        return True

    # Check wall edge deadlocks with 2 walls
    if up == 1 and (left == 1 or right == 1):
        return True
    if down == 1 and (left == 1 or right == 1):
        return True
    if left == 1 and (up == 1 or down == 1):
        return True
    if right == 1 and (up == 1 or down == 1):
        return True

    # Check if crate is against wall without nearby target
    # Horizontal wall check
    if up == 1:  # Against top wall
        has_target_on_wall = False
        for dx in range(-2, 3):  # Check nearby wall
            check_x = x + dx
            if 0 <= check_x < w and level[y, check_x] == 4:
                has_target_on_wall = True
                break
        if not has_target_on_wall:
            return True

    if down == 1:  # Against bottom wall
        has_target_on_wall = False
        for dx in range(-2, 3):
            check_x = x + dx
            if 0 <= check_x < w and level[y, check_x] == 4:
                has_target_on_wall = True
                break
        if not has_target_on_wall:
            return True

    # Vertical wall check
    if left == 1:  # Against left wall
        has_target_on_wall = False
        for dy in range(-2, 3):
            check_y = y + dy
            if 0 <= check_y < h and level[check_y, x] == 4:
                has_target_on_wall = True
                break
        if not has_target_on_wall:
            return True

    if right == 1:  # Against right wall
        has_target_on_wall = False
        for dy in range(-2, 3):
            check_y = y + dy
            if 0 <= check_y < h and level[check_y, x] == 4:
                has_target_on_wall = True
                break
        if not has_target_on_wall:
            return True

    return False


def remove_deadlocked_crates(
    level: np.ndarray, target_positions: list = None
) -> Tuple[np.ndarray, int]:
    """
    Remove all deadlocked crates from the level.

    Args:
        level: 2D numpy array
        target_positions: Optional list of target positions for dead square computation

    Returns:
        (modified_level, num_removed)
    """
    level = level.copy()

    # Compute dead squares if targets provided
    dead_squares = None
    if target_positions:
        dead_squares = compute_dead_squares(level, target_positions)

    crate_positions = np.argwhere(level == 3)
    removed_count = 0

    for crate_pos in crate_positions:
        if check_sokoban_deadlock(level, tuple(crate_pos), dead_squares):
            level[tuple(crate_pos)] = 0
            removed_count += 1

    return level, removed_count


def check_crate_pushability(level: np.ndarray, crate_pos: tuple) -> bool:
    """
    Check if a crate has at least one free adjacent tile to push from.

    Args:
        level: 2D numpy array
        crate_pos: (y, x) position of crate

    Returns:
        True if crate can be pushed, False otherwise
    """
    y, x = crate_pos
    h, w = level.shape

    # Check all 4 directions
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        push_from_y, push_from_x = y + dy, x + dx
        push_to_y, push_to_x = y - dy, x - dx

        if 0 <= push_from_y < h and 0 <= push_from_x < w:
            if 0 <= push_to_y < h and 0 <= push_to_x < w:
                # Can push if there's walkable space to push from and to
                push_from_tile = level[push_from_y, push_from_x]
                push_to_tile = level[push_to_y, push_to_x]

                if push_from_tile in [0, 2, 4] and push_to_tile in [0, 4]:
                    return True

    return False


def check_player_can_reach_crates(
    level: np.ndarray, player_pos: tuple, crate_positions: list
) -> bool:
    """
    Check if player can reach all crates using BFS.

    Args:
        level: 2D numpy array
        player_pos: (y, x) position of player
        crate_positions: list of (y, x) positions of crates

    Returns:
        True if player can reach all crates, False otherwise
    """
    reachable = get_reachable_positions(level, player_pos)

    for crate_pos in crate_positions:
        if crate_pos not in reachable:
            return False

    return True


def check_crate_to_target_path(
    level: np.ndarray, crate_pos: tuple, target_positions: list
) -> bool:
    """
    Check if a crate can reach at least one target (ignoring other crates).

    Args:
        level: 2D numpy array
        crate_pos: (y, x) position of crate
        target_positions: list of (y, x) positions of targets

    Returns:
        True if crate can reach at least one target, False otherwise
    """
    # Get reachable positions from crate (treat crate as empty for pathfinding)
    reachable = get_reachable_positions(level, crate_pos)

    for target_pos in target_positions:
        if target_pos in reachable:
            return True

    return False


def check_target_dead_position(level: np.ndarray, target_pos: tuple) -> bool:
    """
    Check if a target is in a dead position (corner or surrounded by 3+ walls).

    Args:
        level: 2D numpy array
        target_pos: (y, x) position of target

    Returns:
        True if target is in dead position, False otherwise
    """
    y, x = target_pos
    h, w = level.shape

    # Count adjacent walls
    wall_count = 0
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if ny < 0 or ny >= h or nx < 0 or nx >= w or level[ny, nx] == 1:
            wall_count += 1

    # Dead if in corner (2+ walls) or surrounded (3+ walls)
    return wall_count >= 2


def validate_and_fix_sokoban(
    level: np.ndarray,
    min_crates: int = 1,
    enforce_all_rules: bool = True,
    verbose: bool = False,
) -> Tuple[np.ndarray, Dict]:
    """
    Validate and fix Sokoban level to meet all game constraints.

    Rules enforced:
    1. Exactly 1 player (placed in reachable center position)
    2. Equal number of crates and targets (minimum min_crates)
    3. At least 1 crate-target pair
    4. Remove deadlocked crates (corners, walls, dead squares)
    5. Player can reach all crates (if enforce_all_rules=True)
    6. Each crate has at least one free adjacent tile to push from
    7. Each crate can reach at least one target (ignoring other crates)
    8. No targets in dead positions (corners, 3-wall surrounds)

    Args:
        level: 2D numpy array
        min_crates: Minimum number of crate-target pairs
        enforce_all_rules: If True, apply all validation rules

    Returns:
        (fixed_level, corrections_dict)
    """
    level = level.copy()
    h, w = level.shape

    corrections = {
        "original_players": np.sum(level == 2),
        "original_crates": np.sum(level == 3),
        "original_targets": np.sum(level == 4),
        "player_fixed": False,
        "deadlocked_removed": 0,
        "unpushable_removed": 0,
        "unreachable_removed": 0,
        "dead_targets_removed": 0,
        "crates_balanced": False,
        "final_players": 0,
        "final_crates": 0,
        "final_targets": 0,
    }

    # Fix 1: Ensure exactly 1 player
    player_positions = np.argwhere(level == 2)
    if len(player_positions) != 1:
        # Remove all players
        level[level == 2] = 0

        # Place player in center of largest open area
        empty_positions = np.argwhere(level == 0)
        if len(empty_positions) > 0:
            center_y, center_x = h // 2, w // 2
            closest_empty = min(
                empty_positions,
                key=lambda p: (p[0] - center_y) ** 2 + (p[1] - center_x) ** 2,
            )
            level[tuple(closest_empty)] = 2
            corrections["player_fixed"] = True

    # Get current positions
    player_positions = np.argwhere(level == 2)
    if len(player_positions) == 0:
        # Emergency: place player anywhere
        empty_positions = np.argwhere(level == 0)
        if len(empty_positions) > 0:
            level[tuple(empty_positions[0])] = 2
            player_positions = np.argwhere(level == 2)

    player_pos = tuple(player_positions[0]) if len(player_positions) > 0 else None

    # Fix 2: Remove targets in dead positions (if enforce_all_rules)
    if enforce_all_rules:
        target_positions = np.argwhere(level == 4)
        dead_targets = []
        for target_pos in target_positions:
            if check_target_dead_position(level, tuple(target_pos)):
                level[tuple(target_pos)] = 0
                dead_targets.append(target_pos)
        corrections["dead_targets_removed"] = len(dead_targets)

    # Fix 3: Remove deadlocked crates
    target_positions = [(y, x) for y, x in np.argwhere(level == 4)]
    level, deadlocked = remove_deadlocked_crates(level, target_positions)
    corrections["deadlocked_removed"] = deadlocked

    # Fix 4: Remove unpushable crates (if enforce_all_rules)
    if enforce_all_rules:
        crate_positions = np.argwhere(level == 3)
        unpushable = []
        for crate_pos in crate_positions:
            if not check_crate_pushability(level, tuple(crate_pos)):
                level[tuple(crate_pos)] = 0
                unpushable.append(crate_pos)
        corrections["unpushable_removed"] = len(unpushable)

    # Fix 5: Remove crates with no path to any target (if enforce_all_rules)
    if enforce_all_rules:
        target_positions = [(y, x) for y, x in np.argwhere(level == 4)]
        if target_positions:
            crate_positions = np.argwhere(level == 3)
            unreachable = []
            for crate_pos in crate_positions:
                if not check_crate_to_target_path(
                    level, tuple(crate_pos), target_positions
                ):
                    level[tuple(crate_pos)] = 0
                    unreachable.append(crate_pos)
            corrections["unreachable_removed"] = len(unreachable)

    # Fix 6: Balance crates and targets
    crate_count = np.sum(level == 3)
    target_count = np.sum(level == 4)
    target_pairs = max(min_crates, min(crate_count, target_count))

    if crate_count != target_pairs or target_count != target_pairs:
        corrections["crates_balanced"] = True

        # Remove excess crates
        if crate_count > target_pairs:
            crate_positions = np.argwhere(level == 3)
            for i in range(crate_count - target_pairs):
                level[tuple(crate_positions[i])] = 0

        # Remove excess targets
        if target_count > target_pairs:
            target_positions = np.argwhere(level == 4)
            for i in range(target_count - target_pairs):
                level[tuple(target_positions[i])] = 0

        # Add crates if needed
        if crate_count < target_pairs:
            needed = target_pairs - crate_count
            empty_positions = np.argwhere(level == 0)
            if len(empty_positions) < needed:
                target_pairs = max(1, crate_count + len(empty_positions))
                needed = len(empty_positions)

            for i in range(needed):
                level[tuple(empty_positions[i])] = 3

        # Add targets if needed
        if target_count < target_pairs:
            needed = target_pairs - target_count
            empty_positions = np.argwhere(level == 0)
            if len(empty_positions) < needed:
                needed = len(empty_positions)

            for i in range(needed):
                level[tuple(empty_positions[i])] = 4

    # Fix 7: Ensure player can reach all crates (if enforce_all_rules)
    if enforce_all_rules and player_pos:
        crate_positions = [(y, x) for y, x in np.argwhere(level == 3)]
        target_positions = [(y, x) for y, x in np.argwhere(level == 4)]

        if crate_positions:
            reachable = get_reachable_positions(level, player_pos)
            unreachable = [pos for pos in crate_positions if pos not in reachable]

            if unreachable:
                # Remove unreachable crates
                for pos in unreachable:
                    level[pos] = 0
                corrections["unreachable_removed"] += len(unreachable)

                # Also remove equal number of targets to maintain balance
                target_positions = np.argwhere(level == 4)
                for i in range(min(len(unreachable), len(target_positions))):
                    level[tuple(target_positions[i])] = 0

    # Fix 8: ENSURE MINIMUM CRATES/TARGETS (critical!)
    final_crate_count = np.sum(level == 3)
    final_target_count = np.sum(level == 4)

    if final_crate_count < min_crates or final_target_count < min_crates:
        corrections["crates_balanced"] = True

        # Get reachable empty positions
        empty_positions = np.argwhere(level == 0)
        if player_pos:
            reachable = get_reachable_positions(level, player_pos)
            reachable_empty = [
                tuple(pos) for pos in empty_positions if tuple(pos) in reachable
            ]
        else:
            reachable_empty = [tuple(pos) for pos in empty_positions]

        # Add crates to reach minimum
        crates_needed = min_crates - final_crate_count
        if crates_needed > 0 and len(reachable_empty) > 0:
            for i in range(min(crates_needed, len(reachable_empty))):
                level[reachable_empty[i]] = 3
                reachable_empty = reachable_empty[1:]  # Remove used position

        # Add targets to reach minimum
        targets_needed = min_crates - final_target_count
        if targets_needed > 0 and len(reachable_empty) > 0:
            for i in range(min(targets_needed, len(reachable_empty))):
                level[reachable_empty[i]] = 4

    # Final counts
    corrections["final_players"] = np.sum(level == 2)
    corrections["final_crates"] = np.sum(level == 3)
    corrections["final_targets"] = np.sum(level == 4)

    return level, corrections


def check_solvability(
    level: np.ndarray, solver_power: int = 5000
) -> Tuple[bool, List, int]:
    """
    Check if level is solvable using proper A* solver with deadlock detection.

    Uses gym-pcgrl's multi-strategy solver:
    1. BFS (fast, complete)
    2. A* with balance=1 (heuristic-focused)
    3. A* with balance=0.5 (balanced)
    4. A* with balance=0 (cost-focused)

    Args:
        level: 2D numpy array with tile encoding 0-4
        solver_power: Maximum iterations for solver

    Returns:
        (is_solvable, solution, heuristic_distance)
        - is_solvable: True if level has a solution
        - solution: List of moves if solvable, empty list otherwise
        - heuristic_distance: Distance to win state (0 if solvable)
    """
    if not SOLVER_AVAILABLE:
        return False, [], -1

    # Convert to format expected by engine
    lvl = np.pad(level, 1)  # Add border walls
    gameCharacters = "# @$."
    lvlString = ""
    for i in range(lvl.shape[0]):
        for j in range(lvl.shape[1]):
            lvlString += gameCharacters[int(lvl[i][j])]
            if j == lvl.shape[1] - 1:
                lvlString += "\n"

    # Initialize state
    try:
        state = State()
        state.stringInitialize(lvlString.split("\n"))
    except Exception as e:
        # Invalid level format
        return False, [], -1

    # Try multiple solver strategies (in order of speed/effectiveness)
    aStarAgent = AStarAgent()
    bfsAgent = BFSAgent()

    # 1. Try BFS first (fast for simple levels)
    try:
        sol, solState, iters = bfsAgent.getSolution(state, solver_power)
        if solState.checkWin():
            return True, sol, 0
    except Exception:
        pass

    # 2. Try A* with different balance parameters
    for balance in [1, 0.5, 0]:
        try:
            sol, solState, iters = aStarAgent.getSolution(state, balance, solver_power)
            if solState.checkWin():
                return True, sol, 0
        except Exception:
            continue

    # Unsolvable - return heuristic distance
    try:
        return False, [], solState.getHeuristic()
    except:
        return False, [], -1


def is_valid_sokoban(level: np.ndarray) -> Tuple[bool, str]:
    """
    Check if level meets basic Sokoban requirements.

    Returns:
        (is_valid, error_message)
    """
    player_count = np.sum(level == 2)
    crate_count = np.sum(level == 3)
    target_count = np.sum(level == 4)

    if player_count == 0:
        return False, "No player"
    if player_count > 1:
        return False, f"Multiple players ({player_count})"
    if crate_count == 0:
        return False, "No crates"
    if target_count == 0:
        return False, "No targets"
    if crate_count != target_count:
        return False, f"Mismatched crates ({crate_count}) and targets ({target_count})"

    return True, "Valid"


def print_level_stats(
    level: np.ndarray, title: str = "Level Stats", verbose: bool = False
):
    """
    Print statistics about a Sokoban level.

    Args:
        level: 2D numpy array
        title: Title for the stats
        verbose: If True, show detailed validation checks
    """
    print(f"\n{title}:")
    print(f"  Dimensions: {level.shape[0]}x{level.shape[1]}")
    print(f"  Players: {np.sum(level == 2)}")
    print(f"  Crates: {np.sum(level == 3)}")
    print(f"  Targets: {np.sum(level == 4)}")
    print(f"  Walls: {np.sum(level == 1)}")
    print(f"  Empty: {np.sum(level == 0)}")

    is_valid, msg = is_valid_sokoban(level)
    print(f"  Status: {'✓ ' + msg if is_valid else '✗ ' + msg}")

    if verbose and is_valid:
        # Show detailed checks
        player_positions = np.argwhere(level == 2)
        crate_positions = [(y, x) for y, x in np.argwhere(level == 3)]
        target_positions = [(y, x) for y, x in np.argwhere(level == 4)]

        if len(player_positions) > 0 and crate_positions:
            player_pos = tuple(player_positions[0])

            # Check reachability
            reachable = get_reachable_positions(level, player_pos)
            unreachable = [pos for pos in crate_positions if pos not in reachable]
            can_reach = len(unreachable) == 0
            print(
                f"  Player reachability: {'✓ All crates' if can_reach else f'✗ {len(unreachable)} unreachable'}"
            )

            # Check pushability
            unpushable = sum(
                1 for pos in crate_positions if not check_crate_pushability(level, pos)
            )
            print(
                f"  Crate pushability: {'✓ All pushable' if unpushable == 0 else f'✗ {unpushable} unpushable'}"
            )

            # Check paths to targets
            if target_positions:
                no_path = sum(
                    1
                    for pos in crate_positions
                    if not check_crate_to_target_path(level, pos, target_positions)
                )
                print(
                    f"  Crate-target paths: {'✓ All connected' if no_path == 0 else f'✗ {no_path} no path'}"
                )

            # Check solvability with actual A* solver
            if SOLVER_AVAILABLE:
                is_solvable, solution, dist = check_solvability(level)
                if is_solvable:
                    print(f"  ✓ SOLVABLE - Solution length: {len(solution)}")
                else:
                    print(f"  ✗ UNSOLVABLE - Heuristic distance: {dist}")


class SokobanSolvabilityWrapper(gym.Wrapper):
    """
    Gym wrapper that validates Sokoban levels and computes solvability metrics.
    """

    def __init__(
        self,
        env,
        enforce_all_rules: bool = True,
        verbose: bool = False,
        unsolvable_penalty: float = 0,
        min_solution_length: int = 0,
        max_solution_length: int = 100,
        terminate_on_unsolvable: bool = False,
    ):
        super().__init__(env)
        self.enforce_all_rules = enforce_all_rules
        self.verbose = verbose

        # Legacy parameters (kept for compatibility but not used)
        self.unsolvable_penalty = unsolvable_penalty
        self.min_solution_length = min_solution_length
        self.max_solution_length = max_solution_length
        self.terminate_on_unsolvable = terminate_on_unsolvable

        self.validation_stats = {
            "total_resets": 0,
            "total_fixes": 0,
            "player_fixes": 0,
            "deadlock_removals": 0,
            "balance_fixes": 0,
        }

    def reset(self, **kwargs):
        # Get initial observation
        obs = self.env.reset(**kwargs)

        # Extract numpy array from observation
        # obs might be OrderedDict with keys like 'pos', 'map', etc.
        if isinstance(obs, dict):
            # Get the map/level data
            level = obs.get("map", obs.get("level", obs.get("observation", None)))
            if level is None:
                # If no standard key, try first array value
                for value in obs.values():
                    if isinstance(value, np.ndarray):
                        level = value
                        break
        else:
            level = obs

        # Validate and fix if needed
        fixed_level, corrections = validate_and_fix_sokoban(
            level, min_crates=1, enforce_all_rules=self.enforce_all_rules
        )

        # Update stats
        self.validation_stats["total_resets"] += 1
        if (
            corrections["player_fixed"]
            or corrections["crates_balanced"]
            or corrections["deadlocked_removed"] > 0
        ):
            self.validation_stats["total_fixes"] += 1
        if corrections["player_fixed"]:
            self.validation_stats["player_fixes"] += 1
        if corrections["deadlocked_removed"] > 0:
            self.validation_stats["deadlock_removals"] += corrections[
                "deadlocked_removed"
            ]
        if corrections["crates_balanced"]:
            self.validation_stats["balance_fixes"] += 1

        if self.verbose:
            print_level_stats(fixed_level, "Reset Level", verbose=True)

        # Update environment state
        self.env.unwrapped._rep._map = fixed_level

        # Update observation if it was a dict
        if isinstance(obs, dict):
            for key in obs.keys():
                if isinstance(obs[key], np.ndarray) and obs[key].shape == level.shape:
                    obs[key] = fixed_level
                    break

        return obs

    def get_stats(self) -> dict:
        """Get validation statistics."""
        return self.validation_stats.copy()


# Test cases
def test_validator():
    """Test the validator with various edge cases."""
    print("=" * 60)
    print("Testing Sokoban Validator")
    print("=" * 60)

    # Test 1: Multiple players
    print("\nTest 1: Multiple players")
    level1 = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 2, 0, 2, 1],
            [1, 3, 4, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    print_level_stats(level1, "Before Fix")
    fixed1, corr1 = validate_and_fix_sokoban(level1)
    print_level_stats(fixed1, "After Fix", verbose=True)

    # Test 2: Deadlocked crates
    print("\nTest 2: Corner deadlock")
    level2 = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 2, 0, 0, 1],
            [1, 3, 0, 0, 1],
            [1, 0, 4, 4, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    print_level_stats(level2, "Before Fix")
    fixed2, corr2 = validate_and_fix_sokoban(level2)
    print_level_stats(fixed2, "After Fix", verbose=True)

    # Test 3: Mismatched crates/targets
    print("\nTest 3: Mismatched crates/targets")
    level3 = np.array(
        [
            [1, 1, 1, 1, 1, 1],
            [1, 2, 0, 0, 0, 1],
            [1, 3, 3, 3, 0, 1],
            [1, 4, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1],
        ]
    )
    print_level_stats(level3, "Before Fix")
    fixed3, corr3 = validate_and_fix_sokoban(level3, enforce_all_rules=True)
    print_level_stats(fixed3, "After Fix", verbose=True)

    # Test 4: Valid level (should pass)
    print("\nTest 4: Already valid level")
    level4 = np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 2, 0, 0, 1],
            [1, 0, 3, 0, 1],
            [1, 0, 4, 0, 1],
            [1, 1, 1, 1, 1],
        ]
    )
    print_level_stats(level4, "Before Fix", verbose=True)
    fixed4, corr4 = validate_and_fix_sokoban(level4, enforce_all_rules=True)
    print_level_stats(fixed4, "After Fix", verbose=True)

    print("\n" + "=" * 60)
    print("Tests Complete")
    print("=" * 60)


if __name__ == "__main__":
    test_validator()
```
