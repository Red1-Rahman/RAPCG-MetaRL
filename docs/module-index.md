# Module Index — RAPCG-MetaRL

> Resource-Aware Procedural Content Generation via Meta-Reinforcement Learning
> Thesis project by Redwan Rahman, Daffodil International University — ACM Transactions on Graphics

This document is the authoritative reference for every Python module in the project. It covers purpose, public classes and functions, key parameters, dependencies, and output artifacts for each file.

---

## Table of Contents

1. [Project Layout](#1-project-layout)
2. [Core Training Modules](#2-core-training-modules)
   - [train.py](#trainpy)
   - [train_backward.py](#train_backwardpy)
   - [maml_trainer.py](#maml_trainerpy)
   - [rlhf_trainer.py](#rlhf_trainerpy)
3. [Environment Layer](#3-environment-layer)
   - [wrappers/pcgrl_env.py](#wrappers--pcgrl_envpy)
   - [wrappers/helper.py](#wrappers--helperpy)
4. [Inference Modules](#4-inference-modules)
   - [inference.py](#inferencepy)
   - [inference_timed.py](#inference_timedpy)
   - [maml_inference_timed.py](#maml_inference_timedpy)
5. [Utilities](#5-utilities)
   - [utils.py](#utilspy)
   - [model.py](#modelpy)
   - [sokoban_utils.py](#sokoban_utilspy)
   - [config_hardware.py](#config_hardwarepy)
   - [solvability_config.py](#solvability_configpy)
6. [Analysis and Visualization](#6-analysis-and-visualization)
7. [Test Suite](#7-test-suite)
8. [gym-pcgrl Vendored Layer](#8-gym-pcgrl-vendored-layer)
9. [Dashboard](#9-dashboard)
10. [Dependency Map](#10-dependency-map)
11. [Import Notes](#11-import-notes)

---

## 1. Project Layout

```
RAPCG-MetaRL/
├── train.py                        # PPO / A2C / SAC forward training
├── train_backward.py               # Backward Sokoban training
├── maml_trainer.py                 # MAML meta-training
├── rlhf_trainer.py                 # RLHF 4-phase pipeline
├── inference.py                    # Standard level generation
├── inference_timed.py              # Timed inference with CSV output
├── maml_inference_timed.py         # MAML timed inference
├── utils.py                        # ResourceMonitor, TrainingLogger
├── model.py                        # Model load/save helpers
├── sokoban_utils.py                # Sokoban validation & solvability
├── config_hardware.py              # Hardware presets
├── solvability_config.py           # Per-game reward weight tuning
├── visualize_levels.py             # Level rendering
├── architecture_diagram.py         # System diagram
├── generate_paper_figures.py       # ACM TOG figure generation
├── compare_approaches.py           # Forward vs backward comparison
├── analyze_action_penalties.py     # Action-penalty correlation analysis
├── analyze_maml_results.py         # MAML convergence analysis
├── quickstart.py                   # ⚠ Do not invoke directly (uses bare python)
├── sokoban_utils_backup.py         # ⚠ Broken — do not import
├── wrappers/
│   ├── pcgrl_env.py                # ResourceAwarePCGRLWrapper, make_pcgrl_env()
│   ├── helper.py                   # VGLC parsing, content metrics, level I/O
│   └── __init__.py
├── gym-pcgrl/                      # Vendored base environments (Khalifa et al.)
│   └── gym_pcgrl/
│       ├── envs/
│       │   ├── pcgrl_env.py        # Base gym environment
│       │   ├── helper.py           # Tile utils, Dijkstra, BFS helpers
│       │   └── probs/              # Game problem definitions
│       │       ├── binary_prob.py
│       │       ├── zelda_prob.py
│       │       ├── sokoban_prob.py
│       │       └── sokoban/engine.py  # BFS / A* solver
│       └── __init__.py             # Environment registration
├── dashboard/
│   └── dashboard.py                # Streamlit experiment console
├── test/
│   └── test.py                     # 5-category primary test suite
├── test_action_space.py
├── test_obs_shape.py
├── test_sokoban_solvability.py
├── test_solvability_integration.py
├── test_solver_integration.py
└── test_trust_model.py
```

---

## 2. Core Training Modules

### train.py

**Purpose:** Main training entry point. Creates the resource-aware environment, selects the RL algorithm, attaches the logging callback, and runs `model.learn()`.

#### Classes

**`MetaRLTrainer`**

The primary training class. Orchestrates environment setup, model initialization, and the training loop.

```python
MetaRLTrainer(
    game="zelda",                    # "zelda" | "sokoban" | "binary"
    representation="narrow",         # "narrow" | "wide" | "turtle"
    algorithm="PPO",                 # "PPO" | "A2C" | "SAC"
    total_timesteps=50000,
    n_steps=128,                     # Steps per policy update
    batch_size=64,
    learning_rate=2.5e-4,
    n_envs=1,                        # Parallel envs — keep ≤ 6 on 16 GB RAM
    device="auto",                   # "cpu" | "cuda" | "auto"
    seed=None,
    experiment_name=None,            # Auto-generated if None
    use_gpu_monitoring=True,
    checkpoint_freq=1000,
    log_dir="logs",
    checkpoint_dir="checkpoints",
    sokoban_unsolvable_penalty=25.0, # Sokoban-specific penalty weight
    use_solvability_tuning=True,     # Apply solvability_config reward weights
)
```

| Method                    | Description                                         |
| ------------------------- | --------------------------------------------------- |
| `setup_environments()`    | Builds vectorized PCGRL envs via `make_pcgrl_env()` |
| `setup_model()`           | Initializes PPO / A2C / SAC from Stable-Baselines3  |
| `train()`                 | Runs `model.learn()` with `ResourceAwareCallback`   |
| `evaluate(n_episodes=10)` | Deterministic rollout evaluation                    |
| `load_model(model_path)`  | Loads a pre-trained `.zip` checkpoint               |

**Outputs:** `checkpoints/<experiment>/model_step_N.zip`, `checkpoints/<experiment>/final_model.zip`, `logs/<experiment>.csv`

---

**`ResourceAwareCallback`** _(extends `BaseCallback`)_

Per-step callback attached to `model.learn()`. Reads resource telemetry, extracts penalty breakdowns from `info`, logs to `TrainingLogger`, and saves periodic checkpoints.

```python
ResourceAwareCallback(
    resource_monitor,       # ResourceMonitor instance
    training_logger,        # TrainingLogger instance
    save_freq=1000,         # Steps between checkpoints
    checkpoint_dir="checkpoints",
    verbose=1,
)
```

Reads from `info` dict: `ram_penalty`, `cpu_penalty`, `gpu_penalty`, `total_penalty`.
Reports action-penalty correlation summary every 50 episodes.

---

#### CLI Usage

```powershell
python.exe train.py --game zelda --timesteps 100000 --n-envs 6 --device cuda
python.exe train.py --game sokoban --algorithm A2C --timesteps 10000 --device auto
python.exe train.py --game zelda --load-model checkpoints/exp/final_model.zip --evaluate
```

**All CLI flags:**

| Flag                      | Default  | Description                   |
| ------------------------- | -------- | ----------------------------- |
| `--game`                  | `zelda`  | zelda / sokoban / binary      |
| `--representation`        | `narrow` | narrow / wide / turtle        |
| `--algorithm`             | `PPO`    | PPO / A2C / SAC               |
| `--timesteps`             | `50000`  | Total training steps          |
| `--n-steps`               | `128`    | Steps per update              |
| `--batch-size`            | `64`     | Minibatch size                |
| `--lr`                    | `2.5e-4` | Learning rate                 |
| `--n-envs`                | `1`      | Parallel environments         |
| `--device`                | `auto`   | cpu / cuda / auto             |
| `--checkpoint-freq`       | `1000`   | Steps between saves           |
| `--sokoban-penalty`       | `25.0`   | Unsolvable level penalty      |
| `--no-solvability-tuning` | —        | Disable reward weight tuning  |
| `--evaluate`              | —        | Run evaluation after training |
| `--load-model`            | —        | Path to pre-trained `.zip`    |
| `--experiment-name`       | —        | Custom experiment name        |

**Imports:** `utils.ResourceMonitor`, `utils.TrainingLogger`, `wrappers.pcgrl_env.make_pcgrl_env`, `stable_baselines3.{PPO, A2C, SAC}`

---

### train_backward.py

**Purpose:** Backward-generation training for Sokoban. Constructs levels by reverse-engineering from a known solved state, providing the strongest solvability guarantee in the project. Applies `BackwardRewardWrapper` over the standard environment.

> ⚠ The `sokoban-reverse` gym environment ID registration has a `TODO`. The `BackwardRewardWrapper` is applied over the standard env as a workaround. Check `gym_pcgrl/__init__.py` before attempting to register the reverse env directly.

**Key class:** `BackwardRewardWrapper` — reverses the generation direction so the agent edits backward from a solvable terminal state.

**Outputs:** `checkpoints/<experiment>/final_model.zip`, `logs/<experiment>.csv`

**CLI:**

```powershell
python.exe train_backward.py --game sokoban --timesteps 50000 --device auto
```

---

### maml_trainer.py

**Purpose:** Model-Agnostic Meta-Learning (MAML, Finn et al. 2017) for PCGRL. Trains a policy that can rapidly adapt to new game/representation/reward combinations with few gradient steps. Supports first-order MAML (FOMAML, default) and full second-order MAML.

#### Classes

**`MAMLTrainer`**

Outer training loop. Manages the task distribution, policy, meta-optimizer, and checkpoint saving.

```python
MAMLTrainer(
    games=["zelda", "sokoban", "binary"],
    representations=["narrow", "wide", "turtle"],
    meta_lr=1e-3,            # Outer loop learning rate (β)
    inner_lr=0.01,           # Inner loop learning rate (α)
    inner_steps=5,           # Gradient steps per inner loop (K)
    meta_batch_size=4,       # Tasks per meta-update
    n_trajectories=128,      # Steps per trajectory rollout
    total_meta_iterations=500,
    first_order=True,        # False → full second-order MAML (slower)
    device="auto",
    experiment_name=None,
    log_dir="logs",
    checkpoint_dir="checkpoints",
)
```

| Method                                      | Description                                                             |
| ------------------------------------------- | ----------------------------------------------------------------------- |
| `inner_loop(task)`                          | Adapts policy to a single task; returns `(adapted_params, loss)`        |
| `meta_update(tasks)`                        | Runs inner loop for each task, computes meta-loss, updates meta-weights |
| `train()`                                   | Full meta-training loop; saves best/periodic/final checkpoints          |
| `adapt_to_new_task(task, adaptation_steps)` | Post-training fast adaptation; returns adapted `MAMLPolicy` copy        |
| `load_checkpoint(path)`                     | Restores policy and optimizer state from `.pt` file                     |

**Outputs:** `checkpoints/<experiment>/best_meta_model.pt`, `meta_model_iter_N.pt`, `final_meta_model.pt`, `logs/<experiment>.csv`

---

**`MAMLPolicy`** _(extends `nn.Module`)_

Actor-critic MLP supporting functional forward passes — required so inner-loop gradient updates can be applied to an explicit parameter dict without modifying `self.parameters()`.

```python
MAMLPolicy(
    obs_dim: int,
    action_dim: int,
    hidden_sizes: List[int] = [64, 64],
)
```

| Method                                 | Description                                                |
| -------------------------------------- | ---------------------------------------------------------- |
| `forward(obs)`                         | Standard forward pass → `(action_logits, value)`           |
| `get_action(obs, deterministic=False)` | Sample or argmax action from policy                        |
| `functional_forward(obs, params)`      | Forward with external OrderedDict params (MAML inner loop) |

> **Patch P1:** `functional_forward` uses `_forward_network()` which sorts layer IDs numerically from param keys, replacing fragile string-index iteration. Safe for any number of hidden layers.

---

**`TaskDistribution`**

Generates diverse (game, representation, reward_weights, change_percentage) task configs for meta-learning.

```python
TaskDistribution(
    games=["zelda", "sokoban", "binary"],
    representations=["narrow", "wide", "turtle"],
)
```

| Method                                   | Description                                                                          |
| ---------------------------------------- | ------------------------------------------------------------------------------------ |
| `sample_tasks(n_tasks, fixed_game=None)` | Returns list of task config dicts                                                    |
| `create_env(task, resource_monitor)`     | Builds `DummyVecEnv` for the task, with `SokobanDeadlockGuardrail` for Sokoban tasks |

**Task dict schema:**

```python
{
    "game": "sokoban",
    "representation": "narrow",
    "reward_weights": {"dist-win": 2.0, "sol-length": 1.0, "ratio": 1.0},
    "change_percentage": 0.45,
}
```

---

**`SokobanDeadlockGuardrail`** _(extends `gym.Wrapper`)_

Per-step wrapper that subtracts a dense deadlock penalty for every crate in a geometrically unresolvable position. Applied inside `TaskDistribution.create_env()` for Sokoban tasks only.

```python
SokobanDeadlockGuardrail(env, deadlock_penalty=1.5)
```

Detects corner deadlocks and dead-square positions via `sokoban_utils.check_sokoban_deadlock()` and `compute_dead_squares()`. Crates already on targets are never penalized. Uses old 4-tuple `step()` API for gym-pcgrl compatibility.

---

**`DictFlattenWrapper`** _(extends `gym.Wrapper`)_

Flattens a `Dict` observation space into a 1-D `Box` for compatibility with Stable-Baselines3 and the MAML policy MLP. Also present in `rlhf_trainer.py`.

---

#### Module-level functions

| Function               | Signature                                | Description                                                     |
| ---------------------- | ---------------------------------------- | --------------------------------------------------------------- |
| `collect_trajectories` | `(env, policy, n_steps, device, params)` | Rolls out policy for `n_steps`; returns dict of stacked tensors |
| `compute_policy_loss`  | `(trajectories, policy, params)`         | REINFORCE with GAE baseline + value loss + entropy bonus        |

---

#### CLI Usage

```powershell
python.exe maml_trainer.py --games zelda sokoban binary --meta-batch 4 --iterations 500 --device auto
python.exe maml_trainer.py --games sokoban --second-order --iterations 200 --device cuda
```

**Key CLI flags:**

| Flag               | Default   | Description                 |
| ------------------ | --------- | --------------------------- |
| `--games`          | all three | Games for task distribution |
| `--meta-lr`        | `1e-3`    | Outer loop LR (β)           |
| `--inner-lr`       | `0.01`    | Inner loop LR (α)           |
| `--inner-steps`    | `5`       | Gradient steps per task (K) |
| `--meta-batch`     | `4`       | Tasks per meta-update       |
| `--iterations`     | `500`     | Total meta-iterations       |
| `--n-trajectories` | `128`     | Steps per rollout           |
| `--second-order`   | —         | Full second-order MAML      |

**Imports:** `utils.{ResourceMonitor, TrainingLogger, create_checkpoint_dir}`, `wrappers.pcgrl_env.make_pcgrl_env`, `sokoban_utils.{check_sokoban_deadlock, compute_dead_squares}`

---

### rlhf_trainer.py

**Purpose:** Reinforcement Learning from Human Feedback 4-phase pipeline for PCGRL. Learns a Bradley-Terry reward model from pairwise level preferences, then fine-tunes a PPO policy against a blended environment + human-preference reward.

**Blended reward formula:**

```
R_final = (1 - w) * R_env  +  w * R_human      (default w = 0.5)
```

#### Classes

**`RLHFTrainer`**

Top-level orchestrator for the full 4-phase pipeline.

```python
RLHFTrainer(
    game="zelda",
    representation="narrow",
    base_model_path=None,         # Pre-trained PPO .zip to fine-tune (None = random init)
    rlhf_weight=0.5,              # Human preference reward weight (w)
    reward_model_lr=1e-3,
    reward_model_epochs=100,
    ppo_timesteps=50000,
    device="auto",
    experiment_name=None,
    log_dir="logs",
    checkpoint_dir="checkpoints",
)
```

| Method                                                      | Description                                                        |
| ----------------------------------------------------------- | ------------------------------------------------------------------ |
| `generate_levels_for_feedback(n_levels=50)`                 | Phase 1: roll out current/random policy                            |
| `collect_preferences(levels, n_comparisons, use_synthetic)` | Phase 2: gather pairwise labels                                    |
| `train_reward_model()`                                      | Phase 3: train Bradley-Terry model; saves `reward_model.pt`        |
| `fine_tune_with_rlhf(reward_model=None)`                    | Phase 4: PPO fine-tune with blended reward; saves `rlhf_model.zip` |
| `run_full_pipeline(n_levels, n_comparisons, use_synthetic)` | Runs all 4 phases end-to-end                                       |

**Outputs:** `checkpoints/<experiment>/reward_model.pt`, `checkpoints/<experiment>/rlhf_model.zip`, `data/preferences/<game>/preferences.json`, `logs/<experiment>.csv`

---

**`PreferenceCollector`**

Persists pairwise level preferences to `preferences.json`. Supports interactive CLI collection and synthetic auto-generation.

```python
PreferenceCollector(save_path="data/preferences")
```

| Method                                                        | Description                                                                |
| ------------------------------------------------------------- | -------------------------------------------------------------------------- |
| `add_preference(level_a, level_b, preference, metadata)`      | Record one comparison. `preference`: 0.0 = A wins, 1.0 = B wins, 0.5 = tie |
| `collect_interactive(levels, game, n_comparisons)`            | CLI-based human annotation session                                         |
| `generate_synthetic_preferences(levels, n_comparisons, game)` | Auto-generate labels based on content metrics                              |
| `save()`                                                      | Write `preferences.json` to disk                                           |

**preferences.json entry schema:**

```json
{
  "level_a": [[...], ...],
  "level_b": [[...], ...],
  "preference": 0.0,
  "metrics_a": {"diversity": 0.12, "complexity": 0.88, "size": 100, "unique_tiles": 3},
  "metrics_b": {"diversity": 0.15, "complexity": 0.91, "size": 100, "unique_tiles": 4},
  "metadata": {},
  "timestamp": "2025-01-01T00:00:00"
}
```

---

**`RewardModel`** _(extends `nn.Module`)_

Bradley-Terry reward model. MLP that scores a flattened level observation. Used by `RLHFRewardWrapper` to compute the human-preference reward signal.

---

**`RewardModelTrainer`**

Trains `RewardModel` on collected preferences using binary cross-entropy loss.

---

**`RLHFRewardWrapper`** _(extends `gym.Wrapper`)_

Wraps the PCGRL environment to inject the blended reward at each step:

```
R_final = (1 - rlhf_weight) * R_env  +  rlhf_weight * reward_model(obs)
```

Adds `env_reward` and `human_reward` keys to the `info` dict for per-step monitoring.

---

**`RLHFCallback`** _(extends `BaseCallback`)_

Training callback for the RLHF fine-tuning phase. Logs env vs human reward split every 20 episodes.

---

#### Module-level functions

| Function          | Signature                                              | Description                                             |
| ----------------- | ------------------------------------------------------ | ------------------------------------------------------- |
| `generate_levels` | `(game, representation, n_levels, model_path, device)` | Phase 1 standalone function; returns `List[np.ndarray]` |

---

#### CLI Usage

```powershell
# Full pipeline (synthetic preferences)
python.exe rlhf_trainer.py --game sokoban --synthetic --n-levels 50 --n-comparisons 50 --device auto

# Interactive human annotation session
python.exe rlhf_trainer.py --game zelda --interactive --n-levels 30 --n-comparisons 30

# Resume from saved preferences
python.exe rlhf_trainer.py --game sokoban --use-existing-preferences --device cuda

# Reward model only (skip fine-tuning)
python.exe rlhf_trainer.py --game sokoban --synthetic --reward-model-only
```

**Key CLI flags:**

| Flag                         | Default | Description                                   |
| ---------------------------- | ------- | --------------------------------------------- |
| `--game`                     | `zelda` | zelda / sokoban / binary                      |
| `--base-model`               | —       | Pre-trained PPO `.zip` to fine-tune           |
| `--rlhf-weight`              | `0.5`   | Human reward blend weight (0–1)               |
| `--n-levels`                 | `50`    | Levels to generate for feedback               |
| `--n-comparisons`            | `50`    | Pairwise comparisons to collect               |
| `--synthetic`                | —       | Auto-generate preferences                     |
| `--interactive`              | —       | Human annotation via CLI                      |
| `--timesteps`                | `50000` | PPO fine-tuning steps                         |
| `--reward-epochs`            | `100`   | Reward model training epochs                  |
| `--reward-model-only`        | —       | Skip PPO fine-tuning                          |
| `--use-existing-preferences` | —       | Skip generation, use saved `preferences.json` |

**Imports:** `utils.{ResourceMonitor, TrainingLogger, create_checkpoint_dir}`, `wrappers.pcgrl_env.make_pcgrl_env`, `wrappers.helper.calculate_content_metrics`

---

## 3. Environment Layer

### wrappers / pcgrl_env.py

**Purpose:** Main environment integration boundary. Wraps gym-pcgrl environments with resource-aware reward shaping and optional solvability enforcement.

#### `make_pcgrl_env()`

Factory function — the single entry point for creating any PCGRL environment in this project.

```python
make_pcgrl_env(
    game="zelda",
    representation="narrow",
    resource_monitor=None,
    ram_penalty_weight=0.2,
    cpu_penalty_weight=0.1,
    gpu_penalty_weight=0.1,
    max_complexity=10,
    min_complexity=3,
) -> gym.Env
```

**Build order (wrapper chain, innermost → outermost):**

```
gym_pcgrl base env  (e.g. zelda-narrow-v0)
    └── solvability_config.adjust_param()   [Sokoban/Zelda reward weights]
    └── SokobanSolvabilityWrapper           [Sokoban only]
    └── ResourceAwarePCGRLWrapper           [all games]
```

---

**`ResourceAwarePCGRLWrapper`** _(extends `gym.Wrapper`)_

The core resource-penalty wrapper. On every `step()`:

1. Calls inner env step to get `raw_reward`
2. Reads CPU, RAM, GPU from `ResourceMonitor`
3. Computes shaped reward:
   ```
   shaped_reward = raw_reward
       - max(0, RAM%  - 78) * 0.2
       - max(0, CPU%  - 70) * 0.1
       - max(0, GPU%  - 70) * 0.1
   ```
4. Tracks `env_complexity` (bounded 3–10), adjusting up when resources are low and down when high
5. Injects penalty breakdown into `info` dict

```python
ResourceAwarePCGRLWrapper(
    env,
    resource_monitor,
    ram_penalty_weight=0.2,
    cpu_penalty_weight=0.1,
    gpu_penalty_weight=0.1,
    max_complexity=10,
    min_complexity=3,
)
```

**`info` dict keys added by this wrapper:**

| Key              | Type  | Description                     |
| ---------------- | ----- | ------------------------------- |
| `ram_penalty`    | float | RAM overage penalty this step   |
| `cpu_penalty`    | float | CPU overage penalty this step   |
| `gpu_penalty`    | float | GPU overage penalty this step   |
| `total_penalty`  | float | Sum of all resource penalties   |
| `env_reward`     | float | Raw reward before penalties     |
| `env_complexity` | int   | Current complexity level (3–10) |

> **Known limitation:** `env_complexity` is tracked and reported but not yet wired into actual PCGRL map parameters (e.g. max crate count). This is an open item.

---

### wrappers / helper.py

**Purpose:** VGLC level parsing, content quality metrics, and level I/O utilities.

#### Functions

| Function                    | Signature                            | Returns            | Description                                           |
| --------------------------- | ------------------------------------ | ------------------ | ----------------------------------------------------- |
| `parse_vglc_level`          | `(file_path: str)`                   | `np.ndarray`       | Parses `.txt` or `.json` VGLC level file              |
| `load_vglc_levels`          | `(data_dir: str, game: str)`         | `List[np.ndarray]` | Loads all levels for a game from VGLC directory       |
| `tile_diversity`            | `(level: np.ndarray)`                | `float`            | Unique tiles / total tiles (0–1)                      |
| `pattern_complexity`        | `(level: np.ndarray, window_size=3)` | `float`            | Unique 3×3 patterns / max possible patterns           |
| `calculate_content_metrics` | `(level: np.ndarray)`                | `Dict[str, float]` | Returns `{diversity, complexity, size, unique_tiles}` |
| `save_level`                | `(level, filepath, format="npy")`    | —                  | Saves level as `.npy`, `.txt`, or `.json`             |
| `load_level`                | `(filepath: str)`                    | `np.ndarray`       | Loads level from `.npy`, `.txt`, or `.json`           |

---

## 4. Inference Modules

### inference.py

**Purpose:** Standard level generation from a trained Stable-Baselines3 checkpoint.

**Key class:** `LevelGenerator`

```python
LevelGenerator(model_path, game="zelda", representation="narrow", device="auto")
```

| Method                            | Description                            |
| --------------------------------- | -------------------------------------- |
| `generate(n_levels=10)`           | Returns list of generated level arrays |
| `save_levels(levels, output_dir)` | Saves levels to disk                   |

**CLI:**

```powershell
python.exe inference.py checkpoints/<exp>/final_model.zip --n-levels 10
```

---

### inference_timed.py

**Purpose:** Timed level generation with per-level metrics logged to CSV. Primary tool for gathering inference performance data for the paper.

**Key class:** `TimedLevelGenerator`

```python
TimedLevelGenerator(model_path, game="zelda", representation="narrow", device="auto")
```

| Method                               | Description                            |
| ------------------------------------ | -------------------------------------- |
| `generate_timed(n_levels, log_file)` | Generates levels and writes timing CSV |

**Output CSV columns:** `level_id`, `time_seconds`, `steps_taken`, `reward`

**CLI:**

```powershell
python.exe inference_timed.py checkpoints/<exp>/final_model.zip --game sokoban --n-levels 20 --log-file inference_timing.csv --device auto
```

---

### maml_inference_timed.py

**Purpose:** Timed inference from a MAML `.pt` checkpoint. Supports running directly from meta-weights or performing inner-loop adaptation steps before generation.

**Key class:** `MAMLPolicyWrapper`

```python
MAMLPolicyWrapper(checkpoint_path, game, device="auto")
```

| Method                                            | Description                                                  |
| ------------------------------------------------- | ------------------------------------------------------------ |
| `generate(n_levels, adapt_steps=0)`               | Generates levels; `adapt_steps=0` uses meta-weights directly |
| `generate_timed(n_levels, adapt_steps, log_file)` | Timed generation with CSV output                             |

**CLI:**

```powershell
# No adaptation (fastest)
python.exe maml_inference_timed.py checkpoints/sokoban_MAML_inference/best_meta_model.pt --game sokoban --n-levels 20 --adapt-steps 0 --device auto

# With inner-loop adaptation
python.exe maml_inference_timed.py checkpoints/sokoban_MAML_inference/best_meta_model.pt --game sokoban --n-levels 20 --adapt-steps 5 --device auto
```

---

## 5. Utilities

### utils.py

**Purpose:** Resource monitoring and training logging. Used by every training and inference script.

#### `ResourceMonitor`

Polls CPU, RAM, and GPU metrics via `psutil` and `pynvml`. Gracefully degrades to zero-valued GPU fields when NVIDIA monitoring is unavailable.

```python
ResourceMonitor(use_gpu=True)
```

| Method                    | Signature           | Returns            | Description                              |
| ------------------------- | ------------------- | ------------------ | ---------------------------------------- |
| `get_resources`           | `()`                | `Dict[str, float]` | Current snapshot of all resource metrics |
| `check_resource_pressure` | `(thresholds=None)` | `Tuple[bool, str]` | True if any metric exceeds threshold     |

**`get_resources()` return dict keys:**

| Key                | Source | Description               |
| ------------------ | ------ | ------------------------- |
| `cpu_percent`      | psutil | CPU usage %               |
| `ram_percent`      | psutil | RAM usage %               |
| `ram_used_gb`      | psutil | RAM used in GB            |
| `ram_available_gb` | psutil | RAM available in GB       |
| `gpu_util_percent` | pynvml | GPU compute utilization % |
| `gpu_mem_used_mb`  | pynvml | GPU VRAM used in MB       |
| `gpu_mem_total_mb` | pynvml | GPU VRAM total in MB      |
| `gpu_mem_percent`  | pynvml | GPU VRAM usage %          |

**Default pressure thresholds:** CPU 90%, RAM 90%, GPU mem 85%, GPU util 85%

**Penalty thresholds** (in `ResourceAwarePCGRLWrapper`): RAM 78%, CPU 70%, GPU 70%

---

#### `TrainingLogger`

Accumulates per-step training data and saves to CSV.

```python
TrainingLogger(log_dir="logs", experiment_name=None)
```

| Method                               | Signature                                                    | Description                                                 |
| ------------------------------------ | ------------------------------------------------------------ | ----------------------------------------------------------- |
| `log_step`                           | `(reward, resources, content_metrics, action, penalty_info)` | Record one step                                             |
| `log_episode_end`                    | `()`                                                         | Increment episode counter                                   |
| `save`                               | `()`                                                         | Write accumulated data to `logs/<experiment>.csv`           |
| `get_stats`                          | `()`                                                         | Returns summary dict (mean/std/min/max reward, total steps) |
| `print_stats`                        | `()`                                                         | Prints summary to stdout                                    |
| `analyze_action_penalty_correlation` | `(recent_steps=1000)`                                        | Per-action average penalty breakdown                        |
| `get_high_penalty_actions`           | `(top_n=5, penalty_type, recent_steps)`                      | Actions with highest average penalty                        |

**Output CSV columns:** `episode`, `step`, `reward`, `timestamp`, `cpu_percent`, `ram_percent`, `ram_used_gb`, `ram_available_gb`, `gpu_util_percent`, `gpu_mem_used_mb`, `gpu_mem_total_mb`, `gpu_mem_percent`, `content_diversity`, `content_complexity`, `content_size`, `content_unique_tiles`, `action`, `penalty_ram_penalty`, `penalty_cpu_penalty`, `penalty_gpu_penalty`, `penalty_total_penalty`

---

#### Module-level functions

| Function                 | Signature                                  | Returns | Description                                   |
| ------------------------ | ------------------------------------------ | ------- | --------------------------------------------- |
| `calculate_fps`          | `(start_time, steps)`                      | `float` | Steps per second                              |
| `estimate_training_time` | `(current_steps, total_steps, start_time)` | `str`   | Formatted `HH:MM:SS` remaining                |
| `create_checkpoint_dir`  | `(base_dir, experiment_name)`              | `str`   | Creates and returns checkpoint directory path |

---

### model.py

**Purpose:** Thin compatibility wrapper for loading and saving Stable-Baselines3 models. Adds `gym-pcgrl` to `sys.path` for backward compatibility.

#### Functions

| Function     | Signature                                      | Returns   | Description                                 |
| ------------ | ---------------------------------------------- | --------- | ------------------------------------------- |
| `load_model` | `(model_path, algorithm="PPO", device="auto")` | SB3 model | Loads a PPO or A2C `.zip` checkpoint        |
| `save_model` | `(model, save_path)`                           | —         | Saves model, creating parent dirs if needed |

> Most scripts load models directly via `PPO.load()` / `A2C.load()`. Use `model.py` when algorithm type is not known at call time.

---

### sokoban_utils.py

**Purpose:** Sokoban-specific level validation, deadlock detection, and solvability checking. The active validation layer — `sokoban_utils_backup.py` is broken and must not be imported.

**Tile encoding (authoritative):**

| Value | Tile        |
| ----- | ----------- |
| `0`   | Empty floor |
| `1`   | Wall        |
| `2`   | Player      |
| `3`   | Crate       |
| `4`   | Target      |

#### Classes

**`SokobanSolvabilityWrapper`** _(extends `gym.Wrapper`)_

Validates and repairs Sokoban levels on `reset()`. Does not enforce hard solvability — applies structural fixes only.

```python
SokobanSolvabilityWrapper(
    env,
    enforce_all_rules=True,
    verbose=False,
    unsolvable_penalty=0,        # Legacy param, unused
    min_solution_length=0,       # Legacy param, unused
    max_solution_length=100,     # Legacy param, unused
    terminate_on_unsolvable=False,  # Legacy param, unused
)
```

| Method            | Description                                                                           |
| ----------------- | ------------------------------------------------------------------------------------- |
| `reset(**kwargs)` | Validates/repairs map; updates `env.unwrapped._rep._map`                              |
| `get_stats()`     | Returns `{total_resets, total_fixes, player_fixes, deadlock_removals, balance_fixes}` |

---

#### Module-level functions

| Function                     | Signature                                | Returns                  | Description                                                 |
| ---------------------------- | ---------------------------------------- | ------------------------ | ----------------------------------------------------------- |
| `get_reachable_positions`    | `(level, start_pos, walkable_tiles)`     | `set`                    | BFS reachability from a position                            |
| `compute_dead_squares`       | `(level, target_positions)`              | `set`                    | Reverse-BFS dead square detection                           |
| `check_sokoban_deadlock`     | `(level, crate_pos, dead_squares)`       | `bool`                   | Corner + dead-square deadlock check for one crate           |
| `validate_and_fix_sokoban`   | `(level, min_crates, enforce_all_rules)` | `(np.ndarray, dict)`     | Full validation and repair pass                             |
| `check_solvability`          | `(level, budget)`                        | `(bool, solution, dist)` | BFS + A\* solver (offline use only — expensive)             |
| `is_valid_sokoban`           | `(level)`                                | `(bool, str)`            | Quick structural check (player count, crate/target balance) |
| `print_level_stats`          | `(level, title, verbose)`                | —                        | Debug print of level statistics                             |
| `check_crate_pushability`    | `(level, crate_pos)`                     | `bool`                   | Whether a crate can be pushed in any direction              |
| `check_crate_to_target_path` | `(level, crate_pos, target_positions)`   | `bool`                   | Path exists from crate to any target                        |

> `check_solvability()` uses the gym-pcgrl BFS/A\* engine with a configurable iteration budget (default 5,000 from `solvability_config.py`). It is intended for **post-hoc offline evaluation** of generated level batches, not as an inline training signal.

---

### config_hardware.py

**Purpose:** Hardware presets for different deployment scenarios.

**Key exports:** `PRESET_FAST`, `PRESET_LIGHT`

`PRESET_FAST` — optimized for the project development machine (i5-13500, RTX 3060 Ti, 16 GB RAM):

- `n_envs=6`, `device="cuda"`, `n_steps=128`, `batch_size=64`

`PRESET_LIGHT` — low-resource fallback (CPU-only, 8 GB RAM):

- `n_envs=2`, `device="cpu"`, `n_steps=64`, `batch_size=32`

---

### solvability_config.py

**Purpose:** Per-game reward weight tuning. Applied via `env._prob.adjust_param()` during environment setup in `make_pcgrl_env()`.

**Key function:** `get_solvability_config(game)` → returns reward weight dict for the given game.

**Sokoban notable change:** `sol-length` weight raised from 3 → 5 to amplify the long-horizon pathfinding signal once structural deadlocks are penalized by the guardrail.

**Zelda weights raised:** `path-length`, `regions`, `nearest-enemy`, `player`, `key`, `door`

---

## 6. Analysis and Visualization

| File                          | Key Class / Function | Output               | Description                                                       |
| ----------------------------- | -------------------- | -------------------- | ----------------------------------------------------------------- |
| `generate_paper_figures.py`   | —                    | `figures/` PDF + PNG | ACM TOG publication figures at 300 DPI                            |
| `analyze_maml_results.py`     | —                    | `figures/maml/`      | Meta-loss convergence, resource panels, reward proxy distribution |
| `compare_approaches.py`       | —                    | stdout / CSV         | Forward vs backward Sokoban generation comparison                 |
| `analyze_action_penalties.py` | —                    | stdout / plots       | Correlation between agent actions and resource penalties          |
| `visualize_levels.py`         | `save_level_image()` | PNG files            | Level rendering and image export                                  |
| `architecture_diagram.py`     | —                    | ASCII / image        | System architecture diagram                                       |

**CLI examples:**

```powershell
python.exe generate_paper_figures.py --demo
python.exe compare_approaches.py --forward-dir generated_levels/sokoban_forward --backward-dir generated_levels/sokoban_backward
python.exe analyze_action_penalties.py
python.exe visualize_levels.py
```

---

## 7. Test Suite

### test/test.py

Primary test suite. Run with:

```powershell
python.exe test/test.py
```

**5 test categories (all expected to pass):**

| Category          | What it tests                                                 |
| ----------------- | ------------------------------------------------------------- |
| Resource Monitor  | `ResourceMonitor.get_resources()` returns valid dict          |
| Training Logger   | `TrainingLogger.log_step()` and `save()`                      |
| VGLC Parsing      | `wrappers.helper.parse_vglc_level()` and `load_vglc_levels()` |
| Content Metrics   | `calculate_content_metrics()` output shape and value ranges   |
| PCGRL Environment | `make_pcgrl_env()` creates env with valid obs/action spaces   |

### Standalone Test Scripts

| Script                            | Tests                                                                        |
| --------------------------------- | ---------------------------------------------------------------------------- |
| `test_action_space.py`            | Action space dimensionality across games and representations                 |
| `test_obs_shape.py`               | Observation shape consistency after `DictFlattenWrapper`                     |
| `test_sokoban_solvability.py`     | `validate_and_fix_sokoban()` edge cases (multi-player, deadlocks, imbalance) |
| `test_solvability_integration.py` | End-to-end: `SokobanSolvabilityWrapper` on live env                          |
| `test_solver_integration.py`      | `check_solvability()` against known-solvable/unsolvable levels               |
| `test_trust_model.py`             | Raw model output under `--trust-model` inference path                        |

---

## 8. gym-pcgrl Vendored Layer

Located at `gym-pcgrl/gym_pcgrl/`. Do not modify — treat as a read-only dependency.

| File                           | Purpose                                                              |
| ------------------------------ | -------------------------------------------------------------------- |
| `__init__.py`                  | Registers all gym environment IDs (e.g. `zelda-narrow-v0`)           |
| `envs/pcgrl_env.py`            | Base `PCGRLEnv` gym class; combines problem + representation         |
| `envs/helper.py`               | Tile histograms, Dijkstra, BFS, longest-path, random map gen         |
| `envs/probs/zelda_prob.py`     | Zelda reward: player, key, door, enemies, path length                |
| `envs/probs/sokoban_prob.py`   | Sokoban reward: player, crates, targets, distance-to-win, sol-length |
| `envs/probs/binary_prob.py`    | Binary reward: connected regions, path length                        |
| `envs/probs/sokoban/engine.py` | BFS, DFS, A\* Sokoban solver — used by `check_solvability()`         |
| `envs/reps/narrow_rep.py`      | Narrow representation: one tile edit per step                        |
| `envs/reps/wide_rep.py`        | Wide representation: full map edit per step                          |
| `envs/reps/turtle_rep.py`      | Turtle representation: agent moves and edits current tile            |

**Registered environment IDs (selected):**

| ID                  | Game    | Representation |
| ------------------- | ------- | -------------- |
| `zelda-narrow-v0`   | Zelda   | narrow         |
| `zelda-wide-v0`     | Zelda   | wide           |
| `sokoban-narrow-v0` | Sokoban | narrow         |
| `sokoban-wide-v0`   | Sokoban | wide           |
| `binary-narrow-v0`  | Binary  | narrow         |

---

## 9. Dashboard

### dashboard/dashboard.py

Streamlit application. Exposes training, inference, level visualization, logs, and checkpoint browsing through a browser UI. Runs Python subprocesses and streams output back to the interface.

**Launch:**

```powershell
python.exe -m streamlit run dashboard/dashboard.py
# Binds to 0.0.0.0:8501
```

---

## 10. Dependency Map

```
train.py
    ├── utils.py                   (ResourceMonitor, TrainingLogger)
    ├── wrappers/pcgrl_env.py      (make_pcgrl_env)
    │       ├── sokoban_utils.py   (SokobanSolvabilityWrapper)
    │       ├── solvability_config.py
    │       └── gym-pcgrl/
    └── stable_baselines3

maml_trainer.py
    ├── utils.py
    ├── wrappers/pcgrl_env.py
    │       └── (same chain as above)
    ├── sokoban_utils.py           (check_sokoban_deadlock, compute_dead_squares)
    └── torch

rlhf_trainer.py
    ├── utils.py
    ├── wrappers/pcgrl_env.py
    ├── wrappers/helper.py         (calculate_content_metrics)
    ├── stable_baselines3
    └── torch

inference_timed.py
    ├── utils.py
    ├── wrappers/pcgrl_env.py
    └── stable_baselines3

wrappers/pcgrl_env.py
    ├── utils.py                   (ResourceMonitor)
    ├── sokoban_utils.py           (SokobanSolvabilityWrapper)
    ├── solvability_config.py      (get_solvability_config)
    └── gym-pcgrl/gym_pcgrl       (base envs)
```

---

## 11. Import Notes

### Always use the venv interpreter

```powershell
# Correct
d:/Work/thesis/RAPCG-MetaRL/pcg_env/Scripts/python.exe <script>

# Never use bare python — may resolve to system interpreter
python <script>
```

### gym-pcgrl path injection

Every module that imports from `gym_pcgrl` must add it to `sys.path` before importing:

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gym-pcgrl"))
import gym_pcgrl
```

This is handled automatically in `wrappers/pcgrl_env.py`. Scripts that import `make_pcgrl_env()` do not need to repeat this.

### Root model.py vs gym-pcgrl model.py

There are two `model.py` files:

| Path                 | Purpose                                                         |
| -------------------- | --------------------------------------------------------------- |
| `model.py` (root)    | Load/save helpers for SB3 `.zip` checkpoints                    |
| `gym-pcgrl/model.py` | Original PCGRL CNN policy (legacy, not used in active training) |

The root-level `model.py` adds `gym-pcgrl/` to `sys.path` on import. Import only the root-level version from project scripts.

### Do not import these files

| File                       | Reason                                           |
| -------------------------- | ------------------------------------------------ |
| `sokoban_utils_backup.py`  | Not parseable Python — broken backup             |
| `quickstart.py` (directly) | Uses bare `python` via `os.system()` — will fail |

### CUDA reinstall if needed

```powershell
python.exe -m pip uninstall torch torchvision torchaudio
python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
