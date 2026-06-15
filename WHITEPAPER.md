# RAPCG-MetaRL White Paper

Resource-Aware Procedural Content Generation via Meta-Reinforcement Learning

## Executive Summary

RAPCG-MetaRL is a research framework for training procedural content generation agents that account for both game-level quality and machine resource cost. The system extends PCGRL environments with live CPU, RAM, and GPU memory telemetry, then injects that signal into the reinforcement learning reward path. The result is a PCG training and inference stack where generated content is scored against domain constraints while the agent also receives negative feedback when resource usage crosses configured limits.

The current implementation focuses on Zelda and Sokoban PCGRL tasks, with Binary support exposed through the same environment factory. Stable-Baselines3 provides PPO and A2C training, PyTorch supports MAML and RLHF extensions, and `gym-pcgrl` supplies the underlying problem definitions, representations, search engines, and reward statistics. Training produces Stable-Baselines3 checkpoint archives, step-level CSV logs, and generated levels in NumPy, text, and image form. A Streamlit dashboard and cross-platform command scripts provide repeatable access to training, inference, analysis, and visualization.

The strongest completed comparison in the repository is PPO versus A2C on Zelda and Sokoban. PPO shows higher mean episode reward than A2C in the logged comparison table, with Zelda PPO improving from -8.54 early reward to +11.84 late reward and Sokoban PPO improving from +2.84 to +6.66. CUDA PPO runs reported average CPU use below 4 percent and average RAM use below 50 percent for the Zelda and Sokoban runs documented in `table.md`. MAML and RLHF code paths exist. A completed MAML training run on Sokoban narrow (500 iterations, meta-batch 2, inner steps 3) produced a best meta-loss of 8.1543 at iteration 31, with average CPU at 5.2%, average RAM at 57.8%, and average GPU at 14.8%. Full MAML training metrics are documented in the MAML Training Results section. RLHF training results are not yet fully represented in the main comparison table.

## The Systemic Problem

Procedural content generation through reinforcement learning has a practical cost problem. A policy can learn to improve map statistics while ignoring the hardware load produced by the training or generation process. That is a poor match for heterogeneous game platforms, where the same content system may need to run on a desktop GPU, a CPU-only laptop, or an integrated graphics device.

Classic PCGRL treats the content task as a Markov decision process. The agent edits a tile map, receives rewards from game-specific statistics, and terminates when the task-specific stopping condition is met. That framing is useful, but it leaves resource use outside the agent's feedback loop. Hardware pressure appears only as wall-clock delay, memory exhaustion, or operator tuning after a run fails.

Sokoban adds another difficulty. A generated Sokoban map can satisfy simple tile-count constraints while still being unsolvable because of deadlocked crates, unreachable targets, unpushable objects, or disconnected regions. A generator that is rewarded only for local tile statistics may produce plausible-looking but unusable puzzles. The repository addresses that problem with tuned reward weights, solver-backed solvability checks, validation utilities, and an optional backward-generation path that starts from a solved state. For meta-learning architectures like MAML, this local minimum is particularly acute: policies quickly maximize dense rewards for matching item counts while stalling out on sparse pathfinding rewards. To resolve this, the framework introduces a dense, per-step structural guardrail that penalizes deadlocked tile configurations in real-time, preventing the policy from settling into un-solvable layout plateaus.

## Core Engineering Mechanics and Topology

RAPCG-MetaRL is organized around a feedback path from environment state, hardware telemetry, reward shaping, logging, and checkpointed learning.

The active training path is implemented through `MetaRLTrainer` in `train.py`. It creates a `ResourceMonitor`, builds one or more PCGRL environments, selects PPO, A2C, or SAC from Stable-Baselines3, and attaches a callback that logs rewards, resources, actions, and penalty terms. The command interface exposes game, representation, algorithm, timesteps, PPO step and batch parameters, parallel environment count, device selection, checkpoint cadence, evaluation, and checkpoint loading.

The environment factory in `wrappers/pcgrl_env.py` is the main integration boundary. It builds `gym-pcgrl` environment IDs such as `zelda-narrow-v0` and `sokoban-narrow-v0`, applies solvability reward configuration where available, adds the Sokoban solvability wrapper for Sokoban tasks, and finally wraps the result with `ResourceAwarePCGRLWrapper`.

The resource-aware reward equation implemented in the wrapper is:

```text
shaped_reward =
    raw_reward
    - max(0, RAM_percent - 78) * 0.2
    - max(0, CPU_percent - 70) * 0.1
    - max(0, GPU_memory_percent - 70) * 0.1
```

The same wrapper tracks a bounded complexity value from 3 to 10. It lowers that value when GPU memory exceeds 85 percent, RAM exceeds 90 percent, or CPU exceeds 90 percent. It raises the value when GPU memory is below 60 percent, RAM below 70 percent, and CPU below 70 percent. In the present code, that value is reported through `info["env_complexity"]`; it is not yet wired into internal PCGRL map parameters at runtime.

Hardware telemetry is supplied by `ResourceMonitor` in `utils.py`. CPU and RAM data come from `psutil`. GPU utilization, GPU memory usage, and GPU memory percentage come from `pynvml` when NVIDIA monitoring is available. When GPU monitoring cannot be initialized, the monitor returns zero-valued GPU fields instead of stopping execution.

The logging layer uses `TrainingLogger`. It records episode id, step id, reward, timestamp, hardware readings, optional content metrics, action id, and resource penalty fields. CSV output is written under `logs/`, while checkpoints are created under `checkpoints/`. The training callback writes intermediate models at the configured checkpoint frequency and writes `final_model.zip` when training ends.

### PCGRL Task Layer

The vendored `gym-pcgrl` source supplies the base PCGRL interface. It registers problem and representation combinations through `gym_pcgrl/__init__.py`. Problems include Binary, Zelda, Sokoban, Super Mario Bros, Dangerous Dave, and MiniDungeon. Representations include narrow, wide, turtle, narrowcast, narrowmulti, and turtlecast.

`gym_pcgrl/envs/pcgrl_env.py` defines the base Gym environment. It combines a problem object and a representation object, computes old and new map statistics, calculates reward deltas, and exposes reset, step, render, tile-count, and parameter-adjustment behavior. Helper functions in `gym_pcgrl/envs/helper.py` implement tile histograms, region counting, Dijkstra distance maps, longest-path estimates, reachable tile counts, random map generation, integer probability conversion, and range-based reward changes.

For Zelda, the problem class rewards single-player placement, key and door placement, connected regions, enemy count, nearest enemy distance, and player-to-key-to-door path length. For Sokoban, the problem class scores player, crate, target, region, crate-target ratio, distance to win, and solution length. The Sokoban engine includes BFS, DFS, and A\* search support.

### Solvability Path

The active Sokoban validation code is `sokoban_utils.py`. It enforces one player, balanced crate-target counts, deadlock removal, pushability checks, player reachability, crate-to-target reachability, target placement checks, and minimum crate-target pairs. Solvability checking uses the `gym-pcgrl` Sokoban engine and tries BFS followed by A\* variants with different balance settings under a configurable solver budget. The default solver budget exposed in `solvability_config.py` is 5,000 iterations.

The Sokoban environment can be wrapped by `SokobanSolvabilityWrapper`. On reset it can validate and repair the current map. During step execution it evaluates terminal or reported maps, checks validity and solvability, applies the unsolvable penalty when needed, and records statistics such as total levels checked, solvable levels, unsolvable levels, and solution lengths.

Reward tuning in `solvability_config.py` raises Zelda rewards for player, key, door, connected regions, nearest enemy distance, and path length. For Sokoban, it raises player, crate, target, connected-region, ratio, distance-to-win, and solution-length weights. Specifically, the `sol-length` weight is tightened from 3 to 5 to amplify the long-horizon pathfinding signal relative to dense item counting once structural deadlocks are successfully penalized. These weights are applied through `env._prob.adjust_param()` when the base environment exposes the expected PCGRL problem object.

### MAML Extension

The MAML implementation is in `maml_trainer.py`. It defines a task distribution over games, representations, reward-weight variations, and change percentages. It uses a PyTorch actor-critic MLP with functional forward passes so inner-loop updates can be applied to an ordered parameter dictionary. The trainer supports first-order MAML for lower compute cost and second-order MAML when requested.

The MAML path collects trajectories, computes a policy-gradient loss with generalized advantage estimation, applies inner-loop updates per task, and aggregates meta-loss across sampled tasks for the outer optimizer. To maintain structural viability during layout generation, tasks assigned to the Sokoban domain are intercepted inside `TaskDistribution.create_env()` and wrapped with a custom `SokobanDeadlockGuardrail`. This wrapper hooks into the environment step loop, runs real-time grid adjacency checks on crate tiles via `sokoban_utils`, and subtracts a dense penalty (default `-1.5`) per deadlocked crate to reshape the gradient topology away from corner traps. It writes `best_meta_model.pt`, iteration checkpoints, and `final_meta_model.pt`. Timed MAML inference in `maml_inference_timed.py` can run directly from meta-weights or perform a configurable number of adaptation steps before generation, mirroring the exact same `SokobanDeadlockGuardrail` inside its environment pipeline to ensure reward landscape and gradient consistency during inner-loop meta-adaptation.

### RLHF Extension

The RLHF pipeline in `rlhf_trainer.py` has four stages: generate candidate levels, collect pairwise preferences, train a reward model with a Bradley-Terry preference objective, and fine-tune PPO against a blended environment-plus-human reward. The preference collector supports interactive CLI labels and synthetic labels based on content metrics. Preferences are persisted as JSON under `data/preferences/<game>/preferences.json`.

The reward model is a PyTorch MLP that maps flattened level arrays to scalar rewards. The wrapper `RLHFRewardWrapper` computes the learned reward for the current observation or map, blends it with the environment reward using `rlhf_weight`, and writes `env_reward`, `human_reward`, and `blended_reward` into the environment info dictionary. The RLHF callback logs these fields with resource readings.

The repository contains a Sokoban RLHF reward-model checkpoint, but not a completed `rlhf_model.zip` in `checkpoints/sokoban_RLHF_cuda/`. The available log shows a run configured for 50,000 PPO timesteps, with fine-tuning output present through 5,248 timesteps.

### Inference, Visualization, and Analysis

`inference.py` loads PPO or A2C checkpoints, generates levels, extracts final maps from environment internals or `info`, calculates diversity and pattern complexity, and saves NumPy, text, and image outputs. `inference_timed.py` adds timing around reset, generation, extraction, validation, metrics, solvability checks, saving, and resource deltas. It writes CSV logs and a LaTeX timing table.

Visualization uses PNG tile assets when present and fallback color tiles otherwise. `visualize_levels.py` renders individual levels, grids, before-and-after comparisons, and legends. `generate_paper_figures.py` produces figure sets for generated-level showcases, training progression, resource-quality tradeoff plots, algorithm comparisons, and level statistics.

Analysis support includes action-penalty correlation in `analyze_action_penalties.py`, forward-versus-backward Sokoban comparison in `compare_approaches.py`, and an ASCII system diagram in `architecture_diagram.py`.

## Verification and Results

The repository contains a primary test suite in `test/test.py` with five categories: resource monitor, training logger, VGLC parsing, content metrics, and PCGRL environment creation. Additional targeted tests cover action spaces, observation shape handling, Sokoban solvability wrapping, solver integration, combined solvability and resource-aware behavior, and raw model output under the `--trust-model` inference path.

The following results are taken from `table.md` and repository artifacts.

### Hardware Profiles

| Component |          Training and CUDA Inference Platform |           Cross-Platform CPU Inference Platform |
| --------- | --------------------------------------------: | ----------------------------------------------: |
| CPU       | Intel i5-13500, 14 cores, 20 threads, 2.5 GHz | AMD Ryzen 5 3550H, 8 cores, 16 threads, 3.7 GHz |
| GPU/APU   |                NVIDIA RTX 3060 Ti, 8 GB GDDR6 |                       AMD RX 560x / Vega 8 iGPU |
| RAM       |                               16 GB DDR4-3600 |                           8 GB DDR4-2667 SODIMM |
| Use case  |                        Training and inference |                        CPU-based inference only |

### PPO and A2C Training Comparison

| Algorithm |  Domain | Episodes |  Duration | Mean Episode Reward | Early to Late Reward | Avg CPU | Avg RAM |
| --------- | ------: | -------: | --------: | ------------------: | -------------------: | ------: | ------: |
| A2C       |   Zelda |      492 | 128.2 min |               -8.92 |        -14.6 to -3.0 |    4.7% |   58.8% |
| A2C       | Sokoban |    1,291 |  56.3 min |                0.89 |         -1.9 to +3.2 |    4.4% |   63.2% |
| PPO       |   Zelda |      566 | 113.2 min |                5.60 |      -8.54 to +11.84 |   3.68% |  48.15% |
| PPO       | Sokoban |    2,018 | 112.6 min |                5.39 |       +2.84 to +6.66 |   3.81% |  49.37% |

These results indicate PPO reached stronger final rewards than A2C on both documented domains. The resource figures also show the configured resource penalties were not triggered in the published runs, because CPU, RAM, and GPU memory usage stayed below the wrapper thresholds.

### CUDA PPO Training Runs

| Metric                  |        Zelda PPO CUDA |      Sokoban PPO CUDA |
| ----------------------- | --------------------: | --------------------: |
| Training steps          |                20,096 |                20,096 |
| Episodes                |                   566 |                 2,018 |
| Duration                |             113.2 min |             112.6 min |
| Mean episode reward     |                  5.60 |                  5.39 |
| Episode reward range    |        -39.0 to +53.0 |        -19.0 to +34.0 |
| Early to late reward    |       -8.54 to +11.84 |        +2.84 to +6.66 |
| Average CPU             |                 3.68% |                 3.81% |
| Average RAM             |       48.15%, 7.60 GB |       49.37%, 7.79 GB |
| Average GPU utilization |                 0.86% |                 0.84% |
| Average GPU memory      | 824.4 MB, 10.06% VRAM |  815.9 MB, 9.96% VRAM |
| Peak GPU memory         | 938.2 MB, 11.45% VRAM | 880.9 MB, 10.75% VRAM |

### PPO Inference Metrics

| Metric                  | Zelda CUDA, 20 levels | Sokoban CUDA, 20 levels |        Zelda AMD CPU, 20 levels |      Sokoban AMD CPU, 20 levels |
| ----------------------- | --------------------: | ----------------------: | ------------------------------: | ------------------------------: |
| Mean generation time    |            5,176.1 ms |              1,586.3 ms |                      4,933.2 ms |                      1,442.9 ms |
| Mean inference per step |               2.64 ms |                 3.19 ms |                         1.01 ms |                         1.15 ms |
| Mean steps per level    |                  45.6 |                    13.4 |                            47.1 |                            13.1 |
| Mean total reward       |                 14.10 |                    8.25 |                           15.10 |                            5.95 |
| Mean diversity          |                0.0675 |                  0.1540 |                          0.0760 |                          0.1440 |
| Mean complexity         |                0.8300 |                  0.9667 |                          0.8800 |                          0.9667 |
| Resource note           |     7.2% to 8.3% VRAM |     10.3% to 10.6% VRAM | 7.0% mean CPU, 0.07% RAM change | 2.5% mean CPU, 0.00% RAM change |

### MAML Training Results

A completed MAML training run was conducted on Sokoban with the narrow representation. Training used first-order MAML (FOMAML) for computational efficiency. The experiment name is `sokoban_MAML_inference` and the log is available at `logs/sokoban_MAML_inference.csv`.

| Metric                 | MAML Sokoban Narrow (CUDA) |
| ---------------------- | -------------------------: |
| Iterations             |                        500 |
| Meta-batch size        |                          2 |
| Inner steps (K)        |                          3 |
| Trajectories per task  |                         64 |
| Best meta-loss         |                     8.1543 |
| Best loss at iteration |                         31 |
| Final meta-loss        |                  22386.084 |
| Mean meta-loss         |                  14307.817 |
| Loss std               |                  12450.299 |
| Average CPU            |                       5.2% |
| Average RAM            |              57.8%, ~9.2GB |
| Average GPU VRAM       |                      14.8% |
| Device                 |   NVIDIA RTX 3060 Ti, CUDA |

Publication-quality convergence analysis figures are produced by `analyze_maml_results.py` and saved under `figures/maml/sokoban_MAML_inference/`. Output includes meta-loss convergence curves, resource utilization panels, reward proxy distribution, training phase analysis, and a summary dashboard. All figures are saved as PDF (ACM TOG vector-safe) and PNG at 300 DPI.

### MAML Timed Inference Artifacts

The repository also contains a prior MAML timed inference CSV. The file `inference_timing_maml.csv` contains 1,000 rows for Sokoban MAML inference. Its measured averages are:

| Metric                  | MAML Sokoban Timed Inference |
| ----------------------- | ---------------------------: |
| Levels                  |                        1,000 |
| Mean total time         |                  1,604.63 ms |
| Mean generation time    |                  1,594.54 ms |
| Mean inference per step |                     1.879 ms |
| Mean steps per level    |                        12.54 |
| Mean total reward       |                         7.77 |
| Mean diversity          |                       0.1947 |
| Mean complexity         |                       0.9988 |

### RLHF Artifact Status

The repository contains a Sokoban preference file with 30 preferences and a reward model checkpoint at `checkpoints/sokoban_RLHF_cuda/reward_model.pt`. The RLHF log reports 30 generated levels, 30 synthetic pairwise preferences, 100 reward-model epochs, and final validation accuracy of 66.7 percent. PPO fine-tuning was configured for 50,000 timesteps, but the available log reaches 5,248 timesteps and no final `rlhf_model.zip` is present in the checkpoint directory.

| Metric                                           |     RLHF Sokoban |
| ------------------------------------------------ | ---------------: |
| Preference source                                |        Synthetic |
| Preference count                                 |               30 |
| Reward model epochs                              |              100 |
| Reward model final validation accuracy           |            66.7% |
| Fine-tuning timesteps completed in available log |            5,248 |
| Final RLHF PPO checkpoint                        | TBD, not present |
| Comparable inference result                      | TBD, not present |

## Operational Topology and Strategic Outlook

The root `Dockerfile` packages the current project on `python:3.9-slim`, installs `requirements.txt`, installs Streamlit, copies the project, installs `gym-pcgrl` in editable mode, creates runtime directories, exposes port 8501, and starts `dashboard/dashboard.py`. This container is oriented toward dashboard-driven experiment control and review.

The vendored `gym-pcgrl/Dockerfile` reflects the original PCGRL project stack on TensorFlow 1.15 GPU images and Stable Baselines 2. It is useful as historical context for the base environment, but the active RAPCG-MetaRL training path uses Stable-Baselines3.

PowerShell and Bash scripts under `console/` define repeatable command sets for tree inspection, training, testing, inference, and analysis. The training runbooks include PPO, A2C, Sokoban backward generation, Binary PPO, MAML, RLHF, checkpoint resume, low-resource fallback, and immediate evaluation commands. The inference runbooks include standard inference, timed inference, MAML inference with and without adaptation, large generation batches, visualization, and timing CSV analysis. Dashboard launchers install Streamlit if missing and start the app on `0.0.0.0:8501`.

The Streamlit dashboard exposes training, inference, level visualization, logs, checkpoints, and file discovery through a browser interface. It runs Python subprocesses and streams process output back into the UI. This makes the repository usable both as a command-line research project and as an operator-facing experiment console.

The next technical steps are clear from the code state:

1. Wire the tracked environment complexity value into actual PCGRL problem or representation parameters.
2. Complete comparable MAML training tables for Zelda, Sokoban, and Binary rather than relying only on timed inference artifacts.
3. Complete RLHF fine-tuning through the configured timestep budget, save the final PPO checkpoint, and run timed inference against the RLHF policy.
4. Replace or remove the broken archival `sokoban_utils_backup.py`, which is not parseable Python.
5. Plan a Gymnasium migration path, since logs show the current stack uses older Gym APIs through compatibility wrappers.

## Evidence Scope

This white paper is based on project-owned Python files, the vendored `gym-pcgrl` Python source, root and vendored Dockerfiles, Bash and PowerShell command scripts, documentation files, log and checkpoint inventories, `table.md`, and available CSV artifacts. The local virtual environment under `pcg_env/` was excluded from code analysis because it contains third-party installed packages rather than project source.
