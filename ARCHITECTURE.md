# RAPCG-MetaRL Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAPCG-MetaRL System                          │
│                                                                       │
│  ┌────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │  User Scripts  │  │   Wrappers      │  │  gym-pcgrl      │      │
│  │                │  │                 │  │                 │      │
│  │  • train.py    │──│  • pcgrl_env   │──│  • Zelda-v0     │      │
│  │  • inference.py│  │  • helper.py   │  │  • Sokoban-v0   │      │
│  │  • test.py     │  │                │  │  • Binary-v0    │      │
│  └────────────────┘  └─────────────────┘  └─────────────────┘      │
│          │                    │                     │                │
│          └────────────────────┴─────────────────────┘                │
│                              │                                       │
│                    ┌─────────▼─────────┐                            │
│                    │   utils.py         │                            │
│                    │                    │                            │
│                    │  • ResourceMonitor │                            │
│                    │  • TrainingLogger  │                            │
│                    └────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

## Component Architecture

### 1. Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                       Training Flow                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  [Start] ──► [Create Environment] ──► [Initialize Agent]        │
│                      │                        │                  │
│                      ▼                        ▼                  │
│            [Resource-Aware Wrapper]   [PPO/A2C Model]           │
│                      │                        │                  │
│                      └────────┬───────────────┘                  │
│                               │                                  │
│                               ▼                                  │
│                      [Training Loop]                             │
│                      │          │                                │
│                      │          ├─► [Step Environment]           │
│                      │          ├─► [Collect Experience]         │
│                      │          ├─► [Update Policy]              │
│                      │          ├─► [Monitor Resources]          │
│                      │          ├─► [Log Metrics]                │
│                      │          ├─► [Save Checkpoints]           │
│                      │          │                                │
│                      └──────────┘                                │
│                               │                                  │
│                               ▼                                  │
│                    [Save Final Model] ──► [End]                  │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Resource Monitoring System

```
┌──────────────────────────────────────────────────────────────────┐
│                    Resource Monitor                               │
├──────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │     CPU      │    │     RAM      │    │     GPU      │       │
│  │              │    │              │    │              │       │
│  │  • psutil    │    │  • psutil    │    │  • pynvml    │       │
│  │  • percent   │    │  • percent   │    │  • util %    │       │
│  │  • per-core  │    │  • used GB   │    │  • mem MB    │       │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘       │
│         │                   │                    │                │
│         └───────────────────┴────────────────────┘                │
│                             │                                     │
│                             ▼                                     │
│                  ┌────────────────────┐                           │
│                  │  Resource Metrics  │                           │
│                  │                    │                           │
│                  │  • Aggregation     │                           │
│                  │  • Thresholding    │                           │
│                  │  • Pressure Check  │                           │
│                  └─────────┬──────────┘                           │
│                            │                                      │
│                            ▼                                      │
│              ┌──────────────────────────┐                         │
│              │  Adaptive Response       │                         │
│              │                          │                         │
│              │  • Reduce Complexity     │                         │
│              │  • Log Warning           │                         │
│              │  • Trigger Checkpoint    │                         │
│              └──────────────────────────┘                         │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 3. Data Flow

```
┌────────────────────────────────────────────────────────────────────┐
│                          Data Pipeline                              │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input Sources                Processing               Output        │
│  ──────────────               ──────────               ──────        │
│                                                                      │
│  ┌─────────┐                                                        │
│  │ VGLC    │──┐                                                     │
│  │ Levels  │  │             ┌──────────────┐                        │
│  └─────────┘  ├────────────►│ Level Parser │                        │
│                │             └──────┬───────┘                        │
│  ┌─────────┐  │                    │                                │
│  │ Custom  │──┘                    ▼                                │
│  │ Levels  │              ┌─────────────────┐                       │
│  └─────────┘              │ Preprocessing   │                       │
│                           └────────┬────────┘                       │
│                                    │                                │
│  ┌─────────┐                      │                                │
│  │ Env     │◄─────────────────────┘                                │
│  │ State   │                                                        │
│  └────┬────┘                                                        │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────┐        ┌──────────────┐                           │
│  │ RL Agent    │───────►│ Action       │                           │
│  └─────────────┘        └──────┬───────┘                           │
│       │                        │                                    │
│       │                        ▼                                    │
│       │                  ┌──────────┐       ┌─────────────┐        │
│       │                  │ Env Step │──────►│ New State   │        │
│       │                  └──────────┘       │ + Reward    │        │
│       │                                     └──────┬──────┘        │
│       └─────────────────────────────────────────────┘              │
│                                                      │              │
│                                                      ▼              │
│                                            ┌──────────────────┐    │
│                                            │ Logging System   │    │
│                                            │                  │    │
│                                            │ • CSV Files      │    │
│                                            │ • Checkpoints    │    │
│                                            │ • Generated      │    │
│                                            │   Levels         │    │
│                                            └──────────────────┘    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4. File System Organization

```
RAPCG-MetaRL/
│
├── Core Scripts
│   ├── train.py              [Training orchestration]
│   ├── inference.py          [Level generation]
│   └── utils.py              [Utilities & monitoring]
│
├── Wrappers/                 [Environment adaptations]
│   ├── pcgrl_env.py         [Resource-aware wrapper]
│   └── helper.py            [VGLC parsing & metrics]
│
├── Test/                     [Validation suite]
│   └── test.py              [Comprehensive tests]
│
├── Data/                     [Game level data]
│   ├── SMB.json             [Super Mario Bros]
│   └── zelda.json           [Zelda dungeons]
│
├── Runtime Directories       [Created during execution]
│   ├── logs/                [Training logs (CSV)]
│   ├── checkpoints/         [Model checkpoints (.zip)]
│   └── generated_levels/    [Generated content]
│
└── External Repos            [Submodules]
    ├── gym-pcgrl/           [PCGRL environments]
    ├── pcg_benchmark/       [PCG benchmark]
    └── TheVGLC/             [Level corpus]
```

## Class Hierarchy

### Resource Monitoring

```
ResourceMonitor
├── __init__(use_gpu: bool)
├── get_resources() → Dict[str, float]
│   ├── CPU percentage
│   ├── RAM percentage & GB
│   └── GPU util & memory (if available)
├── check_resource_pressure(thresholds) → (bool, str)
└── __del__()  # Cleanup
```

### Training Logger

```
TrainingLogger
├── __init__(log_dir, experiment_name)
├── log_step(reward, resources, content_metrics)
├── log_episode_end()
├── save() → CSV file
├── get_stats() → Dict[str, float]
└── print_stats()
```

### Environment Wrapper

```
ResourceAwarePCGRLWrapper(gym.Wrapper)
├── __init__(env, max_complexity, min_complexity)
├── adapt_complexity(cpu, ram, gpu) → int
├── reset(**kwargs) → observation
└── step(action) → (obs, reward, done, info)
```

### Meta-RL Trainer

```
MetaRLTrainer
├── __init__(game, algorithm, hyperparams...)
├── make_env(rank) → callable
├── setup_environments()
├── setup_model()
├── train()
├── evaluate(n_episodes)
└── load_model(path)
```

### Level Generator

```
LevelGenerator
├── __init__(model_path, game, representation)
├── generate(n_levels, ...) → List[levels]
├── _extract_level(info) → ndarray
├── _visualize_level(level, title)
└── close()
```

## Data Flow Diagram

```
User Input
    │
    ├─► Configuration (args)
    │       │
    │       ▼
    │   MetaRLTrainer
    │       │
    │       ├─► Environment Setup
    │       │       │
    │       │       ├─► gym-pcgrl
    │       │       └─► ResourceAwareWrapper
    │       │
    │       ├─► Model Setup
    │       │       │
    │       │       └─► stable-baselines3 (PPO/A2C)
    │       │
    │       └─► Training Loop
    │               │
    │               ├─► ResourceMonitor ──► Metrics
    │               ├─► TrainingLogger ──► CSV Logs
    │               ├─► Callback ──► Checkpoints
    │               │
    │               └─► Final Model
    │
    └─► Inference
            │
            ├─► Load Model
            ├─► Generate Levels
            └─► Save & Visualize
```

## Integration Points

### 1. gym-pcgrl Integration

```python
# RAPCG-MetaRL → gym-pcgrl
from wrappers.pcgrl_env import make_pcgrl_env

env = make_pcgrl_env('zelda', 'narrow')
# Returns: ResourceAwareWrapper(gym-pcgrl environment)
```

### 2. stable-baselines3 Integration

```python
# RAPCG-MetaRL → stable-baselines3
from stable_baselines3 import PPO

model = PPO('MlpPolicy', env, **hyperparams)
model.learn(total_timesteps, callback=ResourceAwareCallback())
```

### 3. VGLC Integration

```python
# RAPCG-MetaRL → TheVGLC
from wrappers.helper import load_vglc_levels

levels = load_vglc_levels('data', 'SMB')
# Returns: List[ndarray] of parsed levels
```

## Callback System

```
Training Step
     │
     ├─► ResourceAwareCallback._on_step()
     │       │
     │       ├─► Get resources
     │       ├─► Log step
     │       ├─► Check episode end
     │       ├─► Save checkpoint (periodic)
     │       └─► Check resource pressure
     │
     └─► Continue training or adapt
```

## Extension Points

### Add New Game

1. Implement in gym-pcgrl
2. Update `make_pcgrl_env()` in `wrappers/pcgrl_env.py`
3. Add test in `test/test.py`

### Add New Metric

1. Implement in `wrappers/helper.py`
2. Call in `calculate_content_metrics()`
3. Add to logger in training loop

### Add New Algorithm

1. Import from stable-baselines3
2. Add to `MetaRLTrainer.setup_model()`
3. Add command-line option
4. Update docs

### Add Meta-RL Algorithm

1. Implement task distribution
2. Create meta-learning loop
3. Add inner/outer optimization
4. Integrate with existing training

## Performance Considerations

### Memory Management

```
Factor                  Impact              Mitigation
──────────────────────  ──────────────────  ────────────────────
Batch Size              High                Reduce to 32-64
Parallel Envs           Medium              Use 1-4 based on RAM
Replay Buffer           Medium              Use on-policy (PPO)
Model Checkpoints       Low                 Clean old checkpoints
Logging Frequency       Low                 Batch log writes
```

### Computational Efficiency

```
Operation               Cost                Optimization
──────────────────────  ──────────────────  ────────────────────
Environment Step        Medium              Vectorized envs
Policy Forward Pass     Medium              GPU acceleration
Resource Monitoring     Low                 Sample rate control
Logging                 Low                 Async writes
Checkpoint Saving       Low                 Periodic only
```

## Configuration Flow

```
Command Line Args
       │
       ▼
argparse.ArgumentParser
       │
       ├─► game
       ├─► algorithm
       ├─► hyperparameters
       ├─► device
       └─► experiment_name
           │
           ▼
       MetaRLTrainer.__init__()
           │
           ├─► Create experiment dir
           ├─► Initialize ResourceMonitor
           ├─► Initialize TrainingLogger
           └─► Setup environments & model
               │
               ▼
           Training Execution
```

## Output Artifacts

```
Project Root/
├── logs/
│   └── experiment_name.csv
│       ├── episode, step, reward
│       ├── cpu_percent, ram_percent
│       ├── gpu_util_percent, gpu_mem_percent
│       └── content_diversity, content_complexity
│
├── checkpoints/
│   └── experiment_name/
│       ├── model_step_1000.zip
│       ├── model_step_2000.zip
│       └── final_model.zip
│
└── generated_levels/
    └── game_name/
        ├── level_1.npy
        ├── level_1.txt
        ├── level_2.npy
        └── level_2.txt
```

---

This architecture supports:

- ✅ Modular design
- ✅ Easy extension
- ✅ Resource awareness
- ✅ Comprehensive logging
- ✅ Meta-RL foundation
