# RAPCG-MetaRL

**Resource-Aware Procedural Content Generation via Meta-Reinforcement Learning**

> Published research framework for training Meta-RL agents to generate procedural game content with dynamic hardware adaptation. Evaluated on PCGRL benchmarks (Zelda, Sokoban) using PPO and A2C algorithms on consumer hardware (Intel i5-13500, RTX 3060 Ti, 16GB RAM).

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-orange.svg)](https://pytorch.org/)
[![stable-baselines3](https://img.shields.io/badge/RL-stable--baselines3-green.svg)](https://github.com/DLR-RM/stable-baselines3)

---

## 🎯 Overview

RAPCG-MetaRL integrates real-time hardware telemetry into a reinforcement learning reward signal, creating a feedback loop that teaches PCG agents to balance content quality with computational efficiency. The framework targets heterogeneous gaming platforms — from budget laptops to high-end workstations — without requiring separate builds.

### Key Results (Published in ACM TOG 2025)

| Algorithm | Domain  | Early Reward | Late Reward | Improvement |
| --------- | ------- | ------------ | ----------- | ----------- |
| PPO       | Zelda   | −8.54        | +11.84      | +20.38 pts  |
| PPO       | Sokoban | +2.84        | +6.66       | +3.82 pts   |
| A2C       | Zelda   | −14.6        | −3.0        | +11.6 pts   |
| A2C       | Sokoban | −1.9         | +3.2        | +5.1 pts    |

Both algorithms maintained CPU usage under 5% and RAM under 65%, with resource-aware penalties remaining at zero — demonstrating operation well within configured hardware thresholds.

---

## 🎯 Features

- **Resource-Aware Reward Shaping**: Real-time CPU/RAM/GPU penalties shape the reward signal, teaching agents to generate content efficiently
- **Multi-Algorithm Support**: PPO and A2C via stable-baselines3; SAC for continuous action spaces
- **Dynamic Complexity Adaptation**: Environment complexity scales with resource pressure
- **Hardware Telemetry**: Asynchronous CPU, RAM, and GPU monitoring via `psutil`/`pynvml`
- **PCGRL Benchmarks**: Zelda, Sokoban, and Binary environments via gym-pcgrl
- **Solvability Optimization**: Tuned reward weights for high solvability rates
- **Comprehensive Logging**: Per-step CSV logs with reward, resource, and content metrics
- **VGLC Integration**: Parse and analyze levels from The Video Game Level Corpus
- **Paper Figure Generation**: Publication-ready figures for academic reporting

---

## 📁 Project Structure

```
RAPCG-MetaRL/
│
├── Core Scripts
│   ├── train.py                    # Main training script (PPO/A2C/SAC)
│   ├── inference.py                # Level generation from trained models
│   ├── inference_timed.py          # Timed inference with detailed metrics
│   ├── utils.py                    # ResourceMonitor, TrainingLogger, utilities
│   ├── maml_trainer.py             # MAML meta-learning trainer
│   ├── rlhf_trainer.py             # RLHF fine-tuning pipeline
│   ├── dashboard/                  # Interactive Streamlit dashboard UI
│   └── quickstart.py               # Guided demo
│
├── Wrappers/
│   ├── pcgrl_env.py                # ResourceAwarePCGRLWrapper + make_pcgrl_env()
│   └── helper.py                   # VGLC parsing, content metrics, level I/O
│
├── Configuration
│   ├── config_hardware.py          # Hardware-specific presets (i5-13500 + RTX 3060 Ti)
│   ├── solvability_config.py       # Per-game reward weight tuning
│   └── sokoban_utils.py            # Sokoban solvability validation
│
├── Test/
│   └── test/test.py                # Full test suite (5 test categories)
│
├── Analysis
│   ├── generate_paper_figures.py   # ACM TOG paper figure generator
│   ├── compare_approaches.py       # Forward vs. backward generation comparison
│   ├── analyze_action_penalties.py # Action-penalty correlation analysis
│   ├── architecture_diagram.py     # ASCII architecture diagram
│   └── graph.ipynb / inference_graph.ipynb  # Notebooks for result visualization
│
├── Data/
│   ├── data/SMB.json               # Super Mario Bros levels (VGLC)
│   └── data/zelda.json             # Zelda levels (VGLC)
│
├── Results (created during training/inference)
│   ├── logs/                       # Training CSVs (per-step metrics)
│   ├── checkpoints/                # Model checkpoints (.zip)
│   └── generated_levels/           # Generated level files (.npy, .txt)
│
├── Documentation
│   ├── README.md                   # This file
│   ├── ARCHITECTURE.md             # System architecture reference
│   ├── IMPLEMENTATION_SUMMARY.md   # Feature completion checklist
│   ├── RESOURCE_AWARE_IMPLEMENTATION.md  # Reward shaping details
│   ├── SOLVABILITY_INTEGRATION.md  # Solvability mechanism docs
│   ├── HARDWARE_COMPATIBILITY.md   # Hardware tuning guide
│   └── VERIFICATION_CHECKLIST.md   # Setup verification steps
│
├── External Repos (submodules)
│   ├── gym-pcgrl/                  # PCGRL environments (Khalifa et al.)
│   ├── pcg_benchmark/              # PCG benchmark suite
│   └── TheVGLC/                    # Video Game Level Corpus
│
├── setup.ps1                       # Windows PowerShell setup script
├── quickstart_optimized.ps1        # Optimized quickstart for target hardware
├── Dockerfile                      # Container support
└── requirements.txt                # Python dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites

```bash
# Python 3.8+
python --version

# Install dependencies
pip install -r requirements.txt

# Install gym-pcgrl
cd gym-pcgrl
pip install -e .
cd ..
```

> [Spacer]
> [!IMPORTANT]
> **Virtual Environment Python Interpreter (Windows)**  
> Always run scripts using the project's virtual environment Python interpreter to ensure dependencies are loaded correctly:
> ```powershell
> .\pcg_env\Scripts\python.exe <script> <args>
> ```

### 2. Run Tests

```powershell
.\pcg_env\Scripts\python.exe test/test.py
```

Expected: 5/5 tests pass (Resource Monitor, Training Logger, VGLC Parsing, Content Metrics, PCGRL Environment).

### 3. Train a Model

```powershell
# Quick test (10k steps, CPU)
.\pcg_env\Scripts\python.exe train.py --game zelda --timesteps 10000

# GPU-accelerated training (matches paper results)
.\pcg_env\Scripts\python.exe train.py --game zelda --timesteps 20000 --device cuda

# Sokoban PPO
.\pcg_env\Scripts\python.exe train.py --game sokoban --timesteps 20000 --device cuda

# A2C comparison
.\pcg_env\Scripts\python.exe train.py --game zelda --algorithm A2C --timesteps 10000 --device cuda
```

### 4. Generate Levels

```powershell
# Standard inference
.\pcg_env\Scripts\python.exe inference.py checkpoints/zelda_PPO_<timestamp>/final_model.zip --n-levels 10

# Timed inference (with detailed metrics CSV)
.\pcg_env\Scripts\python.exe inference_timed.py checkpoints/zelda_PPO_<timestamp>/final_model.zip --game zelda --n-levels 20 --log-file inference_timing.csv --device cuda
```

### 5. Windows PowerShell (Optimized)

```powershell
# Activate venv and run hardware-optimized config
.\quickstart_optimized.ps1
```

### 6. Start the Interactive Dashboard

You can use the Streamlit-based dashboard to configure and trigger training, monitor runs with real-time logs, generate and visualize levels, compare metrics from different runs, and annotate pairs of levels for RLHF:

```powershell
# Start the dashboard using the venv streamlit command
.\pcg_env\Scripts\streamlit run dashboard/dashboard.py
```

---

## 🔧 Training Parameters

| Parameter                 | Description                  | Default  |
| ------------------------- | ---------------------------- | -------- |
| `--game`                  | Game environment             | `zelda`  |
| `--representation`        | Representation type          | `narrow` |
| `--algorithm`             | RL algorithm (PPO/A2C/SAC)   | `PPO`    |
| `--timesteps`             | Total training steps         | `50000`  |
| `--n-steps`               | Steps per update             | `128`    |
| `--batch-size`            | Batch size                   | `64`     |
| `--lr`                    | Learning rate                | `2.5e-4` |
| `--n-envs`                | Parallel environments        | `1`      |
| `--device`                | Device (`cpu`/`cuda`/`auto`) | `auto`   |
| `--checkpoint-freq`       | Checkpoint save frequency    | `1000`   |
| `--no-solvability-tuning` | Disable solvability weights  | off      |

### Hardware Presets (from `config_hardware.py`)

```bash
# Check your hardware compatibility first
python config_hardware.py
```

| Preset         | n_envs | timesteps | batch_size | Use Case           |
| -------------- | ------ | --------- | ---------- | ------------------ |
| `PRESET_FAST`  | 6      | 50000     | 128        | Full training run  |
| `PRESET_LIGHT` | 4      | 50000     | 64         | Running other apps |

---

## 📊 Resource Monitoring

The framework automatically monitors hardware resources at every training step:

```python
from utils import ResourceMonitor

monitor = ResourceMonitor(use_gpu=True)
resources = monitor.get_resources()

print(f"CPU:      {resources['cpu_percent']:.1f}%")
print(f"RAM:      {resources['ram_percent']:.1f}%")
print(f"GPU Mem:  {resources['gpu_mem_percent']:.1f}%")
print(f"GPU Util: {resources['gpu_util_percent']:.1f}%")
```

### Resource-Aware Reward Shaping

Penalties are subtracted from the reward when usage exceeds thresholds:

```
R_total = R_quality
        − 0.2 × max(0, RAM%  − 78%)
        − 0.1 × max(0, CPU%  − 70%)
        − 0.1 × max(0, GPU%  − 70%)
```

In the published training runs, all penalties remained at zero — indicating the system operated well within hardware limits throughout training.

---

## 📈 Training Logs

All runs generate detailed CSV logs in `logs/`:

```
logs/zelda_PPO_YYYYMMDD_HHMMSS.csv
  ├── episode, step, reward, shaped_reward
  ├── ram_penalty, cpu_penalty, gpu_penalty
  ├── cpu_percent, ram_percent
  ├── gpu_util_percent, gpu_mem_percent
  └── content_diversity, content_complexity
```

Analyze with pandas:

```python
import pandas as pd
df = pd.read_csv('logs/zelda_PPO_20260220_140931.csv')
print(df[['episode','reward','ram_percent']].tail(20))
```

---

## 🎮 Supported Environments

### Games

| Game    | Grid   | Goal                                    |
| ------- | ------ | --------------------------------------- |
| Zelda   | 16×16  | Dungeon with valid key→door→player path |
| Sokoban | 10×10  | Solvable box-pushing puzzles (NP-hard)  |
| Binary  | Varies | Connected binary pattern generation     |

### Representations

- **Narrow**: Agent edits one tile at a time (default)
- **Wide**: Agent selects position and tile type simultaneously
- **Turtle**: Agent moves through the map and places tiles

### Algorithms

- ✅ **PPO** — Best overall performance (recommended)
- ✅ **A2C** — Lower sample efficiency, more predictable learning curves
- ✅ **SAC** — Continuous action spaces only

---

## 🧪 Development

### Test Suite

```bash
# Run all tests
python test/test.py

# Test individual components
python -c "from test.test import test_environment; test_environment()"
python -c "from utils import ResourceMonitor; m = ResourceMonitor(); print(m.get_resources())"
```

### Generate Paper Figures

```bash
# Procedural demo (no trained model required)
python generate_paper_figures.py --demo

# From trained models
python generate_paper_figures.py --model \
  --zelda-model checkpoints/zelda_PPO_.../final_model.zip \
  --sokoban-model checkpoints/sokoban_PPO_.../final_model.zip
```

### Compare Approaches

```bash
python compare_approaches.py \
  --forward-dir generated_levels/sokoban_forward \
  --backward-dir generated_levels/sokoban_backward
```

### VGLC Integration

```python
from wrappers.helper import load_vglc_levels, calculate_content_metrics

levels = load_vglc_levels('data', 'SMB')
metrics = calculate_content_metrics(levels[0])
print(f"Diversity:  {metrics['diversity']:.3f}")
print(f"Complexity: {metrics['complexity']:.3f}")
```

---

## 🐛 Troubleshooting

| Issue                 | Solution                                                         |
| --------------------- | ---------------------------------------------------------------- |
| GPU not detected      | `pip install nvidia-ml-py3` or use `--device cpu`                |
| `gym-pcgrl` not found | `cd gym-pcgrl && pip install -e .`                               |
| Import errors         | `$env:PYTHONPATH += ";D:\Work\thesis\RAPCG-MetaRL"` (PowerShell) |
| Out of memory         | `--batch-size 32 --n-envs 1`                                     |
| Training too slow     | Enable CUDA: `--device cuda`                                     |

---

## 🏗️ Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for full system diagrams.

Core component hierarchy:

```
train.py / inference.py
    └── MetaRLTrainer / LevelGenerator
            └── make_pcgrl_env()
                    └── ResourceAwarePCGRLWrapper (gym.Wrapper)
                            ├── gym-pcgrl (Zelda-v0, Sokoban-v0, ...)
                            └── ResourceMonitor (psutil / pynvml)
```

### Implementation Status

| Component                          | Status         |
| ---------------------------------- | -------------- |
| PPO/A2C Training Pipeline          | ✅ Implemented |
| Resource-Aware Reward Shaping      | ✅ Implemented |
| Hardware Telemetry (psutil/pynvml) | ✅ Implemented |
| Solvability Optimization           | ✅ Implemented |
| MAML Meta-RL Controller            | ✅ Implemented |
| Adaptive Batch Scheduling          | 🔄 Proposed    |
| Hybrid PCG Ensemble                | 🔄 Proposed    |
| Unity/Unreal Integration           | 🔄 Proposed    |

---

## 📦 Dependencies

**Core (Required)**

- Python 3.8+
- PyTorch 2.1+
- stable-baselines3
- gym
- numpy, pandas, psutil, pillow

**Optional**

- `nvidia-ml-py3` — GPU monitoring
- `jupyter` — Notebooks
- `matplotlib` — Figure generation

See [requirements.txt](requirements.txt) for full list.

---

## 📖 Citation

If you use this framework, please cite:

```bibtex
@article{rahman2025rapcg,
  title={Resource-Aware Procedural Content Generation via Meta-Reinforcement
         Learning for Heterogeneous Gaming Platforms},
  author={Rahman, Redwan and Kabir, Md. Alamgir},
  journal={ACM Transactions on Graphics},
  year={2025},
  publisher={ACM}
}
```

Please also cite the foundational work this project builds upon:

```bibtex
@inproceedings{khalifa2020pcgrl,
  title={PCGRL: Procedural Content Generation via Reinforcement Learning},
  author={Khalifa, Ahmed and Bontrager, Philip and Earle, Sam and Togelius, Julian},
  booktitle={Artificial Intelligence and Interactive Digital Entertainment},
  volume={16}, number={1}, pages={95--101},
  year={2020}, organization={AAAI}
}
```

---

## 🙏 Acknowledgments

- **[gym-pcgrl](https://github.com/amidos2006/gym-pcgrl)** — Ahmed Khalifa et al.'s foundational PCGRL framework
- **[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)** — High-quality RL implementations
- **[The Video Game Level Corpus (VGLC)](https://github.com/TheVGLC/TheVGLC)** — Level dataset
- **[PCG Benchmark](https://github.com/amidos2006/gym-pcgrl)** — PCG evaluation testbed

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

## 📧 Contact

Redwan Rahman — rahman22205101127@diu.edu.bd  
Department of Computer Science and Engineering, Daffodil International University

Code: <https://github.com/Red1-Rahman/RAPCG-MetaRL>

---

_RAPCG-MetaRL — Resource-Aware PCG that adapts to your hardware. 🎮_
