# RAPCG-MetaRL

**Resource-Aware Procedural Content Generation with Meta-Reinforcement Learning**

A framework for training Meta-RL agents to generate procedural game content with dynamic resource adaptation. This project combines gym-pcgrl environments with resource-aware training to optimize content generation within hardware constraints.

## 🎯 Features

- **Meta-RL Training**: PPO and A2C algorithms with stable-baselines3
- **Resource Monitoring**: Real-time CPU, GPU, and RAM tracking
- **Dynamic Adaptation**: Automatic complexity adjustment based on resource usage
- **Multi-Environment**: Support for Zelda, Sokoban, Binary, and custom levels
- **Content Metrics**: Diversity, complexity, and quality evaluation
- **VGLC Integration**: Parse and use levels from The Video Game Level Corpus
- **Comprehensive Logging**: CSV-based training logs with resource tracking

## 📁 Project Structure

```
RAPCG-MetaRL/
├── train.py                 # Main training script
├── inference.py             # Level generation script
├── utils.py                 # Resource monitoring and logging utilities
├── wrappers/                # Environment wrappers
│   ├── __init__.py
│   ├── pcgrl_env.py        # PCGRL environment wrapper
│   └── helper.py           # VGLC parsing and metrics
├── test/                    # Test suite
│   ├── __init__.py
│   └── test.py             # Comprehensive tests
├── data/                    # Game level data (VGLC)
│   ├── SMB.json
│   └── zelda.json
├── gym-pcgrl/              # gym-pcgrl submodule
├── TheVGLC/                # Video Game Level Corpus
├── logs/                   # Training logs (created during training)
├── checkpoints/            # Model checkpoints (created during training)
└── generated_levels/       # Generated levels (created during inference)
```

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

### 2. Run Tests

Verify your setup:

```bash
python test/test.py
```

This will test:

- Resource monitoring
- Training logger
- VGLC level parsing
- Content metrics
- Environment creation

### 3. Train a Model

Simple training with default settings (Zelda, PPO):

```bash
python train.py --game zelda --timesteps 50000
```

Advanced training with custom parameters:

```bash
python train.py \
  --game zelda \
  --representation narrow \
  --algorithm PPO \
  --timesteps 100000 \
  --n-steps 128 \
  --batch-size 64 \
  --lr 2.5e-4 \
  --device cuda \
  --experiment-name zelda_experiment_1
```

Training with multiple environments (parallel):

```bash
python train.py \
  --game sokoban \
  --n-envs 4 \
  --timesteps 200000
```

### 4. Generate Levels

Generate levels using a trained model:

```bash
python inference.py checkpoints/zelda_PPO_*/final_model.zip \
  --n-levels 10 \
  --save-dir generated_levels/zelda
```

With custom settings:

```bash
python inference.py path/to/model.zip \
  --game zelda \
  --representation narrow \
  --n-levels 5 \
  --max-steps 1000 \
  --stochastic
```

## 📊 Resource Monitoring

The framework automatically monitors system resources during training:

### CPU, RAM, GPU Usage

```python
from utils import ResourceMonitor

monitor = ResourceMonitor(use_gpu=True)
resources = monitor.get_resources()

print(f"CPU: {resources['cpu_percent']}%")
print(f"RAM: {resources['ram_percent']}%")
print(f"GPU: {resources['gpu_mem_percent']}%")
```

### Resource-Aware Training

The training script automatically:

- Monitors resources every step
- Logs resource usage to CSV
- Adapts environment complexity when under pressure
- Saves checkpoints with resource snapshots

## 📈 Training Logs

All training runs generate detailed CSV logs:

```
logs/
└── zelda_PPO_20231208_143022.csv
    ├── episode
    ├── step
    ├── reward
    ├── timestamp
    ├── cpu_percent
    ├── ram_percent
    ├── gpu_util_percent
    ├── gpu_mem_percent
    ├── content_diversity
    └── content_complexity
```

Analyze logs with pandas:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load logs
df = pd.read_csv('logs/zelda_PPO_20231208_143022.csv')

# Plot reward over time
plt.plot(df['step'], df['reward'])
plt.xlabel('Step')
plt.ylabel('Reward')
plt.show()

# Analyze resource usage
print(f"Mean GPU usage: {df['gpu_mem_percent'].mean():.1f}%")
```

## 🎮 Supported Environments

### Games

- **Zelda**: Top-down dungeon generation
- **Sokoban**: Puzzle box-pushing levels
- **Binary**: Simple binary pattern generation

### Representations

- **Narrow**: Agent edits one tile at a time
- **Wide**: Agent selects position and tile type
- **Turtle**: Agent moves and places tiles

## 🔧 Configuration

### Training Parameters

| Parameter           | Description            | Default  |
| ------------------- | ---------------------- | -------- |
| `--game`            | Game environment       | `zelda`  |
| `--representation`  | Representation type    | `narrow` |
| `--algorithm`       | RL algorithm (PPO/A2C) | `PPO`    |
| `--timesteps`       | Total training steps   | `50000`  |
| `--n-steps`         | Steps per update       | `128`    |
| `--batch-size`      | Batch size             | `64`     |
| `--lr`              | Learning rate          | `2.5e-4` |
| `--n-envs`          | Parallel environments  | `1`      |
| `--device`          | Device (cpu/cuda/auto) | `auto`   |
| `--checkpoint-freq` | Checkpoint frequency   | `1000`   |

### Resource Thresholds

Modify in `utils.py`:

```python
thresholds = {
    'cpu_percent': 90.0,
    'ram_percent': 90.0,
    'gpu_mem_percent': 85.0,
    'gpu_util_percent': 85.0,
}
```

## 🧪 Development

### Running Tests

```bash
# Run all tests
python test/test.py

# Test specific component
python -c "from test.test import test_environment; test_environment()"
```

### Adding Custom Environments

1. Add environment wrapper in `wrappers/pcgrl_env.py`
2. Update `make_pcgrl_env()` function
3. Add tests in `test/test.py`

### Custom Metrics

Add custom content metrics in `wrappers/helper.py`:

```python
def custom_metric(level: np.ndarray) -> float:
    """Calculate custom metric."""
    # Your implementation
    return score
```

## 📚 VGLC Integration

Use levels from The Video Game Level Corpus:

```python
from wrappers.helper import load_vglc_levels, calculate_content_metrics

# Load levels
levels = load_vglc_levels('data', 'SMB')

# Analyze a level
metrics = calculate_content_metrics(levels[0])
print(f"Diversity: {metrics['diversity']:.3f}")
print(f"Complexity: {metrics['complexity']:.3f}")
```

## 🎯 Meta-RL (Future Work)

The current implementation provides a solid foundation for Meta-RL. Future enhancements:

### MAML (Model-Agnostic Meta-Learning)

```python
# Train on multiple tasks
tasks = ['zelda', 'sokoban', 'binary']

for task in tasks:
    env = make_pcgrl_env(task)
    # Inner loop: adapt to task
    # Outer loop: update meta-parameters
```

### PEARL (Probabilistic Embeddings for Actor-Critic RL)

```python
# Context encoder for task inference
# Train on diverse level distributions
# Fast adaptation with few samples
```

## 📦 Dependencies

Core requirements:

- Python 3.8+
- PyTorch 1.10+
- stable-baselines3
- gym
- numpy
- pandas
- psutil
- pynvml (for GPU monitoring)

See `requirements.txt` for full list.

## 🐛 Troubleshooting

### GPU Monitoring Not Working

If GPU monitoring fails:

```bash
pip install nvidia-ml-py3
```

Or disable GPU monitoring:

```bash
python train.py --no-gpu-monitoring
```

### Environment Creation Fails

Make sure gym-pcgrl is installed:

```bash
cd gym-pcgrl
pip install -e .
```

### Import Errors

Add project to Python path:

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/RAPCG-MetaRL"
```

Or on Windows PowerShell:

```powershell
$env:PYTHONPATH += ";D:\Work\thesis\RAPCG-MetaRL"
```

### Out of Memory

Reduce batch size and parallel environments:

```bash
python train.py --batch-size 32 --n-envs 1
```

## 📖 Citation

If you use this framework, please cite:

```bibtex
@software{rapcg_metarl,
  title={RAPCG-MetaRL: Resource-Aware Procedural Content Generation with Meta-RL},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/RAPCG-MetaRL}
}
```

Please also cite the foundational work this project builds upon:

```bibtex
@inproceedings{khalifa2020pcgrl,
  title={PCGRL: Procedural Content Generation via Reinforcement Learning},
  author={Khalifa, Ahmed and Bontrager, Philip and Earle, Sam and Togelius, Julian},
  booktitle={Artificial Intelligence and Interactive Digital Entertainment},
  volume={16},
  number={1},
  pages={95--101},
  year={2020},
  organization={AAAI}
}

@inproceedings{khalifa2025pcgbenchmark,
  title={The Procedural Content Generation Benchmark: An Open-source Testbed for Generative Challenges in Games},
  author={Khalifa, Ahmed and Gallota, Roberto and Barthet, Matthew and Liapis, Antonios and Togelius, Julian and Yannakakis, Georgios N.},
  booktitle={Foundations of Digital Games Conference},
  year={2025},
  publisher={ACM}
}

@misc{summerville2016vglc,
  title={The Video Game Level Corpus},
  author={Summerville, Adam and Snodgrass, Sam and Mateas, Michael and Ontañón, Santiago},
  year={2016},
  eprint={1606.07487},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://doi.org/10.48550/arXiv.1606.07487}
}
```

## 🙏 Acknowledgments

This project builds upon several excellent research works and open-source projects:

- **[gym-pcgrl](https://github.com/amidos2006/gym-pcgrl)** - Ahmed Khalifa et al.'s foundational PCGRL framework
- **[PCG Benchmark](https://github.com/amidos2006/gym-pcgrl)** - Comprehensive testbed for PCG challenges
- **[The Video Game Level Corpus (VGLC)](https://github.com/TheVGLC/TheVGLC)** - Large-scale level dataset
- **[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)** - High-quality RL algorithm implementations
- **[OpenAI Gym](https://github.com/openai/gym)** - Standard RL environment interface

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Happy Level Generating! 🎮✨**
