# ✅ RAPCG-MetaRL Setup Verification Checklist

Use this checklist to verify your RAPCG-MetaRL installation is complete and ready to use.

## 📋 Pre-Installation Checklist

- [ ] Python 3.8+ installed
- [ ] pip package manager available
- [ ] Git installed (for cloning repositories)
- [ ] CUDA installed (optional, for GPU training)
- [ ] 16GB+ RAM available (recommended)

## 📦 Installation Verification

### Core Files Present

- [ ] `train.py` - Main training script
- [ ] `inference.py` - Level generation script
- [ ] `utils.py` - Utilities and monitoring
- [ ] `requirements.txt` - Python dependencies
- [ ] `README.md` - Documentation
- [ ] `Dockerfile` - Container support
- [ ] `setup.ps1` - Windows setup script
- [ ] `quickstart.py` - Quick start demo

### Directory Structure

- [ ] `wrappers/` - Environment wrappers
  - [ ] `wrappers/__init__.py`
  - [ ] `wrappers/pcgrl_env.py`
  - [ ] `wrappers/helper.py`
- [ ] `test/` - Test suite
  - [ ] `test/__init__.py`
  - [ ] `test/test.py`
- [ ] `data/` - Game data
  - [ ] `data/SMB.json` (optional)
  - [ ] `data/zelda.json` (optional)
- [ ] `gym-pcgrl/` - PCGRL environments (submodule)
- [ ] `TheVGLC/` - Level corpus (optional)

### Dependencies Installed

Run these commands to verify:

```powershell
# Check Python version
python --version  # Should be 3.8+

# Check pip
pip --version

# Verify key packages
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import stable_baselines3; print('SB3:', stable_baselines3.__version__)"
python -c "import gym; print('Gym:', gym.__version__)"
python -c "import numpy; print('NumPy:', numpy.__version__)"
python -c "import pandas; print('Pandas:', pandas.__version__)"
python -c "import psutil; print('psutil:', psutil.__version__)"
```

- [ ] PyTorch installed
- [ ] stable-baselines3 installed
- [ ] gym installed
- [ ] numpy installed
- [ ] pandas installed
- [ ] psutil installed

### Optional Dependencies

```powershell
# GPU monitoring
python -c "import pynvml; print('pynvml: OK')"

# Visualization
python -c "import matplotlib; print('matplotlib: OK')"

# Jupyter (for notebooks)
python -c "import jupyter; print('jupyter: OK')"
```

- [ ] pynvml (GPU monitoring)
- [ ] matplotlib (visualization)
- [ ] jupyter (notebooks)

### gym-pcgrl Installation

```powershell
cd gym-pcgrl
pip install -e .
cd ..

# Verify installation
python -c "import gym_pcgrl; print('gym-pcgrl: OK')"
```

- [ ] gym-pcgrl installed
- [ ] gym-pcgrl imports successfully

## 🧪 Functionality Testing

### Run Test Suite

```powershell
python test/test.py
```

Expected results:

- [ ] ✓ PASS: Resource Monitor
- [ ] ✓ PASS: Training Logger
- [ ] ✓ PASS: VGLC Parsing
- [ ] ✓ PASS: Content Metrics
- [ ] ✓ PASS: PCGRL Environment

### Quick Functionality Checks

```powershell
# Test resource monitoring
python -c "from utils import ResourceMonitor; m = ResourceMonitor(); print(m.get_resources())"

# Test environment creation
python -c "from wrappers.pcgrl_env import make_pcgrl_env; env = make_pcgrl_env('zelda'); print('OK')"

# Test training script help
python train.py --help

# Test inference script help
python inference.py --help
```

- [ ] Resource monitoring works
- [ ] Environment creation works
- [ ] Training script accessible
- [ ] Inference script accessible

## 🚀 Ready to Train

### Quick Test Training (5 minutes)

```powershell
python train.py --game zelda --timesteps 5000 --experiment-name test_run
```

Check that:

- [ ] Training starts without errors
- [ ] Resource monitoring displays
- [ ] Progress updates appear
- [ ] Log file created in `logs/`
- [ ] Checkpoint created in `checkpoints/`

### Quick Test Inference

```powershell
python inference.py checkpoints/test_run/final_model.zip --n-levels 1 --no-visualize
```

Check that:

- [ ] Model loads successfully
- [ ] Level generation completes
- [ ] Output saved to `generated_levels/`
- [ ] Metrics displayed

## 🎯 Advanced Features

### GPU Support

```powershell
# Check CUDA availability
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

# Test GPU training
python train.py --game zelda --timesteps 1000 --device cuda
```

- [ ] CUDA available (if GPU present)
- [ ] GPU training works
- [ ] GPU monitoring active

### Multi-Environment Training

```powershell
python train.py --game zelda --timesteps 5000 --n-envs 2
```

- [ ] Parallel environments work
- [ ] Training speed increases

### Custom Experiments

```powershell
python train.py \
  --game zelda \
  --algorithm PPO \
  --timesteps 10000 \
  --n-steps 128 \
  --batch-size 64 \
  --lr 2.5e-4 \
  --experiment-name my_experiment
```

- [ ] Custom parameters accepted
- [ ] Training runs successfully
- [ ] Logs saved with experiment name

## 📊 Data Analysis

### Check Logs

```powershell
# View log files
Get-ChildItem logs/*.csv

# Quick analysis (if pandas available)
python -c "import pandas as pd; df = pd.read_csv('logs/test_run.csv'); print(df.describe())"
```

- [ ] Log files created
- [ ] CSV format correct
- [ ] All metrics present (reward, CPU, RAM, GPU)

### Check Checkpoints

```powershell
# View checkpoints
Get-ChildItem checkpoints/* -Recurse -Filter *.zip
```

- [ ] Checkpoint directory created
- [ ] Model files (.zip) present
- [ ] final_model.zip exists

### Check Generated Levels

```powershell
# View generated levels
Get-ChildItem generated_levels/* -Recurse
```

- [ ] Levels directory created
- [ ] Level files present (.npy, .txt)
- [ ] Metrics computed

## 🔧 Troubleshooting

### Common Issues

#### Issue: Import errors

```powershell
# Add to Python path
$env:PYTHONPATH += ";D:\Work\thesis\RAPCG-MetaRL"
```

#### Issue: GPU not detected

```powershell
# Install CUDA toolkit
# Or use CPU: python train.py --device cpu
```

#### Issue: Out of memory

```powershell
# Reduce batch size
python train.py --batch-size 32 --n-envs 1
```

#### Issue: gym-pcgrl not found

```powershell
cd gym-pcgrl
pip install -e .
cd ..
```

## ✅ Final Verification

Run the complete quick start:

```powershell
python quickstart.py
```

This should:

- [ ] Run all tests
- [ ] Train a small model (10k steps)
- [ ] Generate sample levels
- [ ] Complete without errors

## 🎉 Ready to Go!

If all items are checked, your RAPCG-MetaRL installation is complete!

### Next Steps:

1. **Read the README.md** for detailed documentation
2. **Review IMPLEMENTATION_SUMMARY.md** for architecture overview
3. **Start training**: `python train.py --game zelda --timesteps 50000`
4. **Join the discussion**: Open issues or discussions on GitHub

### Quick Reference Commands

```powershell
# Full training run
python train.py --game zelda --timesteps 50000

# Monitor training in real-time
Get-Content logs/your_experiment.csv -Wait

# Generate levels
python inference.py checkpoints/your_experiment/final_model.zip --n-levels 10

# View training progress
python -c "import pandas as pd; df = pd.read_csv('logs/your_experiment.csv'); print(df.tail(20))"
```

## 📝 Notes

- First run may be slower due to compilation
- GPU training requires CUDA-compatible GPU
- Logs can get large - clean periodically
- Checkpoints are resumable - use `--load-model` flag

---

**Installation verified! Happy training! 🚀**

Date: ******\_\_\_******
Completed by: ******\_\_\_******
