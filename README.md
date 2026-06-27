# RAPCG-MetaRL

**Resource-Aware Procedural Content Generation via Meta-Reinforcement Learning**   
   
![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c)
![Docker](https://img.shields.io/badge/Container-Docker-2496ED)
![OCI](https://img.shields.io/badge/OCI-Compliant-2496ED)
![ISO/IEC/IEEE 12207](https://img.shields.io/badge/ISO%2FIEC%2FIEEE-12207-blue)
![SemVer](https://img.shields.io/badge/SemVer-2.0.0-orange)
![Git](https://img.shields.io/badge/Version_Control-Git-F05032)
![Reproducible Research](https://img.shields.io/badge/Reproducible-Research-success)
![Cross Platform](https://img.shields.io/badge/Cross--Platform-Windows%20%7C%20Linux-blueviolet)
![RL](https://img.shields.io/badge/Reinforcement-Learning-red)
![Meta Learning](https://img.shields.io/badge/Meta-Learning-purple)
![Washington Accord](https://img.shields.io/badge/Washington_Accord-Compliant-blue)
![Complex Engineering Problem](https://img.shields.io/badge/Engineering-Complex_Problem-success)
![Knowledge Profile](https://img.shields.io/badge/Knowledge_Profile-K2--K8-success)
![Engineering Activities](https://img.shields.io/badge/Engineering_Activities-EA1--EA5-success)


## 🎯 Overview

RAPCG-MetaRL integrates real-time hardware telemetry into a reinforcement learning reward signal, creating a feedback loop that teaches PCG agents to balance content quality with computational efficiency. The framework targets heterogeneous gaming platforms — from budget laptops to high-end workstations — without requiring separate builds.




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

- Python 3.10
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
@repo{RAPCG-MetaRL,
  title={Resource-Aware Procedural Content Generation via Meta-Reinforcement
         Learning for Heterogeneous Gaming Platforms},
  author={Redwan Rahman},
  link={https://github.com/Red1-Rahman/RAPCG-MetaRL}
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


## 📧 Contact

Redwan Rahman — rahman22205101127@diu.edu.bd  
Department of Computer Science and Engineering, Daffodil International University

Code: <https://github.com/Red1-Rahman/RAPCG-MetaRL>

---

_RAPCG-MetaRL — Resource-Aware PCG that adapts to your hardware. 🎮_
