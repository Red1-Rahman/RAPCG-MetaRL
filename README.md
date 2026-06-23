# RAPCG-MetaRL

**Resource-Aware Procedural Content Generation via Meta-Reinforcement Learning**


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
