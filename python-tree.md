# Codebase tree of `.py` files:

```
root
├── dashboard
│ └── dashboard.py
├── gym-pcgrl
│ ├── gym_pcgrl
│ │ ├── envs
│ │ │ ├── probs
│ │ │ │ ├── ddave
│ │ │ │ │ ├── engine.py
│ │ │ │ │ └── **init**.py
│ │ │ │ ├── mdungeon
│ │ │ │ │ ├── engine.py
│ │ │ │ │ └── **init**.py
│ │ │ │ ├── smb
│ │ │ │ │ ├── engine.py
│ │ │ │ │ └── **init**.py
│ │ │ │ ├── sokoban
│ │ │ │ │ ├── engine.py
│ │ │ │ │ └── **init**.py
│ │ │ │ ├── binary_prob.py
│ │ │ │ ├── ddave_prob.py
│ │ │ │ ├── mdungeon_prob.py
│ │ │ │ ├── problem.py
│ │ │ │ ├── smb_prob.py
│ │ │ │ ├── sokoban_prob.py
│ │ │ │ ├── zelda_prob.py
│ │ │ │ └── **init**.py
│ │ │ ├── reps
│ │ │ │ ├── narrow_cast_rep.py
│ │ │ │ ├── narrow_multi_rep.py
│ │ │ │ ├── narrow_rep.py
│ │ │ │ ├── representation.py
│ │ │ │ ├── turtle_cast_rep.py
│ │ │ │ ├── turtle_rep.py
│ │ │ │ ├── wide_rep.py
│ │ │ │ └── **init**.py
│ │ │ ├── helper.py
│ │ │ ├── pcgrl_env.py
│ │ │ ├── sokoban_reverse_env.py
│ │ │ └── **init**.py
│ │ ├── wrappers.py
│ │ └── **init**.py
│ ├── inference.ipynb
│ ├── inference.py
│ ├── model.py
│ ├── setup.py
│ ├── train.py
│ └── utils.py
├── test
│ ├── test.py
│ └── **init**.py
├── wrappers
│ ├── helper.py
│ ├── pcgrl_env.py
│ └── **init**.py
├── analyze_action_penalties.py
├── analyze_maml_results.py
├── architecture_diagram.py
├── compare_approaches.py
├── config_hardware.py
├── fix_sokoban_prefered_levels.py
├── generate_paper_figures.py
├── graph.ipynb
├── inference.py
├── inference_graph.ipynb
├── inference_timed.py
├── maml_inference_timed.py
├── maml_trainer.py
├── model.py
├── quickstart.py
├── rlhf_trainer.py
├── sokoban_utils.py
├── sokoban_utils_backup.py
├── solvability_config.py
├── test_action_space.py
├── test_obs_shape.py
├── test_sokoban_solvability.py
├── test_solvability_integration.py
├── test_solver_integration.py
├── test_trust_model.py
├── train.py
├── train_backward.py
├── utils.py
└── visualize_levels.py
```
