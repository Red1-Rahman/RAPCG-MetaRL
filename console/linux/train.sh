#!/usr/bin/env bash
# train.sh — Training workflows for RAPCG-MetaRL
# Usage: bash console/train.sh
# Run from project root: ~/Work/thesis/RAPCG-MetaRL/

PY="./pcg_env/bin/python"

# ── Quick smoke test (10k steps) ──────────────────────────────────────────────
# $PY train.py --game zelda --timesteps 10000 --checkpoint-freq 2000 --device auto

# ── Zelda PPO (recommended, 100k steps) ──────────────────────────────────────
# $PY train.py --game zelda --algorithm PPO --timesteps 100000 --n-envs 6 --device cuda

# ── Zelda A2C ─────────────────────────────────────────────────────────────────
# $PY train.py --game zelda --algorithm A2C --timesteps 100000 --n-envs 6 --device cuda

# ── Sokoban PPO (forward, with solvability tuning) ───────────────────────────
# $PY train.py --game sokoban --algorithm PPO --timesteps 100000 --n-envs 4 \
#     --sokoban-penalty 25.0 --device cuda

# ── Sokoban BACKWARD (guaranteed solvability) ─────────────────────────────────
# $PY train_backward.py --game sokoban --timesteps 50000 --device auto

# ── Binary PPO ────────────────────────────────────────────────────────────────
# $PY train.py --game binary --algorithm PPO --timesteps 100000 --n-envs 6 --device cuda

# ── MAML (multi-game task distribution, first-order) ─────────────────────────
# $PY maml_trainer.py --games zelda sokoban binary --meta-batch 4 --iterations 500 --device auto

# ── MAML (single game, second-order, slower but more accurate) ────────────────
# $PY maml_trainer.py --games sokoban --meta-batch 2 --iterations 500 --second-order --device cuda

# ── RLHF (synthetic preferences — safe for testing) ──────────────────────────
# $PY rlhf_trainer.py --game zelda --synthetic --n-levels 50 --n-comparisons 50 --device auto

# ── RLHF (real human annotation session) ─────────────────────────────────────
# $PY rlhf_trainer.py --game zelda --interactive --n-levels 50 --n-comparisons 50 --device auto

# ── Resume from checkpoint ────────────────────────────────────────────────────
# $PY train.py --game zelda --load-model checkpoints/<experiment>/final_model.zip \
#     --timesteps 50000 --device auto

# ── Low-resource fallback (OOM safety) ───────────────────────────────────────
# $PY train.py --game zelda --timesteps 50000 --n-envs 1 --batch-size 32 --device auto

# ── Evaluate immediately after training ──────────────────────────────────────
# $PY train.py --game zelda --timesteps 50000 --evaluate --device auto

# NOTE: SAC is NOT recommended — gym-pcgrl uses discrete action spaces.
#       SAC requires continuous action spaces and will fail or produce poor results.
#       Always use PPO or A2C.
