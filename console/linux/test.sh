#!/usr/bin/env bash
# test.sh — Test suite for RAPCG-MetaRL
# Usage: bash console/test.sh
# Run from project root: ~/Work/thesis/RAPCG-MetaRL/

PY="./pcg_env/bin/python"

# ── Primary test suite (5 categories — all should pass) ──────────────────────
# $PY test/test.py

# ── Standalone targeted tests ─────────────────────────────────────────────────
# $PY test_action_space.py
# $PY test_obs_shape.py
# $PY test_sokoban_solvability.py
# $PY test_solvability_integration.py
# $PY test_solver_integration.py
# $PY test_trust_model.py

# ── Quick environment sanity check ────────────────────────────────────────────
# $PY wrappers/pcgrl_env.py

# ── Verify CUDA is available ──────────────────────────────────────────────────
# $PY -c "import torch; print('CUDA:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

# ── Verify gym-pcgrl is installed ────────────────────────────────────────────
# $PY -c "import gym_pcgrl; print('gym-pcgrl OK')"

# ── Verify all key imports ────────────────────────────────────────────────────
# $PY -c "from utils import ResourceMonitor, TrainingLogger; from wrappers.pcgrl_env import make_pcgrl_env; print('All imports OK')"

# ── Venv integrity check (run if all 5 tests fail) ────────────────────────────
# $PY -m pip check

# ── Reinstall gym-pcgrl (run if import errors persist) ───────────────────────
# pushd gym-pcgrl && $PY -m pip install -e . && popd
