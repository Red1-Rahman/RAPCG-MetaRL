# inference.ps1 — Inference & level generation for RAPCG-MetaRL
# Usage: .\console\inference.ps1
# Run from project root: D:\Work\thesis\RAPCG-MetaRL\

$PY = "d:/Work/thesis/RAPCG-MetaRL/pcg_env/Scripts/python.exe"

# ── Standard inference (basic) ────────────────────────────────────────────────
# $PY inference.py checkpoints/<experiment>/final_model.zip --n-levels 10

# ── Timed inference — Zelda (paper metrics, CSV output) ──────────────────────
# $PY inference_timed.py checkpoints/<experiment>/final_model.zip `
#     --game zelda --n-levels 20 --log-file inference_timing_zelda.csv --device auto

# ── Timed inference — Sokoban ─────────────────────────────────────────────────
# $PY inference_timed.py checkpoints/<experiment>/final_model.zip `
#     --game sokoban --n-levels 20 --log-file inference_timing_sokoban.csv --device auto

# ── MAML inference (meta-weights directly, fastest) ──────────────────────────
# $PY maml_inference_timed.py checkpoints/sokoban_MAML_inference/best_meta_model.pt `
#     --game sokoban --n-levels 20 --adapt-steps 0 --device auto

# ── MAML inference (with 5-step inner-loop adaptation) ───────────────────────
# $PY maml_inference_timed.py checkpoints/sokoban_MAML_inference/best_meta_model.pt `
#     --game sokoban --n-levels 20 --adapt-steps 5 --device auto

# ── MAML inference (stochastic policy) ───────────────────────────────────────
# $PY maml_inference_timed.py checkpoints/sokoban_MAML_inference/best_meta_model.pt `
#     --game sokoban --n-levels 20 --stochastic --device auto

# ── Generate many levels for comparison study ─────────────────────────────────
# $PY inference_timed.py checkpoints/<experiment>/final_model.zip `
#     --game sokoban --n-levels 100 --max-steps 500 `
#     --log-file generated_levels/sokoban_forward/timing.csv `
#     --device auto

# ── Visualize generated levels ────────────────────────────────────────────────
# $PY visualize_levels.py
