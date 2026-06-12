# analysis.ps1 — Analysis & paper figures for RAPCG-MetaRL
# Usage: .\console\analysis.ps1
# Run from project root: D:\Work\thesis\RAPCG-MetaRL\

$PY = "d:/Work/thesis/RAPCG-MetaRL/pcg_env/Scripts/python.exe"

# ── Generate all paper figures (demo mode, no checkpoint needed) ──────────────
# $PY generate_paper_figures.py --demo

# ── Compare forward vs backward Sokoban generation ───────────────────────────
# $PY compare_approaches.py `
#     --forward-dir generated_levels/sokoban_forward `
#     --backward-dir generated_levels/sokoban_backward

# ── Action penalty analysis (which actions cause resource spikes) ─────────────
# $PY analyze_action_penalties.py

# ── Architecture diagram ──────────────────────────────────────────────────────
# $PY architecture_diagram.py

# ── Analyze training CSV log with pandas ─────────────────────────────────────
# $PY -c "
# import pandas as pd
# df = pd.read_csv('logs/<experiment>.csv')
# print(df.describe())
# print(df[['reward','ram_penalty','cpu_penalty','gpu_penalty']].tail(20))
# "

# ── Analyze inference timing CSV ─────────────────────────────────────────────
# $PY -c "
# import pandas as pd
# df = pd.read_csv('inference_timing.csv')
# print(df[['total_time_ms','generation_time_ms','mean_inference_ms','is_solvable']].describe())
# "

# ── Launch Jupyter for notebook analysis ─────────────────────────────────────
# $PY -m jupyter notebook graph.ipynb
# $PY -m jupyter notebook inference_graph.ipynb
# $PY -m jupyter notebook gym-pcgrl/inference.ipynb
