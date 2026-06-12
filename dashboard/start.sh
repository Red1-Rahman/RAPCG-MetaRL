#!/usr/bin/env bash
# start.sh — Launch RAPCG-MetaRL dashboard
# Usage: bash dashboard/start.sh
# Run from project root: ~/Work/thesis/RAPCG-MetaRL/

PY="./pcg_env/bin/python"

# Install streamlit if missing
$PY -m pip install streamlit --quiet

# Launch
$PY -m streamlit run dashboard/dashboard.py \
    --server.port 8501 \
    --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false