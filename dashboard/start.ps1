# start.ps1 — Launch RAPCG-MetaRL dashboard
# Usage: .\dashboard\start.ps1
# Run from project root: D:\Work\thesis\RAPCG-MetaRL\

$PY = "d:/Work/thesis/RAPCG-MetaRL/pcg_env/Scripts/python.exe"

# Install streamlit if missing
& $PY -m pip install streamlit --quiet

# Launch
& $PY -m streamlit run dashboard/dashboard.py `
    --server.port 8501 `
    --server.address 0.0.0.0 `
    --server.headless true `
    --browser.gatherUsageStats false

# Opens at http://localhost:8501