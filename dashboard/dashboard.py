"""
RAPCG-MetaRL Dashboard
Streamlit UI for training, inference, and level visualization.
"""

import os
import sys
import time
import subprocess
import threading
import queue
import glob
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAPCG-MetaRL",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PY = str(PROJECT_ROOT / "pcg_env" / "bin" / "python")
# Fallback for Windows
if not Path(PY).exists():
    PY = str(PROJECT_ROOT / "pcg_env" / "Scripts" / "python.exe")

sys.path.insert(0, str(PROJECT_ROOT))

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    /* Base */
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark terminal feel */
    .stApp {
        background-color: #0d1117;
        color: #e6edf3;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Cards */
    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 16px 20px;
        margin-bottom: 8px;
    }
    .metric-card .label {
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-card .value {
        font-size: 28px;
        font-weight: 600;
        color: #58a6ff;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 4px;
    }
    .metric-card .sub {
        font-size: 12px;
        color: #8b949e;
        margin-top: 2px;
    }

    /* Log box */
    .log-box {
        background: #0d1117;
        border: 1px solid #30363d;
        border-radius: 6px;
        padding: 12px 16px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #7ee787;
        min-height: 200px;
        max-height: 400px;
        overflow-y: auto;
        white-space: pre-wrap;
        word-break: break-word;
    }

    /* Status badges */
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .badge-running  { background: #1f3a1f; color: #7ee787; border: 1px solid #3fb950; }
    .badge-idle     { background: #1c2128; color: #8b949e; border: 1px solid #30363d; }
    .badge-done     { background: #1a2b4a; color: #58a6ff; border: 1px solid #388bfd; }
    .badge-error    { background: #3b1212; color: #f85149; border: 1px solid #da3633; }

    /* Section headers */
    .section-header {
        font-size: 11px;
        font-family: 'JetBrains Mono', monospace;
        color: #8b949e;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        border-bottom: 1px solid #21262d;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #21262d;
        color: #e6edf3;
        border: 1px solid #30363d;
        border-radius: 6px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #30363d;
        border-color: #58a6ff;
        color: #58a6ff;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background-color: #238636;
        border-color: #2ea043;
        color: #fff;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #2ea043;
    }

    /* Inputs */
    .stSelectbox > div, .stNumberInput > div, .stSlider {
        background-color: #161b22 !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
        border-bottom: 1px solid #30363d;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #8b949e;
        padding: 10px 20px;
        border-radius: 0;
    }
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
        background: transparent !important;
    }

    /* Level grid tiles */
    .level-grid { font-family: 'JetBrains Mono', monospace; font-size: 18px; line-height: 1.4; }
    .tile-0 { color: #21262d; }   /* empty */
    .tile-1 { color: #484f58; }   /* wall */
    .tile-2 { color: #f0883e; }   /* player */
    .tile-3 { color: #7ee787; }   /* crate / path */
    .tile-4 { color: #58a6ff; }   /* target / key */
    .tile-5 { color: #f85149; }   /* enemy */
    .tile-6 { color: #ffa657; }   /* door */

    /* Divider */
    hr { border-color: #21262d; }

    /* Dataframe */
    .stDataFrame { border: 1px solid #30363d; border-radius: 6px; }

    /* Progress bar */
    .stProgress > div > div { background-color: #238636; }

    /* Expander */
    .streamlit-expanderHeader {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        color: #8b949e;
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── Session state init ────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "train_process": None,
        "train_log": [],
        "train_status": "idle",
        "train_start_time": None,
        "infer_process": None,
        "infer_log": [],
        "infer_status": "idle",
        "generated_levels": [],
        "log_queue": queue.Queue(),
        "last_refresh": time.time(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ── Helpers ───────────────────────────────────────────────────────────────────
TILE_CHARS = {
    0: "·",  # empty
    1: "█",  # wall
    2: "☺",  # player
    3: "▣",  # crate
    4: "◎",  # target
    5: "☠",  # enemy
    6: "▤",  # door / key
}
TILE_CLASSES = {
    0: "tile-0",
    1: "tile-1",
    2: "tile-2",
    3: "tile-3",
    4: "tile-4",
    5: "tile-5",
    6: "tile-6",
}


def render_level_html(level: np.ndarray) -> str:
    rows = []
    for row in level:
        cells = ""
        for val in row:
            v = int(val)
            ch = TILE_CHARS.get(v, "?")
            cls = TILE_CLASSES.get(v, "tile-0")
            cells += f'<span class="{cls}">{ch}</span>'
        rows.append(cells)
    inner = "<br>".join(rows)
    return f'<div class="level-grid">{inner}</div>'


def status_badge(status: str) -> str:
    labels = {"idle": "IDLE", "running": "RUNNING", "done": "DONE", "error": "ERROR"}
    return f'<span class="badge badge-{status}">{labels.get(status, status.upper())}</span>'


def stream_process(proc, log_list: list, status_key: str):
    """Stream stdout/stderr from subprocess into log list."""

    def _read(stream):
        for line in iter(stream.readline, b""):
            decoded = line.decode("utf-8", errors="replace").rstrip()
            log_list.append(decoded)
        stream.close()

    t_out = threading.Thread(target=_read, args=(proc.stdout,), daemon=True)
    t_err = threading.Thread(target=_read, args=(proc.stderr,), daemon=True)
    t_out.start()
    t_err.start()

    def _wait():
        proc.wait()
        t_out.join()
        t_err.join()
        st.session_state[status_key] = "done" if proc.returncode == 0 else "error"

    threading.Thread(target=_wait, daemon=True).start()


def find_checkpoints() -> list:
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    if not ckpt_dir.exists():
        return []
    models = sorted(glob.glob(str(ckpt_dir / "**" / "*.zip"), recursive=True))
    models += sorted(glob.glob(str(ckpt_dir / "**" / "*.pt"), recursive=True))
    return models


def find_level_files() -> list:
    gen_dir = PROJECT_ROOT / "generated_levels"
    if not gen_dir.exists():
        return []
    return sorted(glob.glob(str(gen_dir / "**" / "*.npy"), recursive=True))


def find_log_csvs() -> list:
    log_dir = PROJECT_ROOT / "logs"
    if not log_dir.exists():
        return []
    return sorted(glob.glob(str(log_dir / "*.csv")), reverse=True)


def load_level(path: str) -> np.ndarray:
    try:
        return np.load(path)
    except Exception:
        return None


def resource_color(pct: float) -> str:
    if pct > 85:
        return "#f85149"
    if pct > 70:
        return "#ffa657"
    return "#7ee787"


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<div class="section-header">RAPCG-MetaRL</div>', unsafe_allow_html=True
    )
    st.markdown("**Thesis Dashboard**")
    st.markdown("Redwan Rahman · DIU", unsafe_allow_html=True)
    st.markdown("---")

    # Live resource monitor
    st.markdown('<div class="section-header">System</div>', unsafe_allow_html=True)
    try:
        import psutil

        cpu = psutil.cpu_percent(interval=0.2)
        ram = psutil.virtual_memory()
        ram_pct = ram.percent
        ram_used = ram.used / (1024**3)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="label">CPU</div>
                <div class="value" style="color:{resource_color(cpu)}">{cpu:.0f}%</div>
            </div>""",
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="label">RAM</div>
                <div class="value" style="color:{resource_color(ram_pct)}">{ram_pct:.0f}%</div>
                <div class="sub">{ram_used:.1f} / {ram.total / (1024**3):.0f} GB</div>
            </div>""",
                unsafe_allow_html=True,
            )

        # GPU
        try:
            import pynvml

            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            gpu_pct = (mem.used / mem.total) * 100
            gpu_util = util.gpu
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="label">GPU VRAM</div>
                <div class="value" style="color:{resource_color(gpu_pct)}">{gpu_pct:.0f}%</div>
                <div class="sub">{mem.used / (1024**2):.0f} / {mem.total / (1024**2):.0f} MB · util {gpu_util}%</div>
            </div>""",
                unsafe_allow_html=True,
            )
        except Exception:
            st.markdown(
                """
            <div class="metric-card">
                <div class="label">GPU</div>
                <div class="value" style="color:#8b949e">N/A</div>
            </div>""",
                unsafe_allow_html=True,
            )
    except ImportError:
        st.warning("psutil not available")

    st.markdown("---")

    # Process status
    st.markdown('<div class="section-header">Processes</div>', unsafe_allow_html=True)
    st.markdown(
        f"Training &nbsp; {status_badge(st.session_state.train_status)}",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"Inference {status_badge(st.session_state.infer_status)}",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("⟳ Refresh", use_container_width=True):
        st.rerun()

    # Auto-refresh toggle
    auto_refresh = st.checkbox("Auto-refresh (3s)", value=False)
    if auto_refresh:
        time.sleep(3)
        st.rerun()

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_train, tab_infer, tab_levels, tab_logs, tab_compare = st.tabs(
    ["⚡ Train", "🎲 Inference", "🗺 Levels", "📊 Logs", "⚖ Compare"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRAIN
# ══════════════════════════════════════════════════════════════════════════════
with tab_train:
    st.markdown(
        '<div class="section-header">Training Configuration</div>',
        unsafe_allow_html=True,
    )

    col_cfg, col_log = st.columns([1, 1], gap="large")

    with col_cfg:
        # Config form
        game = st.selectbox("Game", ["zelda", "sokoban", "binary"], index=0)
        algo = st.selectbox("Algorithm", ["PPO", "A2C"], index=0)
        representation = st.selectbox(
            "Representation", ["narrow", "wide", "turtle"], index=0
        )
        timesteps = st.number_input(
            "Timesteps", min_value=1000, max_value=1_000_000, value=50_000, step=10_000
        )

        col_a, col_b = st.columns(2)
        with col_a:
            n_envs = st.number_input(
                "Parallel envs",
                min_value=1,
                max_value=6,
                value=1,
                help="Max 6 on this hardware",
            )
            batch_size = st.number_input(
                "Batch size", min_value=16, max_value=512, value=64, step=16
            )
        with col_b:
            n_steps = st.number_input(
                "Steps/update", min_value=32, max_value=2048, value=128, step=32
            )
            lr = st.number_input(
                "Learning rate",
                min_value=1e-5,
                max_value=1e-2,
                value=2.5e-4,
                format="%.5f",
            )

        checkpoint_freq = st.number_input(
            "Checkpoint every N steps",
            min_value=500,
            max_value=50_000,
            value=5_000,
            step=500,
        )

        if game == "sokoban":
            sokoban_penalty = st.slider(
                "Sokoban unsolvable penalty",
                min_value=0.0,
                max_value=50.0,
                value=25.0,
                step=1.0,
            )
            use_backward = st.checkbox(
                "Use backward generation (guaranteed solvable)", value=False
            )
        else:
            sokoban_penalty = 25.0
            use_backward = False

        device = st.selectbox("Device", ["auto", "cuda", "cpu"], index=0)
        experiment_name = st.text_input(
            "Experiment name (optional)",
            value="",
            placeholder="auto-generated if blank",
        )

        st.markdown("---")

        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            start_clicked = st.button(
                "▶ Start Training",
                type="primary",
                use_container_width=True,
                disabled=(st.session_state.train_status == "running"),
            )
        with col_btn2:
            stop_clicked = st.button(
                "■ Stop",
                use_container_width=True,
                disabled=(st.session_state.train_status != "running"),
            )

    with col_log:
        st.markdown(
            '<div class="section-header">Live Output</div>', unsafe_allow_html=True
        )

        # Status + elapsed
        elapsed = ""
        if (
            st.session_state.train_start_time
            and st.session_state.train_status == "running"
        ):
            secs = int(time.time() - st.session_state.train_start_time)
            elapsed = f" · {secs // 3600:02d}:{(secs % 3600) // 60:02d}:{secs % 60:02d}"
        st.markdown(
            f"{status_badge(st.session_state.train_status)}{elapsed}",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        log_text = (
            "\n".join(st.session_state.train_log[-200:]) or "Waiting for output..."
        )
        st.markdown(f'<div class="log-box">{log_text}</div>', unsafe_allow_html=True)

        # Progress estimate from log
        if st.session_state.train_log:
            progress_val = 0.0
            for line in reversed(st.session_state.train_log):
                if "timesteps" in line.lower() and "/" in line:
                    try:
                        parts = [p.strip() for p in line.split("|")]
                        for p in parts:
                            if "/" in p and any(c.isdigit() for c in p):
                                nums = [
                                    int(x.replace(",", ""))
                                    for x in p.split("/")
                                    if x.strip().replace(",", "").isdigit()
                                ]
                                if len(nums) == 2 and nums[1] > 0:
                                    progress_val = min(nums[0] / nums[1], 1.0)
                                    break
                    except Exception:
                        pass
                    if progress_val > 0:
                        break
            if progress_val > 0:
                st.progress(progress_val, text=f"{progress_val * 100:.1f}% complete")

    # ── Start / stop logic ────────────────────────────────────────────────────
    if start_clicked:
        st.session_state.train_log = []
        st.session_state.train_status = "running"
        st.session_state.train_start_time = time.time()

        if use_backward and game == "sokoban":
            cmd = [
                PY,
                str(PROJECT_ROOT / "train_backward.py"),
                "--game",
                game,
                "--timesteps",
                str(timesteps),
                "--device",
                device,
            ]
        else:
            cmd = [
                PY,
                str(PROJECT_ROOT / "train.py"),
                "--game",
                game,
                "--algorithm",
                algo,
                "--representation",
                representation,
                "--timesteps",
                str(timesteps),
                "--n-envs",
                str(n_envs),
                "--batch-size",
                str(batch_size),
                "--n-steps",
                str(n_steps),
                "--lr",
                str(lr),
                "--checkpoint-freq",
                str(checkpoint_freq),
                "--device",
                device,
                "--sokoban-penalty",
                str(sokoban_penalty),
            ]
            if experiment_name.strip():
                cmd += ["--experiment-name", experiment_name.strip()]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(PROJECT_ROOT)
        )
        st.session_state.train_process = proc
        stream_process(proc, st.session_state.train_log, "train_status")
        st.rerun()

    if stop_clicked and st.session_state.train_process:
        st.session_state.train_process.terminate()
        st.session_state.train_status = "idle"
        st.session_state.train_log.append("— Training stopped by user —")
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
with tab_infer:
    st.markdown(
        '<div class="section-header">Inference Configuration</div>',
        unsafe_allow_html=True,
    )

    col_icfg, col_ilog = st.columns([1, 1], gap="large")

    with col_icfg:
        checkpoints = find_checkpoints()
        if checkpoints:
            ckpt_labels = [
                Path(c).relative_to(PROJECT_ROOT).as_posix() for c in checkpoints
            ]
            ckpt_idx = st.selectbox(
                "Checkpoint",
                range(len(ckpt_labels)),
                format_func=lambda i: ckpt_labels[i],
            )
            selected_ckpt = checkpoints[ckpt_idx]
        else:
            st.warning("No checkpoints found. Train a model first.")
            selected_ckpt = None

        infer_game = st.selectbox(
            "Game ", ["zelda", "sokoban", "binary"], key="infer_game"
        )
        infer_mode = st.selectbox("Mode", ["Standard PPO/A2C", "MAML"], index=0)
        n_levels = st.number_input(
            "Levels to generate", min_value=1, max_value=100, value=10
        )
        max_steps = st.number_input(
            "Max steps/level", min_value=50, max_value=2000, value=500, step=50
        )
        infer_device = st.selectbox(
            "Device ", ["auto", "cuda", "cpu"], index=0, key="infer_device"
        )

        if infer_mode == "MAML":
            adapt_steps = st.number_input(
                "Adaptation steps (0 = meta-weights directly)",
                min_value=0,
                max_value=20,
                value=0,
            )
        else:
            adapt_steps = 0

        log_file = st.text_input("Output CSV name", value="inference_timing.csv")

        st.markdown("---")
        run_infer = st.button(
            "▶ Run Inference",
            type="primary",
            use_container_width=True,
            disabled=(
                st.session_state.infer_status == "running" or selected_ckpt is None
            ),
        )

    with col_ilog:
        st.markdown(
            '<div class="section-header">Live Output</div>', unsafe_allow_html=True
        )
        st.markdown(status_badge(st.session_state.infer_status), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        ilog_text = (
            "\n".join(st.session_state.infer_log[-200:]) or "Waiting for output..."
        )
        st.markdown(f'<div class="log-box">{ilog_text}</div>', unsafe_allow_html=True)

    if run_infer and selected_ckpt:
        st.session_state.infer_log = []
        st.session_state.infer_status = "running"

        if infer_mode == "MAML":
            cmd = [
                PY,
                str(PROJECT_ROOT / "maml_inference_timed.py"),
                selected_ckpt,
                "--game",
                infer_game,
                "--n-levels",
                str(n_levels),
                "--max-steps",
                str(max_steps),
                "--adapt-steps",
                str(adapt_steps),
                "--log-file",
                str(PROJECT_ROOT / log_file),
                "--device",
                infer_device,
            ]
        else:
            cmd = [
                PY,
                str(PROJECT_ROOT / "inference_timed.py"),
                selected_ckpt,
                "--game",
                infer_game,
                "--n-levels",
                str(n_levels),
                "--max-steps",
                str(max_steps),
                "--log-file",
                str(PROJECT_ROOT / log_file),
                "--device",
                infer_device,
            ]

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(PROJECT_ROOT)
        )
        st.session_state.infer_process = proc
        stream_process(proc, st.session_state.infer_log, "infer_status")
        st.rerun()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — LEVELS
# ══════════════════════════════════════════════════════════════════════════════
with tab_levels:
    st.markdown(
        '<div class="section-header">Generated Levels</div>', unsafe_allow_html=True
    )

    level_files = find_level_files()

    if not level_files:
        st.info("No generated levels found. Run inference first.")
    else:
        col_ctrl, col_info = st.columns([2, 1])
        with col_ctrl:
            # Group by subdirectory
            dirs = sorted(set(str(Path(f).parent) for f in level_files))
            dir_labels = [Path(d).relative_to(PROJECT_ROOT).as_posix() for d in dirs]
            selected_dir_idx = st.selectbox(
                "Level set", range(len(dir_labels)), format_func=lambda i: dir_labels[i]
            )
            selected_dir = dirs[selected_dir_idx]

        files_in_dir = [f for f in level_files if str(Path(f).parent) == selected_dir]

        with col_info:
            st.markdown(
                f"""
            <div class="metric-card">
                <div class="label">Levels in set</div>
                <div class="value">{len(files_in_dir)}</div>
            </div>""",
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # Grid display — 4 per row
        cols_per_row = 4
        for row_start in range(0, len(files_in_dir), cols_per_row):
            row_files = files_in_dir[row_start : row_start + cols_per_row]
            cols = st.columns(cols_per_row)
            for col, fpath in zip(cols, row_files):
                level = load_level(fpath)
                if level is None:
                    continue
                name = Path(fpath).stem
                with col:
                    st.markdown(f"**{name}**")

                    # Try PNG first
                    png_path = fpath.replace(".npy", ".png")
                    if Path(png_path).exists():
                        st.image(png_path, use_column_width=True)
                    else:
                        # Render as ASCII grid
                        html = render_level_html(level)
                        st.markdown(html, unsafe_allow_html=True)

                    # Mini stats
                    unique = len(np.unique(level))
                    size = f"{level.shape[0]}×{level.shape[1]}"
                    st.caption(f"{size} · {unique} tile types")

                    # Download button
                    txt_path = fpath.replace(".npy", ".txt")
                    if Path(txt_path).exists():
                        with open(txt_path) as f:
                            st.download_button(
                                "↓ .txt",
                                f.read(),
                                file_name=Path(txt_path).name,
                                key=f"dl_{fpath}",
                            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — LOGS
# ══════════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.markdown(
        '<div class="section-header">Training Logs</div>', unsafe_allow_html=True
    )

    log_csvs = find_log_csvs()

    if not log_csvs:
        st.info("No log files found yet.")
    else:
        log_labels = [Path(f).name for f in log_csvs]
        selected_log = st.selectbox(
            "Log file", log_csvs, format_func=lambda f: Path(f).name
        )

        try:
            df = pd.read_csv(selected_log)
            st.markdown(
                f"**{len(df):,} steps · {df['episode'].max() if 'episode' in df.columns else '?'} episodes**"
            )

            # Summary metrics row
            if "reward" in df.columns:
                m1, m2, m3, m4 = st.columns(4)
                with m1:
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="label">Mean Reward</div>
                        <div class="value">{df["reward"].mean():.3f}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )
                with m2:
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="label">Max Reward</div>
                        <div class="value">{df["reward"].max():.3f}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )
                with m3:
                    penalty_cols = [c for c in df.columns if "penalty_total" in c]
                    avg_pen = df[penalty_cols[0]].mean() if penalty_cols else 0.0
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="label">Avg Penalty</div>
                        <div class="value" style="color:#f85149">{avg_pen:.3f}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )
                with m4:
                    st.markdown(
                        f"""<div class="metric-card">
                        <div class="label">Total Steps</div>
                        <div class="value">{len(df):,}</div>
                    </div>""",
                        unsafe_allow_html=True,
                    )

            st.markdown("---")

            # Charts
            chart_col1, chart_col2 = st.columns(2)
            with chart_col1:
                if "reward" in df.columns:
                    # Rolling mean
                    smoothed = (
                        df["reward"].rolling(window=min(200, len(df) // 10 or 1)).mean()
                    )
                    chart_df = pd.DataFrame(
                        {"reward": df["reward"], "smoothed": smoothed}
                    )
                    st.markdown("**Reward**")
                    st.line_chart(chart_df, color=["#30363d", "#58a6ff"])

            with chart_col2:
                resource_cols = [
                    c
                    for c in ["ram_percent", "cpu_percent", "gpu_mem_percent"]
                    if c in df.columns
                ]
                if resource_cols:
                    st.markdown("**Resource Usage %**")
                    st.line_chart(
                        df[resource_cols].iloc[:: max(1, len(df) // 500)],
                        color=["#f85149", "#ffa657", "#7ee787"][: len(resource_cols)],
                    )

            # Penalty breakdown chart
            penalty_cols = [
                c
                for c in df.columns
                if c.startswith("penalty_") and c != "penalty_total_penalty"
            ]
            if penalty_cols:
                st.markdown("**Penalty Breakdown**")
                st.line_chart(df[penalty_cols].iloc[:: max(1, len(df) // 500)])

            st.markdown("---")
            with st.expander("Raw data (last 500 rows)"):
                st.dataframe(df.tail(500), use_container_width=True)

            # Download
            st.download_button(
                "↓ Download CSV",
                df.to_csv(index=False),
                file_name=Path(selected_log).name,
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Error reading log: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — COMPARE
# ══════════════════════════════════════════════════════════════════════════════
with tab_compare:
    st.markdown(
        '<div class="section-header">Compare Runs</div>', unsafe_allow_html=True
    )

    log_csvs = find_log_csvs()

    if len(log_csvs) < 2:
        st.info("Need at least 2 training runs to compare.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            run_a = st.selectbox(
                "Run A", log_csvs, format_func=lambda f: Path(f).name, key="cmp_a"
            )
        with col_b:
            run_b = st.selectbox(
                "Run B",
                log_csvs,
                format_func=lambda f: Path(f).name,
                index=min(1, len(log_csvs) - 1),
                key="cmp_b",
            )

        if run_a and run_b and run_a != run_b:
            try:
                df_a = pd.read_csv(run_a)
                df_b = pd.read_csv(run_b)

                # Summary table
                def run_summary(df, name):
                    s = {"Run": Path(name).stem, "Steps": len(df)}
                    if "reward" in df.columns:
                        s["Mean Reward"] = f"{df['reward'].mean():.4f}"
                        s["Max Reward"] = f"{df['reward'].max():.4f}"
                    if "ram_percent" in df.columns:
                        s["Avg RAM %"] = f"{df['ram_percent'].mean():.1f}"
                    pen = [c for c in df.columns if "penalty_total" in c]
                    if pen:
                        s["Avg Penalty"] = f"{df[pen[0]].mean():.4f}"
                    return s

                summary = pd.DataFrame(
                    [
                        run_summary(df_a, run_a),
                        run_summary(df_b, run_b),
                    ]
                )
                st.dataframe(summary, use_container_width=True, hide_index=True)

                st.markdown("---")

                # Reward comparison chart
                if "reward" in df_a.columns and "reward" in df_b.columns:
                    st.markdown("**Reward Comparison (rolling mean)**")
                    window = 200
                    min_len = min(len(df_a), len(df_b))
                    smooth_a = df_a["reward"].rolling(window).mean().iloc[:min_len]
                    smooth_b = df_b["reward"].rolling(window).mean().iloc[:min_len]
                    cmp_df = pd.DataFrame(
                        {
                            Path(run_a).stem[:30]: smooth_a.values,
                            Path(run_b).stem[:30]: smooth_b.values,
                        }
                    )
                    st.line_chart(cmp_df, color=["#58a6ff", "#7ee787"])

                # Resource comparison
                if "ram_percent" in df_a.columns and "ram_percent" in df_b.columns:
                    st.markdown("**RAM Usage Comparison**")
                    stride = max(1, min_len // 500)
                    ram_df = pd.DataFrame(
                        {
                            Path(run_a).stem[:30]: df_a["ram_percent"]
                            .iloc[::stride]
                            .values[: min_len // stride],
                            Path(run_b).stem[:30]: df_b["ram_percent"]
                            .iloc[::stride]
                            .values[: min_len // stride],
                        }
                    )
                    st.line_chart(ram_df, color=["#58a6ff", "#7ee787"])

            except Exception as e:
                st.error(f"Error comparing runs: {e}")

        elif run_a == run_b:
            st.warning("Select two different runs to compare.")
