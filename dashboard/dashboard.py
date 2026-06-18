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
import random
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

try:
    from wrappers.helper import calculate_content_metrics
except Exception:

    def calculate_content_metrics(level: np.ndarray) -> dict:
        unique, counts = np.unique(level, return_counts=True)
        probs = counts / max(1, counts.sum())
        entropy = float(-(probs * np.log2(probs + 1e-12)).sum())
        return {
            "diversity": float(len(unique) / max(1, level.size)),
            "complexity": entropy,
        }


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
        "rlhf_pair": None,
        "rlhf_pair_source": None,
        "log_queue": queue.Queue(),
        "last_refresh": time.time(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()

# ── Poll background processes ──────────────────────────────────────────────────
if (
    st.session_state.train_status == "running"
    and st.session_state.train_process is not None
):
    poll = st.session_state.train_process.poll()
    if poll is not None:
        st.session_state.train_status = "done" if poll == 0 else "error"
        st.session_state.train_process = None

if (
    st.session_state.infer_status == "running"
    and st.session_state.infer_process is not None
):
    poll = st.session_state.infer_process.poll()
    if poll is not None:
        st.session_state.infer_status = "done" if poll == 0 else "error"
        st.session_state.infer_process = None

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


def stream_process(proc, log_list: list):
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


def preference_file(game: str) -> Path:
    return PROJECT_ROOT / "data" / "preferences" / game / "preferences.json"


def load_preferences(game: str) -> list:
    path = preference_file(game)
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_dashboard_preference(
    game: str,
    level_a: np.ndarray,
    level_b: np.ndarray,
    preference: float,
    metadata: dict,
) -> int:
    path = preference_file(game)
    path.parent.mkdir(parents=True, exist_ok=True)
    prefs = load_preferences(game)
    prefs.append(
        {
            "level_a": level_a.tolist(),
            "level_b": level_b.tolist(),
            "preference": preference,
            "metrics_a": calculate_content_metrics(level_a),
            "metrics_b": calculate_content_metrics(level_b),
            "metadata": metadata,
            "timestamp": datetime.now().isoformat(),
        }
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2, default=str)
    return len(prefs)


def format_metrics(level: np.ndarray) -> str:
    metrics = calculate_content_metrics(level)
    return "div={:.3f}  complexity={:.3f}".format(
        metrics.get("diversity", 0.0),
        metrics.get("complexity", 0.0),
    )


def select_feedback_pair(files: list, source_key: str) -> tuple:
    if len(files) < 2:
        return None
    cached = st.session_state.rlhf_pair
    if cached and st.session_state.rlhf_pair_source == source_key:
        if all(Path(p).exists() for p in cached):
            return cached
    pair = tuple(random.sample(files, 2))
    st.session_state.rlhf_pair = pair
    st.session_state.rlhf_pair_source = source_key
    return pair


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
_legacy_tab_labels = ["⚡ Train", "🎲 Inference", "🗺 Levels", "📊 Logs", "⚖ Compare"]

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TRAIN
# ══════════════════════════════════════════════════════════════════════════════
tab_train, tab_infer, tab_levels, tab_feedback, tab_logs, tab_compare = st.tabs(
    ["Train", "Inference", "Levels", "RLHF Feedback", "Logs", "Compare"]
)

with tab_train:
    st.markdown(
        '<div class="section-header">Training Configuration</div>',
        unsafe_allow_html=True,
    )

    col_cfg, col_log = st.columns([1, 1], gap="large")

    with col_cfg:
        # Config form
        training_mode = st.selectbox(
            "Workflow",
            ["Standard PPO/A2C/SAC", "MAML meta-training", "RLHF fine-tuning"],
            index=0,
        )
        game = st.selectbox("Game", ["zelda", "sokoban", "binary"], index=0)
        algo = st.selectbox("Algorithm", ["PPO", "A2C", "SAC"], index=0)
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

        if training_mode == "MAML meta-training":
            st.markdown("**MAML settings**")
            maml_games = st.multiselect(
                "Task games",
                ["zelda", "sokoban", "binary"],
                default=["zelda", "sokoban", "binary"],
            )
            maml_representations = st.multiselect(
                "Task representations",
                ["narrow", "wide", "turtle"],
                default=["narrow", "wide", "turtle"],
            )
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                maml_iterations = st.number_input(
                    "Meta iterations", min_value=1, max_value=10_000, value=500
                )
                maml_meta_batch = st.number_input(
                    "Meta batch", min_value=1, max_value=16, value=4
                )
                maml_inner_steps = st.number_input(
                    "Inner steps", min_value=1, max_value=50, value=5
                )
            with col_m2:
                maml_trajectories = st.number_input(
                    "Trajectory steps", min_value=16, max_value=2048, value=128, step=16
                )
                maml_meta_lr = st.number_input(
                    "Meta LR", min_value=1e-5, max_value=1e-1, value=1e-3, format="%.5f"
                )
                maml_inner_lr = st.number_input(
                    "Inner LR",
                    min_value=1e-5,
                    max_value=1e-1,
                    value=1e-2,
                    format="%.5f",
                )
            maml_second_order = st.checkbox("Use second-order MAML", value=False)
        else:
            maml_games = []
            maml_representations = []
            maml_iterations = 500
            maml_meta_batch = 4
            maml_inner_steps = 5
            maml_trajectories = 128
            maml_meta_lr = 1e-3
            maml_inner_lr = 1e-2
            maml_second_order = False

        if training_mode == "RLHF fine-tuning":
            st.markdown("**RLHF settings**")
            zip_checkpoints = [c for c in find_checkpoints() if c.endswith(".zip")]
            base_options = ["(random init)"] + zip_checkpoints
            base_model_choice = st.selectbox(
                "Base PPO model",
                base_options,
                format_func=lambda p: p
                if p == "(random init)"
                else Path(p).relative_to(PROJECT_ROOT).as_posix(),
            )
            feedback_source = st.selectbox(
                "Feedback source",
                ["Dashboard preferences", "Synthetic preferences"],
                index=0,
                help="Collect real human labels in the RLHF Feedback tab.",
            )
            existing_pref_count = len(load_preferences(game))
            if feedback_source == "Dashboard preferences":
                st.caption(f"{existing_pref_count} saved preference(s) for {game}.")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                rlhf_weight = st.slider(
                    "Human reward weight", min_value=0.0, max_value=1.0, value=0.5
                )
                rlhf_levels = st.number_input(
                    "Feedback levels", min_value=2, max_value=500, value=50
                )
                rlhf_comparisons = st.number_input(
                    "Comparisons", min_value=1, max_value=500, value=50
                )
            with col_r2:
                rlhf_timesteps = st.number_input(
                    "Fine-tune timesteps",
                    min_value=0,
                    max_value=1_000_000,
                    value=50_000,
                    step=5_000,
                )
                reward_epochs = st.number_input(
                    "Reward epochs", min_value=1, max_value=1000, value=100
                )
                reward_model_only = st.checkbox("Reward model only", value=False)
        else:
            base_model_choice = "(random init)"
            feedback_source = "Synthetic preferences"
            existing_pref_count = 0
            rlhf_weight = 0.5
            rlhf_levels = 50
            rlhf_comparisons = 50
            rlhf_timesteps = 50_000
            reward_epochs = 100
            reward_model_only = False

        st.markdown("---")

        col_btn1, col_btn2 = st.columns(2)
        start_disabled = st.session_state.train_status == "running"
        if training_mode == "MAML meta-training":
            start_disabled = (
                start_disabled or not maml_games or not maml_representations
            )
        if (
            training_mode == "RLHF fine-tuning"
            and feedback_source == "Dashboard preferences"
        ):
            start_disabled = start_disabled or existing_pref_count == 0
        with col_btn1:
            start_clicked = st.button(
                "▶ Start Training",
                type="primary",
                use_container_width=True,
                disabled=start_disabled,
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

        if training_mode == "MAML meta-training":
            cmd = [
                PY,
                str(PROJECT_ROOT / "maml_trainer.py"),
                "--games",
                *maml_games,
                "--representations",
                *maml_representations,
                "--meta-lr",
                str(maml_meta_lr),
                "--inner-lr",
                str(maml_inner_lr),
                "--inner-steps",
                str(maml_inner_steps),
                "--meta-batch",
                str(maml_meta_batch),
                "--iterations",
                str(maml_iterations),
                "--n-trajectories",
                str(maml_trajectories),
                "--device",
                device,
            ]
            if maml_second_order:
                cmd.append("--second-order")
            if experiment_name.strip():
                cmd += ["--experiment-name", experiment_name.strip()]
        elif training_mode == "RLHF fine-tuning":
            cmd = [
                PY,
                str(PROJECT_ROOT / "rlhf_trainer.py"),
                "--game",
                game,
                "--representation",
                representation,
                "--rlhf-weight",
                str(rlhf_weight),
                "--n-levels",
                str(rlhf_levels),
                "--n-comparisons",
                str(rlhf_comparisons),
                "--timesteps",
                str(rlhf_timesteps),
                "--reward-epochs",
                str(reward_epochs),
                "--device",
                device,
            ]
            if base_model_choice != "(random init)":
                cmd += ["--base-model", base_model_choice]
            if feedback_source == "Dashboard preferences":
                cmd.append("--use-existing-preferences")
            else:
                cmd.append("--synthetic")
            if reward_model_only:
                cmd.append("--reward-model-only")
            if experiment_name.strip():
                cmd += ["--experiment-name", experiment_name.strip()]
        elif use_backward and game == "sokoban":
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

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(PROJECT_ROOT), env=env
        )
        st.session_state.train_process = proc
        stream_process(proc, st.session_state.train_log)
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

        # Select algorithm (only relevant for standard SB3 models)
        if infer_mode == "Standard PPO/A2C":
            infer_algo = st.selectbox(
                "Algorithm ", ["PPO", "A2C"], index=0, key="infer_algo"
            )
        else:
            infer_algo = "PPO"

        infer_repr = st.selectbox(
            "Representation ", ["narrow", "wide", "turtle"], index=0, key="infer_repr"
        )

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
                "--representation",
                infer_repr,
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
                "--algorithm",
                infer_algo,
                "--representation",
                infer_repr,
                "--n-levels",
                str(n_levels),
                "--max-steps",
                str(max_steps),
                "--log-file",
                str(PROJECT_ROOT / log_file),
                "--device",
                infer_device,
                "--save-dir",
                str(
                    PROJECT_ROOT
                    / "generated_levels"
                    / f"{infer_game}_{infer_algo}_{infer_repr}_standard"
                ),
            ]

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(PROJECT_ROOT), env=env
        )
        st.session_state.infer_process = proc
        stream_process(proc, st.session_state.infer_log)
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
                        st.image(png_path, width="stretch")
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
with tab_feedback:
    st.markdown(
        '<div class="section-header">RLHF Human Preference Collection</div>',
        unsafe_allow_html=True,
    )

    feedback_game = st.selectbox(
        "Feedback game", ["zelda", "sokoban", "binary"], key="feedback_game"
    )
    pref_count = len(load_preferences(feedback_game))
    pref_path = preference_file(feedback_game).relative_to(PROJECT_ROOT).as_posix()
    st.caption(f"{pref_count} saved preference(s) in {pref_path}")

    level_files = find_level_files()
    if len(level_files) < 2:
        st.info("Generate at least two levels first, then return here to label pairs.")
    else:
        dirs = sorted(set(str(Path(f).parent) for f in level_files))
        dir_labels = [Path(d).relative_to(PROJECT_ROOT).as_posix() for d in dirs]
        selected_feedback_dir_idx = st.selectbox(
            "Level source",
            range(len(dir_labels)),
            format_func=lambda i: dir_labels[i],
            key="feedback_dir",
        )
        selected_feedback_dir = dirs[selected_feedback_dir_idx]
        files_in_feedback_dir = [
            f for f in level_files if str(Path(f).parent) == selected_feedback_dir
        ]

        source_key = f"{feedback_game}:{selected_feedback_dir}"
        if st.button("New Pair", key="rlhf_new_pair"):
            st.session_state.rlhf_pair = None
            st.session_state.rlhf_pair_source = None
            st.rerun()

        pair = select_feedback_pair(files_in_feedback_dir, source_key)
        if pair is None:
            st.warning("This level source needs at least two .npy files.")
        else:
            level_a = load_level(pair[0])
            level_b = load_level(pair[1])
            if level_a is None or level_b is None:
                st.error("Could not load one of the selected level files.")
            else:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**A**")
                    png_a = pair[0].replace(".npy", ".png")
                    if Path(png_a).exists():
                        st.image(png_a, width="stretch")
                    else:
                        st.markdown(render_level_html(level_a), unsafe_allow_html=True)
                    st.caption(f"{Path(pair[0]).name} | {format_metrics(level_a)}")
                with col_b:
                    st.markdown("**B**")
                    png_b = pair[1].replace(".npy", ".png")
                    if Path(png_b).exists():
                        st.image(png_b, width="stretch")
                    else:
                        st.markdown(render_level_html(level_b), unsafe_allow_html=True)
                    st.caption(f"{Path(pair[1]).name} | {format_metrics(level_b)}")

                col_p1, col_p2, col_p3 = st.columns(3)
                preference = None
                if col_p1.button("Prefer A", type="primary", use_container_width=True):
                    preference = 0.0
                if col_p2.button("Tie", use_container_width=True):
                    preference = 0.5
                if col_p3.button("Prefer B", type="primary", use_container_width=True):
                    preference = 1.0

                if preference is not None:
                    total = save_dashboard_preference(
                        feedback_game,
                        level_a,
                        level_b,
                        preference,
                        {
                            "game": feedback_game,
                            "type": "dashboard_interactive",
                            "source_a": Path(pair[0])
                            .relative_to(PROJECT_ROOT)
                            .as_posix(),
                            "source_b": Path(pair[1])
                            .relative_to(PROJECT_ROOT)
                            .as_posix(),
                        },
                    )
                    st.session_state.rlhf_pair = None
                    st.session_state.rlhf_pair_source = None
                    st.success(f"Saved preference #{total}.")
                    st.rerun()

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

# ── Auto-refresh / Rerun while running ────────────────────────────────────────
if (
    st.session_state.train_status == "running"
    or st.session_state.infer_status == "running"
):
    time.sleep(1.0)
    st.rerun()
