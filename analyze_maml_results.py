"""
analyze_maml_results.py
=======================
Publication-quality analysis and plotting for RAPCG-MetaRL MAML training logs.
Targets ACM Transactions on Graphics figure standards:
  - 300 DPI minimum
  - vector-safe fonts (no Type 3)
  - column-width and double-column figure sizes
  - consistent color palette across all figures

Usage:
    python analyze_maml_results.py --log logs/sokoban_MAML_inference.csv
    python analyze_maml_results.py --log logs/sokoban_MAML_inference.csv --out figures/
    python analyze_maml_results.py --log logs/sokoban_MAML_inference.csv --compare logs/zelda_MAML_run2.csv

Output (all in --out directory, default: figures/maml/):
    fig1_meta_loss_convergence.pdf/.png
    fig2_resource_usage.pdf/.png
    fig3_reward_distribution.pdf/.png
    fig4_penalty_breakdown.pdf/.png
    fig5_training_summary_dashboard.pdf/.png
    maml_summary_stats.json
"""

import os
import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore", category=UserWarning)

# ── ACM TOG figure standards ──────────────────────────────────────────────────
matplotlib.rcParams.update(
    {
        # Fonts — Type 1 / TrueType only, no Type 3
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.title_fontsize": 8,
        # Lines & markers
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.5,
        "ytick.minor.width": 0.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        # Grid
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        # Layout
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# ── ACM column widths (inches) ────────────────────────────────────────────────
COL_SINGLE = 3.33  # single column
COL_DOUBLE = 7.00  # double column (full page width)
COL_HEIGHT = 2.2  # standard panel height

# ── Color palette — dark terminal aesthetic matching the project ───────────────
C = {
    "loss": "#58a6ff",  # blue     — meta-loss
    "smooth": "#1f6feb",  # dark blue — smoothed loss
    "reward": "#7ee787",  # green    — reward proxy
    "ram": "#f85149",  # red      — RAM
    "cpu": "#ffa657",  # orange   — CPU
    "gpu": "#d2a8ff",  # purple   — GPU
    "ram_pen": "#ff7b72",  # salmon   — RAM penalty
    "cpu_pen": "#ffa657",  # orange   — CPU penalty
    "gpu_pen": "#79c0ff",  # light blue — GPU penalty
    "fill": "#161b22",  # dark bg fill
    "grid": "#30363d",  # grid lines
    "threshold": "#e3b341",  # yellow   — threshold lines
    "compare": "#f0883e",  # orange   — comparison run
}

SAVE_EXTS = [".pdf", ".png"]


# ── I/O helpers ───────────────────────────────────────────────────────────────


def load_log(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names — TrainingLogger may prefix content_ keys
    rename = {}
    for col in df.columns:
        if col == "content_meta_loss":
            rename[col] = "meta_loss"
        if col == "content_iteration":
            rename[col] = "iteration"
    df = df.rename(columns=rename)

    # Derive meta_loss from reward proxy if not already present
    if "meta_loss" not in df.columns and "reward" in df.columns:
        df["meta_loss"] = -df["reward"]

    # Derive iteration from step if not present
    if "iteration" not in df.columns and "step" in df.columns:
        df["iteration"] = df["step"] + 1

    # Smoothed loss (EMA)
    df["meta_loss_smooth"] = df["meta_loss"].ewm(span=20, adjust=False).mean()

    # Rolling stats
    df["loss_roll_mean"] = df["meta_loss"].rolling(window=20, min_periods=1).mean()
    df["loss_roll_std"] = (
        df["meta_loss"].rolling(window=20, min_periods=1).std().fillna(0)
    )

    return df


def save_fig(fig: plt.Figure, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in SAVE_EXTS:
        fpath = out_dir / (name + ext)
        fig.savefig(fpath)
        print(f"  [OK] {fpath}")
    plt.close(fig)


def label_run(path: str) -> str:
    return Path(path).stem.replace("_", " ")


# ── Figure 1: Meta-loss convergence ───────────────────────────────────────────


def fig_meta_loss_convergence(
    df: pd.DataFrame,
    out_dir: Path,
    compare_df: pd.DataFrame = None,
    compare_label: str = None,
):
    """
    Single-column figure.
    Top panel: raw + smoothed meta-loss with ±1 std band.
    Bottom panel: first-difference (loss delta per iteration) to show stability.
    """
    fig, (ax_main, ax_delta) = plt.subplots(
        2,
        1,
        figsize=(COL_SINGLE, COL_HEIGHT * 2),
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
        sharex=True,
    )

    iters = df["iteration"].values
    loss = df["meta_loss"].values
    smooth = df["meta_loss_smooth"].values
    lo = df["loss_roll_mean"].values - df["loss_roll_std"].values
    hi = df["loss_roll_mean"].values + df["loss_roll_std"].values

    # ── Main panel ────────────────────────────────────────────────────────────
    ax_main.fill_between(iters, lo, hi, alpha=0.15, color=C["loss"], linewidth=0)
    ax_main.plot(
        iters, loss, color=C["loss"], alpha=0.35, linewidth=0.8, label="Raw loss"
    )
    ax_main.plot(iters, smooth, color=C["smooth"], linewidth=1.8, label="EMA (span=20)")

    if compare_df is not None:
        c_iters = compare_df["iteration"].values
        c_smooth = compare_df["meta_loss_smooth"].values
        ax_main.plot(
            c_iters,
            c_smooth,
            color=C["compare"],
            linewidth=1.5,
            linestyle="--",
            label=compare_label or "Comparison",
        )

    # Best loss annotation
    best_iter = df.loc[df["meta_loss"].idxmin(), "iteration"]
    best_loss = df["meta_loss"].min()
    ax_main.axvline(
        best_iter, color=C["threshold"], linewidth=0.9, linestyle=":", alpha=0.8
    )
    y_top = ax_main.get_ylim()[1] if ax_main.get_ylim()[1] != 1.0 else smooth.max()
    ax_main.annotate(
        f"Best: {best_loss:.4f}\n(iter {best_iter})",
        xy=(best_iter, best_loss),
        xytext=(min(best_iter + len(iters) * 0.12, iters[-1] * 0.85),
                smooth.max() * 0.55),
        fontsize=7,
        color=C["threshold"],
        arrowprops=dict(arrowstyle="->", color=C["threshold"], lw=0.8,
                        connectionstyle="arc3,rad=0.2"),
    )

    ax_main.set_ylabel("Meta-Loss $\\mathcal{L}_{\\mathrm{meta}}$")
    ax_main.legend(loc="upper right", framealpha=0.9, edgecolor="#30363d")
    ax_main.yaxis.set_major_locator(MaxNLocator(5, prune="both"))

    # ── Delta panel ───────────────────────────────────────────────────────────
    delta = np.diff(smooth, prepend=smooth[0])
    ax_delta.axhline(0, color="#484f58", linewidth=0.7, linestyle="--")
    ax_delta.fill_between(
        iters,
        delta,
        0,
        where=(delta < 0),
        color=C["reward"],
        alpha=0.45,
        label="Decreasing",
    )
    ax_delta.fill_between(
        iters,
        delta,
        0,
        where=(delta >= 0),
        color=C["ram"],
        alpha=0.35,
        label="Increasing",
    )
    ax_delta.set_ylabel("$\\Delta$ Loss")
    ax_delta.set_xlabel("Meta-Training Iteration")
    ax_delta.yaxis.set_major_locator(MaxNLocator(3, prune="both"))
    ax_delta.legend(
        loc="upper right", framealpha=0.9, edgecolor="#30363d", ncol=2, fontsize=7
    )

    fig.suptitle("MAML Meta-Loss Convergence (Sokoban, Narrow)", fontsize=10, y=1.01)

    save_fig(fig, out_dir, "fig1_meta_loss_convergence")


# ── Figure 2: Resource usage over training ────────────────────────────────────


def fig_resource_usage(df: pd.DataFrame, out_dir: Path):
    """
    Double-column figure: CPU / RAM / GPU over training iterations.
    Threshold lines matching the reward shaping formula in pcgrl_env.py.
    """
    resource_cols = {
        "cpu_percent": ("CPU Usage (%)", C["cpu"], 70.0),
        "ram_percent": ("RAM Usage (%)", C["ram"], 78.0),
        "gpu_mem_percent": ("GPU VRAM (%)", C["gpu"], 70.0),
    }
    available = {k: v for k, v in resource_cols.items() if k in df.columns}
    if not available:
        print("  [SKIP] No resource columns found in log.")
        return

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(COL_DOUBLE, COL_HEIGHT), sharey=False)
    if n == 1:
        axes = [axes]

    iters = df["iteration"].values

    for ax, (col, (ylabel, color, threshold)) in zip(axes, available.items()):
        vals = df[col].values
        smooth = pd.Series(vals).ewm(span=30, adjust=False).mean().values

        ax.fill_between(iters, vals, alpha=0.12, color=color, linewidth=0)
        ax.plot(iters, vals, color=color, alpha=0.25, linewidth=0.7)
        ax.plot(iters, smooth, color=color, linewidth=1.6, label="EMA")

        # Penalty threshold line
        ax.axhline(
            threshold,
            color=C["threshold"],
            linewidth=0.9,
            linestyle="--",
            alpha=0.85,
            label=f"Penalty threshold ({threshold:.0f}%)",
        )

        # Shade above threshold
        ax.fill_between(
            iters,
            smooth,
            threshold,
            where=(smooth > threshold),
            color=C["threshold"],
            alpha=0.12,
            linewidth=0,
        )

        # Stats box
        mean_v = np.mean(vals)
        max_v = np.max(vals)
        pct_over = 100 * np.mean(vals > threshold)
        textstr = (
            f"$\\mu$={mean_v:.1f}%\n"
            f"max={max_v:.1f}%\n"
            f">{threshold:.0f}%: {pct_over:.1f}%"
        )
        ax.text(
            0.97,
            0.97,
            textstr,
            transform=ax.transAxes,
            fontsize=7,
            va="top",
            ha="right",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                alpha=0.85,
                edgecolor="#30363d",
            ),
        )

        ax.set_xlabel("Meta-Training Iteration")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 105)
        ax.legend(loc="upper left", framealpha=0.9, edgecolor="#30363d", fontsize=7)

    fig.suptitle(
        "Hardware Resource Utilization During MAML Training", fontsize=10, y=1.02
    )
    fig.tight_layout()
    save_fig(fig, out_dir, "fig2_resource_usage")


# ── Figure 3: Reward proxy distribution ───────────────────────────────────────


def fig_reward_distribution(df: pd.DataFrame, out_dir: Path):
    """
    Single-column figure: violin + box plot of reward (−meta_loss) across
    training quartiles to show how the distribution shifts.
    """
    if "reward" not in df.columns:
        print("  [SKIP] No reward column.")
        return

    fig, (ax_violin, ax_trend) = plt.subplots(
        1,
        2,
        figsize=(COL_DOUBLE * 0.7, COL_HEIGHT),
        gridspec_kw={"width_ratios": [2, 3]},
    )

    # Split into 4 quartile windows
    n = len(df)
    quartile_size = n // 4
    quartiles = []
    labels = []
    for q in range(4):
        start = q * quartile_size
        end = (q + 1) * quartile_size if q < 3 else n
        chunk = df["reward"].iloc[start:end].dropna().values
        quartiles.append(chunk)
        pct = int((start + end) / (2 * n) * 100)
        #labels.append(f"Q{q + 1}\n({start}–{end})")
        labels.append(f"Q{q + 1}\n[{start}–{end}]")

    # Violin
    parts = ax_violin.violinplot(
        quartiles, positions=range(1, 5), showmedians=True, showextrema=False
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(C["loss"])
        pc.set_alpha(0.55)
    parts["cmedians"].set_color(C["smooth"])
    parts["cmedians"].set_linewidth(1.5)

    # Overlay box plot
    bp = ax_violin.boxplot(
        quartiles,
        positions=range(1, 5),
        widths=0.18,
        patch_artist=True,
        medianprops=dict(color=C["smooth"], linewidth=1.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        flierprops=dict(marker=".", markersize=2, alpha=0.4),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("white")
        patch.set_alpha(0.7)

    ax_violin.set_xticks(range(1, 5))
    ax_violin.set_xticklabels(labels, fontsize=6.5, linespacing=1.3)
    ax_violin.tick_params(axis="x", pad=4)
    ax_violin.set_ylabel("Reward Proxy ($-\\mathcal{L}_{\\mathrm{meta}}$)")
    ax_violin.set_xlabel("Training Quartile")
    ax_violin.set_title("Reward Distribution by Quartile", fontsize=9)

    # Trend: cumulative mean reward
    cum_mean = df["reward"].expanding().mean()
    iters = df["iteration"].values
    ax_trend.plot(
        iters,
        df["reward"].values,
        color=C["reward"],
        alpha=0.2,
        linewidth=0.6,
        label="Raw",
    )
    ax_trend.plot(
        iters,
        cum_mean.values,
        color=C["smooth"],
        linewidth=1.6,
        label="Cumulative mean",
    )
    ax_trend.axhline(0, color="#484f58", linewidth=0.7, linestyle="--")
    ax_trend.set_xlabel("Meta-Training Iteration")
    ax_trend.set_ylabel("Reward Proxy")
    ax_trend.set_title("Cumulative Mean Reward Trend", fontsize=9)
    ax_trend.legend(framealpha=0.9, edgecolor="#30363d", fontsize=7)

    fig.suptitle("Meta-Training Reward Proxy Analysis", fontsize=10, y=1.08)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, out_dir, "fig3_reward_distribution")


# ── Figure 4: Penalty breakdown ───────────────────────────────────────────────


def fig_penalty_breakdown(df: pd.DataFrame, out_dir: Path):
    """
    Stacked area chart of RAM / CPU / GPU penalties over training,
    plus a pie chart of mean penalty composition.
    """
    pen_cols = {
        "penalty_ram_penalty": ("RAM Penalty", C["ram_pen"]),
        "penalty_cpu_penalty": ("CPU Penalty", C["cpu_pen"]),
        "penalty_gpu_penalty": ("GPU Penalty", C["gpu_pen"]),
    }
    available = {k: v for k, v in pen_cols.items() if k in df.columns}

    if not available:
        print("  [SKIP] No penalty columns found in log.")
        return

    fig, (ax_stack, ax_pie) = plt.subplots(
        1,
        2,
        figsize=(COL_DOUBLE * 0.8, COL_HEIGHT),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    iters = df["iteration"].values
    labels = [v[0] for v in available.values()]
    colors = [v[1] for v in available.values()]
    data = np.array([df[k].fillna(0).values for k in available])

    # Smooth each penalty series
    smoothed = np.array(
        [pd.Series(d).ewm(span=20, adjust=False).mean().values for d in data]
    )

    ax_stack.stackplot(iters, smoothed, labels=labels, colors=colors, alpha=0.75)
    ax_stack.set_xlabel("Meta-Training Iteration")
    ax_stack.set_ylabel("Cumulative Penalty")
    ax_stack.set_title("Resource Penalty Breakdown (Stacked)", fontsize=9)
    ax_stack.legend(loc="upper right", framealpha=0.9, edgecolor="#30363d", fontsize=7)

    # Pie: mean composition
    means = [d.mean() for d in data]
    total = sum(means)
    if total > 0:
        wedges, texts, autotexts = ax_pie.pie(
            means,
            labels=labels,
            colors=colors,
            autopct="%1.1f%%",
            startangle=90,
            textprops={"fontsize": 7},
            wedgeprops={"linewidth": 0.5, "edgecolor": "white"},
        )
        for at in autotexts:
            at.set_fontsize(6.5)
        ax_pie.set_title("Mean Penalty\nComposition", fontsize=9)
    else:
        ax_pie.text(
            0.5,
            0.5,
            "No penalties\nrecorded",
            ha="center",
            va="center",
            transform=ax_pie.transAxes,
            fontsize=9,
            color="#8b949e",
        )
        ax_pie.axis("off")

    fig.suptitle("Resource-Aware Reward Penalty Analysis", fontsize=10, y=1.02)
    fig.tight_layout()
    save_fig(fig, out_dir, "fig4_penalty_breakdown")


# ── Figure 5: Summary dashboard (double-column) ───────────────────────────────


def fig_summary_dashboard(df: pd.DataFrame, out_dir: Path, run_name: str):
    """
    Full double-column summary figure combining all key metrics.
    Designed as the primary figure for the paper's results section.
    """
    fig = plt.figure(figsize=(COL_DOUBLE, COL_HEIGHT * 3.2))
    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        hspace=0.55,
        wspace=0.42,
    )

    ax_loss = fig.add_subplot(gs[0, :2])  # top-left wide: loss curve
    ax_hist = fig.add_subplot(gs[0, 2])  # top-right: loss histogram
    ax_cpu = fig.add_subplot(gs[1, 0])  # middle row: CPU
    ax_ram = fig.add_subplot(gs[1, 1])  # middle row: RAM
    ax_gpu = fig.add_subplot(gs[1, 2])  # middle row: GPU
    ax_stats = fig.add_subplot(gs[2, :])  # bottom: stats table

    iters = df["iteration"].values
    loss = df["meta_loss"].values
    smooth = df["meta_loss_smooth"].values

    # ── Loss curve ────────────────────────────────────────────────────────────
    lo = df["loss_roll_mean"].values - df["loss_roll_std"].values
    hi = df["loss_roll_mean"].values + df["loss_roll_std"].values
    ax_loss.fill_between(iters, lo, hi, alpha=0.15, color=C["loss"], linewidth=0)
    ax_loss.plot(iters, loss, color=C["loss"], alpha=0.3, linewidth=0.7)
    ax_loss.plot(iters, smooth, color=C["smooth"], linewidth=1.6, label="EMA")
    ax_loss.axhline(
        loss.min(),
        color=C["threshold"],
        linewidth=0.8,
        linestyle=":",
        alpha=0.7,
        label=f"Best: {loss.min():.4f}",
    )
    ax_loss.set_xlabel("Iteration")
    ax_loss.set_ylabel("Meta-Loss")
    ax_loss.set_title("Meta-Loss Convergence", fontsize=9, pad=3)
    ax_loss.legend(fontsize=7, framealpha=0.9, edgecolor="#30363d")

    # ── Loss histogram ────────────────────────────────────────────────────────
    ax_hist.hist(
        loss,
        bins=30,
        color=C["loss"],
        alpha=0.75,
        edgecolor="white",
        linewidth=0.3,
        density=True,
    )
    ax_hist.axvline(
        loss.mean(),
        color=C["smooth"],
        linewidth=1.2,
        linestyle="--",
        label=f"Mean: {loss.mean():.3f}",
    )
    ax_hist.axvline(
        np.median(loss),
        color=C["reward"],
        linewidth=1.2,
        linestyle=":",
        label=f"Median: {np.median(loss):.3f}",
    )
    ax_hist.set_xlabel("Meta-Loss")
    ax_hist.set_ylabel("Density")
    ax_hist.set_title("Loss Distribution", fontsize=9, pad=3)
    ax_hist.legend(fontsize=6.5, framealpha=0.9, edgecolor="#30363d")

    # ── Resource panels ───────────────────────────────────────────────────────
    res_map = [
        ("cpu_percent", ax_cpu, "CPU (%)", C["cpu"], 70.0),
        ("ram_percent", ax_ram, "RAM (%)", C["ram"], 78.0),
        ("gpu_mem_percent", ax_gpu, "GPU VRAM (%)", C["gpu"], 70.0),
    ]
    for col, ax, ylabel, color, thr in res_map:
        if col in df.columns:
            vals = df[col].values
            sm = pd.Series(vals).ewm(span=30, adjust=False).mean().values
            ax.fill_between(iters, vals, alpha=0.12, color=color, linewidth=0)
            ax.plot(iters, sm, color=color, linewidth=1.4)
            ax.axhline(
                thr, color=C["threshold"], linewidth=0.8, linestyle="--", alpha=0.8
            )
            ax.set_ylim(0, 105)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("Iteration")
            ax.set_title(ylabel.replace(" (%)", ""), fontsize=9, pad=3)
            pct_over = 100 * np.mean(vals > thr)
            ax.text(
                0.97,
                0.05,
                f">{thr:.0f}%: {pct_over:.1f}%",
                transform=ax.transAxes,
                fontsize=6.5,
                ha="right",
                color=C["threshold"],
            )
        else:
            ax.text(
                0.5,
                0.5,
                "N/A",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color="#8b949e",
            )
            ax.set_title(ylabel, fontsize=9, pad=3)
            ax.axis("off")

    # ── Stats table — split into two rows to avoid column crowding ────────────
    ax_stats.axis("off")
    stats = compute_summary_stats(df)
    items = list(stats.items())
    mid = len(items) // 2
    row1_keys = [k for k, _ in items[:mid]]
    row1_vals = [v for _, v in items[:mid]]
    row2_keys = [k for k, _ in items[mid:]]
    row2_vals = [v for _, v in items[mid:]]
    # Pad shorter row to equal length
    while len(row2_keys) < len(row1_keys):
        row2_keys.append(""); row2_vals.append("")

    col_labels = row1_keys
    cell_vals  = [row1_vals, row2_keys, row2_vals]

    tbl = ax_stats.table(
        cellText=cell_vals,
        colLabels=col_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.0)
    tbl.scale(1, 1.5)

    n_cols = len(col_labels)
    # Header row (row 0)
    for j in range(n_cols):
        tbl[(0, j)].set_facecolor("#1f3a5f")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    # Data rows
    for j in range(n_cols):
        tbl[(1, j)].set_facecolor("#e8f0fe")   # row1 values
        tbl[(2, j)].set_facecolor("#d0ddf7")   # row2 keys (act as sub-headers)
        tbl[(2, j)].set_text_props(fontweight="bold", color="#1a2a4a")
        tbl[(3, j)].set_facecolor("#e8f0fe")   # row2 values

    ax_stats.set_title("Training Summary Statistics", fontsize=9, pad=6, loc="left")

    fig.suptitle(
        f"RAPCG-MetaRL: MAML Training Results — {run_name}",
        fontsize=10,
        fontweight="bold",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_fig(fig, out_dir, "fig5_training_summary_dashboard")


# ── Figure 6: Phase analysis (early / mid / late convergence) ─────────────────


def fig_phase_analysis(df: pd.DataFrame, out_dir: Path):
    """
    Double-column figure: loss curve divided into 3 training phases,
    with a zoomed inset on the final convergence phase.
    """
    n = len(df)
    phase = n // 3
    phase_bounds = [
        (0, phase, "Early", C["ram"]),
        (phase, 2 * phase, "Mid", C["cpu"]),
        (2 * phase, n, "Late", C["reward"]),
    ]

    fig, ax = plt.subplots(figsize=(COL_DOUBLE * 0.75, COL_HEIGHT * 1.3))

    iters = df["iteration"].values
    smooth = df["meta_loss_smooth"].values

    ax.plot(iters, df["meta_loss"].values, color=C["loss"], alpha=0.2, linewidth=0.7)
    ax.plot(iters, smooth, color=C["smooth"], linewidth=1.5, zorder=5)

    patch_handles = []
    for start, end, label, color in phase_bounds:
        ax.axvspan(
            iters[start], iters[min(end, n - 1)], alpha=0.08, color=color, linewidth=0
        )
        ax.axvline(iters[start], color=color, linewidth=0.7, linestyle="--", alpha=0.6)

        phase_data = df["meta_loss"].iloc[start:end]
        mid_iter = iters[(start + min(end, n - 1)) // 2]
        y_label = smooth.max() * (0.92 - 0.06 * ["Early","Mid","Late"].index(label))
        ax.text(
            mid_iter,
            y_label,
            label,
            ha="center",
            va="top",
            fontsize=7.5,
            color=color,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.7, edgecolor=color, linewidth=0.6),
        )

        patch_handles.append(
            mpatches.Patch(
                color=color,
                alpha=0.4,
                label=f"{label}: μ={phase_data.mean():.3f}, σ={phase_data.std():.3f}",
            )
        )

    # Zoomed inset — last 20% of training
    zoom_start = int(0.8 * n)
    #ax_ins = ax.inset_axes([0.55, 0.45, 0.42, 0.48])
    ax_ins = ax.inset_axes([0.52, 0.38, 0.44, 0.50])
    ax_ins.plot(
        iters[zoom_start:], smooth[zoom_start:], color=C["smooth"], linewidth=1.2
    )
    ax_ins.fill_between(
        iters[zoom_start:],
        df["loss_roll_mean"].values[zoom_start:]
        - df["loss_roll_std"].values[zoom_start:],
        df["loss_roll_mean"].values[zoom_start:]
        + df["loss_roll_std"].values[zoom_start:],
        alpha=0.2,
        color=C["loss"],
        linewidth=0,
    )
    ax_ins.set_title("Final 20%", fontsize=7, pad=2)
    ax_ins.tick_params(labelsize=6)
    ax_ins.yaxis.set_major_locator(MaxNLocator(3))
    ax_ins.xaxis.set_major_locator(MaxNLocator(3))
    ax.indicate_inset_zoom(ax_ins, edgecolor="#30363d", linewidth=0.8)

    ax.set_xlabel("Meta-Training Iteration")
    ax.set_ylabel("Meta-Loss $\\mathcal{L}_{\\mathrm{meta}}$")
    ax.set_title("Training Phase Analysis with Convergence Zoom", fontsize=9)
    ax.legend(
        handles=patch_handles,
        fontsize=7,
        framealpha=0.9,
        edgecolor="#30363d",
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
    )

    fig.tight_layout()
    save_fig(fig, out_dir, "fig6_phase_analysis")


# ── Summary statistics ────────────────────────────────────────────────────────


def compute_summary_stats(df: pd.DataFrame) -> dict:
    loss = df["meta_loss"].values
    n = len(df)

    stats = {
        "Iterations": str(n),
        "Best Loss": f"{loss.min():.4f}",
        #"Final Loss": f"{loss.iloc[-1]:.4f}",
        "Final Loss": f"{float(loss[-1]):.4f}",
        "Mean Loss": f"{loss.mean():.4f}",
        "Loss Std": f"{loss.std():.4f}",
        "Convergence Iter": str(int(df.loc[df["meta_loss"].idxmin(), "iteration"])),
        "Improvement": f"{((loss[0] - loss.min()) / (abs(loss[0]) + 1e-8)) * 100:.1f}%",
    }

    for col, label in [
        ("cpu_percent", "Avg CPU %"),
        ("ram_percent", "Avg RAM %"),
        ("gpu_mem_percent", "Avg GPU %"),
    ]:
        if col in df.columns:
            stats[label] = f"{df[col].mean():.1f}%"

    return stats


def save_summary_json(df: pd.DataFrame, out_dir: Path, run_name: str):
    stats = compute_summary_stats(df)
    stats["run_name"] = run_name
    stats["log_rows"] = len(df)
    stats["columns"] = list(df.columns)

    path = out_dir / "maml_summary_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  [OK] {path}")
    return stats


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Publication-quality MAML results analysis for RAPCG-MetaRL"
    )
    parser.add_argument(
        "--log",
        required=True,
        help="Path to training log CSV (e.g. logs/sokoban_MAML_inference.csv)",
    )
    parser.add_argument(
        "--compare", default=None, help="Optional second log CSV for comparison overlay"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for figures (default: figures/maml/<run_name>/)",
    )
    parser.add_argument(
        "--no-show", action="store_true", help="Do not display figures interactively"
    )
    args = parser.parse_args()

    # Load
    print(f"\nLoading: {args.log}")
    df = load_log(args.log)
    run_name = Path(args.log).stem
    out_dir = Path(args.out) if args.out else Path("figures") / "maml" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    compare_df = None
    compare_label = None
    if args.compare:
        print(f"Loading comparison: {args.compare}")
        compare_df = load_log(args.compare)
        compare_label = label_run(args.compare)

    print(f"\nDataset: {len(df)} rows, columns: {list(df.columns)}")
    print(f"Output : {out_dir}/\n")

    # Stats
    stats = save_summary_json(df, out_dir, run_name)
    print("\n── Summary Statistics ──────────────────────────────────────")
    for k, v in stats.items():
        if k not in ("run_name", "log_rows", "columns"):
            print(f"  {k:<22}: {v}")
    print()

    # Generate all figures
    print("── Generating Figures ──────────────────────────────────────")
    fig_meta_loss_convergence(df, out_dir, compare_df, compare_label)
    fig_resource_usage(df, out_dir)
    fig_reward_distribution(df, out_dir)
    fig_penalty_breakdown(df, out_dir)
    fig_summary_dashboard(df, out_dir, run_name)
    fig_phase_analysis(df, out_dir)

    print(f"\n[DONE] All figures saved to: {out_dir}/")
    print("       Formats: PDF (vector, ACM-safe) + PNG (300 DPI)\n")


if __name__ == "__main__":
    main()
