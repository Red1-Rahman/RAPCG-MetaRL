"""
Action-Penalty Correlation Analysis
Analyze which agent actions cause the highest resource penalties.
"""

import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


def load_training_log(log_file: str) -> pd.DataFrame:
    """Load training log CSV file."""
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    df = pd.DataFrame(pd.read_csv(log_file))
    print(f"✓ Loaded {len(df)} steps from {log_file}")
    return df


def analyze_action_correlations(
    df: pd.DataFrame,
    penalty_type: str = "penalty_ram_penalty",
    min_action_count: int = 10,
) -> pd.DataFrame:
    """
    Analyze correlation between actions and penalties.

    Args:
        df: Training log dataframe
        penalty_type: Column name for penalty to analyze
        min_action_count: Minimum times an action must appear

    Returns:
        DataFrame with action statistics
    """
    if "action" not in df.columns:
        print("⚠ No action data in log file")
        return pd.DataFrame()

    if penalty_type not in df.columns:
        print(f"⚠ Penalty column '{penalty_type}' not found")
        print(f"Available columns: {df.columns.tolist()}")
        return pd.DataFrame()

    # Group by action
    action_stats = (
        df.groupby("action")
        .agg(
            {
                penalty_type: ["mean", "std", "max", "count"],
                "ram_percent": ["mean", "std"],
                "cpu_percent": ["mean", "std"],
            }
        )
        .reset_index()
    )

    # Flatten column names
    action_stats.columns = [
        "_".join(col).strip("_") if col[1] else col[0]
        for col in action_stats.columns.values
    ]

    # Filter by minimum count
    action_stats = action_stats[
        action_stats[f"{penalty_type}_count"] >= min_action_count
    ]

    # Sort by mean penalty (descending)
    action_stats = action_stats.sort_values(f"{penalty_type}_mean", ascending=False)

    return action_stats


def print_action_analysis(
    action_stats: pd.DataFrame, penalty_type: str, top_n: int = 10
):
    """Print action-penalty analysis results."""
    if action_stats.empty:
        print("No action statistics available")
        return

    penalty_mean_col = f"{penalty_type}_mean"
    penalty_count_col = f"{penalty_type}_count"
    ram_mean_col = "ram_percent_mean"

    print("\n" + "=" * 80)
    print("ACTION-PENALTY CORRELATION ANALYSIS")
    print("=" * 80)

    print(f"\nTop {top_n} Most Resource-Intensive Actions:")
    print("-" * 80)
    print(
        f"{'Action':>8} {'Count':>8} {'Avg Penalty':>12} {'Max Penalty':>12} {'Avg RAM %':>10}"
    )
    print("-" * 80)

    for idx, row in action_stats.head(top_n).iterrows():
        action = int(row["action"])
        count = int(row[penalty_count_col])
        avg_penalty = row[penalty_mean_col]
        max_penalty = row[f"{penalty_type}_max"]
        avg_ram = row[ram_mean_col]

        print(
            f"{action:8d} {count:8d} {avg_penalty:12.4f} {max_penalty:12.4f} {avg_ram:10.2f}"
        )

    print("-" * 80)

    # Summary statistics
    print(f"\nSummary Statistics:")
    print(f"  Total unique actions: {len(action_stats)}")
    print(f"  Highest avg penalty: {action_stats[penalty_mean_col].max():.4f}")
    print(f"  Lowest avg penalty: {action_stats[penalty_mean_col].min():.4f}")
    print(
        f"  Mean penalty across all actions: {action_stats[penalty_mean_col].mean():.4f}"
    )
    print("=" * 80 + "\n")


def plot_action_heatmap(
    df: pd.DataFrame, output_file: str = "action_penalty_heatmap.png"
):
    """Create heatmap of action vs penalty over time."""
    if "action" not in df.columns or "penalty_ram_penalty" not in df.columns:
        print("⚠ Cannot create heatmap: missing columns")
        return

    # Sample data if too large
    if len(df) > 5000:
        df_sample = df.sample(n=5000, random_state=42).sort_values("step")
    else:
        df_sample = df.sort_values("step")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Plot 1: Actions over time
    ax1 = axes[0]
    ax1.scatter(df_sample["step"], df_sample["action"], alpha=0.3, s=1)
    ax1.set_ylabel("Action")
    ax1.set_title("Actions Taken Over Training")
    ax1.grid(True, alpha=0.3)

    # Plot 2: RAM penalty over time
    ax2 = axes[1]
    ax2.plot(
        df_sample["step"], df_sample["penalty_ram_penalty"], alpha=0.6, linewidth=0.5
    )
    ax2.set_ylabel("RAM Penalty")
    ax2.set_title("RAM Penalty Over Training")
    ax2.grid(True, alpha=0.3)

    # Plot 3: RAM usage over time
    ax3 = axes[2]
    ax3.plot(
        df_sample["step"],
        df_sample["ram_percent"],
        alpha=0.6,
        linewidth=0.5,
        color="red",
    )
    ax3.axhline(y=60, color="orange", linestyle="--", label="Penalty Threshold (60%)")
    ax3.set_ylabel("RAM Usage (%)")
    ax3.set_xlabel("Training Step")
    ax3.set_title("RAM Usage Over Training")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Heatmap saved to {output_file}")
    plt.close()


def plot_action_distribution(
    action_stats: pd.DataFrame,
    penalty_type: str = "penalty_ram_penalty",
    output_file: str = "action_distribution.png",
):
    """Plot distribution of penalties by action."""
    if action_stats.empty:
        print("⚠ Cannot create distribution plot: no data")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    penalty_mean_col = f"{penalty_type}_mean"
    penalty_count_col = f"{penalty_type}_count"

    # Plot 1: Average penalty by action
    ax1 = axes[0]
    actions = action_stats["action"].values
    penalties = action_stats[penalty_mean_col].values
    colors = ["red" if p > 3.0 else "orange" if p > 1.0 else "green" for p in penalties]

    ax1.bar(range(len(actions)), penalties, color=colors, alpha=0.7)
    ax1.set_ylabel("Average RAM Penalty")
    ax1.set_title("Average RAM Penalty by Action")
    ax1.set_xticks(range(len(actions)))
    ax1.set_xticklabels([int(a) for a in actions], rotation=45)
    ax1.axhline(y=3.0, color="red", linestyle="--", alpha=0.5, label="High (3.0)")
    ax1.axhline(y=1.0, color="orange", linestyle="--", alpha=0.5, label="Medium (1.0)")
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Plot 2: Action frequency
    ax2 = axes[1]
    counts = action_stats[penalty_count_col].values
    ax2.bar(range(len(actions)), counts, alpha=0.7, color="steelblue")
    ax2.set_ylabel("Frequency (count)")
    ax2.set_xlabel("Action")
    ax2.set_title("Action Frequency")
    ax2.set_xticks(range(len(actions)))
    ax2.set_xticklabels([int(a) for a in actions], rotation=45)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"✓ Distribution plot saved to {output_file}")
    plt.close()


def main():
    """Main analysis function."""
    import argparse

    parser = argparse.ArgumentParser(description="Analyze action-penalty correlations")
    parser.add_argument("log_file", help="Path to training log CSV file")
    parser.add_argument(
        "--penalty",
        default="penalty_ram_penalty",
        help="Penalty column to analyze (default: penalty_ram_penalty)",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top actions to show (default: 10)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=10,
        help="Minimum action count to include (default: 10)",
    )
    parser.add_argument(
        "--plot", action="store_true", help="Generate visualization plots"
    )

    args = parser.parse_args()

    # Load data
    try:
        df = load_training_log(args.log_file)
    except Exception as e:
        print(f"✗ Error loading log file: {e}")
        return 1

    # Analyze correlations
    action_stats = analyze_action_correlations(
        df, penalty_type=args.penalty, min_action_count=args.min_count
    )

    # Print results
    print_action_analysis(action_stats, args.penalty, top_n=args.top_n)

    # Generate plots if requested
    if args.plot:
        try:
            plot_action_heatmap(df)
            plot_action_distribution(action_stats, penalty_type=args.penalty)
        except Exception as e:
            print(f"⚠ Error generating plots: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
