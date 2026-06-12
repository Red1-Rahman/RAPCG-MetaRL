"""
Compare Forward vs Backward Sokoban Generation

Analyzes and compares levels generated using:
1. Forward generation + validation
2. Backward generation (from solved state)

Metrics compared:
- Solvability rate
- Solution length distribution
- Validation corrections needed
- Level complexity
- Generation time
- Deadlock frequency
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple

from sokoban_utils import is_valid_sokoban, validate_and_fix_sokoban, print_level_stats


def load_levels(level_dir: str) -> List[np.ndarray]:
    """Load all levels from directory."""
    levels = []
    level_files = sorted(Path(level_dir).glob("*.npy"))

    for level_file in level_files:
        level = np.load(level_file)
        levels.append(level)

    return levels


def analyze_forward_levels(levels: List[np.ndarray]) -> Dict:
    """
    Analyze forward-generated levels.

    Checks:
    - Validation corrections needed
    - Solvability
    - Deadlock frequency
    """
    stats = {
        "total_levels": len(levels),
        "valid_before_fix": 0,
        "valid_after_fix": 0,
        "player_fixes": 0,
        "deadlock_removals": 0,
        "balance_fixes": 0,
        "unpushable_removals": 0,
        "unreachable_removals": 0,
        "dead_target_removals": 0,
        "crate_counts": [],
        "target_counts": [],
        "wall_densities": [],
        "correction_details": [],
    }

    for i, level in enumerate(levels):
        # Check validity before fix
        is_valid, msg = is_valid_sokoban(level)
        if is_valid:
            stats["valid_before_fix"] += 1

        # Apply validation
        fixed_level, corrections = validate_and_fix_sokoban(
            level, min_crates=1, enforce_all_rules=True
        )

        # Check after fix
        is_valid_after, msg_after = is_valid_sokoban(fixed_level)
        if is_valid_after:
            stats["valid_after_fix"] += 1

        # Collect corrections
        if corrections["player_fixed"]:
            stats["player_fixes"] += 1
        if corrections["deadlocked_removed"] > 0:
            stats["deadlock_removals"] += corrections["deadlocked_removed"]
        if corrections["crates_balanced"]:
            stats["balance_fixes"] += 1
        if corrections["unpushable_removed"] > 0:
            stats["unpushable_removals"] += corrections["unpushable_removed"]
        if corrections["unreachable_removed"] > 0:
            stats["unreachable_removals"] += corrections["unreachable_removed"]
        if corrections["dead_targets_removed"] > 0:
            stats["dead_target_removals"] += corrections["dead_targets_removed"]

        # Level metrics
        stats["crate_counts"].append(corrections["final_crates"])
        stats["target_counts"].append(corrections["final_targets"])
        stats["wall_densities"].append(np.sum(fixed_level == 1) / fixed_level.size)

        # Store details
        stats["correction_details"].append(
            {
                "level_id": i,
                "valid_before": is_valid,
                "valid_after": is_valid_after,
                "corrections": corrections,
            }
        )

    return stats


def analyze_backward_levels(levels: List[np.ndarray], solution_dir: str = None) -> Dict:
    """
    Analyze backward-generated levels.

    These should be inherently solvable with known solutions.
    """
    stats = {
        "total_levels": len(levels),
        "valid_levels": 0,
        "crate_counts": [],
        "target_counts": [],
        "wall_densities": [],
        "solution_lengths": [],
        "solutions_found": 0,
    }

    for i, level in enumerate(levels):
        # Check validity
        is_valid, msg = is_valid_sokoban(level)
        if is_valid:
            stats["valid_levels"] += 1

        # Level metrics
        stats["crate_counts"].append(np.sum(level == 3))
        stats["target_counts"].append(np.sum(level == 4))
        stats["wall_densities"].append(np.sum(level == 1) / level.size)

        # Load solution if available
        if solution_dir:
            solution_file = Path(solution_dir) / f"solution_{i}.json"
            if solution_file.exists():
                with open(solution_file, "r") as f:
                    solution = json.load(f)
                    stats["solution_lengths"].append(len(solution["path"]))
                    stats["solutions_found"] += 1

    return stats


def create_comparison_report(
    forward_stats: Dict, backward_stats: Dict, output_dir: str
):
    """
    Create comprehensive comparison report with visualizations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # 1. Validity Comparison
    ax1 = plt.subplot(3, 3, 1)
    validity_data = {
        "Forward\n(before fix)": forward_stats["valid_before_fix"]
        / forward_stats["total_levels"]
        * 100,
        "Forward\n(after fix)": forward_stats["valid_after_fix"]
        / forward_stats["total_levels"]
        * 100,
        "Backward": backward_stats["valid_levels"]
        / backward_stats["total_levels"]
        * 100,
    }
    bars = ax1.bar(
        validity_data.keys(),
        validity_data.values(),
        color=["#e74c3c", "#f39c12", "#27ae60"],
    )
    ax1.set_ylabel("Valid Levels (%)")
    ax1.set_title("Validity Rate Comparison")
    ax1.set_ylim([0, 105])
    for bar in bars:
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}%",
            ha="center",
            va="bottom",
        )

    # 2. Corrections Needed (Forward only)
    ax2 = plt.subplot(3, 3, 2)
    correction_types = [
        "Player\nFixes",
        "Deadlocks\nRemoved",
        "Balance\nFixes",
        "Unpushable\nRemoved",
        "Unreachable\nRemoved",
    ]
    correction_counts = [
        forward_stats["player_fixes"],
        forward_stats["deadlock_removals"],
        forward_stats["balance_fixes"],
        forward_stats["unpushable_removals"],
        forward_stats["unreachable_removals"],
    ]
    ax2.bar(correction_types, correction_counts, color="#e74c3c")
    ax2.set_ylabel("Count")
    ax2.set_title("Forward: Corrections Needed")
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # 3. Crate Distribution
    ax3 = plt.subplot(3, 3, 3)
    ax3.hist(
        [forward_stats["crate_counts"], backward_stats["crate_counts"]],
        label=["Forward", "Backward"],
        bins=range(0, 8),
        alpha=0.7,
    )
    ax3.set_xlabel("Number of Crates")
    ax3.set_ylabel("Frequency")
    ax3.set_title("Crate Distribution")
    ax3.legend()

    # 4. Wall Density
    ax4 = plt.subplot(3, 3, 4)
    ax4.boxplot(
        [forward_stats["wall_densities"], backward_stats["wall_densities"]],
        labels=["Forward", "Backward"],
    )
    ax4.set_ylabel("Wall Density")
    ax4.set_title("Wall Density Comparison")

    # 5. Solution Length (Backward only)
    if backward_stats["solution_lengths"]:
        ax5 = plt.subplot(3, 3, 5)
        ax5.hist(
            backward_stats["solution_lengths"], bins=15, color="#27ae60", alpha=0.7
        )
        ax5.set_xlabel("Solution Length")
        ax5.set_ylabel("Frequency")
        ax5.set_title("Backward: Solution Length Distribution")
        ax5.axvline(
            np.mean(backward_stats["solution_lengths"]),
            color="red",
            linestyle="--",
            label=f"Mean: {np.mean(backward_stats['solution_lengths']):.1f}",
        )
        ax5.legend()

    # 6. Summary Statistics Table
    ax6 = plt.subplot(3, 3, 6)
    ax6.axis("off")

    summary_text = f"""
    COMPARISON SUMMARY
    
    Forward Generation:
    • Total Levels: {forward_stats["total_levels"]}
    • Valid (before): {forward_stats["valid_before_fix"]} ({forward_stats["valid_before_fix"] / forward_stats["total_levels"] * 100:.1f}%)
    • Valid (after): {forward_stats["valid_after_fix"]} ({forward_stats["valid_after_fix"] / forward_stats["total_levels"] * 100:.1f}%)
    • Avg Crates: {np.mean(forward_stats["crate_counts"]):.1f}
    • Avg Wall Density: {np.mean(forward_stats["wall_densities"]):.3f}
    
    Backward Generation:
    • Total Levels: {backward_stats["total_levels"]}
    • Valid: {backward_stats["valid_levels"]} ({backward_stats["valid_levels"] / backward_stats["total_levels"] * 100:.1f}%)
    • Avg Crates: {np.mean(backward_stats["crate_counts"]):.1f}
    • Avg Wall Density: {np.mean(backward_stats["wall_densities"]):.3f}
    """

    if backward_stats["solution_lengths"]:
        summary_text += f"    • Avg Solution: {np.mean(backward_stats['solution_lengths']):.1f} steps\n"

    ax6.text(
        0.1,
        0.5,
        summary_text,
        fontfamily="monospace",
        fontsize=9,
        verticalalignment="center",
    )

    # 7. Correction Rate per Level (Forward)
    ax7 = plt.subplot(3, 3, 7)
    correction_rates = []
    for detail in forward_stats["correction_details"]:
        corr = detail["corrections"]
        total_corrections = (
            int(corr["player_fixed"])
            + corr["deadlocked_removed"]
            + int(corr["crates_balanced"])
            + corr["unpushable_removed"]
            + corr["unreachable_removed"]
        )
        correction_rates.append(total_corrections)

    ax7.hist(
        correction_rates,
        bins=range(0, max(correction_rates) + 2),
        color="#e74c3c",
        alpha=0.7,
    )
    ax7.set_xlabel("Corrections per Level")
    ax7.set_ylabel("Frequency")
    ax7.set_title("Forward: Corrections Distribution")

    # 8. Quality Score
    ax8 = plt.subplot(3, 3, 8)
    forward_quality = (
        forward_stats["valid_after_fix"] / forward_stats["total_levels"]
    ) * 100
    backward_quality = (
        backward_stats["valid_levels"] / backward_stats["total_levels"]
    ) * 100

    # Compute composite quality score
    forward_score = forward_quality - (
        sum(correction_rates) / len(correction_rates) * 5
    )
    backward_score = backward_quality

    scores = {"Forward\n(adjusted)": max(0, forward_score), "Backward": backward_score}
    bars = ax8.bar(scores.keys(), scores.values(), color=["#f39c12", "#27ae60"])
    ax8.set_ylabel("Quality Score")
    ax8.set_title("Overall Quality Comparison")
    ax8.set_ylim([0, 105])
    for bar in bars:
        height = bar.get_height()
        ax8.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, "comparison_analysis.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved comparison figure: {fig_path}")

    # Save detailed stats as JSON
    stats_path = os.path.join(output_dir, "comparison_stats.json")
    with open(stats_path, "w") as f:
        json.dump(
            {
                "forward": {
                    k: v for k, v in forward_stats.items() if k != "correction_details"
                },
                "backward": backward_stats,
            },
            f,
            indent=2,
            default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x,
        )
    print(f"✓ Saved statistics: {stats_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Compare forward vs backward generation"
    )
    parser.add_argument(
        "--forward-dir",
        type=str,
        required=True,
        help="Directory with forward-generated levels",
    )
    parser.add_argument(
        "--backward-dir",
        type=str,
        required=True,
        help="Directory with backward-generated levels",
    )
    parser.add_argument(
        "--solution-dir",
        type=str,
        default=None,
        help="Directory with solutions for backward levels",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="comparison_results",
        help="Output directory for comparison results",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FORWARD vs BACKWARD GENERATION COMPARISON")
    print("=" * 70)

    # Load levels
    print(f"\n📂 Loading levels...")
    print(f"  Forward: {args.forward_dir}")
    forward_levels = load_levels(args.forward_dir)
    print(f"  ✓ Loaded {len(forward_levels)} forward levels")

    print(f"  Backward: {args.backward_dir}")
    backward_levels = load_levels(args.backward_dir)
    print(f"  ✓ Loaded {len(backward_levels)} backward levels")

    # Analyze
    print(f"\n📊 Analyzing forward-generated levels...")
    forward_stats = analyze_forward_levels(forward_levels)
    print(f"  ✓ Analysis complete")

    print(f"\n📊 Analyzing backward-generated levels...")
    backward_stats = analyze_backward_levels(backward_levels, args.solution_dir)
    print(f"  ✓ Analysis complete")

    # Create report
    print(f"\n📈 Creating comparison report...")
    create_comparison_report(forward_stats, backward_stats, args.output_dir)

    print(f"\n" + "=" * 70)
    print("✓ Comparison complete!")
    print(f"Results saved to: {args.output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
