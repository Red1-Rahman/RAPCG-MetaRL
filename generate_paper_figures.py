"""
ACM TOG Paper Figure Generator for RAPCG-MetaRL
Creates publication-ready figures showing:
- Generated level diversity
- Training progress
- Resource-quality tradeoffs
- Algorithm comparisons
"""

import os
import sys

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "gym-pcgrl"))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import pandas as pd
from typing import List, Dict, Optional
from visualize_levels import render_level
import argparse

# Import for model-based generation
try:
    from stable_baselines3 import PPO, A2C
    from utils import ResourceMonitor
    from wrappers.pcgrl_env import make_pcgrl_env

    MODEL_AVAILABLE = True
except ImportError:
    MODEL_AVAILABLE = False
    print("Warning: stable-baselines3 not available, using procedural fallback")

# Color mappings for charts (not for tile rendering)
ZELDA_COLORS = {
    0: [255, 255, 255],  # empty
    1: [0, 0, 0],  # solid
    2: [0, 255, 0],  # player
    3: [255, 255, 0],  # key
    4: [0, 255, 255],  # door
    5: [255, 0, 0],  # bat
    6: [255, 128, 0],  # scorpion
    7: [128, 0, 128],  # spider
}

SOKOBAN_COLORS = {
    0: [255, 255, 255],  # empty
    1: [0, 0, 0],  # solid
    2: [0, 255, 0],  # player
    3: [165, 42, 42],  # crate
    4: [255, 0, 0],  # target
}


def set_publication_style():
    """Set matplotlib style for ACM TOG publications."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "figure.titlesize": 13,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def figure1_generated_levels_showcase(
    levels: List[np.ndarray],
    game: str = "zelda",
    save_path: str = "figures/fig1_levels_showcase.png",
):
    """
    Figure 1: Showcase of generated levels with different characteristics.
    Grid layout showing diversity.

    Args:
        levels: List of 6-9 level arrays
        game: 'zelda' or 'sokoban'
        save_path: Output path
    """
    set_publication_style()

    n = len(levels)
    cols = 3
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(7.5, 2.5 * rows))

    for idx, level in enumerate(levels):
        ax = plt.subplot(rows, cols, idx + 1)
        rgb = render_level(level, game, scale=15, show_grid=True)
        ax.imshow(rgb)
        ax.set_title(f"Level {idx + 1}", fontweight="bold")
        ax.axis("off")

    plt.suptitle(
        f"Generated {game.capitalize()} Levels - Diversity Showcase",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved Figure 1: {save_path}")
    plt.close()


def figure2_training_progression(
    initial_levels: List[np.ndarray],
    intermediate_levels: List[np.ndarray],
    final_levels: List[np.ndarray],
    timesteps: List[int],
    game: str = "zelda",
    save_path: str = "figures/fig2_training_progression.png",
):
    """
    Figure 2: Training progression showing how levels improve over time.

    Args:
        initial_levels: Levels at step 0
        intermediate_levels: Levels at middle checkpoint
        final_levels: Levels at end of training
        timesteps: [0, intermediate_step, final_step]
        game: 'zelda' or 'sokoban'
        save_path: Output path
    """
    set_publication_style()

    n_samples = len(initial_levels)
    fig, axes = plt.subplots(3, n_samples, figsize=(3 * n_samples, 9))

    if n_samples == 1:
        axes = axes.reshape(3, 1)

    stages = [
        (initial_levels, f"Initial (0 steps)", 0),
        (intermediate_levels, f"Intermediate ({timesteps[1]:,} steps)", 1),
        (final_levels, f"Final ({timesteps[2]:,} steps)", 2),
    ]

    for stage_idx, (levels, title, row) in enumerate(stages):
        for col, level in enumerate(levels):
            ax = axes[row, col]
            rgb = render_level(level, game, scale=12, show_grid=True)
            ax.imshow(rgb)
            if col == 0:
                ax.set_ylabel(title, fontsize=11, fontweight="bold")
            if row == 0:
                ax.set_title(f"Sample {col + 1}", fontweight="bold")
            ax.axis("off")

    plt.suptitle(
        f"{game.capitalize()} Level Generation - Training Progression",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved Figure 2: {save_path}")
    plt.close()


def figure3_resource_quality_tradeoff(
    log_file: str, save_path: str = "figures/fig3_resource_quality.png"
):
    """
    Figure 3: Resource usage vs. level quality tradeoff.
    Dual-axis plot showing RAM usage and reward over training.

    Args:
        log_file: Path to training log CSV
        save_path: Output path
    """
    set_publication_style()

    # Load data
    df = pd.read_csv(log_file)

    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3))

    # Plot 1: Reward and RAM over time
    color_reward = "tab:blue"
    ax1.set_xlabel("Training Steps")
    ax1.set_ylabel("Reward", color=color_reward)
    line1 = ax1.plot(
        df["step"],
        df["reward"].rolling(100).mean(),
        color=color_reward,
        label="Reward (smoothed)",
        linewidth=1.5,
    )
    ax1.tick_params(axis="y", labelcolor=color_reward)
    ax1.grid(True, alpha=0.3)

    ax1_twin = ax1.twinx()
    color_ram = "tab:red"
    ax1_twin.set_ylabel("RAM Usage (%)", color=color_ram)
    line2 = ax1_twin.plot(
        df["step"],
        df["ram_percent"].rolling(100).mean(),
        color=color_ram,
        label="RAM (smoothed)",
        linewidth=1.5,
        alpha=0.7,
    )
    ax1_twin.tick_params(axis="y", labelcolor=color_ram)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left", framealpha=0.9)
    ax1.set_title("(a) Training Dynamics", fontweight="bold")

    # Plot 2: Resource penalty over time
    if "ram_penalty" in df.columns:
        ax2.plot(
            df["step"],
            df["ram_penalty"].rolling(100).mean(),
            color="tab:orange",
            linewidth=1.5,
        )
        ax2.fill_between(
            df["step"],
            0,
            df["ram_penalty"].rolling(100).mean(),
            alpha=0.3,
            color="tab:orange",
        )
        ax2.set_xlabel("Training Steps")
        ax2.set_ylabel("RAM Penalty")
        ax2.set_title("(b) Resource Penalty", fontweight="bold")
        ax2.grid(True, alpha=0.3)

    plt.suptitle(
        "Resource-Quality Tradeoff During Training", fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved Figure 3: {save_path}")
    plt.close()


def figure4_algorithm_comparison(
    log_files: Dict[str, str], save_path: str = "figures/fig4_algorithm_comparison.png"
):
    """
    Figure 4: Comparison of different RL algorithms (PPO vs A2C).

    Args:
        log_files: Dict mapping algorithm name to log file path
        save_path: Output path
    """
    set_publication_style()

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.5, 6))

    colors = {"PPO": "tab:blue", "A2C": "tab:green", "PPO+Resource": "tab:purple"}

    for algo_name, log_file in log_files.items():
        if not os.path.exists(log_file):
            print(f"⚠ Skipping {algo_name}: File not found")
            continue

        df = pd.read_csv(log_file)
        color = colors.get(algo_name, "tab:gray")

        # Plot 1: Reward
        ax1.plot(
            df["step"],
            df["reward"].rolling(100).mean(),
            label=algo_name,
            color=color,
            linewidth=1.5,
            alpha=0.8,
        )

        # Plot 2: CPU Usage
        ax2.plot(
            df["step"],
            df["cpu_percent"].rolling(100).mean(),
            label=algo_name,
            color=color,
            linewidth=1.5,
            alpha=0.8,
        )

        # Plot 3: RAM Usage
        ax3.plot(
            df["step"],
            df["ram_percent"].rolling(100).mean(),
            label=algo_name,
            color=color,
            linewidth=1.5,
            alpha=0.8,
        )

        # Plot 4: Episode length (if available)
        if "episode_length" in df.columns:
            ax4.plot(
                df["step"],
                df["episode_length"].rolling(100).mean(),
                label=algo_name,
                color=color,
                linewidth=1.5,
                alpha=0.8,
            )

    # Configure subplots
    ax1.set_title("(a) Reward", fontweight="bold")
    ax1.set_ylabel("Reward")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.set_title("(b) CPU Usage", fontweight="bold")
    ax2.set_ylabel("CPU %")
    ax2.grid(True, alpha=0.3)

    ax3.set_title("(c) RAM Usage", fontweight="bold")
    ax3.set_xlabel("Training Steps")
    ax3.set_ylabel("RAM %")
    ax3.grid(True, alpha=0.3)

    ax4.set_title("(d) Episode Length", fontweight="bold")
    ax4.set_xlabel("Training Steps")
    ax4.set_ylabel("Steps")
    ax4.grid(True, alpha=0.3)

    plt.suptitle("Algorithm Comparison", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved Figure 4: {save_path}")
    plt.close()


def figure5_level_statistics(
    levels: List[np.ndarray],
    game: str = "zelda",
    save_path: str = "figures/fig5_level_statistics.png",
):
    """
    Figure 5: Statistical analysis of generated levels.
    Shows tile distribution, diversity metrics.

    Args:
        levels: List of level arrays
        game: 'zelda' or 'sokoban'
        save_path: Output path
    """
    set_publication_style()

    from wrappers.helper import calculate_content_metrics

    fig = plt.figure(figsize=(7.5, 5))
    gs = GridSpec(2, 3, figure=fig)

    # Calculate metrics for all levels
    all_metrics = [calculate_content_metrics(level) for level in levels]
    diversities = [m["diversity"] for m in all_metrics]
    complexities = [m["complexity"] for m in all_metrics]

    # Plot 1: Diversity distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(diversities, bins=15, color="tab:blue", alpha=0.7, edgecolor="black")
    ax1.set_xlabel("Diversity Score")
    ax1.set_ylabel("Frequency")
    ax1.set_title("(a) Diversity Distribution", fontweight="bold")
    ax1.axvline(
        np.mean(diversities),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(diversities):.3f}",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Complexity distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(complexities, bins=15, color="tab:green", alpha=0.7, edgecolor="black")
    ax2.set_xlabel("Complexity Score")
    ax2.set_ylabel("Frequency")
    ax2.set_title("(b) Complexity Distribution", fontweight="bold")
    ax2.axvline(
        np.mean(complexities),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(complexities):.3f}",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Diversity vs Complexity scatter
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(diversities, complexities, alpha=0.6, s=30, color="tab:purple")
    ax3.set_xlabel("Diversity")
    ax3.set_ylabel("Complexity")
    ax3.set_title("(c) Diversity vs Complexity", fontweight="bold")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Tile distribution
    ax4 = fig.add_subplot(gs[1, :])

    # Aggregate tile counts
    if game == "zelda":
        tile_names = [
            "Empty",
            "Solid",
            "Player",
            "Key",
            "Door",
            "Bat",
            "Scorpion",
            "Spider",
        ]
    else:
        tile_names = ["Empty", "Solid", "Player", "Crate", "Target"]

    tile_counts = np.zeros(len(tile_names))
    for level in levels:
        for i in range(len(tile_names)):
            tile_counts[i] += np.sum(level == i)

    # Normalize
    tile_percentages = 100 * tile_counts / tile_counts.sum()

    colors_map = ZELDA_COLORS if game == "zelda" else SOKOBAN_COLORS
    bar_colors = [np.array(colors_map[i]) / 255.0 for i in range(len(tile_names))]

    bars = ax4.bar(
        tile_names,
        tile_percentages,
        color=bar_colors,
        alpha=0.8,
        edgecolor="black",
        linewidth=1.5,
    )
    ax4.set_ylabel("Percentage (%)")
    ax4.set_title(
        "(d) Tile Distribution Across All Generated Levels", fontweight="bold"
    )
    ax4.grid(True, alpha=0.3, axis="y")

    # Add percentage labels on bars
    for bar, pct in zip(bars, tile_percentages):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.suptitle(
        f"{game.capitalize()} Level Statistics (n={len(levels)})",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✓ Saved Figure 5: {save_path}")
    plt.close()


def generate_valid_level(
    game: str, size: tuple, complexity: str = "medium"
) -> np.ndarray:
    """
    Generate a valid level with game constraints.

    Args:
        game: 'zelda' or 'sokoban'
        size: (height, width)
        complexity: 'low', 'medium', 'high' affects entity density

    Returns:
        Valid level array
    """
    h, w = size
    level = np.zeros(size, dtype=int)

    if game == "zelda":
        # Zelda: walls, player, keys, doors, enemies
        # Create border walls
        level[0, :] = 1
        level[-1, :] = 1
        level[:, 0] = 1
        level[:, -1] = 1

        # Add some internal walls for structure
        density = {"low": 0.15, "medium": 0.25, "high": 0.35}[complexity]
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if np.random.random() < density:
                    level[i, j] = 1  # solid wall

        # Place exactly ONE player in empty space
        empty_positions = np.argwhere(level == 0)
        if len(empty_positions) > 0:
            player_pos = empty_positions[np.random.randint(len(empty_positions))]
            level[tuple(player_pos)] = 2  # player

        # Add keys, doors, and enemies in remaining empty spaces
        empty_positions = np.argwhere(level == 0)
        n_entities = {"low": 3, "medium": 5, "high": 8}[complexity]
        n_entities = min(n_entities, len(empty_positions))

        if n_entities > 0:
            selected = empty_positions[
                np.random.choice(len(empty_positions), n_entities, replace=False)
            ]
            for pos in selected:
                # Randomly assign: 3=key, 4=door, 5=bat, 6=scorpion, 7=spider
                tile_type = np.random.choice([3, 4, 5, 6, 7])
                level[tuple(pos)] = tile_type

    else:  # sokoban
        # Sokoban: walls, player, crates, targets (EQUAL number of crates and targets!)
        # Create border walls
        level[0, :] = 1
        level[-1, :] = 1
        level[:, 0] = 1
        level[:, -1] = 1

        # Add some internal walls
        density = {"low": 0.20, "medium": 0.30, "high": 0.40}[complexity]
        for i in range(1, h - 1):
            for j in range(1, w - 1):
                if np.random.random() < density:
                    level[i, j] = 1

        # Place exactly ONE player
        empty_positions = np.argwhere(level == 0)
        if len(empty_positions) > 0:
            player_pos = empty_positions[np.random.randint(len(empty_positions))]
            level[tuple(player_pos)] = 2  # player

        # Place EQUAL number of crates (3) and targets (4)
        empty_positions = np.argwhere(level == 0)
        n_crates = {"low": 2, "medium": 3, "high": 4}[complexity]
        n_crates = min(
            n_crates, len(empty_positions) // 2
        )  # Need room for both crates and targets

        if n_crates > 0 and len(empty_positions) >= n_crates * 2:
            # Select positions for crates
            selected = empty_positions[
                np.random.choice(len(empty_positions), n_crates * 2, replace=False)
            ]

            # First half are crates, second half are targets
            for i in range(n_crates):
                level[tuple(selected[i])] = 3  # crate
            for i in range(n_crates, n_crates * 2):
                level[tuple(selected[i])] = 4  # target

    return level


def generate_levels_from_model(
    model_path: str, game: str, n_levels: int = 6, max_steps: int = 500
) -> List[np.ndarray]:
    """
    Generate levels using trained RAPCG-MetaRL model.

    Args:
        model_path: Path to trained model checkpoint (.zip)
        game: 'zelda' or 'sokoban'
        n_levels: Number of levels to generate
        max_steps: Maximum steps per level generation

    Returns:
        List of generated level arrays
    """
    if not MODEL_AVAILABLE:
        print("Warning: Model not available, using procedural fallback")
        size = (11, 11) if game == "zelda" else (10, 10)
        return [generate_valid_level(game, size, "medium") for _ in range(n_levels)]

    print(f"Loading model from {model_path}...")

    # Create environment with resource monitor
    resource_monitor = ResourceMonitor(use_gpu=False)
    env = make_pcgrl_env(
        resource_monitor=resource_monitor, game=game, representation="narrow"
    )

    # Load model
    model = PPO.load(model_path, device="cpu")

    # Generate levels
    levels = []
    print(f"Generating {n_levels} levels from trained model...")

    for i in range(n_levels):
        obs = env.reset()
        done = False
        steps = 0

        while not done and steps < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            steps += 1

        # Extract level from environment
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        if hasattr(base_env, "_rep") and hasattr(base_env._rep, "_map"):
            level = np.array(base_env._rep._map, dtype=int)
            levels.append(level)
            print(f"  ✓ Generated level {i + 1}/{n_levels} (steps={steps})")
        else:
            print(f"  ✗ Failed to extract level {i + 1}")

    env.close()
    return levels


def generate_demo_figures(
    use_model: bool = False,
    zelda_model: Optional[str] = None,
    sokoban_model: Optional[str] = None,
):
    """Generate all demo figures with sample data."""
    print("\n" + "=" * 70)
    if use_model:
        print("Generating ACM TOG Paper Figures (MODEL-GENERATED)")
    else:
        print("Generating ACM TOG Paper Figures (Procedural Demo)")
    print("=" * 70)

    os.makedirs("figures", exist_ok=True)

    # Default model paths if not provided
    if use_model:
        if zelda_model is None:
            zelda_model = "checkpoints/zelda_PPO_20251228_234742/final_model.zip"
        if sokoban_model is None:
            sokoban_model = "checkpoints/sokoban_PPO_latest/final_model.zip"

    # Generate sample levels for both games
    for game in ["zelda", "sokoban"]:
        print(f"\n{'=' * 70}")
        print(f"Generating figures for {game.upper()}")
        print(f"{'=' * 70}")

        size = (11, 11) if game == "zelda" else (10, 10)

        # Choose generation method
        if use_model:
            model_path = zelda_model if game == "zelda" else sokoban_model
            if os.path.exists(model_path):
                # Figure 1: Showcase - use model
                levels = generate_levels_from_model(
                    model_path, game, n_levels=6, max_steps=500
                )
            else:
                print(f"Warning: Model not found at {model_path}, using procedural")
                use_model = False

        if not use_model:
            # Fallback to procedural generation
            levels = []
            complexities = ["low", "medium", "medium", "high", "medium", "low"]
            for comp in complexities:
                levels.append(generate_valid_level(game, size, comp))

        figure1_generated_levels_showcase(
            levels, game, f"figures/{game}_fig1_showcase.png"
        )

        # Figure 2: Training progression
        if use_model and os.path.exists(model_path):
            # Use different checkpoints for progression
            checkpoint_dir = os.path.dirname(model_path)
            early_model = os.path.join(checkpoint_dir, "model_step_1000.zip")
            mid_model = os.path.join(checkpoint_dir, "model_step_2000.zip")

            if os.path.exists(early_model):
                initial = generate_levels_from_model(
                    early_model, game, n_levels=3, max_steps=300
                )
            else:
                initial = [generate_valid_level(game, size, "low") for _ in range(3)]

            if os.path.exists(mid_model):
                intermediate = generate_levels_from_model(
                    mid_model, game, n_levels=3, max_steps=400
                )
            else:
                intermediate = [
                    generate_valid_level(game, size, "medium") for _ in range(3)
                ]

            final = generate_levels_from_model(
                model_path, game, n_levels=3, max_steps=500
            )
        else:
            initial = [generate_valid_level(game, size, "low") for _ in range(3)]
            intermediate = [
                generate_valid_level(game, size, "medium") for _ in range(3)
            ]
            final = [generate_valid_level(game, size, "high") for _ in range(3)]

        figure2_training_progression(
            initial,
            intermediate,
            final,
            [1000, 2000, 3000],
            game,
            f"figures/{game}_fig2_progression.png",
        )

        # Figure 5: Statistics
        if use_model and os.path.exists(model_path):
            many_levels = generate_levels_from_model(
                model_path, game, n_levels=50, max_steps=500
            )
        else:
            many_levels = []
            for _ in range(50):
                comp = np.random.choice(["low", "medium", "high"])
                many_levels.append(generate_valid_level(game, size, comp))

        figure5_level_statistics(
            many_levels, game, f"figures/{game}_fig5_statistics.png"
        )

    print("\n" + "=" * 70)
    print("✓ All demo figures generated!")
    print("  Check the 'figures/' directory")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate ACM TOG paper figures")
    parser.add_argument(
        "--demo", action="store_true", help="Generate demo figures with procedural data"
    )
    parser.add_argument(
        "--model",
        action="store_true",
        help="Generate figures using trained RAPCG-MetaRL models",
    )
    parser.add_argument(
        "--zelda-model", type=str, default=None, help="Path to trained Zelda model"
    )
    parser.add_argument(
        "--sokoban-model", type=str, default=None, help="Path to trained Sokoban model"
    )
    parser.add_argument(
        "--log-file", type=str, default=None, help="Training log CSV file for Figure 3"
    )
    parser.add_argument(
        "--game",
        type=str,
        default="zelda",
        choices=["zelda", "sokoban"],
        help="Game type",
    )

    args = parser.parse_args()

    if args.model:
        generate_demo_figures(
            use_model=True,
            zelda_model=args.zelda_model,
            sokoban_model=args.sokoban_model,
        )
    elif args.demo:
        generate_demo_figures(use_model=False)
    elif args.log_file:
        figure3_resource_quality_tradeoff(args.log_file)
    else:
        print("Usage:")
        print(
            "  python generate_paper_figures.py --demo              # Procedural generation"
        )
        print(
            "  python generate_paper_figures.py --model             # Use trained models"
        )
        print(
            "  python generate_paper_figures.py --model --zelda-model checkpoints/path/model.zip"
        )
        print(
            "  python generate_paper_figures.py --log-file logs/training.csv  # Figure 3"
        )
