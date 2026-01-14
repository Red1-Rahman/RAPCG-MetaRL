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
sys.path.append(os.path.join(project_root, 'gym-pcgrl'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import pandas as pd
from typing import List, Dict, Optional
from visualize_levels import render_level
import argparse

# Color mappings for charts (not for tile rendering)
ZELDA_COLORS = {
    0: [255, 255, 255],  # empty
    1: [0, 0, 0],        # solid
    2: [0, 255, 0],      # player
    3: [255, 255, 0],    # key
    4: [0, 255, 255],    # door
    5: [255, 0, 0],      # bat
    6: [255, 128, 0],    # scorpion
    7: [128, 0, 128],    # spider
}

SOKOBAN_COLORS = {
    0: [255, 255, 255],  # empty
    1: [0, 0, 0],        # solid
    2: [0, 255, 0],      # player
    3: [165, 42, 42],    # crate
    4: [255, 0, 0],      # target
}


def set_publication_style():
    """Set matplotlib style for ACM TOG publications."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman'],
        'font.size': 10,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
    })


def figure1_generated_levels_showcase(levels: List[np.ndarray], 
                                     game: str = 'zelda',
                                     save_path: str = 'figures/fig1_levels_showcase.png'):
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
        ax.set_title(f'Level {idx + 1}', fontweight='bold')
        ax.axis('off')
    
    plt.suptitle(f'Generated {game.capitalize()} Levels - Diversity Showcase', 
                fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 1: {save_path}")
    plt.close()


def figure2_training_progression(initial_levels: List[np.ndarray],
                                intermediate_levels: List[np.ndarray],
                                final_levels: List[np.ndarray],
                                timesteps: List[int],
                                game: str = 'zelda',
                                save_path: str = 'figures/fig2_training_progression.png'):
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
    fig, axes = plt.subplots(3, n_samples, figsize=(3*n_samples, 9))
    
    if n_samples == 1:
        axes = axes.reshape(3, 1)
    
    stages = [
        (initial_levels, f'Initial (0 steps)', 0),
        (intermediate_levels, f'Intermediate ({timesteps[1]:,} steps)', 1),
        (final_levels, f'Final ({timesteps[2]:,} steps)', 2)
    ]
    
    for stage_idx, (levels, title, row) in enumerate(stages):
        for col, level in enumerate(levels):
            ax = axes[row, col]
            rgb = render_level(level, game, scale=12, show_grid=True)
            ax.imshow(rgb)
            if col == 0:
                ax.set_ylabel(title, fontsize=11, fontweight='bold')
            if row == 0:
                ax.set_title(f'Sample {col+1}', fontweight='bold')
            ax.axis('off')
    
    plt.suptitle(f'{game.capitalize()} Level Generation - Training Progression', 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 2: {save_path}")
    plt.close()


def figure3_resource_quality_tradeoff(log_file: str,
                                     save_path: str = 'figures/fig3_resource_quality.png'):
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
    color_reward = 'tab:blue'
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Reward', color=color_reward)
    line1 = ax1.plot(df['step'], df['reward'].rolling(100).mean(), 
                     color=color_reward, label='Reward (smoothed)', linewidth=1.5)
    ax1.tick_params(axis='y', labelcolor=color_reward)
    ax1.grid(True, alpha=0.3)
    
    ax1_twin = ax1.twinx()
    color_ram = 'tab:red'
    ax1_twin.set_ylabel('RAM Usage (%)', color=color_ram)
    line2 = ax1_twin.plot(df['step'], df['ram_percent'].rolling(100).mean(), 
                          color=color_ram, label='RAM (smoothed)', 
                          linewidth=1.5, alpha=0.7)
    ax1_twin.tick_params(axis='y', labelcolor=color_ram)
    
    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', framealpha=0.9)
    ax1.set_title('(a) Training Dynamics', fontweight='bold')
    
    # Plot 2: Resource penalty over time
    if 'ram_penalty' in df.columns:
        ax2.plot(df['step'], df['ram_penalty'].rolling(100).mean(), 
                color='tab:orange', linewidth=1.5)
        ax2.fill_between(df['step'], 0, df['ram_penalty'].rolling(100).mean(), 
                         alpha=0.3, color='tab:orange')
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('RAM Penalty')
        ax2.set_title('(b) Resource Penalty', fontweight='bold')
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Resource-Quality Tradeoff During Training', 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 3: {save_path}")
    plt.close()


def figure4_algorithm_comparison(log_files: Dict[str, str],
                                 save_path: str = 'figures/fig4_algorithm_comparison.png'):
    """
    Figure 4: Comparison of different RL algorithms (PPO vs A2C).
    
    Args:
        log_files: Dict mapping algorithm name to log file path
        save_path: Output path
    """
    set_publication_style()
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(7.5, 6))
    
    colors = {'PPO': 'tab:blue', 'A2C': 'tab:green', 'PPO+Resource': 'tab:purple'}
    
    for algo_name, log_file in log_files.items():
        if not os.path.exists(log_file):
            print(f"⚠ Skipping {algo_name}: File not found")
            continue
            
        df = pd.read_csv(log_file)
        color = colors.get(algo_name, 'tab:gray')
        
        # Plot 1: Reward
        ax1.plot(df['step'], df['reward'].rolling(100).mean(), 
                label=algo_name, color=color, linewidth=1.5, alpha=0.8)
        
        # Plot 2: CPU Usage
        ax2.plot(df['step'], df['cpu_percent'].rolling(100).mean(), 
                label=algo_name, color=color, linewidth=1.5, alpha=0.8)
        
        # Plot 3: RAM Usage
        ax3.plot(df['step'], df['ram_percent'].rolling(100).mean(), 
                label=algo_name, color=color, linewidth=1.5, alpha=0.8)
        
        # Plot 4: Episode length (if available)
        if 'episode_length' in df.columns:
            ax4.plot(df['step'], df['episode_length'].rolling(100).mean(), 
                    label=algo_name, color=color, linewidth=1.5, alpha=0.8)
    
    # Configure subplots
    ax1.set_title('(a) Reward', fontweight='bold')
    ax1.set_ylabel('Reward')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2.set_title('(b) CPU Usage', fontweight='bold')
    ax2.set_ylabel('CPU %')
    ax2.grid(True, alpha=0.3)
    
    ax3.set_title('(c) RAM Usage', fontweight='bold')
    ax3.set_xlabel('Training Steps')
    ax3.set_ylabel('RAM %')
    ax3.grid(True, alpha=0.3)
    
    ax4.set_title('(d) Episode Length', fontweight='bold')
    ax4.set_xlabel('Training Steps')
    ax4.set_ylabel('Steps')
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Algorithm Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 4: {save_path}")
    plt.close()


def figure5_level_statistics(levels: List[np.ndarray],
                             game: str = 'zelda',
                             save_path: str = 'figures/fig5_level_statistics.png'):
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
    diversities = [m['diversity'] for m in all_metrics]
    complexities = [m['complexity'] for m in all_metrics]
    
    # Plot 1: Diversity distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(diversities, bins=15, color='tab:blue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Diversity Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('(a) Diversity Distribution', fontweight='bold')
    ax1.axvline(np.mean(diversities), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(diversities):.3f}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Complexity distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(complexities, bins=15, color='tab:green', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Complexity Score')
    ax2.set_ylabel('Frequency')
    ax2.set_title('(b) Complexity Distribution', fontweight='bold')
    ax2.axvline(np.mean(complexities), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {np.mean(complexities):.3f}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Diversity vs Complexity scatter
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.scatter(diversities, complexities, alpha=0.6, s=30, color='tab:purple')
    ax3.set_xlabel('Diversity')
    ax3.set_ylabel('Complexity')
    ax3.set_title('(c) Diversity vs Complexity', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Tile distribution
    ax4 = fig.add_subplot(gs[1, :])
    
    # Aggregate tile counts
    if game == 'zelda':
        tile_names = ['Empty', 'Solid', 'Player', 'Key', 'Door', 'Bat', 'Scorpion', 'Spider']
    else:
        tile_names = ['Empty', 'Solid', 'Player', 'Crate', 'Target']
    
    tile_counts = np.zeros(len(tile_names))
    for level in levels:
        for i in range(len(tile_names)):
            tile_counts[i] += np.sum(level == i)
    
    # Normalize
    tile_percentages = 100 * tile_counts / tile_counts.sum()
    
    colors_map = ZELDA_COLORS if game == 'zelda' else SOKOBAN_COLORS
    bar_colors = [np.array(colors_map[i]) / 255.0 for i in range(len(tile_names))]
    
    bars = ax4.bar(tile_names, tile_percentages, color=bar_colors, 
                   alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.set_ylabel('Percentage (%)')
    ax4.set_title('(d) Tile Distribution Across All Generated Levels', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels on bars
    for bar, pct in zip(bars, tile_percentages):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.suptitle(f'{game.capitalize()} Level Statistics (n={len(levels)})', 
                fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved Figure 5: {save_path}")
    plt.close()


def generate_demo_figures():
    """Generate all demo figures with sample data."""
    print("\n" + "="*70)
    print("Generating ACM TOG Paper Figures (Demo)")
    print("="*70)
    
    os.makedirs('figures', exist_ok=True)
    
    # Generate sample levels for both games
    for game in ['zelda', 'sokoban']:
        print(f"\n{'='*70}")
        print(f"Generating figures for {game.upper()}")
        print(f"{'='*70}")
        
        n_tiles = 8 if game == 'zelda' else 5
        size = (11, 11) if game == 'zelda' else (10, 10)
        
        # Figure 1: Showcase
        levels = [np.random.randint(0, n_tiles, size=size) for _ in range(6)]
        figure1_generated_levels_showcase(levels, game, 
                                         f'figures/{game}_fig1_showcase.png')
        
        # Figure 2: Training progression
        initial = [np.random.randint(0, n_tiles, size=size) for _ in range(3)]
        intermediate = [np.random.randint(0, n_tiles, size=size) for _ in range(3)]
        final = [np.random.randint(0, n_tiles, size=size) for _ in range(3)]
        figure2_training_progression(initial, intermediate, final, 
                                    [0, 50000, 100000], game,
                                    f'figures/{game}_fig2_progression.png')
        
        # Figure 5: Statistics
        many_levels = [np.random.randint(0, n_tiles, size=size) for _ in range(50)]
        figure5_level_statistics(many_levels, game, 
                                f'figures/{game}_fig5_statistics.png')
    
    print("\n" + "="*70)
    print("✓ All demo figures generated!")
    print("  Check the 'figures/' directory")
    print("="*70)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ACM TOG paper figures')
    parser.add_argument('--demo', action='store_true',
                       help='Generate demo figures with sample data')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Training log CSV file for Figure 3')
    parser.add_argument('--game', type=str, default='zelda',
                       choices=['zelda', 'sokoban'],
                       help='Game type')
    
    args = parser.parse_args()
    
    if args.demo:
        generate_demo_figures()
    elif args.log_file:
        figure3_resource_quality_tradeoff(args.log_file)
    else:
        print("Usage:")
        print("  python generate_paper_figures.py --demo              # Generate all demo figures")
        print("  python generate_paper_figures.py --log-file logs/training.csv  # Generate Figure 3")
