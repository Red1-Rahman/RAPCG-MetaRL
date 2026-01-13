"""
Level Visualization for RAPCG-MetaRL
Generates high-quality images of Zelda and Sokoban levels for ACM TOG paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from typing import List, Dict, Tuple, Optional
import argparse


# ============================================================================
# TILE COLOR MAPPINGS (Based on gym-pcgrl)
# ============================================================================

ZELDA_TILES = {
    'empty': 0,
    'solid': 1,
    'player': 2,
    'key': 3,
    'door': 4,
    'bat': 5,
    'scorpion': 6,
    'spider': 7,
}

ZELDA_COLORS = {
    0: [255, 255, 255],  # empty - white
    1: [0, 0, 0],        # solid - black
    2: [0, 255, 0],      # player - green
    3: [255, 255, 0],    # key - yellow
    4: [0, 255, 255],    # door - cyan
    5: [255, 0, 0],      # bat - red
    6: [255, 128, 0],    # scorpion - orange
    7: [128, 0, 128],    # spider - purple
}

SOKOBAN_TILES = {
    'empty': 0,
    'solid': 1,
    'player': 2,
    'crate': 3,
    'target': 4,
}

SOKOBAN_COLORS = {
    0: [255, 255, 255],  # empty - white
    1: [0, 0, 0],        # solid - black
    2: [0, 255, 0],      # player - green
    3: [165, 42, 42],    # crate - brown
    4: [255, 0, 0],      # target - red
}


def level_to_rgb(level: np.ndarray, game: str = 'zelda') -> np.ndarray:
    """
    Convert a level array to RGB image.
    
    Args:
        level: 2D numpy array with tile indices
        game: 'zelda' or 'sokoban'
    
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    colors = ZELDA_COLORS if game == 'zelda' else SOKOBAN_COLORS
    
    h, w = level.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    for tile_id, color in colors.items():
        mask = level == tile_id
        rgb[mask] = color
    
    return rgb


def render_level(level: np.ndarray, game: str = 'zelda', 
                 scale: int = 20, show_grid: bool = True) -> np.ndarray:
    """
    Render level with grid lines for better visibility.
    
    Args:
        level: 2D numpy array with tile indices
        game: 'zelda' or 'sokoban'
        scale: Pixels per tile
        show_grid: Whether to show grid lines
    
    Returns:
        RGB image as numpy array
    """
    rgb = level_to_rgb(level, game)
    h, w = level.shape
    
    # Scale up
    rgb_scaled = np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)
    
    if show_grid:
        # Add grid lines
        for i in range(h + 1):
            y = i * scale
            rgb_scaled[y:y+1, :] = [200, 200, 200]
        for j in range(w + 1):
            x = j * scale
            rgb_scaled[:, x:x+1] = [200, 200, 200]
    
    return rgb_scaled


def save_level_image(level: np.ndarray, filepath: str, game: str = 'zelda',
                     scale: int = 20, show_grid: bool = True, dpi: int = 300):
    """
    Save level as high-resolution image.
    
    Args:
        level: 2D numpy array with tile indices
        filepath: Output file path (PNG)
        game: 'zelda' or 'sokoban'
        scale: Pixels per tile
        show_grid: Whether to show grid lines
        dpi: DPI for publication quality (300 for ACM TOG)
    """
    rgb = render_level(level, game, scale, show_grid)
    img = Image.fromarray(rgb)
    
    # Create directory if needed
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    
    # Save with high DPI
    img.save(filepath, dpi=(dpi, dpi))
    print(f"✓ Saved: {filepath}")


def create_level_grid(levels: List[np.ndarray], titles: List[str], 
                      game: str = 'zelda', scale: int = 15,
                      save_path: Optional[str] = None, dpi: int = 300):
    """
    Create a grid of multiple levels for comparison.
    
    Args:
        levels: List of level arrays
        titles: List of titles for each level
        game: 'zelda' or 'sokoban'
        scale: Pixels per tile
        save_path: Path to save figure (None = show only)
        dpi: DPI for publication
    """
    n = len(levels)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows), dpi=dpi)
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (level, title) in enumerate(zip(levels, titles)):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        rgb = render_level(level, game, scale, show_grid=True)
        ax.imshow(rgb)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
    
    # Hide empty subplots
    for idx in range(n, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved grid: {save_path}")
    else:
        plt.show()
    
    plt.close()


def create_training_comparison(levels_before: List[np.ndarray],
                               levels_after: List[np.ndarray],
                               game: str = 'zelda',
                               save_path: Optional[str] = None,
                               dpi: int = 300):
    """
    Create before/after comparison for training progress.
    
    Args:
        levels_before: Initial levels
        levels_after: Trained levels
        game: 'zelda' or 'sokoban'
        save_path: Path to save figure
        dpi: DPI for publication
    """
    n = len(levels_before)
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8), dpi=dpi)
    
    if n == 1:
        axes = axes.reshape(2, 1)
    
    scale = 15
    
    for i in range(n):
        # Before
        rgb_before = render_level(levels_before[i], game, scale)
        axes[0, i].imshow(rgb_before)
        axes[0, i].set_title(f'Before Training - Sample {i+1}', fontsize=12, fontweight='bold')
        axes[0, i].axis('off')
        
        # After
        rgb_after = render_level(levels_after[i], game, scale)
        axes[1, i].imshow(rgb_after)
        axes[1, i].set_title(f'After Training - Sample {i+1}', fontsize=12, fontweight='bold')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved comparison: {save_path}")
    else:
        plt.show()
    
    plt.close()


def add_legend(game: str = 'zelda', save_path: Optional[str] = None, dpi: int = 300):
    """
    Create a legend showing tile types and colors.
    
    Args:
        game: 'zelda' or 'sokoban'
        save_path: Path to save legend
        dpi: DPI for publication
    """
    if game == 'zelda':
        tiles = ZELDA_TILES
        colors = ZELDA_COLORS
    else:
        tiles = SOKOBAN_TILES
        colors = SOKOBAN_COLORS
    
    fig, ax = plt.subplots(figsize=(6, len(tiles) * 0.5), dpi=dpi)
    ax.axis('off')
    
    y_pos = 0
    for name, tile_id in sorted(tiles.items(), key=lambda x: x[1]):
        color = np.array(colors[tile_id]) / 255.0
        
        # Draw color box
        rect = patches.Rectangle((0, y_pos), 0.5, 0.8, 
                                 facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # Add label
        ax.text(0.7, y_pos + 0.4, f'{name.capitalize()} (ID: {tile_id})',
               va='center', fontsize=12, fontweight='bold')
        
        y_pos -= 1
    
    ax.set_xlim(-0.1, 5)
    ax.set_ylim(y_pos, 1)
    
    title = f'{game.capitalize()} Tile Legend'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✓ Saved legend: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_from_file(filepath: str, game: str = 'zelda', 
                       output_dir: str = 'visualizations'):
    """
    Load and visualize level from file.
    
    Args:
        filepath: Path to .npy level file
        game: 'zelda' or 'sokoban'
        output_dir: Directory to save visualization
    """
    level = np.load(filepath)
    
    basename = os.path.basename(filepath).replace('.npy', '')
    output_path = os.path.join(output_dir, f'{basename}.png')
    
    save_level_image(level, output_path, game, scale=20, show_grid=True, dpi=300)
    
    return output_path


def demo_visualizations(game: str = 'zelda'):
    """
    Create demo visualizations with random levels.
    
    Args:
        game: 'zelda' or 'sokoban'
    """
    print(f"\nGenerating demo visualizations for {game.upper()}...")
    
    # Create output directory
    output_dir = f'figures/{game}_demo'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate random levels
    if game == 'zelda':
        levels = [np.random.randint(0, 8, size=(11, 11)) for _ in range(4)]
        tiles = ZELDA_TILES
    else:
        levels = [np.random.randint(0, 5, size=(10, 10)) for _ in range(4)]
        tiles = SOKOBAN_TILES
    
    # 1. Individual level images
    print("\n1. Individual level images:")
    for i, level in enumerate(levels[:2]):
        filepath = os.path.join(output_dir, f'level_{i+1}.png')
        save_level_image(level, filepath, game, scale=25, show_grid=True, dpi=300)
    
    # 2. Grid of levels
    print("\n2. Level grid:")
    titles = [f'Level {i+1}' for i in range(len(levels))]
    grid_path = os.path.join(output_dir, 'level_grid.png')
    create_level_grid(levels, titles, game, scale=15, save_path=grid_path, dpi=300)
    
    # 3. Training comparison
    print("\n3. Training comparison:")
    comparison_path = os.path.join(output_dir, 'training_comparison.png')
    create_training_comparison(levels[:2], levels[2:4], game, save_path=comparison_path, dpi=300)
    
    # 4. Legend
    print("\n4. Tile legend:")
    legend_path = os.path.join(output_dir, 'tile_legend.png')
    add_legend(game, save_path=legend_path, dpi=300)
    
    print(f"\n✓ Demo visualizations saved to: {output_dir}/")
    print(f"  Total files: {len(os.listdir(output_dir))}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize PCGRL levels')
    parser.add_argument('--game', type=str, default='zelda', choices=['zelda', 'sokoban'],
                       help='Game type')
    parser.add_argument('--demo', action='store_true',
                       help='Generate demo visualizations')
    parser.add_argument('--file', type=str, default=None,
                       help='Path to .npy level file to visualize')
    parser.add_argument('--output-dir', type=str, default='figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    if args.demo:
        # Generate demos for both games
        demo_visualizations('zelda')
        demo_visualizations('sokoban')
    elif args.file:
        visualize_from_file(args.file, args.game, args.output_dir)
    else:
        print("Usage:")
        print("  python visualize_levels.py --demo                    # Generate demo figures")
        print("  python visualize_levels.py --file level.npy --game zelda  # Visualize specific level")
