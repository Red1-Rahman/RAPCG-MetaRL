"""
Level Visualization for RAPCG-MetaRL
Generates high-quality images of Zelda and Sokoban levels for ACM TOG paper.
Uses actual tile PNG images for authentic game visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import os
from typing import List, Dict, Tuple, Optional
import argparse


# ============================================================================
# TILE IMAGE PATHS
# ============================================================================

ZELDA_TILE_PATHS = {
    0: 'zelda_tiles/empty.png',
    1: 'zelda_tiles/solid.png',
    2: 'zelda_tiles/player.png',
    3: 'zelda_tiles/key.png',
    4: 'zelda_tiles/door.png',
    5: 'zelda_tiles/bat.png',
    6: 'zelda_tiles/scorpion.png',
    7: 'zelda_tiles/spider.png',
}

SOKOBAN_TILE_PATHS = {
    0: 'sokoban_tiles/empty.png',
    1: 'sokoban_tiles/solid.png',
    2: 'sokoban_tiles/player.png',
    3: 'sokoban_tiles/crate.png',
    4: 'sokoban_tiles/target.png',
}

# Cache for loaded tile images
_TILE_CACHE = {}


def load_tile_images(game: str = 'zelda', tile_size: int = 16) -> Dict[int, np.ndarray]:
    """
    Load tile images from PNG files.
    
    Args:
        game: 'zelda' or 'sokoban'
        tile_size: Target size for tiles (will resize if needed)
    
    Returns:
        Dictionary mapping tile ID to RGB array
    """
    cache_key = f"{game}_{tile_size}"
    
    if cache_key in _TILE_CACHE:
        return _TILE_CACHE[cache_key]
    
    tile_paths = ZELDA_TILE_PATHS if game == 'zelda' else SOKOBAN_TILE_PATHS
    tile_images = {}
    
    for tile_id, path in tile_paths.items():
        if os.path.exists(path):
            img = Image.open(path).convert('RGB')
            
            # Resize to target size
            if img.size != (tile_size, tile_size):
                img = img.resize((tile_size, tile_size), Image.Resampling.NEAREST)
            
            tile_images[tile_id] = np.array(img, dtype=np.uint8)
        else:
            # Fallback to solid color if image not found
            print(f"⚠ Warning: Tile image not found: {path}, using fallback color")
            tile_images[tile_id] = create_fallback_tile(tile_id, game, tile_size)
    
    _TILE_CACHE[cache_key] = tile_images
    return tile_images


def create_fallback_tile(tile_id: int, game: str, size: int = 16) -> np.ndarray:
    """Create a solid color fallback tile."""
    # Fallback colors (from original implementation)
    if game == 'zelda':
        colors = {
            0: [255, 255, 255], 1: [0, 0, 0], 2: [0, 255, 0],
            3: [255, 255, 0], 4: [0, 255, 255], 5: [255, 0, 0],
            6: [255, 128, 0], 7: [128, 0, 128]
        }
    else:
        colors = {
            0: [255, 255, 255], 1: [0, 0, 0], 2: [0, 255, 0],
            3: [165, 42, 42], 4: [255, 0, 0]
        }
    
    color = colors.get(tile_id, [128, 128, 128])
    tile = np.full((size, size, 3), color, dtype=np.uint8)
    return tile

def level_to_rgb(level: np.ndarray, game: str = 'zelda', tile_size: int = 16) -> np.ndarray:
    """
    Convert a level array to RGB image using tile PNG images.
    
    Args:
        level: 2D numpy array with tile indices
        game: 'zelda' or 'sokoban'
        tile_size: Size of each tile in pixels
    
    Returns:
        RGB image as numpy array (H, W, 3)
    """
    tile_images = load_tile_images(game, tile_size)
    
    h, w = level.shape
    rgb = np.zeros((h * tile_size, w * tile_size, 3), dtype=np.uint8)
    
    for i in range(h):
        for j in range(w):
            tile_id = int(level[i, j])
            if tile_id in tile_images:
                tile_img = tile_images[tile_id]
                y_start = i * tile_size
                y_end = y_start + tile_size
                x_start = j * tile_size
                x_end = x_start + tile_size
                rgb[y_start:y_end, x_start:x_end] = tile_img
    
    return rgb


def render_level(level: np.ndarray, game: str = 'zelda', 
                 scale: int = 20, show_grid: bool = True) -> np.ndarray:
    """
    Render level with tile images and optional grid lines.
    
    Args:
        level: 2D numpy array with tile indices
        game: 'zelda' or 'sokoban'
        scale: Pixels per tile (tiles will be this size)
        show_grid: Whether to show grid lines
    
    Returns:
        RGB image as numpy array
    """
    # Render using tile images at the specified scale
    rgb = level_to_rgb(level, game, tile_size=scale)
    
    if show_grid:
        h, w = level.shape
        # Add grid lines
        grid_color = [80, 80, 80]  # Dark gray for better visibility
        line_width = max(1, scale // 16)  # Scale line width with tile size
        
        for i in range(h + 1):
            y = i * scale
            if y < rgb.shape[0]:
                rgb[y:min(y+line_width, rgb.shape[0]), :] = grid_color
        for j in range(w + 1):
            x = j * scale
            if x < rgb.shape[1]:
                rgb[:, x:min(x+line_width, rgb.shape[1])] = grid_color
    
    return rgb


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
    Create a legend showing tile types with actual tile images.
    
    Args:
        game: 'zelda' or 'sokoban'
        save_path: Path to save legend
        dpi: DPI for publication
    """
    # Tile names mapping
    if game == 'zelda':
        tile_names = ['Empty', 'Solid', 'Player', 'Key', 'Door', 'Bat', 'Scorpion', 'Spider']
    else:
        tile_names = ['Empty', 'Solid', 'Player', 'Crate', 'Target']
    
    # Load tile images
    tile_size = 48  # Larger for legend
    tile_images = load_tile_images(game, tile_size)
    
    fig, ax = plt.subplots(figsize=(6, len(tile_names) * 0.6), dpi=dpi)
    ax.axis('off')
    
    y_pos = 0
    for tile_id, name in enumerate(tile_names):
        if tile_id in tile_images:
            # Display actual tile image
            tile_img = tile_images[tile_id]
            
            # Create an inset axes for the tile
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            axins = ax.inset_axes([0, y_pos, 0.08, 0.08], transform=ax.transData)
            axins.imshow(tile_img)
            axins.axis('off')
            
            # Add label
            ax.text(0.12, y_pos + 0.04, f'{name} (ID: {tile_id})',
                   va='center', fontsize=12, fontweight='bold',
                   transform=ax.transData)
        
        y_pos -= 0.12
    
    ax.set_xlim(-0.02, 0.5)
    ax.set_ylim(y_pos + 0.1, 0.15)
    
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
    
    # Generate random levels with only one player
    levels = []
    if game == 'zelda':
        for _ in range(4):
            # Generate random level (excluding player tile initially)
            level = np.random.choice([0, 1, 3, 4, 5, 6, 7], size=(11, 11))
            # Place exactly one player at random position
            player_y, player_x = np.random.randint(1, 10), np.random.randint(1, 10)
            level[player_y, player_x] = 2  # Player tile
            levels.append(level)
    else:
        for _ in range(4):
            # Generate random level (excluding player tile initially)
            level = np.random.choice([0, 1, 3, 4], size=(10, 10))
            # Place exactly one player at random position
            player_y, player_x = np.random.randint(1, 9), np.random.randint(1, 9)
            level[player_y, player_x] = 2  # Player tile
            levels.append(level)
    
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
