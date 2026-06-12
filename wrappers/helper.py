# wrappers/helper.py
"""
Helper utilities for RAPCG-MetaRL
Includes resource monitoring, level parsing, and other utility functions.
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional


def parse_vglc_level(file_path: str) -> np.ndarray:
    """
    Convert VGLC tile-based level file into numpy array.

    Args:
        file_path: Path to VGLC level file (.txt or .json)

    Returns:
        numpy array representing the level grid
    """
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            data = json.load(f)
            # Handle different JSON structures
            if isinstance(data, dict) and "level" in data:
                level = data["level"]
            elif isinstance(data, list):
                level = data
            else:
                raise ValueError(f"Unsupported JSON structure in {file_path}")

            # Convert to numpy array
            return np.array(level)

    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            lines = f.readlines()
        grid = [list(line.strip()) for line in lines if line.strip()]
        return np.array(grid)

    else:
        raise ValueError(f"Unsupported file format: {file_path}")


def load_vglc_levels(data_dir: str, game: str) -> List[np.ndarray]:
    """
    Load all VGLC levels for a specific game.

    Args:
        data_dir: Directory containing VGLC data
        game: Game name (e.g., 'SMB', 'zelda')

    Returns:
        List of level arrays
    """
    levels = []
    json_file = os.path.join(data_dir, f"{game}.json")

    if os.path.exists(json_file):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)

            # Parse JSON structure
            if isinstance(data, dict):
                for level_name, level_data in data.items():
                    if isinstance(level_data, list):
                        levels.append(np.array(level_data))
            elif isinstance(data, list):
                levels = [np.array(level) for level in data]

        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    return levels


def tile_diversity(level: np.ndarray) -> float:
    """
    Calculate diversity metric for a level based on unique tiles.

    Args:
        level: Level array

    Returns:
        Diversity score (0-1)
    """
    unique_tiles = len(np.unique(level))
    total_tiles = level.size
    return unique_tiles / total_tiles if total_tiles > 0 else 0.0


def pattern_complexity(level: np.ndarray, window_size: int = 3) -> float:
    """
    Calculate pattern complexity using sliding window of unique patterns.

    Args:
        level: Level array
        window_size: Size of pattern window

    Returns:
        Complexity score
    """
    if level.size == 0:
        return 0.0

    patterns = set()
    h, w = level.shape if len(level.shape) == 2 else (1, level.shape[0])

    for i in range(h - window_size + 1):
        for j in range(w - window_size + 1):
            if len(level.shape) == 2:
                pattern = tuple(
                    level[i : i + window_size, j : j + window_size].flatten()
                )
            else:
                pattern = tuple(level[j : j + window_size])
            patterns.add(pattern)

    max_patterns = (h - window_size + 1) * (w - window_size + 1)
    return len(patterns) / max_patterns if max_patterns > 0 else 0.0


def calculate_content_metrics(level: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive content quality metrics for a level.

    Args:
        level: Level array

    Returns:
        Dictionary of metrics
    """
    metrics = {
        "diversity": tile_diversity(level),
        "complexity": pattern_complexity(level),
        "size": level.size,
        "unique_tiles": len(np.unique(level)),
    }

    return metrics


def save_level(level: np.ndarray, filepath: str, format: str = "npy"):
    """
    Save generated level to file.

    Args:
        level: Level array
        filepath: Output file path
        format: Save format ('npy', 'txt', 'json')
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    if format == "npy":
        np.save(filepath, level)
    elif format == "txt":
        with open(filepath, "w") as f:
            if len(level.shape) == 2:
                for row in level:
                    f.write("".join(map(str, row)) + "\n")
            else:
                f.write("".join(map(str, level)) + "\n")
    elif format == "json":
        with open(filepath, "w") as f:
            json.dump(level.tolist(), f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


def load_level(filepath: str) -> np.ndarray:
    """
    Load level from file.

    Args:
        filepath: Path to level file

    Returns:
        Level array
    """
    if filepath.endswith(".npy"):
        return np.load(filepath)
    elif filepath.endswith(".txt"):
        with open(filepath, "r") as f:
            lines = f.readlines()
        return np.array([list(line.strip()) for line in lines if line.strip()])
    elif filepath.endswith(".json"):
        with open(filepath, "r") as f:
            data = json.load(f)
        return np.array(data)
    else:
        raise ValueError(f"Unsupported file format: {filepath}")
