"""
Solvability Configuration for RAPCG-MetaRL
Tuned reward weights that emphasize solvability while maintaining resource awareness.
"""

# Zelda Solvability Configuration
ZELDA_SOLVABILITY_REWARDS = {
    "player": 5,           # Strong reward for single player (was 3)
    "key": 5,              # Strong reward for key placement (was 3)
    "door": 5,             # Strong reward for door placement (was 3)
    "regions": 10,         # CRITICAL: Single connected region (was 5)
    "enemies": 1,          # Moderate reward for enemies (unchanged)
    "nearest-enemy": 3,    # Good enemy distance (was 2)
    "path-length": 2       # Longer paths are better (was 1)
}

ZELDA_SOLVABILITY_PARAMS = {
    "target_enemy_dist": 4,   # Minimum distance from player to nearest enemy
    "target_path": 16,        # Target path length (player → key → door)
    "max_enemies": 5          # Maximum number of enemies
}

# Sokoban Solvability Configuration
SOKOBAN_SOLVABILITY_REWARDS = {
    "player": 5,           # Strong reward for single player (was 3)
    "crate": 3,            # Good crate placement (was 2)
    "target": 3,           # Good target placement (was 2)
    "regions": 10,         # CRITICAL: Single connected region (was 5)
    "ratio": 5,            # CRITICAL: Crates must equal targets (was 2)
    "dist-win": 5,         # CRITICAL: Solvability check (was 0.0!)
    "sol-length": 3        # Longer solutions are more interesting (was 1)
}

SOKOBAN_SOLVABILITY_PARAMS = {
    "solver_power": 5000,     # Computational budget for solver
    "max_crates": 3,          # Maximum number of crates
    "min_solution": 18        # Target solution length
}

# Resource-Aware Penalty Weights (unchanged)
RESOURCE_PENALTY_WEIGHTS = {
    "ram_penalty_weight": 0.2,
    "cpu_penalty_weight": 0.1,
    "gpu_penalty_weight": 0.1
}

# Combined configuration for easy access
SOLVABILITY_CONFIG = {
    'zelda': {
        'rewards': ZELDA_SOLVABILITY_REWARDS,
        'params': ZELDA_SOLVABILITY_PARAMS
    },
    'sokoban': {
        'rewards': SOKOBAN_SOLVABILITY_REWARDS,
        'params': SOKOBAN_SOLVABILITY_PARAMS
    }
}


def get_solvability_config(game: str):
    """
    Get solvability configuration for a specific game.
    
    Args:
        game: Game name ('zelda', 'sokoban')
        
    Returns:
        Dictionary with 'rewards' and 'params' keys
    """
    game_lower = game.lower()
    if game_lower in SOLVABILITY_CONFIG:
        return SOLVABILITY_CONFIG[game_lower]
    return None


def print_solvability_config(game: str):
    """Print solvability configuration for debugging."""
    config = get_solvability_config(game)
    if config:
        print(f"\n{'='*60}")
        print(f"Solvability Configuration: {game.upper()}")
        print(f"{'='*60}")
        print("Reward Weights:")
        for key, value in config['rewards'].items():
            print(f"  {key:20s}: {value}")
        print("\nParameters:")
        for key, value in config['params'].items():
            print(f"  {key:20s}: {value}")
        print(f"{'='*60}\n")
