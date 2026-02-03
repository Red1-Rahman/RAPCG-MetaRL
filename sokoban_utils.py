"""
Comprehensive Sokoban level validation and solvability utilities.

Implements all Sokoban game rules with advanced deadlock detection.
"""

import numpy as np
import gym
import sys
import os
from typing import Tuple, Dict, List
from collections import deque

# Add gym-pcgrl to path for solver access
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'gym-pcgrl'))

# Import proper Sokoban solver with deadlock detection
try:
    from gym_pcgrl.envs.probs.sokoban.engine import State, AStarAgent, BFSAgent
    SOLVER_AVAILABLE = True
except ImportError:
    print("Warning: gym-pcgrl solver not available. Solvability checking disabled.")
    SOLVER_AVAILABLE = False


# Tile encoding:
# 0 = empty, 1 = wall, 2 = player, 3 = crate, 4 = target


def get_reachable_positions(level: np.ndarray, start_pos: tuple, 
                            walkable_tiles: list = None) -> set:
    """
    Get all positions reachable from start_pos using BFS.
    
    Args:
        level: 2D numpy array
        start_pos: (y, x) starting position
        walkable_tiles: List of walkable tile values (default: [0, 2, 4])
        
    Returns:
        Set of reachable (y, x) positions
    """
    if walkable_tiles is None:
        walkable_tiles = [0, 2, 4]  # empty, player, target
    
    h, w = level.shape
    visited = set()
    queue = deque([start_pos])
    visited.add(start_pos)
    
    while queue:
        y, x = queue.popleft()
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            
            if (0 <= ny < h and 0 <= nx < w and 
                (ny, nx) not in visited and 
                level[ny, nx] in walkable_tiles):
                visited.add((ny, nx))
                queue.append((ny, nx))
    
    return visited


def compute_dead_squares(level: np.ndarray, target_positions: list) -> set:
    """
    Compute dead squares - positions where a crate can never reach any target.
    Uses reverse BFS from all targets.
    
    Args:
        level: 2D numpy array
        target_positions: List of (y, x) target positions
        
    Returns:
        Set of dead square (y, x) positions
    """
    h, w = level.shape
    
    # Start BFS from all targets simultaneously
    reachable_from_targets = set()
    queue = deque(target_positions)
    
    for pos in target_positions:
        reachable_from_targets.add(pos)
    
    while queue:
        y, x = queue.popleft()
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            
            if (0 <= ny < h and 0 <= nx < w and 
                (ny, nx) not in reachable_from_targets and 
                level[ny, nx] in [0, 2, 3, 4]):  # walkable or crate
                reachable_from_targets.add((ny, nx))
                queue.append((ny, nx))
    
    # Dead squares are all non-wall positions NOT reachable from targets
    dead_squares = set()
    for y in range(h):
        for x in range(w):
            if level[y, x] != 1 and (y, x) not in reachable_from_targets:
                dead_squares.add((y, x))
    
    return dead_squares


def check_sokoban_deadlock(level: np.ndarray, crate_pos: tuple, 
                          dead_squares: set = None) -> bool:
    """
    Check if a crate is in a deadlock position.
    
    Detects:
    - Corner deadlocks (2 adjacent walls)
    - Wall-edge deadlocks (3+ adjacent walls)
    - Dead square positions (precomputed)
    - Wall-adjacent crates without nearby targets
    
    Args:
        level: 2D numpy array
        crate_pos: (y, x) position of crate
        dead_squares: Precomputed set of dead squares
        
    Returns:
        True if crate is deadlocked, False otherwise
    """
    y, x = crate_pos
    h, w = level.shape
    
    # Check dead squares first (fast lookup)
    if dead_squares is not None and crate_pos in dead_squares:
        return True
    
    # Get adjacent tiles
    up = level[y-1, x] if y > 0 else 1
    down = level[y+1, x] if y < h-1 else 1
    left = level[y, x-1] if x > 0 else 1
    right = level[y, x+1] if x < w-1 else 1
    
    # Count walls
    wall_count = sum([up == 1, down == 1, left == 1, right == 1])
    
    # Check corner deadlocks (2 adjacent walls)
    if (up == 1 and left == 1) or (up == 1 and right == 1) or \
       (down == 1 and left == 1) or (down == 1 and right == 1):
        return True
    
    # Check wall-edge deadlocks (3+ walls)
    if wall_count >= 3:
        return True
    
    # Check wall edge deadlocks with 2 walls
    if up == 1 and (left == 1 or right == 1):
        return True
    if down == 1 and (left == 1 or right == 1):
        return True
    if left == 1 and (up == 1 or down == 1):
        return True
    if right == 1 and (up == 1 or down == 1):
        return True
    
    # Check if crate is against wall without nearby target
    # Horizontal wall check
    if up == 1:  # Against top wall
        has_target_on_wall = False
        for dx in range(-2, 3):  # Check nearby wall
            check_x = x + dx
            if 0 <= check_x < w and level[y, check_x] == 4:
                has_target_on_wall = True
                break
        if not has_target_on_wall:
            return True
    
    if down == 1:  # Against bottom wall
        has_target_on_wall = False
        for dx in range(-2, 3):
            check_x = x + dx
            if 0 <= check_x < w and level[y, check_x] == 4:
                has_target_on_wall = True
                break
        if not has_target_on_wall:
            return True
    
    # Vertical wall check
    if left == 1:  # Against left wall
        has_target_on_wall = False
        for dy in range(-2, 3):
            check_y = y + dy
            if 0 <= check_y < h and level[check_y, x] == 4:
                has_target_on_wall = True
                break
        if not has_target_on_wall:
            return True
    
    if right == 1:  # Against right wall
        has_target_on_wall = False
        for dy in range(-2, 3):
            check_y = y + dy
            if 0 <= check_y < h and level[check_y, x] == 4:
                has_target_on_wall = True
                break
        if not has_target_on_wall:
            return True
    
    return False


def remove_deadlocked_crates(level: np.ndarray, target_positions: list = None) -> Tuple[np.ndarray, int]:
    """
    Remove all deadlocked crates from the level.
    
    Args:
        level: 2D numpy array
        target_positions: Optional list of target positions for dead square computation
        
    Returns:
        (modified_level, num_removed)
    """
    level = level.copy()
    
    # Compute dead squares if targets provided
    dead_squares = None
    if target_positions:
        dead_squares = compute_dead_squares(level, target_positions)
    
    crate_positions = np.argwhere(level == 3)
    removed_count = 0
    
    for crate_pos in crate_positions:
        if check_sokoban_deadlock(level, tuple(crate_pos), dead_squares):
            level[tuple(crate_pos)] = 0
            removed_count += 1
    
    return level, removed_count


def check_crate_pushability(level: np.ndarray, crate_pos: tuple) -> bool:
    """
    Check if a crate has at least one free adjacent tile to push from.
    
    Args:
        level: 2D numpy array
        crate_pos: (y, x) position of crate
        
    Returns:
        True if crate can be pushed, False otherwise
    """
    y, x = crate_pos
    h, w = level.shape
    
    # Check all 4 directions
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        push_from_y, push_from_x = y + dy, x + dx
        push_to_y, push_to_x = y - dy, x - dx
        
        if 0 <= push_from_y < h and 0 <= push_from_x < w:
            if 0 <= push_to_y < h and 0 <= push_to_x < w:
                # Can push if there's walkable space to push from and to
                push_from_tile = level[push_from_y, push_from_x]
                push_to_tile = level[push_to_y, push_to_x]
                
                if push_from_tile in [0, 2, 4] and push_to_tile in [0, 4]:
                    return True
    
    return False


def check_player_can_reach_crates(level: np.ndarray, player_pos: tuple, crate_positions: list) -> bool:
    """
    Check if player can reach all crates using BFS.
    
    Args:
        level: 2D numpy array
        player_pos: (y, x) position of player
        crate_positions: list of (y, x) positions of crates
        
    Returns:
        True if player can reach all crates, False otherwise
    """
    reachable = get_reachable_positions(level, player_pos)
    
    for crate_pos in crate_positions:
        if crate_pos not in reachable:
            return False
    
    return True


def check_crate_to_target_path(level: np.ndarray, crate_pos: tuple, target_positions: list) -> bool:
    """
    Check if a crate can reach at least one target (ignoring other crates).
    
    Args:
        level: 2D numpy array
        crate_pos: (y, x) position of crate
        target_positions: list of (y, x) positions of targets
        
    Returns:
        True if crate can reach at least one target, False otherwise
    """
    # Get reachable positions from crate (treat crate as empty for pathfinding)
    reachable = get_reachable_positions(level, crate_pos)
    
    for target_pos in target_positions:
        if target_pos in reachable:
            return True
    
    return False


def check_target_dead_position(level: np.ndarray, target_pos: tuple) -> bool:
    """
    Check if a target is in a dead position (corner or surrounded by 3+ walls).
    
    Args:
        level: 2D numpy array
        target_pos: (y, x) position of target
        
    Returns:
        True if target is in dead position, False otherwise
    """
    y, x = target_pos
    h, w = level.shape
    
    # Count adjacent walls
    wall_count = 0
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if ny < 0 or ny >= h or nx < 0 or nx >= w or level[ny, nx] == 1:
            wall_count += 1
    
    # Dead if in corner (2+ walls) or surrounded (3+ walls)
    return wall_count >= 2


def validate_and_fix_sokoban(level: np.ndarray, min_crates: int = 1,
                              enforce_all_rules: bool = True, verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Validate and fix Sokoban level to meet all game constraints.
    
    Rules enforced:
    1. Exactly 1 player (placed in reachable center position)
    2. Equal number of crates and targets (minimum min_crates)
    3. At least 1 crate-target pair
    4. Remove deadlocked crates (corners, walls, dead squares)
    5. Player can reach all crates (if enforce_all_rules=True)
    6. Each crate has at least one free adjacent tile to push from
    7. Each crate can reach at least one target (ignoring other crates)
    8. No targets in dead positions (corners, 3-wall surrounds)
    
    Args:
        level: 2D numpy array
        min_crates: Minimum number of crate-target pairs
        enforce_all_rules: If True, apply all validation rules
        
    Returns:
        (fixed_level, corrections_dict)
    """
    level = level.copy()
    h, w = level.shape
    
    corrections = {
        'original_players': np.sum(level == 2),
        'original_crates': np.sum(level == 3),
        'original_targets': np.sum(level == 4),
        'player_fixed': False,
        'deadlocked_removed': 0,
        'unpushable_removed': 0,
        'unreachable_removed': 0,
        'dead_targets_removed': 0,
        'crates_balanced': False,
        'final_players': 0,
        'final_crates': 0,
        'final_targets': 0
    }
    
    # Fix 1: Ensure exactly 1 player
    player_positions = np.argwhere(level == 2)
    if len(player_positions) != 1:
        # Remove all players
        level[level == 2] = 0
        
        # Place player in center of largest open area
        empty_positions = np.argwhere(level == 0)
        if len(empty_positions) > 0:
            center_y, center_x = h // 2, w // 2
            closest_empty = min(empty_positions, 
                              key=lambda p: (p[0]-center_y)**2 + (p[1]-center_x)**2)
            level[tuple(closest_empty)] = 2
            corrections['player_fixed'] = True
    
    # Get current positions
    player_positions = np.argwhere(level == 2)
    if len(player_positions) == 0:
        # Emergency: place player anywhere
        empty_positions = np.argwhere(level == 0)
        if len(empty_positions) > 0:
            level[tuple(empty_positions[0])] = 2
            player_positions = np.argwhere(level == 2)
    
    player_pos = tuple(player_positions[0]) if len(player_positions) > 0 else None
    
    # Fix 2: Remove targets in dead positions (if enforce_all_rules)
    if enforce_all_rules:
        target_positions = np.argwhere(level == 4)
        dead_targets = []
        for target_pos in target_positions:
            if check_target_dead_position(level, tuple(target_pos)):
                level[tuple(target_pos)] = 0
                dead_targets.append(target_pos)
        corrections['dead_targets_removed'] = len(dead_targets)
    
    # Fix 3: Remove deadlocked crates
    target_positions = [(y, x) for y, x in np.argwhere(level == 4)]
    level, deadlocked = remove_deadlocked_crates(level, target_positions)
    corrections['deadlocked_removed'] = deadlocked
    
    # Fix 4: Remove unpushable crates (if enforce_all_rules)
    if enforce_all_rules:
        crate_positions = np.argwhere(level == 3)
        unpushable = []
        for crate_pos in crate_positions:
            if not check_crate_pushability(level, tuple(crate_pos)):
                level[tuple(crate_pos)] = 0
                unpushable.append(crate_pos)
        corrections['unpushable_removed'] = len(unpushable)
    
    # Fix 5: Remove crates with no path to any target (if enforce_all_rules)
    if enforce_all_rules:
        target_positions = [(y, x) for y, x in np.argwhere(level == 4)]
        if target_positions:
            crate_positions = np.argwhere(level == 3)
            unreachable = []
            for crate_pos in crate_positions:
                if not check_crate_to_target_path(level, tuple(crate_pos), target_positions):
                    level[tuple(crate_pos)] = 0
                    unreachable.append(crate_pos)
            corrections['unreachable_removed'] = len(unreachable)
    
    # Fix 6: Balance crates and targets
    crate_count = np.sum(level == 3)
    target_count = np.sum(level == 4)
    target_pairs = max(min_crates, min(crate_count, target_count))
    
    if crate_count != target_pairs or target_count != target_pairs:
        corrections['crates_balanced'] = True
        
        # Remove excess crates
        if crate_count > target_pairs:
            crate_positions = np.argwhere(level == 3)
            for i in range(crate_count - target_pairs):
                level[tuple(crate_positions[i])] = 0
        
        # Remove excess targets
        if target_count > target_pairs:
            target_positions = np.argwhere(level == 4)
            for i in range(target_count - target_pairs):
                level[tuple(target_positions[i])] = 0
        
        # Add crates if needed
        if crate_count < target_pairs:
            needed = target_pairs - crate_count
            empty_positions = np.argwhere(level == 0)
            if len(empty_positions) < needed:
                target_pairs = max(1, crate_count + len(empty_positions))
                needed = len(empty_positions)
            
            for i in range(needed):
                level[tuple(empty_positions[i])] = 3
        
        # Add targets if needed
        if target_count < target_pairs:
            needed = target_pairs - target_count
            empty_positions = np.argwhere(level == 0)
            if len(empty_positions) < needed:
                needed = len(empty_positions)
            
            for i in range(needed):
                level[tuple(empty_positions[i])] = 4
    
    # Fix 7: Ensure player can reach all crates (if enforce_all_rules)
    if enforce_all_rules and player_pos:
        crate_positions = [(y, x) for y, x in np.argwhere(level == 3)]
        target_positions = [(y, x) for y, x in np.argwhere(level == 4)]
        
        if crate_positions:
            reachable = get_reachable_positions(level, player_pos)
            unreachable = [pos for pos in crate_positions if pos not in reachable]
            
            if unreachable:
                # Remove unreachable crates
                for pos in unreachable:
                    level[pos] = 0
                corrections['unreachable_removed'] += len(unreachable)
                
                # Also remove equal number of targets to maintain balance
                target_positions = np.argwhere(level == 4)
                for i in range(min(len(unreachable), len(target_positions))):
                    level[tuple(target_positions[i])] = 0
    
    # Fix 8: ENSURE MINIMUM CRATES/TARGETS (critical!)
    final_crate_count = np.sum(level == 3)
    final_target_count = np.sum(level == 4)
    
    if final_crate_count < min_crates or final_target_count < min_crates:
        corrections['crates_balanced'] = True
        
        # Get reachable empty positions
        empty_positions = np.argwhere(level == 0)
        if player_pos:
            reachable = get_reachable_positions(level, player_pos)
            reachable_empty = [tuple(pos) for pos in empty_positions if tuple(pos) in reachable]
        else:
            reachable_empty = [tuple(pos) for pos in empty_positions]
        
        # Add crates to reach minimum
        crates_needed = min_crates - final_crate_count
        if crates_needed > 0 and len(reachable_empty) > 0:
            for i in range(min(crates_needed, len(reachable_empty))):
                level[reachable_empty[i]] = 3
                reachable_empty = reachable_empty[1:]  # Remove used position
        
        # Add targets to reach minimum
        targets_needed = min_crates - final_target_count
        if targets_needed > 0 and len(reachable_empty) > 0:
            for i in range(min(targets_needed, len(reachable_empty))):
                level[reachable_empty[i]] = 4
    
    # Final counts
    corrections['final_players'] = np.sum(level == 2)
    corrections['final_crates'] = np.sum(level == 3)
    corrections['final_targets'] = np.sum(level == 4)
    
    return level, corrections


def check_solvability(level: np.ndarray, solver_power: int = 5000) -> Tuple[bool, List, int]:
    """
    Check if level is solvable using proper A* solver with deadlock detection.
    
    Uses gym-pcgrl's multi-strategy solver:
    1. BFS (fast, complete)
    2. A* with balance=1 (heuristic-focused)
    3. A* with balance=0.5 (balanced)
    4. A* with balance=0 (cost-focused)
    
    Args:
        level: 2D numpy array with tile encoding 0-4
        solver_power: Maximum iterations for solver
        
    Returns:
        (is_solvable, solution, heuristic_distance)
        - is_solvable: True if level has a solution
        - solution: List of moves if solvable, empty list otherwise
        - heuristic_distance: Distance to win state (0 if solvable)
    """
    if not SOLVER_AVAILABLE:
        return False, [], -1
    
    # Convert to format expected by engine
    lvl = np.pad(level, 1)  # Add border walls
    gameCharacters = "# @$."
    lvlString = ""
    for i in range(lvl.shape[0]):
        for j in range(lvl.shape[1]):
            lvlString += gameCharacters[int(lvl[i][j])]
            if j == lvl.shape[1] - 1:
                lvlString += "\n"
    
    # Initialize state
    try:
        state = State()
        state.stringInitialize(lvlString.split("\n"))
    except Exception as e:
        # Invalid level format
        return False, [], -1
    
    # Try multiple solver strategies (in order of speed/effectiveness)
    aStarAgent = AStarAgent()
    bfsAgent = BFSAgent()
    
    # 1. Try BFS first (fast for simple levels)
    try:
        sol, solState, iters = bfsAgent.getSolution(state, solver_power)
        if solState.checkWin():
            return True, sol, 0
    except Exception:
        pass
    
    # 2. Try A* with different balance parameters
    for balance in [1, 0.5, 0]:
        try:
            sol, solState, iters = aStarAgent.getSolution(state, balance, solver_power)
            if solState.checkWin():
                return True, sol, 0
        except Exception:
            continue
    
    # Unsolvable - return heuristic distance
    try:
        return False, [], solState.getHeuristic()
    except:
        return False, [], -1


def is_valid_sokoban(level: np.ndarray) -> Tuple[bool, str]:
    """
    Check if level meets basic Sokoban requirements.
    
    Returns:
        (is_valid, error_message)
    """
    player_count = np.sum(level == 2)
    crate_count = np.sum(level == 3)
    target_count = np.sum(level == 4)
    
    if player_count == 0:
        return False, "No player"
    if player_count > 1:
        return False, f"Multiple players ({player_count})"
    if crate_count == 0:
        return False, "No crates"
    if target_count == 0:
        return False, "No targets"
    if crate_count != target_count:
        return False, f"Mismatched crates ({crate_count}) and targets ({target_count})"
    
    return True, "Valid"


def print_level_stats(level: np.ndarray, title: str = "Level Stats", verbose: bool = False):
    """
    Print statistics about a Sokoban level.
    
    Args:
        level: 2D numpy array
        title: Title for the stats
        verbose: If True, show detailed validation checks
    """
    print(f"\n{title}:")
    print(f"  Dimensions: {level.shape[0]}x{level.shape[1]}")
    print(f"  Players: {np.sum(level == 2)}")
    print(f"  Crates: {np.sum(level == 3)}")
    print(f"  Targets: {np.sum(level == 4)}")
    print(f"  Walls: {np.sum(level == 1)}")
    print(f"  Empty: {np.sum(level == 0)}")
    
    is_valid, msg = is_valid_sokoban(level)
    print(f"  Status: {'✓ ' + msg if is_valid else '✗ ' + msg}")
    
    if verbose and is_valid:
        # Show detailed checks
        player_positions = np.argwhere(level == 2)
        crate_positions = [(y, x) for y, x in np.argwhere(level == 3)]
        target_positions = [(y, x) for y, x in np.argwhere(level == 4)]
        
        if len(player_positions) > 0 and crate_positions:
            player_pos = tuple(player_positions[0])
            
            # Check reachability
            reachable = get_reachable_positions(level, player_pos)
            unreachable = [pos for pos in crate_positions if pos not in reachable]
            can_reach = len(unreachable) == 0
            print(f"  Player reachability: {'✓ All crates' if can_reach else f'✗ {len(unreachable)} unreachable'}")
            
            # Check pushability
            unpushable = sum(1 for pos in crate_positions if not check_crate_pushability(level, pos))
            print(f"  Crate pushability: {'✓ All pushable' if unpushable == 0 else f'✗ {unpushable} unpushable'}")
            
            # Check paths to targets
            if target_positions:
                no_path = sum(1 for pos in crate_positions 
                            if not check_crate_to_target_path(level, pos, target_positions))
                print(f"  Crate-target paths: {'✓ All connected' if no_path == 0 else f'✗ {no_path} no path'}")
            
            # Check solvability with actual A* solver
            if SOLVER_AVAILABLE:
                is_solvable, solution, dist = check_solvability(level)
                if is_solvable:
                    print(f"  ✓ SOLVABLE - Solution length: {len(solution)}")
                else:
                    print(f"  ✗ UNSOLVABLE - Heuristic distance: {dist}")


class SokobanSolvabilityWrapper(gym.Wrapper):
    """
    Gym wrapper that validates Sokoban levels and computes solvability metrics.
    """
    
    def __init__(self, env, enforce_all_rules: bool = True, verbose: bool = False,
                 unsolvable_penalty: float = 0, min_solution_length: int = 0,
                 max_solution_length: int = 100, terminate_on_unsolvable: bool = False):
        super().__init__(env)
        self.enforce_all_rules = enforce_all_rules
        self.verbose = verbose
        
        # Legacy parameters (kept for compatibility but not used)
        self.unsolvable_penalty = unsolvable_penalty
        self.min_solution_length = min_solution_length
        self.max_solution_length = max_solution_length
        self.terminate_on_unsolvable = terminate_on_unsolvable
        
        self.validation_stats = {
            'total_resets': 0,
            'total_fixes': 0,
            'player_fixes': 0,
            'deadlock_removals': 0,
            'balance_fixes': 0
        }
    
    def reset(self, **kwargs):
        # Get initial observation
        obs = self.env.reset(**kwargs)
        
        # Extract numpy array from observation
        # obs might be OrderedDict with keys like 'pos', 'map', etc.
        if isinstance(obs, dict):
            # Get the map/level data
            level = obs.get('map', obs.get('level', obs.get('observation', None)))
            if level is None:
                # If no standard key, try first array value
                for value in obs.values():
                    if isinstance(value, np.ndarray):
                        level = value
                        break
        else:
            level = obs
        
        # Validate and fix if needed
        fixed_level, corrections = validate_and_fix_sokoban(
            level, 
            min_crates=1,
            enforce_all_rules=self.enforce_all_rules
        )
        
        # Update stats
        self.validation_stats['total_resets'] += 1
        if corrections['player_fixed'] or corrections['crates_balanced'] or corrections['deadlocked_removed'] > 0:
            self.validation_stats['total_fixes'] += 1
        if corrections['player_fixed']:
            self.validation_stats['player_fixes'] += 1
        if corrections['deadlocked_removed'] > 0:
            self.validation_stats['deadlock_removals'] += corrections['deadlocked_removed']
        if corrections['crates_balanced']:
            self.validation_stats['balance_fixes'] += 1
        
        if self.verbose:
            print_level_stats(fixed_level, "Reset Level", verbose=True)
        
        # Update environment state
        self.env.unwrapped._rep._map = fixed_level
        
        # Update observation if it was a dict
        if isinstance(obs, dict):
            for key in obs.keys():
                if isinstance(obs[key], np.ndarray) and obs[key].shape == level.shape:
                    obs[key] = fixed_level
                    break
        
        return obs
    
    def get_stats(self) -> dict:
        """Get validation statistics."""
        return self.validation_stats.copy()


# Test cases
def test_validator():
    """Test the validator with various edge cases."""
    print("="*60)
    print("Testing Sokoban Validator")
    print("="*60)
    
    # Test 1: Multiple players
    print("\nTest 1: Multiple players")
    level1 = np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 0, 2, 1],
        [1, 3, 4, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 1, 1, 1, 1]
    ])
    print_level_stats(level1, "Before Fix")
    fixed1, corr1 = validate_and_fix_sokoban(level1)
    print_level_stats(fixed1, "After Fix", verbose=True)
    
    # Test 2: Deadlocked crates
    print("\nTest 2: Corner deadlock")
    level2 = np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 0, 0, 1],
        [1, 3, 0, 0, 1],
        [1, 0, 4, 4, 1],
        [1, 1, 1, 1, 1]
    ])
    print_level_stats(level2, "Before Fix")
    fixed2, corr2 = validate_and_fix_sokoban(level2)
    print_level_stats(fixed2, "After Fix", verbose=True)
    
    # Test 3: Mismatched crates/targets
    print("\nTest 3: Mismatched crates/targets")
    level3 = np.array([
        [1, 1, 1, 1, 1, 1],
        [1, 2, 0, 0, 0, 1],
        [1, 3, 3, 3, 0, 1],
        [1, 4, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 1]
    ])
    print_level_stats(level3, "Before Fix")
    fixed3, corr3 = validate_and_fix_sokoban(level3, enforce_all_rules=True)
    print_level_stats(fixed3, "After Fix", verbose=True)
    
    # Test 4: Valid level (should pass)
    print("\nTest 4: Already valid level")
    level4 = np.array([
        [1, 1, 1, 1, 1],
        [1, 2, 0, 0, 1],
        [1, 0, 3, 0, 1],
        [1, 0, 4, 0, 1],
        [1, 1, 1, 1, 1]
    ])
    print_level_stats(level4, "Before Fix", verbose=True)
    fixed4, corr4 = validate_and_fix_sokoban(level4, enforce_all_rules=True)
    print_level_stats(fixed4, "After Fix", verbose=True)
    
    print("\n" + "="*60)
    print("Tests Complete")
    print("="*60)


if __name__ == "__main__":
    test_validator()
