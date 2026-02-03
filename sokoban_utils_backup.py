"""
Sokoban Utilities for RAPCG-MetaRL
Unified module containing:
1. Level validation and correction (post-processing)
2. Solvability wrapper (training-time enforcement)
3. Deadlock detection
4. Quality metrics

This consolidates sokoban_validator.py and sokoban_solvability_wrapper.py
into a single, well-organized module.
"""
import gym
import numpy as np
from typing import Tuple, Dict


# ============================================================================
# REACHABILITY AND PATHFINDING
# ============================================================================

def get_reachable_positions(level: np.ndarray, start_pos: tuple, 
                            walkable_tiles: list = [0, 2, 4]) -> set:
    """
    Get all positions reachable from start position using BFS.
    
    Args:
        level: 2D numpy array
        start_pos: (y, x) starting position
        walkable_tiles: List of tile IDs that can be walked on
        
    Returns:
        Set of (y, x) positions reachable from start
    """
    from collections import deque
    
    h, w = level.shape
    visited = set()
    queue = deque([start_pos])
    visited.add(start_pos)
    
    while queue:
        y, x = queue.popleft()
        
        # Check 4 adjacent positions
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                if level[ny, nx] in walkable_tiles:
                    visited.add((ny, nx))
                    queue.append((ny, nx))
    
    return visited


def compute_dead_squares(level: np.ndarray, target_positions: list) -> set:
    """
    Compute dead squares using reverse BFS from all targets.
    A dead square is one from which no crate can reach any target.
    
    Args:
        level: 2D numpy array
        target_positions: List of (y, x) target positions
        
    Returns:
        Set of (y, x) positions that are dead squares
    """
    from collections import deque
    
    h, w = level.shape
    reachable_from_targets = set()
    
    # BFS from all targets simultaneously
    queue = deque(target_positions)
    for pos in target_positions:
        reachable_from_targets.add(pos)
    
    while queue:
        y, x = queue.popleft()
        
        # Check 4 directions - can we push a crate FROM this position?
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in reachable_from_targets:
                # Check if this is a valid position for a crate
                if level[ny, nx] in [0, 4]:  # Empty or target
                    # Check if we can push FROM the opposite direction
                    push_y, push_x = ny - dy, nx - dx
                    if 0 <= push_y < h and 0 <= push_x < w:
                        if level[push_y, push_x] in [0, 2, 4]:  # Walkable
                            reachable_from_targets.add((ny, nx))
                            queue.append((ny, nx))
    
    # Dead squares are all empty positions NOT reachable from targets
    dead_squares = set()
    for y in range(h):
        for x in range(w):
            if level[y, x] in [0, 4] and (y, x) not in reachable_from_targets:
                dead_squares.add((y, x))
    
    return dead_squares


# ============================================================================
# DEADLOCK DETECTION
# ============================================================================

def check_sokoban_deadlock(level: np.ndarray, crate_pos: tuple, 
                          dead_squares: set = None) -> bool:
    """
    Check if a crate is in a deadlock position (cannot be moved to any target).
    
    Deadlock cases:
    1. Crate in corner (2 adjacent walls)
    2. Crate on wall edge with walls on both perpendicular sides
    3. Crate on a precomputed dead square
    4. Crate against wall without target on that wall
    
    Args:
        level: 2D numpy array with tile IDs
        crate_pos: (y, x) position of crate to check
        dead_squares: Optional precomputed set of dead square positions
        
    Returns:
        True if deadlocked, False otherwise
    """
    y, x = crate_pos
    h, w = level.shape
    
    # Check if this position is a target (targets can't be deadlocks)
    if level[y, x] == 4:
        return False
    
    # Check if on a precomputed dead square
    if dead_squares and (y, x) in dead_squares:
        return True
    
    # Get adjacent tiles (up, down, left, right)
    up = level[y-1, x] if y > 0 else 1
    down = level[y+1, x] if y < h-1 else 1
    left = level[y, x-1] if x > 0 else 1
    right = level[y, x+1] if x < w-1 else 1
    
    # Check corner deadlocks (two adjacent walls or crates)
    if (up == 1 or up == 3) and (left == 1 or left == 3):
        return True
    if (up == 1 or up == 3) and (right == 1 or right == 3):
        return True
    if (down == 1 or down == 3) and (left == 1 or left == 3):
        return True
    if (down == 1 or down == 3) and (right == 1 or right == 3):
        return True
    
    # Check wall edge deadlocks
    if up == 1 and (left == 1 or right == 1):
        return True
    if down == 1 and (left == 1 or right == 1):
        return True
    if left == 1 and (up == 1 or down == 1):
        return True
    if right == 1 and (up == 1 or down == 1):
        return True
    
    # Check if crate is against wall without target on that wall
    # Horizontal wall check
    if up == 1:  # Against top wall
        # Check if there's a target on this horizontal wall section
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
    Remove all deadlocked crates from level.
    
    Args:
        level: 2D numpy array with tile IDs
        target_positions: Optional list of target positions for dead square computation
        
    Returns:
        (corrected_level, num_removed)
    """
    level = level.copy()
    crate_positions = np.argwhere(level == 3)
    
    # Compute dead squares if targets provided
    dead_squares = None
    if target_positions is None:
        target_positions = [tuple(pos) for pos in np.argwhere(level == 4)]
    if target_positions:
        dead_squares = compute_dead_squares(level, target_positions)
    
    removed = 0
    for pos in crate_positions:
        if check_sokoban_deadlock(level, tuple(pos), dead_squares):
            level[tuple(pos)] = 0  # Remove deadlocked crate
            removed += 1
    
    return level, removed


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
                              enforce_all_rules: bool = True) -> Tuple[np.ndarray, Dict]:
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
        level: 2D numpy array with tile IDs
        min_crates: Minimum number of crate-target pairs
        verbose: Print correction details
        enforce_all_rules: If True, enforce advanced rules (may be slow)n
        crate_positions: List of (y, x) crate positions
        
    Returns:
        (all_reachable, unreachable_crates)
    """
    # Get all positions reachable by player (can walk on empty, targets, and player tile)
    reachable = get_reachable_positions(level, player_pos, walkable_tiles=[0, 2, 4])
    
    unreachable_crates = []
    for crate_pos in crate_positions:
        # Player needs to reach adjacent to crate, not the crate itself
        y, x = crate_pos
        adjacent_reachable = False
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj_y, adj_x = y + dy, x + dx
            if (adj_y, adj_x) in reachable:
                adjacent_reachable = True
                break
        
        if not adjacent_reachable:
            unreachable_crates.append(crate_pos)
    
    return len(unreachable_crates) == 0, unreachable_crates


def check_crate_to_target_path(level: np.ndarray, crate_pos: tuple, 
                                target_positions: list) -> bool:
    """
    Check if there exists a path from crate to any target (ignoring other crates).
    
    Args:
        level: 2D numpy array
        crate_pos: (y, x) crate position
        target_positions: List of (y, x) target positions
        
    Returns:
        True if path exists to at least one target
    """
    # Temporarily remove this crate and all other crates
    level_temp = level.copy()
    level_temp[level_temp == 3] = 0  # Remove all crates
    
    # BFS from crate position to any target
    reachable = get_reachable_positions(level_temp, crate_pos, walkable_tiles=[0, 4])
    
    # Check if any target is reachable
    for target_pos in target_positions:
        if target_pos in reachable:
            return True
    
    return False


def check_target_dead_position(level: np.ndarray, target_pos: tuple) -> bool:
    """
    Check if a target is placed in a static dead position.
    A target in a corner or surrounded by walls is useless.
    
    Args:
        level: 2D numpy array
        target_pos: (y, x) target position
        
    Returns:
        True if target is in dead position, False otherwise
    """
    y, x = target_pos
    h, w = level.shape
    
    # Get adjacent tiles
    up = level[y-1, x] if y > 0 else 1
    down = level[y+1, x] if y < h-1 else 1
    left = level[y, x-1] if x > 0 else 1
    right = level[y, x+1] if x < w-1 else 1
    
    # Check corner positions (2 adjacent walls)
    if (up == 1 and left == 1) or (up == 1 and right == 1) or \
       (down == 1 and left == 1) or (down == 1 and right == 1):
        return True
    
    # Check if surrounded by walls on 3 sides
    walls = sum([up == 1, down == 1, left == 1, right == 1])
    if walls >= 3:
        return True
    
    return Falset_on_wal targets (targets in corners/surrounded by walls)
    if enforce_all_rules:
        target_positions = np.argwhere(level == 4)
        dead_targets = []
        for pos in target_positions:
            if check_target_dead_position(level, tuple(pos)):
                level[tuple(pos)] = 0
                dead_targets.append(pos)
        corrections['dead_targets_removed'] = len(dead_targets)
        if dead_targets and verbose:
          4: Remove unpushable crates (no free adjacent space)
    if enforce_all_rules:
        crate_positions = np.argwhere(level == 3)
        unpushable = []
        for pos in crate_positions:
            if not check_crate_pushability(level, tuple(pos)):
                level[tuple(pos)] = 0
                unpushable.append(pos)
        corrections['unpushable_removed'] = len(unpushable)
        if unpushable and verbose:
            print(f"  ✓ Removed {len(unpushable)} unpushable crates")
        crate_count = np.sum(level == 3)
    
    # Fix 5: Remove crates with no path to any target
    if enforce_all_rules and target_count > 0:
        crate_positions = np.argwhere(level == 3)
        target_positions = [tuple(pos) for pos in np.argwhere(level == 4)]
        unreachable = []
        for pos in crate_positions:
            if not check_crate_to_target_path(level, tuple(pos), target_positions):
                level[tuple(pos)] = 0
                unreachable.append(pos)
        if unreachable:
            corrections['unreachable_removed'] = len(unreachable)
            if verbose:
                print(f"  ✓ Removed {len(unreachable)} crates with no path to target")
        crate_count = np.sum(level == 3)
    
    # Fix 6 print(f"  ✓ Removed {len(dead_targets)} dead targets")
    
    # Fix 3: Remove deadlocked crates (with dead square computation)
    target_positions = [tuple(pos) for pos in np.argwhere(level == 4)]
    level, removed = remove_deadlocked_crates(level, target_positions
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
    
    if down == 1 and (left == 1 or right == 1):
        return True
    if left == 1 and (up == 1 or down == 1):
        return True
    if right == 1 and (up == 1 or down == 1):
        return True
    
    return False


def remove_deadlocked_crates(level: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Remove all deadlocked crates from level.
    
    Args:
        level: 2D numpy array with tile IDs
        
    Returns:
        (corrected_level, num_removed)
    """
    level = level.copy()
    crate_positions = np.argwhere(level == 3)
    removed = 0
    
    for pos in crate_positions:
        if check_sokoban_deadlock(level, tuple(pos)):
            level[tuple(pos)] = 0  # Remove deadlocked crate
            removed += 1
    
    return level, removed


# ============================================================================
# LEVEL VALIDATION AND CORRECTION
# ============================================================================

def is_valid_sokoban(level: np.ndarray) -> Tuple[bool, str]:
    """
    Check if a Sokoban level is valid.
    
    Rules:
    - Exactly 1 player
    - Equal number of crates and targets
    - At least 1 crate-target pair
    
    Returns:
        (is_valid, error_message)
    """
    player_count = np.sum(level == 2)
    crate_count = np.sum(level == 3)
    target_count = np.sum(level == 4)
    
    if player_count != 1:
        return False, f"Invalid player count: {player_count} (expected 1)"
    
    if crate_count != target_count:
        return False, f"Crate/target mismatch: {crate_count} crates, {target_count} targets"
    
    if crate_count < 1:
        return False, "No crates/targets (no goal)"
    
    return True, "Valid"


def validate_and_fix_sokoban(level: np.ndarray, min_crates: int = 2, 
                              verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Validate and fix Sokoban level to meet all game constraints.
    
    Fixes applied:
    1. Ensure exactly 1 player (placed in center-ish position)
    2. Remove deadlocked crates
    3. Balance crates and targets (minimum min_crates pairs)
    4. Ensure at least 1 crate-target pair exists
    
    Args:
        level: 2D numpy array with tile IDs
        min_crates: Minimum number of crate-target pairs
        verbose: Print correction details
        
    Returns:
        (corrected_level, corrections_dict)
    """
    level = level.copy()
    corrections = {
        'player_fixed': False,
        'deadlocks_removed': 0,
        'crate_target_fixed': False,
        'original_players': np.sum(level == 2),
        'original_crates': np.sum(level == 3),
        'original_targets': np.sum(level == 4),
        'final_players': 0,
        'final_crates': 0,
        'final_targets': 0
    }
    
    if verbose:
        print(f"  Initial: {corrections['original_players']} players, "
              f"{corrections['original_crates']} crates, "
              f"{corrections['original_targets']} targets")
    
    # Fix 1: Ensure exactly 1 player
    if corrections['original_players'] != 1:
        corrections['player_fixed'] = True
        
        # Remove all existing players
        level[level == 2] = 0
        
        # Place exactly 1 player in center-ish position
        empty_positions = np.argwhere(level == 0)
        if len(empty_positions) > 0:
            h, w = level.shape
            center_y, center_x = h // 2, w // 2
            distances = np.sqrt((empty_positions[:, 0] - center_y)**2 + 
                              (empty_positions[:, 1] - center_x)**2)
            best_pos_idx = np.argmin(distances)
            player_pos = empty_positions[best_pos_idx]
            level[tuple(player_pos)] = 2
            
            if verbose:
                print(f"  ✓ Fixed player: {corrections['original_players']} → 1")
        else:
            # Last resort: replace a wall with player
            wall_positions = np.argwhere(level == 1)
            if len(wall_positions) > 0:
                level[tuple(wall_positions[0])] = 2
                if verbose:
                    print(f"  ✓ Fixed player (replaced wall)")
    
    # Fix 2: Remove deadlocked crates
    level, removed = remove_deadlocked_crates(level)
    corrections['deadlocks_removed'] = removed
    if removed > 0 and verbose:
        print(f"  ✓ Removed {removed} deadlocked crates")
    
    # Recount after fixes
    crate_count = np.sum(level == 3)
    target_count = np.sum(level == 4)
    
    # Fix 3: Balance crates and targets (minimum min_crates)
    if crate_count != target_count or crate_count < min_crates:
        corrections['crate_target_fixed'] = True
        
        # Target number of pairs (at least min_crates)
        target_pairs = max(min_crates, min(crate_count, target_count))
        
        # Get positions
        crate_positions = np.argwhere(level == 3)
        target_positions = np.argwhere(level == 4)
        empty_positions = np.argwhere(level == 0)
        
        # Adjust crates
        if crate_count > target_pairs:
            # Remove extra crates
            for i in range(crate_count - target_pairs):
                if i < len(crate_positions):
                    level[tuple(crate_positions[i])] = 0
        elif crate_count < target_pairs:
    # Fix 7: Ensure player can reach all crates (final check)
    if enforce_all_rules and corrections['final_crates'] > 0:
        player_positions = np.argwhere(level == 2)
        if len(player_positions) > 0:, verbose: bool = False):
    """Print detailed statistics about a Sokoban level."""
    print(f"\n{title}:")
    print(f"  Shape: {level.shape}")
    print(f"  Empty (0):   {np.sum(level == 0):3d}")
    print(f"  Solid (1):   {np.sum(level == 1):3d}")
    print(f"  Player (2):  {np.sum(level == 2):3d} {'✓' if np.sum(level == 2) == 1 else '✗'}")
    print(f"  Crate (3):   {np.sum(level == 3):3d}")
    print(f"  Target (4):  {np.sum(level == 4):3d} {'✓' if np.sum(level == 3) == np.sum(level == 4) else '✗'}")
    
    is_valid, msg = is_valid_sokoban(level)
    print(f"  Status: {msg}")
    
    if verbose and is_valid:
        # Additional detailed checks
        player_positions = np.argwhere(level == 2)
        crate_positions = [tuple(pos) for pos in np.argwhere(level == 3)]
        target_positions = [tuple(pos) for pos in np.argwhere(level == 4)]
        
        if len(player_positions) > 0 and len(crate_positions) > 0:
            player_pos = tuple(player_positions[0])
            can_reach, unreachable = check_player_can_reach_crates(level, player_pos, crate_positions)
            print(f"  Player reachability: {'✓ All crates' if can_reach else f'✗ {len(unreachable)} unreachable'}")
            
            # Check pushability
            unpushable = sum(1 for pos in crate_positions if not check_crate_pushability(level, pos))
            print(f"  Crate pushability: {'✓ All pushable' if unpushable == 0 else f'✗ {unpushable} unpushable'}")
            
            # Check paths to targets
            if target_positions:
                no_path = sum(1 for pos in crate_positions 
                            if not check_crate_to_target_path(level, pos, target_positions))
                print(f"  Crate-target paths: {'✓ All connected' if no_path == 0 else f'✗ {no_path} no path's['final_crates'] = np.sum(level == 3)
                
                # Also remove equal number of targets to maintain balance
                target_positions = np.argwhere(level == 4)
                for i in range(min(len(unreachable), len(target_positions))):
                    level[tuple(target_positions[i])] = 0
                corrections['final_targets'] = np.sum(level == 4)
                
                if verbose:
                    print(f"  ✓ Removed {len(unreachable)} unreachable crates and targets")
    
            # Add crates
            needed = target_pairs - crate_count
            empty_positions = np.argwhere(level == 0)
            if len(empty_positions) < needed:
                target_pairs = max(1, crate_count + len(empty_positions))
                needed = target_pairs - crate_count
            for i in range(min(needed, len(empty_positions))):
                level[tuple(empty_positions[i])] = 3
        
        # Adjust targets
        target_positions = np.argwhere(level == 4)
        current_targets = len(target_positions)
        
        if current_targets > target_pairs:
            # Remove extra targets
            for i in range(current_targets - target_pairs):
                if i < len(target_positions):
                    level[tuple(target_positions[i])] = 0
        elif current_targets < target_pairs:
            # Add targets
            needed = target_pairs - current_targets
            empty_positions = np.argwhere(level == 0)
            
            if len(empty_positions) < needed:
                # Convert walls as last resort
                wall_positions = np.argwhere(level == 1)
                for i in range(min(needed - len(empty_positions), len(wall_positions))):
                    level[tuple(wall_positions[-(i+1)])] = 4
                needed = min(needed, len(empty_positions))
            
            for i in range(min(needed, len(empty_positions))):
                level[tuple(empty_positions[i])] = 4
        
        if verbose:
            print(f"  ✓ Fixed crates/targets: {crate_count}/{target_count} → {target_pairs}/{target_pairs}")
    
    # Final counts
    corrections['final_players'] = np.sum(level == 2)
    corrections['final_crates'] = np.sum(level == 3)
    corrections['final_targets'] = np.sum(level == 4)
    
    # Verify constraints
    assert corrections['final_players'] == 1, \
        f"Player count still wrong: {corrections['final_players']}"
    assert corrections['final_crates'] == corrections['final_targets'], \
        f"Crate/target mismatch: {corrections['final_crates']}/{corrections['final_targets']}"
    
    if corrections['final_crates'] < 1 and verbose:
        print(f"  ⚠ Warning: Level has no crates/targets")
    
    return level, corrections


def print_level_stats(level: np.ndarray, title: str = "Level Stats"):
    """Print detailed statistics about a Sokoban level."""
    print(f"\n{title}:")
    print(f"  Shape: {level.shape}")
    print(f"  Empty (0):   {np.sum(level == 0):3d}")
    print(f"  Solid (1):   {np.sum(level == 1):3d}")
    print(f"  Player (2):  {np.sum(level == 2):3d} {'✓' if np.sum(level == 2) == 1 else '✗'}")
    print(f"  Crate (3):   {np.sum(level == 3):3d}")
    print(f"  Target (4):  {np.sum(level == 4):3d} {'✓' if np.sum(level == 3) == np.sum(level == 4) else '✗'}")
    
    is_valid, msg = is_valid_sokoban(level)
    print(f"  Status: {msg}")


# ============================================================================
# GYM WRAPPER FOR TRAINING-TIME ENFORCEMENT
# ============================================================================

class SokobanSolvabilityWrapper(gym.Wrapper):
    """
    Gym wrapper that enforces solvability during training.
    
    This wrapper:
    1. Checks if generated levels are solvable (using gym-pcgrl's solver)
    2. Applies large penalties for unsolvable levels
    3. Applies rewards for solvable levels with good solution lengths
    4. Tracks solvability statistics
    5. Optionally terminates episodes on unsolvable levels
    
    Use this during TRAINING to teach the agent to generate solvable levels.
    Use validate_and_fix_sokoban() during INFERENCE to correct invalid levels.
    """
    
    def __init__(self, env, unsolvable_penalty=25.0, solvable_reward=10.0,
                 solution_bonus=5.0, min_solution_length=5, 
                 max_solution_length=50, terminate_on_unsolvable=False):
        """
        Args:
            env: Base Sokoban PCGRL environment
            unsolvable_penalty: Penalty for unsolvable levels (default: 25.0)
            solvable_reward: Reward for solvable levels (default: 10.0)
            solution_bonus: Bonus for optimal solution length (default: 5.0)
            min_solution_length: Minimum acceptable solution length
            max_solution_length: Maximum acceptable solution length
            terminate_on_unsolvable: End episode on unsolvable level
        """
        super().__init__(env)
        self.unsolvable_penalty = unsolvable_penalty
        self.solvable_reward = solvable_reward
        self.solution_bonus = solution_bonus
        self.min_solution_length = min_solution_length
        self.max_solution_length = max_solution_length
        self.terminate_on_unsolvable = terminate_on_unsolvable
        
        # Statistics
        self.total_levels = 0
        self.unsolvable_levels = 0
        self.solvable_levels = 0
        self.solution_lengths = []
        
    def reset(self, **kwargs):
        """Reset environment and statistics."""
        obs = self.env.reset(**kwargs)
        
        self.total_levels = 0
        self.unsolvable_levels = 0
        self.solvable_levels = 0
        self.solution_lengths = []
        
        return obs
    
    def step(self, action):
        """Step with solvability checking and reward shaping."""
        obs, reward, done, info = self.env.step(action)
        
        # Extract stats
        if 'stats' in info:
            stats = info['stats']
        else:
            try:
                stats = self.env.unwrapped._prob.get_stats(
                    self.env.unwrapped._rep._map
                )
            except:
                return obs, reward, done, info
        
        # Check if level is complete
        is_complete = (
            stats.get('player', 0) == 1 and
            stats.get('crate', 0) > 0 and
            stats.get('crate', 0) == stats.get('target', 0) and
            stats.get('regions', 0) == 1
        )
        
        if is_complete:
            self.total_levels += 1
            
            # Check solvability
            solution = stats.get('solution', [])
            dist_win = stats.get('dist-win', float('inf'))
            is_solvable = len(solution) > 0 and dist_win == 0
            
            if is_solvable:
                self.solvable_levels += 1
                self.solution_lengths.append(len(solution))
                
                # Big reward for solvability
                solvability_reward = self.solvable_reward
                
                # Bonus for good solution length
                if self.min_solution_length <= len(solution) <= self.max_solution_length:
                    solvability_reward += self.solution_bonus
                
                reward += solvability_reward
                
                if not isinstance(info, dict):
                    info = {}
                info['solvable'] = True
                info['solution_length'] = len(solution)
                info['solvability_reward'] = solvability_reward
                
            else:
                # Heavy penalty for unsolvable levels
                self.unsolvable_levels += 1
                penalty = self.unsolvable_penalty
                reward -= penalty
                
                if self.terminate_on_unsolvable:
                    done = True
                
                if not isinstance(info, dict):
                    info = {}
                info['solvable'] = False
                info['solution_length'] = 0
                info['solvability_penalty'] = penalty
                info['dist_win'] = dist_win
                info['terminated_unsolvable'] = self.terminate_on_unsolvable
        
        # Add statistics
        if self.total_levels > 0:
            info['solvability_rate'] = self.solvable_levels / self.total_levels
            info['unsolvable_rate'] = self.unsolvable_levels / self.total_levels
            if len(self.solution_lengths) > 0:
                info['avg_solution_length'] = np.mean(self.solution_lengths)
        
        return obs, reward, done, info
    
    def get_statistics(self) -> Dict:
        """Get accumulated solvability statistics."""
        if self.total_levels == 0:
            return {
                'total_levels': 0,
                'solvable_levels': 0,
                'unsolvable_levels': 0,
                'solvability_rate': 0.0,
                'avg_solution_length': 0.0
            }
        
        return {
            'total_levels': self.total_levels,
            'solvable_levels': self.solvable_levels,
            'unsolvable_levels': self.unsolvable_levels,
            'solvability_rate': self.solvable_levels / self.total_levels,
            'avg_solution_length': (
                np.mean(self.solution_lengths) 
                if len(self.solution_lengths) > 0 else 0.0
            )
        }


# ============================================================================
# TESTING AND DEMO
# ============================================================================

def test_validator():
    """Test the validator with various invalid levels."""
    print("\n" + "="*70)
    print("TESTING SOKOBAN VALIDATOR")
    print("="*70)
    
    # Test 1: No player
    print("\nTest 1: No player")
    level = np.random.choice([0, 1, 3, 4], size=(10, 10))
    print_level_stats(level, "Before")
    fixed, corrections = validate_and_fix_sokoban(level, verbose=True)
    print_level_stats(fixed, "After")
    
    # Test 2: Multiple players
    print("\nTest 2: Multiple players")
    level = np.random.choice([0, 1, 2, 3, 4], size=(10, 10))
    print_level_stats(level, "Before")
    fixed, corrections = validate_and_fix_sokoban(level, verbose=True)
    print_level_stats(fixed, "After")
    
    # Test 3: Mismatched crates/targets
    print("\nTest 3: Mismatched crates/targets")
    level = np.zeros((10, 10), dtype=int)
    level[5, 5] = 2  # Player
    level[2, 2] = 3  # Crate
    level[3, 3] = 3  # Crate
    level[4, 4] = 3  # Crate
    level[7, 7] = 4  # Target
    print_level_stats(level, "Before")
    fixed, corrections = validate_and_fix_sokoban(level, verbose=True)
    print_level_stats(fixed, "After")
    
    # Test 4: Deadlocked crate (corner)
    print("\nTest 4: Deadlocked crate in corner")
    level = np.zeros((10, 10), dtype=int)
    level[0, :] = 1  # Top wall
    level[-1, :] = 1  # Bottom wall
    level[:, 0] = 1  # Left wall
    level[:, -1] = 1  # Right wall
    level[5, 5] = 2  # Player
    level[1, 1] = 3  # Deadlocked crate (corner)
    level[7, 7] = 4  # Target
    print_level_stats(level, "Before")
    fixed, corrections = validate_and_fix_sokoban(level, verbose=True)
    print_level_stats(fixed, "After")
    
    print("\n" + "="*70)
    print("✓ All validator tests completed")


if __name__ == '__main__':
    test_validator()
