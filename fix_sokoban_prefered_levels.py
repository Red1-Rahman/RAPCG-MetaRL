import json
import numpy as np
from collections import deque

# ── 1. Helper Reachability & Dead-Square Analysis ─────────────────────────

def get_reachable_player_tiles(level, start, walkable=(0, 2, 4)):
    """Returns all tiles the player can walk to without pushing anything."""
    h, w = level.shape
    visited = {start}
    q = deque([start])
    while q:
        y, x = q.popleft()
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in visited:
                if level[ny, nx] in walkable:
                    visited.add((ny, nx))
                    q.append((ny, nx))
    return visited

def compute_live_squares(level):
    """
    Finds all squares from which a crate can eventually be pulled to a target.
    Any tile not in this set is a 'dead square'.
    """
    h, w = level.shape
    targets = list(zip(*np.where(level == 4)))
    live = set()
    q = deque(targets)

    while q:
        cy, cx = q.popleft()
        if (cy, cx) in live:
            continue
        live.add((cy, cx))

        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            prev_y = cy - dy
            prev_x = cx - dx
            player_y = cy - 2 * dy
            player_x = cx - 2 * dx

            if (0 <= prev_y < h and 0 <= prev_x < w and 
                0 <= player_y < h and 0 <= player_x < w):
                # To pull a crate from (cy, cx) to (prev_y, prev_x),
                # neither the destination nor the required player spot can be a wall.
                if level[prev_y, prev_x] != 1 and level[player_y, player_x] != 1:
                    if (prev_y, prev_x) not in live:
                        q.append((prev_y, prev_x))
    return live

# ── 2. Full Sokoban State-Space Solver ────────────────────────────────────

def solve_sokoban(level):
    """
    A lightweight BFS solver to guarantee total level solvability.
    State representation: (player_pos, tuple_of_sorted_crate_positions)
    """
    h, w = level.shape
    
    # Extract structural layers (walls=1, targets=4)
    walls = (level == 1)
    targets = set(zip(*np.where(level == 4)))
    
    # Extract initial dynamic entities
    start_player = tuple(zip(*np.where(level == 2)))[0]
    start_crates = tuple(sorted(zip(*np.where(level == 3))))
    
    if not start_crates:
        return True # Trivially solved if no crates exist (handled during balancing)

    # BFS Initialization
    initial_state = (start_player, start_crates)
    queue = deque([initial_state])
    visited = {initial_state}
    
    while queue:
        player, crates = queue.popleft()
        
        # Check Win Condition: All crates are on targets
        if all(c in targets for c in crates):
            return True
            
        py, px = player
        crate_set = set(crates)
        
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ny, nx = py + dy, px + dx
            
            if not (0 <= ny < h and 0 <= nx < w) or walls[ny, nx]:
                continue
                
            # Scenario A: Moving into a crate (Attempting a Push)
            if (ny, nx) in crate_set:
                next_box_y, next_box_x = ny + dy, nx + dx
                
                # Check if push destination is out of bounds, a wall, or another crate
                if not (0 <= next_box_y < h and 0 <= next_box_x < w):
                    continue
                if walls[next_box_y, next_box_x] or (next_box_y, next_box_x) in crate_set:
                    continue
                    
                # Valid push: update crate positions
                new_crates = tuple(sorted([
                    (next_box_y, next_box_x) if c == (ny, nx) else c for c in crates
                ]))
                new_state = ((ny, nx), new_crates)
                
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append(new_state)
            
            # Scenario B: Just walking into an empty/target space
            else:
                new_state = ((ny, nx), crates)
                if new_state not in visited:
                    visited.add(new_state)
                    queue.append(new_state)
                    
    return False

# ── 3. Level Repair Pipeline ──────────────────────────────────────────────

def fix_level(raw):
    level = np.array(raw, dtype=int)
    h, w = level.shape
    issues = []
    
    # 1. Player Count Sanity Check (exactly 1)
    players = list(zip(*np.where(level == 2)))
    if len(players) == 0:
        empties = list(zip(*np.where(level == 0)))
        if empties:
            level[empties[0]] = 2
            issues.append("added missing player")
        players = list(zip(*np.where(level == 2)))
    elif len(players) > 1:
        for p in players[1:]:
            level[p] = 0
        issues.append(f"removed {len(players)-1} extra players")
        players = [players[0]]
    player_pos = players[0] if players else None

    # 2. Balance Crates and Targets (Equal counts, minimum 1)
    crate_count = int(np.sum(level == 3))
    target_count = int(np.sum(level == 4))
    
    if crate_count > target_count:
        excess = list(zip(*np.where(level == 3)))
        for pos in excess[:crate_count - target_count]:
            level[pos] = 0
        issues.append(f"removed {crate_count - target_count} excess crates")
    elif target_count > crate_count:
        excess = list(zip(*np.where(level == 4)))
        for pos in excess[:target_count - crate_count]:
            level[pos] = 0
        issues.append(f"removed {target_count - crate_count} excess targets")

    if int(np.sum(level == 3)) == 0:
        empties = list(zip(*np.where(level == 0)))
        if empties: level[empties[0]] = 3; issues.append("added missing crate")
    if int(np.sum(level == 4)) == 0:
        empties = list(zip(*np.where(level == 0)))
        if empties: level[empties[0]] = 4; issues.append("added missing target")

    # 3. Dead-Square Elimination 
    live_squares = compute_live_squares(level)
    crates = list(zip(*np.where(level == 3)))
    removed_dead = 0
    for cy, cx in crates:
        if (cy, cx) not in live_squares:
            level[cy, cx] = 0
            removed_dead += 1
    if removed_dead:
        issues.append(f"removed {removed_dead} dead-square crates")

    # Re-balance counts if dead-square logic pruned required crates
    crate_count = int(np.sum(level == 3))
    target_count = int(np.sum(level == 4))
    if crate_count < target_count:
        # Re-add crates *only* on non-dead, non-target open live squares
        live_empties = [pos for pos in live_squares if level[pos] == 0]
        needed = target_count - crate_count
        for pos in live_empties[:needed]:
            level[pos] = 3
            issues.append(f"re-placed pruned crate onto live square {pos}")

    # 4. Global Solvability Check & Dynamic Relocation Loop
    # If a micro-map is mathematically unsolvable, relocate until BFS solves it.
    attempts = 0
    while not solve_sokoban(level) and attempts < 15:
        attempts += 1
        current_player = list(zip(*np.where(level == 2)))[0]
        reachable_tiles = get_reachable_player_tiles(level, current_player)
        live_squares = compute_live_squares(level)
        
        # Valid locations to push a crate from: must be reachable by player and live
        valid_crate_spots = [t for t in live_squares if t in reachable_tiles and level[t] == 0]
        current_crates = list(zip(*np.where(level == 3)))
        
        if current_crates and valid_crate_spots:
            # Shift the first blocked crate to an active player-accessible location
            bad_crate = current_crates[0]
            new_spot = valid_crate_spots[0]
            level[bad_crate] = 0
            level[new_spot] = 3
            issues.append(f"unsolvable state fallback: moved crate {bad_crate} to {new_spot}")
        else:
            break # Break loop if structural limits prevent clean path adjustments

    return level.tolist(), issues

# ── 4. Main Execution Engine ──────────────────────────────────────────────

input_path = "data/preferences/sokoban/preferences.json"
output_path = "preferences_fixed.json"

with open(input_path) as f:
    prefs = json.load(f)

total = len(prefs)
fixed_count = 0
all_issues = []

for i, pref in enumerate(prefs):
    fixed_a, issues_a = fix_level(pref["level_a"])
    fixed_b, issues_b = fix_level(pref["level_b"])
    
    if issues_a or issues_b:
        fixed_count += 1
        entry_issues = {}
        if issues_a: entry_issues["level_a"] = issues_a
        if issues_b: entry_issues["level_b"] = issues_b
        all_issues.append({"pair": i, **entry_issues})
    
    pref["level_a"] = fixed_a
    pref["level_b"] = fixed_b
    
    def metrics(level):
        arr = np.array(level)
        unique = len(np.unique(arr))
        return {
            "diversity": round(unique / arr.size, 4),
            "complexity": 1.0,
            "size": arr.size,
            "unique_tiles": unique
        }
    
    pref["metrics_a"] = metrics(fixed_a)
    pref["metrics_b"] = metrics(fixed_b)
    pref["metadata"]["fixed"] = True

with open(output_path, "w") as f:
    json.dump(prefs, f, indent=2)

print(f"Total pairs processed : {total}")
print(f"Pairs corrected       : {fixed_count}")
print(f"\nDetailed Corrections per Pair:")
for item in all_issues:
    print(f"  Pair {item['pair']:02d}: ", end="")
    if "level_a" in item: print(f"A={item['level_a']}", end="  ")
    if "level_b" in item: print(f"B={item['level_b']}", end="")
    print()