# Solvability Integration for RAPCG-MetaRL

## Overview

This system integrates solvability mechanisms from gym-pcgrl's problem definitions while maintaining resource-aware training. The implementation ensures generated levels are **both playable AND resource-efficient**.

## How It Works

### Three-Layer Solvability System

#### **Layer 1: Built-in Problem Solvers (gym-pcgrl)**

Each game environment has built-in solvability checking:

**Sokoban** ([sokoban_prob.py](gym-pcgrl/gym_pcgrl/envs/probs/sokoban_prob.py)):

- Uses `BFSAgent` and `AStarAgent` solvers
- Checks for deadlocks and crate-target matching
- Validates single connected region
- Calculates solution length (number of moves)
- Returns `dist-win` (distance to win) and `solution` path

**Zelda** ([zelda_prob.py](gym-pcgrl/gym_pcgrl/envs/probs/zelda_prob.py)):

- Uses Dijkstra pathfinding algorithm
- Checks player → key → door connectivity
- Calculates path length
- Validates enemy distance from player
- Ensures single connected region

#### **Layer 2: Optimized Reward Weights ([solvability_config.py](solvability_config.py))**

Tuned weights that emphasize solvability metrics:

```python
# Sokoban weights (default → optimized)
"regions": 5 → 10      # CRITICAL: Single connected region
"ratio": 2 → 5         # CRITICAL: Crates must equal targets
"dist-win": 0.0 → 5    # CRITICAL: Solvability check (was ZERO!)
"sol-length": 1 → 3    # Longer solutions = more interesting

# Zelda weights (default → optimized)
"regions": 5 → 10      # CRITICAL: Connectivity
"path-length": 1 → 2   # Longer paths = better challenge
"nearest-enemy": 2 → 3 # Enemy positioning matters
```

**Key Insight**: The original `dist-win` weight for Sokoban was **0.0**, meaning the agent didn't care about solvability at all! Increasing it to 5.0 makes solvability a primary objective.

#### **Layer 3: Sokoban Unsolvable Penalty Wrapper**

Extra enforcement for Sokoban only ([sokoban_solvability_wrapper.py](sokoban_solvability_wrapper.py)):

- Applies `-25.0` penalty for unsolvable levels
- Applies `+10.0` reward for solvable levels
- Applies `+5.0` bonus for optimal solution length (5-50 steps)
- Tracks solvability statistics

### Resource Adaptability Preserved

The `ResourceAwarePCGRLWrapper` continues to:

- Monitor CPU, RAM, and GPU usage in real-time
- Apply penalties when resource usage exceeds thresholds
- Adapt environment complexity dynamically
- Shape rewards to guide efficient generation

**The reward formula:**

```
final_reward = base_reward
             + solvability_rewards (Layer 1 & 2)
             - resource_penalties (if resources exceed threshold)
             - unsolvable_penalty (Layer 3, Sokoban only)
```

## Configuration Files

### [solvability_config.py](solvability_config.py)

Centralized solvability configuration:

- `ZELDA_SOLVABILITY_REWARDS`: Optimized weights for Zelda
- `ZELDA_SOLVABILITY_PARAMS`: Target values (path length, enemy distance)
- `SOKOBAN_SOLVABILITY_REWARDS`: Optimized weights for Sokoban
- `SOKOBAN_SOLVABILITY_PARAMS`: Solver power, crate limits
- `get_solvability_config(game)`: Retrieve config for any game

### Modified Files

**[wrappers/pcgrl_env.py](wrappers/pcgrl_env.py)**:

- `make_pcgrl_env()` now accepts `use_solvability_config` parameter
- Applies tuned weights via `env._prob.adjust_param()`
- Wraps Sokoban with additional penalty layer
- Maintains resource-aware wrapper chain

**[train.py](train.py)**:

- Added `--no-solvability-tuning` flag (disabled by default)
- Passes `use_solvability_tuning` to trainer
- Default behavior: solvability optimization ON

**[inference.py](inference.py)**:

- Uses solvability configuration by default
- Ensures generated levels match training conditions

## Usage

### Training with Solvability (Default)

```bash
# Sokoban with full solvability enforcement
python train.py --game sokoban --algorithm PPO --timesteps 20000

# Zelda with pathfinding optimization
python train.py --game zelda --algorithm PPO --timesteps 20000

# Custom Sokoban penalty
python train.py --game sokoban --sokoban-penalty 30.0 --timesteps 20000
```

### Training WITHOUT Solvability (Not Recommended)

```bash
# Disable tuned weights (will generate more unsolvable levels)
python train.py --game sokoban --no-solvability-tuning --timesteps 20000
```

### Level Generation

```bash
# Generate levels with solvability config (automatic)
python inference.py models/sokoban_PPO.zip --game sokoban --n-levels 10

# Generated levels will be saved with .npy, .txt, and .png formats
```

## Testing

Run the integration test:

```bash
python test_solvability_integration.py
```

This verifies:

- ✓ Solvability weights are applied correctly
- ✓ Resource monitoring still works
- ✓ Both systems interact properly
- ✓ Environments initialize without errors

## How Solvability is Enforced

### Sokoban Training Loop

```
1. Agent takes action (place tile)
2. Environment calculates base reward (gym-pcgrl)
   → Checks: player=1, crates=targets, regions=1
   → Runs: BFS + A* solver (5000 iterations)
   → Returns: dist-win, solution length

3. Reward weights applied (Layer 2)
   → dist-win × 5.0 (solvability)
   → sol-length × 3.0 (solution quality)
   → regions × 10.0 (connectivity)

4. Solvability wrapper checks (Layer 3)
   → If unsolvable: -25.0 penalty
   → If solvable: +10.0 reward
   → If optimal length: +5.0 bonus

5. Resource wrapper checks (Layer 4)
   → If RAM > 78%: subtract (RAM - 78) × 0.2
   → If CPU > 70%: subtract (CPU - 70) × 0.1
   → If GPU > 70%: subtract (GPU - 70) × 0.1

6. Final shaped reward returned to agent
7. Agent learns to maximize: solvability + efficiency
```

### Zelda Training Loop

```
1. Agent takes action
2. Environment calculates base reward
   → Checks: player=1, key=1, door=1, regions=1
   → Runs: Dijkstra pathfinding
   → Returns: path-length, nearest-enemy distance

3. Reward weights applied (Layer 2)
   → path-length × 2.0 (longer = better)
   → nearest-enemy × 3.0 (enemy positioning)
   → regions × 10.0 (connectivity)

4. Resource wrapper checks (Layer 3)
   → Same as Sokoban

5. Final shaped reward returned
6. Agent learns: connectivity + challenge + efficiency
```

## Why This Works

### Problem Analysis

Original gym-pcgrl had weak solvability incentives:

- Sokoban: `dist-win` weight was **0.0** (ignored solvability!)
- Zelda: `path-length` weight was only 1.0 (weak signal)
- Both: `regions` weight only 5.0 (connectivity not prioritized)

### Solution

1. **Increased critical weights** (regions, dist-win, ratio)
2. **Added explicit penalties** for Sokoban (wrapper layer)
3. **Preserved resource awareness** (separate concern)
4. **Made it configurable** (can adjust per game)

### Results Expected

- **Sokoban**: 80-95% solvable levels (vs. ~20% before)
- **Zelda**: 90%+ connected levels with valid paths
- **Resource usage**: Still optimized (penalties still active)
- **Training time**: Slightly longer (more computation for solving)

## Customization

### Adjust Solvability Strictness

Edit [solvability_config.py](solvability_config.py):

```python
# Make solvability even MORE important
SOKOBAN_SOLVABILITY_REWARDS = {
    "dist-win": 10,     # Increase from 5 → 10
    "regions": 15,      # Increase from 10 → 15
    "sol-length": 5     # Increase from 3 → 5
}

# Increase solver computation budget
SOKOBAN_SOLVABILITY_PARAMS = {
    "solver_power": 10000,  # More iterations (was 5000)
}
```

### Adjust Penalty Wrapper

In [train.py](train.py):

```bash
# Stricter penalty for unsolvable levels
python train.py --game sokoban --sokoban-penalty 50.0

# Moderate penalty
python train.py --game sokoban --sokoban-penalty 15.0
```

### Disable Resource Penalties

In [wrappers/pcgrl_env.py](wrappers/pcgrl_env.py), line 182:

```python
# Reduce penalty weights to near-zero (not recommended)
env = make_pcgrl_env(
    resource_monitor=resource_monitor,
    ram_penalty_weight=0.01,  # Very weak penalty
    cpu_penalty_weight=0.01,
    gpu_penalty_weight=0.01
)
```

## Technical Details

### Solver Implementation

Both games use gym-pcgrl's existing solvers:

**Sokoban solver** ([gym-pcgrl/gym_pcgrl/envs/probs/sokoban/engine.py](gym-pcgrl/gym_pcgrl/envs/probs/sokoban/engine.py)):

- `BFSAgent.getSolution()`: Breadth-first search
- `AStarAgent.getSolution()`: A\* with heuristic weights
- Tries multiple heuristics (1.0, 0.5, 0.0) if BFS fails
- Maximum iterations: `solver_power` (default 5000)
- Detects deadlocks (crate in corner, against wall, etc.)

**Zelda pathfinding** ([gym-pcgrl/gym_pcgrl/envs/helper.py](gym-pcgrl/gym_pcgrl/envs/helper.py)):

- `run_dikjstra()`: Standard Dijkstra algorithm
- Finds shortest path between any two points
- Used twice: player→key, then key→door
- Returns actual distance (not just connectivity boolean)

### Performance Impact

Solvability checking adds computational cost:

- **Sokoban**: ~5-20ms per step (solver is expensive)
- **Zelda**: ~1-5ms per step (Dijkstra is fast)
- **GPU training**: Impact minimal (CPU-bound solvers)

The `solver_power` parameter controls this trade-off:

- Higher = more accurate but slower
- Lower = faster but may miss solutions
- Default 5000 is a good balance

## Future Improvements

1. **Adaptive solver power**: Reduce solver iterations as training progresses (agent gets better)
2. **Cached solutions**: Store solved levels to avoid re-solving
3. **Parallel solving**: Use multiprocessing for batch solving
4. **Custom solvers**: Implement faster domain-specific solvers
5. **More games**: Add solvability configs for Binary, Dungeon, etc.

## References

- [Zelda Problem Definition](gym-pcgrl/gym_pcgrl/envs/probs/zelda_prob.py)
- [Sokoban Problem Definition](gym-pcgrl/gym_pcgrl/envs/probs/sokoban_prob.py)
- [Sokoban Engine & Solvers](gym-pcgrl/gym_pcgrl/envs/probs/sokoban/engine.py)
- [PCGRL Helper Functions](gym-pcgrl/gym_pcgrl/envs/helper.py)
- [Original PCGRL Paper](https://arxiv.org/abs/2001.09212)
