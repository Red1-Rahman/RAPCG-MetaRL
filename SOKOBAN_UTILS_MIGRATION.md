# Sokoban Utilities - Unified Module

## Overview

The Sokoban-related code has been consolidated into a single, well-organized module: **`sokoban_utils.py`**

### Previous Structure (3 separate files):

1. ❌ `sokoban_validator.py` - Level validation and correction
2. ❌ `sokoban_solvability_wrapper.py` - Training-time gym wrapper
3. ✅ `solvability_config.py` - Configuration (kept separate, used by multiple games)

### New Structure (1 unified file):

1. ✅ **`sokoban_utils.py`** - All Sokoban-specific utilities
2. ✅ `solvability_config.py` - Configuration (unchanged, shared across games)

## Benefits of Consolidation

### 1. **Single Import Point**

```python
# Before (scattered imports)
from sokoban_validator import validate_and_fix_sokoban
from sokoban_solvability_wrapper import SokobanSolvabilityWrapper

# After (unified import)
from sokoban_utils import validate_and_fix_sokoban, SokobanSolvabilityWrapper
```

### 2. **Shared Code**

- Deadlock detection logic is now shared between validator and wrapper
- No code duplication
- Easier to maintain and extend

### 3. **Better Organization**

The file is organized into clear sections:

- **Deadlock Detection** (lines 20-80)
- **Level Validation** (lines 85-250)
- **Gym Wrapper** (lines 255-390)
- **Testing** (lines 395-end)

### 4. **Logical Separation of Concerns**

```
sokoban_utils.py
├── Deadlock Detection
│   ├── check_sokoban_deadlock()      # Check single crate
│   └── remove_deadlocked_crates()    # Fix entire level
│
├── Validation & Correction (Post-processing)
│   ├── is_valid_sokoban()            # Check validity
│   ├── validate_and_fix_sokoban()    # Fix all issues
│   └── print_level_stats()           # Debug output
│
└── Solvability Wrapper (Training-time)
    └── SokobanSolvabilityWrapper      # Gym wrapper class
```

## Why `solvability_config.py` Stays Separate

Configuration is kept separate because:

1. **Multi-game**: Contains config for both Zelda AND Sokoban
2. **Used by multiple modules**: train.py, inference.py, wrappers/pcgrl_env.py
3. **Configuration vs Logic**: Pure data vs executable code
4. **Easy modification**: Users can adjust weights without touching logic

## Module Contents

### Deadlock Detection

```python
def check_sokoban_deadlock(level: np.ndarray, crate_pos: tuple) -> bool
```

- Checks if a crate is in an unsolvable position (corner, wall edge)
- Used by both validator and wrapper

```python
def remove_deadlocked_crates(level: np.ndarray) -> Tuple[np.ndarray, int]
```

- Removes all deadlocked crates from level
- Returns corrected level and count removed

### Validation & Correction

```python
def is_valid_sokoban(level: np.ndarray) -> Tuple[bool, str]
```

- Checks if level meets all Sokoban constraints
- Returns (is_valid, error_message)

```python
def validate_and_fix_sokoban(level: np.ndarray, min_crates: int = 2,
                              verbose: bool = False) -> Tuple[np.ndarray, Dict]
```

- **Main validation function**
- Fixes:
  1. Ensures exactly 1 player (placed near center)
  2. Removes deadlocked crates
  3. Balances crates and targets (minimum `min_crates` pairs)
  4. Ensures at least 1 goal exists
- Returns corrected level and corrections dictionary
- Used in **inference** to fix generated levels

```python
def print_level_stats(level: np.ndarray, title: str = "Level Stats")
```

- Prints detailed level statistics for debugging

### Gym Wrapper

```python
class SokobanSolvabilityWrapper(gym.Wrapper)
```

- **Training-time enforcement**
- Wraps PCGRL environment during training
- Checks solvability using gym-pcgrl's built-in solver
- Applies penalties/rewards:
  - `-25.0` penalty for unsolvable levels
  - `+10.0` reward for solvable levels
  - `+5.0` bonus for optimal solution length
- Tracks solvability statistics
- Used in **training** to teach agent to generate solvable levels

## Usage Examples

### During Training (wrappers/pcgrl_env.py)

```python
from sokoban_utils import SokobanSolvabilityWrapper

# Wrap environment during training
env = gym.make('sokoban-narrow-v0')
env = SokobanSolvabilityWrapper(
    env,
    unsolvable_penalty=25.0,
    solvable_reward=10.0,
    solution_bonus=5.0
)
```

### During Inference (inference_timed.py)

```python
from sokoban_utils import validate_and_fix_sokoban, is_valid_sokoban

# Generate level
level = generate_level()

# Validate and fix
is_valid, msg = is_valid_sokoban(level)
if not is_valid:
    level, corrections = validate_and_fix_sokoban(level, min_crates=2, verbose=True)
    print(f"Fixed: {corrections}")
```

### Standalone Testing

```bash
# Run built-in tests
python sokoban_utils.py
```

## Migration Guide

### Old Code

```python
# Multiple imports
from sokoban_validator import validate_and_fix_sokoban, is_valid_sokoban
from sokoban_solvability_wrapper import SokobanSolvabilityWrapper
from solvability_config import get_solvability_config

# Use functions...
```

### New Code

```python
# Single import
from sokoban_utils import (
    validate_and_fix_sokoban,
    is_valid_sokoban,
    SokobanSolvabilityWrapper,
    check_sokoban_deadlock,
    remove_deadlocked_crates
)
from solvability_config import get_solvability_config  # Still separate

# Use functions... (same API)
```

## File Status

### Active Files

- ✅ **sokoban_utils.py** (NEW, unified)
- ✅ **solvability_config.py** (unchanged)

### Deprecated Files (can be deleted)

- ❌ sokoban_validator.py → merged into sokoban_utils.py
- ❌ sokoban_solvability_wrapper.py → merged into sokoban_utils.py

### Updated Files (imports changed)

- ✅ wrappers/pcgrl_env.py
- ✅ inference_timed.py

## Testing

The unified module includes comprehensive tests:

```bash
python sokoban_utils.py
```

Tests validate:

- ✅ Player fixing (0 players, multiple players)
- ✅ Deadlock removal (corner crates, wall edges)
- ✅ Crate/target balancing (mismatched counts)
- ✅ Minimum goal enforcement (at least 1 pair)

## Advantages Summary

| Aspect                | Before (3 files)          | After (1 file)           |
| --------------------- | ------------------------- | ------------------------ |
| **Lines of code**     | 450 lines total           | 400 lines (deduplicated) |
| **Import statements** | 2 imports needed          | 1 import needed          |
| **Code duplication**  | Deadlock logic duplicated | Shared functions         |
| **Maintenance**       | Update 2 files            | Update 1 file            |
| **Navigation**        | Jump between files        | Single file navigation   |
| **Testing**           | Separate test files       | Integrated tests         |
| **Organization**      | Scattered                 | Logically grouped        |

## Conclusion

The consolidation into `sokoban_utils.py` provides:

- ✅ **Better organization** - All Sokoban logic in one place
- ✅ **Less duplication** - Shared deadlock detection
- ✅ **Easier maintenance** - Single file to update
- ✅ **Clearer purpose** - Validation vs Training wrapper clearly separated
- ✅ **Same API** - No breaking changes to existing code

The configuration file (`solvability_config.py`) remains separate as it serves multiple games and is pure configuration data.
