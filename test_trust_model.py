"""
Quick test: Regenerate levels trusting the model (no aggressive validation)
This tests if the trained model actually generates crates/targets
"""
import numpy as np
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, 'gym-pcgrl'))

from utils import ResourceMonitor
from wrappers.pcgrl_env import make_pcgrl_env
from stable_baselines3 import PPO

print("="*70)
print("TESTING: Model Output WITHOUT Aggressive Validation")
print("="*70)

# Setup
resource_monitor = ResourceMonitor(use_gpu=False)
env = make_pcgrl_env(
    resource_monitor=resource_monitor,
    game='sokoban',
    representation='narrow',
    sokoban_unsolvable_penalty=25.0
)

# Load model
model_path = 'checkpoints/sokoban_PPO_20260130_162717/best_model.zip'
print(f"\nLoading model: {model_path}")
model = PPO.load(model_path, env=env)

# Generate one level
print("\nGenerating 1 test level...")
obs = env.reset()
done = False
steps = 0

while not done and steps < 1000:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    steps += 1

# Extract level WITHOUT validation
print(f"Generated in {steps} steps")

# Get level from info
if 'map' in info:
    level = np.array(info['map'])
elif hasattr(env.unwrapped, '_rep') and hasattr(env.unwrapped._rep, '_map'):
    level = np.array(env.unwrapped._rep._map)
else:
    print("ERROR: Could not extract level")
    sys.exit(1)

# Check counts
print("\n" + "="*70)
print("RAW MODEL OUTPUT (no validation):")
print("="*70)
print(f"Empty (0): {np.sum(level == 0)}")
print(f"Walls (1): {np.sum(level == 1)}")
print(f"Player (2): {np.sum(level == 2)}")
print(f"Crates (3): {np.sum(level == 3)}")  # ← KEY METRIC
print(f"Targets (4): {np.sum(level == 4)}")  # ← KEY METRIC

print("\nLevel array:")
print(level)

# Check stats from environment (includes solver results)
if 'stats' in info:
    stats = info['stats']
    print("\nEnvironment stats:")
    print(f"  Player count: {stats.get('player', 'N/A')}")
    print(f"  Crate count: {stats.get('crate', 'N/A')}")
    print(f"  Target count: {stats.get('target', 'N/A')}")
    print(f"  Regions: {stats.get('regions', 'N/A')}")
    print(f"  Distance to win: {stats.get('dist-win', 'N/A')}")
    print(f"  Solution length: {len(stats.get('solution', []))}")

env.close()

print("\n" + "="*70)
if np.sum(level == 3) > 0 and np.sum(level == 4) > 0:
    print("✅ SUCCESS: Model generates crates and targets!")
    print("   The issue was aggressive validation removing them.")
    print("   Use --trust-model flag during inference.")
else:
    print("❌ PROBLEM: Model doesn't generate crates/targets")
    print("   May need to retrain with proper solvability wrapper.")
print("="*70)
