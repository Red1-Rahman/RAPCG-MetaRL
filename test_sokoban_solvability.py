"""
Test script to verify Sokoban solvability wrapper works correctly.
"""
import sys
import os

project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, 'gym-pcgrl'))

from utils import ResourceMonitor
from wrappers.pcgrl_env import make_pcgrl_env


def test_sokoban_solvability():
    """Test that the solvability wrapper is working."""
    print("Testing Sokoban Solvability Wrapper...")
    print("=" * 60)
    
    # Create resource monitor
    resource_monitor = ResourceMonitor(use_gpu=False)
    
    # Create Sokoban environment with solvability wrapper
    env = make_pcgrl_env(
        resource_monitor=resource_monitor,
        game='sokoban',
        representation='narrow',
        sokoban_unsolvable_penalty=15.0
    )
    
    print("✓ Environment created successfully")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Test a few episodes
    print("\nTesting 3 episodes...")
    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        obs = env.reset()
        done = False
        step = 0
        total_reward = 0
        
        while not done and step < 100:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Check if info contains solvability information
            if 'solvable' in info:
                solvable = info['solvable']
                sol_len = info.get('solution_length', 0)
                print(f"  Step {step}: Solvable={solvable}, "
                      f"Solution Length={sol_len}, Reward={reward:.2f}")
                if done:
                    break
        
        print(f"  Episode finished: {step} steps, Total reward: {total_reward:.2f}")
    
    # Get wrapper statistics if available
    print("\n" + "=" * 60)
    print("Wrapper Statistics:")
    
    # Unwrap to find SokobanSolvabilityWrapper
    current_env = env
    while hasattr(current_env, 'env'):
        if hasattr(current_env, 'get_statistics'):
            stats = current_env.get_statistics()
            print(f"  Total levels checked: {stats['total_levels']}")
            print(f"  Solvable levels: {stats['solvable_levels']}")
            print(f"  Unsolvable levels: {stats['unsolvable_levels']}")
            if stats['total_levels'] > 0:
                print(f"  Solvability rate: {stats['solvability_rate']*100:.1f}%")
                if stats['avg_solution_length'] > 0:
                    print(f"  Avg solution length: {stats['avg_solution_length']:.1f}")
            break
        current_env = current_env.env
    
    env.close()
    print("\n✓ Test completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    test_sokoban_solvability()
