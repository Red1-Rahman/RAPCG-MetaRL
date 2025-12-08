"""
Test Script for RAPCG-MetaRL
Test environment setup, resource monitoring, and basic functionality.
"""
import os
import sys
import numpy as np

# Add project paths - prioritize project root over gym-pcgrl
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)  # Project root first for utils.py
sys.path.append(os.path.join(project_root, 'gym-pcgrl'))  # gym-pcgrl last to avoid conflicts

from utils import ResourceMonitor, TrainingLogger
from wrappers.pcgrl_env import make_pcgrl_env
from wrappers.helper import (
    parse_vglc_level, load_vglc_levels, 
    calculate_content_metrics, save_level, load_level
)


def test_resource_monitor():
    """Test resource monitoring."""
    print("\n" + "="*60)
    print("Testing Resource Monitor")
    print("="*60)
    
    try:
        monitor = ResourceMonitor(use_gpu=True)
        resources = monitor.get_resources()
        
        print("Current Resource Usage:")
        print(f"  CPU: {resources['cpu_percent']:.1f}%")
        print(f"  RAM: {resources['ram_percent']:.1f}% "
              f"({resources['ram_used_gb']:.2f}GB / "
              f"{resources['ram_used_gb'] + resources['ram_available_gb']:.2f}GB)")
        
        if resources['gpu_mem_total_mb'] > 0:
            print(f"  GPU Util: {resources['gpu_util_percent']:.1f}%")
            print(f"  GPU Mem: {resources['gpu_mem_percent']:.1f}% "
                  f"({resources['gpu_mem_used_mb']:.0f}MB / "
                  f"{resources['gpu_mem_total_mb']:.0f}MB)")
        else:
            print("  GPU: Not available")
        
        # Test pressure check
        is_pressure, msg = monitor.check_resource_pressure()
        print(f"\nResource Pressure: {msg}")
        
        print("✓ Resource Monitor test passed")
        return True
        
    except Exception as e:
        print(f"✗ Resource Monitor test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logger():
    """Test training logger."""
    print("\n" + "="*60)
    print("Testing Training Logger")
    print("="*60)
    
    try:
        logger = TrainingLogger(log_dir='test/logs', experiment_name='test_exp')
        
        # Log some dummy data
        for i in range(100):
            reward = np.random.randn()
            resources = {
                'cpu_percent': 50 + np.random.randn() * 10,
                'ram_percent': 60 + np.random.randn() * 5,
                'gpu_util_percent': 70 + np.random.randn() * 10,
                'gpu_mem_percent': 65 + np.random.randn() * 5,
            }
            content_metrics = {
                'diversity': np.random.rand(),
                'complexity': np.random.rand(),
            }
            
            logger.log_step(reward, resources, content_metrics)
            
            if i % 20 == 0:
                logger.log_episode_end()
        
        # Save logs
        logger.save()
        
        # Print stats
        stats = logger.get_stats()
        print(f"Total Episodes: {stats['total_episodes']}")
        print(f"Total Steps: {stats['total_steps']}")
        print(f"Mean Reward: {stats['mean_reward']:.3f}")
        
        print("✓ Logger test passed")
        return True
        
    except Exception as e:
        print(f"✗ Logger test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vglc_parsing():
    """Test VGLC level parsing."""
    print("\n" + "="*60)
    print("Testing VGLC Level Parsing")
    print("="*60)
    
    try:
        data_dir = os.path.join(project_root, 'data')
        
        # Check if data files exist
        smb_file = os.path.join(data_dir, 'SMB.json')
        zelda_file = os.path.join(data_dir, 'zelda.json')
        
        if os.path.exists(smb_file):
            print(f"Loading SMB levels from {smb_file}...")
            smb_levels = load_vglc_levels(data_dir, 'SMB')
            print(f"  Loaded {len(smb_levels)} SMB levels")
            
            if smb_levels:
                print(f"  First level shape: {smb_levels[0].shape}")
        else:
            print(f"  SMB data not found at {smb_file}")
        
        if os.path.exists(zelda_file):
            print(f"Loading Zelda levels from {zelda_file}...")
            zelda_levels = load_vglc_levels(data_dir, 'zelda')
            print(f"  Loaded {len(zelda_levels)} Zelda levels")
            
            if zelda_levels:
                print(f"  First level shape: {zelda_levels[0].shape}")
        else:
            print(f"  Zelda data not found at {zelda_file}")
        
        print("✓ VGLC parsing test passed")
        return True
        
    except Exception as e:
        print(f"✗ VGLC parsing test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_content_metrics():
    """Test content quality metrics."""
    print("\n" + "="*60)
    print("Testing Content Metrics")
    print("="*60)
    
    try:
        # Create dummy level
        level = np.random.randint(0, 5, size=(10, 20))
        
        metrics = calculate_content_metrics(level)
        
        print("Metrics for random level:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
        
        # Test save/load
        test_dir = 'test/levels'
        os.makedirs(test_dir, exist_ok=True)
        
        test_path = os.path.join(test_dir, 'test_level')
        save_level(level, test_path + '.npy', format='npy')
        loaded_level = load_level(test_path + '.npy')
        
        if np.array_equal(level, loaded_level):
            print("✓ Level save/load successful")
        else:
            print("✗ Level save/load mismatch")
            return False
        
        print("✓ Content metrics test passed")
        return True
        
    except Exception as e:
        print(f"✗ Content metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_environment():
    """Test PCGRL environment creation."""
    print("\n" + "="*60)
    print("Testing PCGRL Environment")
    print("="*60)
    
    try:
        # Import gym_pcgrl to verify installation
        try:
            import gym_pcgrl
            print("✓ gym_pcgrl imported successfully")
        except ImportError as e:
            print(f"✗ gym_pcgrl import failed: {e}")
            print("  Install with: cd gym-pcgrl && pip install -e .")
            return False
        
        # List available environments
        print("Available PCGRL environments:")
        import gym
        # gym.envs.registry is a dict in gym 0.26.2
        if hasattr(gym.envs.registry, 'all'):
            envs = [env.id for env in gym.envs.registry.all() if 'pcgrl' in env.id.lower()]
        else:
            # For newer gym versions where registry is a dict
            envs = [env_id for env_id in gym.envs.registry.keys() if 'pcgrl' in env_id.lower()]
        
        for env_id in envs:
            print(f"  - {env_id}")
        
        if not envs:
            print("  No PCGRL environments found")
            print("  Make sure gym-pcgrl is properly installed")
            return False
        
        # Try to create an environment
        print("\nTesting environment creation...")
        env = make_pcgrl_env('zelda', 'narrow')
        
        print(f"Environment created successfully")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Test reset
        obs = env.reset()
        print(f"✓ Reset successful")
        
        # Test step
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print(f"✓ Step successful (reward: {reward})")
        
        env.close()
        
        print("✓ Environment test passed")
        return True
        
    except Exception as e:
        print(f"✗ Environment test failed: {e}")
        print("\nThis might be because gym-pcgrl is not properly set up.")
        print("Make sure to install gym-pcgrl dependencies:")
        print("  cd gym-pcgrl")
        print("  pip install -e .")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("RAPCG-MetaRL Test Suite")
    print("="*60)
    
    tests = [
        ("Resource Monitor", test_resource_monitor),
        ("Training Logger", test_logger),
        ("VGLC Parsing", test_vglc_parsing),
        ("Content Metrics", test_content_metrics),
        ("PCGRL Environment", test_environment),
    ]
    
    results = []
    for name, test_func in tests:
        result = test_func()
        results.append((name, result))
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*60)
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
