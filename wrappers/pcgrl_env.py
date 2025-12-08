"""
Resource-Aware PCGRL Environment Wrapper
Wraps gym-pcgrl environments with dynamic resource adaptation.
"""
import sys
import os
import numpy as np

# gym-pcgrl uses old gym, so we need to import it directly
import gym

# Add gym-pcgrl to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'gym-pcgrl'))
import gym_pcgrl


class ResourceAwarePCGRLWrapper(gym.Wrapper):
    """
    Wrapper that adapts PCGRL environment complexity based on resource usage
    AND introduces resource-based reward shaping to make the agent truly resource-aware.
    """
    
    def __init__(self, env, resource_monitor, ram_penalty_weight=0.2, 
                 cpu_penalty_weight=0.1, gpu_penalty_weight=0.1,
                 max_complexity=10, min_complexity=3):
        """
        Args:
            env: Base gym-pcgrl environment
            resource_monitor: ResourceMonitor instance for tracking resource usage
            ram_penalty_weight: Weight for RAM usage penalty (higher = stronger penalty)
            cpu_penalty_weight: Weight for CPU usage penalty
            gpu_penalty_weight: Weight for GPU usage penalty
            max_complexity: Maximum environment complexity (e.g., max enemies)
            min_complexity: Minimum environment complexity
        """
        super().__init__(env)
        self.resource_monitor = resource_monitor
        self.ram_penalty_weight = ram_penalty_weight
        self.cpu_penalty_weight = cpu_penalty_weight
        self.gpu_penalty_weight = gpu_penalty_weight
        self.max_complexity = max_complexity
        self.min_complexity = min_complexity
        self.current_complexity = max_complexity
        
        # Track penalty statistics for debugging
        self.total_ram_penalty = 0.0
        self.total_cpu_penalty = 0.0
        self.total_gpu_penalty = 0.0
        self.penalty_steps = 0
        
    def adapt_complexity(self, cpu_usage, ram_usage, gpu_usage):
        """
        Dynamically adjust environment complexity based on resource usage.
        
        Args:
            cpu_usage: CPU usage percentage (0-100)
            ram_usage: RAM usage percentage (0-100)
            gpu_usage: GPU usage percentage (0-100)
        """
        # If any resource is critically high, reduce complexity
        if gpu_usage > 85 or ram_usage > 90 or cpu_usage > 90:
            self.current_complexity = max(
                self.min_complexity,
                self.current_complexity - 1
            )
        # If resources are comfortable, increase complexity
        elif gpu_usage < 60 and ram_usage < 70 and cpu_usage < 70:
            self.current_complexity = min(
                self.max_complexity,
                self.current_complexity + 1
            )
            
        return self.current_complexity
    
    def reset(self, **kwargs):
        """Reset the environment."""
        return self.env.reset(**kwargs)
    
    def step(self, action):
        """
        Step the environment with resource-aware reward shaping.
        
        This implements the feedback loop that makes the agent resource-aware:
        1. Monitor current resource usage (CPU, RAM, GPU)
        2. Adapt environment complexity dynamically
        3. Apply resource-based penalties to the reward
        4. Return modified reward that guides agent toward efficient generation
        """
        # 1. Get current resource usage
        resources = self.resource_monitor.get_resources()
        cpu_usage = resources['cpu_percent']
        ram_usage = resources['ram_percent']
        gpu_usage = resources['gpu_mem_percent']  # GPU memory usage
        
        # 2. ACTIVATE COMPLEXITY ADAPTATION (previously dormant)
        new_complexity = self.adapt_complexity(cpu_usage, ram_usage, gpu_usage)
        if new_complexity != self.current_complexity:
            self.current_complexity = new_complexity
            # Note: Real PCGRL env adaptation would require modifying internal parameters
            # For now, we track complexity for logging purposes
        
        # 3. Take the normal environment step
        obs, reward, done, info = self.env.step(action)
        
        # 4. IMPLEMENT RESOURCE REWARD SHAPING (The crucial fix!)
        # This creates the feedback loop that makes the agent resource-aware
        
        # RAM Penalty: Penalize if RAM usage exceeds 60%
        # Example: 80% RAM -> (80 - 60) * 0.2 = -4.0 penalty
        ram_penalty = 0.0
        if ram_usage > 78.0:
            ram_penalty = (ram_usage - 78.0) * self.ram_penalty_weight
        
        # CPU Penalty: Penalize if CPU usage exceeds 70%
        cpu_penalty = 0.0
        if cpu_usage > 70.0:
            cpu_penalty = (cpu_usage - 70.0) * self.cpu_penalty_weight
        
        # GPU Penalty: Penalize if GPU memory exceeds 70%
        gpu_penalty = 0.0
        if gpu_usage > 70.0:
            gpu_penalty = (gpu_usage - 70.0) * self.gpu_penalty_weight
        
        # Total resource penalty
        total_penalty = ram_penalty + cpu_penalty + gpu_penalty
        
        # Apply penalty to reward (this is what makes the agent learn!)
        shaped_reward = reward - total_penalty
        
        # Track statistics
        self.total_ram_penalty += ram_penalty
        self.total_cpu_penalty += cpu_penalty
        self.total_gpu_penalty += gpu_penalty
        self.penalty_steps += 1
        
        # Add debugging info
        if not isinstance(info, dict):
            info = {}
        
        info['raw_reward'] = reward
        info['shaped_reward'] = shaped_reward
        info['ram_penalty'] = ram_penalty
        info['cpu_penalty'] = cpu_penalty
        info['gpu_penalty'] = gpu_penalty
        info['total_penalty'] = total_penalty
        info['env_complexity'] = self.current_complexity
        info['ram_usage'] = ram_usage
        info['cpu_usage'] = cpu_usage
        info['gpu_usage'] = gpu_usage
        
        return obs, shaped_reward, done, info
    
    def get_penalty_stats(self):
        """Get accumulated penalty statistics."""
        if self.penalty_steps == 0:
            return {
                'avg_ram_penalty': 0.0,
                'avg_cpu_penalty': 0.0,
                'avg_gpu_penalty': 0.0,
                'total_steps': 0
            }
        
        return {
            'avg_ram_penalty': self.total_ram_penalty / self.penalty_steps,
            'avg_cpu_penalty': self.total_cpu_penalty / self.penalty_steps,
            'avg_gpu_penalty': self.total_gpu_penalty / self.penalty_steps,
            'total_steps': self.penalty_steps
        }


def make_pcgrl_env(resource_monitor, game='zelda', representation='narrow', 
                   ram_penalty_weight=0.2, cpu_penalty_weight=0.1, 
                   gpu_penalty_weight=0.1, crop_size=None, **kwargs):
    """
    Create a resource-aware PCGRL environment.
    
    Args:
        resource_monitor: ResourceMonitor instance for tracking resource usage
        game: Game name ('zelda', 'sokoban', 'binary', etc.)
        representation: Representation type ('narrow', 'wide', 'turtle')
        ram_penalty_weight: Weight for RAM penalty (higher = agent cares more about RAM)
        cpu_penalty_weight: Weight for CPU penalty
        gpu_penalty_weight: Weight for GPU penalty
        crop_size: Crop size for observation
        **kwargs: Additional arguments for the environment
        
    Returns:
        Gym environment with resource-aware reward shaping
    """
    # gym-pcgrl registers environments as '{game}-{representation}-v0'
    game_name = game.lower()
    rep_name = representation.lower()
    env_name = f'{game_name}-{rep_name}-v0'
    
    try:
        # Create environment using registered name
        env = gym.make(env_name)
        
        # Wrap with resource-aware wrapper (passing the monitor for feedback loop)
        env = ResourceAwarePCGRLWrapper(
            env, 
            resource_monitor=resource_monitor,
            ram_penalty_weight=ram_penalty_weight,
            cpu_penalty_weight=cpu_penalty_weight,
            gpu_penalty_weight=gpu_penalty_weight
        )
        
        return env
        
    except Exception as e:
        print(f"Error creating environment '{env_name}': {e}")
        print(f"\nAvailable games: zelda, sokoban, binary")
        print(f"Available representations: narrow, wide, turtle")
        print(f"Environment should be registered as: {env_name}")
        raise


def test_environment():
    """Test function to verify environment creation."""
    print("Testing PCGRL Environment Creation...")
    
    try:
        # Test Zelda environment
        env = make_pcgrl_env('zelda', 'narrow')
        print(f"✓ Created Zelda environment")
        print(f"  Observation space: {env.observation_space}")
        print(f"  Action space: {env.action_space}")
        
        # Test reset
        state = env.reset()
        print(f"✓ Environment reset successful")
        print(f"  State shape: {state.shape if hasattr(state, 'shape') else type(state)}")
        
        # Test step
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        print(f"✓ Environment step successful")
        print(f"  Reward: {reward}")
        
        env.close()
        print("\n✓ All environment tests passed!")
        return True
        
    except Exception as e:
        print(f"\n✗ Environment test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_environment()
