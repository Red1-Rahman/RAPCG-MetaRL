"""
RAPCG-MetaRL Inference Script
Generate levels using trained Meta-RL agents.
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Add project paths - prioritize main project over gym-pcgrl
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, 'gym-pcgrl'))

from utils import ResourceMonitor
from wrappers.pcgrl_env import make_pcgrl_env
from wrappers.helper import calculate_content_metrics, save_level
from visualize_levels import save_level_image, render_level

try:
    from stable_baselines3 import PPO, A2C
except ImportError:
    print("Error: stable-baselines3 not installed")
    sys.exit(1)


class LevelGenerator:
    """
    Generate levels using trained Meta-RL model.
    """
    
    def __init__(self, model_path, game='zelda', representation='narrow', 
                 algorithm='PPO', device='auto'):
        """
        Initialize level generator.
        
        Args:
            model_path: Path to trained model
            game: Game environment
            representation: Representation type
            algorithm: RL algorithm used
            device: Device for inference
        """
        self.model_path = model_path
        self.game = game
        self.representation = representation
        self.algorithm = algorithm
        self.device = device
        
        # Create environment
        print(f"Creating {game} environment...")
        self.env = make_pcgrl_env(game=game, representation=representation)
        
        # Load model
        print(f"Loading model from {model_path}...")
        if algorithm == 'PPO':
            self.model = PPO.load(model_path, device=device)
        elif algorithm == 'A2C':
            self.model = A2C.load(model_path, device=device)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        print("✓ Generator ready")
    
    def generate(self, n_levels=1, max_steps=1000, deterministic=True, 
                 save_dir=None, visualize=True):
        """
        Generate levels.
        
        Args:
            n_levels: Number of levels to generate
            max_steps: Maximum steps per level
            deterministic: Use deterministic policy
            save_dir: Directory to save levels
            visualize: Visualize generated levels
            
        Returns:
            List of generated levels with metadata
        """
        generated_levels = []
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            print(f"Levels will be saved to: {save_dir}")
        
        for i in range(n_levels):
            print(f"\nGenerating level {i+1}/{n_levels}...")
            
            obs = self.env.reset()
            done = False
            steps = 0
            total_reward = 0
            
            while not done and steps < max_steps:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
            
            # Extract level from environment
            level = self._extract_level(info)
            
            # Calculate metrics
            metrics = calculate_content_metrics(level)
            metrics['total_reward'] = total_reward
            metrics['steps'] = steps
            
            # Store result
            result = {
                'level': level,
                'metrics': metrics,
                'info': info
            }
            generated_levels.append(result)
            
            # Print metrics
            print(f"  Steps: {steps}, Reward: {total_reward:.2f}")
            print(f"  Diversity: {metrics['diversity']:.3f}, "
                  f"Complexity: {metrics['complexity']:.3f}")
            
            # Save level
            if save_dir:
                level_path = os.path.join(save_dir, f'level_{i+1}')
                save_level(level, level_path + '.npy', format='npy')
                save_level(level, level_path + '.txt', format='txt')
                
                # Save visualization as high-res PNG
                img_path = level_path + '.png'
                save_level_image(level, img_path, game=self.game, 
                               scale=25, show_grid=True, dpi=300)
                print(f"  ✓ Saved to {level_path} (.npy, .txt, .png)")
            
            # Visualize
            if visualize:
                self._visualize_level(level, f"Generated Level {i+1}")
        
        return generated_levels
    
    def _extract_level(self, info):
        """Extract level array from environment info."""
        # This depends on the environment structure
        # Adjust based on actual gym-pcgrl info dict
        if 'level' in info:
            return np.array(info['level'])
        elif hasattr(self.env, 'get_level'):
            return self.env.get_level()
        else:
            # Fallback: return dummy level
            print("Warning: Could not extract level from environment")
            return np.zeros((10, 10), dtype=int)
    
    def _visualize_level(self, level, title="Level"):
        """Visualize level with proper tile colors."""
        rgb = render_level(level, game=self.game, scale=20, show_grid=True)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(rgb)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def close(self):
        """Close environment."""
        self.env.close()


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description='Generate levels using trained model')
    
    parser.add_argument('model_path', type=str,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--game', type=str, default='zelda',
                       help='Game environment')
    parser.add_argument('--representation', type=str, default='narrow',
                       help='Representation type')
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'A2C'],
                       help='RL algorithm')
    parser.add_argument('--n-levels', type=int, default=5,
                       help='Number of levels to generate')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per level')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy')
    parser.add_argument('--save-dir', type=str, default='generated_levels',
                       help='Directory to save levels')
    parser.add_argument('--no-visualize', action='store_true',
                       help='Disable visualization')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for inference')
    
    args = parser.parse_args()
    
    # Create generator
    generator = LevelGenerator(
        model_path=args.model_path,
        game=args.game,
        representation=args.representation,
        algorithm=args.algorithm,
        device=args.device
    )
    
    # Generate levels
    levels = generator.generate(
        n_levels=args.n_levels,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        save_dir=args.save_dir,
        visualize=not args.no_visualize
    )
    
    print(f"\n✓ Generated {len(levels)} levels")
    
    # Print summary statistics
    all_metrics = [l['metrics'] for l in levels]
    print(f"\nSummary Statistics:")
    print(f"  Mean Diversity: {np.mean([m['diversity'] for m in all_metrics]):.3f}")
    print(f"  Mean Complexity: {np.mean([m['complexity'] for m in all_metrics]):.3f}")
    print(f"  Mean Reward: {np.mean([m['total_reward'] for m in all_metrics]):.2f}")
    
    generator.close()


if __name__ == '__main__':
    main()
