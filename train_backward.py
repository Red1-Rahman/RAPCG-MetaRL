"""
Train Sokoban using BACKWARD generation (reverse from solved state).

This approach guarantees solvability by generating levels backward from a solved state.

Usage:
    python train_backward.py --game sokoban --timesteps 50000
"""

import os
import argparse
from datetime import datetime
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
import gym

# Import custom environments
import gym_pcgrl
from wrappers.pcgrl_env import make_pcgrl_env
from utils import ResourceMonitor


def train_backward_sokoban(
    game: str = 'sokoban',
    representation: str = 'narrow',
    timesteps: int = 50000,
    checkpoint_freq: int = 5000,
    eval_freq: int = 2000,
    n_eval_episodes: int = 10,
    learning_rate: float = 0.0001,
    batch_size: int = 256,
    n_steps: int = 2048,
    device: str = 'auto'
):
    """
    Train backward Sokoban generator.
    
    Args:
        game: Game name (default: 'sokoban')
        representation: Representation type (narrow/wide)
        timesteps: Total training timesteps
        checkpoint_freq: Save checkpoint every N steps
        eval_freq: Evaluate every N steps
        n_eval_episodes: Number of evaluation episodes
        learning_rate: Learning rate for PPO
        batch_size: Batch size for PPO
        n_steps: Steps per update for PPO
        device: Device to use (auto/cuda/cpu)
    """
    
    # Create timestamp for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{game}_backward_PPO_{timestamp}"
    
    # Create directories
    checkpoint_dir = f"checkpoints/{run_name}"
    log_dir = f"logs/{run_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    print("="*70)
    print("BACKWARD GENERATION TRAINING")
    print("="*70)
    print(f"Game: {game}")
    print(f"Representation: {representation}")
    print(f"Approach: REVERSE (from solved state)")
    print(f"Timesteps: {timesteps:,}")
    print(f"Device: {device}")
    print(f"Run: {run_name}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")
    print("="*70)
    
    # Create environment (use regular env with custom reward shaping)
    # TODO: Register sokoban-reverse environment in gym_pcgrl/__init__.py
    # For now, use standard env with custom wrapper
    
    print("\n📦 Creating backward generation environment...")
    
    # Create resource monitor
    resource_monitor = ResourceMonitor(use_gpu=True)
    
    # Custom reward wrapper for backward generation
    class BackwardRewardWrapper(gym.Wrapper):
        """
        Reward shaping for backward generation.
        
        Rewards:
        - Creating longer solution paths
        - Maintaining solvability
        - Creating interesting puzzle structures
        - Avoiding trivial configurations
        """
        
        def __init__(self, env):
            super().__init__(env)
            self.solution_length = 0
            self.prev_solution_length = 0
            
        def reset(self, **kwargs):
            obs = self.env.reset(**kwargs)
            self.solution_length = 0
            self.prev_solution_length = 0
            return obs
        
        def step(self, action):
            obs, reward, done, info = self.env.step(action)
            
            # Get current solution length
            self.prev_solution_length = self.solution_length
            self.solution_length = info.get('solution_length', 0)
            
            # Custom reward shaping
            custom_reward = reward
            
            # Reward for extending solution
            if self.solution_length > self.prev_solution_length:
                custom_reward += 5.0
            
            # Bonus for reaching milestones
            if self.solution_length >= 10:
                custom_reward += 2.0
            if self.solution_length >= 20:
                custom_reward += 3.0
            if self.solution_length >= 30:
                custom_reward += 5.0
            
            # Penalty for too short solutions at episode end
            if done and self.solution_length < 10:
                custom_reward -= 10.0
            
            # Update info
            info['original_reward'] = reward
            info['backward_reward'] = custom_reward
            info['solution_length'] = self.solution_length
            
            return obs, custom_reward, done, info
    
    # Create base environment
    env = make_pcgrl_env(
        game=game,
        representation=representation,
        device=device,
        resource_monitor=resource_monitor,
        verbose=True
    )
    
    # Wrap with backward reward shaping
    env = BackwardRewardWrapper(env)
    
    # Create evaluation environment
    eval_env = make_pcgrl_env(
        game=game,
        representation=representation,
        device=device,
        resource_monitor=resource_monitor,
        verbose=False
    )
    eval_env = BackwardRewardWrapper(eval_env)
    
    print("✓ Environments created")
    
    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="backward_rl_model",
        verbose=1
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=checkpoint_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # Create PPO model
    print(f"\n🤖 Creating PPO model...")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print(f"  Steps per update: {n_steps}")
    
    model = PPO(
        "MultiInputPolicy",  # Use MultiInputPolicy for dict observation spaces
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=1,
        device=device
    )
    
    print("✓ Model created")
    
    # Train
    print(f"\n🚀 Starting backward generation training for {timesteps:,} timesteps...")
    print("="*70)
    
    model.learn(
        total_timesteps=timesteps,
        callback=[checkpoint_callback, eval_callback]
    )
    
    # Save final model
    final_model_path = os.path.join(checkpoint_dir, "final_model.zip")
    model.save(final_model_path)
    
    print("\n" + "="*70)
    print("✓ Training complete!")
    print(f"Final model saved to: {final_model_path}")
    print(f"Checkpoints: {checkpoint_dir}")
    print(f"Logs: {log_dir}")
    print("="*70)
    
    return model, checkpoint_dir


def main():
    parser = argparse.ArgumentParser(description='Train Sokoban backward generator')
    
    parser.add_argument('--game', type=str, default='sokoban',
                       help='Game name (default: sokoban)')
    parser.add_argument('--representation', type=str, default='narrow',
                       choices=['narrow', 'wide', 'turtle'],
                       help='Representation type')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Total training timesteps')
    parser.add_argument('--checkpoint-freq', type=int, default=5000,
                       help='Checkpoint save frequency')
    parser.add_argument('--eval-freq', type=int, default=2000,
                       help='Evaluation frequency')
    parser.add_argument('--n-eval-episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--learning-rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--n-steps', type=int, default=2048,
                       help='Steps per update')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use')
    
    args = parser.parse_args()
    
    train_backward_sokoban(
        game=args.game,
        representation=args.representation,
        timesteps=args.timesteps,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        device=args.device
    )


if __name__ == "__main__":
    main()
