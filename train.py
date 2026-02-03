"""
RAPCG-MetaRL Training Script
Train Meta-RL agents for resource-aware procedural content generation.
"""
import os
import sys
import argparse
import numpy as np
from datetime import datetime

# Add project paths - prioritize main project over gym-pcgrl
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, 'gym-pcgrl'))

# Import local modules
from utils import ResourceMonitor, TrainingLogger, create_checkpoint_dir
from wrappers.pcgrl_env import make_pcgrl_env

# Stable Baselines3 for RL
try:
    from stable_baselines3 import PPO, A2C, SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from stable_baselines3.common.callbacks import BaseCallback
except ImportError:
    print("Error: stable-baselines3 not installed. Install with: pip install stable-baselines3")
    sys.exit(1)


class ResourceAwareCallback(BaseCallback):
    """
    Callback for resource-aware training with dynamic adaptation.
    """
    
    def __init__(self, resource_monitor, training_logger, save_freq=1000, 
                 checkpoint_dir='checkpoints', verbose=1):
        super().__init__(verbose)
        self.resource_monitor = resource_monitor
        self.training_logger = training_logger
        self.save_freq = save_freq
        self.checkpoint_dir = checkpoint_dir
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        
    def _on_step(self) -> bool:
        """Called at each step with resource-aware penalty tracking."""
        # Get current resources
        resources = self.resource_monitor.get_resources()
        
        # Get reward (already shaped by ResourceAwarePCGRLWrapper)
        reward = self.locals.get('rewards', [0])[0]
        self.current_episode_reward += reward
        self.current_episode_length += 1
        
        # Get additional info from environment (penalties, raw reward, etc.)
        infos = self.locals.get('infos', [{}])
        info = infos[0] if infos else {}
        
        # Extract penalty information from wrapper
        ram_penalty = info.get('ram_penalty', 0.0)
        cpu_penalty = info.get('cpu_penalty', 0.0)
        gpu_penalty = info.get('gpu_penalty', 0.0)
        total_penalty = info.get('total_penalty', 0.0)
        
        # Get action taken
        actions = self.locals.get('actions', [None])
        action = actions[0] if actions is not None and len(actions) > 0 else None
        if hasattr(action, 'item'):
            action = action.item()  # Convert numpy/torch to Python int
        
        # Prepare penalty breakdown
        penalty_info = {
            'ram_penalty': ram_penalty,
            'cpu_penalty': cpu_penalty,
            'gpu_penalty': gpu_penalty,
            'total_penalty': total_penalty,
        }
        
        # Log step with action and penalty tracking
        self.training_logger.log_step(reward, resources, action=action, penalty_info=penalty_info)
        
        # Track penalties for analysis
        if not hasattr(self, 'total_penalties'):
            self.total_penalties = []
        self.total_penalties.append(total_penalty)
        
        # Check if episode ended
        done = self.locals.get('dones', [False])[0]
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.training_logger.log_episode_end()
            
            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                # Show penalty info
                recent_penalties = self.total_penalties[-100:] if hasattr(self, 'total_penalties') else []
                avg_penalty = np.mean(recent_penalties) if recent_penalties else 0.0
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Mean Reward (last 10): {mean_reward:.2f}, "
                      f"Avg Penalty: {avg_penalty:.2f}, "
                      f"CPU: {resources['cpu_percent']:.1f}%, "
                      f"RAM: {resources['ram_percent']:.1f}%, "
                      f"GPU: {resources['gpu_mem_percent']:.1f}%")
            
            # Show action-penalty correlation every 50 episodes
            if self.verbose > 0 and len(self.episode_rewards) % 50 == 0:
                high_penalty_actions = self.training_logger.get_high_penalty_actions(
                    top_n=5, penalty_type='ram_penalty', recent_steps=1000
                )
                if high_penalty_actions:
                    print("\n  Top RAM-intensive actions (recent 1000 steps):")
                    for action, avg_penalty in high_penalty_actions:
                        print(f"    Action {action}: Avg RAM Penalty = {avg_penalty:.3f}")
                    print()
            
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        # Save checkpoint periodically
        if self.n_calls % self.save_freq == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, 
                f'model_step_{self.n_calls}.zip'
            )
            self.model.save(checkpoint_path)
            self.training_logger.save()
            
            if self.verbose > 0:
                print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Check resource pressure and adapt if needed
        is_pressure, msg = self.resource_monitor.check_resource_pressure()
        if is_pressure and self.verbose > 0:
            print(f"⚠ Resource pressure detected: {msg}")
            # Could implement dynamic adaptation here
        
        return True
    
    def _on_training_end(self) -> None:
        """Called at end of training."""
        self.training_logger.save()
        self.training_logger.print_stats()


class MetaRLTrainer:
    """
    Meta-RL trainer for PCGRL environments.
    Supports PPO, A2C, and other algorithms with resource-aware training.
    """
    
    def __init__(self, 
                 game='zelda',
                 representation='narrow',
                 algorithm='PPO',
                 total_timesteps=50000,
                 n_steps=128,
                 batch_size=64,
                 learning_rate=2.5e-4,
                 n_envs=1,
                 device='auto',
                 seed=None,
                 experiment_name=None,
                 use_gpu_monitoring=True,
                 checkpoint_freq=1000,
                 log_dir='logs',
                 checkpoint_dir='checkpoints',
                 sokoban_unsolvable_penalty=25.0,
                 use_solvability_tuning=True):
        """
        Initialize Meta-RL trainer.
        
        Args:
            game: Game environment ('zelda', 'sokoban', 'binary')
            representation: Representation type ('narrow', 'wide', 'turtle')
            algorithm: RL algorithm ('PPO', 'A2C', 'SAC')
            total_timesteps: Total training steps
            n_steps: Steps per update
            batch_size: Batch size for training
            learning_rate: Learning rate
            n_envs: Number of parallel environments
            device: Device ('cpu', 'cuda', 'auto')
            seed: Random seed
            experiment_name: Experiment name for logging
            use_gpu_monitoring: Enable GPU monitoring
            checkpoint_freq: Checkpoint save frequency
            log_dir: Log directory
            checkpoint_dir: Checkpoint directory
        """
        self.game = game
        self.representation = representation
        self.algorithm = algorithm
        self.total_timesteps = total_timesteps
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_envs = n_envs
        self.device = device
        self.seed = seed
        self.use_solvability_tuning = use_solvability_tuning
        self.sokoban_unsolvable_penalty = sokoban_unsolvable_penalty
        
        # Detect GPU availability and adjust device
        if device == 'auto':
            import torch
            if torch.cuda.is_available():
                self.device = 'cuda'
                print("✓ GPU detected: Using CUDA for training")
            else:
                self.device = 'cpu'
                print("⚠ No GPU detected: Using CPU for training")
                print("  To enable GPU: pip install torch --index-url https://download.pytorch.org/whl/cu121")
        
        # Set up experiment tracking
        if experiment_name is None:
            # Include device type (CUDA/CPU) in experiment name
            device_suffix = "CUDA" if self.device == 'cuda' else "CPU"
            experiment_name = f"{game}_{algorithm}_{device_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.experiment_name = experiment_name
        
        # Initialize monitoring and logging
        # Only monitor GPU if we're actually using it for training
        use_gpu_monitoring_actual = use_gpu_monitoring and (self.device == 'cuda')
        self.resource_monitor = ResourceMonitor(use_gpu=use_gpu_monitoring_actual)
        self.logger = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name)
        self.checkpoint_dir = create_checkpoint_dir(checkpoint_dir, experiment_name)
        self.checkpoint_freq = checkpoint_freq
        
        # Will be set during training
        self.env = None
        self.model = None
        
        print(f"\n{'='*60}")
        print(f"Meta-RL Trainer Initialized")
        print(f"{'='*60}")
        print(f"Game: {game}")
        print(f"Algorithm: {algorithm}")
        print(f"Total Timesteps: {total_timesteps:,}")
        print(f"Device: {self.device}")
        if self.device == 'cuda':
            import torch
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Monitoring: {'Enabled' if use_gpu_monitoring_actual else 'Disabled'}")
        print(f"Experiment: {experiment_name}")
        print(f"{'='*60}\n")
    
    def make_env(self, rank=0):
        """
        Create and wrap environment with resource awareness.
        
        Args:
            rank: Environment rank for seeding
            
        Returns:
            Callable that creates the environment
        """
        def _init():
            # AND apply solvability-optimized reward weights
            env = make_pcgrl_env(
                resource_monitor=self.resource_monitor,  # CRUCIAL: Enable feedback loop
                game=self.game,
                representation=self.representation,
                ram_penalty_weight=0.2,  # Configurable penalty weights
                cpu_penalty_weight=0.1,
                gpu_penalty_weight=0.1,
                sokoban_unsolvable_penalty=self.sokoban_unsolvable_penalty,
                use_solvability_config=self.use_solvability_tuning  # Apply tuned weights
            )
            # Don't use Monitor - we have our own logging via TrainingLogger
            if self.seed is not None:
                env.seed(self.seed + rank)
            return env
        return _init
    
    def setup_environments(self):
        """Set up training environments."""
        print(f"Setting up {self.n_envs} environment(s)...")
        
        if self.n_envs == 1:
            self.env = DummyVecEnv([self.make_env(0)])
        else:
            self.env = SubprocVecEnv([self.make_env(i) for i in range(self.n_envs)])
        
        print(f"✓ Environments ready")
    
    def setup_model(self):
        """Set up RL model."""
        print(f"Setting up {self.algorithm} model...")
        
        # Determine policy type based on observation space
        obs_space = self.env.observation_space
        if hasattr(obs_space, 'spaces') and isinstance(obs_space.spaces, dict):
            policy_type = 'MultiInputPolicy'
        else:
            policy_type = 'MlpPolicy'
        
        print(f"Using policy: {policy_type}")
        
        # Common parameters
        model_kwargs = {
            'learning_rate': self.learning_rate,
            'verbose': 1,
            'device': self.device,
            'seed': self.seed,
        }
        
        # Algorithm-specific parameters
        if self.algorithm == 'PPO':
            model_kwargs.update({
                'n_steps': self.n_steps,
                'batch_size': self.batch_size,
                'n_epochs': 10,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'clip_range': 0.2,
                'ent_coef': 0.01,
            })
            self.model = PPO(policy_type, self.env, **model_kwargs)
            
        elif self.algorithm == 'A2C':
            model_kwargs.update({
                'n_steps': self.n_steps,
                'gamma': 0.99,
                'gae_lambda': 0.95,
                'ent_coef': 0.01,
            })
            self.model = A2C(policy_type, self.env, **model_kwargs)
            
        elif self.algorithm == 'SAC':
            # SAC requires continuous action spaces
            # For discrete actions, it will fail - consider using PPO or A2C instead
            action_space = self.env.action_space
            from gym import spaces
            if isinstance(action_space, spaces.Discrete):
                print("⚠ WARNING: SAC is designed for continuous action spaces!")
                print("  gym-pcgrl uses discrete actions. Consider using PPO or A2C instead.")
                print("  Training may fail or produce poor results.")
            
            model_kwargs.update({
                'buffer_size': 100000,  # Replay buffer size
                'learning_starts': 1000,  # Start learning after N steps
                'batch_size': 256,  # Larger batch for off-policy
                'tau': 0.005,  # Soft update coefficient
                'gamma': 0.99,
                'train_freq': 1,
                'gradient_steps': 1,
                'ent_coef': 'auto',  # Automatic entropy tuning
            })
            self.model = SAC(policy_type, self.env, **model_kwargs)
            
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        print(f"✓ Model initialized")
        print(f"  Policy: {policy_type}")
        print(f"  Learning rate: {self.learning_rate}")
        print(f"  Batch size: {self.batch_size}")
    
    def train(self):
        """Run training loop."""
        print(f"\nStarting training...")
        print(f"Target: {self.total_timesteps:,} timesteps\n")
        
        # Set up callback
        callback = ResourceAwareCallback(
            resource_monitor=self.resource_monitor,
            training_logger=self.logger,
            save_freq=self.checkpoint_freq,
            checkpoint_dir=self.checkpoint_dir,
            verbose=1
        )
        
        try:
            # Train model
            self.model.learn(
                total_timesteps=self.total_timesteps,
                callback=callback
            )
            
            print("\n✓ Training completed!")
            
            # Save final model
            final_model_path = os.path.join(self.checkpoint_dir, 'final_model.zip')
            self.model.save(final_model_path)
            print(f"✓ Final model saved: {final_model_path}")
            
            # Save logs
            self.logger.save()
            self.logger.print_stats()
            
        except KeyboardInterrupt:
            print("\n\n⚠ Training interrupted by user")
            self.logger.save()
            print("✓ Logs saved")
        
        except Exception as e:
            print(f"\n✗ Training error: {e}")
            import traceback
            traceback.print_exc()
            self.logger.save()
        
        finally:
            if self.env is not None:
                self.env.close()
    
    def evaluate(self, n_episodes=10):
        """
        Evaluate trained model.
        
        Args:
            n_episodes: Number of episodes to evaluate
        """
        if self.model is None:
            print("Error: No model to evaluate. Train first or load a model.")
            return
        
        print(f"\nEvaluating model for {n_episodes} episodes...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(n_episodes):
            obs = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                episode_reward += reward[0]
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, Length={episode_length}")
        
        print(f"\nEvaluation Results:")
        print(f"  Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
        print(f"  Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    
    def load_model(self, model_path):
        """
        Load a trained model.
        
        Args:
            model_path: Path to model file
        """
        print(f"Loading model from {model_path}...")
        
        if self.algorithm == 'PPO':
            self.model = PPO.load(model_path, env=self.env)
        elif self.algorithm == 'A2C':
            self.model = A2C.load(model_path, env=self.env)
        elif self.algorithm == 'SAC':
            self.model = SAC.load(model_path, env=self.env)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        print("✓ Model loaded")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Meta-RL agent for PCGRL')
    
    parser.add_argument('--game', type=str, default='zelda',
                       choices=['zelda', 'sokoban', 'binary', 'zelda-narrow'],
                       help='Game environment')
    parser.add_argument('--representation', type=str, default='narrow',
                       choices=['narrow', 'wide', 'turtle'],
                       help='Representation type')
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'A2C', 'SAC'],
                       help='RL algorithm (SAC only works with continuous action spaces)')
    parser.add_argument('--timesteps', type=int, default=50000,
                       help='Total training timesteps')
    parser.add_argument('--n-steps', type=int, default=128,
                       help='Steps per update')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=2.5e-4,
                       help='Learning rate')
    parser.add_argument('--sokoban-penalty', type=float, default=25.0,
                       help='Penalty for unsolvable Sokoban levels (default: 25.0 - very strict)')
    parser.add_argument('--no-solvability-tuning', action='store_true',
                       help='Disable solvability-optimized reward weights (not recommended)')
    parser.add_argument('--n-envs', type=int, default=1,
                       help='Number of parallel environments')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['cpu', 'cuda', 'auto'],
                       help='Device for training')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--experiment-name', type=str, default=None,
                       help='Experiment name')
    parser.add_argument('--no-gpu-monitoring', action='store_true',
                       help='Disable GPU monitoring')
    parser.add_argument('--checkpoint-freq', type=int, default=1000,
                       help='Checkpoint save frequency')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate after training')
    parser.add_argument('--load-model', type=str, default=None,
                       help='Load pre-trained model')
    
    args = parser.parse_args()
    
    # Create trainer
    trainer = MetaRLTrainer(
        game=args.game,
        representation=args.representation,
        algorithm=args.algorithm,
        total_timesteps=args.timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        n_envs=args.n_envs,
        device=args.device,
        seed=args.seed,
        experiment_name=args.experiment_name,
        use_gpu_monitoring=not args.no_gpu_monitoring,
        checkpoint_freq=args.checkpoint_freq,
        sokoban_unsolvable_penalty=args.sokoban_penalty,
        use_solvability_tuning=not args.no_solvability_tuning
    )
    
    # Set up environments and model
    trainer.setup_environments()
    
    if args.load_model:
        trainer.load_model(args.load_model)
    else:
        trainer.setup_model()
    
    # Train
    trainer.train()
    
    # Evaluate if requested
    if args.evaluate:
        trainer.evaluate(n_episodes=10)


if __name__ == '__main__':
    main()
