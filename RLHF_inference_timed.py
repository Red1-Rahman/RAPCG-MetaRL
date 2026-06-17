import os
import time
import argparse
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd
import gym
from gym.core import ObservationWrapper
from gym.spaces import Box
import gym_pcgrl  # Registers the pcgrl environments
from stable_baselines3 import PPO

# Direct import from project utilities
from sokoban_utils import SokobanSolvabilityWrapper


class RLHFFlattenWrapper(ObservationWrapper):
    """
    Flattens dictionary observations into a 1D array to match
    the Box(52,) space expected by the RLHF model.
    """
    def __init__(self, env: gym.Env):
        super().__init__(env)
        # The RLHF model expects a flattened 52-dimensional Box
        self.observation_space = Box(low=0.0, high=5.0, shape=(52,), dtype=np.float32)

    def reset(self, **kwargs: Any) -> np.ndarray:
        """
        Overrides reset to ensure compatibility with older Gym API
        and force flattening of the returned observation.
        """
        # Call the underlying environment's reset
        result = self.env.reset(**kwargs)
        # Extract obs (handle tuple vs single return)
        obs = result[0] if isinstance(result, tuple) else result
        # Return as the flattened observation
        return self.observation(obs)

    def observation(self, observation: Any) -> np.ndarray:
        # If it's a dictionary (like OrderedDict from PCGRL)
        if isinstance(observation, dict):
            # Flatten all values in the dict and concatenate them into a single 1D array
            flat_obs = np.concatenate([np.array(v).flatten() for v in observation.values()])
            return flat_obs.astype(np.float32)
        
        # Fallback if it's already an array
        if isinstance(observation, np.ndarray):
            return observation.flatten().astype(np.float32)
            
        return np.array(observation, dtype=np.float32)


class RLHFLevelGenerator:
    def __init__(self, model_path: str, game: str, device: str = "auto"):
        self.game = game
        self.device = device
        
        print(f"Loading RLHF model from {model_path} on {device}...")
        self.model = PPO.load(model_path, device=device)
        
        # Initialize the base environment
        env_name = f"{game}-narrow-v0"
        self.env: gym.Env = gym.make(env_name)
        
        # 1. Apply Solvability Configuration
        if game.lower() == "sokoban":
            print("Applying solvability-optimized configuration for SOKOBAN")
            # Use 'unsolvable_penalty' to match the actual class signature in sokoban_utils.py
            wrapper_kwargs: Dict[str, Any] = {"unsolvable_penalty": 25.0}
            self.env = SokobanSolvabilityWrapper(self.env, **wrapper_kwargs)
            
        # 2. Apply the crucial Flattening Wrapper
        self.env = RLHFFlattenWrapper(self.env)
        
        print("[OK] Setup complete")

    def generate_with_timing(self, n_levels: int, deterministic: bool = True) -> pd.DataFrame:
        print("======================================================================")
        print(f"GENERATING {n_levels} LEVELS WITH TIMING")
        
        timing_data = []
        
        for i in range(n_levels):
            print(f"Level {i + 1}/{n_levels}:")
            start_time = time.time()
            
            # The wrapper override handles reset and returns the flattened observation
            obs = self.env.reset()
                
            done = False
            steps = 0
            reward: float = 0.0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                
                step_result: Tuple[Any, ...] = self.env.step(action)
                
                # Handle old (obs, reward, done, info) vs new (obs, reward, terminated, truncated, info)
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                    
                steps += 1
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Safe attribute access
            unwrapped_env: Any = self.env.unwrapped
            timing_data.append({
                "level_id": i + 1,
                "time_seconds": elapsed_time,
                "steps_taken": steps,
                "reward": reward
            })
            
            print(f"  -> Finished in {elapsed_time:.3f}s ({steps} steps)")
            
        return pd.DataFrame(timing_data)


def main():
    parser = argparse.ArgumentParser(description="Timed Inference for RLHF Models")
    parser.add_argument("model_path", type=str, help="Path to the RLHF model (.zip)")
    parser.add_argument("--game", type=str, required=True, help="Game name (e.g., sokoban)")
    parser.add_argument("--n-levels", type=int, default=20, help="Number of levels to generate")
    parser.add_argument("--log-file", type=str, default="inference_timing_rlhf.csv", help="Output CSV file")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cpu, cuda, auto)")
    
    args = parser.parse_args()
    
    generator = RLHFLevelGenerator(args.model_path, args.game, args.device)
    df = generator.generate_with_timing(n_levels=args.n_levels)
    
    df.to_csv(args.log_file, index=False)
    print("======================================================================")
    print(f"[OK] Timed inference complete. Results saved to {args.log_file}")

if __name__ == "__main__":
    main()