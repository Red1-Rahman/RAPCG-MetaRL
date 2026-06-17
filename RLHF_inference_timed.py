import os
import time
import argparse
from typing import Any, Tuple
import numpy as np
import pandas as pd
import gym
from gym.core import ObservationWrapper
from gym.spaces import Box
import gym_pcgrl  # Registers the pcgrl environments
from stable_baselines3 import PPO

# Fix: Direct import removes the re-assignment type conflict.
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

    # Fix: Renamed parameter 'obs' to 'observation' to perfectly match the base class
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
        
        # 1. Apply Solvability Configuration (as seen in logs)
        if game.lower() == "sokoban":
            print("Applying solvability-optimized configuration for SOKOBAN")
            self.env = SokobanSolvabilityWrapper(self.env, penalty=25.0)
            
        # 2. Apply the crucial Flattening Wrapper for the RLHF model
        self.env = RLHFFlattenWrapper(self.env)
        
        print("[OK] Setup complete")

    def generate_with_timing(self, n_levels: int, deterministic: bool = True) -> pd.DataFrame:
        print("======================================================================")
        print(f"GENERATING {n_levels} LEVELS WITH TIMING")
        
        timing_data = []
        
        for i in range(n_levels):
            print(f"Level {i + 1}/{n_levels}:")
            
            start_time = time.time()
            
            obs_raw = self.env.reset()
            # Handle Gym API differences (tuple vs single observation)
            if isinstance(obs_raw, tuple):
                obs = obs_raw[0]
            else:
                obs = obs_raw
                
            done = False
            steps = 0
            # Fix: Ensure reward is bound even if the loop somehow doesn't run
            reward: float = 0.0
            
            while not done:
                # The obs is now guaranteed to be a flattened 52-dim array by the wrapper
                action, _ = self.model.predict(obs, deterministic=deterministic)
                
                # Fix: explicitly type step_result as a Tuple so len() is statically valid
                step_result: Tuple[Any, ...] = self.env.step(action)
                
                # Handle old vs new Gym step returns safely
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                    
                steps += 1
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Fix: Cast unwrapped environment to Any to bypass strict attribute checking safely
            unwrapped_env: Any = self.env.unwrapped
            final_map = unwrapped_env._rep._map if hasattr(unwrapped_env, '_rep') else None
            
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
    
    print("TIMED INFERENCE SETUP")
    print(f"Model: {args.model_path}")
    print(f"Game: {args.game}")
    print(f"Device: {args.device}")
    
    generator = RLHFLevelGenerator(
        model_path=args.model_path,
        game=args.game,
        device=args.device
    )
    
    df = generator.generate_with_timing(n_levels=args.n_levels)
    
    # Save the timing logs
    df.to_csv(args.log_file, index=False)
    print("======================================================================")
    print(f"[OK] Timed inference complete. Results saved to {args.log_file}")
    print(f"Average time per level: {df['time_seconds'].mean():.4f} seconds")

if __name__ == "__main__":
    main()