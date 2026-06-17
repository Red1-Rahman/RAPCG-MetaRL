import os
import time
import argparse
import numpy as np
import pandas as pd
import gym
import gym_pcgrl  # Registers the pcgrl environments
from gym.spaces import Box
from stable_baselines3 import PPO

# Import your existing solvability wrapper 
# (Adjust this import if your wrapper is located in a different file, like wrappers.py)
try:
    from sokoban_utils import SokobanSolvabilityWrapper
except ImportError:
    # Fallback/Dummy if the exact import path is different in your repo
    print("[WARNING] Could not import SokobanSolvabilityWrapper. Check your import path.")
    class SokobanSolvabilityWrapper(gym.Wrapper):
        def __init__(self, env, penalty=25.0):
            super().__init__(env)

class RLHFFlattenWrapper(gym.ObservationWrapper):
    """
    Flattens dictionary observations into a 1D array to match
    the Box(52,) space expected by the RLHF model.
    """
    def __init__(self, env):
        super().__init__(env)
        # The RLHF model expects a flattened 52-dimensional Box
        self.observation_space = Box(low=0.0, high=5.0, shape=(52,), dtype=np.float32)

    def observation(self, obs):
        # If it's a dictionary (like OrderedDict from PCGRL)
        if isinstance(obs, dict):
            # Flatten all values in the dict and concatenate them into a single 1D array
            flat_obs = np.concatenate([np.array(v).flatten() for v in obs.values()])
            return flat_obs.astype(np.float32)
        
        # Fallback if it's already an array
        if isinstance(obs, np.ndarray):
            return obs.flatten().astype(np.float32)
            
        return obs

class RLHFLevelGenerator:
    def __init__(self, model_path, game, device="auto"):
        self.game = game
        self.device = device
        
        print(f"Loading RLHF model from {model_path} on {device}...")
        self.model = PPO.load(model_path, device=device)
        
        # Initialize the base environment
        env_name = f"{game}-narrow-v0"
        self.env = gym.make(env_name)
        
        # 1. Apply Solvability Configuration (as seen in your logs)
        if game.lower() == "sokoban":
            print("Applying solvability-optimized configuration for SOKOBAN")
            self.env = SokobanSolvabilityWrapper(self.env, penalty=25.0)
            
        # 2. Apply the crucial Flattening Wrapper for the RLHF model
        self.env = RLHFFlattenWrapper(self.env)
        
        print("[OK] Setup complete")

    def generate_with_timing(self, n_levels, deterministic=True):
        print("======================================================================")
        print(f"GENERATING {n_levels} LEVELS WITH TIMING")
        
        timing_data = []
        
        for i in range(n_levels):
            print(f"Level {i + 1}/{n_levels}:")
            
            start_time = time.time()
            
            obs = self.env.reset()
            # Handle Gym API differences (tuple vs single observation)
            if isinstance(obs, tuple):
                obs = obs[0]
                
            done = False
            steps = 0
            
            while not done:
                # The obs is now guaranteed to be a flattened 52-dim array by the wrapper
                action, _ = self.model.predict(obs, deterministic=deterministic)
                
                step_result = self.env.step(action)
                
                # Handle old vs new Gym step returns
                if len(step_result) == 4:
                    obs, reward, done, info = step_result
                else:
                    obs, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                    
                steps += 1
                
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Extract final map (assuming it's stored in the environment)
            # Adjust according to how gym-pcgrl stores the final grid in your version
            final_map = self.env.unwrapped._rep._map if hasattr(self.env.unwrapped, '_rep') else None
            
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