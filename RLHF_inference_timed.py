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
        result = self.env.reset(**kwargs)
        obs = result[0] if isinstance(result, tuple) else result
        return self.observation(obs)

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Overrides step to intercept the inner environment's return,
        preventing gym.Wrapper from breaking on a 4-vs-5 item unpack error.
        Always returns a 5-tuple format matching Gym's updated expected API.
        """
        # Call the inner environment step directly bypassing the wrapper's default unpacker
        step_result = self.env.step(action)

        if len(step_result) == 4:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        else:
            obs, reward, terminated, truncated, info = step_result

        # Return the processed observation and elements as a strict 5-tuple
        return (
            self.observation(obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def observation(self, observation: Any) -> np.ndarray:
        if isinstance(observation, dict):
            flat_obs = np.concatenate(
                [np.array(v).flatten() for v in observation.values()]
            )
            return flat_obs.astype(np.float32)
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
            wrapper_kwargs: Dict[str, Any] = {"unsolvable_penalty": 25.0}
            self.env = SokobanSolvabilityWrapper(self.env, **wrapper_kwargs)

        # 2. Apply the crucial Flattening and API Bridging Wrapper
        self.env = RLHFFlattenWrapper(self.env)

        print("[OK] Setup complete")

    def generate_with_timing(
        self, n_levels: int, deterministic: bool = True
    ) -> pd.DataFrame:
        print("======================================================================")
        print(f"GENERATING {n_levels} LEVELS WITH TIMING")

        timing_data = []

        for i in range(n_levels):
            print(f"Level {i + 1}/{n_levels}:")
            start_time = time.time()

            # This returns the flattened obs directly
            obs = self.env.reset()

            done = False
            steps = 0
            reward: float = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)

                # Because RLHFFlattenWrapper.step handles everything, this returns a valid 5-tuple
                obs, step_reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                reward += step_reward
                steps += 1

            end_time = time.time()
            elapsed_time = end_time - start_time

            timing_data.append(
                {
                    "level_id": i + 1,
                    "time_seconds": elapsed_time,
                    "steps_taken": steps,
                    "reward": reward,
                }
            )

            print(f"  -> Finished in {elapsed_time:.3f}s ({steps} steps)")

        return pd.DataFrame(timing_data)


def main():
    parser = argparse.ArgumentParser(description="Timed Inference for RLHF Models")
    parser.add_argument("model_path", type=str, help="Path to the RLHF model (.zip)")
    parser.add_argument(
        "--game", type=str, required=True, help="Game name (e.g., sokoban)"
    )
    parser.add_argument(
        "--n-levels", type=int, default=20, help="Number of levels to generate"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default="inference_timing_rlhf.csv",
        help="Output CSV file",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (cpu, cuda, auto)"
    )

    args = parser.parse_args()

    generator = RLHFLevelGenerator(args.model_path, args.game, args.device)
    df = generator.generate_with_timing(n_levels=args.n_levels)

    df.to_csv(args.log_file, index=False)
    print("======================================================================")
    print(f"[OK] Timed inference complete. Results saved to {args.log_file}")


if __name__ == "__main__":
    main()
