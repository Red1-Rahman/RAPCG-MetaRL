import os
import time
import argparse
from typing import Any, Dict, Tuple, cast
import numpy as np
import pandas as pd
import gym
from gym.core import ObservationWrapper
from gym.spaces import Box
import gym_pcgrl  # Registers the pcgrl environments
from stable_baselines3 import PPO

# Direct import from project utilities
from sokoban_utils import SokobanSolvabilityWrapper

# Attempt to import visualization logic
try:
    from visualize_levels import save_level_image

    VISUALIZATION_AVAILABLE = True
except ImportError:
    print(
        "[WARNING] visualize_levels.py not found in path. PNG generation will be disabled."
    )
    VISUALIZATION_AVAILABLE = False


class RLHFFlattenWrapper(ObservationWrapper):
    """
    Flattens dictionary observations into a 1D array to match
    the Box(52,) space expected by the RLHF model.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # The RLHF model expects a flattened 52-dimensional Box
        self.observation_space = Box(low=0.0, high=5.0, shape=(52,), dtype=np.float32)

    def reset(self, **kwargs: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Overrides reset to ensure compatibility with older Gym API
        while conforming to ObservationWrapper's expected Tuple type.
        """
        result = self.env.reset(**kwargs)

        # Parse observation and metadata safely
        if isinstance(result, tuple) and len(result) == 2:
            obs_raw, info = result
        else:
            obs_raw, info = result, {}

        return self.observation(obs_raw), info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Overrides step to intercept the inner environment's return.
        """
        step_result = self.env.step(action)

        if len(step_result) == 4:
            obs_raw, reward, done, info = step_result
            terminated = done
            truncated = False
        else:
            obs_raw, reward, terminated, truncated, info = step_result

        return (
            self.observation(obs_raw),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def observation(self, observation: Any) -> np.ndarray:
        if isinstance(observation, dict):
            # Cast observation to a specific dictionary type to avoid 'Never' iterability error
            obs_dict = cast(Dict[str, Any], observation)
            arrays = [np.array(v).flatten() for v in obs_dict.values()]
            flat_obs = np.concatenate(arrays)
            return flat_obs.astype(np.float32)

        if isinstance(observation, np.ndarray):
            return observation.flatten().astype(np.float32)

        return np.array(observation, dtype=np.float32)


class RLHFLevelGenerator:
    def __init__(self, model_path: str, game: str, device: str = "auto"):
        self.game = game
        self.device = device
        self.model_path = model_path

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

    def _get_output_dir(self) -> str:
        """Determines the target folder path dynamically from the model filename."""
        base_folder = os.path.basename(os.path.dirname(self.model_path))
        if not base_folder or base_folder == "checkpoints":
            base_folder = os.path.splitext(os.path.basename(self.model_path))[0]

        return os.path.join("generated_levels", base_folder)

    def generate_with_timing(
        self, n_levels: int, deterministic: bool = True
    ) -> pd.DataFrame:
        print("======================================================================")
        print(f"GENERATING {n_levels} LEVELS WITH TIMING")

        output_dir = self._get_output_dir()
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"Created output directory: {output_dir}")
        else:
            print(f"Using existing output directory: {output_dir}")

        timing_data = []

        for i in range(n_levels):
            level_id = i + 1
            print(f"Level {level_id}/{n_levels}:")

            npy_path = os.path.join(output_dir, f"level_{level_id}.npy")
            txt_path = os.path.join(output_dir, f"level_{level_id}.txt")
            png_path = os.path.join(output_dir, f"level_{level_id}.png")

            # Check if this level's artifact files already exist
            if (
                os.path.exists(npy_path)
                and os.path.exists(txt_path)
                and (not VISUALIZATION_AVAILABLE or os.path.exists(png_path))
            ):
                print(f"  -> Skipping generation: level files already exist.")
                continue

            start_time = time.time()
            obs, info = self.env.reset()

            done = False
            steps = 0
            reward: float = 0.0

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, step_reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                reward += step_reward
                steps += 1

            end_time = time.time()
            elapsed_time = end_time - start_time

            try:
                # Cast unwrapped environment to Any to bypass base gym.Env attribute limits
                unwrapped_env = cast(Any, self.env.unwrapped)
                final_map = unwrapped_env._rep._map.copy()

                # Save numerical formats
                np.save(npy_path, final_map)
                np.savetxt(txt_path, final_map, fmt="%d")

                # Save image format if visualization script is accessible
                if VISUALIZATION_AVAILABLE:
                    save_level_image(
                        level=final_map,
                        filepath=png_path,
                        game=self.game.lower(),
                        scale=20,
                        show_grid=True,
                        dpi=300,
                    )
            except Exception as e:
                print(f"  Warning: Could not save layout artifacts: {e}")

            timing_data.append(
                {
                    "level_id": level_id,
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

    if not df.empty:
        df.to_csv(args.log_file, index=False)
        print("======================================================================")
        print(f"[OK] Timed inference complete. Metrics saved to {args.log_file}")
    else:
        print("======================================================================")
        print("[INFO] No new levels were generated. Log file unchanged.")


if __name__ == "__main__":
    main()
