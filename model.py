"""
Model utilities for RAPCG-MetaRL
Compatible wrapper for gym-pcgrl models
"""

import os
import sys

# Add gym-pcgrl to path for backward compatibility
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "gym-pcgrl"))

# Import from stable-baselines3
try:
    from stable_baselines3 import PPO, A2C
    from stable_baselines3.common.policies import ActorCriticPolicy
except ImportError:
    print("Warning: stable-baselines3 not installed")
    print("Install with: pip install stable-baselines3")


def load_model(model_path, algorithm="PPO", device="auto"):
    """
    Load a trained model.

    Args:
        model_path: Path to model file (.zip)
        algorithm: Algorithm type ('PPO', 'A2C')
        device: Device for inference ('cpu', 'cuda', 'auto')

    Returns:
        Loaded model
    """
    if algorithm == "PPO":
        return PPO.load(model_path, device=device)
    elif algorithm == "A2C":
        return A2C.load(model_path, device=device)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")


def save_model(model, save_path):
    """
    Save a model.

    Args:
        model: Model to save
        save_path: Path to save location
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    print("Model utilities loaded")
    print("Available algorithms: PPO, A2C")
