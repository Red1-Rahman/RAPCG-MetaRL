"""
Quick Start Script for RAPCG-MetaRL
Run this to quickly test the setup and train a simple model.
"""

import os
import sys


def main():
    print("=" * 60)
    print("RAPCG-MetaRL Quick Start")
    print("=" * 60)

    print("\n1. Running tests...")
    os.system("python test/test.py")

    print("\n2. Training a small model (10,000 steps)...")
    os.system(
        "python train.py --game zelda --timesteps 10000 --experiment-name quickstart"
    )

    print("\n3. Generating levels...")
    os.system("python inference.py checkpoints/quickstart/final_model.zip --n-levels 3")

    print("\n" + "=" * 60)
    print("Quick start complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("- Check logs/ for training metrics")
    print("- Check checkpoints/ for saved models")
    print("- Check generated_levels/ for output")
    print("\nFor full training:")
    print("  python train.py --game zelda --timesteps 50000")


if __name__ == "__main__":
    main()
