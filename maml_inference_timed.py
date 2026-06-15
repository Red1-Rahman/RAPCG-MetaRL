"""
MAML Inference Timed Script
Loads a MAML checkpoint (best_meta_model.pt / final_meta_model.pt) and
generates levels with the same timing/metrics pipeline as inference_timed.py.

Key additions vs v1:
  - TrainingLogger live CSV written after every level (matches maml_trainer.py log format)
  - Per-level row flushed to --log-file immediately so no data lost on crash
  - Full terminal output with warnings (env creation, SB3 compat, etc.) visible

Usage:
    python maml_inference_timed.py checkpoints/sokoban_MAML_inference/best_meta_model.pt \\
        --game sokoban --n-levels 20 --max-steps 500 --device cuda
"""

import os
import sys
import csv
import argparse
import numpy as np
import pandas as pd
import time
import torch
from collections import OrderedDict
from typing import Optional, Tuple
from datetime import datetime

# --------------------------------------------------------------------------
# Project paths
# --------------------------------------------------------------------------
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, "gym-pcgrl"))

from utils import ResourceMonitor, TrainingLogger
from wrappers.pcgrl_env import make_pcgrl_env
from wrappers.helper import calculate_content_metrics, save_level
from maml_trainer import MAMLPolicy, DictFlattenWrapper, SokobanDeadlockGuardrail

try:
    from visualize_levels import save_level_image
except ImportError:

    def save_level_image(level, path, **kwargs):
        print(f"  [SKIP] visualize_levels not available: {path}")


try:
    from sokoban_utils import is_valid_sokoban
except ImportError:

    def is_valid_sokoban(level):
        players = int(np.sum(level == 2))
        crates = int(np.sum(level == 3))
        targets = int(np.sum(level == 4))
        ok = players == 1 and crates >= 1 and targets >= 1
        return ok, f"players={players}, crates={crates}, targets={targets}"


import gym

try:
    from stable_baselines3.common.vec_env import DummyVecEnv
except ImportError:
    print("Error: stable-baselines3 not installed.  pip install stable-baselines3")
    sys.exit(1)


# ==========================================================================
# Policy wrapper  (gives SB3-style .predict() over MAMLPolicy)
# ==========================================================================


class MAMLPolicyWrapper:
    def __init__(
        self,
        policy: MAMLPolicy,
        device: str,
        obs_dim: int,
        adapted_params: Optional[OrderedDict] = None,
    ):
        self.policy = policy
        self.device = device
        self.obs_dim = obs_dim
        self.adapted_params = adapted_params

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        obs_t = torch.FloatTensor(obs).to(self.device)
        if obs_t.dim() == 1:
            obs_t = obs_t.unsqueeze(0)
        obs_flat = obs_t.reshape(obs_t.shape[0], -1)

        feat = obs_flat.shape[-1]
        if feat < self.obs_dim:
            pad = torch.zeros(
                obs_flat.shape[0], self.obs_dim - feat, device=self.device
            )
            obs_flat = torch.cat([obs_flat, pad], dim=-1)
        elif feat > self.obs_dim:
            obs_flat = obs_flat[:, : self.obs_dim]

        with torch.no_grad():
            if self.adapted_params is not None:
                logits, _ = self.policy.functional_forward(
                    obs_flat, self.adapted_params
                )
            else:
                logits, _ = self.policy(obs_flat)

            if deterministic:
                action = logits.argmax(dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                action = torch.distributions.Categorical(probs).sample()

        return action.cpu().numpy(), None


# ==========================================================================
# Checkpoint loader
# ==========================================================================


def load_maml_checkpoint(
    checkpoint_path: str, obs_dim: int, action_dim: int, device: str
) -> Tuple[MAMLPolicy, dict]:
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})

    policy = MAMLPolicy(obs_dim, action_dim).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    print(f"[OK] Loaded MAML checkpoint : {checkpoint_path}")
    print(f"     Saved at iteration     : {ckpt.get('iteration', '?')}")
    print(f"     Meta-loss at save      : {ckpt.get('meta_loss', float('nan')):.4f}")
    print(f"     Training config        : {config}")
    return policy, config


# ==========================================================================
# Environment helpers
# ==========================================================================


def build_env(
    game: str, representation: str, resource_monitor: ResourceMonitor
) -> DummyVecEnv:
    def _make():
        env = make_pcgrl_env(
            resource_monitor=resource_monitor,
            game=game,
            representation=representation,
            use_solvability_config=True,
        )
        # Mirror the training-time deadlock guardrail so adaptation gradients
        # match the reward landscape the meta-policy was trained against.
        if game == "sokoban":
            env = SokobanDeadlockGuardrail(env, deadlock_penalty=1.5)
        if isinstance(env.observation_space, gym.spaces.Dict):
            env = DictFlattenWrapper(env)
        return env

    return DummyVecEnv([_make])


def get_env_dims(env: DummyVecEnv) -> Tuple[int, int]:
    obs_dim = int(np.prod(env.observation_space.shape))
    act_space = env.action_space
    action_dim = (
        act_space.n if hasattr(act_space, "n") else int(np.prod(act_space.shape))
    )
    return obs_dim, action_dim


def extract_level(env: DummyVecEnv, info) -> np.ndarray:
    inner = env.envs[0]
    while hasattr(inner, "env"):
        inner = inner.env
    if hasattr(inner, "_rep") and hasattr(inner._rep, "_map"):
        return np.array(inner._rep._map, dtype=int)
    if isinstance(info, dict) and "level" in info:
        return np.array(info["level"], dtype=int)
    if isinstance(info, (list, tuple)) and info and "level" in info[0]:
        return np.array(info[0]["level"], dtype=int)
    print("  [WARN] Could not extract level – returning zeros")
    return np.zeros((10, 10), dtype=int)


# ==========================================================================
# Optional fast task-adaptation (MAML inner loop)
# ==========================================================================


def adapt_policy(
    policy: MAMLPolicy,
    env: DummyVecEnv,
    inner_lr: float,
    inner_steps: int,
    n_trajectories: int,
    device: str,
) -> OrderedDict:
    from maml_trainer import collect_trajectories, compute_policy_loss

    adapted = OrderedDict(
        (name, param.clone()) for name, param in policy.named_parameters()
    )
    for k in range(inner_steps):
        traj = collect_trajectories(env, policy, n_trajectories, device, adapted)
        loss = compute_policy_loss(traj, policy, adapted)
        grads = torch.autograd.grad(loss, adapted.values(), allow_unused=True)
        adapted = OrderedDict(
            (name, p - inner_lr * (g if g is not None else torch.zeros_like(p)))
            for (name, p), g in zip(adapted.items(), grads)
        )
        print(f"    adapt step {k + 1}/{inner_steps}  loss={loss.item():.4f}")
    return adapted


# ==========================================================================
# CSV live-writer  (flushes one row per level immediately)
# ==========================================================================

_CSV_COLUMNS = [
    "level_id",
    "timestamp",
    "game",
    "algorithm",
    "adapt_steps",
    "reset_time_ms",
    "generation_time_ms",
    "extract_time_ms",
    "validation_time_ms",
    "metrics_time_ms",
    "solvability_time_ms",
    "save_time_ms",
    "total_time_ms",
    "steps",
    "mean_inference_ms",
    "std_inference_ms",
    "min_inference_ms",
    "max_inference_ms",
    "total_reward",
    "diversity",
    "complexity",
    "unique_tiles",
    "is_solvable",
    "ram_start_pct",
    "ram_end_pct",
    "ram_delta_pct",
    "cpu_start_pct",
    "cpu_end_pct",
    "cpu_delta_pct",
    "gpu_start_pct",
    "gpu_end_pct",
    "gpu_delta_pct",
]


class LiveCSVWriter:
    """Opens the CSV immediately, writes header, then flushes one row at a time."""

    def __init__(self, path: str):
        self.path = path
        self._fh = open(path, "w", newline="", encoding="utf-8")
        self._w = csv.DictWriter(
            self._fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore"
        )
        self._w.writeheader()
        self._fh.flush()
        print(f"[OK] Timing CSV  -> {path}  (flushed after every level)")

    def write(self, row: dict):
        self._w.writerow(row)
        self._fh.flush()

    def close(self):
        self._fh.close()


# ==========================================================================
# Main generation loop
# ==========================================================================


def generate_timed(
    checkpoint_path: str,
    game: str = "sokoban",
    representation: str = "narrow",
    n_levels: int = 20,
    max_steps: int = 500,
    deterministic: bool = True,
    adapt_steps: int = 0,
    inner_lr: float = 0.01,
    n_trajectories: int = 64,
    device: str = "auto",
    save_dir: str = "generated_levels/maml",
    log_file: str = "inference_timing_maml.csv",
    log_dir: str = "logs",
    experiment_name: str = None,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # ---- Experiment name -------------------------------------------------
    if experiment_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"MAML_inference_{game}_{ts}"

    # ---- Loggers ---------------------------------------------------------
    # 1. TrainingLogger  -> logs/<experiment_name>.csv  (step-level live log)
    training_logger = TrainingLogger(log_dir=log_dir, experiment_name=experiment_name)
    print(f"[OK] TrainingLogger -> {log_dir}/{experiment_name}.csv")

    # 2. LiveCSVWriter   -> --log-file (level-level timing CSV, flushed live)
    csv_writer = LiveCSVWriter(log_file)

    use_gpu = device == "cuda"
    resource_monitor = ResourceMonitor(use_gpu=use_gpu)

    print(f"\n{'=' * 70}")
    print(f"MAML TIMED INFERENCE")
    print(f"{'=' * 70}")
    print(f"  Checkpoint   : {checkpoint_path}")
    print(f"  Game         : {game}")
    print(f"  Device       : {device}")
    print(f"  Levels       : {n_levels}   Max steps: {max_steps}")
    print(f"  Adapt steps  : {adapt_steps}  (0 = direct meta-weights)")
    print(f"  Experiment   : {experiment_name}")
    print(f"{'=' * 70}\n")

    # ---- Build env, discover dims ----------------------------------------
    setup_start = time.perf_counter()
    env = build_env(game, representation, resource_monitor)
    obs_dim, act_dim = get_env_dims(env)
    print(f"[OK] Env dims -> obs={obs_dim}, actions={act_dim}")

    # ---- Load checkpoint -------------------------------------------------
    policy, config = load_maml_checkpoint(checkpoint_path, obs_dim, act_dim, device)

    # ---- Optional task adaptation ----------------------------------------
    adapted_params: Optional[OrderedDict] = None
    if adapt_steps > 0:
        print(f"\n[ADAPT] {adapt_steps} inner-loop steps on {game} ...")
        t0 = time.perf_counter()
        adapted_params = adapt_policy(
            policy,
            env,
            inner_lr=config.get("inner_lr", inner_lr),
            inner_steps=adapt_steps,
            n_trajectories=n_trajectories,
            device=device,
        )
        print(f"[OK] Adaptation done in {time.perf_counter() - t0:.2f}s")
    else:
        print("[INFO] Using meta-learned weights directly (no adaptation)\n")

    model = MAMLPolicyWrapper(policy, device, obs_dim, adapted_params)
    setup_time = time.perf_counter() - setup_start
    print(f"[OK] Setup complete in {setup_time:.3f}s\n")

    # ---- Generation loop -------------------------------------------------
    all_results = []

    for i in range(n_levels):
        level_id = i + 1
        print(f"Level {level_id}/{n_levels}:")

        # Reset
        reset_start = time.perf_counter()
        obs = env.reset()
        reset_time = time.perf_counter() - reset_start

        # Generate
        gen_start = time.perf_counter()
        res_start = resource_monitor.get_resources()
        done = False
        steps = 0
        total_reward = 0.0
        inference_times = []

        while not done and steps < max_steps:
            t0 = time.perf_counter()
            action, _ = model.predict(obs, deterministic=deterministic)
            inference_times.append(time.perf_counter() - t0)

            obs, reward, done, info = env.step(action)
            total_reward += float(reward)
            steps += 1

            # Log every 50 steps (and on done) via TrainingLogger.log_step()
            if steps % 50 == 0 or done:
                step_res = resource_monitor.get_resources()
                training_logger.log_step(
                    reward=float(reward),
                    resources=step_res,
                    content_metrics={
                        "level_id": level_id,
                        "total_reward": total_reward,
                        "done": int(bool(done)),
                    },
                )

        gen_time = time.perf_counter() - gen_start
        res_end = resource_monitor.get_resources()

        # Extract & validate level
        extract_start = time.perf_counter()
        level = extract_level(env, info)
        val_msg = "N/A"
        validation_time = 0.0

        if game == "sokoban":
            vs = time.perf_counter()
            is_valid, val_msg = is_valid_sokoban(level)
            validation_time = time.perf_counter() - vs

        extract_time = time.perf_counter() - extract_start

        # Metrics
        metrics_start = time.perf_counter()
        metrics = calculate_content_metrics(level)
        metrics_time = time.perf_counter() - metrics_start

        # Solvability flag from env info
        solve_start = time.perf_counter()
        if isinstance(info, dict):
            is_solvable = info.get("solvable", None)
        elif isinstance(info, (list, tuple)) and info:
            is_solvable = info[0].get("solvable", None)
        else:
            is_solvable = None
        solvability_time = time.perf_counter() - solve_start

        # Save files
        save_start = time.perf_counter()
        lpath = os.path.join(save_dir, f"level_{level_id:03d}")
        save_level(level, lpath + ".npy", format="npy")
        save_level(level, lpath + ".txt", format="txt")
        try:
            save_level_image(
                level, lpath + ".png", game=game, scale=25, show_grid=True, dpi=300
            )
        except Exception as e:
            print(f"  [WARN] Image save failed: {e}")
        save_time = time.perf_counter() - save_start

        total_time = (
            reset_time
            + gen_time
            + extract_time
            + validation_time
            + metrics_time
            + solvability_time
            + save_time
        )

        result = {
            "level_id": level_id,
            "timestamp": datetime.now().isoformat(),
            "game": game,
            "algorithm": "MAML",
            "adapt_steps": adapt_steps,
            "reset_time_ms": reset_time * 1000,
            "generation_time_ms": gen_time * 1000,
            "extract_time_ms": extract_time * 1000,
            "validation_time_ms": validation_time * 1000,
            "metrics_time_ms": metrics_time * 1000,
            "solvability_time_ms": solvability_time * 1000,
            "save_time_ms": save_time * 1000,
            "total_time_ms": total_time * 1000,
            "steps": steps,
            "mean_inference_ms": np.mean(inference_times) * 1000,
            "std_inference_ms": np.std(inference_times) * 1000,
            "min_inference_ms": np.min(inference_times) * 1000,
            "max_inference_ms": np.max(inference_times) * 1000,
            "total_reward": total_reward,
            "diversity": metrics["diversity"],
            "complexity": metrics["complexity"],
            "unique_tiles": metrics["unique_tiles"],
            "is_solvable": is_solvable,
            "ram_start_pct": res_start["ram_percent"],
            "ram_end_pct": res_end["ram_percent"],
            "ram_delta_pct": res_end["ram_percent"] - res_start["ram_percent"],
            "cpu_start_pct": res_start["cpu_percent"],
            "cpu_end_pct": res_end["cpu_percent"],
            "cpu_delta_pct": res_end["cpu_percent"] - res_start["cpu_percent"],
            "gpu_start_pct": res_start["gpu_mem_percent"],
            "gpu_end_pct": res_end["gpu_mem_percent"],
            "gpu_delta_pct": res_end["gpu_mem_percent"] - res_start["gpu_mem_percent"],
        }

        # Flush this row immediately — no data lost even if run crashes
        csv_writer.write(result)
        all_results.append(result)

        # Mark episode end and save TrainingLogger CSV
        training_logger.log_episode_end()
        training_logger.save()

        # Console summary
        print(
            f"  Total: {total_time * 1000:.1f} ms | "
            f"gen: {gen_time * 1000:.1f} ms ({steps} steps) | "
            f"infer/step: {np.mean(inference_times) * 1000:.3f} ms"
        )
        print(
            f"  diversity={metrics['diversity']:.3f}  "
            f"complexity={metrics['complexity']:.3f}  "
            f"solvable={is_solvable}"
        )
        if game == "sokoban":
            print(f"  sokoban check : {val_msg}")
        print(
            f"  CPU={res_end['cpu_percent']:.0f}%  "
            f"RAM={res_end['ram_percent']:.0f}%  "
            f"GPU={res_end['gpu_mem_percent']:.0f}%"
        )
        print(f"  Saved -> {lpath}.*\n")

    # ---- Cleanup ---------------------------------------------------------
    csv_writer.close()
    env.close()

    df = pd.DataFrame(all_results)
    _print_summary(df)
    _write_latex(df, log_file.replace(".csv", "_table.tex"))

    print(f"[OK] Timing CSV  -> {log_file}")
    print(f"[OK] Live log    -> {log_dir}/{experiment_name}.csv")
    print(f"[OK] LaTeX table -> {log_file.replace('.csv', '_table.tex')}")
    print(f"[OK] Levels      -> {save_dir}/\n")
    return df


# ==========================================================================
# Summary / LaTeX helpers
# ==========================================================================


def _print_summary(df: pd.DataFrame):
    print(f"\n{'=' * 70}")
    print("SUMMARY STATISTICS")
    print(f"{'=' * 70}")

    print("\nTIMING PERFORMANCE:")
    print(
        f"  Total time (mean)         : "
        f"{df['total_time_ms'].mean():.2f} ± {df['total_time_ms'].std():.2f} ms"
    )
    print(
        f"  Generation time (mean)    : "
        f"{df['generation_time_ms'].mean():.2f} ± {df['generation_time_ms'].std():.2f} ms"
    )
    print(
        f"  Per-step inference (mean) : "
        f"{df['mean_inference_ms'].mean():.3f} ± {df['mean_inference_ms'].std():.3f} ms"
    )
    print(
        f"  Solvability check (mean)  : "
        f"{df['solvability_time_ms'].mean():.2f} ± {df['solvability_time_ms'].std():.2f} ms"
    )

    print("\nGENERATION QUALITY:")
    print(f"  Mean steps      : {df['steps'].mean():.1f} ± {df['steps'].std():.1f}")
    print(
        f"  Mean reward     : {df['total_reward'].mean():.2f} ± {df['total_reward'].std():.2f}"
    )
    print(
        f"  Mean diversity  : {df['diversity'].mean():.3f} ± {df['diversity'].std():.3f}"
    )
    print(
        f"  Mean complexity : {df['complexity'].mean():.3f} ± {df['complexity'].std():.3f}"
    )

    if df["is_solvable"].notna().any():
        rate = df["is_solvable"].sum() / len(df) * 100
        print(f"  Solvability     : {rate:.1f}%")

    print("\nRESOURCE USAGE:")
    print(f"  RAM delta (mean) : {df['ram_delta_pct'].mean():.2f}%")
    print(f"  CPU usage (mean) : {df['cpu_end_pct'].mean():.1f}%")
    print(f"  GPU usage (mean) : {df['gpu_end_pct'].mean():.1f}%")
    print(f"{'=' * 70}\n")


def _write_latex(df: pd.DataFrame, tex_path: str):
    rows = [
        ("Total Time (ms)", "total_time_ms"),
        ("Generation Time (ms)", "generation_time_ms"),
        ("Per-Step Inference (ms)", "mean_inference_ms"),
        ("Steps", "steps"),
        ("Diversity", "diversity"),
        ("Complexity", "complexity"),
    ]
    with open(tex_path, "w") as f:
        f.write("% MAML Inference Timing — auto-generated\n")
        f.write("\\begin{table}[t]\n\\centering\n")
        f.write("\\caption{MAML Inference Timing Performance}\n")
        f.write("\\label{tab:maml_inference_timing}\n")
        f.write("\\begin{tabular}{lcc}\\hline\n")
        f.write("Metric & Mean & Std Dev \\\\\n\\hline\n")
        for label, col in rows:
            f.write(f"{label} & {df[col].mean():.2f} & {df[col].std():.2f} \\\\\n")
        if df["is_solvable"].notna().any():
            rate = df["is_solvable"].sum() / len(df) * 100
            f.write(f"Solvability (\\%) & {rate:.1f} & -- \\\\\n")
        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")
    print(f"[OK] LaTeX table -> {tex_path}")


# ==========================================================================
# CLI
# ==========================================================================


def main():
    parser = argparse.ArgumentParser(description="Timed MAML inference – paper metrics")

    parser.add_argument(
        "checkpoint",
        help="Path to MAML .pt checkpoint "
        "(e.g. checkpoints/sokoban_MAML_inference/best_meta_model.pt)",
    )
    parser.add_argument("--game", default="sokoban")
    parser.add_argument("--representation", default="narrow")
    parser.add_argument("--n-levels", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Stochastic policy (default: deterministic)",
    )
    parser.add_argument(
        "--adapt-steps",
        type=int,
        default=0,
        help="MAML inner-loop adaptation steps before inference "
        "(0 = use meta-weights directly, fastest)",
    )
    parser.add_argument("--inner-lr", type=float, default=0.01)
    parser.add_argument("--n-trajectories", type=int, default=64)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--save-dir", default="generated_levels/maml")
    parser.add_argument(
        "--log-file",
        default="inference_timing_maml.csv",
        help="Per-level timing CSV (row flushed after every level)",
    )
    parser.add_argument(
        "--log-dir", default="logs", help="Directory for TrainingLogger step-level CSV"
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Experiment tag used in TrainingLogger filename",
    )

    args = parser.parse_args()

    generate_timed(
        checkpoint_path=args.checkpoint,
        game=args.game,
        representation=args.representation,
        n_levels=args.n_levels,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        adapt_steps=args.adapt_steps,
        inner_lr=args.inner_lr,
        n_trajectories=args.n_trajectories,
        device=args.device,
        save_dir=args.save_dir,
        log_file=args.log_file,
        log_dir=args.log_dir,
        experiment_name=args.experiment_name,
    )


if __name__ == "__main__":
    main()
