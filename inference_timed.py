# inference_timed.py
"""
RAPCG-MetaRL Timed Inference Script
Generate levels with detailed timing measurements for academic paper reporting.
Logs: per-level inference time, resource usage, solvability metrics.
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
import time
from datetime import datetime
import json

# Add project paths
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)
sys.path.append(os.path.join(project_root, 'gym-pcgrl'))

from utils import ResourceMonitor
from wrappers.pcgrl_env import make_pcgrl_env
from wrappers.helper import calculate_content_metrics, save_level
from visualize_levels import save_level_image, render_level
from sokoban_utils import validate_and_fix_sokoban, is_valid_sokoban

try:
    from stable_baselines3 import PPO, A2C
except ImportError:
    print("Error: stable-baselines3 not installed")
    sys.exit(1)


class TimedLevelGenerator:
    """
    Generate levels with comprehensive timing and performance metrics.
    """
    
    def __init__(self, model_path, game='sokoban', representation='narrow', 
                 algorithm='PPO', device='auto', trust_model=True):
        """
        Initialize timed generator.
        
        Args:
            trust_model: If True, trust trained model output without aggressive validation.
                        KEY STRATEGY: Let reward shaping during training do its job.
        """
        self.model_path = model_path
        self.game = game
        self.representation = representation
        self.algorithm = algorithm
        self.device = device
        self.trust_model = trust_model
        
        # Create resource monitor
        use_gpu_monitor = (device == 'cuda' or (device == 'auto' and self._is_cuda_available()))
        self.resource_monitor = ResourceMonitor(use_gpu=use_gpu_monitor)
        
        # Timing logs
        self.timing_logs = []
        self.level_metrics = []
        
        # Load model and environment
        print(f"\n{'='*70}")
        print(f"TIMED INFERENCE SETUP")
        print(f"{'='*70}")
        print(f"Model: {model_path}")
        print(f"Game: {game}")
        print(f"Device: {device}")
        print(f"GPU Monitoring: {'ENABLED' if use_gpu_monitor else 'DISABLED'}")
        
        # Start timing: Model loading
        load_start = time.perf_counter()
        
        # Create environment
        self.env = make_pcgrl_env(
            resource_monitor=self.resource_monitor,
            game=game, 
            representation=representation,
            use_solvability_config=True
        )
        
        # Load model
        if algorithm == 'PPO':
            self.model = PPO.load(model_path, device=device)
        elif algorithm == 'A2C':
            self.model = A2C.load(model_path, device=device)
        else:
            raise ValueError(f"Unsupported algorithm: {algorithm}")
        
        load_time = time.perf_counter() - load_start
        print(f"\n[OK] Setup complete in {load_time:.3f}s")
        print(f"{'='*70}\n")
    
    def _is_cuda_available(self):
        """Check CUDA availability."""
        try:
            import torch
            return torch.cuda.is_available()
        except:
            return False
    
    def generate_with_timing(self, n_levels=10, max_steps=1000, deterministic=True, 
                            save_dir='generated_levels', log_file='inference_timing.csv'):
        """
        Generate levels with detailed timing measurements.
        
        Returns:
            DataFrame with timing and performance metrics
        """
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"{'='*70}")
        print(f"GENERATING {n_levels} LEVELS WITH TIMING")
        print(f"{'='*70}\n")
        
        all_results = []
        
        for i in range(n_levels):
            level_id = i + 1
            print(f"Level {level_id}/{n_levels}:")
            
            # === TIMING: Environment Reset ===
            reset_start = time.perf_counter()
            obs = self.env.reset()
            reset_time = time.perf_counter() - reset_start
            
            # === TIMING: Level Generation ===
            gen_start = time.perf_counter()
            resources_start = self.resource_monitor.get_resources()
            
            done = False
            steps = 0
            total_reward = 0
            inference_times = []
            
            while not done and steps < max_steps:
                # Time individual inference step
                step_start = time.perf_counter()
                action, _ = self.model.predict(obs, deterministic=deterministic)
                inference_time = time.perf_counter() - step_start
                inference_times.append(inference_time)
                
                obs, reward, done, info = self.env.step(action)
                total_reward += reward
                steps += 1
            
            gen_time = time.perf_counter() - gen_start
            resources_end = self.resource_monitor.get_resources()
            
            # === TIMING: Level Extraction ===
            extract_start = time.perf_counter()
            level = self._extract_level(info)
            
            # === TIMING: Sokoban Validation & Correction ===
            validation_time = 0.0
            corrections = {}
            if self.game == 'sokoban':
                validate_start = time.perf_counter()
                if self.trust_model:
                    # KEY STRATEGY: Trust the trained model, only validate completeness
                    # Don't fix/modify - let reward shaping during training do its job
                    is_valid, msg = is_valid_sokoban(level)
                    corrections = {
                        'trusted_model': True,
                        'valid': is_valid,
                        'message': msg,
                        'final_players': np.sum(level == 2),
                        'final_crates': np.sum(level == 3),
                        'final_targets': np.sum(level == 4)
                    }
                else:
                    # Aggressive validation - may remove crates/targets
                    level, corrections = validate_and_fix_sokoban(level, min_crates=2, enforce_all_rules=True)
                validation_time = time.perf_counter() - validate_start
            
            extract_time = time.perf_counter() - extract_start
            
            # === TIMING: Metrics Calculation ===
            metrics_start = time.perf_counter()
            metrics = calculate_content_metrics(level)
            metrics_time = time.perf_counter() - metrics_start
            
            # === TIMING: Solvability Check (if Sokoban) ===
            solvability_time = 0.0
            is_solvable = None
            if self.game == 'sokoban':
                solve_start = time.perf_counter()
                is_solvable = info.get('solvable', None)
                solvability_time = time.perf_counter() - solve_start
            
            # === TIMING: Save & Visualize ===
            save_start = time.perf_counter()
            level_path = os.path.join(save_dir, f'level_{level_id:03d}')
            save_level(level, level_path + '.npy', format='npy')
            save_level(level, level_path + '.txt', format='txt')
            save_level_image(level, level_path + '.png', game=self.game, 
                           scale=25, show_grid=True, dpi=300)
            save_time = time.perf_counter() - save_start
            
            # === TOTAL TIME ===
            total_time = reset_time + gen_time + extract_time + validation_time + metrics_time + solvability_time + save_time
            
            # Calculate resource deltas
            ram_delta = resources_end['ram_percent'] - resources_start['ram_percent']
            cpu_delta = resources_end['cpu_percent'] - resources_start['cpu_percent']
            gpu_delta = resources_end['gpu_mem_percent'] - resources_start['gpu_mem_percent']
            
            # Compile results
            result = {
                'level_id': level_id,
                'timestamp': datetime.now().isoformat(),
                'game': self.game,
                'algorithm': self.algorithm,
                
                # Timing breakdown (milliseconds for paper)
                'reset_time_ms': reset_time * 1000,
                'generation_time_ms': gen_time * 1000,
                'extract_time_ms': extract_time * 1000,
                'validation_time_ms': validation_time * 1000,
                'metrics_time_ms': metrics_time * 1000,
                'solvability_time_ms': solvability_time * 1000,
                'save_time_ms': save_time * 1000,
                'total_time_ms': total_time * 1000,
                
                # Inference statistics
                'steps': steps,
                'mean_inference_ms': np.mean(inference_times) * 1000,
                'std_inference_ms': np.std(inference_times) * 1000,
                'min_inference_ms': np.min(inference_times) * 1000,
                'max_inference_ms': np.max(inference_times) * 1000,
                
                # Quality metrics
                'total_reward': total_reward,
                'diversity': metrics['diversity'],
                'complexity': metrics['complexity'],
                'unique_tiles': metrics['unique_tiles'],
                'is_solvable': is_solvable,
                
                # Resource usage
                'ram_start_pct': resources_start['ram_percent'],
                'ram_end_pct': resources_end['ram_percent'],
                'ram_delta_pct': ram_delta,
                'cpu_start_pct': resources_start['cpu_percent'],
                'cpu_end_pct': resources_end['cpu_percent'],
                'cpu_delta_pct': cpu_delta,
                'gpu_start_pct': resources_start['gpu_mem_percent'],
                'gpu_end_pct': resources_end['gpu_mem_percent'],
                'gpu_delta_pct': gpu_delta,
            }
            
            all_results.append(result)
            
            # Print summary
            print(f"  Total time: {total_time*1000:.1f} ms")
            print(f"    - Generation: {gen_time*1000:.1f} ms ({steps} steps)")
            print(f"    - Mean inference: {np.mean(inference_times)*1000:.2f} ms/step")
            if validation_time > 0:
                print(f"    - Validation: {validation_time*1000:.1f} ms")
            print(f"    - Solvability check: {solvability_time*1000:.1f} ms")
            print(f"  Quality: diversity={metrics['diversity']:.3f}, complexity={metrics['complexity']:.3f}")
            if is_solvable is not None:
                print(f"  Solvable: {is_solvable}")
            print(f"  Saved: {level_path}.*\n")
        
        # Create DataFrame
        df = pd.DataFrame(all_results)
        
        # Save to CSV
        df.to_csv(log_file, index=False)
        print(f"\n{'='*70}")
        print(f"[OK] Timing log saved: {log_file}")
        print(f"{'='*70}\n")
        
        # Print summary statistics
        self._print_summary(df)
        
        return df
    
    def _extract_level(self, info):
        """Extract level from environment."""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        
        if hasattr(env, '_rep') and hasattr(env._rep, '_map'):
            return np.array(env._rep._map, dtype=int)
        
        if 'level' in info:
            return np.array(info['level'], dtype=int)
        
        print("Warning: Could not extract level")
        return np.zeros((10, 10), dtype=int)
    
    def _print_summary(self, df):
        """Print summary statistics for paper."""
        print("SUMMARY STATISTICS (for paper)")
        print("="*70)
        
        print("\n📊 TIMING PERFORMANCE:")
        print(f"  Total time (mean):        {df['total_time_ms'].mean():.2f} ± {df['total_time_ms'].std():.2f} ms")
        print(f"  Generation time (mean):   {df['generation_time_ms'].mean():.2f} ± {df['generation_time_ms'].std():.2f} ms")
        print(f"  Inference per step (mean):{df['mean_inference_ms'].mean():.2f} ± {df['mean_inference_ms'].std():.2f} ms")
        print(f"  Solvability check (mean): {df['solvability_time_ms'].mean():.2f} ± {df['solvability_time_ms'].std():.2f} ms")
        
        print("\n🎮 GENERATION QUALITY:")
        print(f"  Mean steps:     {df['steps'].mean():.1f} ± {df['steps'].std():.1f}")
        print(f"  Mean reward:    {df['total_reward'].mean():.2f} ± {df['total_reward'].std():.2f}")
        print(f"  Mean diversity: {df['diversity'].mean():.3f} ± {df['diversity'].std():.3f}")
        print(f"  Mean complexity:{df['complexity'].mean():.3f} ± {df['complexity'].std():.3f}")
        
        if 'was_corrected' in df.columns:
            corrected_count = df['was_corrected'].sum()
            print(f"  Levels corrected: {corrected_count}/{len(df)} ({corrected_count/len(df)*100:.1f}%)")
        
        if 'is_solvable' in df.columns and df['is_solvable'].notna().any():
            solvable_rate = df['is_solvable'].sum() / len(df) * 100
            print(f"  Solvability rate: {solvable_rate:.1f}%")
            print(f"  Solvability rate: {solvable_rate:.1f}%")
        
        print("\n💻 RESOURCE USAGE:")
        print(f"  RAM delta (mean):  {df['ram_delta_pct'].mean():.2f}%")
        print(f"  CPU usage (mean):  {df['cpu_end_pct'].mean():.1f}%")
        print(f"  GPU usage (mean):  {df['gpu_end_pct'].mean():.1f}%")
        
        print("\n" + "="*70)
        
        # Generate LaTeX table snippet
        self._generate_latex_table(df)
    
    def _generate_latex_table(self, df):
        """Generate LaTeX table for paper."""
        latex_file = 'inference_timing_table.tex'
        
        with open(latex_file, 'w') as f:
            f.write("% LaTeX table for paper - Inference Timing Results\n")
            f.write("\\begin{table}[t]\n")
            f.write("\\centering\n")
            f.write("\\caption{Inference Timing Performance}\n")
            f.write("\\label{tab:inference_timing}\n")
            f.write("\\begin{tabular}{lcc}\n")
            f.write("\\hline\n")
            f.write("Metric & Mean & Std Dev \\\\\n")
            f.write("\\hline\n")
            f.write(f"Total Time (ms) & {df['total_time_ms'].mean():.2f} & {df['total_time_ms'].std():.2f} \\\\\n")
            f.write(f"Generation Time (ms) & {df['generation_time_ms'].mean():.2f} & {df['generation_time_ms'].std():.2f} \\\\\n")
            f.write(f"Per-Step Inference (ms) & {df['mean_inference_ms'].mean():.2f} & {df['mean_inference_ms'].std():.2f} \\\\\n")
            f.write(f"Steps & {df['steps'].mean():.1f} & {df['steps'].std():.1f} \\\\\n")
            f.write(f"Diversity & {df['diversity'].mean():.3f} & {df['diversity'].std():.3f} \\\\\n")
            f.write(f"Complexity & {df['complexity'].mean():.3f} & {df['complexity'].std():.3f} \\\\\n")
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"[OK] LaTeX table saved: {latex_file}")
    
    def close(self):
        """Close environment."""
        self.env.close()


def main():
    """Main inference function with timing."""
    parser = argparse.ArgumentParser(description='Timed level generation for paper')
    
    parser.add_argument('model_path', type=str,
                       help='Path to trained model (.zip file)')
    parser.add_argument('--game', type=str, default='sokoban',
                       help='Game environment')
    parser.add_argument('--representation', type=str, default='narrow',
                       help='Representation type')
    parser.add_argument('--algorithm', type=str, default='PPO',
                       choices=['PPO', 'A2C'],
                       help='RL algorithm')
    parser.add_argument('--n-levels', type=int, default=10,
                       help='Number of levels to generate')
    parser.add_argument('--max-steps', type=int, default=1000,
                       help='Maximum steps per level')
    parser.add_argument('--stochastic', action='store_true',
                       help='Use stochastic policy')
    parser.add_argument('--trust-model', action='store_true', default=True,
                       help='Trust trained model without aggressive validation (KEY STRATEGY - recommended)')
    parser.add_argument('--aggressive-validation', dest='trust_model', action='store_false',
                       help='Use aggressive validation (may remove crates/targets)')
    parser.add_argument('--save-dir', type=str, default='generated_levels',
                       help='Directory to save levels')
    parser.add_argument('--log-file', type=str, default='inference_timing.csv',
                       help='CSV file for timing logs')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device for inference')
    
    args = parser.parse_args()
    
    # Create generator
    generator = TimedLevelGenerator(
        model_path=args.model_path,
        game=args.game,
        representation=args.representation,
        algorithm=args.algorithm,
        device=args.device,
        trust_model=args.trust_model
    )
    
    # Generate levels with timing
    df = generator.generate_with_timing(
        n_levels=args.n_levels,
        max_steps=args.max_steps,
        deterministic=not args.stochastic,
        save_dir=args.save_dir,
        log_file=args.log_file
    )
    
    print(f"\n[OK] Generated {len(df)} levels")
    print(f"[OK] Logs saved to: {args.log_file}")
    print(f"[OK] LaTeX table: inference_timing_table.tex")
    
    generator.close()


if __name__ == '__main__':
    main()
