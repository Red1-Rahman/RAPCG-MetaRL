# utils.py
"""
RAPCG-MetaRL Utilities
Resource monitoring, logging, and training utilities.
"""
import os
import time
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class ResourceMonitor:
    """
    Monitor system resources (CPU, RAM, GPU) during training.
    """
    
    def __init__(self, use_gpu: bool = True):
        """
        Initialize resource monitor.
        
        Args:
            use_gpu: Whether to monitor GPU (requires pynvml)
        """
        self.use_gpu = use_gpu
        self.gpu_available = False
        
        if use_gpu:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.pynvml = pynvml
                self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.gpu_available = True
                print("[OK] GPU monitoring enabled")
            except Exception as e:
                print(f"[WARNING] GPU monitoring unavailable: {e}")
                self.gpu_available = False
    
    def get_resources(self) -> Dict[str, float]:
        """
        Get current resource usage.
        
        Returns:
            Dictionary with CPU, RAM, GPU metrics
        """
        resources = {
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'ram_percent': psutil.virtual_memory().percent,
            'ram_used_gb': psutil.virtual_memory().used / (1024**3),
            'ram_available_gb': psutil.virtual_memory().available / (1024**3),
        }
        
        if self.gpu_available:
            try:
                gpu_info = self.pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_util = self.pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                
                resources.update({
                    'gpu_util_percent': gpu_util.gpu,
                    'gpu_mem_used_mb': gpu_info.used / (1024**2),
                    'gpu_mem_total_mb': gpu_info.total / (1024**2),
                    'gpu_mem_percent': (gpu_info.used / gpu_info.total) * 100,
                })
            except Exception as e:
                print(f"Warning: GPU metrics error: {e}")
        else:
            resources.update({
                'gpu_util_percent': 0.0,
                'gpu_mem_used_mb': 0.0,
                'gpu_mem_total_mb': 0.0,
                'gpu_mem_percent': 0.0,
            })
        
        return resources
    
    def check_resource_pressure(self, thresholds: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
        """
        Check if resources are under pressure.
        
        Args:
            thresholds: Custom thresholds for CPU, RAM, GPU
            
        Returns:
            Tuple of (is_under_pressure, message)
        """
        if thresholds is None:
            thresholds = {
                'cpu_percent': 90.0,
                'ram_percent': 90.0,
                'gpu_mem_percent': 85.0,
                'gpu_util_percent': 85.0,
            }
        
        resources = self.get_resources()
        
        for key, threshold in thresholds.items():
            if key in resources and resources[key] > threshold:
                return True, f"{key} exceeds threshold: {resources[key]:.1f}% > {threshold}%"
        
        return False, "Resources normal"
    
    def __del__(self):
        """Cleanup GPU monitoring."""
        if self.gpu_available:
            try:
                self.pynvml.nvmlShutdown()
            except:
                pass


class TrainingLogger:
    """
    Log training metrics including rewards, resources, and content quality.
    """
    
    def __init__(self, log_dir: str = 'logs', experiment_name: Optional[str] = None):
        """
        Initialize training logger.
        
        Args:
            log_dir: Directory to save logs
            experiment_name: Name of experiment (auto-generated if None)
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.experiment_name = experiment_name
        self.log_file = os.path.join(log_dir, f"{experiment_name}.csv")
        
        self.episodes = []
        self.steps = []
        self.rewards = []
        self.resource_logs = []
        self.content_metrics = []
        self.timestamps = []
        self.actions = []  # NEW: Track actions
        self.penalties = []  # NEW: Track penalties per step
        
        self.episode_count = 0
        self.total_steps = 0
        self.start_time = time.time()
        
        print(f"[OK] Logger initialized: {self.log_file}")
    
    def log_step(self, reward: float, resources: Dict[str, float], 
                  content_metrics: Optional[Dict[str, float]] = None,
                  action: Optional[int] = None,
                  penalty_info: Optional[Dict[str, float]] = None):
        """
        Log a single training step.
        
        Args:
            reward: Step reward (shaped)
            resources: Resource usage dict
            content_metrics: Content quality metrics
            action: Action taken by agent
            penalty_info: Penalty breakdown (ram_penalty, cpu_penalty, etc.)
        """
        self.steps.append(self.total_steps)
        self.episodes.append(self.episode_count)
        self.rewards.append(reward)
        self.resource_logs.append(resources)
        self.content_metrics.append(content_metrics or {})
        self.timestamps.append(time.time() - self.start_time)
        self.actions.append(action if action is not None else -1)
        self.penalties.append(penalty_info or {})
        
        self.total_steps += 1
    
    def log_episode_end(self):
        """Mark end of episode."""
        self.episode_count += 1
    
    def save(self):
        """Save logs to CSV file."""
        if not self.steps:
            print("No data to save")
            return
        
        # Prepare dataframe
        data = {
            'episode': self.episodes,
            'step': self.steps,
            'reward': self.rewards,
            'timestamp': self.timestamps,
        }
        
        # Add resource metrics
        if self.resource_logs:
            for key in self.resource_logs[0].keys():
                data[key] = [r.get(key, 0) for r in self.resource_logs]
        
        # Add content metrics
        if self.content_metrics and any(self.content_metrics):
            all_keys = set()
            for cm in self.content_metrics:
                all_keys.update(cm.keys())
            
            for key in all_keys:
                data[f'content_{key}'] = [cm.get(key, 0) for cm in self.content_metrics]
        
        # Add action tracking
        if self.actions:
            data['action'] = self.actions
        
        # Add penalty breakdown
        if self.penalties and any(self.penalties):
            penalty_keys = set()
            for p in self.penalties:
                penalty_keys.update(p.keys())
            
            for key in penalty_keys:
                data[f'penalty_{key}'] = [p.get(key, 0.0) for p in self.penalties]
        
        df = pd.DataFrame(data)
        df.to_csv(self.log_file, index=False)
        print(f"[OK] Logs saved to {self.log_file} ({len(df)} steps)")
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get summary statistics.
        
        Returns:
            Dictionary of training statistics
        """
        if not self.rewards:
            return {}
        
        rewards_array = np.array(self.rewards)
        
        stats = {
            'total_episodes': self.episode_count,
            'total_steps': self.total_steps,
            'mean_reward': np.mean(rewards_array),
            'std_reward': np.std(rewards_array),
            'min_reward': np.min(rewards_array),
            'max_reward': np.max(rewards_array),
            'elapsed_time': time.time() - self.start_time,
        }
        
        return stats
    
    def print_stats(self):
        """Print summary statistics."""
        stats = self.get_stats()
        
        print("\n" + "="*50)
        print(f"Training Statistics - {self.experiment_name}")
        print("="*50)
        for key, value in stats.items():
            if 'time' in key:
                print(f"{key:20s}: {value:.2f}s")
            else:
                print(f"{key:20s}: {value:.4f}")
        print("="*50 + "\n")
    
    def analyze_action_penalty_correlation(self, recent_steps: int = 1000) -> Dict:
        """
        Analyze correlation between actions and resource penalties.
        
        Args:
            recent_steps: Number of recent steps to analyze
        
        Returns:
            Dictionary with correlation analysis
        """
        if not self.actions or not self.penalties:
            return {}
        
        # Get recent data
        start_idx = max(0, len(self.actions) - recent_steps)
        recent_actions = self.actions[start_idx:]
        recent_penalties = self.penalties[start_idx:]
        
        # Extract penalty types
        penalty_types = ['ram_penalty', 'cpu_penalty', 'gpu_penalty', 'total_penalty']
        
        # Group by action
        action_stats = {}
        unique_actions = set(a for a in recent_actions if a >= 0)
        
        for action in unique_actions:
            # Find indices where this action was taken
            action_indices = [i for i, a in enumerate(recent_actions) if a == action]
            
            if not action_indices:
                continue
            
            action_stats[action] = {
                'count': len(action_indices),
                'avg_penalties': {},
                'max_penalties': {},
            }
            
            for penalty_type in penalty_types:
                penalties = [recent_penalties[i].get(penalty_type, 0.0) 
                           for i in action_indices]
                
                if penalties:
                    action_stats[action]['avg_penalties'][penalty_type] = np.mean(penalties)
                    action_stats[action]['max_penalties'][penalty_type] = np.max(penalties)
        
        return action_stats
    
    def get_high_penalty_actions(self, top_n: int = 5, penalty_type: str = 'ram_penalty',
                                  recent_steps: int = 1000) -> List[Tuple[int, float]]:
        """
        Get actions that cause highest penalties.
        
        Args:
            top_n: Number of top actions to return
            penalty_type: Type of penalty to analyze ('ram_penalty', 'cpu_penalty', etc.)
            recent_steps: Number of recent steps to analyze
        
        Returns:
            List of (action, avg_penalty) tuples, sorted by penalty (descending)
        """
        stats = self.analyze_action_penalty_correlation(recent_steps)
        
        if not stats:
            return []
        
        # Extract action-penalty pairs
        action_penalties = []
        for action, data in stats.items():
            avg_penalty = data['avg_penalties'].get(penalty_type, 0.0)
            action_penalties.append((action, avg_penalty))
        
        # Sort by penalty (descending)
        action_penalties.sort(key=lambda x: x[1], reverse=True)
        
        return action_penalties[:top_n]


def calculate_fps(start_time: float, steps: int) -> float:
    """
    Calculate frames per second (steps per second).
    
    Args:
        start_time: Start timestamp
        steps: Number of steps
        
    Returns:
        FPS value
    """
    elapsed = time.time() - start_time
    return steps / elapsed if elapsed > 0 else 0.0


def estimate_training_time(current_steps: int, total_steps: int, 
                          start_time: float) -> str:
    """
    Estimate remaining training time.
    
    Args:
        current_steps: Steps completed
        total_steps: Total steps to complete
        start_time: Training start time
        
    Returns:
        Formatted time string
    """
    if current_steps == 0:
        return "Calculating..."
    
    elapsed = time.time() - start_time
    steps_per_sec = current_steps / elapsed
    remaining_steps = total_steps - current_steps
    remaining_secs = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
    
    hours = int(remaining_secs // 3600)
    minutes = int((remaining_secs % 3600) // 60)
    seconds = int(remaining_secs % 60)
    
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def create_checkpoint_dir(base_dir: str = 'checkpoints', 
                          experiment_name: Optional[str] = None) -> str:
    """
    Create checkpoint directory.
    
    Args:
        base_dir: Base checkpoint directory
        experiment_name: Experiment name
        
    Returns:
        Path to checkpoint directory
    """
    if experiment_name is None:
        experiment_name = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    checkpoint_dir = os.path.join(base_dir, experiment_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    return checkpoint_dir
