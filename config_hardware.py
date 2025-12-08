"""
Hardware-Optimized Configuration for RAPCG-MetaRL
Optimized for:
- CPU: Intel i5-13500 (14 cores, 20 threads)
- GPU: RTX 3060 Ti (8GB VRAM) - Currently PyTorch CPU-only
- RAM: 16GB DDR4-3600
- Storage: Samsung 980 PRO 1TB NVMe SSD
"""

# System Specifications
HARDWARE_CONFIG = {
    'cpu': {
        'model': 'Intel i5-13500',
        'cores': 14,
        'threads': 20,
        'base_clock': 2.5,  # GHz
    },
    'gpu': {
        'model': 'NVIDIA RTX 3060 Ti',
        'vram_gb': 8,
        'cuda_version': '12.6',
        'driver_version': '560.94',
        'available': False,  # PyTorch is CPU-only currently
    },
    'ram': {
        'total_gb': 16,
        'speed_mhz': 3600,
        'type': 'DDR4',
    },
    'storage': {
        'model': 'Samsung 980 PRO 1TB',
        'type': 'NVMe SSD',
    }
}

# ============================================================================
# OPTIMIZED TRAINING CONFIGURATION
# ============================================================================

# Parallel Environment Configuration
# With 20 threads and 16GB RAM, we can safely run multiple environments
# Each environment uses ~200-500MB RAM depending on the game
PARALLEL_ENV_CONFIG = {
    'n_envs_light': 8,      # For quick testing (zelda, binary)
    'n_envs_medium': 6,     # For medium complexity (sokoban)
    'n_envs_heavy': 4,      # For heavy games or limited RAM scenarios
    'recommended': 6,       # Balanced for most use cases
}

# CPU Thread Allocation
# Leave some threads for system and monitoring
CPU_CONFIG = {
    'n_workers': 16,        # Max parallel workers (leave 4 threads for system)
    'torch_threads': 12,    # PyTorch intraop parallelism
    'monitoring_interval': 1.0,  # Resource monitoring interval (seconds)
}

# Memory Configuration (16GB total)
MEMORY_CONFIG = {
    'max_ram_percent': 70,              # Max 70% RAM usage (~11.2GB)
    'per_env_mb': 400,                  # Estimated RAM per environment
    'buffer_size_small': 50000,         # For quick tests
    'buffer_size_medium': 100000,       # Balanced
    'buffer_size_large': 200000,        # Max for 16GB RAM
    'batch_size_small': 64,
    'batch_size_medium': 128,
    'batch_size_large': 256,
}

# GPU Configuration (Currently CPU-only)
# Note: To enable GPU training, install PyTorch with CUDA:
#   pip uninstall torch
#   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
GPU_CONFIG = {
    'enabled': False,       # Set to True after installing CUDA PyTorch
    'device': 'cpu',        # Will be 'cuda' when GPU is enabled
    'mixed_precision': False,  # Enable after GPU setup for faster training
}

# Storage Configuration
# NVMe SSD allows fast checkpoint saving and large replay buffers
STORAGE_CONFIG = {
    'checkpoint_dir': 'checkpoints',
    'log_dir': 'logs',
    'data_dir': 'data',
    'save_freq': 10000,     # Save checkpoint every N steps
    'keep_checkpoints': 5,  # Keep last 5 checkpoints (auto-cleanup)
}

# ============================================================================
# RESOURCE MONITORING THRESHOLDS
# ============================================================================

RESOURCE_THRESHOLDS = {
    'cpu_percent': 85,      # Alert if CPU > 85%
    'ram_percent': 75,      # Alert if RAM > 75% (12GB)
    'gpu_util_percent': 90, # Alert if GPU > 90% (when enabled)
    'gpu_mem_percent': 85,  # Alert if VRAM > 85% (when enabled)
}

# ============================================================================
# TRAINING PRESETS
# ============================================================================

# Quick Test (for debugging and quick validation)
PRESET_QUICK = {
    'n_envs': 4,
    'total_timesteps': 10000,
    'buffer_size': 10000,
    'batch_size': 64,
    'n_steps': 128,         # For PPO
    'learning_rate': 3e-4,
    'device': GPU_CONFIG['device'],
}

# Balanced (recommended for most experiments)
PRESET_BALANCED = {
    'n_envs': 6,
    'total_timesteps': 100000,
    'buffer_size': 50000,
    'batch_size': 128,
    'n_steps': 256,
    'learning_rate': 3e-4,
    'device': GPU_CONFIG['device'],
}

# Full Training (for publication-quality results)
PRESET_FULL = {
    'n_envs': 8,
    'total_timesteps': 500000,
    'buffer_size': 100000,
    'batch_size': 256,
    'n_steps': 512,
    'learning_rate': 3e-4,
    'device': GPU_CONFIG['device'],
}

# Memory-Constrained (if running other applications)
PRESET_LIGHT = {
    'n_envs': 4,
    'total_timesteps': 50000,
    'buffer_size': 25000,
    'batch_size': 64,
    'n_steps': 128,
    'learning_rate': 3e-4,
    'device': GPU_CONFIG['device'],
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_optimal_config(complexity='medium'):
    """
    Get optimal configuration based on game complexity.
    
    Args:
        complexity: 'light', 'medium', 'heavy', or 'custom'
    
    Returns:
        dict: Configuration parameters
    """
    presets = {
        'quick': PRESET_QUICK,
        'light': PRESET_LIGHT,
        'medium': PRESET_BALANCED,
        'balanced': PRESET_BALANCED,
        'heavy': PRESET_LIGHT,  # Use light preset for heavy games
        'full': PRESET_FULL,
    }
    
    return presets.get(complexity, PRESET_BALANCED).copy()


def estimate_training_time(timesteps, n_envs=6, steps_per_sec=1000):
    """
    Estimate training time.
    
    Args:
        timesteps: Total training timesteps
        n_envs: Number of parallel environments
        steps_per_sec: Estimated steps per second (CPU: ~1000, GPU: ~3000+)
    
    Returns:
        dict: Time estimates
    """
    effective_steps_per_sec = steps_per_sec * n_envs
    total_seconds = timesteps / effective_steps_per_sec
    
    hours = total_seconds / 3600
    minutes = (total_seconds % 3600) / 60
    
    return {
        'total_seconds': total_seconds,
        'hours': hours,
        'minutes': minutes,
        'formatted': f"{int(hours)}h {int(minutes)}m",
    }


def check_compatibility():
    """Check system compatibility and provide recommendations."""
    import psutil
    
    print("="*70)
    print("RAPCG-MetaRL Hardware Compatibility Check")
    print("="*70)
    
    # CPU Check
    cpu_count = psutil.cpu_count(logical=True)
    print(f"\n✓ CPU: {cpu_count} threads detected")
    if cpu_count >= 8:
        print(f"  Excellent! Can run {PARALLEL_ENV_CONFIG['recommended']} parallel environments")
    else:
        print(f"  Recommend reducing to {min(4, cpu_count//2)} parallel environments")
    
    # RAM Check
    ram_gb = psutil.virtual_memory().total / (1024**3)
    print(f"\n✓ RAM: {ram_gb:.1f} GB detected")
    if ram_gb >= 16:
        print(f"  Excellent! Can handle large replay buffers and multiple environments")
    elif ram_gb >= 8:
        print(f"  Good! Use medium-sized configurations")
    else:
        print(f"  Limited RAM. Use light presets and reduce n_envs to 2-4")
    
    # GPU Check
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"\n{'✓' if cuda_available else '⚠'} GPU: {'Available' if cuda_available else 'Not available (CPU-only PyTorch)'}")
        
        if cuda_available:
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Training will be 3-5x faster with GPU!")
        else:
            print(f"  To enable GPU acceleration:")
            print(f"    pip uninstall torch")
            print(f"    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    except ImportError:
        print(f"\n⚠ PyTorch not installed")
    
    # Storage Check
    disk = psutil.disk_usage('.')
    disk_gb = disk.free / (1024**3)
    print(f"\n✓ Storage: {disk_gb:.1f} GB free")
    if disk_gb >= 50:
        print(f"  Excellent! Plenty of space for checkpoints and logs")
    elif disk_gb >= 10:
        print(f"  Good! Monitor disk usage during long training runs")
    else:
        print(f"  Warning! Less than 10GB free. Clean up disk space")
    
    print("\n" + "="*70)
    print("Recommended Training Preset: BALANCED")
    print("  - Parallel Environments: 6")
    print("  - Timesteps: 100,000")
    print("  - Estimated Time: ~30-45 minutes (CPU)")
    print("="*70)
    
    return True


if __name__ == '__main__':
    check_compatibility()
    
    print("\n" + "="*70)
    print("Configuration Examples")
    print("="*70)
    
    print("\n1. Quick Test (debugging):")
    config = get_optimal_config('quick')
    time_est = estimate_training_time(config['total_timesteps'], config['n_envs'])
    print(f"   Timesteps: {config['total_timesteps']:,}")
    print(f"   Environments: {config['n_envs']}")
    print(f"   Estimated Time: {time_est['formatted']}")
    
    print("\n2. Balanced Training (recommended):")
    config = get_optimal_config('balanced')
    time_est = estimate_training_time(config['total_timesteps'], config['n_envs'])
    print(f"   Timesteps: {config['total_timesteps']:,}")
    print(f"   Environments: {config['n_envs']}")
    print(f"   Estimated Time: {time_est['formatted']}")
    
    print("\n3. Full Training (publication-quality):")
    config = get_optimal_config('full')
    time_est = estimate_training_time(config['total_timesteps'], config['n_envs'])
    print(f"   Timesteps: {config['total_timesteps']:,}")
    print(f"   Environments: {config['n_envs']}")
    print(f"   Estimated Time: {time_est['formatted']}")
    
    print("\n" + "="*70)
