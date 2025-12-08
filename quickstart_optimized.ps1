# Quick Start Training Script for Intel i5-13500 + RTX 3060 Ti + 16GB RAM
# Optimized configuration for your hardware

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "RAPCG-MetaRL Quick Start - Hardware Optimized" -ForegroundColor Cyan
Write-Host "System: Intel i5-13500 (20 threads) + RTX 3060 Ti + 16GB RAM" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host ""

# Activate virtual environment
Write-Host "[1/5] Activating virtual environment..." -ForegroundColor Yellow
& "D:\Work\thesis\RAPCG-MetaRL\pcg_env\Scripts\Activate.ps1"

# Check system compatibility
Write-Host "`n[2/5] Checking hardware compatibility..." -ForegroundColor Yellow
python config_hardware.py

# Quick test (optional - uncomment to run)
# Write-Host "`n[3/5] Running test suite..." -ForegroundColor Yellow
# python test/test.py

Write-Host "`n[3/5] Training Configuration Options:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  [A] Quick Test (10k steps, ~30 min)" -ForegroundColor Green
Write-Host "      python train.py --game zelda --timesteps 10000 --n-envs 4"
Write-Host ""
Write-Host "  [B] Balanced Training (100k steps, ~6 hours) - RECOMMENDED" -ForegroundColor Green
Write-Host "      python train.py --game zelda --timesteps 100000 --n-envs 6"
Write-Host ""
Write-Host "  [C] Full Training (500k steps, ~30 hours)" -ForegroundColor Green
Write-Host "      python train.py --game zelda --timesteps 500000 --n-envs 6"
Write-Host ""
Write-Host "  [D] Memory-Constrained (100k steps, 8-10GB RAM)" -ForegroundColor Green
Write-Host "      python train.py --game zelda --timesteps 100000 --n-envs 4"
Write-Host ""

$choice = Read-Host "Select configuration [A/B/C/D] or press Enter to skip"

switch ($choice.ToUpper()) {
    "A" {
        Write-Host "`n[4/5] Starting Quick Test Training..." -ForegroundColor Yellow
        Write-Host "Game: Zelda | Algorithm: PPO | Timesteps: 10,000 | Envs: 4" -ForegroundColor Cyan
        python train.py --game zelda --timesteps 10000 --n-envs 4
    }
    "B" {
        Write-Host "`n[4/5] Starting Balanced Training..." -ForegroundColor Yellow
        Write-Host "Game: Zelda | Algorithm: PPO | Timesteps: 100,000 | Envs: 6" -ForegroundColor Cyan
        python train.py --game zelda --timesteps 100000 --n-envs 6
    }
    "C" {
        Write-Host "`n[4/5] Starting Full Training..." -ForegroundColor Yellow
        Write-Host "Game: Zelda | Algorithm: PPO | Timesteps: 500,000 | Envs: 6" -ForegroundColor Cyan
        Write-Host "WARNING: This will take approximately 30 hours!" -ForegroundColor Red
        $confirm = Read-Host "Continue? [Y/N]"
        if ($confirm.ToUpper() -eq "Y") {
            python train.py --game zelda --timesteps 500000 --n-envs 6
        }
    }
    "D" {
        Write-Host "`n[4/5] Starting Memory-Constrained Training..." -ForegroundColor Yellow
        Write-Host "Game: Zelda | Algorithm: PPO | Timesteps: 100,000 | Envs: 4" -ForegroundColor Cyan
        python train.py --game zelda --timesteps 100000 --n-envs 4 --buffer-size 25000
    }
    default {
        Write-Host "`n[4/5] Skipping training. You can run manually:" -ForegroundColor Yellow
        Write-Host "  python train.py --game zelda --timesteps 10000" -ForegroundColor Cyan
    }
}

Write-Host "`n[5/5] Next Steps:" -ForegroundColor Yellow
Write-Host ""
Write-Host "  1. Enable GPU Acceleration (3-5x faster training):" -ForegroundColor Green
Write-Host "     pip uninstall torch"
Write-Host "     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
Write-Host ""
Write-Host "  2. View Training Logs:" -ForegroundColor Green
Write-Host "     Check 'logs/' directory for CSV files with detailed metrics"
Write-Host ""
Write-Host "  3. Load Trained Models:" -ForegroundColor Green
Write-Host "     python inference.py checkpoints/zelda_PPO_*/final_model.zip --n-levels 5"
Write-Host ""
Write-Host "  4. Monitor Resources:" -ForegroundColor Green
Write-Host "     python config_hardware.py"
Write-Host ""

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "For detailed documentation, see HARDWARE_COMPATIBILITY.md" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
