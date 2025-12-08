# RAPCG-MetaRL Setup Script for Windows PowerShell
# Run this script to set up the project environment

Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "RAPCG-MetaRL Setup" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan

# Check Python version
Write-Host "`nChecking Python version..." -ForegroundColor Yellow
python --version

# Create virtual environment (optional)
$createVenv = Read-Host "`nCreate virtual environment? (y/n)"
if ($createVenv -eq 'y') {
    Write-Host "Creating virtual environment..." -ForegroundColor Yellow
    python -m venv pcg_env
    
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    .\pcg_env\Scripts\Activate.ps1
}

# Install requirements
Write-Host "`nInstalling Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# Install gym-pcgrl
Write-Host "`nInstalling gym-pcgrl..." -ForegroundColor Yellow
Set-Location gym-pcgrl
pip install -e .
Set-Location ..

# Create necessary directories
Write-Host "`nCreating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path logs | Out-Null
New-Item -ItemType Directory -Force -Path checkpoints | Out-Null
New-Item -ItemType Directory -Force -Path generated_levels | Out-Null

# Run tests
Write-Host "`nRunning tests..." -ForegroundColor Yellow
python test\test.py

Write-Host "`n" -NoNewline
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "Setup Complete!" -ForegroundColor Green
Write-Host "=" -NoNewline -ForegroundColor Cyan
Write-Host ("=" * 59) -ForegroundColor Cyan

Write-Host "`nNext steps:" -ForegroundColor Yellow
Write-Host "  1. Train a model:" -ForegroundColor White
Write-Host "     python train.py --game zelda --timesteps 50000" -ForegroundColor Gray
Write-Host "  2. Generate levels:" -ForegroundColor White
Write-Host "     python inference.py checkpoints/*/final_model.zip" -ForegroundColor Gray
Write-Host "  3. Run quick start:" -ForegroundColor White
Write-Host "     python quickstart.py" -ForegroundColor Gray

Write-Host "`nFor help with any script, use --help flag:" -ForegroundColor Yellow
Write-Host "  python train.py --help" -ForegroundColor Gray
