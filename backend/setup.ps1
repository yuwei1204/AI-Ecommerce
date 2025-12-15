# Check for NVIDIA GPU
$has_gpu = $false
try {
    nvidia-smi | Out-Null
    if ($?) {
        $has_gpu = $true
        Write-Host "Detected NVIDIA GPU." -ForegroundColor Green
    }
} catch {
    Write-Host "No NVIDIA GPU detected (or drivers not installed)." -ForegroundColor Yellow
}

# Define PyTorch installation command based on GPU presence
if ($has_gpu) {
    Write-Host "Installing PyTorch with CUDA support..." -ForegroundColor Cyan
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
} else {
    Write-Host "Installing PyTorch (CPU only version) to save space..." -ForegroundColor Cyan
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
}

# Install other requirements
Write-Host "Installing other dependencies from requirements.txt..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "Setup complete!" -ForegroundColor Green

