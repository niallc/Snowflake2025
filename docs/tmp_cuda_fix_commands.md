# CUDA Fix Commands

Quick fix for PyTorch CUDA installation issue on Windows.

## Problem
- `torch.__version__` shows `2.7.1+cpu` (CPU-only version installed)
- `torch.cuda.is_available()` returns `False`
- Need to reinstall PyTorch with CUDA support

## Solution

Run these commands in PowerShell:

```powershell
# Navigate to project directory
cd C:\path\to\Snowflake2025

# Activate virtual environment
.\hex_ai_env\Scripts\Activate.ps1

# Set PYTHONPATH
$env:PYTHONPATH = "."

# Uninstall CPU-only version
pip uninstall torch torchvision torchaudio

# Install CUDA version (cu121 is more stable than cu130)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify installation
python -c "import torch; print(f'Version: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## Expected Results
After successful installation:
- `Version: 2.7.1+cu121` (note `+cu121` instead of `+cpu`)
- `CUDA: True`

## If cu121 doesn't work, try cu130:
```powershell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## Delete this file after use
This is a temporary file for fixing the CUDA installation issue.
