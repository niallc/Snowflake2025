#!/usr/bin/env python3
"""
Check available devices for PyTorch acceleration.

This script detects and reports available compute devices (CUDA, MPS, CPU)
and provides recommendations for optimal performance.
"""

import torch
import sys

def check_devices():
    """Check and report available PyTorch devices."""
    print("=" * 60)
    print("PYTORCH DEVICE DETECTION")
    print("=" * 60)
    print()
    
    # Basic PyTorch info
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # CUDA detection
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"  - CUDA version: {torch.version.cuda}")
        print(f"  - GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    print()
    
    # MPS detection (Apple Silicon)
    mps_available = False
    if hasattr(torch.backends, 'mps'):
        mps_available = torch.backends.mps.is_available()
        print(f"MPS available: {mps_available}")
        if mps_available:
            print("  - Apple Silicon GPU acceleration enabled")
    else:
        print("MPS available: Not supported (requires PyTorch 1.12+)")
    print()
    
    # CPU info
    print(f"CPU threads: {torch.get_num_threads()}")
    print()
    
    # Recommendations
    print("RECOMMENDATIONS:")
    if cuda_available:
        print("✅ CUDA GPU detected - optimal for training and inference")
        print("   Current installation should work well.")
        print("   For optimization: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    elif mps_available:
        print("✅ MPS (Apple Silicon) detected - good for training and inference")
        print("   Current installation should work well.")
        print("   For optimization: pip install torch torchvision torchaudio")
    else:
        print("⚠️  CPU-only mode - slower but functional")
        print("   Current installation should work.")
        print("   For better performance, consider:")
        print("   - CUDA GPU: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        print("   - Apple Silicon: pip install torch torchvision torchaudio")
    print()
    
    # Test device selection
    try:
        from hex_ai.training_utils import get_device
        selected_device = get_device()
        print(f"Hex AI will use device: {selected_device}")
    except ImportError:
        print("Note: Hex AI device detection not available (run from project root with PYTHONPATH=.)")
    
    print()
    print("=" * 60)

if __name__ == "__main__":
    check_devices()
