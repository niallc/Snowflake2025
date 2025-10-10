#!/usr/bin/env python3
"""
Simple PyTorch installation helper.

This script helps you choose the right PyTorch installation for your system.
"""

import platform
import subprocess

def main():
    print("=" * 50)
    print("PYTORCH INSTALLATION HELPER")
    print("=" * 50)
    print()
    
    # Detect system
    system = platform.system()
    machine = platform.machine()
    
    print(f"System: {system} ({machine})")
    
    # Check for NVIDIA GPU
    has_nvidia = False
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        has_nvidia = result.returncode == 0
    except FileNotFoundError:
        pass
    
    # Detect Apple Silicon
    is_apple_silicon = system == "Darwin" and machine == "arm64"
    
    print(f"Apple Silicon: {is_apple_silicon}")
    print(f"NVIDIA GPU: {has_nvidia}")
    print()
    
    # Give simple recommendation
    print("RECOMMENDED INSTALLATION:")
    if has_nvidia:
        print("✅ NVIDIA GPU detected")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    elif is_apple_silicon:
        print("✅ Apple Silicon detected")
        print("   pip install torch torchvision torchaudio")
    else:
        print("⚠️  No GPU detected")
        print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu")
    
    print()
    print("=" * 50)

if __name__ == "__main__":
    main()
