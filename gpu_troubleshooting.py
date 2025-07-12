#!/usr/bin/env python3
"""
GPU troubleshooting script for Windows.
Helps diagnose and fix GPU detection issues.
"""

import sys
import subprocess
import torch

def check_nvidia_drivers():
    """Check if NVIDIA drivers are installed."""
    print("=== Checking NVIDIA Drivers ===")
    
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ NVIDIA drivers are installed")
            print("GPU Information:")
            print(result.stdout)
            return True
        else:
            print("❌ NVIDIA drivers not found or not working")
            print("Error:", result.stderr)
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - NVIDIA drivers may not be installed")
        return False

def check_cuda_installation():
    """Check CUDA installation."""
    print("\n=== Checking CUDA Installation ===")
    
    try:
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ CUDA toolkit is installed")
            print("CUDA Version:", result.stdout.split('release ')[1].split(',')[0])
            return True
        else:
            print("❌ CUDA toolkit not found")
            return False
    except FileNotFoundError:
        print("❌ nvcc not found - CUDA toolkit may not be installed")
        return False

def check_pytorch_cuda():
    """Check PyTorch CUDA installation."""
    print("\n=== Checking PyTorch CUDA ===")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
        return True
    else:
        print("❌ PyTorch CUDA not available")
        return False

def suggest_fixes():
    """Suggest fixes for GPU issues."""
    print("\n=== Suggested Fixes ===")
    
    print("1. Install NVIDIA drivers:")
    print("   Download from: https://www.nvidia.com/Download/index.aspx")
    
    print("\n2. Install CUDA toolkit:")
    print("   Download from: https://developer.nvidia.com/cuda-downloads")
    print("   Choose Windows > x86_64 > 10 > exe (local)")
    
    print("\n3. Reinstall PyTorch with CUDA:")
    print("   conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia")
    print("   or")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    
    print("\n4. Verify installation:")
    print("   python -c \"import torch; print(torch.cuda.is_available())\"")
    
    print("\n5. If still having issues:")
    print("   - Check Windows Device Manager for GPU")
    print("   - Restart computer after driver installation")
    print("   - Try different CUDA version (11.8, 12.1, 12.4)")

def test_gpu_computation():
    """Test GPU computation if available."""
    print("\n=== Testing GPU Computation ===")
    
    if not torch.cuda.is_available():
        print("Skipping GPU test - CUDA not available")
        return False
    
    try:
        # Create tensor on GPU
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        
        # Perform computation
        start_time = torch.cuda.Event(enable_timing=True)
        end_time = torch.cuda.Event(enable_timing=True)
        
        start_time.record()
        z = torch.mm(x, y)
        end_time.record()
        
        torch.cuda.synchronize()
        elapsed_time = start_time.elapsed_time(end_time)
        
        print(f"✅ GPU computation test passed")
        print(f"  Matrix multiplication: 1000x1000")
        print(f"  Time: {elapsed_time:.2f} ms")
        print(f"  Result shape: {z.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU computation test failed: {e}")
        return False

def main():
    """Run GPU troubleshooting."""
    print("GPU Troubleshooting for Windows")
    print("=" * 40)
    
    # Run checks
    drivers_ok = check_nvidia_drivers()
    cuda_ok = check_cuda_installation()
    pytorch_ok = check_pytorch_cuda()
    computation_ok = test_gpu_computation()
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"  NVIDIA Drivers: {'✅' if drivers_ok else '❌'}")
    print(f"  CUDA Toolkit: {'✅' if cuda_ok else '❌'}")
    print(f"  PyTorch CUDA: {'✅' if pytorch_ok else '❌'}")
    print(f"  GPU Computation: {'✅' if computation_ok else '❌'}")
    
    if all([drivers_ok, cuda_ok, pytorch_ok, computation_ok]):
        print("\n🎉 GPU setup is working correctly!")
    else:
        print("\n⚠️  GPU setup needs attention")
        suggest_fixes()

if __name__ == "__main__":
    main()
