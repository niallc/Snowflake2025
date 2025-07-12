#!/usr/bin/env python3
"""
Quick test script for Windows setup.
Tests basic functionality with minimal data requirements.
"""

import sys
import torch
from pathlib import Path
import numpy as np

def test_basic_imports():
    """Test basic imports."""
    print("=== Testing Basic Imports ===")
    
    try:
        from hex_ai.models import TwoHeadedResNet
        print("✅ Models imported successfully")
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False
    
    try:
        from hex_ai.training import Trainer
        print("✅ Training imported successfully")
    except Exception as e:
        print(f"❌ Training import failed: {e}")
        return False
    
    try:
        from hex_ai.data_processing import create_processed_dataloader
        print("✅ Data processing imported successfully")
    except Exception as e:
        print(f"❌ Data processing import failed: {e}")
        return False
    
    return True

def test_gpu_detection():
    """Test GPU detection and PyTorch installation."""
    print("\n=== Testing GPU Detection ===")
    
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
        print("⚠️  No CUDA detected - this might be a PyTorch installation issue")
        print("   Try: conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia")
        return False

def test_model_creation():
    """Test model creation and forward pass."""
    print("\n=== Testing Model Creation ===")
    
    try:
        from hex_ai.models import TwoHeadedResNet
        
        # Create model
        model = TwoHeadedResNet()
        print(f"✅ Model created successfully")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test forward pass
        batch_size = 2
        x = torch.randn(batch_size, 2, 13, 13)  # 2 channels for 2 players
        
        with torch.no_grad():
            policy, value = model(x)
        
        print(f"✅ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Policy output shape: {policy.shape}")
        print(f"  Value output shape: {value.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_data_loading():
    """Test data loading with minimal requirements."""
    print("\n=== Testing Data Loading ===")
    
    # Check if any processed data exists
    processed_dir = Path("data/processed")
    if not processed_dir.exists():
        print("⚠️  No processed data found - creating dummy data for test")
        
        # Create dummy shard for testing
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple dummy shard
        dummy_data = {
            'boards': np.random.randn(100, 2, 13, 13).astype(np.float32),
            'policies': np.random.randn(100, 169).astype(np.float32),
            'values': np.random.randn(100, 1).astype(np.float32),
            'num_games': 100
        }
        
        import pickle
        import gzip
        
        dummy_file = processed_dir / "dummy_test_shard.pkl.gz"
        with gzip.open(dummy_file, 'wb') as f:
            pickle.dump(dummy_data, f)
        
        print(f"✅ Created dummy test shard: {dummy_file}")
    
    # Test data loading
    try:
        from hex_ai.data_processing import create_processed_dataloader
        
        # Find any shard files
        shard_files = list(processed_dir.glob("*.pkl.gz")) + list(processed_dir.glob("*.pkl"))
        
        if not shard_files:
            print("❌ No shard files found")
            return False
        
        print(f"Found {len(shard_files)} shard files")
        print(f"Using first shard: {shard_files[0].name}")
        
        # Create dataloader with just one shard
        dataloader = create_processed_dataloader(
            [shard_files[0]], 
            batch_size=4, 
            shuffle=False,
            num_workers=0
        )
        
        # Test loading one batch
        batch = next(iter(dataloader))
        boards, policies, values = batch
        
        print(f"✅ Data loading successful")
        print(f"  Batch boards shape: {boards.shape}")
        print(f"  Batch policies shape: {policies.shape}")
        print(f"  Batch values shape: {values.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False

def test_mini_training():
    """Test a minimal training step."""
    print("\n=== Testing Mini Training ===")
    
    try:
        from hex_ai.models import TwoHeadedResNet
        from hex_ai.training import Trainer
        from hex_ai.data_processing import create_processed_dataloader
        
        # Create model
        model = TwoHeadedResNet()
        
        # Find a shard file
        processed_dir = Path("data/processed")
        shard_files = list(processed_dir.glob("*.pkl.gz")) + list(processed_dir.glob("*.pkl"))
        
        if not shard_files:
            print("❌ No shard files for training test")
            return False
        
        # Create minimal dataloader
        train_loader = create_processed_dataloader(
            [shard_files[0]], 
            batch_size=2, 
            shuffle=True,
            num_workers=0
        )
        
        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,  # No validation for quick test
            learning_rate=0.001,
            device=None,  # Let it auto-detect
            enable_system_analysis=False  # Disable for quick test
        )
        
        # Test one training step
        print("Testing one training step...")
        trainer.train_epoch(epoch=0)
        
        print(f"✅ Mini training test successful")
        return True
        
    except Exception as e:
        print(f"❌ Mini training test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all quick tests."""
    print("Quick Test for Windows Setup")
    print("=" * 40)
    
    tests = [
        test_basic_imports,
        test_gpu_detection,
        test_model_creation,
        test_data_loading,
        test_mini_training
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 40)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All tests passed! Ready for hyperparameter tuning.")
    else:
        print("❌ Some tests failed. Please fix issues before proceeding.")

if __name__ == "__main__":
    main()
