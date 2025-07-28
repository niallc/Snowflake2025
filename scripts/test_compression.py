#!/usr/bin/env python3
"""
Test script for checkpoint compression functionality.

This script tests that the new compression features work correctly:
1. Saving compressed checkpoints
2. Loading compressed checkpoints
3. Backward compatibility with uncompressed checkpoints
"""

import sys
import tempfile
import os
from pathlib import Path

# Add the project root to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from hex_ai.models import TwoHeadedResNet
from hex_ai.training import Trainer
from hex_ai.training_utils import save_checkpoint, load_checkpoint


def test_compression():
    """Test checkpoint compression functionality."""
    print("Testing checkpoint compression...")
    
    # Create a simple model
    model = TwoHeadedResNet()
    optimizer = torch.optim.Adam(model.parameters())
    
    # Test data
    test_metrics = {
        'total_loss': 0.5,
        'policy_loss': 0.3,
        'value_loss': 0.2
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Test 1: Save uncompressed checkpoint
        print("\n1. Testing uncompressed checkpoint...")
        uncompressed_path = tmp_path / "test_uncompressed.pt"
        save_checkpoint(model, optimizer, epoch=1, loss=0.5, 
                       filepath=str(uncompressed_path), compress=False)
        
        # Verify file exists and is not compressed
        assert uncompressed_path.exists(), "Uncompressed file not created"
        with open(uncompressed_path, 'rb') as f:
            magic_bytes = f.read(2)
            assert magic_bytes != b'\x1f\x8b', "Uncompressed file appears to be gzipped"
        
        # Test 2: Save compressed checkpoint
        print("2. Testing compressed checkpoint...")
        compressed_path = tmp_path / "test_compressed.pt"
        save_checkpoint(model, optimizer, epoch=2, loss=0.4, 
                       filepath=str(compressed_path), compress=True)
        
        # Verify file exists and is compressed
        assert compressed_path.with_suffix('.pt.gz').exists(), "Compressed file not created"
        with open(compressed_path.with_suffix('.pt.gz'), 'rb') as f:
            magic_bytes = f.read(2)
            assert magic_bytes == b'\x1f\x8b', "Compressed file is not gzipped"
        
        # Test 3: Load uncompressed checkpoint
        print("3. Testing loading uncompressed checkpoint...")
        new_model1 = TwoHeadedResNet()
        new_optimizer1 = torch.optim.Adam(new_model1.parameters())
        epoch, loss = load_checkpoint(new_model1, new_optimizer1, str(uncompressed_path))
        assert epoch == 1, f"Expected epoch 1, got {epoch}"
        assert abs(loss - 0.5) < 1e-6, f"Expected loss 0.5, got {loss}"
        
        # Test 4: Load compressed checkpoint
        print("4. Testing loading compressed checkpoint...")
        new_model2 = TwoHeadedResNet()
        new_optimizer2 = torch.optim.Adam(new_model2.parameters())
        epoch, loss = load_checkpoint(new_model2, new_optimizer2, str(compressed_path.with_suffix('.pt.gz')))
        assert epoch == 2, f"Expected epoch 2, got {epoch}"
        assert abs(loss - 0.4) < 1e-6, f"Expected loss 0.4, got {loss}"
        
        # Test 5: Test Trainer save_checkpoint
        print("5. Testing Trainer.save_checkpoint...")
        trainer = Trainer(model, train_loader=None, enable_system_analysis=False, enable_csv_logging=False)
        trainer_path = tmp_path / "trainer_test.pt"
        trainer.save_checkpoint(trainer_path, test_metrics, test_metrics, compress=True)
        
        # Verify compressed file was created
        assert trainer_path.with_suffix('.pt.gz').exists(), "Trainer compressed file not created"
        
        # Test 6: Test Trainer load_checkpoint
        print("6. Testing Trainer.load_checkpoint...")
        new_trainer = Trainer(model, train_loader=None, enable_system_analysis=False, enable_csv_logging=False)
        new_trainer.load_checkpoint(trainer_path.with_suffix('.pt.gz'))
        assert new_trainer.current_epoch == 0, f"Expected epoch 0, got {new_trainer.current_epoch}"
        
        # Test 7: Check file sizes
        print("7. Checking compression ratios...")
        uncompressed_size = uncompressed_path.stat().st_size
        compressed_size = compressed_path.with_suffix('.pt.gz').stat().st_size
        compression_ratio = (1 - compressed_size / uncompressed_size) * 100
        
        print(f"   Uncompressed size: {uncompressed_size:,} bytes")
        print(f"   Compressed size: {compressed_size:,} bytes")
        print(f"   Compression ratio: {compression_ratio:.1f}%")
        
        assert compression_ratio > 0, "Compression should reduce file size"
        
        print("\n✅ All compression tests passed!")
        return True


if __name__ == "__main__":
    try:
        test_compression()
        print("\n🎉 Compression implementation is working correctly!")
    except Exception as e:
        print(f"\n❌ Compression test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 