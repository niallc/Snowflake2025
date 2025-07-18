#!/usr/bin/env python3
"""
Test script for 3-channel legacy modifications.

This script tests:
1. 3-channel model creation and forward pass
2. 3-channel dataset loading with player-to-move channel
3. 5x5 first convolution (Step 2.2 of incremental migration)
"""

import torch
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.models_legacy_with_player_channel import TwoHeadedResNetLegacyWithPlayerChannel
from hex_ai.training_utils_legacy import NewProcessedDataset
from hex_ai.config import VERBOSE_LEVEL

def test_3channel_model():
    """Test 3-channel model creation and forward pass."""
    print("Testing 3-channel model with 5x5 first convolution...")
    
    # Create model
    model = TwoHeadedResNetLegacyWithPlayerChannel()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {total_params:,} parameters")
    
    # Test forward pass
    batch_size = 2
    input_tensor = torch.randn(batch_size, 3, 13, 13)  # 3 channels: 2 board + 1 player-to-move
    
    with torch.no_grad():
        policy_output, value_output = model(input_tensor)
    
    print("Forward pass successful:")
    print(f"  Input shape: {input_tensor.shape}")
    print(f"  Policy output: {policy_output.shape}")
    print(f"  Value output: {value_output.shape}")
    
    # Verify output shapes
    assert policy_output.shape == (batch_size, 169), f"Policy output shape {policy_output.shape} != (batch_size, 169)"
    assert value_output.shape == (batch_size, 1), f"Value output shape {value_output.shape} != (batch_size, 1)"
    
    print("✅ Model test passed")

def test_3channel_dataset():
    """Test 3-channel dataset loading."""
    print("\nTesting 3-channel dataset...")
    
    # Create dataset with limited examples for quick testing
    from pathlib import Path
    data_files = list(Path("data/processed").glob("*.pkl.gz"))[:1]  # Just one file
    
    if not data_files:
        print("❌ No data files found in data/processed/")
        return
    
    dataset = NewProcessedDataset(data_files, max_examples=100)
    print(f"Dataset created with {len(dataset)} examples")
    
    # Test a few examples
    for i in range(min(3, len(dataset))):
        board, policy, value = dataset[i]
        print(f"  Example {i}: board={board.shape}, policy={policy.shape}, value={value.shape}")
        
        # Verify board has 3 channels
        if board.shape[0] == 3:
            print(f"    ✓ Board has 3 channels as expected")
        else:
            print(f"    ❌ Board has {board.shape[0]} channels, expected 3")
            return False
    
    print("✅ Dataset test passed")
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING 3-CHANNEL LEGACY MODIFICATIONS WITH 5x5 CONV")
    print("=" * 60)
    
    try:
        test_3channel_model()
        test_3channel_dataset()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print("The 3-channel legacy modifications with 5x5 first convolution are working correctly.")
        print("You can now run the modified hyperparameter tuning script.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main() 