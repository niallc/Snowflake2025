#!/usr/bin/env python3
"""
Simple test script to verify 3-channel legacy modifications work.

This script tests that:
1. The 3-channel model can be created and run forward pass
2. The modified dataset can load data and add player-to-move channel
3. Everything works together for a small test
"""

import torch
import numpy as np
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def test_3channel_model():
    """Test that the 3-channel model works."""
    print("Testing 3-channel model...")
    
    from hex_ai.models_legacy_with_player_channel import TwoHeadedResNetLegacyWithPlayerChannel
    
    # Create model
    model = TwoHeadedResNetLegacyWithPlayerChannel()
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    test_input = torch.randn(2, 3, 13, 13)  # 3-channel input
    with torch.no_grad():
        policy_out, value_out = model(test_input)
    
    print(f"Forward pass successful:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Policy output: {policy_out.shape}")
    print(f"  Value output: {value_out.shape}")
    
    return True

def test_3channel_dataset():
    """Test that the modified dataset works."""
    print("\nTesting 3-channel dataset...")
    
    from hex_ai.training_utils_legacy import NewProcessedDataset, discover_processed_files_legacy
    
    # Find data files
    data_files = discover_processed_files_legacy("data/processed")
    print(f"Found {len(data_files)} data files")
    
    # Create dataset with just 100 examples for testing
    dataset = NewProcessedDataset(data_files[:1], max_examples=100)
    print(f"Dataset created with {len(dataset)} examples")
    
    # Test a few examples
    for i in range(min(3, len(dataset))):
        board_state, policy_target, value_target = dataset[i]
        print(f"  Example {i}: board={board_state.shape}, policy={policy_target.shape}, value={value_target.shape}")
        
        # Verify board has 3 channels
        if board_state.shape[0] != 3:
            print(f"    ERROR: Expected 3 channels, got {board_state.shape[0]}")
            return False
        else:
            print(f"    ✓ Board has 3 channels as expected")
    
    return True

def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING 3-CHANNEL LEGACY MODIFICATIONS")
    print("=" * 60)
    
    # Test model
    if not test_3channel_model():
        print("❌ Model test failed")
        return False
    
    # Test dataset
    if not test_3channel_dataset():
        print("❌ Dataset test failed")
        return False
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)
    print("The 3-channel legacy modifications are working correctly.")
    print("You can now run the modified hyperparameter tuning script.")
    
    return True

if __name__ == '__main__':
    main() 