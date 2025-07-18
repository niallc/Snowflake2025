#!/usr/bin/env python3
"""
Test script for data augmentation pipeline.
Loads a few examples, applies augmentation, and displays the results.
"""
import sys
import os
import torch
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hex_ai.data_pipeline import StreamingAugmentedProcessedDataset
from hex_ai.training_utils_legacy import augmented_collate_fn
from hex_ai.inference.board_display import display_hex_board
from hex_ai.utils.format_conversion import tensor_to_rowcol, tensor_to_trmph

def test_augmentation():
    """Test the augmentation pipeline with a small dataset."""
    
    # Use a small data file for testing
    data_file = Path("data/processed/twoNetGames_13x13_mk45_d1b20_v1816_2s0_p2551k_vt25_pt10_processed.pkl.gz")
    
    if not data_file.exists():
        print(f"Data file not found: {data_file}")
        return
    
    print("Testing StreamingAugmentedProcessedDataset...")
    
    # Create augmented dataset
    dataset = StreamingAugmentedProcessedDataset([data_file], enable_augmentation=True, chunk_size=1000)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Expected effective size: {len(dataset) * 4} (4x augmentation)")
    
    # Test single item retrieval
    print("\n=== Testing single item retrieval ===")
    item = dataset[1]  # Skip first item (empty board)
    
    if isinstance(item, list):
        print(f"Retrieved {len(item)} augmented examples")
        for i, (board, policy, value) in enumerate(item):
            print(f"\n--- Augmentation {i+1} ---")
            print(f"Board shape: {board.shape}")
            print(f"Policy shape: {policy.shape}")
            print(f"Value: {value.item()}")
            
            # Display board
            board_2ch = board[:2].numpy()  # Extract 2-channel board for display
            display_hex_board(board_2ch)
            
            # Show policy move
            nonzero_indices = torch.where(policy > 1e-6)[0]
            if len(nonzero_indices) > 0:
                label_idx = nonzero_indices[0].item()
                row, col = tensor_to_rowcol(label_idx)
                trmph_move = tensor_to_trmph(label_idx)
                print(f"Policy move: index {label_idx} -> ({row},{col}) -> {trmph_move}")
    else:
        print("Single example returned (no augmentation)")
    
    # Test DataLoader with custom collate function
    print("\n=== Testing DataLoader with augmentation ===")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=2,  # This will give us 8 examples (2 * 4 augmentations)
        shuffle=False,
        collate_fn=augmented_collate_fn,
        num_workers=0
    )
    
    for batch_idx, (boards, policies, values) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}:")
        print(f"  Boards shape: {boards.shape}")
        print(f"  Policies shape: {policies.shape}")
        print(f"  Values shape: {values.shape}")
        print(f"  Number of examples in batch: {boards.size(0)}")
        
        # Show first example from batch
        print(f"\nFirst example in batch:")
        board_2ch = boards[0, :2].numpy()
        display_hex_board(board_2ch)
        
        policy = policies[0]
        value = values[0]
        print(f"Value: {value.item()}")
        
        nonzero_indices = torch.where(policy > 1e-6)[0]
        if len(nonzero_indices) > 0:
            label_idx = nonzero_indices[0].item()
            row, col = tensor_to_rowcol(label_idx)
            trmph_move = tensor_to_trmph(label_idx)
            print(f"Policy move: index {label_idx} -> ({row},{col}) -> {trmph_move}")
        
        # Only show first batch
        break
    
    print("\n=== Augmentation test completed ===")

if __name__ == "__main__":
    test_augmentation() 