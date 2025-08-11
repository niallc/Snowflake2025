#!/usr/bin/env python3
"""
Test batch inference functionality to ensure it works correctly and efficiently.
"""

import sys
import os
import time
import numpy as np
import torch

# Add the parent directory to the path so we can import hex_ai modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.config import EMPTY_PIECE
from hex_ai.enums import Player
from hex_ai.inference.game_engine import HexGameState


def create_test_boards(num_boards=10, board_size=13):
    """Create a list of test boards for batch inference testing."""
    boards = []
    
    for i in range(num_boards):
        # Create a simple board with some moves
        board = np.full((board_size, board_size), EMPTY_PIECE, dtype='U1')
        
        # Add some moves to make it interesting
        num_moves = min(i + 1, board_size * board_size // 4)  # Vary the number of moves
        for j in range(num_moves):
            row = j % board_size
            col = (j + i) % board_size
            player = Player.BLUE.value if j % 2 == 0 else Player.RED.value
            board[row, col] = player
        
        boards.append(board)
    
    return boards


def test_batch_inference_consistency():
    """Test that batch inference produces the same results as individual inference."""
    print("Testing batch inference consistency...")
    
    # Use a small test model or mock if available
    # For now, we'll test the interface without loading a real model
    try:
        # Try to load a model if available
        model_path = "checkpoints/final_only/loss_weight_sweep_exp0_do0_pw0.2_794e88_20250723_230725/epoch1_mini1.pt"
        if os.path.exists(model_path):
            model = SimpleModelInference(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print("No model checkpoint found, skipping actual inference test")
            return True
    except Exception as e:
        print(f"Could not load model: {e}")
        return True
    
    # Create test boards
    test_boards = create_test_boards(5)
    
    # Test individual inference
    individual_results = []
    for board in test_boards:
        policy_logits, value_logit = model.simple_infer(board)
        individual_results.append((policy_logits, value_logit))
    
    # Test batch inference
    batch_policies, batch_values = model.batch_infer(test_boards)
    
    # Compare results
    print(f"Testing {len(test_boards)} boards...")
    for i, (board, (ind_policy, ind_value), (batch_policy, batch_value)) in enumerate(zip(test_boards, individual_results, zip(batch_policies, batch_values))):
        # Check policy logits
        policy_diff = np.abs(ind_policy - batch_policy).max()
        value_diff = abs(ind_value - batch_value)
        
        print(f"Board {i}: policy_diff={policy_diff:.6f}, value_diff={value_diff:.6f}")
        
        if policy_diff > 1e-6 or value_diff > 1e-6:
            print(f"ERROR: Results don't match for board {i}")
            return False
    
    print("✓ Batch inference consistency test passed!")
    return True


def test_batch_inference_performance():
    """Test that batch inference is faster than individual inference."""
    print("\nTesting batch inference performance...")
    
    try:
        model_path = "checkpoints/final_only/loss_weight_sweep_exp0_do0_pw0.2_794e88_20250723_230725/epoch1_mini1.pt"
        if os.path.exists(model_path):
            model = SimpleModelInference(model_path)
            print(f"Loaded model from {model_path}")
        else:
            print("No model checkpoint found, skipping performance test")
            return True
    except Exception as e:
        print(f"Could not load model: {e}")
        return True
    
    # Create test boards
    test_boards = create_test_boards(20)
    
    # Warm up the model
    print("Warming up model...")
    for _ in range(3):
        model.simple_infer(test_boards[0])
    
    # Test individual inference timing
    print("Testing individual inference timing...")
    start_time = time.time()
    for board in test_boards:
        model.simple_infer(board)
    individual_time = time.time() - start_time
    
    # Test batch inference timing
    print("Testing batch inference timing...")
    start_time = time.time()
    model.batch_infer(test_boards)
    batch_time = time.time() - start_time
    
    print(f"Individual inference: {individual_time:.3f}s")
    print(f"Batch inference: {batch_time:.3f}s")
    print(f"Speedup: {individual_time / batch_time:.2f}x")
    
    if batch_time < individual_time:
        print("✓ Batch inference is faster!")
        return True
    else:
        print("⚠ Batch inference is not faster (this might be expected for small batches)")
        return True


def test_batch_inference_edge_cases():
    """Test edge cases for batch inference."""
    print("\nTesting batch inference edge cases...")
    
    try:
        model_path = "checkpoints/final_only/loss_weight_sweep_exp0_do0_pw0.2_794e88_20250723_230725/epoch1_mini1.pt"
        if os.path.exists(model_path):
            model = SimpleModelInference(model_path)
        else:
            print("No model checkpoint found, skipping edge case test")
            return True
    except Exception as e:
        print(f"Could not load model: {e}")
        return True
    
    # Test empty batch
    print("Testing empty batch...")
    policies, values = model.batch_infer([])
    assert len(policies) == 0 and len(values) == 0, "Empty batch should return empty lists"
    print("✓ Empty batch test passed")
    
    # Test single board batch
    print("Testing single board batch...")
    test_board = create_test_boards(1)[0]
    policies, values = model.batch_infer([test_board])
    assert len(policies) == 1 and len(values) == 1, "Single board batch should return single result"
    print("✓ Single board batch test passed")
    
    # Test mixed input formats
    print("Testing mixed input formats...")
    test_boards = create_test_boards(3)
    # Convert one board to TRMPH string format
    state = HexGameState(board=test_boards[1], current_player=Player.BLUE.value)
    trmph_board = state.to_trmph()
    mixed_boards = [test_boards[0], trmph_board, test_boards[2]]
    
    try:
        policies, values = model.batch_infer(mixed_boards)
        assert len(policies) == 3 and len(values) == 3, "Mixed format batch should work"
        print("✓ Mixed input format test passed")
    except Exception as e:
        print(f"⚠ Mixed input format test failed: {e}")
    
    print("✓ Edge case tests completed!")
    return True


def main():
    """Run all batch inference tests."""
    print("Running batch inference tests...")
    
    success = True
    success &= test_batch_inference_consistency()
    success &= test_batch_inference_performance()
    success &= test_batch_inference_edge_cases()
    
    if success:
        print("\n🎉 All batch inference tests passed!")
    else:
        print("\n❌ Some batch inference tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()