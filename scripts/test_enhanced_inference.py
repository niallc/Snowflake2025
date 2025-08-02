#!/usr/bin/env python3
"""
Test script for the enhanced inference system with caching and performance monitoring.
"""

import sys
import os
import time
import numpy as np
import torch

# Add the parent directory to the path so we can import hex_ai modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.selfplay.selfplay_engine import SelfPlayEngine
from hex_ai.config import BLUE_PLAYER, RED_PLAYER, EMPTY_PIECE


def create_test_boards(num_boards=20):
    """Create test boards for inference testing."""
    boards = []
    
    for i in range(num_boards):
        # Create a simple board with some moves
        board = np.full((13, 13), EMPTY_PIECE, dtype='U1')
        
        # Add some moves to make it interesting
        num_moves = min(i + 1, 13 * 13 // 4)
        for j in range(num_moves):
            row = j % 13
            col = (j + i) % 13
            player = BLUE_PLAYER if j % 2 == 0 else RED_PLAYER
            board[row, col] = player
        
        boards.append(board)
    
    return boards


def test_enhanced_inference():
    """Test the enhanced inference system with caching and performance monitoring."""
    print("Testing Enhanced Inference System")
    print("=" * 50)
    
    # Try to find a model checkpoint
    model_paths = [
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch2_mini16.pt.gz",
        "checkpoints/final_only/loss_weight_sweep_exp0_do0_pw0.2_794e88_20250723_230725/epoch1_mini1.pt.gz",
        "checkpoints/latest.pt.gz"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("No model checkpoint found. Please ensure a model checkpoint exists.")
        return False
    
    print(f"Using model: {model_path}")
    
    # Test 1: Basic inference with caching
    print("\n1. Testing basic inference with caching...")
    model = SimpleModelInference(
        checkpoint_path=model_path,
        cache_size=1000,
        enable_caching=True
    )
    
    test_boards = create_test_boards(10)
    
    # First pass - should miss cache
    print("First pass (cache misses expected)...")
    start_time = time.time()
    for i, board in enumerate(test_boards):
        policy, value = model.simple_infer(board)
        if i % 5 == 0:
            print(f"  Board {i}: policy_shape={policy.shape}, value={value:.4f}")
    
    first_pass_time = time.time() - start_time
    print(f"First pass completed in {first_pass_time:.3f}s")
    
    # Second pass - should hit cache
    print("\nSecond pass (cache hits expected)...")
    start_time = time.time()
    for i, board in enumerate(test_boards):
        policy, value = model.simple_infer(board)
        if i % 5 == 0:
            print(f"  Board {i}: policy_shape={policy.shape}, value={value:.4f}")
    
    second_pass_time = time.time() - start_time
    print(f"Second pass completed in {second_pass_time:.3f}s")
    
    # Print performance stats
    model.print_performance_summary()
    
    # Test 2: Batch inference
    print("\n2. Testing batch inference...")
    batch_model = SimpleModelInference(
        checkpoint_path=model_path,
        cache_size=1000,
        max_batch_size=50,
        enable_caching=True
    )
    
    # Test individual vs batch inference
    print("Individual inference...")
    start_time = time.time()
    individual_results = []
    for board in test_boards:
        policy, value = batch_model.simple_infer(board)
        individual_results.append((policy, value))
    individual_time = time.time() - start_time
    
    print("Batch inference...")
    start_time = time.time()
    batch_policies, batch_values = batch_model.batch_infer(test_boards)
    batch_time = time.time() - start_time
    
    print(f"Individual time: {individual_time:.3f}s")
    print(f"Batch time: {batch_time:.3f}s")
    print(f"Speedup: {individual_time / batch_time:.2f}x")
    
    # Verify results are consistent
    print("Verifying batch vs individual consistency...")
    for i, ((ind_policy, ind_value), (batch_policy, batch_value)) in enumerate(zip(individual_results, zip(batch_policies, batch_values))):
        policy_diff = np.abs(ind_policy - batch_policy).max()
        value_diff = abs(ind_value - batch_value)
        
        if policy_diff > 1e-6 or value_diff > 1e-6:
            print(f"  ERROR: Results don't match for board {i}")
            return False
    
    print("✓ Batch inference consistency verified!")
    
    # Test 3: Self-play engine
    print("\n3. Testing self-play engine...")
    try:
        engine = SelfPlayEngine(
            model_path=model_path,
            num_workers=2,  # Use fewer workers for testing
            batch_size=50,
            cache_size=1000,
            search_widths=[2, 1],  # Use smaller search for speed
            temperature=1.0
        )
        
        # Generate a small number of games
        print("Generating 5 games...")
        games = engine.generate_games_with_monitoring(5, progress_interval=1)
        
        print(f"Generated {len(games)} games")
        for i, game in enumerate(games):
            print(f"  Game {i}: {game['num_moves']} moves, winner: {game['winner']}")
        
        # Save games
        output_file = "test_selfplay_games.pkl.gz"
        engine.save_games_to_file(games, output_file)
        print(f"Games saved to {output_file}")
        
        # Print final stats
        engine.shutdown()
        
    except Exception as e:
        print(f"Self-play test failed: {e}")
        return False
    
    print("\n✓ All tests completed successfully!")
    return True


def test_memory_management():
    """Test memory management with large batches."""
    print("\nTesting Memory Management")
    print("=" * 50)
    
    # Find model
    model_paths = [
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch2_mini16.pt.gz",
        "checkpoints/final_only/loss_weight_sweep_exp0_do0_pw0.2_794e88_20250723_230725/epoch1_mini1.pt.gz",
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("No model checkpoint found for memory test.")
        return False
    
    print(f"Using model: {model_path}")
    
    # Test with different batch sizes
    model = SimpleModelInference(
        checkpoint_path=model_path,
        cache_size=1000,
        max_batch_size=200,
        enable_caching=True
    )
    
    test_boards = create_test_boards(100)
    
    print("Testing memory management with large batches...")
    
    # Test different batch sizes
    for batch_size in [10, 25, 50, 100]:
        print(f"\nBatch size: {batch_size}")
        
        # Clear cache and memory
        model.clear_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Get initial memory usage
        initial_memory = model.get_memory_usage()
        print(f"Initial memory: {initial_memory}")
        
        # Process in batches
        start_time = time.time()
        for i in range(0, len(test_boards), batch_size):
            batch = test_boards[i:i + batch_size]
            policies, values = model.batch_infer(batch)
            
            if i == 0:  # Print first batch info
                print(f"  First batch: {len(batch)} boards, {len(policies)} results")
        
        total_time = time.time() - start_time
        
        # Get final memory usage
        final_memory = model.get_memory_usage()
        print(f"Final memory: {final_memory}")
        print(f"Time: {total_time:.3f}s, Throughput: {len(test_boards) / total_time:.1f} boards/s")
    
    model.print_performance_summary()
    return True


def main():
    """Main test function."""
    print("Enhanced Inference System Test")
    print("=" * 60)
    
    success = True
    
    try:
        success &= test_enhanced_inference()
        success &= test_memory_management()
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    if success:
        print("\n🎉 All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)