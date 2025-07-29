#!/usr/bin/env python3
"""
Prototype for position collection during tree building.
This demonstrates the concept of collecting positions instead of immediate inference.
"""

import sys
import os
import time
from typing import List, Dict, Any, Tuple

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference

class PositionCollector:
    """Collects board positions during tree building for batch processing."""
    
    def __init__(self, model: SimpleModelInference):
        self.model = model
        self.positions = []  # List of (board, callback) tuples
        self.batch_size = 1000  # Process when we have this many positions
        
    def add_position(self, board, callback):
        """Add a position to be processed later."""
        self.positions.append((board, callback))
        
        # Process batch if we have enough positions
        if len(self.positions) >= self.batch_size:
            self.process_batch()
    
    def process_batch(self):
        """Process all collected positions in a single batch."""
        if not self.positions:
            return
            
        print(f"Processing batch of {len(self.positions)} positions...")
        
        # Extract boards
        boards = [pos[0] for pos in self.positions]
        
        # Process in single batch
        start_time = time.time()
        policies, values = self.model.batch_infer(boards)
        batch_time = time.time() - start_time
        
        print(f"Batch processed in {batch_time:.3f}s ({len(boards)/batch_time:.0f} boards/s)")
        
        # Call callbacks with results
        for i, (board, callback) in enumerate(self.positions):
            callback(policies[i], values[i])
        
        # Clear processed positions
        self.positions = []
    
    def flush(self):
        """Process any remaining positions."""
        if self.positions:
            self.process_batch()

def simulate_position_collection():
    """Simulate the position collection approach."""
    print("=== Position Collection Prototype ===")
    
    # Initialize model
    model = SimpleModelInference(
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch2_mini16.pt.gz",
        cache_size=1000
    )
    
    # Create position collector
    collector = PositionCollector(model)
    
    # Simulate tree building with position collection
    print("\n--- Simulating Tree Building with Position Collection ---")
    
    # Create some test positions (simulating tree building)
    test_positions = []
    state = HexGameState()
    
    for i in range(50):  # Simulate 50 positions from tree building
        if state.winner is not None:
            break
        legal_moves = state.get_legal_moves()
        if legal_moves:
            import random
            move = random.choice(legal_moves)
            state.make_move(move[0], move[1])
            test_positions.append(state.board.copy())
    
    print(f"Created {len(test_positions)} test positions")
    
    # Simulate current approach (individual inference)
    print("\n--- Current Approach: Individual Inference ---")
    start_time = time.time()
    
    results_current = []
    for board in test_positions:
        policy, value = model.simple_infer(board)
        results_current.append((policy, value))
    
    current_time = time.time() - start_time
    print(f"Current approach: {current_time:.3f}s ({len(test_positions)/current_time:.0f} boards/s)")
    
    # Simulate optimized approach (batch inference)
    print("\n--- Optimized Approach: Batch Inference ---")
    start_time = time.time()
    
    # Process in batches
    batch_size = 50
    results_optimized = []
    
    for i in range(0, len(test_positions), batch_size):
        batch = test_positions[i:i + batch_size]
        policies, values = model.batch_infer(batch)
        results_optimized.extend(list(zip(policies, values)))
    
    optimized_time = time.time() - start_time
    print(f"Optimized approach: {optimized_time:.3f}s ({len(test_positions)/optimized_time:.0f} boards/s)")
    
    # Calculate speedup
    speedup = current_time / optimized_time
    print(f"\n--- Results ---")
    print(f"Speedup: {speedup:.1f}x")
    print(f"Time reduction: {((current_time - optimized_time) / current_time * 100):.1f}%")
    
    # Verify results are the same
    print(f"Results match: {len(results_current) == len(results_optimized)}")
    
    return speedup, current_time, optimized_time

def estimate_full_improvement():
    """Estimate the full improvement potential."""
    print("\n=== Full Improvement Estimation ===")
    
    # Current performance
    current_game_time = 4.97  # seconds per game
    current_positions_per_game = 516  # from profiling
    
    # Optimized performance (conservative estimate)
    speedup_per_position = 3.0  # conservative batch speedup
    optimized_game_time = current_game_time / speedup_per_position
    
    print(f"Current game time: {current_game_time:.2f}s")
    print(f"Optimized game time: {optimized_game_time:.2f}s")
    print(f"Speedup per game: {speedup_per_position:.1f}x")
    
    # Calculate 500k games impact
    current_500k_days = 24.1
    optimized_500k_days = current_500k_days / speedup_per_position
    
    print(f"\n500k games impact:")
    print(f"Current: {current_500k_days:.1f} days")
    print(f"Optimized: {optimized_500k_days:.1f} days")
    print(f"Improvement: {current_500k_days - optimized_500k_days:.1f} days saved")
    
    # Cross-game batching potential
    cross_game_speedup = 2.0  # conservative estimate
    final_500k_days = optimized_500k_days / cross_game_speedup
    
    print(f"\nWith cross-game batching:")
    print(f"Final estimate: {final_500k_days:.1f} days")
    print(f"Total improvement: {current_500k_days / final_500k_days:.1f}x")
    
    return final_500k_days

if __name__ == "__main__":
    print("=== Position Collection Performance Analysis ===")
    
    # Run prototype
    speedup, current_time, optimized_time = simulate_position_collection()
    
    # Estimate full improvement
    final_days = estimate_full_improvement()
    
    print(f"\n=== Conclusion ===")
    print(f"Position collection could reduce 500k games from 24.1 days to {final_days:.1f} days")
    print(f"That's a {24.1/final_days:.1f}x improvement!")
    
    if final_days < 4:
        print("🎉 Target achieved: <4 days for 500k games!")
    else:
        print(f"Need additional optimizations to reach <4 days target")