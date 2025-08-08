#!/usr/bin/env python3
"""
Test script for MCTS performance instrumentation.

This script runs a quick MCTS search to verify that the performance
monitoring is working correctly and to establish a baseline.
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import hex_ai  # This validates the environment
from hex_ai.inference.batched_mcts import BatchedNeuralMCTS
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.utils.perf import PERF


def test_mcts_performance():
    """Test MCTS performance instrumentation."""
    print("Testing MCTS performance instrumentation...")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create a simple mock model for testing
    class MockModel:
        def __init__(self):
            self.cache = {}
        
        def batch_infer(self, boards):
            """Mock batch inference that returns random results."""
            import numpy as np
            
            print(f"MockModel.batch_infer called with {len(boards)} boards")
            
            policies = []
            values = []
            
            for i, board in enumerate(boards):
                # Create a cache key
                cache_key = board.tobytes()
                
                if cache_key in self.cache:
                    # Return cached result
                    policies.append(self.cache[cache_key][0])
                    values.append(self.cache[cache_key][1])
                    print(f"  Board {i}: returning cached result")
                else:
                    # Generate new result
                    policy = np.random.rand(169)  # 13x13 board flattened
                    value = np.random.uniform(-1.0, 1.0)
                    
                    # Cache the result
                    self.cache[cache_key] = (policy, value)
                    
                    policies.append(policy)
                    values.append(value)
                    print(f"  Board {i}: generated new result (policy shape: {policy.shape}, value: {value:.3f})")
            
            print(f"MockModel.batch_infer returning {len(policies)} policies and {len(values)} values")
            return policies, values
    
    # Create mock model and MCTS engine
    model = MockModel()
    mcts = BatchedNeuralMCTS(
        model=model,
        exploration_constant=1.4,
        optimal_batch_size=8,  # Small batch size for testing
        verbose=2  # Increase verbosity for debugging
    )
    
    # Create a game state
    state = HexGameState()
    
    print(f"Starting MCTS search with 100 simulations...")
    
    # Run MCTS search
    start_time = time.time()
    root = mcts.search(state, num_simulations=100)
    total_time = time.time() - start_time
    
    print(f"MCTS search completed in {total_time:.3f}s")
    print(f"Root node state: {root.node_state}")
    print(f"Root node is leaf: {root.is_leaf()}")
    print(f"Root node is terminal: {root.is_terminal()}")
    print(f"Root node children: {len(root.children)}")
    
    # Get search statistics
    search_stats = mcts.get_search_statistics()
    print(f"Search statistics: {search_stats}")
    
    # Get performance snapshot
    perf_snapshot = PERF.snapshot(clear=True, force=True)
    print(f"Performance snapshot: {perf_snapshot}")
    
    # Analyze the results
    if perf_snapshot:
        timings = perf_snapshot.get('timings_s', {})
        counters = perf_snapshot.get('counters', {})
        samples = perf_snapshot.get('samples', {})
        
        print("\n=== Performance Analysis ===")
        print(f"Total simulations: {counters.get('mcts.sim', 0)}")
        print(f"Total batches: {counters.get('nn.batch', 0)}")
        
        total_time = sum(timings.values())
        if total_time > 0:
            print(f"Total time: {total_time:.3f}s")
            print(f"Simulations per second: {counters.get('mcts.sim', 0) / total_time:.1f}")
            
            print("\nTime distribution:")
            for phase, phase_time in timings.items():
                pct = (phase_time / total_time * 100) if total_time > 0 else 0
                print(f"  {phase}: {phase_time:.3f}s ({pct:.1f}%)")
        
        # Batch size analysis
        if 'nn.batch_size' in samples:
            count, total = samples['nn.batch_size']
            avg_batch_size = total / count if count > 0 else 0
            print(f"\nBatch analysis:")
            print(f"  Average batch size: {avg_batch_size:.1f}")
            print(f"  Total batches: {count}")
    
    print("\n✓ MCTS performance instrumentation test completed")


if __name__ == "__main__":
    test_mcts_performance()
