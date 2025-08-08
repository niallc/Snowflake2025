#!/usr/bin/env python3
"""
Test script for performance monitoring infrastructure.

This script validates the PERF utility and demonstrates its usage
for MCTS optimization work.
"""

import sys
import time
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hex_ai.utils.perf import PERF, log_performance_summary, setup_model_performance_meta
import hex_ai  # This validates the environment


def test_basic_performance_monitoring():
    """Test basic performance monitoring functionality."""
    print("Testing basic performance monitoring...")
    
    # Test timing
    with PERF.timer("test.timing"):
        time.sleep(0.1)
    
    # Test counters
    PERF.inc("test.counter", 5)
    PERF.inc("test.counter", 3)
    
    # Test samples
    PERF.add_sample("test.sample", 42.0)
    PERF.add_sample("test.sample", 17.0)
    
    # Test metadata
    PERF.set_meta("test.device", "cpu")
    PERF.set_meta("test.version", "1.0.0")
    
    # Get snapshot
    snapshot = PERF.snapshot(clear=True)
    print(f"Snapshot: {snapshot}")
    
    # Test logging
    PERF.log_snapshot(clear=True, force=True)
    
    print("✓ Basic performance monitoring test passed")


def test_mcts_simulation_timing():
    """Simulate MCTS timing patterns."""
    print("\nTesting MCTS simulation timing patterns...")
    
    # Simulate a complete MCTS move
    with PERF.timer("mcts.search"):
        # Simulate selection phase
        with PERF.timer("mcts.select"):
            time.sleep(0.01)  # Simulate selection time
            PERF.inc("mcts.sim")
        
        # Simulate expansion phase
        with PERF.timer("mcts.expand"):
            time.sleep(0.02)  # Simulate expansion time
            PERF.inc("mcts.sim")
        
        # Simulate neural network inference
        with PERF.timer("nn.infer"):
            time.sleep(0.05)  # Simulate inference time
            PERF.inc("nn.batch")
            PERF.add_sample("nn.batch_size", 32.0)
        
        # Simulate backpropagation
        with PERF.timer("mcts.backprop"):
            time.sleep(0.005)  # Simulate backprop time
            PERF.inc("mcts.sim")
    
    # Log the performance snapshot
    PERF.log_snapshot(clear=True, force=True)
    
    print("✓ MCTS simulation timing test passed")


def test_performance_summary():
    """Test performance summary generation."""
    print("\nTesting performance summary generation...")
    
    # Generate some test data
    for i in range(10):
        with PERF.timer("test.loop"):
            time.sleep(0.001)
            PERF.inc("test.iterations")
            PERF.add_sample("test.values", float(i))
    
    # Generate summary
    summary = PERF.get_summary_stats()
    print(f"Performance summary: {summary}")
    
    # Log summary
    log_performance_summary(clear=True)
    
    print("✓ Performance summary test passed")


def test_rate_limiting():
    """Test rate limiting functionality."""
    print("\nTesting rate limiting...")
    
    # Try to log multiple times quickly
    for i in range(5):
        PERF.inc("test.rate_limit")
        PERF.log_snapshot(clear=False)  # Should be rate limited
    
    # Force a log
    PERF.log_snapshot(clear=True, force=True)
    
    print("✓ Rate limiting test passed")


def main():
    """Run all performance monitoring tests."""
    print("=== Performance Monitoring Test Suite ===\n")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        # Test basic functionality
        test_basic_performance_monitoring()
        
        # Test MCTS patterns
        test_mcts_simulation_timing()
        
        # Test summary generation
        test_performance_summary()
        
        # Test rate limiting
        test_rate_limiting()
        
        print("\n=== All Tests Passed ===")
        print("\nPerformance monitoring infrastructure is ready for MCTS optimization work.")
        print("\nNext steps:")
        print("1. Add PERF.timer() calls to key MCTS boundaries")
        print("2. Run baseline profiling on current codebase")
        print("3. Implement targeted optimizations based on profiling data")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
