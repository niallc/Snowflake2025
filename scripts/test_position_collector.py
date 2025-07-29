#!/usr/bin/env python3
"""
Unit tests for PositionCollector to verify correctness.

This module tests the PositionCollector class, which is responsible for:
1. Collecting board positions during tree building
2. Batching inference requests to minimize GPU calls
3. Mapping results back to the correct callbacks

The tests verify:
- Basic functionality (policy and value requests work)
- Batching behavior (multiple requests processed together)
- Edge cases (empty requests, mixed request types)
- Correctness (batched results match individual results)
"""

import sys
import os
import numpy as np

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.fixed_tree_search import PositionCollector

def test_position_collector_basic():
    """
    Test basic PositionCollector functionality.
    
    **Aim**: Verify that PositionCollector can handle basic policy and value requests.
    
    **Design**: 
    1. Create a PositionCollector with a real model
    2. Make 2 policy requests and 2 value requests for different board states
    3. Process the batches and verify all callbacks are called
    4. Check that results have correct data types and shapes
    
    **Expected**: All 4 callbacks should be called with valid results.
    """
    print("=== Testing PositionCollector Basic Functionality ===")
    
    # Initialize model
    model = SimpleModelInference(
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch2_mini16.pt.gz"
    )
    
    # Create position collector
    collector = PositionCollector(model)
    
    # Create test states
    state1 = HexGameState()
    state2 = state1.make_move(6, 6)  # Center move
    state3 = state2.make_move(7, 7)  # Another move
    
    # Test policy requests
    policy_results = []
    def policy_callback1(policy):
        policy_results.append(("state1", policy))
    
    def policy_callback2(policy):
        policy_results.append(("state2", policy))
    
    collector.request_policy(state1.board, policy_callback1)
    collector.request_policy(state2.board, policy_callback2)
    
    # Test value requests
    value_results = []
    def value_callback1(value):
        value_results.append(("state1", value))
    
    def value_callback2(value):
        value_results.append(("state2", value))
    
    collector.request_value(state1.board, value_callback1)
    collector.request_value(state2.board, value_callback2)
    
    # Process batches
    print(f"Processing {len(collector.policy_requests)} policy requests and {len(collector.value_requests)} value requests...")
    collector.process_batches()
    
    # Verify results
    print(f"Policy results: {len(policy_results)}")
    print(f"Value results: {len(value_results)}")
    
    assert len(policy_results) == 2, f"Expected 2 policy results, got {len(policy_results)}"
    assert len(value_results) == 2, f"Expected 2 value results, got {len(value_results)}"
    
    # Verify callbacks were called with correct data
    for state_name, result in policy_results:
        assert isinstance(result, np.ndarray), f"Policy result for {state_name} should be numpy array"
        assert result.shape == (169,), f"Policy result for {state_name} should have shape (169,), got {result.shape}"
    
    for state_name, result in value_results:
        assert isinstance(result, float), f"Value result for {state_name} should be float"
    
    print("✅ Basic PositionCollector test passed!")

def test_position_collector_batching():
    """
    Test that PositionCollector correctly batches requests.
    
    **Aim**: Verify that multiple requests are processed together in batches.
    
    **Design**:
    1. Create 5 different board states (empty board + 4 moves)
    2. Make policy requests for all 5 states
    3. Process batches and verify all callbacks are called in order
    4. Check that results are correctly mapped to the right indices
    
    **Expected**: All 5 callbacks should be called in the correct order with valid results.
    """
    print("\n=== Testing PositionCollector Batching ===")
    
    # Initialize model
    model = SimpleModelInference(
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch2_mini16.pt.gz"
    )
    
    # Create position collector
    collector = PositionCollector(model)
    
    # Create multiple states
    states = []
    current_state = HexGameState()
    for i in range(5):
        states.append(current_state)
        if i < 4:  # Don't make move after last state
            current_state = current_state.make_move(i % 13, (i + 1) % 13)
    
    # Collect policy requests
    policy_callbacks = []
    for i, state in enumerate(states):
        def make_callback(index):
            def callback(policy):
                policy_callbacks.append((index, policy))
            return callback
        
        collector.request_policy(state.board, make_callback(i))
    
    # Process batches
    print(f"Processing {len(collector.policy_requests)} policy requests...")
    collector.process_batches()
    
    # Verify all callbacks were called
    assert len(policy_callbacks) == 5, f"Expected 5 policy callbacks, got {len(policy_callbacks)}"
    
    # Verify callback order matches request order
    for i, (index, policy) in enumerate(policy_callbacks):
        assert index == i, f"Callback {i} should have index {i}, got {index}"
        assert isinstance(policy, np.ndarray), f"Policy {i} should be numpy array"
        assert policy.shape == (169,), f"Policy {i} should have shape (169,), got {policy.shape}"
    
    print("✅ PositionCollector batching test passed!")

def test_position_collector_empty():
    """
    Test PositionCollector with no requests.
    
    **Aim**: Verify that PositionCollector handles empty request lists gracefully.
    
    **Design**:
    1. Create a PositionCollector
    2. Call process_batches() without making any requests
    3. Verify no errors occur
    
    **Expected**: Should not crash or raise errors.
    """
    print("\n=== Testing PositionCollector Empty ===")
    
    # Initialize model
    model = SimpleModelInference(
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch2_mini16.pt.gz"
    )
    
    # Create position collector
    collector = PositionCollector(model)
    
    # Process empty batches
    print("Processing empty batches...")
    collector.process_batches()
    
    # Should not crash
    print("✅ Empty PositionCollector test passed!")

def test_position_collector_mixed():
    """
    Test PositionCollector with mixed policy and value requests.
    
    **Aim**: Verify that PositionCollector can handle mixed request types correctly.
    
    **Design**:
    1. Create 2 board states
    2. Make mixed requests: policy, value, policy, value
    3. Process batches and verify all callbacks are called
    4. Count policy vs value results and verify correct types
    
    **Expected**: Should get 2 policy results and 2 value results with correct data types.
    """
    print("\n=== Testing PositionCollector Mixed Requests ===")
    
    # Initialize model
    model = SimpleModelInference(
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch2_mini16.pt.gz"
    )
    
    # Create position collector
    collector = PositionCollector(model)
    
    # Create test states
    state1 = HexGameState()
    state2 = state1.make_move(6, 6)
    
    # Collect mixed requests
    results = []
    
    def policy_callback(policy):
        results.append(("policy", policy))
    
    def value_callback(value):
        results.append(("value", value))
    
    # Add requests in mixed order
    collector.request_policy(state1.board, policy_callback)
    collector.request_value(state1.board, value_callback)
    collector.request_policy(state2.board, policy_callback)
    collector.request_value(state2.board, value_callback)
    
    # Process batches
    print(f"Processing {len(collector.policy_requests)} policy and {len(collector.value_requests)} value requests...")
    collector.process_batches()
    
    # Verify results
    assert len(results) == 4, f"Expected 4 results, got {len(results)}"
    
    # Count policy and value results
    policy_count = sum(1 for result_type, _ in results if result_type == "policy")
    value_count = sum(1 for result_type, _ in results if result_type == "value")
    
    assert policy_count == 2, f"Expected 2 policy results, got {policy_count}"
    assert value_count == 2, f"Expected 2 value results, got {value_count}"
    
    # Verify data types
    for result_type, result in results:
        if result_type == "policy":
            assert isinstance(result, np.ndarray), f"Policy result should be numpy array"
            assert result.shape == (169,), f"Policy result should have shape (169,), got {result.shape}"
        else:  # value
            assert isinstance(result, float), f"Value result should be float"
    
    print("✅ Mixed PositionCollector test passed!")

def test_position_collector_correctness():
    """
    Test that PositionCollector produces same results as individual inference.
    
    **Aim**: Verify that batched inference produces identical results to individual inference.
    
    **Design**:
    1. Create 2 board states
    2. Get results using individual model.simple_infer() calls
    3. Get results using PositionCollector batched inference
    4. Compare results and verify they are identical (within floating point precision)
    
    **Expected**: Batched results should be identical to individual results.
    """
    print("\n=== Testing PositionCollector Correctness ===")
    
    # Initialize model
    model = SimpleModelInference(
        "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408/epoch2_mini16.pt.gz"
    )
    
    # Create test states
    state1 = HexGameState()
    state2 = state1.make_move(6, 6)
    
    # Get individual results
    policy1_ind, value1_ind = model.simple_infer(state1.board)
    policy2_ind, value2_ind = model.simple_infer(state2.board)
    
    # Get batched results
    collector = PositionCollector(model)
    
    policy1_batch = None
    policy2_batch = None
    value1_batch = None
    value2_batch = None
    
    def policy1_callback(policy):
        nonlocal policy1_batch
        policy1_batch = policy
    
    def policy2_callback(policy):
        nonlocal policy2_batch
        policy2_batch = policy
    
    def value1_callback(value):
        nonlocal value1_batch
        value1_batch = value
    
    def value2_callback(value):
        nonlocal value2_batch
        value2_batch = value
    
    collector.request_policy(state1.board, policy1_callback)
    collector.request_policy(state2.board, policy2_callback)
    collector.request_value(state1.board, value1_callback)
    collector.request_value(state2.board, value2_callback)
    
    collector.process_batches()
    
    # Compare results
    policy1_diff = np.abs(policy1_ind - policy1_batch).max()
    policy2_diff = np.abs(policy2_ind - policy2_batch).max()
    value1_diff = abs(value1_ind - value1_batch)
    value2_diff = abs(value2_ind - value2_batch)
    
    print(f"Policy 1 max difference: {policy1_diff}")
    print(f"Policy 2 max difference: {policy2_diff}")
    print(f"Value 1 difference: {value1_diff}")
    print(f"Value 2 difference: {value2_diff}")
    
    # Results should be identical (within floating point precision)
    assert policy1_diff < 1e-6, f"Policy 1 difference too large: {policy1_diff}"
    assert policy2_diff < 1e-6, f"Policy 2 difference too large: {policy2_diff}"
    assert value1_diff < 1e-6, f"Value 1 difference too large: {value1_diff}"
    assert value2_diff < 1e-6, f"Value 2 difference too large: {value2_diff}"
    
    print("✅ PositionCollector correctness test passed!")

if __name__ == "__main__":
    print("Running PositionCollector unit tests...")
    
    test_position_collector_basic()
    test_position_collector_batching()
    test_position_collector_empty()
    test_position_collector_mixed()
    test_position_collector_correctness()
    
    print("\n🎉 All PositionCollector tests passed!")