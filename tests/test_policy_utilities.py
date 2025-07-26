#!/usr/bin/env python3
"""
Test the centralized policy processing utilities to ensure they work correctly
and produce the same results as the original duplicated code.
"""

import numpy as np
import pytest
from hex_ai.value_utils import (
    policy_logits_to_probs,
    get_legal_policy_probs,
    select_top_k_moves,
    sample_move_by_value,
    get_top_k_moves_with_probs,
    temperature_scaled_softmax,
)

def test_policy_logits_to_probs():
    """Test that policy_logits_to_probs produces the same result as temperature_scaled_softmax."""
    # Create sample logits
    logits = np.array([1.0, 2.0, 0.5, 3.0])
    temperature = 0.5
    
    # Test both functions produce the same result
    result1 = policy_logits_to_probs(logits, temperature)
    result2 = temperature_scaled_softmax(logits, temperature)
    
    np.testing.assert_array_almost_equal(result1, result2)
    
    # Test with temperature=1.0 (standard softmax)
    result3 = policy_logits_to_probs(logits, 1.0)
    result4 = temperature_scaled_softmax(logits, 1.0)
    np.testing.assert_array_almost_equal(result3, result4)

def test_get_legal_policy_probs():
    """Test legal move filtering."""
    # Create a 4x4 board (16 positions)
    policy_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16])
    legal_moves = [(0, 0), (1, 1), (2, 2), (3, 3)]  # Diagonal moves
    board_size = 4
    
    legal_policy = get_legal_policy_probs(policy_probs, legal_moves, board_size)
    
    # Expected: positions 0, 5, 10, 15 (diagonal)
    expected = np.array([0.1, 0.6, 0.11, 0.16])
    np.testing.assert_array_almost_equal(legal_policy, expected)

def test_select_top_k_moves():
    """Test top-k move selection."""
    legal_policy = np.array([0.1, 0.8, 0.3, 0.9, 0.2])
    legal_moves = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    k = 3
    
    top_moves = select_top_k_moves(legal_policy, legal_moves, k)
    
    # Expected: moves with highest probabilities (0.9, 0.8, 0.3)
    expected = [(1, 0), (0, 1), (0, 2)]
    assert top_moves == expected

def test_sample_move_by_value():
    """Test value-based move sampling."""
    move_values = [1.0, 2.0, 0.5, 3.0]
    temperature = 0.5
    
    # Test with multiple moves
    chosen_idx = sample_move_by_value(move_values, temperature)
    assert 0 <= chosen_idx < len(move_values)
    
    # Test with single move
    chosen_idx = sample_move_by_value([1.0], temperature)
    assert chosen_idx == 0
    
    # Test with empty list (should raise error)
    with pytest.raises(ValueError):
        sample_move_by_value([], temperature)

def test_get_top_k_moves_with_probs():
    """Test the combined utility function."""
    # Create sample logits for a 3x3 board (9 positions)
    policy_logits = np.array([1.0, 2.0, 0.5, 3.0, 1.5, 0.8, 2.5, 1.2, 0.9])
    legal_moves = [(0, 0), (0, 1), (1, 0), (1, 1)]  # Only 4 legal moves
    board_size = 3
    k = 2
    temperature = 1.0
    
    result = get_top_k_moves_with_probs(policy_logits, legal_moves, board_size, k, temperature)
    
    # Should return 2 moves with their probabilities
    assert len(result) == 2
    for move, prob in result:
        assert isinstance(move, tuple) and len(move) == 2
        assert isinstance(prob, float) and 0 <= prob <= 1

def test_consistency_with_original_logic():
    """Test that the new utilities produce the same results as the original duplicated logic."""
    # Simulate the original logic from model_select_move
    policy_logits = np.array([1.0, 2.0, 0.5, 3.0, 1.5, 0.8, 2.5, 1.2, 0.9])
    legal_moves = [(0, 0), (0, 1), (1, 0), (1, 1)]
    board_size = 3
    temperature = 0.5
    
    # Original logic (simulated)
    policy_probs_orig = temperature_scaled_softmax(policy_logits, temperature)
    move_indices_orig = [row * board_size + col for row, col in legal_moves]
    legal_policy_orig = np.array([policy_probs_orig[idx] for idx in move_indices_orig])
    topk_idx_orig = np.argsort(legal_policy_orig)[::-1][:2]
    topk_moves_orig = [legal_moves[i] for i in topk_idx_orig]
    
    # New logic
    result_new = get_top_k_moves_with_probs(policy_logits, legal_moves, board_size, 2, temperature)
    topk_moves_new = [move for move, _ in result_new]
    
    # Should produce the same moves
    assert topk_moves_new == topk_moves_orig

if __name__ == "__main__":
    pytest.main([__file__]) 