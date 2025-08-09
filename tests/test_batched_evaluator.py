"""
Tests for the BatchedEvaluator interface.
"""

import pytest
import numpy as np
import time
from unittest.mock import Mock, MagicMock

from hex_ai.inference.batched_evaluator import BatchedEvaluator
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference


class TestBatchedEvaluator:
    """Test cases for BatchedEvaluator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock model that returns results for any batch size
        self.mock_model = Mock(spec=SimpleModelInference)
        
        def mock_batch_infer(boards):
            # boards is a list of numpy arrays
            batch_size = len(boards)
            return (
                [np.random.randn(169) for _ in range(batch_size)],  # policy logits (list of arrays)
                [np.random.uniform(-1, 1) for _ in range(batch_size)]  # values (list of floats)
            )
        
        self.mock_model.batch_infer.side_effect = mock_batch_infer
        
        # Create evaluator with mock model
        self.evaluator = BatchedEvaluator(
            model=self.mock_model,
            optimal_batch_size=4,
            verbose=0,
            enable_background_processing=False  # Disable for testing
        )
        
        # Create a test game state
        self.test_state = HexGameState()
    
    def test_initialization(self):
        """Test that BatchedEvaluator initializes correctly."""
        assert self.evaluator.model == self.mock_model
        assert self.evaluator.optimal_batch_size == 4
        assert self.evaluator.verbose == 0
        assert self.evaluator.enable_background_processing == False
    
    def test_request_evaluation(self):
        """Test that evaluation requests are handled correctly."""
        callback_called = False
        callback_results = None
        
        def test_callback(policy_logits, value):
            nonlocal callback_called, callback_results
            callback_called = True
            callback_results = (policy_logits, value)
        
        # Request evaluation
        success = self.evaluator.request_evaluation(
            state=self.test_state,
            callback=test_callback,
            metadata={'test': True}
        )
        
        # Should return False because it was queued (not cached)
        assert success == False
        assert self.evaluator.get_queue_size() == 1
    
    def test_process_pending_evaluations(self):
        """Test that pending evaluations are processed correctly."""
        callback_called = False
        
        def test_callback(policy_logits, value):
            nonlocal callback_called
            callback_called = True
        
        # Request evaluation
        self.evaluator.request_evaluation(
            state=self.test_state,
            callback=test_callback
        )
        
        # Process pending evaluations
        processed = self.evaluator.process_pending_evaluations(force=True)
        
        assert processed == 1
        assert callback_called == True
        assert self.evaluator.get_queue_size() == 0
    
    def test_get_statistics(self):
        """Test that statistics are collected correctly."""
        # Request a few evaluations
        for _ in range(3):
            self.evaluator.request_evaluation(
                state=self.test_state,
                callback=lambda p, v: None
            )
        
        # Process them
        self.evaluator.process_pending_evaluations(force=True)
        
        # Get statistics
        stats = self.evaluator.get_statistics()
        
        assert stats['total_evaluations'] == 3
        assert stats['queue_size'] == 0
        assert 'cache_hit_rate' in stats
        assert 'average_batch_size' in stats
    
    def test_reset_statistics(self):
        """Test that statistics can be reset."""
        # Request and process an evaluation
        self.evaluator.request_evaluation(
            state=self.test_state,
            callback=lambda p, v: None
        )
        self.evaluator.process_pending_evaluations(force=True)
        
        # Verify statistics are non-zero
        stats_before = self.evaluator.get_statistics()
        assert stats_before['total_evaluations'] > 0
        
        # Reset statistics
        self.evaluator.reset_statistics()
        
        # Verify statistics are reset
        stats_after = self.evaluator.get_statistics()
        assert stats_after['total_evaluations'] == 0
    
    def test_wait_for_completion(self):
        """Test that wait_for_completion works correctly."""
        callback_called = False
        
        def test_callback(policy_logits, value):
            nonlocal callback_called
            callback_called = True
        
        # Request evaluation
        self.evaluator.request_evaluation(
            state=self.test_state,
            callback=test_callback
        )
        
        # Process the evaluation first
        self.evaluator.process_pending_evaluations(force=True)
        
        # Wait for completion
        completed = self.evaluator.wait_for_completion(timeout_seconds=1.0)
        
        # Should complete successfully
        assert completed == True
        assert callback_called == True
    
    def test_cleanup(self):
        """Test that cleanup works correctly."""
        # This should not raise any exceptions
        self.evaluator.cleanup()
    
    def test_set_optimal_batch_size(self):
        """Test that batch size can be updated."""
        original_size = self.evaluator.optimal_batch_size
        new_size = 128
        
        self.evaluator.set_optimal_batch_size(new_size)
        
        assert self.evaluator.optimal_batch_size == new_size
        assert self.evaluator.batch_processor.optimal_batch_size == new_size


if __name__ == "__main__":
    pytest.main([__file__])
