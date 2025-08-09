"""
Unit tests for the batched MCTS implementation.
"""

import unittest
import numpy as np
import tempfile
import os
from unittest.mock import Mock, patch

from hex_ai.inference.batch_processor import BatchProcessor, BatchRequest
from hex_ai.inference.batched_mcts import BatchedNeuralMCTS, BatchedMCTSNode, NodeState
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.simple_model_inference import SimpleModelInference


class TestBatchProcessor(unittest.TestCase):
    """Test the BatchProcessor class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model that returns results for any batch size
        self.mock_model = Mock()
        
        def mock_batch_infer(boards):
            batch_size = len(boards)
            return (
                [np.random.rand(169) for _ in range(batch_size)],  # policies
                [np.random.uniform(-1, 1) for _ in range(batch_size)]  # values
            )
        
        self.mock_model.batch_infer.side_effect = mock_batch_infer
        
        self.processor = BatchProcessor(self.mock_model, optimal_batch_size=2)
    
    def test_request_evaluation_cache_hit(self):
        """Test that cache hits return immediately."""
        # Add a result to cache
        board_state = np.random.rand(13, 13)
        cache_key = board_state.tobytes()
        cached_policy = np.random.rand(169)
        cached_value = 0.7
        self.processor.result_cache[cache_key] = (cached_policy, cached_value)
        
        # Mock callback
        callback_called = False
        callback_result = None
        
        def callback(policy, value):
            nonlocal callback_called, callback_result
            callback_called = True
            callback_result = (policy, value)
        
        # Request evaluation
        result = self.processor.request_evaluation(board_state, callback)
        
        # Should return True (cache hit) and call callback immediately
        self.assertTrue(result)
        self.assertTrue(callback_called)
        self.assertEqual(callback_result, (cached_policy, cached_value))
        self.assertEqual(self.processor.stats['cache_hits'], 1)
    
    def test_request_evaluation_cache_miss(self):
        """Test that cache misses queue the request."""
        board_state = np.random.rand(13, 13)
        
        # Mock callback
        callback_called = False
        
        def callback(policy, value):
            nonlocal callback_called
            callback_called = True
        
        # Request evaluation
        result = self.processor.request_evaluation(board_state, callback)
        
        # Should return False (cache miss) and queue the request
        self.assertFalse(result)
        self.assertFalse(callback_called)
        self.assertEqual(len(self.processor.request_queue), 1)
        self.assertEqual(self.processor.stats['cache_misses'], 1)
    
    def test_process_batch(self):
        """Test batch processing."""
        # Add requests to queue
        board_states = [np.random.rand(13, 13) for _ in range(3)]
        callbacks = []
        
        for board_state in board_states:
            def make_callback():
                def callback(policy, value):
                    pass
                return callback
            
            callbacks.append(make_callback())
            self.processor.request_evaluation(board_state, callbacks[-1])
        
        # Process batch
        processed = self.processor.process_batch(force=True)
        
        # Should process all requests
        self.assertEqual(processed, 3)
        self.assertEqual(len(self.processor.request_queue), 0)
        self.assertEqual(self.processor.stats['total_batches_processed'], 1)
        self.assertEqual(self.processor.stats['total_inferences'], 3)
        
        # Should have called batch_infer
        self.mock_model.batch_infer.assert_called_once()
        called_boards = self.mock_model.batch_infer.call_args[0][0]
        self.assertEqual(len(called_boards), 3)
    
    def test_get_statistics(self):
        """Test statistics calculation."""
        # Add some activity
        board_state = np.random.rand(13, 13)
        self.processor.request_evaluation(board_state, lambda p, v: None)
        self.processor.process_batch(force=True)
        
        stats = self.processor.get_statistics()
        
        # Check that all expected keys are present
        expected_keys = [
            'total_requests', 'cache_hits', 'cache_misses', 'total_batches_processed',
            'total_inferences', 'total_time', 'average_batch_size', 'cache_hit_rate',
            'inferences_per_second'
        ]
        for key in expected_keys:
            self.assertIn(key, stats)


class TestBatchedMCTSNode(unittest.TestCase):
    """Test the BatchedMCTSNode class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.state = HexGameState()
        self.node = BatchedMCTSNode(state=self.state)
    
    def test_initial_state(self):
        """Test initial node state."""
        self.assertEqual(self.node.node_state, NodeState.UNEXPANDED)
        self.assertEqual(self.node.visits, 0)
        self.assertEqual(self.node.total_value, 0.0)
        self.assertEqual(self.node.virtual_loss, 0)
        self.assertTrue(self.node.is_leaf())
        self.assertFalse(self.node.is_terminal())
    
    def test_virtual_loss(self):
        """Test virtual loss functionality."""
        self.node.add_virtual_loss(2)
        self.assertEqual(self.node.virtual_loss, 2)
        
        self.node.remove_virtual_loss(1)
        self.assertEqual(self.node.virtual_loss, 1)
        
        self.node.remove_virtual_loss(5)  # More than current
        self.assertEqual(self.node.virtual_loss, 0)  # Should not go negative
    
    def test_update_statistics(self):
        """Test statistics update."""
        self.node.update_statistics(0.5)
        self.assertEqual(self.node.visits, 1)
        self.assertEqual(self.node.total_value, 0.5)
        self.assertEqual(self.node.mean_value, 0.5)
        
        self.node.update_statistics(-0.3)
        self.assertEqual(self.node.visits, 2)
        self.assertEqual(self.node.total_value, 0.2)
        self.assertEqual(self.node.mean_value, 0.1)


class TestBatchedNeuralMCTS(unittest.TestCase):
    """Test the BatchedNeuralMCTS class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model that returns results for any batch size
        self.mock_model = Mock()
        
        def mock_batch_infer(boards):
            batch_size = len(boards)
            return (
                [np.random.rand(169) for _ in range(batch_size)],  # policies
                [np.random.uniform(-1, 1) for _ in range(batch_size)]  # values
            )
        
        self.mock_model.batch_infer.side_effect = mock_batch_infer
        
        self.mcts = BatchedNeuralMCTS(
            model=self.mock_model,
            exploration_constant=1.4,
            optimal_batch_size=2,
            verbose=0
        )
    
    def test_initialization(self):
        """Test MCTS initialization."""
        self.assertEqual(self.mcts.exploration_constant, 1.4)
        self.assertEqual(self.mcts.optimal_batch_size, 2)
        self.assertIsNotNone(self.mcts.evaluator)
    
    def test_search_new_state(self):
        """Test search with new game state."""
        state = HexGameState()
        
        # Mock the evaluator to avoid actual inference
        with patch.object(self.mcts.evaluator, 'request_evaluation') as mock_request:
            mock_request.return_value = False  # Cache miss
            
            with patch.object(self.mcts.evaluator, 'process_pending_evaluations') as mock_process:
                mock_process.return_value = 0  # No processing
                
                root = self.mcts.search(state, num_simulations=10)
                
                # Should create a new root node
                self.assertIsInstance(root, BatchedMCTSNode)
                # Compare the board arrays instead of the full state objects
                np.testing.assert_array_equal(root.state.board, state.board)
                # The node state may change during search, so just check it's a valid state
                self.assertIn(root.node_state, [NodeState.UNEXPANDED, NodeState.EVALUATION_PENDING, NodeState.EXPANDED])
    
    def test_search_existing_node(self):
        """Test search with existing node."""
        state = HexGameState()
        existing_node = BatchedMCTSNode(state=state)
        existing_node.visits = 5
        existing_node.total_value = 2.0
        
        # Mock the evaluator
        with patch.object(self.mcts.evaluator, 'request_evaluation') as mock_request:
            mock_request.return_value = False
            
            with patch.object(self.mcts.evaluator, 'process_pending_evaluations') as mock_process:
                mock_process.return_value = 0
                
                root = self.mcts.search(existing_node, num_simulations=10)
                
                # Should return the same node
                self.assertIs(root, existing_node)
    
    def test_get_search_statistics(self):
        """Test statistics retrieval."""
        stats = self.mcts.get_search_statistics()
        
        # Check that all expected keys are present
        expected_keys = [
            'total_simulations', 'total_inferences', 'total_batches_processed',
            'start_time', 'end_time', 'total_evaluations', 'queue_size', 'cache_size',
            'cache_hit_rate', 'average_batch_size', 'inferences_per_second'
        ]
        for key in expected_keys:
            self.assertIn(key, stats)
    
    def test_reset_statistics(self):
        """Test statistics reset."""
        # Set some statistics
        self.mcts.stats['total_simulations'] = 100
        self.mcts.stats['total_inferences'] = 50
        
        self.mcts.reset_search_statistics()
        
        # Should be reset
        self.assertEqual(self.mcts.stats['total_simulations'], 0)
        self.assertEqual(self.mcts.stats['total_inferences'], 0)


if __name__ == '__main__':
    unittest.main()
