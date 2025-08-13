"""
Batched evaluator for neural network inference in MCTS.

This module provides a centralized interface for all neural network evaluations,
separating NN concerns from MCTS logic and enabling better performance monitoring.
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Callable
import numpy as np

from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.inference.batch_processor import BatchProcessor
from hex_ai.inference.game_engine import HexGameState
from hex_ai.utils.perf import PERF

logger = logging.getLogger(__name__)


class BatchedEvaluator:
    """
    Centralized interface for all neural network evaluations in MCTS.
    
    This class provides a clean separation between MCTS logic and neural network
    inference, enabling better performance monitoring and optimization.
    
    Key Features:
    - Automatic batching of evaluation requests
    - Performance monitoring and statistics
    - Result caching to avoid redundant evaluations
    - Clean callback-based result distribution
    """
    
    def __init__(self, model: SimpleModelInference, optimal_batch_size: int = 64, 
                 verbose: int = 1, max_wait_ms: int = 15, 
                 enable_background_processing: bool = True):
        """
        Initialize the batched evaluator.
        
        Args:
            model: Neural network model for policy and value predictions
            optimal_batch_size: Target batch size for optimal GPU utilization
            verbose: Verbosity level (0=quiet, 1=normal, 2=detailed, 3=debug)
            max_wait_ms: Maximum wait time in milliseconds before processing small batches
            enable_background_processing: Whether to enable background thread for automatic processing
        """
        self.model = model
        self.optimal_batch_size = optimal_batch_size
        self.verbose = verbose
        self.max_wait_ms = max_wait_ms
        self.enable_background_processing = enable_background_processing
        
        # Initialize batch processor
        self.batch_processor = BatchProcessor(
            model=model,
            optimal_batch_size=optimal_batch_size,
            verbose=verbose,
            max_wait_ms=max_wait_ms,
            enable_background_processing=enable_background_processing
        )
        
        # Performance statistics
        self.stats = {
            'total_evaluations': 0,
            'start_time': None,
            'end_time': None
        }
        
        logger.info(
            f"BatchedEvaluator initialized with optimal_batch_size={optimal_batch_size}, "
            f"max_wait_ms={max_wait_ms}, background_processing={enable_background_processing}"
        )

    def request_evaluation(self, state: HexGameState, 
                          callback: Callable[[np.ndarray, float], None],
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Request evaluation of a game state.
        
        Args:
            state: Game state to evaluate
            callback: Function to call with results (policy_logits, value)
            metadata: Optional metadata for debugging
            
        Returns:
            True if request was queued successfully, False otherwise
        """
        with PERF.timer("evaluator.request"):
            # Convert state to board tensor (keep as torch tensor on CPU)
            board_tensor = state.get_board_tensor()
            if not hasattr(board_tensor, 'cpu'):
                raise ValueError(f"Expected PyTorch tensor from get_board_tensor(), got {type(board_tensor)}")
            board_tensor = board_tensor.detach().to('cpu', copy=True).contiguous()

            # Minimal metadata; avoid heavy hashing
            if metadata is None:
                metadata = {}
            metadata.update({'evaluator_timestamp': time.time()})  # TODO: consider lightweight state hash if needed for debugging

            # Request evaluation from batch processor (pass torch tensor)
            success = self.batch_processor.request_evaluation(
                board_state=board_tensor,
                callback=callback,
                metadata=metadata
            )
            
            # Increment counter for all requests (both cached and queued)
            self.stats['total_evaluations'] += 1
                
            return success

    def process_pending_evaluations(self, force: bool = False) -> int:
        """
        Process any pending evaluation requests.
        
        Args:
            force: Whether to force processing even if batch is not full
            
        Returns:
            Number of evaluations processed
        """
        with PERF.timer("evaluator.process"):
            return self.batch_processor.process_batch(force=force)

    def wait_for_completion(self, timeout_seconds: float = 1.0) -> bool:
        """
        Wait for all pending evaluations to complete.
        
        Args:
            timeout_seconds: Maximum time to wait
            
        Returns:
            True if all evaluations completed, False if timeout
        """
        start_time = time.time()
        while self.get_queue_size() > 0:
            if time.time() - start_time > timeout_seconds:
                return False
            time.sleep(0.001)  # Sleep for 1ms
        return True

    def get_queue_size(self) -> int:
        """Get the number of pending evaluation requests."""
        return self.batch_processor.get_queue_size()

    def get_cache_size(self) -> int:
        """Get the number of cached evaluation results."""
        return self.batch_processor.get_cache_size()

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive evaluation statistics."""
        stats = self.stats.copy()
        
        # Add batch processor statistics
        batch_stats = self.batch_processor.get_statistics()
        stats.update({
            'batch_processor': batch_stats,
            'queue_size': self.get_queue_size(),
            'cache_size': self.get_cache_size(),
            'cache_hit_rate': batch_stats.get('cache_hit_rate', 0.0),
            'average_batch_size': batch_stats.get('average_batch_size', 0.0),
            'inferences_per_second': batch_stats.get('inferences_per_second', 0.0)
        })
        
        # Calculate timing statistics
        if stats['start_time'] and stats['end_time']:
            stats['total_time'] = stats['end_time'] - stats['start_time']
            if stats['total_evaluations'] > 0:
                stats['evaluations_per_second'] = stats['total_evaluations'] / stats['total_time']
        
        return stats

    def reset_statistics(self):
        """Reset evaluation statistics."""
        self.stats['total_evaluations'] = 0
        self.stats['start_time'] = None
        self.stats['end_time'] = None
        self.batch_processor.reset_statistics()

    def clear_cache(self):
        """Clear the evaluation result cache."""
        self.batch_processor.clear_cache()

    def set_optimal_batch_size(self, batch_size: int):
        """Update the optimal batch size."""
        self.optimal_batch_size = batch_size
        self.batch_processor.set_optimal_batch_size(batch_size)

    def start_evaluation_session(self):
        """Start a new evaluation session for timing."""
        self.stats['start_time'] = time.time()
        if self.verbose >= 2:
            logger.debug("Started evaluation session")

    def end_evaluation_session(self):
        """End the current evaluation session."""
        self.stats['end_time'] = time.time()
        if self.verbose >= 2:
            total_time = self.stats['end_time'] - self.stats['start_time']
            total_evaluations = self.stats['total_evaluations']
            if total_time > 0 and total_evaluations > 0:
                evals_per_sec = total_evaluations / total_time
                logger.debug(f"Ended evaluation session: {total_evaluations} evaluations "
                           f"in {total_time:.2f}s ({evals_per_sec:.1f} evals/sec)")

    def cleanup(self):
        """Clean up resources and stop background threads."""
        if hasattr(self.batch_processor, '_stop_background_thread'):
            self.batch_processor._stop_background_thread()
        if self.verbose >= 1:
            logger.info("BatchedEvaluator cleanup completed")

    def __del__(self):
        """Cleanup when the object is destroyed."""
        self.cleanup()

    def log_performance_summary(self):
        """Log a summary of evaluation performance."""
        stats = self.get_statistics()
        
        if self.verbose >= 1:
            logger.info("=== BatchedEvaluator Performance Summary ===")
            logger.info(f"Total evaluations: {stats['total_evaluations']}")
            logger.info(f"Cache hit rate: {stats['cache_hit_rate']:.1%}")
            logger.info(f"Average batch size: {stats['average_batch_size']:.1f}")
            logger.info(f"Inferences per second: {stats['inferences_per_second']:.1f}")
            
            if 'evaluations_per_second' in stats:
                logger.info(f"Evaluations per second: {stats['evaluations_per_second']:.1f}")
            
            logger.info(f"Queue size: {stats['queue_size']}")
            logger.info(f"Cache size: {stats['cache_size']}")
            logger.info("=============================================")
