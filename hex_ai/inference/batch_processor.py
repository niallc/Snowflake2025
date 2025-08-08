"""
Batch processor for efficient neural network inference in MCTS.

This module provides a BatchProcessor class that manages batched inference requests
for MCTS search, enabling significant performance improvements through GPU batching.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np

from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.utils.gpu_monitoring import get_gpu_memory_info, log_memory_status
from hex_ai.utils.perf import PERF

logger = logging.getLogger(__name__)


@dataclass
class BatchRequest:
    """Represents a single inference request in the batch."""
    board_state: np.ndarray  # The board state to evaluate
    callback: Callable[[np.ndarray, float], None]  # Callback to receive results
    metadata: Optional[Dict[str, Any]] = None  # Optional metadata for debugging


class BatchProcessor:
    """
    Efficient batch processor for MCTS inference.
    
    This class manages batched inference requests to minimize GPU kernel launches
    and maximize throughput. It provides a simple interface for requesting
    evaluations and processing them in optimal batch sizes.
    
    Key Features:
    - Automatic batch size optimization
    - Result caching to avoid redundant evaluations
    - Callback-based result distribution
    - Performance monitoring and statistics
    """
    
    def __init__(self, model: SimpleModelInference, optimal_batch_size: int = 64, verbose: int = 1, 
                 max_wait_ms: int = 5, enable_background_processing: bool = True):
        """
        Initialize the batch processor.
        
        Args:
            model: Model instance that supports batch_infer()
            optimal_batch_size: Target batch size for optimal GPU utilization
            verbose: Verbosity level (0=quiet, 1=normal, 2=detailed, 3=debug)
            max_wait_ms: Maximum wait time in milliseconds before processing small batches
            enable_background_processing: Whether to enable background thread for automatic processing
        """
        # TODO: PERFORMANCE - Optimize batch utilization and tensor allocation
        # Current batching may underfill batches, creating new tensors per evaluation
        # IMPLEMENTATION PLAN (Phase 3.3):
        # 1) Tune batch collection parameters: adjust max_wait_ms (1-5ms)
        # 2) Pre-seed rollouts to fill queue before first inference
        # 3) Pre-allocate input tensor pool, write into views when stacking
        # 4) Batch CPU→GPU transfers, avoid per-state .cpu() calls
        # 5) Monitor avg_batch_size vs target, aim for >80% utilization
        # 6) Add performance instrumentation using PERF utility
        # Expected gain: 1.5-3x speedup in inference throughput
        self.model = model
        self.optimal_batch_size = optimal_batch_size
        self.verbose = verbose
        self.max_wait_ms = max_wait_ms
        self.enable_background_processing = enable_background_processing
        
        # Request queue and cache
        self.request_queue: List[BatchRequest] = []
        self.result_cache: Dict[bytes, Tuple[np.ndarray, float]] = {}
        
        # Background processing
        self.background_thread = None
        self.shutdown_event = threading.Event()
        self.processing_lock = threading.Lock()
        
        # Rate-limited logging
        self.last_log_time = 0.0
        self.log_interval = 5.0  # Log every 5 seconds for rate-limited output
        
        # Performance statistics
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_batches_processed': 0,
            'total_inferences': 0,
            'total_time': 0.0,
            'average_batch_size': 0.0,
            'batch_processing_times': [],
            'gpu_memory_samples': []
        }
        
        logger.info(f"BatchProcessor initialized with optimal_batch_size={optimal_batch_size}, "
                   f"max_wait_ms={max_wait_ms}, background_processing={enable_background_processing}, verbose={verbose}")
        
        # Start background thread if enabled
        if self.enable_background_processing:
            self._start_background_thread()
        
        # Log initial memory status only for high verbosity
        if self.verbose >= 4:
            log_memory_status("Initial ")
    
    def request_evaluation(self, board_state: np.ndarray, 
                          callback: Callable[[np.ndarray, float], None],
                          metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Request an evaluation for a board state.
        
        Args:
            board_state: Board state to evaluate (numpy array or torch tensor)
            callback: Function to call with (policy, value) results
            metadata: Optional metadata for debugging
            
        Returns:
            True if result was immediately available from cache, False if queued
        """
        # Use lock if background processing is enabled
        if self.enable_background_processing:
            with self.processing_lock:
                return self._request_evaluation_internal(board_state, callback, metadata)
        else:
            return self._request_evaluation_internal(board_state, callback, metadata)
    
    def _request_evaluation_internal(self, board_state: np.ndarray, 
                                   callback: Callable[[np.ndarray, float], None],
                                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Internal evaluation request method (thread-safe when called with lock).
        
        Args:
            board_state: Board state to evaluate (numpy array or torch tensor)
            callback: Function to call with (policy, value) results
            metadata: Optional metadata for debugging
            
        Returns:
            True if result was immediately available from cache, False if queued
        """
        # Convert to numpy array for cache key creation
        # We expect either PyTorch tensors or numpy arrays, so be explicit about what we accept
        if hasattr(board_state, 'cpu'):  # PyTorch tensor
            board_array = board_state.cpu().numpy()
            logger.debug(f"Converted PyTorch tensor to numpy array, shape: {board_array.shape}")
        elif isinstance(board_state, np.ndarray):  # Already numpy array
            board_array = board_state
        else:
            raise ValueError(f"Expected PyTorch tensor or numpy array, got {type(board_state)}")
            
        cache_key = board_array.tobytes()
        
        # Check cache first
        if cache_key in self.result_cache:
            self.stats['cache_hits'] += 1
            policy, value = self.result_cache[cache_key]
            callback(policy, value)
            return True
        
        # Cache miss - add to request queue
        self.stats['cache_misses'] += 1
        self.stats['total_requests'] += 1
        
        request = BatchRequest(
            board_state=board_state,
            callback=callback,
            metadata=metadata
        )
        self.request_queue.append(request)
        
        if self.verbose >= 2:
            logger.debug(f"Added request to queue. Queue size: {len(self.request_queue)}, "
                        f"Cache size: {len(self.result_cache)}")
        elif self.verbose >= 4:
            logger.debug(f"Added request to queue. Queue size: {len(self.request_queue)}, "
                        f"Cache size: {len(self.result_cache)}, "
                        f"Metadata: {metadata}")
        
        return False
    
    def _start_background_thread(self):
        """Start the background processing thread."""
        if self.background_thread is not None:
            return  # Already running
        
        self.shutdown_event.clear()
        self.background_thread = threading.Thread(target=self._background_worker, daemon=True)
        self.background_thread.start()
        
        if self.verbose >= 2:
            logger.debug("Background processing thread started")
    
    def _stop_background_thread(self):
        """Stop the background processing thread."""
        if self.background_thread is None:
            return
        
        self.shutdown_event.set()
        self.background_thread.join(timeout=1.0)
        self.background_thread = None
        
        if self.verbose >= 2:
            logger.debug("Background processing thread stopped")
    
    def _background_worker(self):
        """Background worker that processes batches automatically."""
        while not self.shutdown_event.is_set():
            try:
                with self.processing_lock:
                    queue_size = len(self.request_queue)
                    
                    # Process if we have enough requests or if we've been waiting too long
                    should_process = (
                        queue_size >= self.optimal_batch_size or
                        (queue_size > 0 and self._should_process_small_batch())
                    )
                    
                    if should_process:
                        if self.verbose >= 2:
                            logger.debug(f"Background worker processing batch of {queue_size} requests")
                        result = self._process_batch_internal(force=queue_size < self.optimal_batch_size)
                        if self.verbose >= 2:
                            logger.debug(f"Background worker processed {result} requests")
                
                # Sleep for a short time to avoid busy waiting
                time.sleep(self.max_wait_ms / 1000.0)
                
            except Exception as e:
                logger.error(f"Background worker error: {e}")
                time.sleep(0.001)  # Brief pause on error
    
    def _should_process_small_batch(self) -> bool:
        """Check if we should process a small batch based on timing."""
        # Simple implementation: process if we have any requests
        # This could be enhanced with more sophisticated timing logic
        return len(self.request_queue) > 0
    
    def __del__(self):
        """Cleanup when the object is destroyed."""
        self._stop_background_thread()
    
    def process_batch(self, force: bool = False) -> int:
        """
        Process the current batch of requests.
        
        Args:
            force: If True, process even if batch is smaller than optimal size
            
        Returns:
            Number of requests processed
        """
        # Use lock if background processing is enabled
        if self.enable_background_processing:
            with self.processing_lock:
                return self._process_batch_internal(force)
        else:
            return self._process_batch_internal(force)
    
    def _process_batch_internal(self, force: bool = False) -> int:
        """
        Internal batch processing method (thread-safe when called with lock).
        
        Args:
            force: If True, process even if batch is smaller than optimal size
            
        Returns:
            Number of requests processed
        """
        if not self.request_queue:
            return 0
        
        # Determine batch size
        batch_size = len(self.request_queue)
        if not force and batch_size < self.optimal_batch_size:
            if self.verbose >= 3:
                logger.debug(f"Batch too small ({batch_size} < {self.optimal_batch_size}), waiting for more requests")
            return 0  # Wait for more requests
        
        start_time = time.time()
        
        # Rate-limited logging for batch processing
        current_time = time.time()
        should_log = (current_time - self.last_log_time) >= self.log_interval
        
        if self.verbose >= 1 and should_log:
            logger.info(f"Processing batch of {batch_size} requests (optimal: {self.optimal_batch_size})")
            self.last_log_time = current_time
        
        # Detailed logging only for very high verbosity
        if self.verbose >= 5:
            log_memory_status("Pre-batch ")
        
        # Extract boards and callbacks
        boards = [req.board_state for req in self.request_queue]
        callbacks = [req.callback for req in self.request_queue]
        metadata_list = [req.metadata for req in self.request_queue]
        
        # Process batch with performance monitoring
        with PERF.timer("nn.infer"):
            try:
                if self.verbose >= 2:
                    logger.debug(f"Calling model.batch_infer with {len(boards)} boards")
                policies, values = self.model.batch_infer(boards)
                
                if self.verbose >= 2:
                    logger.debug(f"Model returned {len(policies)} policies and {len(values)} values")
                
                # Validate results
                if len(policies) != len(values) or len(policies) != len(boards):
                    raise RuntimeError(f"Batch inference returned {len(policies)} policies, {len(values)} values for {len(boards)} boards")
                
                # Distribute results and update cache
                for i, (board, policy, value, callback) in enumerate(zip(boards, policies, values, callbacks)):
                    # Store in cache
                    cache_key = board.tobytes()
                    self.result_cache[cache_key] = (policy, value)
                    
                    # Call callback
                    try:
                        if self.verbose >= 2:
                            logger.debug(f"Calling callback {i} with policy shape {policy.shape} and value {value:.4f}")
                        callback(policy, value)
                        if self.verbose >= 2:
                            logger.debug(f"Callback {i} completed successfully")
                    except Exception as e:
                        logger.error(f"Callback error for request {i}: {e}")
                        import traceback
                        logger.error(f"Callback traceback: {traceback.format_exc()}")
                        if metadata_list[i]:
                            logger.error(f"Request metadata: {metadata_list[i]}")
                
                # Update performance statistics
                PERF.inc("nn.batch")
                PERF.add_sample("nn.batch_size", len(boards))
                
            except Exception as e:
                logger.error(f"Batch inference failed: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
                # Clear the queue on error to prevent infinite retries
                processed_count = 0
                return processed_count
        
        # Update statistics
        batch_time = time.time() - start_time
        self.stats['total_batches_processed'] += 1
        self.stats['total_inferences'] += len(boards)
        self.stats['total_time'] += batch_time
        self.stats['batch_processing_times'].append(batch_time)
        
        # Update average batch size
        total_batches = self.stats['total_batches_processed']
        current_avg = self.stats['average_batch_size']
        self.stats['average_batch_size'] = (current_avg * (total_batches - 1) + len(boards)) / total_batches
        
        # Record GPU memory usage
        gpu_mem = get_gpu_memory_info()
        if gpu_mem:
            self.stats['gpu_memory_samples'].append({
                'timestamp': time.time(),
                'allocated_mb': gpu_mem['allocated_mb'],
                'utilization_percent': gpu_mem['utilization_percent']
            })
        
        processed_count = len(boards)
        
        # Rate-limited logging for batch completion
        if self.verbose >= 1 and should_log:
            logger.info(f"Batch processed: {processed_count} requests in {batch_time:.3f}s "
                       f"({processed_count/batch_time:.1f} req/s)")
        
        # Detailed logging only for very high verbosity
        if self.verbose >= 5:
            log_memory_status("Post-batch ")
        
        # Clear the queue
        self.request_queue.clear()
        
        return processed_count
    
    def process_all(self) -> int:
        """
        Process all pending requests, regardless of batch size.
        
        Returns:
            Total number of requests processed
        """
        return self.process_batch(force=True)
    
    def get_queue_size(self) -> int:
        """Get the current number of pending requests."""
        return len(self.request_queue)
    
    def get_cache_size(self) -> int:
        """Get the current number of cached results."""
        return len(self.result_cache)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.stats.copy()
        
        # Calculate derived statistics
        if stats['total_requests'] > 0:
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['total_requests']
        else:
            stats['cache_hit_rate'] = 0.0
        
        if stats['total_time'] > 0:
            stats['inferences_per_second'] = stats['total_inferences'] / stats['total_time']
        else:
            stats['inferences_per_second'] = 0.0
        
        if stats['total_batches_processed'] > 0:
            stats['average_batch_size'] = stats['average_batch_size']
        else:
            stats['average_batch_size'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset performance statistics."""
        self.stats = {
            'total_requests': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'total_batches_processed': 0,
            'total_inferences': 0,
            'total_time': 0.0,
            'average_batch_size': 0.0,
            'batch_processing_times': [],
            'gpu_memory_samples': []
        }
    
    def clear_cache(self):
        """Clear the result cache."""
        self.result_cache.clear()
        logger.info("BatchProcessor cache cleared")
    
    def set_optimal_batch_size(self, batch_size: int):
        """Update the optimal batch size."""
        self.optimal_batch_size = batch_size
        logger.info(f"Optimal batch size updated to {batch_size}")
