"""
Timing utilities for performance measurement.

This module contains utilities for tracking timing information
and performance metrics across different operations.
"""

import time
import math
import numpy as np
from typing import Dict, Any, List


class MCTSTimingTracker:
    """
    Tracks timing information for MCTS operations.
    
    This class provides a centralized way to track timing for various
    MCTS operations including neural network inference, tree traversal,
    and other performance-critical operations.
    """
    
    def __init__(self):
        self.timings = {}
        self.batch_count = 0
        self.batch_sizes = []
        self.forward_ms_list = []
        self.select_times = []
        self.cache_hit_times = []
        self.cache_miss_times = []
        self.puct_calc_times = []
        self.make_move_times = []
        
        # Cumulative totals
        self.h2d_ms_total = 0.0
        self.forward_ms_total = 0.0
        self.pure_forward_ms_total = 0.0
        self.sync_ms_total = 0.0
        self.d2h_ms_total = 0.0
        
        # Current timing
        self.current_timing = None
        self.current_timing_start = None
    
    def start_timing(self, operation: str) -> None:
        """
        Start timing an operation.
        
        Args:
            operation: Name of the operation to time
        """
        self.current_timing = operation
        self.current_timing_start = time.perf_counter()
    
    def end_timing(self, operation: str) -> None:
        """
        End timing an operation.
        
        Args:
            operation: Name of the operation to end timing for
        """
        if self.current_timing == operation and self.current_timing_start is not None:
            duration_ms = (time.perf_counter() - self.current_timing_start) * 1000.0
            if operation not in self.timings:
                self.timings[operation] = 0.0
            self.timings[operation] += duration_ms
            self.current_timing = None
            self.current_timing_start = None
    
    def record_batch_metrics(self, tm: Dict[str, Any]) -> None:
        """
        Record metrics from a neural network batch.
        
        Args:
            tm: Timing metrics dictionary from neural network inference
        """
        self.batch_count += 1
        self.batch_sizes.append(int(tm["batch_size"]))
        self.forward_ms_list.append(float(tm["forward_ms"]))
        self.h2d_ms_total += float(tm["h2d_ms"])
        self.forward_ms_total += float(tm["forward_ms"])
        self.pure_forward_ms_total += float(tm.get("pure_forward_ms", tm["forward_ms"]))
        self.sync_ms_total += float(tm.get("sync_ms", 0.0))
        self.d2h_ms_total += float(tm["d2h_ms"])
    
    def get_final_stats(self) -> Dict[str, Any]:
        """
        Get final timing statistics.
        
        Returns:
            Dictionary containing comprehensive timing statistics
        """
        total_search_time = (
            self.timings.get("encode", 0.0) + self.timings.get("stack", 0.0) + 
            self.h2d_ms_total + self.forward_ms_total + self.d2h_ms_total + 
            self.timings.get("expand", 0.0) + self.timings.get("backprop", 0.0) + 
            self.timings.get("select", 0.0) + self.timings.get("cache_lookup", 0.0) + 
            self.timings.get("state_creation", 0.0)
        ) / 1000.0
        
        return {
            "encode_ms": self.timings.get("encode", 0.0),
            "stack_ms": self.timings.get("stack", 0.0),
            "h2d_ms": self.h2d_ms_total,
            "forward_ms": self.forward_ms_total,
            "pure_forward_ms": self.pure_forward_ms_total,
            "sync_ms": self.sync_ms_total,
            "d2h_ms": self.d2h_ms_total,
            "expand_ms": self.timings.get("expand", 0.0),
            "backprop_ms": self.timings.get("backprop", 0.0),
            "select_ms": self.timings.get("select", 0.0),
            "cache_lookup_ms": self.timings.get("cache_lookup", 0.0),
            "state_creation_ms": self.timings.get("state_creation", 0.0),
            "puct_calc_ms": self.timings.get("puct_calc", 0.0),
            "make_move_ms": self.timings.get("make_move", 0.0),
            "batch_count": self.batch_count,
            "batch_sizes": self.batch_sizes,
            "forward_ms_list": self.forward_ms_list,
            "select_times": self.select_times,
            "cache_hit_times": self.cache_hit_times,
            "cache_miss_times": self.cache_miss_times,
            "puct_calc_times": self.puct_calc_times,
            "make_move_times": self.make_move_times,
            "median_forward_ms_ex_warm": _median_excluding_first(self.forward_ms_list),
            "p90_forward_ms_ex_warm": _p90_excluding_first(self.forward_ms_list),
            "median_select_ms": _median_excluding_first(self.select_times) if self.select_times else 0.0,
            "median_cache_hit_ms": _median_excluding_first(self.cache_hit_times) if self.cache_hit_times else 0.0,
            "median_cache_miss_ms": _median_excluding_first(self.cache_miss_times) if self.cache_miss_times else 0.0,
            "median_puct_calc_ms": _median_excluding_first(self.puct_calc_times) if self.puct_calc_times else 0.0,
            "median_make_move_ms": _median_excluding_first(self.make_move_times) if self.make_move_times else 0.0,
            "total_search_time": total_search_time,
        }


def _median_excluding_first(xs: List[float]) -> float:
    """
    Calculate median of a list excluding the first element.
    
    Args:
        xs: List of float values
        
    Returns:
        Median value excluding the first element
    """
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    arr = np.array(xs[1:], dtype=np.float64)
    return float(np.median(arr))


def _p90_excluding_first(xs: List[float]) -> float:
    """
    Calculate 90th percentile of a list excluding the first element.
    
    Args:
        xs: List of float values
        
    Returns:
        90th percentile value excluding the first element
    """
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    arr = np.array(xs[1:], dtype=np.float64)
    k = max(0, int(math.ceil(0.9 * len(arr)) - 1))
    arr.sort()
    return float(arr[k])
