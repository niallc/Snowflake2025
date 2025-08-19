"""
Profiling utilities for performance analysis.
"""

import time
import functools
import logging
from contextlib import contextmanager
from typing import Dict, Any, Optional, Callable
from collections import defaultdict

logger = logging.getLogger(__name__)

class Profiler:
    """Simple profiler for tracking function calls and timing."""
    
    def __init__(self):
        self.call_counts = defaultdict(int)
        self.total_times = defaultdict(float)
        self.min_times = defaultdict(lambda: float('inf'))
        self.max_times = defaultdict(lambda: 0.0)
        self.current_timers = {}
    
    def start_timer(self, name: str):
        """Start timing a named operation."""
        self.current_timers[name] = time.perf_counter()
    
    def end_timer(self, name: str):
        """End timing a named operation and record statistics."""
        if name in self.current_timers:
            duration = time.perf_counter() - self.current_timers[name]
            self.call_counts[name] += 1
            self.total_times[name] += duration
            self.min_times[name] = min(self.min_times[name], duration)
            self.max_times[name] = max(self.max_times[name], duration)
            del self.current_timers[name]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get profiling statistics."""
        stats = {}
        for name in self.call_counts:
            count = self.call_counts[name]
            total = self.total_times[name]
            avg = total / count if count > 0 else 0.0
            stats[name] = {
                'calls': count,
                'total_time': total,
                'avg_time': avg,
                'min_time': self.min_times[name] if self.min_times[name] != float('inf') else 0.0,
                'max_time': self.max_times[name],
                'total_ms': total * 1000.0,
                'avg_ms': avg * 1000.0,
                'min_ms': self.min_times[name] * 1000.0 if self.min_times[name] != float('inf') else 0.0,
                'max_ms': self.max_times[name] * 1000.0,
            }
        return stats
    
    def print_stats(self):
        """Print profiling statistics to logger."""
        stats = self.get_stats()
        if not stats:
            logger.info("No profiling data collected.")
            return
        
        logger.info("=== PROFILING STATISTICS ===")
        for name, data in sorted(stats.items(), key=lambda x: x[1]['total_time'], reverse=True):
            logger.info(f"{name}:")
            logger.info(f"  Calls: {data['calls']}")
            logger.info(f"  Total: {data['total_ms']:.2f}ms")
            logger.info(f"  Avg: {data['avg_ms']:.2f}ms")
            logger.info(f"  Min: {data['min_ms']:.2f}ms")
            logger.info(f"  Max: {data['max_ms']:.2f}ms")
        logger.info("=== END PROFILING ===")
    
    def reset(self):
        """Reset all profiling data."""
        self.call_counts.clear()
        self.total_times.clear()
        self.min_times.clear()
        self.max_times.clear()
        self.current_timers.clear()

# Global profiler instance
global_profiler = Profiler()

def profile_function(name: Optional[str] = None):
    """Decorator to profile a function."""
    def decorator(func: Callable) -> Callable:
        func_name = name or func.__name__
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            global_profiler.start_timer(func_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                global_profiler.end_timer(func_name)
        
        return wrapper
    return decorator

@contextmanager
def profile_operation(name: str):
    """Context manager for profiling operations."""
    global_profiler.start_timer(name)
    try:
        yield
    finally:
        global_profiler.end_timer(name)

def get_profiling_stats() -> Dict[str, Any]:
    """Get current profiling statistics."""
    return global_profiler.get_stats()

def print_profiling_stats():
    """Print current profiling statistics."""
    global_profiler.print_stats()

def reset_profiling():
    """Reset profiling data."""
    global_profiler.reset()
