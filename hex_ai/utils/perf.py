"""
Performance profiling utilities for MCTS optimization.

This module provides a centralized performance monitoring system that can be used
across the MCTS codebase to identify bottlenecks and measure optimization gains.
"""

import time
import collections
import contextlib
import threading
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class Perf:
    """
    Thread-safe performance monitoring class.
    
    Provides timing, counters, and sampling capabilities for profiling
    MCTS performance bottlenecks.
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        self.counters = collections.Counter()
        self.timings = collections.defaultdict(float)   # key -> total seconds
        self.samples = collections.defaultdict(list)    # key -> list of floats (for histograms)
        self.meta = {}  # free-form metadata
        
        # Track the last snapshot for rate limiting
        self.last_snapshot_time = 0.0
        self.snapshot_interval = 1.0  # Minimum seconds between snapshots
    
    @contextlib.contextmanager
    def timer(self, key: str):
        """
        Context manager for timing code blocks.
        
        Args:
            key: Identifier for this timing measurement
        """
        t0 = time.perf_counter()
        try:
            yield
        finally:
            dt = time.perf_counter() - t0
            with self.lock:
                self.timings[key] += dt
                self.samples[key].append(dt)
    
    def inc(self, key: str, n: int = 1):
        """
        Increment a counter.
        
        Args:
            key: Counter identifier
            n: Amount to increment (default: 1)
        """
        with self.lock:
            self.counters[key] += n
    
    def add_sample(self, key: str, value: float):
        """
        Add a sample value for statistical analysis.
        
        Args:
            key: Sample identifier
            value: Sample value
        """
        with self.lock:
            self.samples[key].append(value)
    
    def set_meta(self, key: str, value: Any):
        """
        Set metadata value.
        
        Args:
            key: Metadata key
            value: Metadata value
        """
        with self.lock:
            self.meta[key] = value
    
    def snapshot(self, clear: bool = False, force: bool = False) -> Dict[str, Any]:
        """
        Get a snapshot of current performance data.
        
        Args:
            clear: Whether to clear data after snapshot
            force: Whether to force snapshot even if rate limited
            
        Returns:
            Dictionary containing performance data
        """
        current_time = time.time()
        
        # Rate limit snapshots unless forced
        if not force and current_time - self.last_snapshot_time < self.snapshot_interval:
            return {}
        
        with self.lock:
            out = {
                "counters": dict(self.counters),
                "timings_s": {k: round(v, 6) for k, v in self.timings.items()},
                "samples": {k: (len(v), round(sum(v), 6)) for k, v in self.samples.items()},
                "meta": dict(self.meta),
            }
            
            if clear:
                self.counters.clear()
                self.timings.clear()
                self.samples.clear()
            
            self.last_snapshot_time = current_time
            return out
    
    def log_snapshot(self, clear: bool = True, force: bool = False):
        """
        Log a performance snapshot as JSON.
        
        Args:
            clear: Whether to clear data after logging
            force: Whether to force logging even if rate limited
        """
        snapshot = self.snapshot(clear=clear, force=force)
        if snapshot:
            logger.info(f"PERF: {json.dumps(snapshot, sort_keys=True)}")
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics for all measurements.
        
        Returns:
            Dictionary with summary statistics
        """
        with self.lock:
            summary = {}
            
            # Counter summaries
            for key, count in self.counters.items():
                summary[f"counter_{key}"] = count
            
            # Timing summaries
            for key, total_time in self.timings.items():
                samples = self.samples[key]
                if samples:
                    summary[f"timing_{key}_total"] = round(total_time, 6)
                    summary[f"timing_{key}_count"] = len(samples)
                    summary[f"timing_{key}_avg"] = round(total_time / len(samples), 6)
                    summary[f"timing_{key}_min"] = round(min(samples), 6)
                    summary[f"timing_{key}_max"] = round(max(samples), 6)
            
            # Sample summaries
            for key, samples_list in self.samples.items():
                if samples_list and key not in self.timings:  # Don't double-count timing samples
                    summary[f"sample_{key}_count"] = len(samples_list)
                    summary[f"sample_{key}_sum"] = round(sum(samples_list), 6)
                    summary[f"sample_{key}_avg"] = round(sum(samples_list) / len(samples_list), 6)
                    summary[f"sample_{key}_min"] = round(min(samples_list), 6)
                    summary[f"sample_{key}_max"] = round(max(samples_list), 6)
            
            return summary


# Global performance instance
PERF = Perf()


def log_performance_summary(clear: bool = True):
    """
    Log a comprehensive performance summary.
    
    Args:
        clear: Whether to clear data after logging
    """
    summary = PERF.get_summary_stats()
    if summary:
        logger.info(f"PERF_SUMMARY: {json.dumps(summary, sort_keys=True)}")
        if clear:
            PERF.snapshot(clear=True)


def setup_model_performance_meta(model):
    """
    Set up performance metadata for a model.
    
    Args:
        model: PyTorch model to analyze
    """
    import torch
    
    # Device and dtype info
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    
    PERF.set_meta("device", str(device))
    PERF.set_meta("dtype", str(dtype))
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    PERF.set_meta("model_params", total_params)
    
    # Ensure model is in eval mode
    model.eval()
    PERF.set_meta("model_mode", "eval")
    
    logger.info(f"PERF: Model setup - device={device}, dtype={dtype}, params={total_params}")
