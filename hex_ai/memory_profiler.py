"""
Simple memory profiler for tracking RSS vs Python heap and identifying memory growth.

This module provides lightweight memory profiling with minimal overhead.
Designed to be easily removable once memory issues are resolved.
"""

import tracemalloc
import psutil
import threading
import time
import csv
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MemoryProfiler:
    """
    Simple memory profiler that tracks RSS vs Python heap and takes periodic snapshots.
    
    Features:
    1. RSS vs Heap Timeline - tracks both metrics over time
    2. Periodic Snapshot Deltas - compares snapshots to find growing allocations
    3. Simple Type Census - shows top allocators by file/line
    """
    
    def __init__(self, output_dir: str = "temp/memoryProfile", interval_seconds: int = 60):
        """
        Initialize memory profiler.
        
        Args:
            output_dir: Directory to write profiling output files
            interval_seconds: How often to log RSS/heap measurements (default: 60s)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval_seconds = interval_seconds
        
        # Generate timestamp for this profiling session
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # State
        self.is_running = False
        self.logging_thread: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        
        # Timeline tracking
        self.timeline_file = self.output_dir / f"memory_timeline_{self.session_timestamp}.txt"
        self.timeline_data: List[Dict] = []
        
        # Snapshot tracking
        self.snapshots: List[tuple] = []  # List of (label, snapshot, timestamp)
        self.previous_snapshot: Optional[tracemalloc.Snapshot] = None
        
        # Process reference
        self.process = psutil.Process()
        
        logger.info(f"Memory profiler initialized. Output directory: {self.output_dir}")
    
    def start(self):
        """Start memory profiling."""
        if self.is_running:
            logger.warning("Memory profiler is already running")
            return
        
        # Start tracemalloc
        tracemalloc.start()
        self.is_running = True
        self.stop_event.clear()
        
        # Write CSV header
        with open(self.timeline_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'rss_gb', 'heap_mb', 'heap_peak_mb', 'gpu_mb', 'label'])
        
        # Start background thread for periodic logging
        self.logging_thread = threading.Thread(target=self._periodic_logging, daemon=True)
        self.logging_thread.start()
        
        logger.info("Memory profiling started")
    
    def stop(self):
        """Stop memory profiling and write final reports."""
        if not self.is_running:
            return
        
        self.is_running = False
        self.stop_event.set()
        
        # Wait for logging thread to finish
        if self.logging_thread and self.logging_thread.is_alive():
            self.logging_thread.join(timeout=5.0)
        
        # Write final timeline data
        self._write_timeline()
        
        # Generate final summary
        self._write_summary()
        
        # Stop tracemalloc
        tracemalloc.stop()
        
        logger.info(f"Memory profiling stopped. Results written to {self.output_dir}")
    
    def _get_gpu_memory_mb(self) -> Optional[float]:
        """Get GPU memory usage in MB if available."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 2)
            elif torch.backends.mps.is_available():
                # MPS doesn't expose memory stats, return None
                return None
        except ImportError:
            pass
        return None
    
    def log_measurement(self, label: Optional[str] = None):
        """
        Log a single RSS/heap measurement.
        
        Args:
            label: Optional label for this measurement (e.g., "epoch_start")
        """
        if not self.is_running:
            return
        
        timestamp = time.time()
        rss_gb = self.process.memory_info().rss / (1024 ** 3)
        heap_current, heap_peak = tracemalloc.get_traced_memory()
        heap_mb = heap_current / (1024 ** 2)
        heap_peak_mb = heap_peak / (1024 ** 2)
        gpu_mb = self._get_gpu_memory_mb()
        
        entry = {
            'timestamp': timestamp,
            'rss_gb': rss_gb,
            'heap_mb': heap_mb,
            'heap_peak_mb': heap_peak_mb,
            'gpu_mb': gpu_mb,
            'label': label
        }
        
        self.timeline_data.append(entry)
        
        # Also write immediately to file (for real-time monitoring)
        with open(self.timeline_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.fromtimestamp(timestamp).isoformat(),
                f"{rss_gb:.4f}",
                f"{heap_mb:.2f}",
                f"{heap_peak_mb:.2f}",
                f"{gpu_mb:.2f}" if gpu_mb is not None else "",
                label or ""
            ])
    
    def take_snapshot(self, label: str):
        """
        Take a memory snapshot at a key point.
        
        Args:
            label: Label for this snapshot (e.g., "pipeline_start", "epoch_1_end")
        """
        if not self.is_running:
            logger.warning("Cannot take snapshot: memory profiler not running")
            return
        
        snapshot = tracemalloc.take_snapshot()
        timestamp = time.time()
        
        self.snapshots.append((label, snapshot, timestamp))
        
        # Compare with previous snapshot if available
        if self.previous_snapshot is not None:
            self._compare_snapshots(self.previous_snapshot, snapshot, label)
        
        self.previous_snapshot = snapshot
        
        logger.info(f"Memory snapshot taken: {label}")
    
    def _periodic_logging(self):
        """Background thread that logs RSS/heap measurements periodically."""
        while not self.stop_event.is_set():
            self.log_measurement()
            # Wait for interval, but check stop_event frequently
            self.stop_event.wait(self.interval_seconds)
    
    def _write_timeline(self):
        """Write final timeline data to file."""
        # Timeline is already being written incrementally, so this is just for completeness
        pass
    
    def _compare_snapshots(self, old_snapshot: tracemalloc.Snapshot, 
                          new_snapshot: tracemalloc.Snapshot, label: str):
        """
        Compare two snapshots and write delta report.
        
        Args:
            old_snapshot: Previous snapshot
            new_snapshot: Current snapshot
            label: Label for this comparison
        """
        top_stats = new_snapshot.compare_to(old_snapshot, 'lineno')
        
        # Write delta report
        delta_file = self.output_dir / f"snapshot_delta_{self.session_timestamp}_{label}.txt"
        with open(delta_file, 'w') as f:
            f.write(f"Memory Growth Analysis: {label}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write("=" * 80 + "\n\n")
            f.write("Top 20 growing allocations:\n")
            f.write("-" * 80 + "\n")
            
            for index, stat in enumerate(top_stats[:20], 1):
                f.write(f"{index}. {stat}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"Total memory growth: {sum(stat.size_diff for stat in top_stats) / (1024**2):.2f} MB\n")
            f.write(f"Number of growing allocations: {len([s for s in top_stats if s.size_diff > 0])}\n")
    
    def write_epoch_summary(self, epoch: int):
        """
        Write summary report for a specific epoch.
        
        Args:
            epoch: Epoch number
        """
        if not self.snapshots:
            return
        
        summary_file = self.output_dir / f"summary_{self.session_timestamp}_epoch_{epoch}.txt"
        
        with open(summary_file, 'w') as f:
            f.write(f"Memory Profiling Summary - Epoch {epoch}\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Session timestamp: {self.session_timestamp}\n")
            f.write(f"Epoch: {epoch}\n")
            f.write(f"Total snapshots up to this point: {len(self.snapshots)}\n")
            f.write(f"Timeline entries up to this point: {len(self.timeline_data)}\n\n")
            
            # Write snapshot timeline up to this epoch
            f.write("Snapshot Timeline:\n")
            f.write("-" * 80 + "\n")
            for label, snapshot, timestamp in self.snapshots:
                # Get heap stats from the snapshot itself
                stats = snapshot.statistics('traceback')
                total_size = sum(stat.size for stat in stats)
                f.write(f"{datetime.fromtimestamp(timestamp).isoformat()}: {label}\n")
                f.write(f"  Heap: {total_size / (1024**2):.2f} MB\n")
            
            # Write RSS/heap stats up to this point
            if self.timeline_data:
                f.write("\nRSS/Heap/GPU Statistics:\n")
                f.write("-" * 80 + "\n")
                rss_values = [d['rss_gb'] for d in self.timeline_data]
                heap_values = [d['heap_mb'] for d in self.timeline_data]
                gpu_values = [d['gpu_mb'] for d in self.timeline_data if d['gpu_mb'] is not None]
                
                f.write(f"RSS - Min: {min(rss_values):.2f} GB, Max: {max(rss_values):.2f} GB, "
                       f"Current: {rss_values[-1]:.2f} GB\n")
                f.write(f"Heap - Min: {min(heap_values):.2f} MB, Max: {max(heap_values):.2f} MB, "
                       f"Current: {heap_values[-1]:.2f} MB\n")
                if gpu_values:
                    f.write(f"GPU - Min: {min(gpu_values):.2f} MB, Max: {max(gpu_values):.2f} MB, "
                           f"Current: {gpu_values[-1]:.2f} MB\n")
                
                # Calculate growth from start
                if len(rss_values) > 1:
                    rss_growth = rss_values[-1] - rss_values[0]
                    heap_growth = heap_values[-1] - heap_values[0]
                    f.write(f"\nGrowth since start:\n")
                    f.write(f"RSS: {rss_growth:+.2f} GB\n")
                    f.write(f"Heap: {heap_growth:+.2f} MB\n")
                    if gpu_values and len(gpu_values) > 1:
                        gpu_growth = gpu_values[-1] - gpu_values[0]
                        f.write(f"GPU: {gpu_growth:+.2f} MB\n")
                
                # Calculate epoch-to-epoch growth if we have previous epoch data
                if epoch > 1 and hasattr(self, '_last_epoch_rss'):
                    epoch_rss_growth = rss_values[-1] - self._last_epoch_rss
                    epoch_heap_growth = heap_values[-1] - self._last_epoch_heap
                    f.write(f"\nGrowth since previous epoch:\n")
                    f.write(f"RSS: {epoch_rss_growth:+.2f} GB\n")
                    f.write(f"Heap: {epoch_heap_growth:+.2f} MB\n")
                    if gpu_values and hasattr(self, '_last_epoch_gpu') and self._last_epoch_gpu is not None:
                        epoch_gpu_growth = gpu_values[-1] - self._last_epoch_gpu
                        f.write(f"GPU: {epoch_gpu_growth:+.2f} MB\n")
                
                # Store current values for next epoch comparison
                self._last_epoch_rss = rss_values[-1]
                self._last_epoch_heap = heap_values[-1]
                if gpu_values:
                    self._last_epoch_gpu = gpu_values[-1]
                else:
                    self._last_epoch_gpu = None
        
        # Write type census from the latest snapshot
        if self.snapshots:
            self._write_type_census(self.snapshots[-1][1], epoch=epoch)
    
    def _write_summary(self):
        """Write final summary report."""
        if not self.snapshots:
            return
        
        summary_file = self.output_dir / f"summary_{self.session_timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("Memory Profiling Summary (Final)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Session timestamp: {self.session_timestamp}\n")
            f.write(f"Total snapshots: {len(self.snapshots)}\n")
            f.write(f"Timeline entries: {len(self.timeline_data)}\n\n")
            
            # Write snapshot timeline
            f.write("Snapshot Timeline:\n")
            f.write("-" * 80 + "\n")
            for label, snapshot, timestamp in self.snapshots:
                stats = snapshot.statistics('traceback')
                total_size = sum(stat.size for stat in stats)
                f.write(f"{datetime.fromtimestamp(timestamp).isoformat()}: {label}\n")
                f.write(f"  Heap: {total_size / (1024**2):.2f} MB\n")
            
            # Write final RSS/heap stats
            if self.timeline_data:
                f.write("\nRSS/Heap/GPU Statistics:\n")
                f.write("-" * 80 + "\n")
                rss_values = [d['rss_gb'] for d in self.timeline_data]
                heap_values = [d['heap_mb'] for d in self.timeline_data]
                gpu_values = [d['gpu_mb'] for d in self.timeline_data if d['gpu_mb'] is not None]
                
                f.write(f"RSS - Min: {min(rss_values):.2f} GB, Max: {max(rss_values):.2f} GB, "
                       f"Final: {rss_values[-1]:.2f} GB\n")
                f.write(f"Heap - Min: {min(heap_values):.2f} MB, Max: {max(heap_values):.2f} MB, "
                       f"Final: {heap_values[-1]:.2f} MB\n")
                if gpu_values:
                    f.write(f"GPU - Min: {min(gpu_values):.2f} MB, Max: {max(gpu_values):.2f} MB, "
                           f"Final: {gpu_values[-1]:.2f} MB\n")
                
                # Calculate growth
                if len(rss_values) > 1:
                    rss_growth = rss_values[-1] - rss_values[0]
                    heap_growth = heap_values[-1] - heap_values[0]
                    f.write(f"\nGrowth:\n")
                    f.write(f"RSS: {rss_growth:+.2f} GB\n")
                    f.write(f"Heap: {heap_growth:+.2f} MB\n")
                    if gpu_values and len(gpu_values) > 1:
                        gpu_growth = gpu_values[-1] - gpu_values[0]
                        f.write(f"GPU: {gpu_growth:+.2f} MB\n")
        
        # Write type census from final snapshot
        if self.snapshots:
            self._write_type_census(self.snapshots[-1][1])
    
    def _write_type_census(self, snapshot: tracemalloc.Snapshot, epoch: Optional[int] = None):
        """
        Write type census showing top allocators.
        
        Args:
            snapshot: Snapshot to analyze
            epoch: Optional epoch number for filename suffix
        """
        top_stats = snapshot.statistics('filename')
        
        if epoch is not None:
            census_file = self.output_dir / f"type_census_{self.session_timestamp}_epoch_{epoch}.txt"
        else:
            census_file = self.output_dir / f"type_census_{self.session_timestamp}.txt"
        
        with open(census_file, 'w') as f:
            if epoch is not None:
                f.write(f"Memory Allocation by File - Epoch {epoch}\n")
            else:
                f.write("Memory Allocation by File (Final)\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            stats = snapshot.statistics('traceback')
            total_size = sum(stat.size for stat in stats)
            f.write(f"Total traced memory: {total_size / (1024**2):.2f} MB\n\n")
            f.write("Top 30 files by memory allocation:\n")
            f.write("-" * 80 + "\n")
            
            for index, stat in enumerate(top_stats[:30], 1):
                size_mb = stat.size / (1024 ** 2)
                count = stat.count
                f.write(f"{index}. {stat.traceback.format()[0]}\n")
                f.write(f"   Size: {size_mb:.2f} MB ({stat.size:,} bytes), Count: {count:,}\n\n")


# Global profiler instance (None when not enabled)
_global_profiler: Optional[MemoryProfiler] = None


def get_profiler() -> Optional[MemoryProfiler]:
    """Get the global memory profiler instance, or None if not enabled."""
    return _global_profiler


def start_profiling(output_dir: str = "temp/memoryProfile", interval_seconds: int = 60) -> MemoryProfiler:
    """
    Start global memory profiling.
    
    Args:
        output_dir: Directory to write profiling output
        interval_seconds: How often to log measurements
        
    Returns:
        MemoryProfiler instance
    """
    global _global_profiler
    if _global_profiler is not None:
        logger.warning("Memory profiling already started")
        return _global_profiler
    
    _global_profiler = MemoryProfiler(output_dir=output_dir, interval_seconds=interval_seconds)
    _global_profiler.start()
    return _global_profiler


def stop_profiling():
    """Stop global memory profiling."""
    global _global_profiler
    if _global_profiler is not None:
        _global_profiler.stop()
        _global_profiler = None


def take_snapshot(label: str):
    """Take a memory snapshot (convenience function)."""
    if _global_profiler is not None:
        _global_profiler.take_snapshot(label)


def write_epoch_summary(epoch: int):
    """Write epoch-specific summary (convenience function)."""
    if _global_profiler is not None:
        _global_profiler.write_epoch_summary(epoch)

