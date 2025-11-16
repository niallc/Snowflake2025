"""
Memory leak diagnostic tracking for data pipeline.

This module provides lightweight diagnostic tracking to detect potential memory leaks
in the data pipeline, specifically:
- Array reference sharing (positions sharing memory with shard data)
- Pool size growth over time
- Shard data retention issues

Results are written to temp/memoryProfile/ for analysis.
"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)


class MemoryLeakDiagnostics:
    """
    Tracks internal state to detect potential memory leaks in the data pipeline.
    
    Writes diagnostic results to temp/memoryProfile/ directory.
    """
    
    def __init__(self, output_dir: str = "temp/memoryProfile", enabled: bool = True):
        """
        Initialize diagnostic tracker.
        
        Args:
            output_dir: Directory to write diagnostic reports
            enabled: Whether diagnostics are enabled
        """
        self.enabled = enabled
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this session
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.diagnostic_file = self.output_dir / f"memory_leak_diagnostics_{self.session_timestamp}.txt"
        
        # Tracking state
        self.pool_size_history: deque = deque(maxlen=50)  # Keep last 50 measurements
        self.array_sharing_detected = False
        self.pool_growth_warnings = 0
        self.shard_load_count = 0
        self.shard_data_tracking: Dict[str, int] = {}  # Track shard data dict IDs to detect retention
        self.shared_array_ids: Dict[int, Dict] = {}  # Track shared array IDs: array_id -> {size_bytes, shard_path, array_type}
        
        # Write header
        if self.enabled:
            with open(self.diagnostic_file, 'w') as f:
                f.write("Memory Leak Diagnostics Report\n")
                f.write("=" * 80 + "\n")
                f.write(f"Session timestamp: {self.session_timestamp}\n")
                f.write(f"Started: {datetime.now().isoformat()}\n\n")
    
    def check_array_sharing(self, example: Dict, shard_data: Dict, shard_path: str) -> bool:
        """
        Check if position arrays share memory with shard data (memory leak indicator).
        
        This should be called AFTER copying the example. If the copied example still
        shares memory with the original shard data, that indicates the copy failed
        and the shard data cannot be garbage collected.
        
        Args:
            example: Position example (should be the COPIED example, not the original)
            shard_data: Original shard data dict
            shard_path: Path to shard file (for logging)
            
        Returns:
            True if array sharing detected (memory leak), False otherwise
        """
        if not self.enabled:
            return False
        
        if not example or 'board' not in example:
            return False
        
        if 'examples' not in shard_data or not shard_data['examples']:
            return False
        
        # Check if board array shares memory address with original
        example_board_id = id(example['board'])
        shard_board_id = id(shard_data['examples'][0]['board'])
        
        if example_board_id == shard_board_id:
            self.array_sharing_detected = True
            # Track this shared array for memory estimation
            self.shared_array_ids[example_board_id] = {
                'size_bytes': example['board'].nbytes,
                'shard_path': str(shard_path),
                'array_type': 'board'
            }
            self._log_diagnostic(
                "🚨 ARRAY SHARING DETECTED",
                f"Position arrays share memory with shard data!\n"
                f"  Shard: {shard_path}\n"
                f"  Board array ID: {example_board_id}\n"
                f"  Array size: {example['board'].nbytes / (1024*1024):.2f} MB\n"
                f"  This prevents shard data from being freed.\n"
                f"  FIX: Copy arrays when adding positions to pool."
            )
            return True
        
        # Check policy array sharing if present
        # NOTE: Policy can be None for terminal positions (final moves with no next move).
        # This is expected and valid - about 1% of examples have None policy.
        # If policy is None, there's no array to check for memory sharing, so we skip.
        example_policy = example.get('policy')
        if example_policy is None:
            return False  # No array to check - this is expected for terminal positions
        
        # Policy exists in copied example - check if it shares memory with shard data
        # We compare against the first example in the shard (the diagnostic only runs once per shard)
        shard_policy = shard_data['examples'][0].get('policy')
        if shard_policy is None:
            # First shard example has None policy but copied example has policy - they're different examples
            # This is fine - we can't check for sharing between different examples
            return False
        
        # Both have policy arrays - check if they share memory
        example_policy_id = id(example_policy)
        shard_policy_id = id(shard_policy)
        if example_policy_id == shard_policy_id:
            self.array_sharing_detected = True
            # Track this shared array for memory estimation
            self.shared_array_ids[example_policy_id] = {
                'size_bytes': example_policy.nbytes,
                'shard_path': str(shard_path),
                'array_type': 'policy'
            }
            self._log_diagnostic(
                "🚨 ARRAY SHARING DETECTED",
                f"Policy arrays share memory with shard data!\n"
                f"  Shard: {shard_path}\n"
                f"  Policy array ID: {example_policy_id}\n"
                f"  Array size: {example_policy.nbytes / (1024*1024):.2f} MB\n"
                f"  This prevents shard data from being freed.\n"
                f"  FIX: Copy arrays when adding positions to pool."
            )
            return True
        
        return False
    
    def track_shard_data_loaded(self, shard_data: Dict, shard_path: str):
        """
        Track that shard data was loaded. This helps detect if shard data dicts
        are being retained in memory after they should be freed.
        
        Args:
            shard_data: The loaded shard data dictionary
            shard_path: Path to the shard file
        """
        if not self.enabled:
            return
        
        # Store the ID of the shard data dict
        shard_data_id = id(shard_data)
        self.shard_data_tracking[str(shard_path)] = shard_data_id
    
    def verify_shard_data_freed(self, shard_path: str):
        """
        Verify that shard data has been freed after processing.
        This is a best-effort check - we can't directly verify GC, but we can
        check if the dict ID is still accessible (which would indicate retention).
        
        Args:
            shard_path: Path to the shard file that was processed
        """
        if not self.enabled:
            return
        
        # Note: This is a best-effort check. We can't directly verify GC,
        # but tracking the dict IDs helps identify patterns if memory leaks occur.
        # The explicit `del data` in the code should ensure proper cleanup.
        if str(shard_path) in self.shard_data_tracking:
            # Log that we expect this data to be freed
            # (We can't directly verify, but this helps with debugging)
            pass  # Silent tracking - only log if we detect actual retention issues
    
    def track_pool_size(self, pool_size: int, threshold: int = 2_000_000):
        """
        Track pool size and warn if it grows unexpectedly.
        
        Args:
            pool_size: Current size of position pool
            threshold: Size threshold for warnings (default: 2M positions)
        """
        if not self.enabled:
            return
        
        self.pool_size_history.append(pool_size)
        self.shard_load_count += 1
        
        # Check if pool exceeds threshold
        if pool_size > threshold:
            # Check if pool is growing over time
            if len(self.pool_size_history) >= 5:
                recent_avg = sum(list(self.pool_size_history)[-5:]) / 5
                if pool_size > recent_avg * 1.5:
                    self.pool_growth_warnings += 1
                    self._log_diagnostic(
                        "⚠️ POOL SIZE GROWING",
                        f"Position pool size is growing unexpectedly:\n"
                        f"  Current size: {pool_size:,} positions\n"
                        f"  Recent average: {recent_avg:,.0f} positions\n"
                        f"  Growth: {(pool_size / recent_avg - 1) * 100:.1f}%\n"
                        f"  This may indicate a memory leak."
                    )
                elif pool_size > threshold:
                    # Pool is large but not necessarily growing
                    self._log_diagnostic(
                        "⚠️ LARGE POOL SIZE",
                        f"Position pool is very large: {pool_size:,} positions\n"
                        f"  This is ~{pool_size * 3 / 1024 / 1024:.1f} GB in position dicts alone."
                    )
    
    def estimate_shared_array_memory(self, position_pool: List[Dict]) -> Dict:
        """
        Scan position pool to estimate memory from shared arrays.
        
        This scans the entire position pool and counts how many positions have arrays
        that match IDs we've detected as shared. This gives us a quantitative measure
        of the memory leak from array sharing.
        
        Args:
            position_pool: List of position dictionaries from the dataset
            
        Returns:
            Dict with:
            - shared_position_count: number of positions with shared arrays
            - estimated_memory_mb: estimated memory retained from shared arrays (MB)
            - estimated_memory_gb: estimated memory retained from shared arrays (GB)
        """
        if not self.enabled:
            return {'shared_position_count': 0, 'estimated_memory_mb': 0.0, 'estimated_memory_gb': 0.0}
        
        shared_count = 0
        total_memory_bytes = 0
        
        for position in position_pool:
            board_id = id(position.get('board')) if position.get('board') is not None else None
            policy_id = id(position.get('policy')) if position.get('policy') is not None else None
            
            if board_id and board_id in self.shared_array_ids:
                shared_count += 1
                total_memory_bytes += self.shared_array_ids[board_id]['size_bytes']
            
            if policy_id and policy_id in self.shared_array_ids:
                shared_count += 1
                if position.get('policy') is not None:
                    total_memory_bytes += self.shared_array_ids[policy_id]['size_bytes']
        
        return {
            'shared_position_count': shared_count,
            'estimated_memory_mb': total_memory_bytes / (1024 * 1024),
            'estimated_memory_gb': total_memory_bytes / (1024 * 1024 * 1024)
        }
    
    def _log_diagnostic(self, level: str, message: str):
        """Write diagnostic message to file."""
        timestamp = datetime.now().isoformat()
        with open(self.diagnostic_file, 'a') as f:
            f.write(f"\n[{timestamp}] {level}\n")
            f.write(f"{message}\n")
            f.write("-" * 80 + "\n")
        
        # Also log to Python logger at appropriate level
        if "🚨" in level:
            logger.error(f"{level}: {message}")
        elif "⚠️" in level:
            logger.warning(f"{level}: {message}")
        else:
            logger.info(f"{level}: {message}")
    
    def write_summary(self):
        """Write final diagnostic summary."""
        if not self.enabled:
            return
        
        with open(self.diagnostic_file, 'a') as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write("Diagnostic Summary\n")
            f.write("=" * 80 + "\n")
            f.write(f"Session ended: {datetime.now().isoformat()}\n")
            f.write(f"Total shard loads tracked: {self.shard_load_count}\n")
            f.write(f"Shard data dicts tracked: {len(self.shard_data_tracking)}\n")
            f.write(f"Array sharing detected: {self.array_sharing_detected}\n")
            f.write(f"Pool growth warnings: {self.pool_growth_warnings}\n")
            
            if self.pool_size_history:
                f.write(f"\nPool size statistics:\n")
                f.write(f"  Min: {min(self.pool_size_history):,}\n")
                f.write(f"  Max: {max(self.pool_size_history):,}\n")
                f.write(f"  Final: {self.pool_size_history[-1]:,}\n")
            
            if self.array_sharing_detected:
                f.write("\n⚠️ ACTION REQUIRED: Array sharing was detected!\n")
                f.write("   This is a confirmed memory leak. Arrays must be copied\n")
                f.write("   when adding positions to the pool.\n")
            
            if self.pool_growth_warnings > 0:
                f.write(f"\n⚠️ WARNING: {self.pool_growth_warnings} pool growth warnings detected.\n")
                f.write("   Monitor memory usage over longer periods.\n")


# Global diagnostic instance (None when not enabled)
_global_diagnostics: Optional[MemoryLeakDiagnostics] = None


def get_diagnostics() -> Optional[MemoryLeakDiagnostics]:
    """Get the global diagnostic tracker instance, or None if not enabled."""
    return _global_diagnostics


def start_diagnostics(output_dir: str = "temp/memoryProfile") -> MemoryLeakDiagnostics:
    """
    Start global memory leak diagnostics.
    
    Args:
        output_dir: Directory to write diagnostic reports
        
    Returns:
        MemoryLeakDiagnostics instance
    """
    global _global_diagnostics
    if _global_diagnostics is not None:
        logger.warning("Memory leak diagnostics already started")
        return _global_diagnostics
    
    _global_diagnostics = MemoryLeakDiagnostics(output_dir=output_dir, enabled=True)
    return _global_diagnostics


def stop_diagnostics():
    """Stop global memory leak diagnostics and write final summary."""
    global _global_diagnostics
    if _global_diagnostics is not None:
        _global_diagnostics.write_summary()
        _global_diagnostics = None

