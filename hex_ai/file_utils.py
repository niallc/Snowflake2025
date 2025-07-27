"""
File handling utilities for batch processing operations.

This module provides:
- Graceful shutdown handling for long-running processes
- Atomic file write operations
- Progress reporting utilities
- File validation functions
"""

import signal
import sys
import logging
import gzip
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """Handle graceful shutdown on SIGTERM/SIGINT."""
    
    def __init__(self):
        self.shutdown_requested = False
        self.emergency_exit = False
        self.signal_count = 0
        
        # Register signal handlers
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        self.signal_count += 1
        if self.signal_count == 1:
            logger.info(f"Received signal {signum} - initiating graceful shutdown...")
            self.shutdown_requested = True
        elif self.signal_count == 2:
            logger.warning(f"Received second signal {signum} - forcing emergency exit!")
            self.emergency_exit = True
            sys.exit(1)
        else:
            logger.error(f"Received signal {signum} for the {self.signal_count}th time - forcing exit!")
            sys.exit(1)


def atomic_write_pickle_gz(data: Dict[str, Any], output_file: Path, temp_suffix: str = '.tmp') -> None:
    """
    Write data to a compressed pickle file using atomic write.
    
    Args:
        data: Dictionary to save
        output_file: Target file path
        temp_suffix: Suffix for temporary file
    
    Raises:
        RuntimeError: If target file already exists (race condition)
        Exception: If write operation fails
    """
    temp_file = output_file.with_suffix(temp_suffix)
    
    # Check if target file already exists (race condition check)
    if output_file.exists():
        logger.warning(f"Target file {output_file} already exists - this may indicate a race condition")
        logger.warning(f"Will attempt to overwrite existing file")
    
    try:
        # Write data
        with gzip.open(temp_file, 'wb') as f:
            pickle.dump(data, f)
        
        # Atomic move - this will fail if target exists on most filesystems
        try:
            temp_file.rename(output_file)
        except FileExistsError:
            # Target file exists - this is unexpected and indicates a bug
            temp_file.unlink()  # Clean up temp file
            logger.error(f"Target file {output_file} already exists - possible race condition or duplicate processing")
            raise RuntimeError(f"Target file {output_file} already exists - possible race condition or duplicate processing")
        except OSError as e:
            # Handle other filesystem errors
            temp_file.unlink()  # Clean up temp file
            logger.error(f"Filesystem error during atomic move: {e}")
            raise RuntimeError(f"Filesystem error during atomic move: {e}")
            
    except Exception as e:
        # Clean up temp file if it exists
        if temp_file.exists():
            temp_file.unlink(missing_ok=True)
        raise e


def validate_output_directory(output_dir: Path, data_dir: Optional[Path] = None) -> None:
    """
    Validate output directory permissions and available space.
    
    Args:
        output_dir: Directory to validate
        data_dir: Optional data directory for space estimation
    
    Raises:
        ValueError: If directory is not writable or has insufficient space
    """
    try:
        # Create directory if it doesn't exist
        output_dir.mkdir(exist_ok=True)
        
        # Check write permissions
        test_file = output_dir / ".test_write_permission"
        try:
            test_file.write_text("test")
            test_file.unlink()  # Clean up
        except (PermissionError, OSError) as e:
            raise ValueError(f"Output directory {output_dir} is not writable: {e}")
        
        # Check available disk space
        import shutil
        total, used, free = shutil.disk_usage(output_dir)
        free_gb = free / (1024**3)
        
        # Estimate space needed if data directory provided
        if data_dir:
            estimated_files = len(list(data_dir.rglob("*.trmph")))
            # Rough estimate: each trmph file might produce 10-100MB of processed data
            estimated_needed_gb = estimated_files * 0.1  # Conservative estimate
            
            if free_gb + 20 < estimated_needed_gb:
                logger.warning(f"Low disk space: {free_gb:.1f}GB free, estimated need: {estimated_needed_gb:.1f}GB")
                logger.warning(f"Consider freeing up space or processing fewer files")
            elif free_gb < 40.0:
                logger.warning(f"Low disk space: {free_gb:.1f}GB free in {output_dir}")
        else:
            if free_gb < 10.0:
                logger.warning(f"Low disk space: {free_gb:.1f}GB free in {output_dir}")
        
        logger.info(f"Output directory validated: {output_dir}")
        
    except Exception as e:
        raise ValueError(f"Failed to validate output directory {output_dir}: {e}")


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to be safe for filesystem.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename safe for filesystem use
    """
    import re
    import unicodedata
    
    # Normalize unicode characters
    filename = unicodedata.normalize('NFKD', filename)
    
    # Remove or replace problematic characters
    # Keep alphanumeric, dots, hyphens, underscores
    filename = re.sub(r'[^\w\-_.]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    filename = filename.strip('._ ')
    
    # Ensure filename is not empty
    if not filename:
        filename = "unnamed"
    
    # Limit length to prevent filesystem issues
    if len(filename) > 200:
        filename = filename[:200]
    
    return filename


def save_progress_report(stats: Dict[str, Any], output_dir: Path, shutdown_handler: Optional[GracefulShutdown] = None, 
                        current_file_index: Optional[int] = None, total_files: Optional[int] = None, 
                        current_file: Optional[str] = None) -> None:
    """
    Save a progress report with current statistics.
    
    Args:
        stats: Current processing statistics
        output_dir: Directory to save progress report
        shutdown_handler: Optional shutdown handler for status
        current_file_index: Current file being processed
        total_files: Total number of files to process
        current_file: Current file path being processed
    """
    try:
        # Create progress report
        progress_data = {
            'stats': stats.copy(),
            'timestamp': datetime.now().isoformat(),
            'shutdown_requested': shutdown_handler.shutdown_requested if shutdown_handler else False
        }
        
        # Add current file being processed if available
        if current_file_index is not None and total_files is not None:
            progress_data['current_file_index'] = current_file_index
            progress_data['total_files'] = total_files
            if current_file:
                progress_data['current_file'] = current_file
        
        # Add summary statistics
        if stats.get('all_games', 0) > 0:
            progress_data['summary'] = {
                'success_rate_percent': round(stats['valid_games'] / stats['all_games'] * 100, 1),
                'examples_per_game': round(stats['total_examples'] / stats['valid_games'], 1) if stats['valid_games'] > 0 else 0,
                'files_per_second': round(stats.get('files_per_second', 0), 2),
                'elapsed_hours': round(stats.get('elapsed_time', 0) / 3600, 1)
            }
        
        # Save progress report
        progress_file = output_dir / "processing_progress.json"
        import json
        with open(progress_file, 'w') as f:
            json.dump(progress_data, f, indent=2)
        
        logger.info(f"Progress report saved to {progress_file}")
        
    except Exception as e:
        logger.error(f"Failed to save progress report: {e}") 