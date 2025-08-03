"""
File handling utilities for batch processing operations.

This module provides:
- Graceful shutdown handling for long-running processes
- Atomic file write operations
- Progress reporting utilities
- File validation functions
- Model checkpoint discovery and management
"""

import signal
import sys
import logging
import gzip
import pickle
import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
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


def get_unique_checkpoint_path(base_path: Path) -> Path:
    """
    Get a unique checkpoint path by appending timestamp if the file already exists.
    
    Args:
        base_path: The desired checkpoint path (e.g., Path("epoch3_mini19.pt.gz"))
        
    Returns:
        A unique path that won't overwrite existing files.
        If base_path doesn't exist, returns base_path.
        If base_path exists, returns base_path with timestamp appended.
        
    Example:
        If "epoch3_mini19.pt.gz" exists, returns "epoch3_mini19_250802_1041.pt.gz"
    """
    if not base_path.exists():
        return base_path
    
    # Extract stem and suffix
    stem = base_path.stem
    suffix = base_path.suffix
    
    # Generate timestamp (yymmdd_hrmin format)
    from datetime import datetime
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    
    # Create new path with timestamp
    new_stem = f"{stem}_{timestamp}"
    new_path = base_path.parent / f"{new_stem}{suffix}"
    
    # If this path also exists, add seconds to make it unique
    if new_path.exists():
        timestamp_with_seconds = datetime.now().strftime("%y%m%d_%H%M%S")
        new_stem = f"{stem}_{timestamp_with_seconds}"
        new_path = base_path.parent / f"{new_stem}{suffix}"
    
    return new_path 


def scan_checkpoint_directory(checkpoints_dir: Path = Path("checkpoints")) -> List[Dict[str, Any]]:
    """
    Scan the checkpoints directory to discover available model files.
    
    Args:
        checkpoints_dir: Path to the checkpoints directory
        
    Returns:
        List of model file information dictionaries with keys:
        - path: Full path to the model file
        - relative_path: Path relative to checkpoints directory
        - directory: Directory name containing the model
        - filename: Just the filename
        - size_mb: File size in MB
        - modified_time: Last modified timestamp
        - epoch: Extracted epoch number (if available)
        - mini: Extracted mini number (if available)
    """
    if not checkpoints_dir.exists():
        logger.warning(f"Checkpoints directory {checkpoints_dir} does not exist")
        return []
    
    models = []
    
    # Pattern to match checkpoint files: epochX_miniY.pt.gz
    checkpoint_pattern = re.compile(r'epoch(\d+)_mini(\d+)\.pt\.gz$')
    
    try:
        # Walk through all subdirectories
        for root, dirs, files in os.walk(checkpoints_dir):
            root_path = Path(root)
            
            for file in files:
                if file.endswith('.pt.gz'):
                    file_path = root_path / file
                    relative_path = file_path.relative_to(checkpoints_dir)
                    
                    # Extract epoch and mini numbers if possible
                    match = checkpoint_pattern.match(file)
                    epoch = int(match.group(1)) if match else None
                    mini = int(match.group(2)) if match else None
                    
                    # Get file stats
                    try:
                        stat = file_path.stat()
                        size_mb = stat.st_size / (1024 * 1024)
                        modified_time = datetime.fromtimestamp(stat.st_mtime)
                    except OSError as e:
                        logger.warning(f"Could not get stats for {file_path}: {e}")
                        size_mb = 0
                        modified_time = datetime.now()
                    
                    model_info = {
                        'path': str(file_path),
                        'relative_path': str(relative_path),
                        'directory': str(relative_path.parent),
                        'filename': file,
                        'size_mb': round(size_mb, 1),
                        'modified_time': modified_time.isoformat(),
                        'epoch': epoch,
                        'mini': mini
                    }
                    
                    models.append(model_info)
    
    except Exception as e:
        logger.error(f"Error scanning checkpoint directory: {e}")
        return []
    
    # Sort by directory, then epoch, then mini
    models.sort(key=lambda x: (x['directory'], x['epoch'] or 0, x['mini'] or 0))
    
    logger.info(f"Found {len(models)} model files in {checkpoints_dir}")
    return models


def validate_model_file(model_path: str) -> Tuple[bool, str]:
    """
    Validate that a model file exists and is accessible.
    
    Args:
        model_path: Path to the model file (can be relative to checkpoints/)
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Handle relative paths
        if not model_path.startswith('/'):
            full_path = Path("checkpoints") / model_path
        else:
            full_path = Path(model_path)
        
        if not full_path.exists():
            return False, f"Model file does not exist: {full_path}"
        
        if not full_path.is_file():
            return False, f"Path is not a file: {full_path}"
        
        # Try to get file stats to verify accessibility
        stat = full_path.stat()
        if stat.st_size == 0:
            return False, f"Model file is empty: {full_path}"
        
        return True, ""
        
    except Exception as e:
        return False, f"Error validating model file: {e}"


def get_model_directories(checkpoints_dir: Path = Path("checkpoints")) -> List[str]:
    """
    Get list of all directories containing model files.
    
    Args:
        checkpoints_dir: Path to the checkpoints directory
        
    Returns:
        List of directory names (relative to checkpoints/)
    """
    models = scan_checkpoint_directory(checkpoints_dir)
    directories = list(set(model['directory'] for model in models))
    directories.sort()
    return directories


def get_models_in_directory(directory: str, checkpoints_dir: Path = Path("checkpoints")) -> List[Dict[str, Any]]:
    """
    Get all models in a specific directory.
    
    Args:
        directory: Directory name (relative to checkpoints/)
        checkpoints_dir: Path to the checkpoints directory
        
    Returns:
        List of model file information dictionaries
    """
    models = scan_checkpoint_directory(checkpoints_dir)
    return [model for model in models if model['directory'] == directory]


def save_recent_models(recent_models: List[str], max_count: int = 10) -> None:
    """
    Save list of recently used models to a file.
    
    Args:
        recent_models: List of model paths (most recent first)
        max_count: Maximum number of models to save
    """
    try:
        # Limit to max_count
        recent_models = recent_models[:max_count]
        
        # Save to user's home directory
        home_dir = Path.home()
        config_dir = home_dir / ".hex_ai"
        config_dir.mkdir(exist_ok=True)
        
        config_file = config_dir / "recent_models.json"
        
        import json
        with open(config_file, 'w') as f:
            json.dump({
                'recent_models': recent_models,
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.debug(f"Saved {len(recent_models)} recent models to {config_file}")
        
    except Exception as e:
        logger.error(f"Failed to save recent models: {e}")


def load_recent_models() -> List[str]:
    """
    Load list of recently used models from file.
    
    Returns:
        List of model paths (most recent first)
    """
    try:
        home_dir = Path.home()
        config_file = home_dir / ".hex_ai" / "recent_models.json"
        
        if not config_file.exists():
            return []
        
        import json
        with open(config_file, 'r') as f:
            data = json.load(f)
        
        recent_models = data.get('recent_models', [])
        
        # Validate that all models still exist
        valid_models = []
        for model_path in recent_models:
            is_valid, _ = validate_model_file(model_path)
            if is_valid:
                valid_models.append(model_path)
            else:
                logger.debug(f"Removing invalid recent model: {model_path}")
        
        # Save back the cleaned list
        if len(valid_models) != len(recent_models):
            save_recent_models(valid_models)
        
        return valid_models
        
    except Exception as e:
        logger.error(f"Failed to load recent models: {e}")
        return []


def add_recent_model(model_path: str) -> None:
    """
    Add a model to the recent models list.
    
    Args:
        model_path: Path to the model file
    """
    try:
        recent_models = load_recent_models()
        
        # Remove if already exists (will be added to front)
        if model_path in recent_models:
            recent_models.remove(model_path)
        
        # Add to front
        recent_models.insert(0, model_path)
        
        # Save updated list
        save_recent_models(recent_models)
        
        logger.debug(f"Added {model_path} to recent models")
        
    except Exception as e:
        logger.error(f"Failed to add recent model: {e}") 