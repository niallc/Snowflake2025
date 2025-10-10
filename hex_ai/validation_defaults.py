"""
Validation data defaults and configuration resolution.

This module provides hardcoded defaults for validation data and simple
resolution logic for command-line overrides.
"""

from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Default validation configuration - update in code when needed
DEFAULT_VALIDATION_DIRS = [
    "data/processed/sf18_shuffled",
    "data/processed/shuffled_selfplay_20250924_105628"
]

DEFAULT_VALIDATION_RANGES = [
    "498-498",  # Use only 1 shard from sf18_shuffled (the last one)
    "99-99"     # Keep 1 shard from selfplay directory
]


def get_default_validation_config() -> Tuple[List[str], List[str]]:
    """
    Get the default validation directories and shard ranges.
    
    Returns:
        Tuple of (validation_dirs, validation_shard_ranges)
    """
    return DEFAULT_VALIDATION_DIRS.copy(), DEFAULT_VALIDATION_RANGES.copy()


def resolve_validation_config(
    validation_dirs: Optional[List[str]] = None,
    validation_shard_ranges: Optional[List[str]] = None,
    no_validation: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Resolve validation configuration with simple logic.
    
    Note: Training and validation datasets are constructed independently.
    If using the same directory for both, ensure shard ranges don't overlap
    to avoid data leakage.
    
    Args:
        validation_dirs: Override default validation directories
        validation_shard_ranges: Override default validation shard ranges
        no_validation: Disable validation entirely
        
    Returns:
        Tuple of (validation_data_dirs, validation_shard_ranges)
    """
    if no_validation:
        logger.info("Validation disabled via --no-validation flag")
        return [], []
    
    if validation_dirs and validation_shard_ranges:
        # Use explicit specification
        logger.info(f"Using explicit validation config: dirs={validation_dirs}, ranges={validation_shard_ranges}")
        return validation_dirs, validation_shard_ranges
    
    if validation_dirs or validation_shard_ranges:
        # Partial specification - use defaults for missing parts
        default_dirs, default_ranges = get_default_validation_config()
        resolved_dirs = validation_dirs or default_dirs
        resolved_ranges = validation_shard_ranges or default_ranges
        logger.info(f"Using partial validation config: dirs={resolved_dirs}, ranges={resolved_ranges}")
        return resolved_dirs, resolved_ranges
    
    # Use defaults
    default_dirs, default_ranges = get_default_validation_config()
    logger.info(f"Using default validation config: dirs={default_dirs}, ranges={default_ranges}")
    return default_dirs, default_ranges


def log_validation_summary(validation_dirs: List[str], validation_ranges: List[str]) -> None:
    """
    Log a summary of the validation configuration being used.
    
    Args:
        validation_dirs: Validation data directories
        validation_ranges: Validation shard ranges
    """
    if not validation_dirs:
        logger.info("No validation data configured")
        return
    
    logger.info("=" * 60)
    logger.info("VALIDATION DATA CONFIGURATION")
    logger.info("=" * 60)
    for i, (data_dir, shard_range) in enumerate(zip(validation_dirs, validation_ranges)):
        logger.info(f"Validation directory {i+1}: {data_dir} (range: {shard_range})")
    logger.info("=" * 60)
