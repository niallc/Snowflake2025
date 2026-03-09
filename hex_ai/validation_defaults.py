"""
Validation data configuration resolution.

Validation is disabled by default. Callers that want validation must provide
both directories and shard ranges explicitly.
"""

from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

def get_default_validation_config() -> Tuple[List[str], List[str]]:
    """
    Get the default validation directories and shard ranges.
    
    Returns:
        Tuple of (validation_dirs, validation_shard_ranges)
    """
    return [], []


def resolve_validation_config(
    validation_dirs: Optional[List[str]] = None,
    validation_shard_ranges: Optional[List[str]] = None,
    no_validation: bool = False
) -> Tuple[List[str], List[str]]:
    """
    Resolve validation configuration.

    Note: Training and validation datasets are constructed independently.
    If using the same directory for both, ensure shard ranges don't overlap
    to avoid data leakage.
    
    Args:
        validation_dirs: Explicit validation directories
        validation_shard_ranges: Explicit validation shard ranges
        no_validation: Disable validation entirely
        
    Returns:
        Tuple of (validation_data_dirs, validation_shard_ranges)
    """
    if no_validation:
        logger.info("Validation disabled via --no-validation flag")
        return [], []

    if validation_dirs is None and validation_shard_ranges is None:
        logger.info("Validation disabled by default (no validation dirs/ranges provided)")
        return [], []

    resolved_dirs = validation_dirs or []
    resolved_ranges = validation_shard_ranges or []

    if not resolved_dirs and not resolved_ranges:
        logger.info("Validation disabled explicitly with empty validation dirs/ranges")
        return [], []

    if bool(resolved_dirs) != bool(resolved_ranges):
        raise ValueError(
            "Validation requires both validation_dirs and validation_shard_ranges. "
            "Provide both, or neither, or use --no-validation."
        )

    logger.info(
        f"Using explicit validation config: dirs={resolved_dirs}, ranges={resolved_ranges}"
    )
    return resolved_dirs, resolved_ranges


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
