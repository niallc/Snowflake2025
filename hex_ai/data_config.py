"""
Data configuration for Hex AI.

This module centralizes default paths and configuration for data processing.
"""

from pathlib import Path
from typing import List

# Default data directories
DEFAULT_DATA_ROOT = Path("data")

# Source directories for data collection
DEFAULT_SOURCE_DIRS = [
    DEFAULT_DATA_ROOT / "tournament_play",
    DEFAULT_DATA_ROOT / "sf25"
]

# Processed data directories
DEFAULT_PROCESSED_DATA_DIRS = [
    DEFAULT_DATA_ROOT / "processed" / "sf18_shuffled",
    DEFAULT_DATA_ROOT / "processed" / "shuffled_sf25_20250828"
]
DEFAULT_SF25_SHUFFLED_DIR = DEFAULT_DATA_ROOT / "processed" / "sf25_shuffled"

# Intermediate processing directories
DEFAULT_COLLECTED_DIR = DEFAULT_DATA_ROOT / "collected"
DEFAULT_CLEANED_DIR = DEFAULT_DATA_ROOT / "cleaned"

# Processed file tracking
PROCESSED_SUBDIR_NAME = "processed"

# Default chunk sizes
DEFAULT_CHUNK_SIZE = 20000

# Default output naming patterns
def get_sf25_shuffled_dir_name(date_suffix: str = None) -> Path:
    """Get the default SF25 shuffled directory name with optional date suffix."""
    if date_suffix is None:
        from datetime import datetime
        date_suffix = datetime.now().strftime("%Y%m%d")
    
    return DEFAULT_DATA_ROOT / "processed" / f"sf25_shuffled_{date_suffix}"


def get_collected_dir_name(name: str = None) -> Path:
    """Get the default collected directory name."""
    if name is None:
        from datetime import datetime
        name = f"sf25_{datetime.now().strftime('%Y%m%d')}"
    
    return DEFAULT_COLLECTED_DIR / name


def get_cleaned_dir_name(name: str = None) -> Path:
    """Get the default cleaned directory name."""
    if name is None:
        from datetime import datetime
        name = f"cleaned_{datetime.now().strftime('%Y%m%d')}"
    
    return DEFAULT_CLEANED_DIR / name


# Validation functions
def validate_data_directories() -> List[str]:
    """
    Validate that all default data directories exist.
    
    Returns:
        List of missing directory paths
    """
    missing_dirs = []
    
    # Check source directories
    for source_dir in DEFAULT_SOURCE_DIRS:
        if not source_dir.exists():
            missing_dirs.append(str(source_dir))
    
    # Check processed directories
    for processed_dir in DEFAULT_PROCESSED_DATA_DIRS:
        if not processed_dir.exists():
            missing_dirs.append(str(processed_dir))
    
    return missing_dirs


def ensure_data_directories():
    """Create default data directories if they don't exist."""
    directories = [
        DEFAULT_DATA_ROOT,
        DEFAULT_DATA_ROOT / "processed",
        DEFAULT_COLLECTED_DIR,
        DEFAULT_CLEANED_DIR,
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


def get_processed_subdir(source_dir: Path) -> Path:
    """Get the processed subdirectory for a source directory."""
    return source_dir / PROCESSED_SUBDIR_NAME


def mark_files_as_processed(source_dir: Path, processed_files: List[Path]) -> None:
    """
    Move processed files to a 'processed' subdirectory to mark them as handled.
    
    WARNING: This permanently moves files. Only call after successful completion.
    
    Args:
        source_dir: The source directory containing the files
        processed_files: List of file paths that have been processed
    """
    processed_dir = get_processed_subdir(source_dir)
    processed_dir.mkdir(exist_ok=True)
    
    for file_path in processed_files:
        if file_path.exists():
            # Move to processed subdirectory
            new_path = processed_dir / file_path.name
            file_path.rename(new_path)


def get_processed_files_log_path(source_dir: Path) -> Path:
    """Get the path to a log file that tracks processed files without moving them."""
    return source_dir / "processed_files.log"


def log_processed_files(source_dir: Path, processed_files: List[Path]) -> None:
    """
    Log processed files without moving them. This allows for recovery if processing fails.
    
    Args:
        source_dir: The source directory containing the files
        processed_files: List of file paths that have been processed
    """
    log_path = get_processed_files_log_path(source_dir)
    
    with open(log_path, 'a') as f:
        for file_path in processed_files:
            f.write(f"{file_path.name}\n")


def get_unprocessed_files_from_log(source_dir: Path) -> List[Path]:
    """
    Get list of unprocessed .trmph files based on log file.
    
    Args:
        source_dir: The source directory to check
        
    Returns:
        List of unprocessed .trmph file paths
    """
    log_path = get_processed_files_log_path(source_dir)
    
    # Get all .trmph files in source directory
    all_files = list(source_dir.glob("*.trmph"))
    
    if not log_path.exists():
        return all_files
    
    # Read processed files from log
    with open(log_path, 'r') as f:
        processed_filenames = {line.strip() for line in f if line.strip()}
    
    # Filter out processed files
    unprocessed_files = []
    for file_path in all_files:
        if file_path.name not in processed_filenames:
            unprocessed_files.append(file_path)
    
    return unprocessed_files


def get_unprocessed_files(source_dir: Path) -> List[Path]:
    """
    Get list of unprocessed .trmph files in a source directory.
    
    Args:
        source_dir: The source directory to check
        
    Returns:
        List of unprocessed .trmph file paths
    """
    processed_dir = get_processed_subdir(source_dir)
    
    # Get all .trmph files in source directory
    all_files = list(source_dir.glob("*.trmph"))
    
    # Filter out files that are in the processed subdirectory
    unprocessed_files = []
    for file_path in all_files:
        if not (processed_dir / file_path.name).exists():
            unprocessed_files.append(file_path)
    
    return unprocessed_files
