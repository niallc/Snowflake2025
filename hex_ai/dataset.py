"""
Dataset and data loading utilities for Hex AI training.

This module provides PyTorch Dataset classes and data loading utilities
for training the Hex AI model on game data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, List
import numpy as np
from pathlib import Path
import logging
logger = logging.getLogger(__name__)

from .config import BOARD_SIZE, NUM_PLAYERS, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
from .data_utils import load_trmph_file, augment_board


def get_dataset_info(data_dir: str) -> dict:
    """
    Get information about the dataset.
    
    Args:
        data_dir: Directory containing .trmph files
        
    Returns:
        Dictionary with dataset information
    """
    data_path = Path(data_dir)
    files = list(data_path.glob("*.trmph"))
    
    return {
        "num_files": len(files),
        "data_dir": str(data_path),
        "file_extensions": [f.suffix for f in files[:5]],  # First 5 file extensions
        "total_size_mb": sum(f.stat().st_size for f in files) / (1024 * 1024)
    } 