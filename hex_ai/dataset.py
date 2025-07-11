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
from .data_utils import load_trmph_file, convert_to_matrix_format, augment_board


class HexDataset(Dataset):
    """Dataset for Hex game data."""
    
    def __init__(self, data_source, board_size: int = BOARD_SIZE):
        """
        Initialize dataset.
        
        Args:
            data_source: Either a directory path (str/Path) containing .trmph files,
                        or a list of trmph strings
            board_size: Size of the board
        """
        self.board_size = board_size
        
        if isinstance(data_source, (str, Path)):
            # Load from directory
            self.data_dir = Path(data_source)
            self.game_files = list(self.data_dir.glob("*.trmph"))
            self.game_data = []
            
            for file_path in self.game_files:
                with open(file_path, 'r') as f:
                    content = f.read().strip()
                    if content:
                        self.game_data.append(content)
        else:
            # Direct list of trmph strings
            self.data_dir = None
            self.game_files = []
            self.game_data = data_source
        
        logger.info(f"Loaded {len(self.game_data)} games")
    
    def __len__(self):
        return len(self.game_data)
    
    def __getitem__(self, idx):
        """Get a single training example."""
        game_data = self.game_data[idx]
        
        # Convert to matrix format
        board_state, policy_target, value_target = convert_to_matrix_format(game_data)
        
        # Convert to tensors
        board_tensor = torch.FloatTensor(board_state)
        policy_tensor = torch.FloatTensor(policy_target)
        value_tensor = torch.FloatTensor([value_target])
        
        return board_tensor, policy_tensor, value_tensor


def create_sample_data(batch_size: int = 8) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sample data for testing purposes.
    
    Args:
        batch_size: Number of samples to create
        
    Returns:
        Tuple of (boards, policies, values) tensors
    """
    # Create random board states (2 channels for 2 players)
    boards = torch.randn(batch_size, NUM_PLAYERS, BOARD_SIZE, BOARD_SIZE)
    
    # Create random policy targets (169 possible moves)
    policies = torch.randn(batch_size, POLICY_OUTPUT_SIZE)
    policies = torch.softmax(policies, dim=1)  # Convert to probabilities
    
    # Create random value targets (single value per board)
    values = torch.randn(batch_size, VALUE_OUTPUT_SIZE)
    values = torch.sigmoid(values)  # Convert to [0, 1] range
    
    return boards, policies, values


def create_dataloader(data_dir: str, batch_size: int = 32, 
                     shuffle: bool = True, num_workers: int = 4,
                     augment: bool = True) -> DataLoader:
    """
    Create a DataLoader for the Hex dataset.
    
    Args:
        data_dir: Directory containing .trmph files
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes
        augment: Whether to apply data augmentation
        
    Returns:
        PyTorch DataLoader
    """
    dataset = HexDataset(data_dir, augment=augment)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )


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