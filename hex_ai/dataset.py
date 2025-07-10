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

from .config import BOARD_SIZE, NUM_PLAYERS, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE
from .data_utils import load_trmph_file, convert_to_matrix_format, augment_board


class HexDataset(Dataset):
    """
    PyTorch Dataset for Hex game data.
    
    This dataset loads .trmph files and converts them to the format
    expected by the neural network model.
    """
    
    def __init__(self, data_dir: str, augment: bool = True):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing .trmph files
            augment: Whether to apply data augmentation
        """
        self.data_dir = Path(data_dir)
        self.augment = augment
        
        # Find all .trmph files
        self.files = list(self.data_dir.glob("*.trmph"))
        
        if not self.files:
            raise ValueError(f"No .trmph files found in {data_dir}")
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.files)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (board_state, policy_target, value_target)
        """
        # Load the .trmph file
        file_path = self.files[idx]
        game_data = load_trmph_file(str(file_path))
        
        # Convert to matrix format
        board_state, policy_target, value_target = convert_to_matrix_format(game_data)
        
        # Apply augmentation if enabled
        if self.augment:
            board_state, policy_target = augment_board(board_state, policy_target)
        
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