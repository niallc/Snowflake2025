"""
Utility functions for the Hex AI project.

This module contains helper functions for data processing, model utilities,
and other common operations used throughout the project.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import logging

from .config import BOARD_SIZE, NUM_PLAYERS


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Set up logging configuration for the project.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model: torch.nn.Module, 
                   optimizer: torch.optim.Optimizer,
                   epoch: int,
                   loss: float,
                   filepath: str):
    """
    Save a model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        filepath: Path to save the checkpoint
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)


def load_checkpoint(model: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   filepath: str) -> Tuple[int, float]:
    """
    Load a model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        filepath: Path to the checkpoint file
        
    Returns:
        Tuple of (epoch, loss)
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def board_to_tensor(board: np.ndarray) -> torch.Tensor:
    """
    Convert a board array to a PyTorch tensor.
    
    Args:
        board: Numpy array representing the board
        
    Returns:
        PyTorch tensor of shape (2, 13, 13)
    """
    # TODO: Implement board conversion
    # Should handle different input formats and convert to standard tensor format
    pass


def tensor_to_board(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor back to a board array.
    
    Args:
        tensor: PyTorch tensor of shape (2, 13, 13)
        
    Returns:
        Numpy array representing the board
    """
    # TODO: Implement tensor to board conversion
    pass


def validate_board_shape(tensor: torch.Tensor) -> bool:
    """
    Validate that a tensor has the correct board shape.
    
    Args:
        tensor: Tensor to validate
        
    Returns:
        True if shape is correct, False otherwise
    """
    expected_shape = (NUM_PLAYERS, BOARD_SIZE, BOARD_SIZE)
    return tensor.shape == expected_shape


def create_sample_data(num_samples: int = 100) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create sample data for testing.
    
    Args:
        num_samples: Number of samples to create
        
    Returns:
        Tuple of (boards, policies, values) for testing
    """
    boards = torch.randn(num_samples, NUM_PLAYERS, BOARD_SIZE, BOARD_SIZE)
    policies = torch.randn(num_samples, BOARD_SIZE * BOARD_SIZE)
    values = torch.randn(num_samples, 1)
    
    return boards, policies, values


def get_device() -> torch.device:
    """
    Get the appropriate device for training.
    
    Returns:
        torch.device (cuda if available, else cpu)
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed) 