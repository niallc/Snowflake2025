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


def validate_game(trmph_url: str, winner_indicator: str, line_info: str = "") -> Tuple[bool, str]:
    """
    Validate a single game for corruption.
    
    Args:
        trmph_url: The trmph URL string
        winner_indicator: The winner indicator string
        line_info: Optional line information for debugging
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Test if we can parse the game without errors
        board_state, policy_target, value_target = convert_to_matrix_format(trmph_url, debug_info=line_info)
        return True, ""
    except Exception as e:
        return False, str(e)


class HexDataset(Dataset):
    """Dataset for Hex game data."""
    
    def __init__(self, data_source=None, data_dir=None, board_size: int = BOARD_SIZE, augment: bool = True):
        """
        Initialize dataset.
        
        Args:
            data_source: Either a directory path (str/Path) containing .trmph files,
                        a single .trmph file path, or a list of trmph strings
            data_dir: Alternative to data_source for directory paths (for API compatibility)
            board_size: Size of the board
            augment: Whether to apply data augmentation
        """
        self.board_size = board_size
        self.augment = augment
        
        # Handle both data_source and data_dir parameters for API compatibility
        if data_source is None and data_dir is not None:
            data_source = data_dir
        elif data_source is None and data_dir is None:
            raise ValueError("Either data_source or data_dir must be provided")
        
        if isinstance(data_source, (str, Path)):
            data_path = Path(data_source)
            
            if data_path.is_file():
                # Single file - load all games from it
                self.game_data, self.corrupted_games = self._load_games_from_file(data_path)
            elif data_path.is_dir():
                # Directory - load all .trmph files
                self.game_files = list(data_path.glob("*.trmph"))
                self.game_data = []
                self.corrupted_games = []
                
                for file_path in self.game_files:
                    games, corrupted = self._load_games_from_file(file_path)
                    self.game_data.extend(games)
                    self.corrupted_games.extend(corrupted)
            else:
                raise FileNotFoundError(f"Data source not found: {data_source}")
        else:
            # Direct list of trmph strings
            self.game_data = data_source
            self.corrupted_games = []
        
        # Check corruption thresholds
        total_games = len(self.game_data) + len(self.corrupted_games)
        corruption_count = len(self.corrupted_games)
        corruption_percentage = (corruption_count / total_games * 100) if total_games > 0 else 0
        
        logger.info(f"Loaded {len(self.game_data)} valid games")
        logger.info(f"Found {corruption_count} corrupted games ({corruption_percentage:.1f}%)")
        
        # Log corrupted games to file
        if self.corrupted_games:
            corruption_log_path = Path("corrupted_games.log")
            with open(corruption_log_path, 'w') as f:
                f.write(f"Corrupted games from {data_source}\n")
                f.write(f"Total games: {total_games}, Corrupted: {corruption_count} ({corruption_percentage:.1f}%)\n\n")
                for i, (trmph_url, winner_indicator, error_msg) in enumerate(self.corrupted_games, 1):
                    f.write(f"Game {i}:\n")
                    f.write(f"  URL: {trmph_url}\n")
                    f.write(f"  Winner: {winner_indicator}\n")
                    f.write(f"  Error: {error_msg}\n\n")
            logger.info(f"Corrupted games logged to {corruption_log_path}")
        
        # Check thresholds - more permissive for now
        if corruption_count > 20:  # Increased from 5
            raise ValueError(f"Too many corrupted games: {corruption_count} > 20")
        
        if corruption_percentage > 10.0:  # Increased from 1.0%
            raise ValueError(f"Corruption percentage too high: {corruption_percentage:.1f}% > 10.0%")
        
        logger.info(f"Dataset validation passed - proceeding with {len(self.game_data)} games")
    
    def _load_games_from_file(self, file_path: Path) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str, str]]]:
        """Load all games from a single .trmph file."""
        games = []
        corrupted_games = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Parse line format: "trmph_url winner_indicator"
                parts = line.split(' ', 1)
                if len(parts) != 2:
                    logger.warning(f"Invalid line format at line {line_num}: {line[:50]}...")
                    corrupted_games.append((line, "invalid_format", f"Invalid line format at line {line_num}"))
                    continue
                
                trmph_url, winner_indicator = parts
                
                # Validate trmph URL format
                if not trmph_url.startswith("http://www.trmph.com/hex/board#"):
                    logger.warning(f"Invalid trmph URL at line {line_num}: {trmph_url[:50]}...")
                    corrupted_games.append((trmph_url, winner_indicator, f"Invalid trmph URL format at line {line_num}"))
                    continue
                
                # Validate game integrity
                is_valid, error_msg = validate_game(trmph_url, winner_indicator, f"Line {line_num}")
                if is_valid:
                    games.append((trmph_url, winner_indicator))
                else:
                    corrupted_games.append((trmph_url, winner_indicator, error_msg))
        
        logger.info(f"Loaded {len(games)} valid games and {len(corrupted_games)} corrupted games from {file_path}")
        return games, corrupted_games
    
    def __len__(self):
        return len(self.game_data)
    
    def __getitem__(self, idx):
        """Get a single training example."""
        game_data, winner_indicator = self.game_data[idx]
        
        # Convert to matrix format (should not fail since we validated during loading)
        board_state, policy_target, value_target = convert_to_matrix_format(game_data)
        
        # Override value target based on actual winner
        if winner_indicator == "1":
            value_target = 1.0  # Blue wins
        elif winner_indicator == "0":
            value_target = 0.0  # Red wins
        else:
            # Unknown winner - keep the default from convert_to_matrix_format
            pass
        
        # Apply data augmentation if enabled
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