"""
Data processing utilities for Hex AI.

This module contains functions for converting between different Hex data formats,
applying data augmentation, and preparing data for training. It provides a modern,
typed interface to the legacy conversion functions.

Key functions:
- File format conversion (trmph ↔ tensor)
- Data augmentation (rotation, reflection)
- Training data preparation
- Coordinate system conversions
"""

import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
import logging
import re
import string

from .config import BOARD_SIZE, NUM_PLAYERS, TRMPH_EXTENSION, POLICY_OUTPUT_SIZE

logger = logging.getLogger(__name__)

TRMPH_BOARD_PATTERN = re.compile(r"#(\d+),")
LETTERS = string.ascii_lowercase


# ============================================================================
# File Format Conversion Functions
# ============================================================================

def load_trmph_file(file_path: str) -> str:
    """
    Load a single .trmph file.
    
    Args:
        file_path: Path to the .trmph file
        
    Returns:
        Trmph string content
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty or invalid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        if not content:
            raise ValueError(f"Empty file: {file_path}")
        
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        raise ValueError(f"Error reading file {file_path}: {e}")


def convert_to_matrix_format(game_data: str) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Convert game data to matrix format for training.
    
    Args:
        game_data: Trmph string representing a game
        
    Returns:
        Tuple of (board_state, policy_target, value_target):
        - board_state: Shape (2, 13, 13) - current board state
        - policy_target: Shape (169,) - move probabilities
        - value_target: Shape (1,) - win probability (0.0 or 1.0)
    """
    # For now, create placeholder data
    # TODO: Implement actual conversion from trmph to matrix format
    
    # Create a random board state (2 channels for 2 players)
    board_state = np.random.randint(0, 2, size=(2, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    board_state = board_state.astype(np.float32)
    
    # Create random policy target (169 possible moves)
    policy_target = np.random.rand(POLICY_OUTPUT_SIZE).astype(np.float32)
    policy_target = policy_target / policy_target.sum()  # Normalize to probabilities
    
    # Create random value target (0.0 or 1.0 for win/loss)
    value_target = float(np.random.choice([0.0, 1.0]))
    
    return board_state, policy_target, value_target


def augment_board(board: np.ndarray, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to board and policy.
    
    Args:
        board: Board state of shape (2, 13, 13)
        policy: Policy target of shape (169,)
        
    Returns:
        Tuple of (augmented_board, augmented_policy)
    """
    # For now, return the original data
    # TODO: Implement actual augmentation (rotation, reflection)
    return board, policy


def load_trmph_files(data_dir: str) -> List[str]:
    """
    Load all .trmph files from a directory.
    
    Args:
        data_dir: Path to directory containing .trmph files
        
    Returns:
        List of trmph strings (with preamble)
        
    Raises:
        FileNotFoundError: If directory doesn't exist
        ValueError: If no .trmph files found
    """
    # TODO: Implement file loading
    # Should:
    # - Scan directory for .trmph files
    # - Read file contents
    # - Handle encoding issues
    # - Return list of trmph strings
    pass


def strip_trmph_preamble(trmph_text: str) -> str:
    """
    Remove the preamble from a trmph string (e.g., 'http://...#13,a1b2c3' -> 'a1b2c3').
    """
    match = TRMPH_BOARD_PATTERN.search(trmph_text)
    if not match:
        raise ValueError(f"No board preamble found in trmph string: {trmph_text}")
    return trmph_text[match.end():]


def split_trmph_moves(bare_moves: str) -> list[str]:
    """
    Split a bare trmph move string into a list of moves (e.g., 'a1b2c3' -> ['a1','b2','c3']).
    """
    moves = []
    i = 0
    while i < len(bare_moves):
        if bare_moves[i] not in LETTERS:
            raise ValueError(f"Expected letter at position {i} in {bare_moves}")
        j = i + 1
        while j < len(bare_moves) and bare_moves[j].isdigit():
            j += 1
        moves.append(bare_moves[i:j])
        i = j
    return moves


def trmph_move_to_rowcol(move: str, board_size: int = BOARD_SIZE) -> tuple[int, int]:
    """
    Convert a trmph move (e.g., 'a1') to (row, col) coordinates (0-indexed).
    """
    if len(move) < 2 or len(move) > 4:
        raise ValueError(f"Invalid trmph move: {move}")
    letter = move[0]
    number = int(move[1:])
    if letter not in LETTERS[:board_size]:
        raise ValueError(f"Invalid letter in move: {move}")
    if not (1 <= number <= board_size):
        raise ValueError(f"Invalid number in move: {move}")
    row = number - 1
    col = LETTERS.index(letter)
    return row, col


def parse_trmph_to_board(trmph_text: str, board_size: int = BOARD_SIZE) -> np.ndarray:
    """
    Parse a trmph string to a board matrix (0=empty, 1=blue, 2=red).
    """
    bare_moves = strip_trmph_preamble(trmph_text)
    moves = split_trmph_moves(bare_moves)
    board = np.zeros((board_size, board_size), dtype=np.int8)
    for i, move in enumerate(moves):
        row, col = trmph_move_to_rowcol(move, board_size)
        color = 1 if i % 2 == 0 else 2  # Blue starts
        if board[row, col] != 0:
            raise ValueError(f"Duplicate move at {(row, col)} in {trmph_text}")
        board[row, col] = color
    return board


def trmph_to_dothex(trmph_text: str) -> np.ndarray:
    """
    Convert trmph format to DotHex matrix format.
    
    Args:
        trmph_text: Trmph string (without preamble)
        
    Returns:
        DotHex matrix: shape (13, 13), values 0/1/2
        
    Example:
        "a1b2c3" → [[1, 0, 0, ...], [0, 2, 0, ...], ...]
    """
    # TODO: Implement conversion
    # Should use legacy trmphToDotHex() logic
    pass


def dothex_to_tensor(dothex_board: np.ndarray) -> torch.Tensor:
    """
    Convert DotHex matrix to PyTorch tensor format.
    
    Args:
        dothex_board: DotHex matrix (13, 13)
        
    Returns:
        Tensor: shape (2, 13, 13)
        - Channel 0: Blue player positions
        - Channel 1: Red player positions
        
    Example:
        [[1, 0, 0], [0, 2, 0], ...] → 
        [[[1, 0, 0], [0, 0, 0], ...],  # Blue channel
         [[0, 0, 0], [0, 1, 0], ...]]  # Red channel
    """
    # TODO: Implement conversion
    # Should convert 0/1/2 matrix to 2-channel tensor
    pass


def trmph_to_tensor(trmph_text: str) -> torch.Tensor:
    """
    Convert trmph format directly to PyTorch tensor.
    
    Args:
        trmph_text: Trmph string (with or without preamble)
        
    Returns:
        Tensor: shape (2, 13, 13)
    """
    # TODO: Implement direct conversion
    # Should combine trmph_to_dothex() + dothex_to_tensor()
    pass


# ============================================================================
# Coordinate System Conversions
# ============================================================================

def trmph_to_rowcol(trmph_move: str) -> Tuple[int, int]:
    """
    Convert trmph move to (row, col) coordinates.
    
    Args:
        trmph_move: Single move (e.g., "a1", "m13")
        
    Returns:
        (row, col) coordinates (0-indexed)
        
    Example:
        "a1" → (0, 0)
        "m13" → (12, 12)
    """
    # TODO: Implement conversion
    # Should use legacy LookUpRowCol() logic
    pass


def rowcol_to_trmph(row: int, col: int) -> str:
    """
    Convert (row, col) coordinates to trmph move.
    
    Args:
        row: Row index (0-12)
        col: Column index (0-12)
        
    Returns:
        Trmph move string
        
    Example:
        (0, 0) → "a1"
        (12, 12) → "m13"
    """
    # TODO: Implement conversion
    # Should use legacy RowColToTrmph() logic
    pass


def tensor_to_rowcol(tensor_pos: int) -> Tuple[int, int]:
    """
    Convert tensor position index to (row, col).
    
    Args:
        tensor_pos: Position in flattened tensor (0-168)
        
    Returns:
        (row, col) coordinates
    """
    # TODO: Implement conversion
    # Should convert linear index to 2D coordinates
    pass


def rowcol_to_tensor(row: int, col: int) -> int:
    """
    Convert (row, col) to tensor position index.
    
    Args:
        row: Row index (0-12)
        col: Column index (0-12)
        
    Returns:
        Position in flattened tensor (0-168)
    """
    # TODO: Implement conversion
    # Should convert 2D coordinates to linear index
    pass


# ============================================================================
# Data Augmentation Functions
# ============================================================================

def rotate_board_180(board: np.ndarray) -> np.ndarray:
    """
    Rotate board 180 degrees and swap player colors.
    
    This preserves the logical game state under the swap rule (π-rule).
    Blue pieces become red and vice versa, while the board is rotated.
    
    Args:
        board: Board array of shape (2, 13, 13) or (13, 13)
        
    Returns:
        Rotated and color-swapped board
    """
    if board.ndim == 3:
        # 2-channel format: (2, 13, 13)
        rotated = np.flip(board, axis=(1, 2))  # Rotate 180°
        # Swap channels (blue <-> red)
        rotated = rotated[::-1]
        return rotated
    else:
        # Single channel format: (13, 13) with values 0/1/2
        rotated = np.flip(board, axis=(0, 1))  # Rotate 180°
        # Swap colors: 1 <-> 2
        rotated = np.where(rotated == 1, 2, np.where(rotated == 2, 1, rotated))
        return rotated


def reflect_board_long_diagonal(board: np.ndarray) -> np.ndarray:
    """
    Reflect board along the long diagonal (top-left to bottom-right) and swap colors.
    
    This preserves the logical game state under the swap rule.
    
    Args:
        board: Board array of shape (2, 13, 13) or (13, 13)
        
    Returns:
        Reflected and color-swapped board
    """
    if board.ndim == 3:
        # 2-channel format: (2, 13, 13)
        reflected = np.transpose(board, (0, 2, 1))  # Transpose
        # Swap channels (blue <-> red)
        reflected = reflected[::-1]
        return reflected
    else:
        # Single channel format: (13, 13) with values 0/1/2
        reflected = np.transpose(board)  # Transpose
        # Swap colors: 1 <-> 2
        reflected = np.where(reflected == 1, 2, np.where(reflected == 2, 1, reflected))
        return reflected


def reflect_board_short_diagonal(board: np.ndarray) -> np.ndarray:
    """
    Reflect board along the short diagonal (top-right to bottom-left) and swap colors.
    
    This preserves the logical game state under the swap rule.
    
    Args:
        board: Board array of shape (2, 13, 13) or (13, 13)
        
    Returns:
        Reflected and color-swapped board
    """
    if board.ndim == 3:
        # 2-channel format: (2, 13, 13)
        # Short diagonal reflection = flip both axes
        reflected = np.flip(board, axis=(1, 2))
        # Swap channels (blue <-> red)
        reflected = reflected[::-1]
        return reflected
    else:
        # Single channel format: (13, 13) with values 0/1/2
        # Short diagonal reflection = flip both axes
        reflected = np.flip(board, axis=(0, 1))
        # Swap colors: 1 <-> 2
        reflected = np.where(reflected == 1, 2, np.where(reflected == 2, 1, reflected))
        return reflected


def augment_board(board: np.ndarray, policy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply data augmentation to board and policy.
    
    Creates 4 augmented versions: original, 180° rotation, long diagonal reflection,
    and short diagonal reflection. All preserve the logical game state under the swap rule.
    
    Args:
        board: Board array of shape (2, 13, 13) or (13, 13)
        policy: Policy array of shape (169,)
        
    Returns:
        Tuple of (augmented_board, augmented_policy)
    """
    # For now, return the original data
    # TODO: Implement random selection of one of the 4 augmentations
    return board, policy


def create_augmented_boards(board: np.ndarray) -> list[np.ndarray]:
    """
    Create all 4 augmented versions of a board.
    
    Args:
        board: Board array of shape (2, 13, 13) or (13, 13)
        
    Returns:
        List of 4 boards: [original, rotated_180, reflected_long, reflected_short]
    """
    rotated = rotate_board_180(board)
    reflected_long = reflect_board_long_diagonal(board)
    reflected_short = reflect_board_short_diagonal(board)
    
    return [board, rotated, reflected_long, reflected_short]


def create_augmented_policies(policy: np.ndarray) -> list[np.ndarray]:
    """
    Create policy arrays corresponding to the 4 board augmentations.
    
    Args:
        policy: Policy array of shape (169,)
        
    Returns:
        List of 4 policies corresponding to the board augmentations
    """
    # Reshape policy to (13, 13) for easier manipulation
    policy_2d = policy.reshape(13, 13)
    
    # Create the 4 augmented policies
    rotated = np.flip(policy_2d, axis=(0, 1))  # 180° rotation
    reflected_long = np.transpose(policy_2d)  # Long diagonal reflection
    reflected_short = np.flip(np.transpose(policy_2d), axis=(0, 1))  # Short diagonal reflection
    
    # Reshape back to (169,)
    return [
        policy,
        rotated.reshape(169),
        reflected_long.reshape(169),
        reflected_short.reshape(169)
    ]


# ============================================================================
# Training Data Preparation
# ============================================================================

def extract_game_positions(trmph_text: str, 
                          positions: str = "all",
                          sample_size: int = 4) -> List[torch.Tensor]:
    """
    Extract board positions from a game.
    
    Args:
        trmph_text: Complete game in trmph format
        positions: Which positions to extract ("all", "last", "sampled")
        sample_size: Number of positions to sample (if "sampled")
        
    Returns:
        List of board tensors
    """
    # TODO: Implement position extraction
    # Should:
    # - Parse game moves
    # - Build board state after each move
    # - Extract requested positions
    # - Convert to tensors
    pass


def prepare_training_data(games: List[str],
                         augment: bool = True,
                         positions: str = "all") -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Prepare training data from games.
    
    Args:
        games: List of trmph game strings
        augment: Whether to apply data augmentation
        positions: Which positions to extract
        
    Returns:
        List of (board, policy_target, value_target) tuples
    """
    # TODO: Implement training data preparation
    # Should:
    # - Extract positions from games
    # - Apply augmentation if requested
    # - Generate policy/value targets
    # - Return training tuples
    pass


# ============================================================================
# Validation and Testing Functions
# ============================================================================

def validate_trmph_format(trmph_text: str) -> bool:
    """
    Validate that a trmph string is properly formatted.
    
    Args:
        trmph_text: Trmph string to validate
        
    Returns:
        True if valid, False otherwise
    """
    # TODO: Implement validation
    # Should check:
    # - Proper preamble
    # - Valid move format
    # - Consistent board size
    pass


def validate_board_tensor(board: torch.Tensor) -> bool:
    """
    Validate that a board tensor has correct format.
    
    Args:
        board: Tensor to validate
        
    Returns:
        True if valid, False otherwise
    """
    # TODO: Implement validation
    # Should check:
    # - Correct shape (2, 13, 13)
    # - Binary values (0 or 1)
    # - No overlapping pieces
    pass


def test_conversion_roundtrip() -> bool:
    """
    Test that conversions work correctly in both directions.
    
    Returns:
        True if all tests pass
    """
    # TODO: Implement roundtrip tests
    # Should test:
    # - trmph ↔ tensor
    # - tensor ↔ rowcol
    # - augmentation reversibility
    pass


# ============================================================================
# Performance Optimization Functions
# ============================================================================

def batch_convert_trmph(trmph_strings: List[str]) -> torch.Tensor:
    """
    Convert multiple trmph strings efficiently.
    
    Args:
        trmph_strings: List of trmph strings
        
    Returns:
        Batch tensor: shape (batch_size, 2, 13, 13)
    """
    # TODO: Implement batch conversion
    # Should be more efficient than individual conversions
    pass


def preprocess_and_cache(data_dir: str, 
                        cache_dir: str,
                        force_reprocess: bool = False) -> str:
    """
    Preprocess all data and cache for faster loading.
    
    Args:
        data_dir: Directory with .trmph files
        cache_dir: Directory to store processed data
        force_reprocess: Whether to reprocess existing cache
        
    Returns:
        Path to cached data
    """
    # TODO: Implement preprocessing and caching
    # Should:
    # - Convert all games to tensors
    # - Apply augmentations
    # - Save in efficient format (e.g., .npy)
    # - Return path to cached data
    pass 