"""
State utilities for Hex game states.

This module contains utility functions for working with Hex game states,
including hashing, validation, and other state-related operations.
"""

from typing import Tuple
from hex_ai.inference.game_engine import HexGameState


def board_key(state: HexGameState) -> int:
    """
    Create a hash key based on the canonical board state and side to move.
    
    This function creates a stable hash that can be used for caching neural network
    evaluations. It keys on the board state (stone positions) and current player,
    ignoring the move history path. This allows cache hits when the same position
    is reached via different move orders.
    
    Args:
        state: The Hex game state to hash
        
    Returns:
        A stable integer hash of the board state and current player
        
    Note:
        The hash is masked to 63 bits to avoid Python hash randomization effects.
    """
    # Convert board to a canonical representation
    # Board is numpy array with string values: "e" (empty), "b" (blue), "r" (red)
    board_tuple = tuple(state.board.flatten())  # Flatten to 1D tuple for hashing
    
    # Include current player (side to move)
    current_player = int(state.current_player_enum.value)
    
    # Create hash key from board state and current player
    key = (board_tuple, current_player)
    h = hash(key) & ((1 << 63) - 1)  # Mask to 63 bits to avoid Python hash randomization
    return h


def validate_move_coordinates(row: int, col: int, board_size: int) -> None:
    """
    Validate that move coordinates are within the board bounds.
    
    Args:
        row: Row coordinate (0-indexed)
        col: Column coordinate (0-indexed)
        board_size: Size of the board (e.g., 13 for 13x13)
        
    Raises:
        ValueError: If coordinates are out of bounds
    """
    if not (0 <= row < board_size):
        raise ValueError(f"Row coordinate {row} is out of bounds for board size {board_size}")
    if not (0 <= col < board_size):
        raise ValueError(f"Column coordinate {col} is out of bounds for board size {board_size}")


def is_valid_move_coordinates(row: int, col: int, board_size: int) -> bool:
    """
    Check if move coordinates are within the board bounds.
    
    Args:
        row: Row coordinate (0-indexed)
        col: Column coordinate (0-indexed)
        board_size: Size of the board (e.g., 13 for 13x13)
        
    Returns:
        True if coordinates are valid, False otherwise
    """
    return 0 <= row < board_size and 0 <= col < board_size
