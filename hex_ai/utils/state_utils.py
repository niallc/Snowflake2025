"""
State utilities for Hex game states.

This module contains utility functions for working with Hex game states,
including hashing, validation, and other state-related operations.
"""

from typing import Tuple
from hex_ai.inference.game_engine import HexGameState


def state_hash_from(state: HexGameState) -> int:
    """
    Create a hash of a game state using only immutable, CPU-native parts.
    
    This function creates a stable hash that can be used for caching and
    state comparison. It only uses the move history and current player,
    avoiding any tensor data that could cause issues with hash randomization.
    
    Args:
        state: The Hex game state to hash
        
    Returns:
        A stable integer hash of the state
        
    Note:
        The hash is masked to 63 bits to avoid Python hash randomization effects.
    """
    # Use move history and current player enum value.
    # Ensure stable, bounded integer (mask to 63 bits to avoid Python hash randomization effects).
    key = (tuple(state.move_history), int(state.current_player_enum.value))
    h = hash(key) & ((1 << 63) - 1)
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
