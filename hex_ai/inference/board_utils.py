"""
Board format conversion utilities for Hex AI.

This module provides conversion functions between different board representations:
- 2×N×N tensor format: Used by neural networks (channels for blue/red)
- N×N array format: Used by game logic (0=empty, 1=blue, 2=red)

The N×N format is more convenient for game logic, adjacency checks, and debugging.
"""

import torch
import numpy as np
from typing import Tuple

# Board value constants for N×N format
from ..config import BOARD_SIZE, BLUE_PLAYER, RED_PLAYER, BLUE_PIECE, RED_PIECE, EMPTY_PIECE
from hex_ai.utils.format_conversion import (
    board_2nxn_to_nxn, board_nxn_to_2nxn
)


def get_piece_at(board_nxn: np.ndarray, row: int, col: int) -> int:
    """
    Get the piece at a given position in N×N format.
    
    Args:
        board_nxn: N×N board array
        row: Row index (0-indexed)
        col: Column index (0-indexed)
        
    Returns:
        Piece value: 0=empty, 1=blue, 2=red
        
    Raises:
        IndexError: If position is out of bounds
    """
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise IndexError(f"Position ({row}, {col}) is out of bounds")
    
    return board_nxn[row, col]


def has_piece_at(board_nxn: np.ndarray, row: int, col: int, color: str) -> bool:
    """
    Check if there's a piece of the given color at the specified position.
    
    Args:
        board_nxn: N×N board array
        row: Row index (0-indexed)
        col: Column index (0-indexed)
        color: "blue" or "red"
        
    Returns:
        True if the position has a piece of the specified color
    """
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return False
    
    piece_value = board_nxn[row, col]
    
    if color == "blue":
        return piece_value == BLUE_PIECE
    elif color == "red":
        return piece_value == RED_PIECE
    else:
        return False


def is_empty(board_nxn: np.ndarray, row: int, col: int) -> bool:
    """
    Check if a position is empty.
    
    Args:
        board_nxn: N×N board array
        row: Row index (0-indexed)
        col: Column index (0-indexed)
        
    Returns:
        True if the position is empty
    """
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return False
    
    return board_nxn[row, col] == EMPTY_PIECE


def place_piece(board_nxn: np.ndarray, row: int, col: int, color: str) -> np.ndarray:
    """
    Place a piece at the specified position.
    
    Args:
        board_nxn: N×N board array
        row: Row index (0-indexed)
        col: Column index (0-indexed)
        color: "blue" or "red"
        
    Returns:
        New board array with the piece placed
        
    Raises:
        ValueError: If position is out of bounds or already occupied
    """
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"Position ({row}, {col}) is out of bounds")
    
    if board_nxn[row, col] != EMPTY_PIECE:
        raise ValueError(f"Position ({row}, {col}) is already occupied")
    
    new_board = board_nxn.copy()
    
    if color == "blue":
        new_board[row, col] = BLUE_PIECE
    elif color == "red":
        new_board[row, col] = RED_PIECE
    else:
        raise ValueError(f"Invalid color: {color}")
    
    return new_board


def board_to_string(board_nxn: np.ndarray) -> str:
    """
    Convert board to a human-readable string representation.
    
    Args:
        board_nxn: N×N board array
        
    Returns:
        String representation with '.' for empty, 'B' for blue, 'R' for red
    """
    symbols = {EMPTY_PIECE: '.', BLUE_PIECE: 'B', RED_PIECE: 'R'}
    
    lines = []
    for row in range(BOARD_SIZE):
        line = " " * row  # Indent for hex shape
        for col in range(BOARD_SIZE):
            line += symbols[board_nxn[row, col]] + " "
        lines.append(line)
    
    return "\n".join(lines)


def validate_board(board_nxn: np.ndarray) -> bool:
    """
    Validate that a board array has correct values.
    
    Args:
        board_nxn: N×N board array
        
    Returns:
        True if the board is valid
    """
    if board_nxn.shape != (BOARD_SIZE, BOARD_SIZE):
        return False
    
    # Check that all values are valid (0, 1, or 2)
    valid_values = {EMPTY_PIECE, BLUE_PIECE, RED_PIECE}
    return np.all(np.isin(board_nxn, list(valid_values)))


def count_pieces(board_nxn: np.ndarray) -> Tuple[int, int]:
    """
    Count the number of blue and red pieces on the board.
    
    Args:
        board_nxn: N×N board array
        
    Returns:
        Tuple of (blue_count, red_count)
    """
    blue_count = np.sum(board_nxn == BLUE_PIECE)
    red_count = np.sum(board_nxn == RED_PIECE)
    return int(blue_count), int(red_count) 