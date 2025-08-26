"""
Board utility functions for Hex game operations.

This module provides utility functions for board operations, including
piece placement, validation, and conversion between different formats.
"""

import numpy as np
from hex_ai.config import BOARD_SIZE
from hex_ai.enums import Piece, char_to_piece, get_piece_display_symbol, piece_to_char


def get_piece_at(board_nxn: np.ndarray, row: int, col: int) -> Piece:
    """
    Get the piece at a specific position.
    
    Args:
        board_nxn: N×N board array
        row: Row index (0-indexed)
        col: Column index (0-indexed)
        
    Returns:
        Piece enum at the position
        
    Raises:
        IndexError: If coordinates are out of bounds
    """
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise IndexError(f"Position ({row}, {col}) is out of bounds")
    
    piece_char = board_nxn[row, col]
    return char_to_piece(piece_char)


def has_piece_at(board_nxn: np.ndarray, row: int, col: int, piece: Piece) -> bool:
    """
    Check if a specific piece is at a position.
    
    Args:
        board_nxn: N×N board array
        row: Row index (0-indexed)
        col: Column index (0-indexed)
        piece: Piece enum to check for
        
    Returns:
        True if the specified piece is at the position
    """
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        return False
    
    piece_value = board_nxn[row, col]
    return piece_value == piece_to_char(piece)


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
    
    return board_nxn[row, col] == piece_to_char(Piece.EMPTY)


def place_piece(board_nxn: np.ndarray, row: int, col: int, piece: Piece) -> np.ndarray:
    """
    Place a piece at the specified position.
    
    Args:
        board_nxn: N×N board array
        row: Row index (0-indexed)
        col: Column index (0-indexed)
        piece: Piece enum to place
        
    Returns:
        New board array with the piece placed
        
    Raises:
        ValueError: If position is out of bounds or already occupied
    """
    if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
        raise ValueError(f"Position ({row}, {col}) is out of bounds")
    
    if board_nxn[row, col] != piece_to_char(Piece.EMPTY):
        raise ValueError(f"Position ({row}, {col}) is already occupied")
    
    new_board = board_nxn.copy()
    new_board[row, col] = piece_to_char(piece)
    
    return new_board


def board_to_string(board_nxn: np.ndarray) -> str:
    """
    Convert board to a human-readable string representation.
    
    Args:
        board_nxn: N×N board array
        
    Returns:
        String representation with '.' for empty, 'B' for blue, 'R' for red
    """
    lines = []
    for row in range(BOARD_SIZE):
        line = " " * row  # Indent for hex shape
        for col in range(BOARD_SIZE):
            piece_char = board_nxn[row, col]
            piece = char_to_piece(piece_char)
            symbol = get_piece_display_symbol(piece)
            line += symbol + " "
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
    
    # Check that all values are valid ('e', 'b', or 'r')
    valid_values = {piece_to_char(Piece.EMPTY), piece_to_char(Piece.BLUE), piece_to_char(Piece.RED)}
    
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board_nxn[row, col] not in valid_values:
                return False
    
    return True


def count_pieces(board_nxn: np.ndarray) -> tuple[int, int]:
    """
    Count the number of blue and red pieces on the board.
    
    Args:
        board_nxn: N×N board array
        
    Returns:
        Tuple of (blue_count, red_count)
    """
    blue_count = np.sum(board_nxn == piece_to_char(Piece.BLUE))
    red_count = np.sum(board_nxn == piece_to_char(Piece.RED))
    
    return int(blue_count), int(red_count) 