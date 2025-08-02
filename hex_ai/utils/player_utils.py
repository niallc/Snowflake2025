"""
Player utilities for Hex AI.

This module contains player-related utilities that are used across multiple modules
to avoid circular imports.
"""

import numpy as np
from hex_ai.value_utils import Player


def get_player_to_move_from_board(board_2ch: np.ndarray, error_tracker=None) -> Player:
    """
    Given a (2, N, N) board, return Player.BLUE if it's blue's move, Player.RED if it's red's move.
    Uses error tracking to handle invalid board states gracefully.
    
    Args:
        board_2ch: np.ndarray of shape (2, N, N), blue and red channels
        error_tracker: Optional BoardStateErrorTracker instance
        
    Returns:
        Player: Player.BLUE or Player.RED
        
    Raises:
        ValueError: If board has invalid state and no error_tracker provided
    """
    if board_2ch.shape[0] != 2:
        raise ValueError(f"Expected board with 2 channels, got shape {board_2ch.shape}")
    
    blue_count = int(np.sum(board_2ch[0]))
    red_count = int(np.sum(board_2ch[1]))
    
    if blue_count == red_count:
        return Player.BLUE
    elif blue_count == red_count + 1:
        return Player.RED
    else:
        # Invalid board state - use error tracking if available
        error_msg = f"Invalid board state: blue_count={blue_count}, red_count={red_count}. Board must have equal or one more blue than red."
        
        if error_tracker is not None:
            error_tracker.record_error(
                board_state=board_2ch,
                error_msg=error_msg,
                file_info=getattr(error_tracker, '_current_file', "Unknown"),
                sample_info=getattr(error_tracker, '_current_sample', "Unknown")
            )
            # Return a default value to continue processing
            # Assume it's blue's turn if we can't determine
            return Player.BLUE
        else:
            # Fall back to original behavior if no error tracker
            raise ValueError(error_msg)