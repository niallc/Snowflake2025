"""
Board visualization utilities for Hex AI.

This module provides utilities for visualizing Hex boards with policy overlays
and move highlighting.
"""

import numpy as np
from hex_ai.inference.board_display import display_hex_board
from hex_ai.utils.format_conversion import tensor_to_rowcol, tensor_to_trmph, board_2nxn_to_nxn


def decode_policy_target(policy):
    """Return (row, col, trmph_move) if policy is one-hot, else None."""
    if policy is not None and np.sum(policy) == 1.0:
        move_idx = int(np.argmax(policy))
        row, col = tensor_to_rowcol(move_idx)
        trmph_move = tensor_to_trmph(move_idx)
        return row, col, trmph_move
    return None


def visualize_board_with_policy(board, policy, file=None):
    """Display the board and highlight the policy target move if present."""
    highlight_move = None
    decoded = decode_policy_target(policy)
    if decoded is not None:
        row, col, trmph_move = decoded
        highlight_move = (row, col)
    # Use only the first two channels for display
    if board.shape[0] >= 2:
        display_board = board[:2]
    else:
        display_board = board
    display_hex_board(display_board, highlight_move=highlight_move, file=file)
    return highlight_move, decoded[2] if decoded is not None else None 