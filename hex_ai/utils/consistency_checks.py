"""
Consistency check utilities for Hex AI.

This module provides utilities for validating the consistency of training data,
including policy moves and player-to-move channels.
"""

import numpy as np
from hex_ai.utils.format_conversion import board_2nxn_to_nxn


def policy_on_empty_cell(board, highlight_move):
    """Return True if the policy target move is on an empty cell."""
    if highlight_move is None:
        return None
    board_nxn = board_2nxn_to_nxn(board[:2])
    return board_nxn[highlight_move] == 0


def player_to_move_channel_valid(player_channel):
    """Return True if player-to-move channel contains only 0.0 or 1.0."""
    unique_vals = np.unique(player_channel)
    return np.all((unique_vals == 0.0) | (unique_vals == 1.0)), unique_vals 