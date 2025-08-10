"""
Centralized enum definitions for Hex AI semantic types.

This module is the single source of truth for representing players, pieces,
channels, winners, and value perspectives. Other modules should import these
Enums rather than duplicating constants.
"""

from enum import Enum


class Winner(Enum):
    BLUE = 0
    RED = 1


class Player(Enum):
    """Player constants for game logic and player-to-move channel."""
    BLUE = 0
    RED = 1


class Piece(Enum):
    """Piece constants for N×N board representation (character encoding)."""
    EMPTY = "e"
    BLUE = "b"
    RED = "r"


class Channel(Enum):
    """Channel indices for one-hot encoded board formats."""
    BLUE = 0
    RED = 1
    PLAYER_TO_MOVE = 2


class ValuePerspective(Enum):
    TRAINING_TARGET = 0   # 0.0 = Blue win, 1.0 = Red win
    BLUE_WIN_PROB = 1     # Probability Blue wins
    RED_WIN_PROB = 2      # Probability Red wins


