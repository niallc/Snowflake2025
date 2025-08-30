"""
Centralized enum definitions for Hex AI semantic types.

This module is the single source of truth for representing players, pieces,
channels, winners, and value perspectives. Other modules should import these
Enums rather than duplicating constants.
"""

from enum import Enum


class StrictEnum(Enum):
    """Base class for enums that prevent cross-type comparisons."""
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            raise TypeError(f"Cannot compare {self.__class__.__name__} with {type(other).__name__}")
        return super().__eq__(other)
    
    def __hash__(self):
        """Make enums hashable so they can be used as dictionary keys."""
        return hash(self.value)


class Winner(StrictEnum):
    BLUE = 0
    RED = 1


class Player(StrictEnum):
    """Player constants for game logic and player-to-move channel."""
    BLUE = 0
    RED = 1


class Piece(StrictEnum):
    """Piece constants for N×N board representation (character encoding)."""
    EMPTY = "e"
    BLUE = "b"
    RED = "r"


class Channel(StrictEnum):
    """Channel indices for one-hot encoded board formats."""
    BLUE = 0
    RED = 1
    PLAYER_TO_MOVE = 2


class ValuePerspective(StrictEnum):
    TRAINING_TARGET = 0   # 0.0 = Blue win, 1.0 = Red win
    BLUE_WIN_PROB = 1     # Probability Blue wins
    RED_WIN_PROB = 2      # Probability Red wins


# ============================================================================
# Helper Functions for Enum-Primitive Conversion
# ============================================================================

def piece_to_char(piece: Piece) -> str:
    """Convert Piece enum to character representation."""
    return piece.value


def char_to_piece(char: str) -> Piece:
    """Convert character to Piece enum."""
    mapping = {"e": Piece.EMPTY, "b": Piece.BLUE, "r": Piece.RED}
    if char not in mapping:
        raise ValueError(f"Invalid piece character: {char}")
    return mapping[char]


def player_to_int(player: Player) -> int:
    """Convert Player enum to integer representation."""
    return player.value


def int_to_player(player_int: int) -> Player:
    """Convert integer to Player enum."""
    if player_int not in (Player.BLUE.value, Player.RED.value):
        raise ValueError(f"Invalid player integer: {player_int}")
    return Player(player_int)


def channel_to_int(channel: Channel) -> int:
    """Convert Channel enum to integer representation."""
    return channel.value


def int_to_channel(channel_int: int) -> Channel:
    """Convert integer to Channel enum."""
    if channel_int not in (Channel.BLUE.value, Channel.RED.value, Channel.PLAYER_TO_MOVE.value):
        raise ValueError(f"Invalid channel integer: {channel_int}")
    return Channel(channel_int)


# ============================================================================
# Board Creation Helpers
# ============================================================================

def create_empty_board(board_size: int) -> str:
    """Create an empty board string representation."""
    return Piece.EMPTY.value


def create_empty_board_array(board_size: int) -> str:
    """Create an empty board array with proper dtype."""
    return Piece.EMPTY.value


# ============================================================================
# Display Helpers
# ============================================================================

def get_piece_display_symbol(piece: Piece) -> str:
    """Get the display symbol for a piece."""
    symbols = {
        Piece.EMPTY: ".",
        Piece.BLUE: "B", 
        Piece.RED: "R"
    }
    return symbols[piece]


def get_piece_unicode_symbol(piece: Piece) -> str:
    """Get the unicode symbol for a piece."""
    symbols = {
        Piece.EMPTY: "◯",
        Piece.BLUE: "●",
        Piece.RED: "●"
    }
    return symbols[piece]


