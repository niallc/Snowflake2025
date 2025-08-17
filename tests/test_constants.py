"""
Test centralized constants from hex_ai.config.

This module tests that all player and piece constants are properly defined
and accessible from the centralized config module.
"""

import pytest
import numpy as np
from hex_ai.enums import Piece, Channel, Player, piece_to_char, player_to_int, channel_to_int
from hex_ai.config import BOARD_SIZE, POLICY_OUTPUT_SIZE, TRMPH_BLUE_WIN, TRMPH_RED_WIN, TRAINING_BLUE_WIN, TRAINING_RED_WIN


class TestPlayerConstants:
    """Test player constants for player-to-move channel and game logic."""
    
    def test_player_constants_defined(self):
        """Test that player constants are properly defined."""
        assert Player.BLUE.value == 0
        assert Player.RED.value == 1
        assert Player.BLUE != Player.RED
    
    def test_player_constants_integer(self):
        """Test that player constants are integers."""
        assert isinstance(Player.BLUE.value, int)
        assert isinstance(Player.RED.value, int)
    
    def test_player_helper_functions(self):
        """Test that player helper functions work correctly."""
        assert player_to_int(Player.BLUE) == 0
        assert player_to_int(Player.RED) == 1
        assert channel_to_int(Channel.BLUE) == 0
        assert channel_to_int(Channel.RED) == 1
        assert channel_to_int(Channel.PLAYER_TO_MOVE) == 2


class TestPieceConstants:
    """Test piece constants for board representation."""
    
    def test_piece_constants_defined(self):
        """Test that piece constants are properly defined."""
        assert Piece.EMPTY.value == "e"
        assert Piece.BLUE.value == "b"
        assert Piece.RED.value == "r"
    
    def test_piece_constants_unique(self):
        """Test that piece constants are unique."""
        assert Piece.EMPTY.value != Piece.BLUE.value
        assert Piece.EMPTY.value != Piece.RED.value
        assert Piece.BLUE.value != Piece.RED.value
    
    def test_piece_constants_string(self):
        """Test that piece constants are strings."""
        assert isinstance(Piece.EMPTY.value, str)
        assert isinstance(Piece.BLUE.value, str)
        assert isinstance(Piece.RED.value, str)
    
    def test_piece_helper_functions(self):
        """Test that piece helper functions work correctly."""
        assert piece_to_char(Piece.EMPTY) == "e"
        assert piece_to_char(Piece.BLUE) == "b"
        assert piece_to_char(Piece.RED) == "r"


class TestWinnerFormatConstants:
    """Test winner format mapping constants."""
    
    def test_trmph_winner_constants(self):
        """Test that TRMPH winner constants have the expected values."""
        assert TRMPH_BLUE_WIN == "b"
        assert TRMPH_RED_WIN == "r"
    
    def test_training_winner_constants(self):
        """Test training format winner constants."""
        assert TRAINING_BLUE_WIN == 0.0
        assert TRAINING_RED_WIN == 1.0
        assert TRAINING_BLUE_WIN != TRAINING_RED_WIN
    
    def test_winner_constants_types(self):
        """Test that winner constants have correct types."""
        assert isinstance(TRMPH_BLUE_WIN, str)
        assert isinstance(TRMPH_RED_WIN, str)
        assert isinstance(TRAINING_BLUE_WIN, float)
        assert isinstance(TRAINING_RED_WIN, float)


class TestConstantsConsistency:
    """Test consistency between different constant types."""
    
    def test_player_piece_consistency(self):
        """Test that player and piece constants are consistent."""
        # BLUE_PLAYER (0) should correspond to BLUE_PIECE ('b') in board representation
        # RED_PLAYER (1) should correspond to RED_PIECE ('r') in board representation
        # This is because board representation now uses 'e'=empty, 'b'=blue, 'r'=red
        # while player-to-move uses 0=blue, 1=red
        assert Player.BLUE.value == 0
        assert Player.RED.value == 1
        assert Piece.BLUE.value == "b"
        assert Piece.RED.value == "r"
    
    def test_trmph_training_consistency(self):
        """Test that TRMPH and training format constants are consistent."""
        # TRMPH: "b"=blue, "r"=red
        # Training: 0.0=blue, 1.0=red
        # So TRMPH "b" corresponds to training 0.0, "r" to 1.0
        assert TRMPH_BLUE_WIN == "b"
        assert TRMPH_RED_WIN == "r"
        assert TRAINING_BLUE_WIN == 0.0
        assert TRAINING_RED_WIN == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 