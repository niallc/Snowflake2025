"""
Test centralized constants from hex_ai.config.

This module tests that all player and piece constants are properly defined
and accessible from the centralized config module.
"""

import pytest
from hex_ai.config import (
    BLUE_PLAYER, RED_PLAYER,
    BLUE_PIECE, RED_PIECE, EMPTY_PIECE,
    TRMPH_BLUE_WIN, TRMPH_RED_WIN,
    TRAINING_BLUE_WIN, TRAINING_RED_WIN
)


class TestPlayerConstants:
    """Test player constants for player-to-move channel and game logic."""
    
    def test_player_constants_defined(self):
        """Test that player constants are properly defined."""
        assert BLUE_PLAYER == 0
        assert RED_PLAYER == 1
        assert BLUE_PLAYER != RED_PLAYER
    
    def test_player_constants_integer(self):
        """Test that player constants are integers."""
        assert isinstance(BLUE_PLAYER, int)
        assert isinstance(RED_PLAYER, int)


class TestPieceConstants:
    """Test piece constants for board representation."""
    
    def test_piece_constants_defined(self):
        """Test that piece constants are properly defined."""
        assert EMPTY_PIECE == 0
        assert BLUE_PIECE == 1
        assert RED_PIECE == 2
    
    def test_piece_constants_unique(self):
        """Test that piece constants are unique."""
        assert EMPTY_PIECE != BLUE_PIECE
        assert EMPTY_PIECE != RED_PIECE
        assert BLUE_PIECE != RED_PIECE
    
    def test_piece_constants_integer(self):
        """Test that piece constants are integers."""
        assert isinstance(EMPTY_PIECE, int)
        assert isinstance(BLUE_PIECE, int)
        assert isinstance(RED_PIECE, int)


class TestWinnerFormatConstants:
    """Test winner format mapping constants."""
    
    def test_trmph_winner_constants(self):
        """Test TRMPH format winner constants."""
        assert TRMPH_BLUE_WIN == "1"
        assert TRMPH_RED_WIN == "2"
        assert TRMPH_BLUE_WIN != TRMPH_RED_WIN
    
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
        # BLUE_PLAYER (0) should correspond to BLUE_PIECE (1) in board representation
        # RED_PLAYER (1) should correspond to RED_PIECE (2) in board representation
        # This is because board representation uses 0=empty, 1=blue, 2=red
        # while player-to-move uses 0=blue, 1=red
        assert BLUE_PLAYER + 1 == BLUE_PIECE
        assert RED_PLAYER + 1 == RED_PIECE
    
    def test_trmph_training_consistency(self):
        """Test that TRMPH and training format constants are consistent."""
        # TRMPH: "1"=blue, "2"=red
        # Training: 0.0=blue, 1.0=red
        # So TRMPH value - 1 = training value
        assert int(TRMPH_BLUE_WIN) - 1 == int(TRAINING_BLUE_WIN)
        assert int(TRMPH_RED_WIN) - 1 == int(TRAINING_RED_WIN)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 