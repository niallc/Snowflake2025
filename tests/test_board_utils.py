# NOTE: This test file covers 2-channel board conversion utilities. 
# The 3-channel format (with player-to-move) is now used for inference/model input.
# TODO: Add/expand tests for 3-channel conversion utilities if/when needed.
"""
Tests for board format conversion utilities.
"""

import pytest
import torch
import numpy as np

from hex_ai.utils.format_conversion import (
    board_2nxn_to_nxn, board_nxn_to_2nxn
)
from hex_ai.inference.board_utils import (
    get_piece_at, has_piece_at,
    is_empty, place_piece, board_to_string, validate_board, count_pieces
)
from hex_ai.enums import Piece, piece_to_char
from hex_ai.config import BOARD_SIZE


class TestBoardConversion:
    """Test board format conversions."""
    
    def test_empty_board_conversion(self):
        """Test conversion of empty board."""
        # Create empty 2×N×N tensor
        board_2nxn = torch.zeros(2, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        
        # Convert to N×N
        board_nxn = board_2nxn_to_nxn(board_2nxn)
        
        # Check all positions are empty
        assert np.all(board_nxn == piece_to_char(Piece.EMPTY))
        
        # Convert back
        board_2nxn_back = board_nxn_to_2nxn(board_nxn)
        
        # Check they're equal
        assert torch.allclose(board_2nxn, board_2nxn_back)
    
    def test_single_piece_conversion(self):
        """Test conversion with single pieces."""
        # Create 2×N×N with blue piece at (0,0)
        board_2nxn = torch.zeros(2, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        board_2nxn[0, 0, 0] = 1.0  # Blue piece
        
        # Convert to N×N
        board_nxn = board_2nxn_to_nxn(board_2nxn)
        
        # Check blue piece is at (0,0)
        assert board_nxn[0, 0] == piece_to_char(Piece.BLUE)
        
        # Check all other positions are empty
        board_nxn[0, 0] = piece_to_char(Piece.EMPTY)  # Temporarily remove the piece
        assert np.all(board_nxn == piece_to_char(Piece.EMPTY))
    
    def test_red_piece_conversion(self):
        """Test conversion with red piece."""
        # Create 2×N×N with red piece at (1,1)
        board_2nxn = torch.zeros(2, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        board_2nxn[1, 1, 1] = 1.0  # Red piece
        
        # Convert to N×N
        board_nxn = board_2nxn_to_nxn(board_2nxn)
        
        # Check red piece is at (1,1)
        assert board_nxn[1, 1] == piece_to_char(Piece.RED)
        
        # Check all other positions are empty
        board_nxn[1, 1] = piece_to_char(Piece.EMPTY)  # Temporarily remove the piece
        assert np.all(board_nxn == piece_to_char(Piece.EMPTY))
    
    def test_multiple_pieces_conversion(self):
        """Test conversion with multiple pieces."""
        # Create 2×N×N with mixed pieces
        board_2nxn = torch.zeros(2, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        board_2nxn[0, 0, 0] = 1.0  # Blue at (0,0)
        board_2nxn[1, 1, 1] = 1.0  # Red at (1,1)
        board_2nxn[0, 2, 2] = 1.0  # Blue at (2,2)
        
        # Convert to N×N
        board_nxn = board_2nxn_to_nxn(board_2nxn)
        
        # Check pieces are in correct positions
        assert board_nxn[0, 0] == piece_to_char(Piece.BLUE)
        assert board_nxn[1, 1] == piece_to_char(Piece.RED)
        assert board_nxn[2, 2] == piece_to_char(Piece.BLUE)
        
        # Convert back
        board_2nxn_back = board_nxn_to_2nxn(board_nxn)
        
        # Check they're equal
        assert torch.allclose(board_2nxn, board_2nxn_back)


class TestBoardOperations:
    """Test board operation functions."""
    
    def test_get_piece_at(self):
        """Test getting piece at position."""
        board_nxn = np.full((BOARD_SIZE, BOARD_SIZE), piece_to_char(Piece.EMPTY), dtype='U1')
        board_nxn[0, 0] = piece_to_char(Piece.BLUE)
        
        # Test getting blue piece
        piece = get_piece_at(board_nxn, 0, 0)
        assert piece == piece_to_char(Piece.BLUE)
        
        # Test getting empty piece
        piece = get_piece_at(board_nxn, 1, 1)
        assert piece == piece_to_char(Piece.EMPTY)
        
        # Test out of bounds
        with pytest.raises(IndexError):
            get_piece_at(board_nxn, BOARD_SIZE, 0)
    
    def test_has_piece_at(self):
        """Test checking for pieces of specific color."""
        board_nxn = np.full((BOARD_SIZE, BOARD_SIZE), piece_to_char(Piece.EMPTY), dtype='U1')
        board_nxn[0, 0] = piece_to_char(Piece.BLUE)
        board_nxn[1, 1] = piece_to_char(Piece.RED)
        
        # Test blue piece
        assert has_piece_at(board_nxn, 0, 0, "blue") == True
        assert has_piece_at(board_nxn, 1, 1, "blue") == False
        
        # Test red piece
        assert has_piece_at(board_nxn, 1, 1, "red") == True
        assert has_piece_at(board_nxn, 0, 0, "red") == False
        
        # Test empty position
        assert has_piece_at(board_nxn, 2, 2, "blue") == False
        assert has_piece_at(board_nxn, 2, 2, "red") == False
        
        # Test out of bounds
        assert has_piece_at(board_nxn, BOARD_SIZE, 0, "blue") == False
    
    def test_is_empty(self):
        """Test checking if position is empty."""
        board_nxn = np.full((BOARD_SIZE, BOARD_SIZE), piece_to_char(Piece.EMPTY), dtype='U1')
        board_nxn[0, 0] = piece_to_char(Piece.BLUE)
        
        # Test empty position
        assert is_empty(board_nxn, 1, 1) == True
        
        # Test occupied position
        assert is_empty(board_nxn, 0, 0) == False
        
        # Test out of bounds
        assert is_empty(board_nxn, BOARD_SIZE, 0) == False
    
    def test_place_piece(self):
        """Test placing pieces."""
        board_nxn = np.full((BOARD_SIZE, BOARD_SIZE), piece_to_char(Piece.EMPTY), dtype='U1')
        
        # Place blue piece
        new_board = place_piece(board_nxn, 0, 0, "blue")
        assert new_board[0, 0] == piece_to_char(Piece.BLUE)
        assert new_board[1, 1] == piece_to_char(Piece.EMPTY)  # Other positions unchanged
        
        # Place red piece
        new_board = place_piece(new_board, 1, 1, "red")
        assert new_board[1, 1] == piece_to_char(Piece.RED)
        
        # Test placing on occupied position
        with pytest.raises(ValueError):
            place_piece(new_board, 0, 0, "red")
        
        # Test out of bounds
        with pytest.raises(ValueError):
            place_piece(board_nxn, BOARD_SIZE, 0, "blue")
    
    def test_board_to_string(self):
        """Test board string representation."""
        board_nxn = np.full((BOARD_SIZE, BOARD_SIZE), piece_to_char(Piece.EMPTY), dtype='U1')
        board_nxn[0, 0] = piece_to_char(Piece.BLUE)
        board_nxn[1, 1] = piece_to_char(Piece.RED)
        
        board_str = board_to_string(board_nxn)
        
        # Check that the string contains the display symbols
        assert 'B' in board_str  # Blue piece displayed as 'B'
        assert 'R' in board_str  # Red piece displayed as 'R'
        assert '.' in board_str  # Empty positions displayed as '.'
        
        # Check that the string has the expected structure (hexagonal shape with indentation)
        lines = board_str.split('\n')
        assert len(lines) == BOARD_SIZE
        
        # First line should have blue piece at start
        assert lines[0].strip().startswith('B')
        
        # Second line should have red piece with indent
        assert 'R' in lines[1]
    
    def test_validate_board(self):
        """Test board validation."""
        # Valid board
        board_nxn = np.full((BOARD_SIZE, BOARD_SIZE), piece_to_char(Piece.EMPTY), dtype='U1')
        board_nxn[0, 0] = piece_to_char(Piece.BLUE)
        board_nxn[1, 1] = piece_to_char(Piece.RED)
        assert validate_board(board_nxn) == True
        
        # Invalid board (wrong shape)
        invalid_board = np.full((BOARD_SIZE, BOARD_SIZE + 1), piece_to_char(Piece.EMPTY), dtype='U1')
        assert validate_board(invalid_board) == False
    
    def test_count_pieces(self):
        """Test counting pieces."""
        board_nxn = np.full((BOARD_SIZE, BOARD_SIZE), piece_to_char(Piece.EMPTY), dtype='U1')
        board_nxn[0, 0] = piece_to_char(Piece.BLUE)
        board_nxn[1, 1] = piece_to_char(Piece.RED)
        board_nxn[2, 2] = piece_to_char(Piece.BLUE)
        
        blue_count, red_count = count_pieces(board_nxn)
        assert blue_count == 2
        assert red_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 