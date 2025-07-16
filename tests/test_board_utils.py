# NOTE: This test file covers 2-channel board conversion utilities. 
# The 3-channel format (with player-to-move) is now used for inference/model input.
# TODO: Add/expand tests for 3-channel conversion utilities if/when needed.
"""
Tests for board format conversion utilities.
"""

import pytest
import torch
import numpy as np

from hex_ai.inference.board_utils import (
    board_2nxn_to_nxn, board_nxn_to_2nxn, get_piece_at, has_piece_at,
    is_empty, place_piece, board_to_string, validate_board, count_pieces,
    EMPTY, BLUE, RED
)
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
        assert np.all(board_nxn == EMPTY)
        
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
        assert board_nxn[0, 0] == BLUE
        
        # Check all other positions are empty
        board_nxn[0, 0] = EMPTY  # Temporarily remove the piece
        assert np.all(board_nxn == EMPTY)
    
    def test_red_piece_conversion(self):
        """Test conversion with red piece."""
        # Create 2×N×N with red piece at (1,1)
        board_2nxn = torch.zeros(2, BOARD_SIZE, BOARD_SIZE, dtype=torch.float32)
        board_2nxn[1, 1, 1] = 1.0  # Red piece
        
        # Convert to N×N
        board_nxn = board_2nxn_to_nxn(board_2nxn)
        
        # Check red piece is at (1,1)
        assert board_nxn[1, 1] == RED
        
        # Check all other positions are empty
        board_nxn[1, 1] = EMPTY  # Temporarily remove the piece
        assert np.all(board_nxn == EMPTY)
    
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
        assert board_nxn[0, 0] == BLUE
        assert board_nxn[1, 1] == RED
        assert board_nxn[2, 2] == BLUE
        
        # Convert back
        board_2nxn_back = board_nxn_to_2nxn(board_nxn)
        
        # Check they're equal
        assert torch.allclose(board_2nxn, board_2nxn_back)


class TestBoardOperations:
    """Test board operation functions."""
    
    def test_get_piece_at(self):
        """Test getting piece at position."""
        board_nxn = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board_nxn[0, 0] = BLUE
        board_nxn[1, 1] = RED
        
        assert get_piece_at(board_nxn, 0, 0) == BLUE
        assert get_piece_at(board_nxn, 1, 1) == RED
        assert get_piece_at(board_nxn, 2, 2) == EMPTY
        
        # Test out of bounds
        with pytest.raises(IndexError):
            get_piece_at(board_nxn, -1, 0)
        with pytest.raises(IndexError):
            get_piece_at(board_nxn, 0, BOARD_SIZE)
    
    def test_has_piece_at(self):
        """Test checking for pieces of specific color."""
        board_nxn = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board_nxn[0, 0] = BLUE
        board_nxn[1, 1] = RED
        
        assert has_piece_at(board_nxn, 0, 0, "blue")
        assert not has_piece_at(board_nxn, 0, 0, "red")
        assert has_piece_at(board_nxn, 1, 1, "red")
        assert not has_piece_at(board_nxn, 1, 1, "blue")
        assert not has_piece_at(board_nxn, 2, 2, "blue")
        assert not has_piece_at(board_nxn, 2, 2, "red")
        
        # Test out of bounds
        assert not has_piece_at(board_nxn, -1, 0, "blue")
        assert not has_piece_at(board_nxn, 0, BOARD_SIZE, "red")
    
    def test_is_empty(self):
        """Test checking if position is empty."""
        board_nxn = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board_nxn[0, 0] = BLUE
        board_nxn[1, 1] = RED
        
        assert not is_empty(board_nxn, 0, 0)
        assert not is_empty(board_nxn, 1, 1)
        assert is_empty(board_nxn, 2, 2)
        
        # Test out of bounds
        assert not is_empty(board_nxn, -1, 0)
        assert not is_empty(board_nxn, 0, BOARD_SIZE)
    
    def test_place_piece(self):
        """Test placing pieces."""
        board_nxn = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        
        # Place blue piece
        new_board = place_piece(board_nxn, 0, 0, "blue")
        assert new_board[0, 0] == BLUE
        assert np.all(new_board[1:, :] == EMPTY)
        assert np.all(new_board[0, 1:] == EMPTY)
        
        # Place red piece
        new_board = place_piece(new_board, 1, 1, "red")
        assert new_board[1, 1] == RED
        
        # Test placing on occupied position
        with pytest.raises(ValueError):
            place_piece(new_board, 0, 0, "red")
        
        # Test out of bounds
        with pytest.raises(ValueError):
            place_piece(board_nxn, -1, 0, "blue")
        with pytest.raises(ValueError):
            place_piece(board_nxn, 0, BOARD_SIZE, "red")
        
        # Test invalid color
        with pytest.raises(ValueError):
            place_piece(board_nxn, 2, 2, "green")
    
    def test_board_to_string(self):
        """Test board string representation."""
        board_nxn = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board_nxn[0, 0] = BLUE
        board_nxn[1, 1] = RED
        
        board_str = board_to_string(board_nxn)
        lines = board_str.split('\n')
        
        # Check first line has blue piece
        assert 'B' in lines[0]
        # Check second line has red piece (with indent)
        assert 'R' in lines[1]
    
    def test_validate_board(self):
        """Test board validation."""
        # Valid board
        board_nxn = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board_nxn[0, 0] = BLUE
        board_nxn[1, 1] = RED
        assert validate_board(board_nxn)
        
        # Invalid board (wrong shape)
        board_invalid = np.zeros((BOARD_SIZE-1, BOARD_SIZE), dtype=np.int8)
        assert not validate_board(board_invalid)
        
        # Invalid board (wrong values)
        board_invalid = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board_invalid[0, 0] = 3  # Invalid value
        assert not validate_board(board_invalid)
    
    def test_count_pieces(self):
        """Test counting pieces."""
        board_nxn = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board_nxn[0, 0] = BLUE
        board_nxn[1, 1] = RED
        board_nxn[2, 2] = BLUE
        
        blue_count, red_count = count_pieces(board_nxn)
        assert blue_count == 2
        assert red_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 