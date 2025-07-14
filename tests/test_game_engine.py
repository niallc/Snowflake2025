"""
Tests for the game engine implementation.

This module tests the HexGameState and HexGameEngine classes,
including move validation, winner detection, and coordinate conversions.
"""

import pytest
import torch
import numpy as np

from hex_ai.inference.game_engine import HexGameState, HexGameEngine
from hex_ai.data_utils import trmph_move_to_rowcol, rowcol_to_trmph


class TestHexGameState:
    """Test the HexGameState class."""
    
    def test_initial_state(self):
        """Test that initial state is correct."""
        state = HexGameState()
        
        # Check board is empty
        assert state.board.shape == (2, 13, 13)
        assert torch.all(state.board == 0.0)
        
        # Check initial player
        assert state.current_player == 0  # Blue goes first
        
        # Check no moves made
        assert len(state.move_history) == 0
        
        # Check game not over
        assert not state.game_over
        assert state.winner is None
    
    def test_valid_move(self):
        """Test that valid moves are accepted."""
        state = HexGameState()
        
        # Test valid move
        assert state.is_valid_move(0, 0)  # Top-left corner
        assert state.is_valid_move(6, 6)  # Center
        assert state.is_valid_move(12, 12)  # Bottom-right corner
    
    def test_invalid_moves(self):
        """Test that invalid moves are rejected."""
        state = HexGameState()
        
        # Test out of bounds
        assert not state.is_valid_move(-1, 0)
        assert not state.is_valid_move(0, -1)
        assert not state.is_valid_move(13, 0)
        assert not state.is_valid_move(0, 13)
        
        # Test occupied position
        new_state = state.make_move(0, 0)
        assert not new_state.is_valid_move(0, 0)  # Already occupied
    
    def test_make_move(self):
        """Test making moves."""
        state = HexGameState()
        
        # Make first move (blue)
        new_state = state.make_move(0, 0)
        
        # Check board updated
        assert new_state.board[0, 0, 0] == 1.0  # Blue piece placed
        assert new_state.board[1, 0, 0] == 0.0  # Red piece not placed
        
        # Check player switched
        assert new_state.current_player == 1  # Red's turn
        
        # Check move history
        assert len(new_state.move_history) == 1
        assert new_state.move_history[0] == (0, 0)
        
        # Check game not over
        assert not new_state.game_over
        assert new_state.winner is None
    
    def test_winner_detection_red_horizontal(self):
        """Test winner detection for red horizontal win."""
        # Create a game where red wins with horizontal line
        moves = [
            (0, 0), (0, 1),  # Blue, Red
            (1, 0), (1, 1),  # Blue, Red
            (2, 0), (2, 1),  # Blue, Red
            (3, 0), (3, 1),  # Blue, Red
            (4, 0), (4, 1),  # Blue, Red
            (5, 0), (5, 1),  # Blue, Red
            (6, 0), (6, 1),  # Blue, Red
            (7, 0), (7, 1),  # Blue, Red
            (8, 0), (8, 1),  # Blue, Red
            (9, 0), (9, 1),  # Blue, Red
            (10, 0), (10, 1),  # Blue, Red
            (11, 0), (11, 1),  # Blue, Red
            (12, 0), (12, 1),  # Blue, Red - Red wins!
        ]
        
        state = HexGameState()
        for row, col in moves:
            state = state.make_move(row, col)
        
        # Check winner
        assert state.game_over
        assert state.winner == "red"
    
    def test_winner_detection_blue_vertical(self):
        """Test winner detection for blue vertical win."""
        # Create a game where blue wins with vertical line
        moves = [
            (0, 0), (1, 0),  # Blue, Red
            (0, 1), (1, 1),  # Blue, Red
            (0, 2), (1, 2),  # Blue, Red
            (0, 3), (1, 3),  # Blue, Red
            (0, 4), (1, 4),  # Blue, Red
            (0, 5), (1, 5),  # Blue, Red
            (0, 6), (1, 6),  # Blue, Red
            (0, 7), (1, 7),  # Blue, Red
            (0, 8), (1, 8),  # Blue, Red
            (0, 9), (1, 9),  # Blue, Red
            (0, 10), (1, 10),  # Blue, Red
            (0, 11), (1, 11),  # Blue, Red
            (0, 12), (1, 12),  # Blue, Red - Blue wins!
        ]
        
        state = HexGameState()
        for row, col in moves:
            state = state.make_move(row, col)
        
        # Check winner
        assert state.game_over
        assert state.winner == "blue"
    
    def test_no_winner_midgame(self):
        """Test that mid-game positions don't have winners."""
        # Create a mid-game position
        moves = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6)]
        
        state = HexGameState()
        for row, col in moves:
            state = state.make_move(row, col)
        
        # Check no winner
        assert not state.game_over
        assert state.winner is None
    
    def test_trmph_conversion(self):
        """Test TRMPH format conversion."""
        # Create a game state
        moves = [(0, 0), (1, 1), (2, 2), (3, 3)]
        state = HexGameState()
        for row, col in moves:
            state = state.make_move(row, col)
        
        # Convert to TRMPH
        trmph = state.to_trmph()
        expected = "#13,a1b2c3d4"
        assert trmph == expected
        
        # Convert back from TRMPH
        new_state = HexGameState.from_trmph(trmph)
        
        # Check states are equivalent
        assert torch.allclose(state.board, new_state.board)
        assert state.current_player == new_state.current_player
        assert state.move_history == new_state.move_history
        assert state.game_over == new_state.game_over
        assert state.winner == new_state.winner
    
    def test_get_legal_moves(self):
        """Test getting legal moves."""
        state = HexGameState()
        
        # Initially all positions should be legal
        legal_moves = state.get_legal_moves()
        assert len(legal_moves) == 13 * 13  # All 169 positions
        
        # After making a move, that position should not be legal
        new_state = state.make_move(0, 0)
        legal_moves = new_state.get_legal_moves()
        assert len(legal_moves) == 13 * 13 - 1  # 168 positions
        assert (0, 0) not in legal_moves
    
    def test_board_tensor(self):
        """Test board tensor conversion."""
        state = HexGameState()
        
        # Make a move
        new_state = state.make_move(0, 0)
        
        # Get board tensor
        board_tensor = new_state.get_board_tensor()
        
        # Check shape
        assert board_tensor.shape == (2, 13, 13)
        
        # Check blue piece placed
        assert board_tensor[0, 0, 0] == 1.0
        
        # Check red piece not placed
        assert board_tensor[1, 0, 0] == 0.0


class TestHexGameEngine:
    """Test the HexGameEngine class."""
    
    def test_engine_creation(self):
        """Test engine creation."""
        engine = HexGameEngine()
        assert engine.board_size == 13
    
    def test_reset(self):
        """Test game reset."""
        engine = HexGameEngine()
        state = engine.reset()
        
        assert isinstance(state, HexGameState)
        assert len(state.move_history) == 0
        assert not state.game_over
    
    def test_make_move(self):
        """Test making moves through engine."""
        engine = HexGameEngine()
        state = engine.reset()
        
        new_state = engine.make_move(state, 0, 0)
        assert new_state.board[0, 0, 0] == 1.0
    
    def test_is_valid_move(self):
        """Test move validation through engine."""
        engine = HexGameEngine()
        state = engine.reset()
        
        assert engine.is_valid_move(state, 0, 0)
        assert not engine.is_valid_move(state, -1, 0)
    
    def test_get_legal_moves(self):
        """Test getting legal moves through engine."""
        engine = HexGameEngine()
        state = engine.reset()
        
        legal_moves = engine.get_legal_moves(state)
        assert len(legal_moves) == 13 * 13
    
    def test_get_winner(self):
        """Test getting winner through engine."""
        engine = HexGameEngine()
        state = engine.reset()
        
        # No winner initially
        assert engine.get_winner(state) is None
        
        # Create winning position
        moves = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), 
                (3, 0), (3, 1), (4, 0), (4, 1), (5, 0), (5, 1), (6, 0), (6, 1)]
        
        for row, col in moves:
            state = engine.make_move(state, row, col)
        
        assert engine.get_winner(state) == "red"
    
    def test_is_game_over(self):
        """Test game over detection through engine."""
        engine = HexGameEngine()
        state = engine.reset()
        
        # Game not over initially
        assert not engine.is_game_over(state)
        
        # Create winning position
        moves = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), 
                (3, 0), (3, 1), (4, 0), (4, 1), (5, 0), (5, 1), (6, 0), (6, 1)]
        
        for row, col in moves:
            state = engine.make_move(state, row, col)
        
        assert engine.is_game_over(state)


class TestCoordinateConsistency:
    """Test coordinate conversion consistency."""
    
    def test_coordinate_conversions(self):
        """Test that coordinate conversions work correctly."""
        # Test some known conversions
        test_cases = [
            ((0, 0), "a1"),
            ((12, 12), "m13"),
            ((6, 6), "g7"),
            ((0, 12), "m1"),
            ((12, 0), "a13"),
        ]
        
        for (row, col), expected_trmph in test_cases:
            # Test rowcol to trmph
            trmph = rowcol_to_trmph(row, col)
            assert trmph == expected_trmph
            
            # Test trmph to rowcol
            parsed_row, parsed_col = trmph_move_to_rowcol(trmph)
            assert (parsed_row, parsed_col) == (row, col)
    
    def test_game_state_coordinate_consistency(self):
        """Test that game state uses consistent coordinates."""
        state = HexGameState()
        
        # Make moves using coordinates
        state = state.make_move(0, 0)  # a1
        state = state.make_move(6, 6)  # g7
        
        # Convert to TRMPH
        trmph = state.to_trmph()
        assert trmph == "#13,a1g7"
        
        # Convert back
        new_state = HexGameState.from_trmph(trmph)
        assert torch.allclose(state.board, new_state.board)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 