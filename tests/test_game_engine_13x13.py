"""
Tests for 13x13 game engine implementation.

This module tests the HexGameState and HexGameEngine classes with proper 13x13 test cases
that follow the pattern of testing positions where adding a piece creates a winner.
"""

import pytest
import torch
import numpy as np

from hex_ai.inference.game_engine import HexGameState, HexGameEngine
from hex_ai.data_utils import trmph_move_to_rowcol

# Validated TRMPH test cases
CASES = {
    "centre_connects": "a1a13a2b12b2c11b3d10c3e9c4f8d4h6d5i5e5j4e6k3f6l2f7m1h7a12h8a11i8a10i9a9j9a8j10a7k10a6k11a5l11a4l12a3m12m2m13",
    "centre_connects_blue_only": "g6f6g5f5g4f4g3f3g2f2g1f1g8f7g9h7g10h8g11h9g12h10g13h11f13h12e13h13",
    "centre_connects_red_only": "a7a8b7b8c7c8d7d8e7e8f7f8h7h6i7i6j7j6k7k6l7l6m7m6",
    "real_play_board_red_win": "m11f5d4h6i6i7h7i5e8e7k4j5c8d9d8e9g7g6d6c7d7d5c6f8f7g9j8j6c5h8b10c9b9c11d10c10l5b12k8k9i10j9i9k7m6l7m7l8m8l10l9k10a13a12m9m10k6j7"
}

def parse_trmph_moves(trmph):
    # Parse a TRMPH string (no #13, prefix) into a list of (row, col) moves
    moves = []
    i = 0
    while i < len(trmph):
        # Find the next letter
        if not trmph[i].isalpha():
            i += 1
            continue
        j = i + 1
        while j < len(trmph) and trmph[j].isdigit():
            j += 1
        move = trmph[i:j]
        moves.append(trmph_move_to_rowcol(move))
        i = j
    return moves

class TestValidatedWinnerDetection:
    def test_centre_connects(self):
        moves = parse_trmph_moves(CASES["centre_connects"])
        state = HexGameState()
        for row, col in moves:
            state = state.make_move(row, col)
        # No winner yet
        assert not state.game_over
        # Red plays g7
        red_state = state.make_move(6, 6)
        assert red_state.game_over
        assert red_state.winner == "red"
        # Blue plays g7 (reset)
        state = HexGameState()
        for row, col in moves:
            state = state.make_move(row, col)
        blue_state = state.make_move(6, 6)
        assert blue_state.game_over
        assert blue_state.winner == "blue"

    def test_centre_connects_blue_only(self):
        moves = parse_trmph_moves(CASES["centre_connects_blue_only"])
        state = HexGameState()
        for row, col in moves:
            state = state.make_move(row, col)
        # No winner yet
        assert not state.game_over
        # Red plays g7
        red_state = state.make_move(6, 6)
        assert not red_state.game_over
        # Blue plays g7 (reset)
        state = HexGameState()
        for row, col in moves:
            state = state.make_move(row, col)
        blue_state = state.make_move(6, 6)
        assert blue_state.game_over
        assert blue_state.winner == "blue"

    def test_centre_connects_red_only(self):
        moves = parse_trmph_moves(CASES["centre_connects_red_only"])
        state = HexGameState()
        for row, col in moves:
            state = state.make_move(row, col)
        # No winner yet
        assert not state.game_over
        # Red plays g7
        red_state = state.make_move(6, 6)
        assert red_state.game_over
        assert red_state.winner == "red"
        # Blue plays g7 (reset)
        state = HexGameState()
        for row, col in moves:
            state = state.make_move(row, col)
        blue_state = state.make_move(6, 6)
        assert not blue_state.game_over

    def test_real_play_board_red_win(self):
        moves = parse_trmph_moves(CASES["real_play_board_red_win"])
        state = HexGameState()
        for row, col in moves:
            state = state.make_move(row, col)
        assert state.game_over
        assert state.winner == "red"


class Test13x13GameEngine:
    """Test the game engine with 13x13 boards."""
    
    def test_engine_13x13(self):
        """Test engine works correctly with 13x13 boards."""
        engine = HexGameEngine()
        state = engine.reset()
        
        # Check board size
        assert engine.board_size == 13
        assert state.board.shape == (2, 13, 13)
        
        # Make some moves
        state = engine.make_move(state, 0, 0)
        state = engine.make_move(state, 1, 1)
        
        # Check legal moves
        legal_moves = engine.get_legal_moves(state)
        assert len(legal_moves) == 13 * 13 - 2  # 167 positions left
        
        # Check no winner yet
        assert not engine.is_game_over(state)
        assert engine.get_winner(state) is None
    
    def test_engine_winner_detection(self):
        """Test engine winner detection with 13x13 boards."""
        engine = HexGameEngine()
        state = engine.reset()
        
        # Create horizontal red win by manually setting board
        board = torch.zeros(2, 13, 13)
        board[1, 0, :] = 1.0  # Red pieces in top row
        state.board = board
        state.current_player = 0
        state.move_history = [(0, i) for i in range(13)]
        
        # Update connections from board state
        state._update_connections_from_board()
        
        # Check red wins
        assert engine.is_game_over(state)
        assert engine.get_winner(state) == "red"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 