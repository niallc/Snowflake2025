"""
Advanced winner detection tests for Hex game engine.

This module tests complex scenarios including:
- Center moves that cause wins for both players
- Blue-only wins from center
- Red-only wins from center  
- Complex realistic game positions
"""

import pytest
import numpy as np
from hex_ai.inference.game_engine import HexGameState, BLUE, RED
from hex_ai.data_utils import strip_trmph_preamble, split_trmph_moves, trmph_move_to_rowcol, rowcol_to_trmph


from hex_ai.inference.board_utils import (
    EMPTY, BLUE, RED, BLUE_PLAYER, RED_PLAYER,
)

def board_state_to_trmph(state: HexGameState) -> str:
    """Convert the move history of a HexGameState to a TRMPH string."""
    moves = state.move_history
    trmph_moves = ''.join([rowcol_to_trmph(row, col) for row, col in moves])
    return f"#13,{trmph_moves}"

class TestTrmphAdvancedCases:
    def play_moves_except(self, state, moves, skip_move):
        """Play all moves except the one matching skip_move (in trmph format)."""
        for i, move in enumerate(moves):
            if move == skip_move:
                continue
            row, col = trmph_move_to_rowcol(move)
            state = state.make_move(row, col)
        return state

    def assert_winner(self, state, expected, context=""):
        winner = state.winner if hasattr(state, 'winner') else state._find_winner()
        try:
            assert winner == expected
        except AssertionError:
            print(f"\n[DEBUG] Assertion failed in {context}")
            print(f"Expected winner: {expected}, Actual winner: {winner}")
            print(f"TRMPH record: {board_state_to_trmph(state)}")
            print(f"Move history: {state.move_history}")
            print(state)
            raise

    def test_centre_connects(self):
        # trmph = "#13,a1a13a2b12b2c11b3d10c3e9c4f8d4h6d5i5e5j4e6k3f6l2f7m1h7a12h8a11i8a10i9a9j9a8j10a7k10a6k11a5l11a4l12a3m12m2m13"
        # Simpler test case that has a win for both players at (6,6)
        trmph = "#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7" # (Blue to play)
        bare_moves = strip_trmph_preamble(trmph)
        moves = split_trmph_moves(bare_moves)
        centre_move = "g7"  # (6,6)
        state = HexGameState()
        state = self.play_moves_except(state, moves, centre_move)
        # No winner yet
        self.assert_winner(state, None, context="centre_connects before g7")
        # Blue plays g7
        state.current_player = BLUE_PLAYER
        blue_state = state.make_move(6, 6)
        self.assert_winner(blue_state, "blue", context="centre_connects blue g7")
        # Red plays g7 (from the same pre-move state)
        state.current_player = RED_PLAYER
        red_state = state.make_move(6, 6)
        self.assert_winner(red_state, "red", context="centre_connects red g7")

    def test_ccts_blue(self):
        # trmph = "#13,g6f6g5f5g4f4g3f3g2f2g1f1g8f7g9h7g10h8g11h9g12h10g13h11f13h12e13h13"
        trmph = "#13,f1g1f2g2f3g3f4g4f5g5f6g6f7g8h7g9h8g10h9g11h10g12h11g13h12f13h13" # (Red to play)
        bare_moves = strip_trmph_preamble(trmph)
        moves = split_trmph_moves(bare_moves)
        centre_move = "g7"  # (6,6)
        state = HexGameState()
        state = self.play_moves_except(state, moves, centre_move)
        self.assert_winner(state, None, context="ccts_blue before g7")
        # Blue plays g7
        state.current_player = BLUE_PLAYER
        blue_state = state.make_move(6, 6)
        self.assert_winner(blue_state, "blue", context="ccts_blue blue g7")
        # Red plays g7 (should not win)
        state.current_player = RED_PLAYER
        red_state = state.make_move(6, 6)
        self.assert_winner(red_state, None, context="ccts_blue red g7")

    def test_ccts_red(self):
        trmph = "#13,a7a8b7b8c7c8d7d8e7e8f7f8h7h6i7i6j7j6k7k6l7l6m7m6"
        bare_moves = strip_trmph_preamble(trmph)
        moves = split_trmph_moves(bare_moves)
        centre_move = "g7"  # (6,6)
        state = HexGameState()
        state = self.play_moves_except(state, moves, centre_move)
        self.assert_winner(state, None, context="ccts_red before g7")
        # Red plays g7
        state.current_player = RED_PLAYER
        red_state = state.make_move(6, 6)
        self.assert_winner(red_state, "red", context="ccts_red red g7")
        # Blue plays g7 (should not win)
        state.current_player = BLUE_PLAYER
        blue_state = state.make_move(6, 6)
        self.assert_winner(blue_state, None, context="ccts_red blue g7")

    def test_real_game_red_win(self):
        trmph = "#13,m11f5d4h6i6i7h7i5e8e7k4j5c8d9d8e9g7g6d6c7d7d5c6f8f7g9j8j6c5h8b10c9b9c11d10c10l5b12k8k9i10j9i9k7m6l7m7l8m8l10l9k10a13a12m9m10k6j7"
        bare_moves = strip_trmph_preamble(trmph)
        moves = split_trmph_moves(bare_moves)
        state = HexGameState()
        for move in moves:
            row, col = trmph_move_to_rowcol(move)
            state = state.make_move(row, col)
        self.assert_winner(state, "red", context="real_game red win")


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 