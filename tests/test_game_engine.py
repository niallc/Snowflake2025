import pytest
import numpy as np
from hex_ai.inference.game_engine import HexGameState

@pytest.fixture
def empty_state():
    return HexGameState()

def test_initial_state(empty_state):
    state = empty_state
    assert state.board.shape == (13, 13)
    assert state.current_player == 0  # Blue
    assert state.move_history == []
    assert not state.game_over
    assert state.winner is None

def test_make_move_and_switch_player(empty_state):
    state = empty_state
    state2 = state.make_move(0, 0)
    assert state2.board[0, 0] == 1  # Blue piece
    assert state2.current_player == 1  # Red
    assert state2.move_history == [(0, 0)]
    # Red moves
    state3 = state2.make_move(0, 1)
    assert state3.board[0, 1] == 2  # Red piece
    assert state3.current_player == 0  # Blue
    assert state3.move_history == [(0, 0), (0, 1)]

def test_blue_win_detection():
    # Blue wins with a vertical line in column 0
    state = HexGameState()
    for row in range(13):
        state = state.make_move(row, 0)
        if row < 12:
            state = state.make_move(row, 1)  # Red plays elsewhere
    assert state.game_over
    assert state.winner == 'blue'

def test_trmph_serialization_and_deserialization():
    # Play a few moves, serialize to trmph, then restore
    moves = [(0, 0), (0, 1), (1, 0), (1, 1)]
    state = HexGameState()
    for move in moves:
        state = state.make_move(*move)
    trmph = state.to_trmph()
    restored = HexGameState.from_trmph(trmph)
    assert np.array_equal(state.board, restored.board)
    assert state.move_history == restored.move_history
    assert state.current_player == restored.current_player
    assert state.game_over == restored.game_over
    assert state.winner == restored.winner

def test_midgame_no_winner():
    # Play a few moves, but no winner
    moves = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 2)]
    state = HexGameState()
    for move in moves:
        state = state.make_move(*move)
    assert not state.game_over
    assert state.winner is None

def test_blue_final_win():
    blueFinal = "#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7g7"
    state = HexGameState.from_trmph(blueFinal)
    assert state.game_over
    assert state.winner == 'blue'

def test_red_final_win():
    redFinal = "#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7a2g7"
    state = HexGameState.from_trmph(redFinal)
    assert state.game_over
    assert state.winner == 'red'

def test_blue_win_one_move_left():
    blueWin = "#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7"
    state = HexGameState.from_trmph(blueWin)
    assert not state.game_over
    assert state.winner is None
    # Blue plays g7 to win
    state2 = state.make_move(6, 6)  # g7 is (6,6)
    assert state2.game_over
    assert state2.winner == 'blue'

def test_red_win_one_move_left():
    redWin = "#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7a2"
    state = HexGameState.from_trmph(redWin)
    assert not state.game_over
    assert state.winner is None
    # Red plays g7 to win
    state2 = state.make_move(6, 6)  # g7 is (6,6)
    assert state2.game_over
    assert state2.winner == 'red' 