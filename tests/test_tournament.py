# NOTE: To skip slow tests, run: pytest -m 'not slow'
import pytest
import tempfile
import os
import time
from hex_ai.utils import tournament_logging
from hex_ai.inference import tournament
from hex_ai.value_utils import get_win_prob_from_model_output, Winner
import numpy as np

ALL_RESULTS_DIR = "checkpoints/hyperparameter_tuning/"
THIS_MODEL_DIR = "loss_weight_sweep_exp0_bs256_98f719_20250724_233408"
CHKPT_FILENAMES = [
    "epoch1_mini1.pt",
    "epoch1_mini5.pt"
]
CHKPT_PATHS = [os.path.join(ALL_RESULTS_DIR, THIS_MODEL_DIR, fname) for fname in CHKPT_FILENAMES]

SLOW = pytest.mark.slow

test_timings = {}

def timed_test(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        test_timings[func.__name__] = elapsed
        print(f"[TIMING] {func.__name__}: {elapsed:.2f} seconds")
        return result
    return wrapper

@pytest.fixture(scope="session", autouse=True)
def print_timing_summary(request):
    yield
    print("\n==== Test Timing Summary ====")
    for name, elapsed in sorted(test_timings.items()):
        print(f"{name:35s} {elapsed:.2f} seconds")
    print("===========================\n")

# ============================================================================
# UNIT TESTS - Pure functions, no model inference or complex mocks
# ============================================================================

def test_get_win_prob_from_model_output_valid():
    """Test win probability calculation from model output logits."""
    # Model output 0.7 (sigmoid), so prob_red_win ~0.668
    import torch
    logit = torch.logit(torch.tensor(0.7)).item()
    # Should work for Winner enums
    p_blue = get_win_prob_from_model_output(logit, Winner.BLUE)
    p_red = get_win_prob_from_model_output(logit, Winner.RED)
    # Should work for strings
    p_blue2 = get_win_prob_from_model_output(logit, 'blue')
    p_red2 = get_win_prob_from_model_output(logit, 'red')
    assert abs(p_blue + p_red - 1.0) < 1e-6
    assert abs(p_blue - p_blue2) < 1e-6
    assert abs(p_red - p_red2) < 1e-6

def test_get_win_prob_from_model_output_invalid():
    """Test win probability calculation with invalid inputs."""
    import torch
    logit = torch.logit(torch.tensor(0.7)).item()
    with pytest.raises(ValueError):
        get_win_prob_from_model_output(logit, 0)  # int not allowed
    with pytest.raises(ValueError):
        get_win_prob_from_model_output(logit, 'invalid')

def test_determine_winner_and_swap_logic():
    """Test that determine_winner correctly maps winner to model and color, with and without swap."""
    from hex_ai.inference.tournament import determine_winner
    class DummyModel: pass
    model_1 = DummyModel()
    model_2 = DummyModel()
    # Dummy state with .winner attribute
    class DummyState:
        def __init__(self, winner):
            self.winner = winner
    # No swap, blue wins
    state = DummyState("blue")
    result, winner_char = determine_winner(state, model_1, model_2, swap=False)
    assert result == "1" and winner_char == "b"
    # No swap, red wins
    state = DummyState("red")
    result, winner_char = determine_winner(state, model_1, model_2, swap=False)
    assert result == "2" and winner_char == "r"
    # Swap, blue wins
    state = DummyState("blue")
    result, winner_char = determine_winner(state, model_1, model_2, swap=True)
    assert result == "2" and winner_char == "b"
    # Swap, red wins
    state = DummyState("red")
    result, winner_char = determine_winner(state, model_1, model_2, swap=True)
    assert result == "1" and winner_char == "r"
    # No winner (should raise)
    state = DummyState(None)
    with pytest.raises(RuntimeError):
        determine_winner(state, model_1, model_2, swap=False)
    with pytest.raises(RuntimeError):
        determine_winner(state, model_1, model_2, swap=True)

def test_network_outputs_are_logits():
    """Test that the network outputs are used as logits, not probabilities, in move selection."""
    from hex_ai.inference.tournament import select_move
    from hex_ai.inference.game_engine import HexGameState
    class DummyModel:
        def __init__(self): 
            self.checkpoint_path = 'dummy'
            self.last_input = None
        def infer(self, board):
            import numpy as np
            self.last_input = board
            # Return logits (not probabilities)
            policy_logits = np.arange(169, dtype=np.float32)
            return policy_logits, 0.0
    state = HexGameState()
    model = DummyModel()
    move = select_move(state, model, search_widths=None, temperature=1.0)
    # The move should be the argmax (since logits are increasing)
    assert move == divmod(np.argmax(np.arange(169)), 13)

def test_handle_pie_rule_swap_and_no_swap():
    """Test handle_pie_rule returns correct swap/model assignment and swap_decision."""
    from hex_ai.inference.tournament import handle_pie_rule, TournamentPlayConfig
    from hex_ai.inference.game_engine import HexGameState
    class DummyModel:
        def __init__(self, win_prob_blue):
            self.win_prob_blue = win_prob_blue
            self.checkpoint_path = "dummy"
        def infer(self, board):
            import numpy as np
            import torch
            policy_logits = np.ones(169, dtype=np.float32)
            # Convert Blue win probability to the correct logit
            # get_win_prob_from_model_output expects: sigmoid(logit) = prob_red_win
            # So if we want Blue to have win_prob_blue, we need sigmoid(logit) = 1.0 - win_prob_blue
            prob_red_win = 1.0 - self.win_prob_blue
            logit = torch.logit(torch.tensor(prob_red_win)).item()
            return policy_logits, logit
    # No swap: win_prob_blue below threshold (Blue's position not good enough to swap)
    state = HexGameState()
    model_1 = DummyModel(0.2)  # This gives win_prob_blue around 0.2
    model_2 = DummyModel(0.2)
    play_config = TournamentPlayConfig(pie_rule=True, swap_threshold=0.5)
    result = handle_pie_rule(state, model_1, model_2, play_config, verbose=0)
    assert result.swap == False
    assert result.swap_decision == "no_swap"
    assert result.model_1 == model_1  # model_1 stays as blue
    assert result.model_2 == model_2  # model_2 stays as red
    # Swap: win_prob_blue above threshold (Blue's position too good, Red should swap)
    state = HexGameState()
    model_1 = DummyModel(0.8)  # This gives win_prob_blue around 0.8
    model_2 = DummyModel(0.8)
    play_config = TournamentPlayConfig(pie_rule=True, swap_threshold=0.5)
    result = handle_pie_rule(state, model_1, model_2, play_config, verbose=0)
    assert result.swap == True
    assert result.swap_decision == "swap"
    assert result.model_1 == model_2  # model_2 becomes blue (first)
    assert result.model_2 == model_1  # model_1 becomes red (second)

def test_select_move_policy_and_tree_search():
    """Test select_move chooses the correct move for both policy and tree search."""
    from hex_ai.inference.tournament import select_move
    from hex_ai.inference.game_engine import HexGameState
    class DummyModel:
        def __init__(self):
            self.checkpoint_path = "dummy"
        def infer(self, board):
            import numpy as np
            policy_logits = np.ones(169, dtype=np.float32)
            return policy_logits, 0.0
    state = HexGameState()
    model = DummyModel()
    # Policy move (no search_widths)
    move = select_move(state, model, search_widths=None, temperature=1.0)
    assert isinstance(move, tuple) and len(move) == 2
    # Tree search move (with search_widths)
    move = select_move(state, model, search_widths=(3, 2), temperature=1.0)
    assert isinstance(move, tuple) and len(move) == 2 

# ============================================================================
# BOARD/GAME LOGIC TESTS - Using known positions and move sequences
# ============================================================================

def test_winner_detection_with_known_positions():
    """Test winner detection with known TRMPH positions that have clear winners."""
    from hex_ai.inference.game_engine import HexGameState
    
    # Test cases: (trmph_string, expected_winner, description)
    test_cases = [
        # Minimal finished games
        ("#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7g7", "blue", "Minimal blue win"),
        ("#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7a2g7", "red", "Minimal red win"),
        
        # Real game positions
        ("#13,a6i2d10d9f8e9g9g10i9h9i8h8i7j4g6g7f7h6g8f10h7i10j10j11h10g4e5e4f4g12i11g2h2g3h4h13g11f12f11e12l11k12h12g13l12k10i12h3j3i4i3h5g5j2k2j12i13h11j9f3d5k1l1", "blue", "Real game blue win"),
        ("#13,a12g5f5f6j4j5h6i4i5h7i7i6g7f9g8g10i9k2h4h8g9f10h9i10h10g12g11f12f11e12e11d12h11h12i11i12j11j12d11c12a13b11a11b10a10b9a9b8a8b7a7b6c11b12a6b5a5b3b4c3c4d3d4e3e4f3g4h2i3j1l2k3i2i1l3k4j2k1l4k6l5l7l6k7l11k12l12k13l13e5f4g2k9l10m7m8m6l8k10e6j3m1j6k5a3a4", "red", "Real game red win"),
    ]
    
    for trmph, expected_winner, description in test_cases:
        state = HexGameState.from_trmph(trmph)
        actual_winner = state._find_winner()
        assert actual_winner == expected_winner, f"{description}: Expected {expected_winner}, got {actual_winner}"
        assert state.game_over, f"{description}: Game should be over"
        assert state.winner == expected_winner, f"{description}: State winner should be {expected_winner}"

def test_one_move_to_win_positions():
    """Test positions where one more move creates a win."""
    from hex_ai.inference.game_engine import HexGameState
    
    # Test cases: (trmph_string, winning_move, expected_winner, description)
    test_cases = [
        ("#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7", (6, 6), "blue", "Blue wins with g7"),
        ("#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7a2", (6, 6), "red", "Red wins with g7"),
    ]
    
    for trmph, winning_move, expected_winner, description in test_cases:
        # Test that game is not over before the winning move
        state = HexGameState.from_trmph(trmph)
        assert not state.game_over, f"{description}: Game should not be over before winning move"
        assert state._find_winner() is None, f"{description}: No winner before winning move"
        
        # Test that the winning move creates a win
        new_state = state.make_move(*winning_move)
        assert new_state.game_over, f"{description}: Game should be over after winning move"
        assert new_state.winner == expected_winner, f"{description}: Expected {expected_winner} win after move {winning_move}"
        assert new_state._find_winner() == expected_winner, f"{description}: _find_winner should return {expected_winner}"

def test_tournament_structural_integrity_with_known_positions():
    """Test structural integrity using known positions to ensure moves are legal and board updates correctly."""
    from hex_ai.inference.game_engine import HexGameState
    
    # Test a simple position
    trmph = "#13,g1a7g2b7g3c7g4d7g5e7g6f7g8h7g9i7g10j7g11k7g12l7g13m7g7"
    state = HexGameState.from_trmph(trmph)
    
    # Verify the game is over
    assert state.game_over, "Game should be over"
    assert state.winner == "blue", "Winner should be blue"
    
    # Verify all moves in the sequence are legal
    test_state = HexGameState()
    seen_moves = set()
    
    for move in state.move_history:
        row, col = move
        # Check move is legal
        assert test_state.is_valid_move(row, col), f"Move {move} should be legal"
        # Check no duplicate moves
        assert (row, col) not in seen_moves, f"Duplicate move {move} detected"
        seen_moves.add((row, col))
        
        # Apply move
        test_state = test_state.make_move(row, col)
    
    # Verify final state matches
    assert test_state.game_over == state.game_over
    assert test_state.winner == state.winner 

# ============================================================================
# TOURNAMENT RESULT API TESTS - Testing the data structures and calculations
# ============================================================================

def test_tournament_result_api():
    """Test TournamentResult API with simulated games."""
    # Simulate a mini-tournament
    result = tournament.TournamentResult(CHKPT_PATHS)
    result.record_game(CHKPT_PATHS[0], CHKPT_PATHS[1])
    result.record_game(CHKPT_PATHS[1], CHKPT_PATHS[0])
    # Should be 2 games
    assert result.total_games == 2
    # Win rates should be 0.5 each
    win_rates = result.win_rates()
    assert abs(win_rates[CHKPT_PATHS[0]] - 0.5) < 1e-6
    assert abs(win_rates[CHKPT_PATHS[1]] - 0.5) < 1e-6
    # Elo keys match participants
    elos = result.elo_ratings()
    assert set(elos.keys()) == set(CHKPT_PATHS)

def test_draw_handling():
    """Test tournament result handling with no games (all win rates should be 0)."""
    # Simulate draws by directly manipulating TournamentResult
    result = tournament.TournamentResult(CHKPT_PATHS)
    # No games played: all win rates should be 0
    win_rates = result.win_rates()
    assert all(v == 0.0 for v in win_rates.values())
    # Record a win and a loss
    result.record_game(CHKPT_PATHS[0], CHKPT_PATHS[1])
    result.record_game(CHKPT_PATHS[1], CHKPT_PATHS[0])
    win_rates = result.win_rates()
    assert abs(win_rates[CHKPT_PATHS[0]] - 0.5) < 1e-6
    assert abs(win_rates[CHKPT_PATHS[1]] - 0.5) < 1e-6

def test_print_methods():
    """Test that print methods don't raise exceptions."""
    result = tournament.TournamentResult(CHKPT_PATHS)
    # Should not raise
    result.print_summary()
    result.print_elo()

# ============================================================================
# LOGGING TESTS - Testing tournament logging functionality
# ============================================================================

@timed_test
def test_append_trmph_winner_line():
    """Test appending TRMPH winner lines to log file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "tournament.log")
        tournament_logging.append_trmph_winner_line("a1b2c3", "b", out_file)
        tournament_logging.append_trmph_winner_line("d4e5f6", "r", out_file)
        with open(out_file, 'r') as f:
            lines = f.readlines()
        assert lines == ["a1b2c3 b\n", "d4e5f6 r\n"]

@timed_test
def test_log_game_csv():
    """Test logging game results to CSV file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "games.csv")
        row1 = {"model_a": "A", "model_b": "B", "color_a": "blue", "trmph": "a1b2", "winner": "b"}
        row2 = {"model_a": "B", "model_b": "A", "color_a": "red", "trmph": "c3d4", "winner": "r"}
        tournament_logging.log_game_csv(row1, csv_file)
        tournament_logging.log_game_csv(row2, csv_file)
        import csv
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["model_a"] == "A"
        assert rows[1]["winner"] == "r" 

# ============================================================================
# INTEGRATION TESTS - Moved to test_tournament_integration.py
# ============================================================================

# Integration tests that require real model checkpoints have been moved to
# tests/test_tournament_integration.py to keep this file focused on unit tests.

# ============================================================================
# TODO: TESTS WE WANT TO IMPLEMENT BUT HAVEN'T YET
# ============================================================================

"""
TODO: TESTS TO IMPLEMENT

1. PIE RULE LOGIC TESTS:
   - Test pie rule swap logic with simple, robust mocks
   - Test that swap decisions are correctly logged
   - Test pie rule threshold edge cases
   - Test pie rule with different temperature settings

2. TOURNAMENT GAME PLAY TESTS:
   - Test single game play with deterministic models that make known moves
   - Test that games always end with a winner (no infinite loops)
   - Test move legality and board state consistency
   - Test that all moves are unique and valid

3. RANDOM SEED REPRODUCIBILITY:
   - Test that same random seed produces identical results
   - Test that different seeds produce different results
   - Test reproducibility with pie rule enabled/disabled

4. TOURNAMENT LOGIC TESTS:
   - Test winner attribution with known outcomes
   - Test that tournament results correctly track wins/losses
   - Test round-robin pairing logic
   - Test tournament with odd number of participants

5. MODEL INTERFACE TESTS:
   - Test that models are called with correct board states
   - Test that model outputs are properly interpreted
   - Test error handling for invalid model outputs
   - Test model loading and initialization

6. INTEGRATION TESTS:
   - Test full tournament pipeline from config to results
   - Test CSV logging and file output
   - Test tournament with different board sizes
   - Test tournament with different game configurations

7. EDGE CASE TESTS:
   - Test tournament with single participant
   - Test tournament with very large number of games
   - Test tournament with models that fail to load
   - Test tournament with corrupted checkpoint files

8. PERFORMANCE TESTS:
   - Test tournament timing and resource usage
   - Test memory usage with large tournaments
   - Test parallel game execution (if implemented)

IMPLEMENTATION NOTES:
- Use simple, well-defined mocks that don't require complex patching
- Focus on testing the logic, not the implementation details
- Use dependency injection where possible instead of monkeypatching
- Keep tests fast and reliable
- Use fixtures for common setup
- Mark slow tests appropriately
""" 