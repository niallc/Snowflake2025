# Integration tests for tournament functionality with real model checkpoints
# These tests are slow and require actual model files to be present

import pytest
import os
import tempfile
import time
from hex_ai.inference import tournament
from hex_ai.inference.game_engine import HexGameState

ALL_RESULTS_DIR = "checkpoints/hyperparameter_tuning/"
THIS_MODEL_DIR = "loss_weight_sweep_exp0_bs256_98f719_20250724_233408"
CHKPT_FILENAMES = [
    "epoch1_mini1.pt",
    "epoch1_mini5.pt"
]
CHKPT_PATHS = [os.path.join(ALL_RESULTS_DIR, THIS_MODEL_DIR, fname) for fname in CHKPT_FILENAMES]

SLOW = pytest.mark.slow

def timed_test(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[TIMING] {func.__name__}: {elapsed:.2f} seconds")
        return result
    return wrapper

# ============================================================================
# INTEGRATION TESTS - Full tournament runs with real checkpoints
# ============================================================================

@SLOW
@pytest.mark.skipif(not all(os.path.exists(p) for p in CHKPT_PATHS), 
                    reason="Checkpoints not found; skipping tournament test.")
@timed_test
def test_round_robin_tournament():
    """Test full round-robin tournament with real model checkpoints."""
    config = tournament.TournamentConfig(checkpoint_paths=CHKPT_PATHS, num_games=2)
    result = tournament.run_round_robin_tournament(config, verbose=False)
    # Print for manual inspection
    result.print_summary()
    result.print_elo()
    # Assert total games
    expected_games = len(CHKPT_PATHS) * (len(CHKPT_PATHS) - 1) * config.num_games
    assert result.total_games == expected_games
    # Assert win rates sum to 1 (or close, allowing for draws)
    win_rates = result.win_rates()
    assert all(0.0 <= v <= 1.0 for v in win_rates.values())
    # Elo keys match participants
    elos = result.elo_ratings()
    assert set(elos.keys()) == set(CHKPT_PATHS)

@SLOW
@pytest.mark.skipif(len(CHKPT_PATHS) < 2 or not all(os.path.exists(p) for p in CHKPT_PATHS), 
                    reason="Not enough checkpoints for 3-way tournament.")
@timed_test
def test_three_way_tournament():
    """Test three-way tournament with real model checkpoints."""
    # Use epoch1_mini20.pt as the third checkpoint
    extra_chkpt = "epoch1_mini20.pt"
    extra_path = os.path.join(ALL_RESULTS_DIR, THIS_MODEL_DIR, extra_chkpt)
    if not os.path.exists(extra_path):
        pytest.skip("Third checkpoint not found.")
    paths = CHKPT_PATHS + [extra_path]
    config = tournament.TournamentConfig(checkpoint_paths=paths, num_games=1)
    result = tournament.run_round_robin_tournament(config, verbose=False)
    assert result.total_games == 6
    win_rates = result.win_rates()
    assert set(win_rates.keys()) == set(paths)
    elos = result.elo_ratings()
    assert set(elos.keys()) == set(paths)

@SLOW
@pytest.mark.skipif(not all(os.path.exists(p) for p in CHKPT_PATHS), 
                    reason="Checkpoints not found; skipping search_widths test.")
@timed_test
def test_tournament_with_search_widths():
    """Test that tournament runs successfully with search_widths parameter specified."""
    config = tournament.TournamentConfig(checkpoint_paths=CHKPT_PATHS, num_games=1, search_widths=[2,2])
    result = tournament.run_round_robin_tournament(config, verbose=False)
    # Just check that it runs and produces results
    assert result.total_games == 2
    win_rates = result.win_rates()
    assert all(0.0 <= v <= 1.0 for v in win_rates.values())

# ============================================================================
# DEBUGGING TESTS - To help understand why games end with no winner
# ============================================================================

@SLOW
@pytest.mark.skipif(not all(os.path.exists(p) for p in CHKPT_PATHS), 
                    reason="Checkpoints not found; skipping model inference test.")
def test_model_inference_and_move_selection():
    """Test basic model inference and move selection functionality."""
    from hex_ai.inference.tournament import play_single_game, TournamentPlayConfig
    from hex_ai.inference.simple_model_inference import SimpleModelInference
    
    # Load a single model and test inference
    model_path = CHKPT_PATHS[0]
    print(f"Testing model: {model_path}")
    
    try:
        model = SimpleModelInference(model_path)
        print(f"Model loaded successfully")
        
        # Test inference on empty board
        state = HexGameState()
        print(f"Initial state: moves={len(state.move_history)}, board_sum={state.board.sum()}")
        
        # Test model inference
        policy_logits, value_logit = model.infer(state.board)
        print(f"Policy logits shape: {policy_logits.shape}, value_logit: {value_logit:.3f}")
        
        # Test move selection
        from hex_ai.inference.tournament import select_move
        move = select_move(state, model, search_widths=None, temperature=1.0)
        print(f"Selected move: {move}")
        
        # Test making the move
        new_state = state.make_move(*move)
        print(f"After move: moves={len(new_state.move_history)}, game_over={new_state.game_over}")
        
    except Exception as e:
        print(f"Error testing model: {e}")
        import traceback
        traceback.print_exc()

@SLOW
@pytest.mark.skipif(not all(os.path.exists(p) for p in CHKPT_PATHS), 
                    reason="Checkpoints not found; skipping full game test.")
def test_full_game_completion():
    """Test that a full game completes successfully with real models."""
    from hex_ai.inference.tournament import play_single_game, TournamentPlayConfig
    from hex_ai.inference.simple_model_inference import SimpleModelInference
    
    model_a = SimpleModelInference(CHKPT_PATHS[0])
    model_b = SimpleModelInference(CHKPT_PATHS[1])
    
    play_config = TournamentPlayConfig(pie_rule=False)
    
    # Test that a full game completes successfully
    result = play_single_game(model_a, model_b, 13, verbose=1, play_config=play_config)
    
    # Verify the game completed properly
    assert result.winner in ["1", "2"], f"Winner should be '1' or '2', got {result.winner}"
    assert result.winner_char in ["b", "r"], f"Winner char should be 'b' or 'r', got {result.winner_char}"
    assert len(result.move_sequence) > 0, "Game should have at least one move"
    assert result.trmph_str.startswith("#13,"), "TRMPH string should start with '#13,'"
    
    print(f"Game completed successfully:")
    print(f"  Winner: Model {result.winner} ({result.winner_char})")
    print(f"  Moves: {len(result.move_sequence)}")
    print(f"  TRMPH: {result.trmph_str[:50]}...")

def test_state_updates_through_tournament_flow():
    """Test state updates through the tournament flow."""
    from hex_ai.inference.tournament import play_single_game, TournamentPlayConfig, play_game_loop, determine_winner
    from hex_ai.inference.simple_model_inference import SimpleModelInference
    from hex_ai.inference.game_engine import HexGameState
    
    model_a = SimpleModelInference(CHKPT_PATHS[0])
    model_b = SimpleModelInference(CHKPT_PATHS[1])
    
    play_config = TournamentPlayConfig(pie_rule=False)
    
    print("=== STATE UPDATE TEST ===")
    
    # Step 1: Initial state
    state = HexGameState()
    print(f"1. Initial state: moves={len(state.move_history)}, game_over={state.game_over}")
    
    # Step 2: After pie rule handling (should make first move)
    from hex_ai.inference.tournament import handle_pie_rule
    pie_result = handle_pie_rule(state, model_a, model_b, play_config, verbose=0)
    print(f"2. After pie rule: moves={len(state.move_history)}, game_over={state.game_over}")
    
    # Step 3: Test play_game_loop directly
    test_state = HexGameState()
    move_sequence, final_state = play_game_loop(test_state, model_a, model_b, None, play_config.temperature, verbose=0)
    print(f"3. After play_game_loop: moves={len(move_sequence)}, final_state_moves={len(final_state.move_history)}")
    
    # Step 4: Test determine_winner with the final_state
    try:
        winner_result, winner_char = determine_winner(final_state, model_a, model_b, False)
        print(f"4. Winner determination: SUCCESS - {winner_result} ({winner_char})")
    except Exception as e:
        print(f"4. Winner determination: ERROR - {e}")
    
    # Step 5: Full game test
    try:
        result = play_single_game(model_a, model_b, 13, verbose=0, play_config=play_config)
        print(f"5. Full game: SUCCESS - winner={result.winner}, moves={len(result.move_sequence)}")
    except Exception as e:
        print(f"5. Full game: ERROR - {e}")
        import traceback
        traceback.print_exc()

def test_move_application_to_state():
    """Test what happens when applying moves to state."""
    from hex_ai.inference.tournament import play_game_loop, select_move
    from hex_ai.inference.simple_model_inference import SimpleModelInference
    from hex_ai.inference.game_engine import HexGameState
    from hex_ai.value_utils import apply_move_to_state
    
    model_a = SimpleModelInference(CHKPT_PATHS[0])
    model_b = SimpleModelInference(CHKPT_PATHS[1])
    
    print("=== APPLY MOVE DEBUG TEST ===")
    
    # Test 1: Direct state manipulation
    print("1. Testing direct state manipulation:")
    state = HexGameState()
    print(f"   Initial state: moves={len(state.move_history)}, board_sum={state.board.sum()}")
    
    # Apply a move directly
    new_state = apply_move_to_state(state, 0, 0)
    print(f"   After apply_move_to_state(0,0): moves={len(new_state.move_history)}, board_sum={new_state.board.sum()}")
    print(f"   Original state unchanged: moves={len(state.move_history)}, board_sum={state.board.sum()}")
    
    # Test 2: Simulate play_game_loop logic
    print("2. Testing play_game_loop logic:")
    state = HexGameState()
    move_sequence = []
    
    print(f"   Initial state: moves={len(state.move_history)}, board_sum={state.board.sum()}")
    
    # Simulate one iteration of the loop
    move = select_move(state, model_a, None, 1.0)
    print(f"   Selected move: {move}")
    
    if move is not None:
        move_sequence.append(move)
        state = apply_move_to_state(state, *move)
        print(f"   After state = apply_move_to_state: moves={len(state.move_history)}, board_sum={state.board.sum()}")
    
    # Test 3: Check if state is actually being updated
    print("3. Testing state update verification:")
    state = HexGameState()
    original_id = id(state)
    print(f"   Original state id: {original_id}")
    
    state = apply_move_to_state(state, 0, 0)
    new_id = id(state)
    print(f"   New state id: {new_id}")
    print(f"   States are different: {original_id != new_id}")
    print(f"   New state moves: {len(state.move_history)}, board_sum: {state.board.sum()}")

def test_play_game_loop_step_by_step():
    """Test play_game_loop functionality step by step."""
    from hex_ai.inference.tournament import play_game_loop, select_move
    from hex_ai.inference.simple_model_inference import SimpleModelInference
    from hex_ai.inference.game_engine import HexGameState
    from hex_ai.value_utils import apply_move_to_state
    
    model_a = SimpleModelInference(CHKPT_PATHS[0])
    model_b = SimpleModelInference(CHKPT_PATHS[1])
    
    print("=== PLAY GAME LOOP DEBUG TEST ===")
    
    # Create a custom play_game_loop with debug output
    def debug_play_game_loop(state: HexGameState, model_1, model_2, search_widths, temperature, verbose):
        move_sequence = []
        iteration = 0
        
        print(f"Starting game loop with state: moves={len(state.move_history)}, game_over={state.game_over}")
        
        while not state.game_over:
            iteration += 1
            print(f"Iteration {iteration}:")
            print(f"  - state.game_over: {state.game_over}")
            print(f"  - state.current_player: {state.current_player}")
            print(f"  - state.move_history length: {len(state.move_history)}")
            print(f"  - state.board.sum(): {state.board.sum()}")
            
            # Determine which model to use
            model = model_1 if state.current_player == 0 else model_2  # 0 = BLUE_PLAYER
            print(f"  - Using model: {'model_1' if state.current_player == 0 else 'model_2'}")
            
            # Select and apply move
            move = select_move(state, model, search_widths, temperature)
            print(f"  - Selected move: {move}")
            
            if move is None:
                print(f"  - No valid moves, breaking")
                break  # No valid moves
            
            move_sequence.append(move)
            print(f"  - Added move to sequence, length now: {len(move_sequence)}")
            
            state = apply_move_to_state(state, *move)
            print(f"  - Applied move, new state: moves={len(state.move_history)}, board_sum={state.board.sum()}")
            
            if verbose >= 2:
                print("-", end="", flush=True)
        
        print(f"Game loop finished after {iteration} iterations")
        print(f"Final state: moves={len(state.move_history)}, game_over={state.game_over}, winner={state.winner}")
        print(f"Move sequence length: {len(move_sequence)}")
        
        return move_sequence
    
    # Test the debug version
    state = HexGameState()
    move_sequence = debug_play_game_loop(state, model_a, model_b, None, 1.0, 0)
    
    # Now test the real version
    print("\n=== TESTING REAL PLAY_GAME_LOOP ===")
    state = HexGameState()
    real_move_sequence = play_game_loop(state, model_a, model_b, None, 1.0, 0)
    print(f"Real play_game_loop returned {len(real_move_sequence)} moves")
    print(f"State after real play_game_loop: moves={len(state.move_history)}, board_sum={state.board.sum()}")

# ============================================================================
# TODO: ADDITIONAL INTEGRATION TESTS TO IMPLEMENT
# ============================================================================

# if __name__ == "__main__":
#     # For debugging: run the specific test directly
#     print("Running test_play_game_loop_debug directly...")
#     test_play_game_loop_debug()

"""
TODO: ADDITIONAL INTEGRATION TESTS

1. TOURNAMENT CONFIGURATION TESTS:
   - Test tournament with different board sizes
   - Test tournament with different temperature settings
   - Test tournament with pie rule enabled/disabled
   - Test tournament with different search widths

2. TOURNAMENT LOGGING TESTS:
   - Test CSV logging during tournament
   - Test TRMPH logging during tournament
   - Test tournament result serialization

3. TOURNAMENT PERFORMANCE TESTS:
   - Test tournament timing with different numbers of games
   - Test memory usage during large tournaments
   - Test tournament with many participants

4. ERROR HANDLING TESTS:
   - Test tournament with corrupted checkpoint files
   - Test tournament with missing checkpoint files
   - Test tournament with models that fail to load
   - Test tournament with models that fail during inference

5. REPRODUCIBILITY TESTS:
   - Test that same random seed produces identical results
   - Test that different seeds produce different results
   - Test reproducibility across different runs

IMPLEMENTATION NOTES:
- These tests should be marked as @SLOW
- Use @pytest.mark.skipif for missing dependencies
- Add proper error handling and debugging output
- Focus on testing the integration points, not unit functionality
- Keep tests focused and avoid testing too many things at once
""" 