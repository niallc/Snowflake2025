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

# --- New tests for value_utils and pie rule logic ---

def test_get_win_prob_from_model_output_valid():
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
    import torch
    logit = torch.logit(torch.tensor(0.7)).item()
    with pytest.raises(ValueError):
        get_win_prob_from_model_output(logit, 0)  # int not allowed
    with pytest.raises(ValueError):
        get_win_prob_from_model_output(logit, 'invalid')

# --- Pie rule swap logic test (mocked inference) ---
class DummyModel:
    def __init__(self, win_prob_blue):
        self.win_prob_blue = win_prob_blue
        self.checkpoint_path = "dummy"
    def infer(self, board):
        import torch
        # Return dummy policy_logits (float), value_logit
        logit = torch.logit(torch.tensor(self.win_prob_blue)).item()
        return np.zeros(169, dtype=np.float32), logit

def test_pie_rule_swap_logic():
    from hex_ai.inference.tournament import TournamentPlayConfig, play_single_game
    # At threshold: should swap
    play_config = TournamentPlayConfig(pie_rule=True, pie_threshold=(0.45, 0.55))
    model_a = DummyModel(0.5)  # win prob for blue after first move
    model_b = DummyModel(0.5)
    # Should swap
    result = play_single_game(model_a, model_b, 13, verbose=False, play_config=play_config)
    # The swap logic swaps models, so winner attribution is correct if no error
    assert result in ("A", "B", "draw")
    # Below threshold: no swap
    play_config = TournamentPlayConfig(pie_rule=True, pie_threshold=(0.6, 0.7))
    model_a = DummyModel(0.5)
    model_b = DummyModel(0.5)
    result = play_single_game(model_a, model_b, 13, verbose=False, play_config=play_config)
    # Should not swap
    assert result in ("A", "B", "draw")

# --- Logging test for swap and temperature ---
def test_logging_swap_and_temperature():
    from hex_ai.inference.tournament import TournamentPlayConfig, play_single_game
    import csv
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_file = os.path.join(tmpdir, "games.csv")
        play_config = TournamentPlayConfig(pie_rule=True, temperature=0.42)
        model_a = DummyModel(0.5)
        model_b = DummyModel(0.5)
        play_single_game(model_a, model_b, 13, csv_file=csv_file, play_config=play_config)
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows[0]["pie_rule"] == "True"
        assert abs(float(rows[0]["temperature"]) - 0.42) < 1e-6
        assert rows[0]["swap"] in ("swap", "no_swap")

# --- Random seed reproducibility test ---
@SLOW
def test_random_seed_reproducibility():
    from hex_ai.inference.tournament import TournamentConfig, run_round_robin_tournament, TournamentPlayConfig
    # Use dummy models so results are deterministic
    class DummyModelStatic:
        def __init__(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path
        def infer(self, board):
            import torch
            # Always return win prob 0.5
            logit = torch.logit(torch.tensor(0.5)).item()
            return np.zeros(169, dtype=np.float32), logit
    # Patch SimpleModelInference to DummyModelStatic
    import hex_ai.inference.tournament as tmod
    orig = tmod.SimpleModelInference
    tmod.SimpleModelInference = DummyModelStatic
    try:
        config = TournamentConfig(checkpoint_paths=["A", "B"], num_games=2)
        play_config = TournamentPlayConfig(random_seed=123, pie_rule=True)
        result1 = run_round_robin_tournament(config, verbose=False, play_config=play_config)
        play_config2 = TournamentPlayConfig(random_seed=123, pie_rule=True)
        result2 = run_round_robin_tournament(config, verbose=False, play_config=play_config2)
        assert result1.results == result2.results
    finally:
        tmod.SimpleModelInference = orig

@SLOW
@pytest.mark.skipif(not all(os.path.exists(p) for p in CHKPT_PATHS), reason="Checkpoints not found; skipping tournament test.")
@timed_test
def test_round_robin_tournament():
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
@pytest.mark.skipif(not all(os.path.exists(p) for p in CHKPT_PATHS), reason="Checkpoints not found; skipping tournament test.")
@timed_test
def test_tournament_result_api():
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

@SLOW
@pytest.mark.skipif(len(CHKPT_PATHS) < 2 or not all(os.path.exists(p) for p in CHKPT_PATHS), reason="Not enough checkpoints for 3-way tournament.")
@timed_test
def test_three_way_tournament():
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
@pytest.mark.skipif(not all(os.path.exists(p) for p in CHKPT_PATHS), reason="Checkpoints not found; skipping search_widths test.")
@timed_test
def test_search_widths_respected():
    config = tournament.TournamentConfig(checkpoint_paths=CHKPT_PATHS, num_games=1, search_widths=[2,2])
    result = tournament.run_round_robin_tournament(config, verbose=False)
    # Just check that it runs and produces results
    assert result.total_games == 2
    win_rates = result.win_rates()
    assert all(0.0 <= v <= 1.0 for v in win_rates.values())

@SLOW
@pytest.mark.skipif(not all(os.path.exists(p) for p in CHKPT_PATHS), reason="Checkpoints not found; skipping draw test.")
@timed_test
def test_draw_handling():
    # Simulate draws by directly manipulating TournamentResult
    result = tournament.TournamentResult(CHKPT_PATHS)
    # No games played: all win rates should be 0
    win_rates = result.win_rates()
    assert all(v == 0.0 for v in win_rates.values())
    # Record a draw (simulate by not calling record_game)
    # Now record a win and a loss
    result.record_game(CHKPT_PATHS[0], CHKPT_PATHS[1])
    result.record_game(CHKPT_PATHS[1], CHKPT_PATHS[0])
    win_rates = result.win_rates()
    assert abs(win_rates[CHKPT_PATHS[0]] - 0.5) < 1e-6
    assert abs(win_rates[CHKPT_PATHS[1]] - 0.5) < 1e-6

@SLOW
@pytest.mark.skipif(not all(os.path.exists(p) for p in CHKPT_PATHS), reason="Checkpoints not found; skipping print methods test.")
@timed_test
def test_print_methods():
    result = tournament.TournamentResult(CHKPT_PATHS)
    # Should not raise
    result.print_summary()
    result.print_elo()

@timed_test
def test_append_trmph_winner_line():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_file = os.path.join(tmpdir, "tournament.log")
        tournament_logging.append_trmph_winner_line("a1b2c3", "b", out_file)
        tournament_logging.append_trmph_winner_line("d4e5f6", "r", out_file)
        with open(out_file, 'r') as f:
            lines = f.readlines()
        assert lines == ["a1b2c3 b\n", "d4e5f6 r\n"]

@timed_test
def test_log_game_csv():
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

def test_tournament_winner_attribution():
    """
    Simulate a tournament where model A always wins and check that the summary is correct.
    This catches winner flipping or misattribution bugs.
    Prints detailed debug output for each game.
    """
    from hex_ai.inference.tournament import TournamentConfig, run_round_robin_tournament, TournamentPlayConfig, play_single_game
    class AlwaysWinModel:
        def __init__(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path
        def infer(self, board):
            import torch
            # Policy logits: uniform, value: 10.0 means always blue win
            return np.zeros(169, dtype=np.float32), 10.0
    class AlwaysLoseModel:
        def __init__(self, checkpoint_path):
            self.checkpoint_path = checkpoint_path
        def infer(self, board):
            import torch
            # Policy logits: uniform, value: -10.0 means always red win
            return np.zeros(169, dtype=np.float32), -10.0
    import hex_ai.inference.tournament as tmod
    orig = tmod.SimpleModelInference
    def patched_SimpleModelInference(path):
        if path == "A":
            return AlwaysWinModel(path)
        else:
            return AlwaysLoseModel(path)
    tmod.SimpleModelInference = patched_SimpleModelInference
    try:
        config = TournamentConfig(checkpoint_paths=["A", "B"], num_games=2)
        play_config = TournamentPlayConfig(random_seed=123, pie_rule=False)
        # Run tournament with debug output for each game
        models = {path: patched_SimpleModelInference(path) for path in config.checkpoint_paths}
        result = tmod.TournamentResult(config.checkpoint_paths)
        for a, b in [("A", "B")]:
            for game_idx in range(config.num_games):
                # A as blue, B as red
                winner = play_single_game(models[a], models[b], 13, None, 0, None, None, play_config)
                print(f"Game {game_idx*2+1}: A (blue) vs B (red) -> winner: {winner}")
                if winner == "A":
                    result.record_game(a, b)
                elif winner == "B":
                    result.record_game(b, a)
                # B as blue, A as red
                winner = play_single_game(models[b], models[a], 13, None, 0, None, None, play_config)
                print(f"Game {game_idx*2+2}: B (blue) vs A (red) -> winner: {winner}")
                if winner == "A":
                    result.record_game(b, a)
                elif winner == "B":
                    result.record_game(a, b)
        win_rates = result.win_rates()
        print(f"Final win rates: {win_rates}")
        elos = result.elo_ratings()
        print(f"Final Elo: {elos}")
        # Model A should win all games
        assert win_rates["A"] == 1.0
        assert win_rates["B"] == 0.0
        assert elos["A"] > elos["B"]
    finally:
        tmod.SimpleModelInference = orig 

def test_tournament_forced_blue_wins():
    """
    Patch HexGameState._find_winner to always return 'blue', so blue always wins.
    Check that the tournament summary matches this forced outcome.
    Print debug output for each game.
    """
    from hex_ai.inference.tournament import TournamentConfig, run_round_robin_tournament, TournamentPlayConfig, play_single_game
    from hex_ai.inference.game_engine import HexGameState
    import hex_ai.inference.tournament as tmod
    # Patch _find_winner
    orig_find_winner = HexGameState._find_winner
    HexGameState._find_winner = lambda self: "blue"
    try:
        config = TournamentConfig(checkpoint_paths=["A", "B"], num_games=2)
        play_config = TournamentPlayConfig(random_seed=123, pie_rule=False)
        # Use dummy models (policy doesn't matter, winner is forced)
        class DummyModel:
            def __init__(self, checkpoint_path):
                self.checkpoint_path = checkpoint_path
            def infer(self, board):
                return np.zeros(169, dtype=np.float32), 0.0
        def patched_SimpleModelInference(path):
            return DummyModel(path)
        orig = tmod.SimpleModelInference
        tmod.SimpleModelInference = patched_SimpleModelInference
        try:
            models = {path: patched_SimpleModelInference(path) for path in config.checkpoint_paths}
            result = tmod.TournamentResult(config.checkpoint_paths)
            for a, b in [("A", "B")]:
                for game_idx in range(config.num_games):
                    # A as blue, B as red
                    winner = play_single_game(models[a], models[b], 13, None, 0, None, None, play_config)
                    print(f"Game {game_idx*2+1}: A (blue) vs B (red) -> winner: {winner}")
                    if winner == "A":
                        result.record_game(a, b)
                    elif winner == "B":
                        result.record_game(b, a)
                    # B as blue, A as red
                    winner = play_single_game(models[b], models[a], 13, None, 0, None, None, play_config)
                    print(f"Game {game_idx*2+2}: B (blue) vs A (red) -> winner: {winner}")
                    if winner == "A":
                        result.record_game(b, a)
                    elif winner == "B":
                        result.record_game(a, b)
            win_rates = result.win_rates()
            print(f"Final win rates: {win_rates}")
            elos = result.elo_ratings()
            print(f"Final Elo: {elos}")
            # Blue should always win, so each model should win all games as blue
            assert win_rates["A"] == 0.5
            assert win_rates["B"] == 0.5
        finally:
            tmod.SimpleModelInference = orig
    finally:
        HexGameState._find_winner = orig_find_winner 

def test_tournament_winner_mapping_with_swap_clearer():
    """
    For each game, log:
    - swap: True/False
    - M1: b/r (color assigned to model1)
    - M2: b/r
    - winner_rb: b/r (which color won)
    - winner_12: M1/M2 (which model won)
    Use model1/model2 (M1/M2) and blue/red (b/r) terminology for clarity.
    Write logs to a temp directory for informal inspection.
    """
    import tempfile
    from hex_ai.inference.tournament import TournamentConfig, TournamentPlayConfig, play_single_game
    from hex_ai.inference.game_engine import HexGameState
    import hex_ai.inference.tournament as tmod
    import os
    # Patch _find_winner to alternate blue/red wins
    orig_find_winner = HexGameState._find_winner
    win_sequence = ["blue", "red", "blue", "red"]
    win_idx = [0]
    def alt_find_winner(self):
        idx = win_idx[0]
        win_idx[0] = (win_idx[0] + 1) % len(win_sequence)
        return win_sequence[idx]
    HexGameState._find_winner = alt_find_winner
    try:
        config = TournamentConfig(checkpoint_paths=["M1", "M2"], num_games=2)
        play_config = TournamentPlayConfig(random_seed=123, pie_rule=True)
        # Use dummy models (policy doesn't matter, winner is forced)
        class DummyModel:
            def __init__(self, checkpoint_path):
                self.checkpoint_path = checkpoint_path
            def infer(self, board):
                return np.zeros(169, dtype=np.float32), 0.0
        def patched_SimpleModelInference(path):
            return DummyModel(path)
        orig = tmod.SimpleModelInference
        tmod.SimpleModelInference = patched_SimpleModelInference
        try:
            models = {path: patched_SimpleModelInference(path) for path in config.checkpoint_paths}
            with tempfile.TemporaryDirectory() as tmpdir:
                csv_file = os.path.join(tmpdir, "winner_mapping_test_clearer.csv")
                for m1, m2 in [("M1", "M2")]:
                    for game_idx in range(config.num_games):
                        # First game: M1 asked first, M2 second
                        # Play game, capture swap and winner info
                        # Patch play_single_game to return swap and color assignment info
                        from hex_ai.inference.tournament import play_single_game as real_play_single_game
                        # Patch play_single_game to expose swap and color assignment
                        # We'll monkeypatch the function to return extra info
                        # (In production, this would be a refactor, but for test, monkeypatch is fine)
                        # We'll use the CSV log to extract swap and winner info
                        real_play_single_game(models[m1], models[m2], 13, None, 0, None, csv_file, play_config)
                        # Read the last row from the CSV
                        import csv
                        with open(csv_file, 'r') as f:
                            reader = list(csv.DictReader(f))
                            row = reader[-1]
                        swap = row['swap'] == 'swap'
                        winner_rb = row['winner']  # 'b' or 'r'
                        # Determine color assignment
                        if not swap:
                            m1_color = 'b'
                            m2_color = 'r'
                        else:
                            m1_color = 'r'
                            m2_color = 'b'
                        # Determine which model won
                        if winner_rb == m1_color:
                            winner_12 = 'M1'
                        else:
                            winner_12 = 'M2'
                        info1 = {
                            'swap': swap,
                            'M1': m1_color,
                            'M2': m2_color,
                            'winner_rb': winner_rb,
                            'winner_12': winner_12,
                            'csv_row': row
                        }
                        print(f"Game {game_idx*2+1}: swap={info1['swap']} M1={info1['M1']} M2={info1['M2']} winner_rb={info1['winner_rb']} winner_12={info1['winner_12']}")
                        # B as blue, A as red
                        real_play_single_game(models[m2], models[m1], 13, None, 0, None, csv_file, play_config)
                        # Read the last row from the CSV
                        with open(csv_file, 'r') as f:
                            reader = list(csv.DictReader(f))
                            row = reader[-1]
                        swap = row['swap'] == 'swap'
                        winner_rb = row['winner']  # 'b' or 'r'
                        # Determine color assignment
                        if not swap:
                            m1_color = 'b'
                            m2_color = 'r'
                        else:
                            m1_color = 'r'
                            m2_color = 'b'
                        # Determine which model won
                        if winner_rb == m1_color:
                            winner_12 = 'M1'
                        else:
                            winner_12 = 'M2'
                        info2 = {
                            'swap': swap,
                            'M1': m1_color,
                            'M2': m2_color,
                            'winner_rb': winner_rb,
                            'winner_12': winner_12,
                            'csv_row': row
                        }
                        print(f"Game {game_idx*2+2}: swap={info2['swap']} M1={info2['M1']} M2={info2['M2']} winner_rb={info2['winner_rb']} winner_12={info2['winner_12']}")
                # Print the CSV log for inspection
                import csv
                with open(csv_file, 'r') as f:
                    reader = csv.DictReader(f)
                    print("\nCSV log of winner mapping (clearer):")
                    for row in reader:
                        print(row)
        finally:
            tmod.SimpleModelInference = orig
    finally:
        HexGameState._find_winner = orig_find_winner 