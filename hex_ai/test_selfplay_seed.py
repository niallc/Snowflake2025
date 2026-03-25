from hex_ai.selfplay.selfplay_engine import NUMPY_SEED_MODULUS, SelfPlayEngine


def test_compute_game_seed_wraps_into_numpy_uint32_range():
    engine = object.__new__(SelfPlayEngine)
    engine.run_seed = NUMPY_SEED_MODULUS - 500

    seed = engine._compute_game_seed(1)

    assert 0 <= seed < NUMPY_SEED_MODULUS
    assert seed == 500


def test_compute_game_seed_stays_deterministic_after_wrap():
    engine = object.__new__(SelfPlayEngine)
    engine.run_seed = NUMPY_SEED_MODULUS - 250

    assert engine._compute_game_seed(3) == (engine.run_seed + 3000) % NUMPY_SEED_MODULUS
    assert engine._compute_game_seed(3) == (engine.run_seed + 3000) % NUMPY_SEED_MODULUS
