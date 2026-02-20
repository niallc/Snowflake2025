from types import SimpleNamespace

import numpy as np
import pytest

import hex_ai.inference.mcts_utils as mcts_utils
from hex_ai.inference.mcts import BaselineMCTS
from hex_ai.inference.mcts_config import BaselineMCTSConfig
from hex_ai.inference.mcts_utils import calculate_visit_count_probs
from hex_ai.utils.temperature import calculate_mcts_root_temperature, calculate_temperature_decay


class _StubState:
    def __init__(self, board_size: int, move_count: int):
        self._board_tensor = np.zeros((board_size, board_size), dtype=np.int8)
        self.move_history = [None] * move_count

    def get_board_tensor(self):
        return self._board_tensor


class _StubRootNode:
    def __init__(self, counts: list[int], legal_moves: list[tuple[int, int]], board_size: int):
        self.N = np.array(counts, dtype=np.float64)
        self.legal_moves = legal_moves
        self.board_size = board_size


def test_root_temperature_method_uses_canonical_helper_contract():
    cfg = BaselineMCTSConfig(
        sims=8,
        batch_cap=32,
        temperature_start=1.0,
        temperature_end=0.2,
        temperature_decay_type="linear",
        temperature_decay_moves=20,
    )
    mcts = object.__new__(BaselineMCTS)
    mcts.cfg = cfg

    expected = calculate_mcts_root_temperature(move_count=7, cfg=cfg, board_size=11)
    observed = BaselineMCTS._root_temperature(mcts, move_idx=7, board_size=11)

    assert observed == pytest.approx(expected, rel=0.0, abs=1e-12)


def test_visit_count_probs_uses_canonical_temperature_helper(monkeypatch):
    cfg = BaselineMCTSConfig(
        sims=8,
        batch_cap=32,
        temperature_deterministic_cutoff=0.02,
    )
    root = _StubRootNode(counts=[3, 1], legal_moves=[(0, 0), (0, 1)], board_size=3)
    state = _StubState(board_size=3, move_count=4)

    captured = SimpleNamespace(move_count=None, cfg=None, board_size=None)

    def _stub_root_temp(move_count, cfg, board_size):
        captured.move_count = move_count
        captured.cfg = cfg
        captured.board_size = board_size
        return 0.01  # Force deterministic cutoff branch for stable expected probabilities.

    monkeypatch.setattr(mcts_utils, "calculate_mcts_root_temperature", _stub_root_temp)

    probs = calculate_visit_count_probs(root, state, cfg)

    assert captured.move_count == 4
    assert captured.cfg is cfg
    assert captured.board_size == 3
    assert sorted(probs.values()) == pytest.approx([0.25, 0.75], rel=0.0, abs=1e-12)


def test_baseline_mcts_config_rejects_unknown_temperature_decay_type():
    with pytest.raises(ValueError, match="temperature_decay_type must be one of"):
        BaselineMCTSConfig(
            sims=8,
            batch_cap=32,
            temperature_decay_type="unknown_mode",
        )


def test_temperature_decay_rejects_unknown_decay_type_runtime():
    with pytest.raises(ValueError, match="Unsupported temperature_decay_type"):
        calculate_temperature_decay(
            temperature_start=1.0,
            temperature_end=0.2,
            temperature_decay_type="unknown_mode",
            temperature_decay_moves=20,
            temperature_step_thresholds=[10, 20],
            temperature_step_values=[0.8, 0.5],
            move_count=3,
            board_size=11,
        )
