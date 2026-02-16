from collections import deque
from types import SimpleNamespace

import pytest

from hex_ai.inference.mcts import BaselineMCTS
from hex_ai.utils.timing import MCTSTimingTracker


def test_run_standard_simulation_loop_fails_fast_on_zero_progress_batch():
    mcts = object.__new__(BaselineMCTS)
    mcts._effective_sims_total = 0
    calls = SimpleNamespace(select=0, process=0)

    def _stub_select_leaves_batch(root, sims_remaining, timing_tracker):
        _ = root
        _ = sims_remaining
        _ = timing_tracker
        calls.select += 1
        return [object(), object()], [[], []]

    def _stub_process_leaves_batch(leaves, paths, timing_tracker, root):
        _ = leaves
        _ = paths
        _ = timing_tracker
        _ = root
        calls.process += 1
        return 0

    mcts._select_leaves_batch = _stub_select_leaves_batch
    mcts._process_leaves_batch = _stub_process_leaves_batch

    with pytest.raises(
        ValueError,
        match="Zero-progress standard simulation batch in _run_standard_simulation_loop",
    ) as err:
        BaselineMCTS._run_standard_simulation_loop(
            mcts,
            root=object(),
            sims_remaining=5,
            timing_tracker=MCTSTimingTracker(),
        )

    message = str(err.value)
    assert "sims_remaining=5" in message
    assert "selected_leaves=2" in message
    assert "selected_paths=2" in message
    assert calls.select == 1
    assert calls.process == 1
    assert mcts._effective_sims_total == 0


def test_run_standard_simulation_loop_preserves_normal_progress_path():
    mcts = object.__new__(BaselineMCTS)
    mcts._effective_sims_total = 0
    batch_sizes = deque([2, 2, 1])
    seen_sims_remaining = []

    def _stub_select_leaves_batch(root, sims_remaining, timing_tracker):
        _ = root
        _ = timing_tracker
        seen_sims_remaining.append(sims_remaining)
        batch_size = batch_sizes[0]
        return [object()] * batch_size, [[] for _ in range(batch_size)]

    def _stub_process_leaves_batch(leaves, paths, timing_tracker, root):
        _ = timing_tracker
        _ = root
        expected = batch_sizes.popleft()
        assert len(leaves) == expected
        assert len(paths) == expected
        return expected

    mcts._select_leaves_batch = _stub_select_leaves_batch
    mcts._process_leaves_batch = _stub_process_leaves_batch

    BaselineMCTS._run_standard_simulation_loop(
        mcts,
        root=object(),
        sims_remaining=5,
        timing_tracker=MCTSTimingTracker(),
    )

    assert seen_sims_remaining == [5, 3, 1]
    assert not batch_sizes
    assert mcts._effective_sims_total == 5


def test_run_standard_simulation_loop_fails_fast_on_over_progress_batch():
    mcts = object.__new__(BaselineMCTS)
    mcts._effective_sims_total = 0
    calls = SimpleNamespace(select=0, process=0)

    def _stub_select_leaves_batch(root, sims_remaining, timing_tracker):
        _ = root
        _ = sims_remaining
        _ = timing_tracker
        calls.select += 1
        return [object(), object(), object()], [[], [], []]

    def _stub_process_leaves_batch(leaves, paths, timing_tracker, root):
        _ = leaves
        _ = paths
        _ = timing_tracker
        _ = root
        calls.process += 1
        return 3

    mcts._select_leaves_batch = _stub_select_leaves_batch
    mcts._process_leaves_batch = _stub_process_leaves_batch

    with pytest.raises(
        ValueError,
        match="Over-progress standard simulation batch in _run_standard_simulation_loop",
    ) as err:
        BaselineMCTS._run_standard_simulation_loop(
            mcts,
            root=object(),
            sims_remaining=2,
            timing_tracker=MCTSTimingTracker(),
        )

    message = str(err.value)
    assert "sims_remaining=2" in message
    assert "batch_simulations=3" in message
    assert "selected_leaves=3" in message
    assert "selected_paths=3" in message
    assert calls.select == 1
    assert calls.process == 1
    assert mcts._effective_sims_total == 0
