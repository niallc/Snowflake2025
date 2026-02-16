import math
from types import SimpleNamespace

import numpy as np
import pytest

from hex_ai.inference.mcts import BaselineMCTS
from hex_ai.inference.mcts_config import BaselineMCTSConfig
from hex_ai.utils.legal_action_contracts import (
    assert_actions_subset_of_legal,
    assert_exact_legal_action_match,
)
from hex_ai.utils.gumbel_utils import (
    build_gumbel_score_rows,
    compute_completed_baseline_v_pi,
    gumbel_alpha_zero_root_batched,
)


class _StubRoot:
    def __init__(self, legal_indices):
        self.legal_indices = list(legal_indices)


class _StubForcedRunner:
    def __init__(self, mutate_root_after_first_call: bool = False):
        self.calls = 0
        self._mutate_root_after_first_call = mutate_root_after_first_call

    def run_forced_root_actions(self, root, actions, verbose=0):
        self.calls += 1
        if self._mutate_root_after_first_call and self.calls == 1:
            root.legal_indices = root.legal_indices[:-1]
        return {"batch_count": 1}


def test_compute_completed_baseline_v_pi_with_visited_and_unvisited_actions():
    pi = np.array([0.5, 0.3, 0.2], dtype=np.float64)
    legal_actions = [0, 1, 2]
    q_map = {0: 0.8, 2: 0.2}
    n_map = {0: 5, 1: 0, 2: 2}

    def q_of_child(action: int) -> float:
        return q_map[action]

    def n_of_child(action: int) -> int:
        return n_map[action]

    value = compute_completed_baseline_v_pi(pi, legal_actions, q_of_child, n_of_child)
    expected = (0.5 * 0.8 + 0.2 * 0.2) / (1.0 - 0.3)
    assert math.isclose(value, expected, rel_tol=0.0, abs_tol=1e-12)


def test_compute_completed_baseline_v_pi_falls_back_to_neutral_when_all_unvisited():
    pi = np.array([0.6, 0.4], dtype=np.float64)
    legal_actions = [0, 1]

    def q_of_child(_: int) -> float:
        return 0.9

    def n_of_child(_: int) -> int:
        return 0

    value = compute_completed_baseline_v_pi(pi, legal_actions, q_of_child, n_of_child)
    assert value == 0.5


def test_build_gumbel_score_rows_reports_components_and_respects_score_mode():
    pi = np.array([0.7, 0.3], dtype=np.float64)
    logits = np.log(pi)
    gumbel_noise = np.array([0.4, -0.1], dtype=np.float64)
    actions = [0, 1]
    c_scale = 2.0
    v_pi_01 = 0.6
    q_map = {0: 0.9}
    n_map = {0: 4, 1: 0}

    def q_of_child(action: int) -> float:
        return q_map[action]

    def n_of_child(action: int) -> int:
        return n_map[action]

    rows_with_g = build_gumbel_score_rows(
        actions=actions,
        pi=pi,
        logits=logits,
        gumbel_noise=gumbel_noise,
        c_scale=c_scale,
        v_pi_01=v_pi_01,
        q_of_child=q_of_child,
        n_of_child=n_of_child,
        include_gumbel_term=True,
    )
    assert [row["tensor_action"] for row in rows_with_g] == [0, 1]

    top = rows_with_g[0]
    expected_top_without_g = float(logits[0] + c_scale * (0.9 - v_pi_01))
    expected_top_with_g = expected_top_without_g + float(gumbel_noise[0])
    assert top["q_source"] == "tree"
    assert math.isclose(top["score_without_gumbel"], expected_top_without_g, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(top["score_with_gumbel"], expected_top_with_g, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(top["score"], expected_top_with_g, rel_tol=0.0, abs_tol=1e-12)

    second = rows_with_g[1]
    expected_second_without_g = float(logits[1])  # adv == 0 for v_pi completion
    expected_second_with_g = expected_second_without_g + float(gumbel_noise[1])
    assert second["q_source"] == "v_pi_completion"
    assert second["q_01"] == v_pi_01
    assert second["adv_01"] == 0.0
    assert math.isclose(second["score_without_gumbel"], expected_second_without_g, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(second["score_with_gumbel"], expected_second_with_g, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(second["score"], expected_second_with_g, rel_tol=0.0, abs_tol=1e-12)

    rows_without_g = build_gumbel_score_rows(
        actions=actions,
        pi=pi,
        logits=logits,
        gumbel_noise=gumbel_noise,
        c_scale=c_scale,
        v_pi_01=v_pi_01,
        q_of_child=q_of_child,
        n_of_child=n_of_child,
        include_gumbel_term=False,
    )
    for row in rows_without_g:
        assert row["gumbel_term"] == 0.0
        assert math.isclose(
            row["score"],
            row["score_without_gumbel"],
            rel_tol=0.0,
            abs_tol=1e-12,
        )


def test_assert_exact_legal_action_match_rejects_mismatched_actions():
    with pytest.raises(ValueError, match="Expected exact legal-action match"):
        assert_exact_legal_action_match(
            expected_actions=[0, 1, 4],
            observed_actions=[0, 1, 2],
            context="unit_test",
            contract_name="Gumbel legality contract",
            expected_label="Provided legal_actions",
            observed_label="Current root legal_indices",
        )


def test_gumbel_alpha_zero_root_batched_fails_fast_on_entry_legality_mismatch():
    root = _StubRoot([0, 1, 2])
    mcts = _StubForcedRunner()
    policy_logits = np.log(np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64))

    def q_of_child(_: int) -> float:
        return 0.5

    def n_of_child(_: int) -> int:
        return 0

    with pytest.raises(ValueError, match="gumbel_alpha_zero_root_batched:entry"):
        gumbel_alpha_zero_root_batched(
            mcts=mcts,
            root=root,
            policy_logits=policy_logits,
            total_sims=4,
            legal_actions=[0, 1, 3],
            q_of_child=q_of_child,
            n_of_child=n_of_child,
            rng=np.random.RandomState(0),
        )

    assert mcts.calls == 0


def test_gumbel_alpha_zero_root_batched_fails_when_root_legality_drifts_between_rounds():
    root = _StubRoot([0, 1, 2, 3])
    mcts = _StubForcedRunner(mutate_root_after_first_call=True)
    policy_logits = np.log(np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64))

    def q_of_child(_: int) -> float:
        return 0.5

    def n_of_child(_: int) -> int:
        return 0

    with pytest.raises(ValueError, match="round_2_pre_forced_actions"):
        gumbel_alpha_zero_root_batched(
            mcts=mcts,
            root=root,
            policy_logits=policy_logits,
            total_sims=8,
            legal_actions=[0, 1, 2, 3],
            q_of_child=q_of_child,
            n_of_child=n_of_child,
            m=4,
            rng=np.random.RandomState(0),
        )

    assert mcts.calls == 1


def test_gumbel_alpha_zero_root_batched_preserves_valid_path_behavior():
    root = _StubRoot([0, 1, 2, 3])
    mcts = _StubForcedRunner()
    policy_logits = np.log(np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64))

    def q_of_child(_: int) -> float:
        return 0.5

    def n_of_child(_: int) -> int:
        return 0

    selected_action, metrics = gumbel_alpha_zero_root_batched(
        mcts=mcts,
        root=root,
        policy_logits=policy_logits,
        total_sims=8,
        legal_actions=[0, 1, 2, 3],
        q_of_child=q_of_child,
        n_of_child=n_of_child,
        m=4,
        rng=np.random.RandomState(0),
    )

    assert selected_action in root.legal_indices
    assert metrics["selected_action"] == selected_action
    assert mcts.calls == 2


def test_gumbel_alpha_zero_root_batched_rejects_non_unit_temperature():
    root = _StubRoot([0, 1, 2, 3])
    mcts = _StubForcedRunner()
    policy_logits = np.log(np.array([0.4, 0.3, 0.2, 0.1], dtype=np.float64))

    def q_of_child(_: int) -> float:
        return 0.5

    def n_of_child(_: int) -> int:
        return 0

    with pytest.raises(ValueError, match="temperature must be 1.0"):
        gumbel_alpha_zero_root_batched(
            mcts=mcts,
            root=root,
            policy_logits=policy_logits,
            total_sims=8,
            legal_actions=[0, 1, 2, 3],
            q_of_child=q_of_child,
            n_of_child=n_of_child,
            m=4,
            temperature=0.75,
            rng=np.random.RandomState(0),
        )

    assert mcts.calls == 0


def test_baseline_mcts_config_rejects_unsupported_gumbel_temperature_controls():
    with pytest.raises(ValueError, match="gumbel_temperature_enabled=False is unsupported"):
        BaselineMCTSConfig(
            sims=4,
            batch_cap=64,
            gumbel_temperature_enabled=False,
        )

    with pytest.raises(ValueError, match="gumbel_temperature_deterministic_cutoff is unsupported"):
        BaselineMCTSConfig(
            sims=4,
            batch_cap=64,
            gumbel_temperature_deterministic_cutoff=0.1,
        )


def test_validate_gumbel_temperature_contract_fails_fast_after_runtime_mutation():
    mcts = object.__new__(BaselineMCTS)
    mcts.cfg = SimpleNamespace(
        gumbel_temperature_enabled=False,
        gumbel_temperature_deterministic_cutoff=-1.0,
    )

    with pytest.raises(ValueError, match="gumbel_temperature_enabled=False is unsupported"):
        BaselineMCTS._validate_gumbel_temperature_contract(mcts)

    mcts.cfg = SimpleNamespace(
        gumbel_temperature_enabled=True,
        gumbel_temperature_deterministic_cutoff=0.1,
    )
    with pytest.raises(ValueError, match="gumbel_temperature_deterministic_cutoff is unsupported"):
        BaselineMCTS._validate_gumbel_temperature_contract(mcts)


def test_run_forced_root_actions_validates_before_processing_any_batch():
    mcts = object.__new__(BaselineMCTS)
    mcts.cfg = SimpleNamespace(batch_cap=4)
    mcts._effective_sims_total = 0
    batch_calls = {"count": 0}

    def _stub_run_forced_root_batch(root, actions, timing_tracker):
        batch_calls["count"] += 1
        return len(actions)

    mcts._run_forced_root_batch = _stub_run_forced_root_batch
    root = _StubRoot([0, 1, 2])

    with pytest.raises(ValueError, match="run_forced_root_actions:entry"):
        BaselineMCTS.run_forced_root_actions(mcts, root, [0, 99, 1])

    assert batch_calls["count"] == 0


def test_assert_actions_subset_of_legal_reports_illegal_positions():
    with pytest.raises(ValueError, match="Illegal positions in action list"):
        assert_actions_subset_of_legal(
            actions=[2, 9, 2],
            legal_actions=[0, 1, 2],
            context="subset_test",
            contract_name="Forced-root legality contract",
            actions_label="Forced actions",
            legal_label="Current root legal_indices",
        )
