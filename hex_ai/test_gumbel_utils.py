import math

import numpy as np

from hex_ai.utils.gumbel_utils import (
    build_gumbel_score_rows,
    compute_completed_baseline_v_pi,
)


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
