from types import SimpleNamespace

import numpy as np
import pytest

from hex_ai.policy_target_construction import (
    build_policy_target_vector_from_gumbel_stage_scores,
)


def test_build_policy_target_vector_from_gumbel_stage_scores_matches_v3_shape():
    # Toy 8-candidate sequential-halving history:
    # A chosen, B final loser, C/D round-2 losers, E/F/G/H round-1 losers.
    mcts_result = SimpleNamespace(
        move=(0, 0),
        stats={
            "gumbel_stage_target_rows": [
                {"tensor_action": 0, "stage_rank": 4, "score_group": 3, "score_without_gumbel": 1.30},
                {"tensor_action": 1, "stage_rank": 3, "score_group": 3, "score_without_gumbel": 1.12},
                {"tensor_action": 2, "stage_rank": 2, "score_group": 2, "score_without_gumbel": 0.90},
                {"tensor_action": 3, "stage_rank": 2, "score_group": 2, "score_without_gumbel": 0.50},
                {"tensor_action": 4, "stage_rank": 1, "score_group": 1, "score_without_gumbel": 0.55},
                {"tensor_action": 5, "stage_rank": 1, "score_group": 1, "score_without_gumbel": 0.30},
                {"tensor_action": 6, "stage_rank": 1, "score_group": 1, "score_without_gumbel": 0.00},
                {"tensor_action": 7, "stage_rank": 1, "score_group": 1, "score_without_gumbel": -0.25},
            ]
        },
    )

    vec = build_policy_target_vector_from_gumbel_stage_scores(
        mcts_result,
        board_size=3,
    )

    expected = np.array(
        [
            0.4611,
            0.1599,
            0.1124,
            0.0790,
            0.0555,
            0.0497,
            0.0435,
            0.0390,
            0.0000,
        ],
        dtype=np.float32,
    )

    np.testing.assert_allclose(vec, expected, atol=1e-4, rtol=0.0)
    assert int(np.argmax(vec)) == 0
    assert np.isclose(float(vec.sum()), 1.0, atol=1e-6)


def test_build_policy_target_vector_from_gumbel_stage_scores_fails_when_selected_move_is_not_top():
    mcts_result = SimpleNamespace(
        move=(0, 1),
        stats={
            "gumbel_stage_target_rows": [
                {"tensor_action": 0, "stage_rank": 4, "score_group": 2, "score_without_gumbel": 1.30},
                {"tensor_action": 1, "stage_rank": 3, "score_group": 2, "score_without_gumbel": 1.12},
            ]
        },
    )

    with pytest.raises(RuntimeError, match="selected move is not the top-scored action"):
        build_policy_target_vector_from_gumbel_stage_scores(
            mcts_result,
            board_size=2,
        )
