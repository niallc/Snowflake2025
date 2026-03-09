"""
Shared policy-target construction helpers for self-play and tournaments.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from hex_ai.utils.format_conversion import rowcol_to_tensor_with_size, trmph_to_tensor


def build_policy_target_vector_from_mcts_result(
    mcts_result: Any, *, board_size: int
) -> np.ndarray:
    """Build dense policy-target row from MCTS root visit distribution."""
    tree_data = getattr(mcts_result, "tree_data", {}) or {}
    mcts_probs_raw = tree_data.get("mcts_probabilities")
    if not isinstance(mcts_probs_raw, dict):
        raise ValueError(
            "MCTS result missing mcts_probabilities dict in tree_data; "
            "cannot build policy target vector."
        )

    vec = np.zeros(board_size * board_size, dtype=np.float32)
    for move_trmph, prob_raw in mcts_probs_raw.items():
        if not isinstance(move_trmph, str):
            raise TypeError(
                f"mcts_probabilities key must be str move, got {type(move_trmph)}"
            )
        prob = float(prob_raw)
        if not np.isfinite(prob) or prob < 0.0:
            raise ValueError(
                f"Invalid probability for move {move_trmph!r}: {prob_raw!r}"
            )
        tensor_idx = trmph_to_tensor(move_trmph, board_size=board_size)
        vec[tensor_idx] = prob

    total = float(vec.sum())
    if total <= 0.0:
        raise RuntimeError(
            "MCTS policy target vector has zero mass despite trainable MCTS move source."
        )
    if not np.isclose(total, 1.0, atol=1e-5):
        vec /= total
    return vec


def _extract_gumbel_top_m_candidate_rows(
    stats: Dict[str, Any],
    *,
    action_count: int,
) -> List[Dict[str, float]]:
    """Validate and return the recorded clean Gumbel top-m candidate rows."""
    rows = stats.get("gumbel_top_m_candidate_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Gumbel move is missing gumbel_top_m_candidate_rows in MCTS stats; "
            "cannot build Gumbel-specific policy target."
        )

    extracted_rows: List[Dict[str, float]] = []
    seen_indices: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(
                f"gumbel_top_m_candidate_rows entries must be dict, got {type(row)}"
            )
        action_raw = row.get("tensor_action")
        log_prior_raw = row.get("log_prior")
        if action_raw is None or log_prior_raw is None:
            raise ValueError(
                "gumbel_top_m_candidate_rows entries must include tensor_action and log_prior"
            )
        action_idx = int(action_raw)
        if action_idx < 0 or action_idx >= action_count:
            raise ValueError(
                f"Invalid tensor_action {action_idx} for action_count {action_count}"
            )
        if action_idx in seen_indices:
            raise ValueError(
                f"Duplicate tensor_action {action_idx} in gumbel_top_m_candidate_rows"
            )
        log_prior = float(log_prior_raw)
        if not np.isfinite(log_prior):
            raise ValueError(
                f"Non-finite log_prior for tensor_action {action_idx}: {log_prior_raw!r}"
            )
        seen_indices.add(action_idx)
        extracted_rows.append(
            {
                "tensor_action": action_idx,
                "log_prior": log_prior,
            }
        )
    return extracted_rows


def _compute_gumbel_policy_target_value_scale(
    visit_counts: List[int],
    *,
    gumbel_c_visit: float,
    gumbel_c_scale: float,
) -> float:
    """
    Convert search-scale Gumbel parameters into a milder training-target scale.
    """
    if not visit_counts:
        raise ValueError("visit_counts must be non-empty for Gumbel policy target construction")
    max_visits = max(int(visits) for visits in visit_counts)
    if max_visits <= 0:
        raise ValueError(
            "Gumbel policy target construction requires at least one visited candidate action"
        )

    c_visit = float(gumbel_c_visit)
    c_scale = float(gumbel_c_scale)
    return float(c_scale * max_visits / (c_visit + max_visits))


def build_policy_target_vector_from_gumbel_candidate_scores(
    mcts_result: Any,
    *,
    board_size: int,
    gumbel_c_visit: float,
    gumbel_c_scale: float,
) -> np.ndarray:
    """
    Build dense policy target from the full Gumbel top-m candidate set.

    Contract:
    - Uses `stats["gumbel_top_m_candidate_rows"]` for the clean pre-search
      log-priors of the searched candidate set.
    - Uses root child Q-values for visited candidates and `gumbel_v_pi_01`
      as the completed-Q fallback for any unvisited candidate.
    - Applies softmax over the searched candidate set only.
    - Leaves all non-candidate legal actions at probability 0.
    """
    stats = getattr(mcts_result, "stats", {}) or {}
    root = getattr(mcts_result, "root_node", None)
    if root is None:
        raise ValueError(
            "Gumbel move is missing root_node on MCTS result; "
            "cannot build Gumbel-specific policy target."
        )

    action_count = board_size * board_size
    candidate_rows = _extract_gumbel_top_m_candidate_rows(
        stats, action_count=action_count
    )
    v_pi_raw = stats.get("gumbel_v_pi_01")
    if v_pi_raw is None:
        raise ValueError(
            "Gumbel move is missing gumbel_v_pi_01 in MCTS stats; "
            "cannot build Gumbel-specific policy target."
        )
    v_pi_01 = float(v_pi_raw)
    if not np.isfinite(v_pi_01):
        raise ValueError(f"Invalid gumbel_v_pi_01 value: {v_pi_raw!r}")

    action_to_legal_idx = {
        int(action): idx for idx, action in enumerate(root.legal_indices)
    }
    candidate_visit_counts: List[int] = []
    for row in candidate_rows:
        action_idx = int(row["tensor_action"])
        legal_idx = action_to_legal_idx.get(action_idx)
        if legal_idx is None:
            raise RuntimeError(
                f"Gumbel candidate tensor_action {action_idx} missing from root legal_indices"
            )
        candidate_visit_counts.append(int(root.N[legal_idx]))

    value_scale = _compute_gumbel_policy_target_value_scale(
        candidate_visit_counts,
        gumbel_c_visit=gumbel_c_visit,
        gumbel_c_scale=gumbel_c_scale,
    )

    action_indices: List[int] = []
    raw_scores: List[float] = []
    for row in candidate_rows:
        action_idx = int(row["tensor_action"])
        log_prior = float(row["log_prior"])
        legal_idx = action_to_legal_idx[action_idx]
        visits = int(root.N[legal_idx])
        if visits > 0:
            q_01 = float((float(root.Q[legal_idx]) + 1.0) / 2.0)
        else:
            q_01 = v_pi_01
        raw_score = float(log_prior + value_scale * (q_01 - v_pi_01))
        action_indices.append(action_idx)
        raw_scores.append(raw_score)

    selected_move = mcts_result.move
    selected_idx = rowcol_to_tensor_with_size(
        int(selected_move[0]), int(selected_move[1]), board_size
    )
    if selected_idx not in action_indices:
        raise RuntimeError(
            "Gumbel target construction mismatch: selected move is missing from "
            f"the searched candidate set (selected={selected_idx}, candidates={action_indices})."
        )

    scores_arr = np.asarray(raw_scores, dtype=np.float64)
    max_score = float(np.max(scores_arr))
    exp_scores = np.exp(scores_arr - max_score)
    exp_sum = float(np.sum(exp_scores))
    if exp_sum <= 0.0 or not np.isfinite(exp_sum):
        raise RuntimeError(
            "Invalid Gumbel candidate-score normalization "
            "(non-positive/invalid softmax denominator)."
        )
    probs = exp_scores / exp_sum

    vec = np.zeros(action_count, dtype=np.float32)
    for action_idx, prob in zip(action_indices, probs):
        vec[action_idx] = float(prob)

    total = float(vec.sum())
    if total <= 0.0:
        raise RuntimeError("Gumbel policy target vector has zero mass.")
    if not np.isclose(total, 1.0, atol=1e-6):
        vec /= total
    return vec
