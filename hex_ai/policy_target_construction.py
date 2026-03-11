"""
Shared policy-target construction helpers for self-play and tournaments.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from hex_ai.utils.format_conversion import rowcol_to_tensor_with_size, trmph_to_tensor


GUMBEL_V3_STAGE_GAP = 0.6
GUMBEL_V3_LOCAL_SCORE_WEIGHT = 0.30
GUMBEL_V3_SOFTMAX_TEMPERATURE = 0.85


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


def _extract_gumbel_stage_target_rows(
    stats: Dict[str, Any],
    *,
    action_count: int,
) -> List[Dict[str, float]]:
    """Validate and return compact per-candidate rows for v3 Gumbel targets."""
    rows = stats.get("gumbel_stage_target_rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Gumbel move is missing gumbel_stage_target_rows in MCTS stats; "
            "cannot build v3 Gumbel policy target."
        )

    extracted_rows: List[Dict[str, float]] = []
    seen_indices: set[int] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise TypeError(
                f"gumbel_stage_target_rows entries must be dict, got {type(row)}"
            )
        action_raw = row.get("tensor_action")
        stage_rank_raw = row.get("stage_rank")
        score_group_raw = row.get("score_group")
        score_raw = row.get("score_without_gumbel")
        if (
            action_raw is None
            or stage_rank_raw is None
            or score_group_raw is None
            or score_raw is None
        ):
            raise ValueError(
                "gumbel_stage_target_rows entries must include tensor_action, "
                "stage_rank, score_group, and score_without_gumbel"
            )
        action_idx = int(action_raw)
        if action_idx < 0 or action_idx >= action_count:
            raise ValueError(
                f"Invalid tensor_action {action_idx} for action_count {action_count}"
            )
        if action_idx in seen_indices:
            raise ValueError(
                f"Duplicate tensor_action {action_idx} in gumbel_stage_target_rows"
            )
        stage_rank = int(stage_rank_raw)
        score_group = int(score_group_raw)
        if stage_rank <= 0:
            raise ValueError(
                f"Invalid stage_rank for tensor_action {action_idx}: {stage_rank_raw!r}"
            )
        if score_group <= 0:
            raise ValueError(
                f"Invalid score_group for tensor_action {action_idx}: {score_group_raw!r}"
            )
        score = float(score_raw)
        if not np.isfinite(score):
            raise ValueError(
                "Non-finite score_without_gumbel for tensor_action "
                f"{action_idx}: {score_raw!r}"
            )
        seen_indices.add(action_idx)
        extracted_rows.append(
            {
                "tensor_action": action_idx,
                "stage_rank": stage_rank,
                "score_group": score_group,
                "score_without_gumbel": score,
            }
        )
    return extracted_rows


def build_policy_target_vector_from_gumbel_stage_scores(
    mcts_result: Any,
    *,
    board_size: int,
) -> np.ndarray:
    """
    Build dense v3 Gumbel policy target from elimination-stage summaries.

    Contract:
    - Uses `stats["gumbel_stage_target_rows"]`, a compact per-candidate summary
      produced by the Gumbel root-selection loop.
    - Each row carries:
      - `stage_rank`: later elimination / final survival gets a larger base
      - `score_group`: candidates sharing the same last-active comparison pool
      - `score_without_gumbel`: noise-free local ranking score
    - Normalizes local scores to [0,1] within each score group and combines them
      with the stage base:
        z(a) = 0.6 * stage_rank(a) + 0.30 * local_scaled_score(a)
      then applies softmax(z / 0.85) over the searched candidate set only.
    - Leaves all non-candidate legal actions at probability 0.
    - Enforces that the selected move is the unique top-scored action under the
      stored v3 target contract.
    """
    stats = getattr(mcts_result, "stats", {}) or {}
    action_count = board_size * board_size
    candidate_rows = _extract_gumbel_stage_target_rows(
        stats, action_count=action_count
    )

    grouped_scores: Dict[int, List[float]] = defaultdict(list)
    for row in candidate_rows:
        grouped_scores[int(row["score_group"])].append(
            float(row["score_without_gumbel"])
        )

    action_indices: List[int] = []
    target_logits: List[float] = []
    for row in candidate_rows:
        action_idx = int(row["tensor_action"])
        stage_rank = int(row["stage_rank"])
        score_group = int(row["score_group"])
        raw_score = float(row["score_without_gumbel"])
        score_pool = grouped_scores[score_group]
        pool_min = min(score_pool)
        pool_max = max(score_pool)
        if pool_max <= pool_min + 1e-12:
            local_scaled = 0.5
        else:
            local_scaled = (raw_score - pool_min) / (pool_max - pool_min)
        target_logit = (
            GUMBEL_V3_STAGE_GAP * float(stage_rank)
            + GUMBEL_V3_LOCAL_SCORE_WEIGHT * float(local_scaled)
        )
        action_indices.append(action_idx)
        target_logits.append(float(target_logit))

    selected_move = mcts_result.move
    selected_idx = rowcol_to_tensor_with_size(
        int(selected_move[0]), int(selected_move[1]), board_size
    )
    if selected_idx not in action_indices:
        raise RuntimeError(
            "Gumbel target construction mismatch: selected move is missing from "
            f"the searched candidate set (selected={selected_idx}, candidates={action_indices})."
        )

    logits_arr = np.asarray(target_logits, dtype=np.float64)
    top_idx = action_indices[int(np.argmax(logits_arr))]
    if selected_idx != top_idx:
        raise RuntimeError(
            "Gumbel v3 target construction mismatch: selected move is not the "
            "top-scored action under the stored stage-aware target "
            f"(selected={selected_idx}, top={top_idx})."
        )

    scaled_logits = logits_arr / float(GUMBEL_V3_SOFTMAX_TEMPERATURE)
    max_score = float(np.max(scaled_logits))
    exp_scores = np.exp(scaled_logits - max_score)
    exp_sum = float(np.sum(exp_scores))
    if exp_sum <= 0.0 or not np.isfinite(exp_sum):
        raise RuntimeError(
            "Invalid Gumbel v3 target normalization "
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
