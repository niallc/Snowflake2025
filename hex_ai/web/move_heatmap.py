"""Utilities for computing per-move study heatmaps for Hex positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import hex_ai.utils.format_conversion as fc
from hex_ai.inference.game_engine import HexGameState, apply_move_to_state_trmph
from hex_ai.value_utils import (
    ValuePredictor,
    get_legal_policy_probs,
    policy_logits_to_probs,
    select_top_k_moves,
    winner_to_color,
)


@dataclass(frozen=True)
class MoveHeatmapResult:
    metric: str
    player: str
    player_enum: str
    selection_mode: str
    requested_top_k: Optional[int]
    effective_top_k: Optional[int]
    legal_move_count: int
    selected_move_count: int
    policy_temperature: float
    scores: Dict[str, float]
    policy_probs: Dict[str, float]
    min_score: Optional[float]
    max_score: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "metric": self.metric,
            "player": self.player,
            "player_enum": self.player_enum,
            "selection_mode": self.selection_mode,
            "requested_top_k": self.requested_top_k,
            "effective_top_k": self.effective_top_k,
            "legal_move_count": self.legal_move_count,
            "selected_move_count": self.selected_move_count,
            "policy_temperature": self.policy_temperature,
            "scores": self.scores,
            "policy_probs": self.policy_probs,
            "min_score": self.min_score,
            "max_score": self.max_score,
        }


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _evaluate_value_signed_batch(model, positions: List[str]) -> List[float]:
    if not positions:
        return []

    if hasattr(model, "batch_infer"):
        try:
            _, values = model.batch_infer(positions)
            return [float(v) for v in values]
        except Exception:
            # Fall back to serial inference if batch path is unavailable at runtime.
            pass

    outputs: List[float] = []
    for pos in positions:
        _, value_signed = model.simple_infer(pos)
        outputs.append(float(value_signed))
    return outputs


def _policy_ranking_for_legal_moves(
    state: HexGameState,
    model,
    policy_temperature: float,
) -> Tuple[List[Tuple[int, int]], Dict[str, float]]:
    legal_moves = state.get_legal_moves()
    if not legal_moves:
        return [], {}

    board_size = state.board.shape[0]
    policy_logits, _ = model.simple_infer(state.to_trmph())
    policy_probs = policy_logits_to_probs(policy_logits, policy_temperature)
    legal_policy = get_legal_policy_probs(policy_probs, legal_moves, board_size)

    move_probs = {
        fc.rowcol_to_trmph(row, col): float(prob)
        for (row, col), prob in zip(legal_moves, legal_policy)
    }
    ranked_moves = sorted(
        legal_moves,
        key=lambda mv: move_probs[fc.rowcol_to_trmph(*mv)],
        reverse=True,
    )
    return ranked_moves, move_probs


def build_policy_value_heatmap(
    state: HexGameState,
    model,
    selection_mode: str = "all_legal",
    top_k: Optional[int] = 12,
    policy_temperature: float = 1.0,
) -> MoveHeatmapResult:
    """
    Evaluate selected legal next moves using value-head win rates.

    Candidate moves are selected from policy ranking:
    - ``policy_top_k``: evaluate only top-k legal moves by policy probability.
    - ``all_legal``: evaluate all legal moves.

    For each selected move, we apply the move and run value inference on the resulting
    state. Since the resulting state's point-of-view is the opponent to move,
    we invert that probability so scores are always from the *current* player's
    perspective in the input state.
    """
    if selection_mode not in {"policy_top_k", "all_legal"}:
        raise ValueError(
            f"selection_mode must be 'policy_top_k' or 'all_legal', got '{selection_mode}'"
        )

    current_player = state.current_player_enum
    ranked_legal_moves, legal_policy_probs = _policy_ranking_for_legal_moves(
        state, model, policy_temperature
    )

    if selection_mode == "all_legal":
        selected_moves = ranked_legal_moves
        requested_top_k = None
    else:
        requested_top_k = max(1, int(top_k or 1))
        if ranked_legal_moves:
            selected_moves = select_top_k_moves(
                [legal_policy_probs[fc.rowcol_to_trmph(*m)] for m in ranked_legal_moves],
                ranked_legal_moves,
                requested_top_k,
            )
        else:
            selected_moves = []

    scores: Dict[str, float] = {}
    selected_policy_probs: Dict[str, float] = {}
    pending_moves: List[str] = []
    pending_next_players = []

    for row, col in selected_moves:
        move_trmph = fc.rowcol_to_trmph(row, col)
        selected_policy_probs[move_trmph] = float(legal_policy_probs.get(move_trmph, 0.0))
        next_state = apply_move_to_state_trmph(state, move_trmph)

        if next_state.game_over and next_state.winner is not None:
            win_prob = 1.0 if next_state.winner == current_player else 0.0
            scores[move_trmph] = _clamp_probability(win_prob)
            continue

        pending_moves.append(next_state.to_trmph())
        pending_next_players.append((move_trmph, next_state.current_player_enum))

    if pending_moves:
        pending_values = _evaluate_value_signed_batch(model, pending_moves)
        for (move_trmph, next_player), value_signed in zip(
            pending_next_players, pending_values
        ):
            opp_win_prob = ValuePredictor.get_win_probability(
                value_signed, next_player
            )
            win_prob = 1.0 - float(opp_win_prob)
            scores[move_trmph] = _clamp_probability(win_prob)

    if scores:
        min_score = min(scores.values())
        max_score = max(scores.values())
    else:
        min_score = None
        max_score = None

    return MoveHeatmapResult(
        metric="policy_value_next_move_win_prob",
        player=winner_to_color(current_player),
        player_enum=current_player.name,
        selection_mode=selection_mode,
        requested_top_k=requested_top_k,
        effective_top_k=len(selected_moves) if selection_mode == "policy_top_k" else None,
        legal_move_count=len(ranked_legal_moves),
        selected_move_count=len(selected_moves),
        policy_temperature=float(policy_temperature),
        scores=scores,
        policy_probs=selected_policy_probs,
        min_score=min_score,
        max_score=max_score,
    )


def build_next_move_value_heatmap(state: HexGameState, model) -> MoveHeatmapResult:
    """
    Backward-compatible wrapper: evaluate all legal next moves.
    """
    return build_policy_value_heatmap(
        state=state,
        model=model,
        selection_mode="all_legal",
        top_k=None,
        policy_temperature=1.0,
    )
