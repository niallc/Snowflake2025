"""Utilities for computing per-move study heatmaps for Hex positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import hex_ai.utils.format_conversion as fc
from hex_ai.inference.game_engine import HexGameState, apply_move_to_state_trmph
from hex_ai.value_utils import ValuePredictor, winner_to_color


@dataclass(frozen=True)
class MoveHeatmapResult:
    metric: str
    player: str
    player_enum: str
    scores: Dict[str, float]
    min_score: Optional[float]
    max_score: Optional[float]

    def to_dict(self) -> Dict[str, object]:
        return {
            "metric": self.metric,
            "player": self.player,
            "player_enum": self.player_enum,
            "scores": self.scores,
            "min_score": self.min_score,
            "max_score": self.max_score,
            "legal_move_count": len(self.scores),
        }


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def build_next_move_value_heatmap(state: HexGameState, model) -> MoveHeatmapResult:
    """
    Evaluate each legal next move for the current player using value-head win rates.

    For each legal move, we apply the move and run value inference on the resulting
    state. Since the resulting state's point-of-view is the opponent to move,
    we invert that probability so scores are always from the *current* player's
    perspective in the input state.
    """
    current_player = state.current_player_enum
    scores: Dict[str, float] = {}

    for row, col in state.get_legal_moves():
        move_trmph = fc.rowcol_to_trmph(row, col)
        next_state = apply_move_to_state_trmph(state, move_trmph)

        if next_state.game_over and next_state.winner is not None:
            win_prob = 1.0 if next_state.winner == current_player else 0.0
        else:
            next_trmph = next_state.to_trmph()
            _, value_signed = model.simple_infer(next_trmph)
            opp_win_prob = ValuePredictor.get_win_probability(
                value_signed, next_state.current_player_enum
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
        metric="value_head_next_move_win_prob",
        player=winner_to_color(current_player),
        player_enum=current_player.name,
        scores=scores,
        min_score=min_score,
        max_score=max_score,
    )
