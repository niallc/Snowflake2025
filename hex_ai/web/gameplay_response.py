"""Shared response and move-sequence helpers for web gameplay endpoints."""

from __future__ import annotations

from typing import Any, Mapping

import hex_ai.utils.format_conversion as fc
from hex_ai.inference.game_engine import apply_move_to_state_trmph
from hex_ai.value_utils import ValuePredictor, policy_logits_to_probs, winner_to_color


def moves_to_trmph(moves):
    """Convert iterable of (row, col) tuples into TRMPH move strings."""
    return [fc.rowcol_to_trmph(row, col) for row, col in moves]


def build_game_state_response(
    state,
    *,
    model,
    temperature: float,
    trmph_for_inference: str | None = None,
    additional_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build the common game response payload including policy/value inference.
    """
    player_enum = state.current_player_enum
    winner = state.winner

    if trmph_for_inference is None:
        trmph_for_inference = state.to_trmph()

    policy_logits, value_signed = model.simple_infer(trmph_for_inference)
    policy_probs = policy_logits_to_probs(policy_logits, temperature)

    response = {
        "board": state.board.tolist(),
        "player": winner_to_color(player_enum),
        "player_enum": player_enum.name,
        "player_index": int(player_enum.value),
        "legal_moves": moves_to_trmph(state.get_legal_moves()),
        "winner": winner_to_color(winner) if winner is not None else None,
        "policy": {
            fc.tensor_to_trmph(index): float(probability)
            for index, probability in enumerate(policy_probs)
        },
        "value_signed": float(value_signed),
        "win_probability": ValuePredictor.get_win_probability(value_signed, player_enum),
    }

    if additional_fields:
        response.update(dict(additional_fields))
    return response


def build_engine_move_response(
    state,
    *,
    new_trmph: str,
    move_made: str | None = None,
    success: bool = True,
    error: str | None = None,
    additional_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the shared move-result payload for engine-driven endpoints."""
    response = {
        "success": success,
        "new_trmph": new_trmph,
        "board": state.board.tolist(),
        "player": winner_to_color(state.current_player_enum),
        "legal_moves": moves_to_trmph(state.get_legal_moves()),
        "winner": winner_to_color(state.winner) if state.winner is not None else None,
        "move_made": move_made,
        "game_over": state.game_over,
    }

    if error:
        response["error"] = error
    if additional_fields:
        response.update(dict(additional_fields))
    return response


def apply_trmph_sequence_to_state(state, trmph_sequence: str):
    """
    Parse and apply a TRMPH sequence, stopping early if the game is over.

    Returns:
        tuple: (updated_state, parsed_moves, moves_applied)
    """
    sequence = (trmph_sequence or "").strip()
    if not sequence:
        return state, [], 0

    moves = fc.split_trmph_moves(sequence)
    moves_applied = 0

    for move in moves:
        if state.game_over:
            break
        state = apply_move_to_state_trmph(state, move)
        moves_applied += 1

    return state, moves, moves_applied
