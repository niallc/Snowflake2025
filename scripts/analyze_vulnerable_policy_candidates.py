#!/usr/bin/env python3
"""Analyze how often top policy candidates are weak/dead/vulnerable in played games.

This script intentionally stays off the hot path. It replays saved tournament
states and recomputes policy/value outputs offline so deeper investigations do
not add instrumentation complexity to the core MCTS execution code.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from hex_ai.enums import Piece, Player
from hex_ai.inference.game_engine import HexGameState, apply_move_to_state, make_empty_hex_state
from hex_ai.inference.model_config import get_model_path
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.utils.format_conversion import (
    rowcol_to_tensor_with_size,
    rowcol_to_trmph,
    split_trmph_moves,
    strip_trmph_preamble,
    trmph_move_to_rowcol,
)
from hex_ai.utils.weaks_cells import select_policy_ordered_weak_moves
from hex_ai.value_utils import ValuePredictor, player_to_winner


RANK_BIN_LABELS: tuple[str, ...] = ("1", "2-3", "4-5", "6-10", "11+")


@dataclass(frozen=True)
class PositionRef:
    game_row_index: int
    opening_idx: int
    opening_source: str
    game_label: str
    move_index_to_play: int
    state_trmph: str
    eventual_winner: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Replay tournament games and report how often top policy-ranked legal moves "
            "would be classified or filtered as dead/vulnerable."
        )
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        required=True,
        help="Tournament CSV file produced by run_tournament.py.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="best",
        help='Model name or checkpoint path to evaluate (default: "best").',
    )
    parser.add_argument(
        "--focus-strategy",
        type=str,
        default="deadmask",
        help=(
            "Exact name or unique substring for the strategy whose turns should be "
            'analyzed (default: "deadmask"). Use "ALL" to analyze every turn.'
        ),
    )
    parser.add_argument(
        "--top-k",
        type=str,
        default="1,3,5,10",
        help='Comma-separated policy cutoffs to summarize (default: "1,3,5,10").',
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=0,
        help="Optional cap on CSV game rows to analyze (0 means all).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=8,
        help="How many sample positions to keep per sample category (default: 8).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write the full summary JSON.",
    )
    return parser.parse_args()


def _parse_top_k(raw_text: str) -> list[int]:
    values: list[int] = []
    for token in raw_text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        value = int(stripped)
        if value <= 0:
            raise ValueError(f"top-k entries must be positive, got {value}")
        values.append(value)
    if not values:
        raise ValueError("At least one top-k value is required")
    return sorted(set(values))


def _resolve_model_path(model_arg: str) -> str:
    candidate = Path(model_arg).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    return str(Path(get_model_path(model_arg)).resolve())


def _load_csv_rows(csv_file: Path, max_games: int) -> list[dict[str, str]]:
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    rows: list[dict[str, str]] = []
    with csv_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=1):
            rows.append(dict(row))
            if max_games > 0 and row_idx >= max_games:
                break
    if not rows:
        raise ValueError(f"No game rows found in {csv_file}")
    return rows


def _resolve_focus_strategy(rows: Sequence[dict[str, str]], focus_query: str) -> Optional[str]:
    normalized = focus_query.strip()
    if not normalized or normalized.upper() == "ALL":
        return None

    names = sorted(
        {
            row["strategy_a"]
            for row in rows
            if row.get("strategy_a")
        }
        | {
            row["strategy_b"]
            for row in rows
            if row.get("strategy_b")
        }
    )

    exact_matches = [name for name in names if name == normalized]
    if len(exact_matches) == 1:
        return exact_matches[0]

    substring_matches = [name for name in names if normalized in name]
    if len(substring_matches) == 1:
        return substring_matches[0]
    if not substring_matches:
        raise ValueError(
            f'Could not match focus strategy "{focus_query}". Available strategies: {names}'
        )
    raise ValueError(
        f'Focus strategy "{focus_query}" matched multiple strategies: {substring_matches}'
    )


def _position_matches_focus(
    current_player: Player,
    *,
    focus_is_blue: bool,
) -> bool:
    if current_player == Player.BLUE:
        return focus_is_blue
    if current_player == Player.RED:
        return not focus_is_blue
    raise ValueError(f"Unexpected current player enum: {current_player}")


def _collect_focus_positions(
    rows: Sequence[dict[str, str]],
    *,
    focus_strategy: Optional[str],
) -> tuple[list[PositionRef], int]:
    positions: list[PositionRef] = []
    games_with_focus = 0

    for row_index, row in enumerate(rows, start=1):
        strategy_a = row["strategy_a"]
        strategy_b = row["strategy_b"]
        if focus_strategy is None:
            focus_is_blue = True
            include_every_turn = True
        else:
            include_every_turn = False
            if strategy_a == focus_strategy:
                focus_is_blue = True
            elif strategy_b == focus_strategy:
                focus_is_blue = False
            else:
                continue

        games_with_focus += 1
        trmph = row["trmph"].strip()
        board_size = int(trmph[1 : trmph.index(",")])
        opening_length = int(row.get("opening_length", "0") or 0)
        moves = split_trmph_moves(strip_trmph_preamble(trmph))
        state = make_empty_hex_state(board_size=board_size)

        for move_index_to_play, move_text in enumerate(moves, start=1):
            if move_index_to_play > opening_length:
                if include_every_turn or _position_matches_focus(
                    state.current_player_enum,
                    focus_is_blue=focus_is_blue,
                ):
                    positions.append(
                        PositionRef(
                            game_row_index=row_index,
                            opening_idx=int(row["opening_idx"]),
                            opening_source=row["opening_source"],
                            game_label=row["game"],
                            move_index_to_play=move_index_to_play,
                            state_trmph=state.to_trmph(),
                            eventual_winner=row["winner"].strip(),
                        )
                    )

            move_row, move_col = trmph_move_to_rowcol(move_text, board_size=board_size)
            state = apply_move_to_state(state, move_row, move_col)

    return positions, games_with_focus


def _has_immediate_terminal_move(
    state: HexGameState,
    legal_moves: Sequence[tuple[int, int]],
) -> bool:
    board_size = int(state.board.shape[0])
    if len(state.move_history) < board_size * 2 - 2:
        return False

    winner = player_to_winner(state.current_player_enum)
    for row, col in legal_moves:
        next_state = state.make_move(row, col)
        if next_state.game_over and next_state.winner == winner:
            return True
    return False


def _rank_bin_label(rank: Optional[int]) -> str:
    if rank is None:
        return "none"
    if rank == 1:
        return "1"
    if rank <= 3:
        return "2-3"
    if rank <= 5:
        return "4-5"
    if rank <= 10:
        return "6-10"
    return "11+"


def _move_index_bucket(move_index: int) -> str:
    start = ((move_index - 1) // 10) * 10 + 1
    end = start + 9
    return f"{start}-{end}"


def _win_probability_bucket(win_probability: float) -> str:
    if win_probability < 0.10:
        return "<0.10"
    if win_probability < 0.25:
        return "0.10-0.25"
    if win_probability < 0.40:
        return "0.25-0.40"
    if win_probability <= 0.60:
        return "0.40-0.60"
    if win_probability <= 0.75:
        return "0.60-0.75"
    if win_probability <= 0.90:
        return "0.75-0.90"
    return ">0.90"


def _safe_rate(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(count / total)


def _summarize_numeric(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "bucket_counts": {},
        }
    return {
        "count": int(len(values)),
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "min": int(min(values)),
        "max": int(max(values)),
        "bucket_counts": dict(
            sorted(Counter(_move_index_bucket(value) for value in values).items())
        ),
    }


def _summarize_probability(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "bucket_counts": {},
        }
    return {
        "count": int(len(values)),
        "mean": float(sum(values) / len(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "bucket_counts": dict(
            sorted(Counter(_win_probability_bucket(value) for value in values).items())
        ),
    }


def _sample_payload(
    position: PositionRef,
    state_summary: dict[str, Any],
) -> dict[str, Any]:
    current_player = state_summary["current_player"]
    return {
        "game_row_index": position.game_row_index,
        "opening_idx": position.opening_idx,
        "opening_source": position.opening_source,
        "game_label": position.game_label,
        "move_index_to_play": position.move_index_to_play,
        "state_trmph": position.state_trmph,
        "current_player": current_player,
        "eventual_winner": position.eventual_winner,
        "eventual_current_player_won": bool(
            position.eventual_winner == ("b" if current_player == "blue" else "r")
        ),
        "root_value_signed": state_summary["root_value_signed"],
        "current_player_win_probability": state_summary["current_player_win_probability"],
        "top_policy_move": state_summary["top_move"]["trmph"],
        "top_policy_move_status": state_summary["top_move"]["status"],
        "top_policy_move_filtered": bool(state_summary["top_move"]["filtered"]),
        "top_policy_move_rules": list(state_summary["top_move"]["rules"]),
        "top_policy_move_vulnerable_reply_moves": list(
            state_summary["top_move"]["vulnerable_reply_moves"]
        ),
        "first_classified_weak_rank": state_summary["classified_min_rank"],
        "first_filtered_rank": state_summary["filtered_min_rank"],
    }


def _build_state_summary(
    state_trmph: str,
    policy_logits: Sequence[float],
    value_signed: float,
    *,
    top_ks: Sequence[int],
) -> dict[str, Any]:
    state = HexGameState.from_trmph(state_trmph)
    board_size = int(state.board.shape[0])
    legal_moves = state.get_legal_moves()
    current_player = state.current_player_enum
    legal_indices = [
        rowcol_to_tensor_with_size(row, col, board_size)
        for row, col in legal_moves
    ]
    legal_scores = [float(policy_logits[index]) for index in legal_indices]
    current_player_win_probability = ValuePredictor.get_win_probability(
        value_signed,
        current_player,
    )

    if _has_immediate_terminal_move(state, legal_moves):
        return {
            "terminal_bypass": True,
            "current_player": str(current_player.name).lower(),
            "legal_move_count": int(len(legal_moves)),
            "root_value_signed": float(value_signed),
            "current_player_win_probability": float(current_player_win_probability),
        }

    order = sorted(
        range(len(legal_moves)),
        key=lambda idx: (-legal_scores[idx], legal_moves[idx][0], legal_moves[idx][1]),
    )
    rank_by_index = {
        idx: rank
        for rank, idx in enumerate(order, start=1)
    }
    player_piece = (
        Piece.BLUE.value if current_player == Player.BLUE else Piece.RED.value
    )
    filter_result = select_policy_ordered_weak_moves(
        state.board,
        legal_moves,
        legal_scores,
        player_color=player_piece,
    )

    weak_filtered_set = set(filter_result.weak_filtered_indices)
    classified_weak_indices = [
        idx
        for idx, classification in filter_result.classifications_by_index.items()
        if classification.status != "safe"
    ]
    top_move_index = order[0]
    top_move = legal_moves[top_move_index]
    top_move_classification = filter_result.classifications_by_index[top_move_index]
    top_move_filtered = top_move_index in weak_filtered_set

    per_k: dict[int, dict[str, Any]] = {}
    for k in top_ks:
        limited = order[: min(k, len(order))]
        statuses = [
            filter_result.classifications_by_index[idx].status
            for idx in limited
        ]
        per_k[k] = {
            "any_classified_weak": any(status != "safe" for status in statuses),
            "any_filtered": any(idx in weak_filtered_set for idx in limited),
            "classified_weak_count": int(sum(status != "safe" for status in statuses)),
            "filtered_count": int(sum(idx in weak_filtered_set for idx in limited)),
            "dead_count": int(sum(status == "dead" for status in statuses)),
            "vulnerable_count": int(sum(status == "vulnerable" for status in statuses)),
        }

    classified_ranks = sorted(rank_by_index[idx] for idx in classified_weak_indices)
    filtered_ranks = sorted(rank_by_index[idx] for idx in weak_filtered_set)

    return {
        "terminal_bypass": False,
        "current_player": str(current_player.name).lower(),
        "root_value_signed": float(value_signed),
        "current_player_win_probability": float(current_player_win_probability),
        "legal_move_count": int(len(legal_moves)),
        "safe_moves_total": int(len(filter_result.safe_indices)),
        "vulnerable_moves_total": int(len(filter_result.vulnerable_indices)),
        "dead_moves_total": int(len(filter_result.dead_indices)),
        "filtered_moves_total": int(len(filter_result.weak_filtered_indices)),
        "only_vulnerable_remaining": bool(filter_result.only_vulnerable_remaining),
        "classified_min_rank": classified_ranks[0] if classified_ranks else None,
        "filtered_min_rank": filtered_ranks[0] if filtered_ranks else None,
        "top_move": {
            "trmph": rowcol_to_trmph(top_move[0], top_move[1], board_size),
            "status": top_move_classification.status,
            "filtered": bool(top_move_filtered),
            "rules": list(top_move_classification.reasons),
            "vulnerable_reply_moves": [
                rowcol_to_trmph(reply_row, reply_col, board_size)
                for reply_row, reply_col in top_move_classification.vulnerable_reply_moves
            ],
        },
        "per_k": per_k,
    }


def _render_pct(count: int, total: int) -> str:
    return f"{count}/{total} ({_safe_rate(count, total) * 100:.2f}%)"


def _winner_matches_current_player(
    eventual_winner: str,
    current_player: str,
) -> bool:
    if eventual_winner not in {"b", "r"}:
        raise ValueError(f"Unexpected winner token: {eventual_winner!r}")
    if current_player == "blue":
        return eventual_winner == "b"
    if current_player == "red":
        return eventual_winner == "r"
    raise ValueError(f"Unexpected current player label: {current_player!r}")


def _print_summary(summary: dict[str, Any]) -> None:
    counts = summary["counts"]
    applicable = counts["weak_filter_applicable_positions"]
    focus_total = counts["focus_positions_total"]
    top_k = summary["top_k"]
    largest_k = max(top_k)
    per_k = summary["per_k"]

    print(f"CSV: {summary['csv_file']}")
    print(f"Model: {summary['model_path']}")
    print(
        f"Focus strategy: {summary['focus_strategy'] or 'ALL'} "
        f"(games with focus positions: {counts['games_with_focus_positions']})"
    )
    print(
        "Focus positions after opening phase: "
        f"{focus_total} (terminal-bypass: {counts['terminal_bypass_positions']}, "
        f"weak-filter-applicable: {applicable})"
    )
    print(
        "Unique focus states: "
        f"{counts['unique_focus_states']} (device: {summary['device_info']['wrapper_device']})"
    )
    print()
    print("Top policy move on weak-filter-applicable positions:")
    print(
        "  classified dead: "
        f"{_render_pct(counts['top_move_dead'], applicable)}"
    )
    print(
        "  classified vulnerable: "
        f"{_render_pct(counts['top_move_vulnerable'], applicable)}"
    )
    print(
        "  classified weak total: "
        f"{_render_pct(counts['top_move_classified_weak'], applicable)}"
    )
    print(
        "  actually filtered out: "
        f"{_render_pct(counts['top_move_filtered'], applicable)}"
    )
    print(
        "  vulnerable but retained (only vulnerable remained): "
        f"{_render_pct(counts['top_move_vulnerable_kept'], applicable)}"
    )
    print()
    print("Any weak / filtered move within top-k:")
    for k in top_k:
        print(
            f"  k={k}: classified weak {per_k[str(k)]['positions_with_any_classified_weak']}/"
            f"{applicable} ({per_k[str(k)]['rate_any_classified_weak'] * 100:.2f}%), "
            f"filtered {per_k[str(k)]['positions_with_any_filtered']}/{applicable} "
            f"({per_k[str(k)]['rate_any_filtered'] * 100:.2f}%)"
        )
    print()
    print("Earliest weak/filter rank on applicable positions:")
    print(
        "  classified weak: "
        f"{summary['rank_bins']['classified_min_rank']}"
    )
    print(
        "  filtered out: "
        f"{summary['rank_bins']['filtered_min_rank']}"
    )
    print()
    print(f"Move-index timing for top-{largest_k} filtered occurrences:")
    print(
        "  all occurrences: "
        f"{summary['per_k'][str(largest_k)]['filtered_move_index_summary']}"
    )
    print(
        "  first occurrence per game: "
        f"{summary['per_k'][str(largest_k)]['first_filtered_move_index_summary']}"
    )
    print()
    print("Root value / eventual result for top-1 filtered positions:")
    print(
        "  current-player win probability: "
        f"{summary['top1_filtered_context']['current_player_win_probability_summary']}"
    )
    print(
        "  eventual current-player wins: "
        + _render_pct(
            summary["top1_filtered_context"]["eventual_current_player_wins"],
            summary["top1_filtered_context"]["positions"],
        )
    )
    print(
        "  value bands: "
        f"{summary['top1_filtered_context']['current_player_win_probability_summary']['bucket_counts']}"
    )


def main() -> None:
    args = parse_args()
    top_ks = _parse_top_k(args.top_k)
    rows = _load_csv_rows(args.csv_file, args.max_games)
    focus_strategy = _resolve_focus_strategy(rows, args.focus_strategy)
    positions, games_with_focus = _collect_focus_positions(
        rows,
        focus_strategy=focus_strategy,
    )
    if not positions:
        raise ValueError("No positions matched the requested filters.")

    unique_states = sorted({position.state_trmph for position in positions})
    model_path = _resolve_model_path(args.model)
    infer = SimpleModelInference(model_path, verbose=0)
    policy_logits_list, value_signed_list = infer.batch_infer(unique_states)

    state_summary_by_trmph: dict[str, dict[str, Any]] = {}
    for state_trmph, policy_logits, value_signed in zip(
        unique_states,
        policy_logits_list,
        value_signed_list,
    ):
        state_summary_by_trmph[state_trmph] = _build_state_summary(
            state_trmph,
            policy_logits,
            value_signed,
            top_ks=top_ks,
        )

    counts = {
        "csv_rows_loaded": int(len(rows)),
        "games_with_focus_positions": int(games_with_focus),
        "focus_positions_total": int(len(positions)),
        "unique_focus_states": int(len(unique_states)),
        "terminal_bypass_positions": 0,
        "weak_filter_applicable_positions": 0,
        "positions_with_only_vulnerable_remaining": 0,
        "top_move_dead": 0,
        "top_move_vulnerable": 0,
        "top_move_classified_weak": 0,
        "top_move_filtered": 0,
        "top_move_vulnerable_kept": 0,
    }
    rank_bins = {
        "classified_min_rank": Counter(),
        "filtered_min_rank": Counter(),
    }
    per_k_counters: dict[int, dict[str, Any]] = {
        k: {
            "positions_with_any_classified_weak": 0,
            "positions_with_any_filtered": 0,
            "all_filtered_move_indices": [],
            "first_filtered_move_index_by_game": {},
        }
        for k in top_ks
    }
    sample_top1_filtered: list[dict[str, Any]] = []
    sample_largest_k_filtered: list[dict[str, Any]] = []
    largest_k = max(top_ks)
    applicable_current_player_win_probabilities: list[float] = []
    top1_filtered_current_player_win_probabilities: list[float] = []
    applicable_current_player_wins = 0
    top1_filtered_eventual_current_player_wins = 0
    top1_filtered_results_by_bucket: dict[str, dict[str, int]] = {}

    for position in positions:
        state_summary = state_summary_by_trmph[position.state_trmph]
        if state_summary["terminal_bypass"]:
            counts["terminal_bypass_positions"] += 1
            continue

        counts["weak_filter_applicable_positions"] += 1
        current_player_win_probability = state_summary["current_player_win_probability"]
        applicable_current_player_win_probabilities.append(current_player_win_probability)
        current_player_eventually_won = _winner_matches_current_player(
            position.eventual_winner,
            state_summary["current_player"],
        )
        if current_player_eventually_won:
            applicable_current_player_wins += 1
        if state_summary["only_vulnerable_remaining"]:
            counts["positions_with_only_vulnerable_remaining"] += 1

        top_move = state_summary["top_move"]
        if top_move["status"] == "dead":
            counts["top_move_dead"] += 1
        if top_move["status"] == "vulnerable":
            counts["top_move_vulnerable"] += 1
        if top_move["status"] != "safe":
            counts["top_move_classified_weak"] += 1
        if top_move["filtered"]:
            counts["top_move_filtered"] += 1
            top1_filtered_current_player_win_probabilities.append(
                current_player_win_probability
            )
            if current_player_eventually_won:
                top1_filtered_eventual_current_player_wins += 1
            bucket = _win_probability_bucket(current_player_win_probability)
            bucket_counts = top1_filtered_results_by_bucket.setdefault(
                bucket,
                {"positions": 0, "eventual_current_player_wins": 0},
            )
            bucket_counts["positions"] += 1
            if current_player_eventually_won:
                bucket_counts["eventual_current_player_wins"] += 1
            if len(sample_top1_filtered) < args.sample_limit:
                sample_top1_filtered.append(_sample_payload(position, state_summary))
        if top_move["status"] == "vulnerable" and not top_move["filtered"]:
            counts["top_move_vulnerable_kept"] += 1

        rank_bins["classified_min_rank"][
            _rank_bin_label(state_summary["classified_min_rank"])
        ] += 1
        rank_bins["filtered_min_rank"][
            _rank_bin_label(state_summary["filtered_min_rank"])
        ] += 1

        for k in top_ks:
            stats = state_summary["per_k"][k]
            aggregate = per_k_counters[k]
            if stats["any_classified_weak"]:
                aggregate["positions_with_any_classified_weak"] += 1
            if stats["any_filtered"]:
                aggregate["positions_with_any_filtered"] += 1
                aggregate["all_filtered_move_indices"].append(position.move_index_to_play)
                aggregate["first_filtered_move_index_by_game"].setdefault(
                    position.game_row_index,
                    position.move_index_to_play,
                )
                if k == largest_k and len(sample_largest_k_filtered) < args.sample_limit:
                    sample_largest_k_filtered.append(_sample_payload(position, state_summary))

    applicable = counts["weak_filter_applicable_positions"]
    device_info = infer.model.get_device_info()
    summary = {
        "csv_file": str(args.csv_file.resolve()),
        "model_path": model_path,
        "focus_strategy": focus_strategy,
        "top_k": top_ks,
        "device_info": device_info,
        "counts": counts,
        "rate_summary": {
            "terminal_bypass_rate_of_focus_positions": _safe_rate(
                counts["terminal_bypass_positions"],
                counts["focus_positions_total"],
            ),
            "only_vulnerable_remaining_rate": _safe_rate(
                counts["positions_with_only_vulnerable_remaining"],
                applicable,
            ),
            "top_move_classified_weak_rate": _safe_rate(
                counts["top_move_classified_weak"],
                applicable,
            ),
            "top_move_filtered_rate": _safe_rate(
                counts["top_move_filtered"],
                applicable,
            ),
        },
        "applicable_position_context": {
            "positions": int(applicable),
            "current_player_win_probability_summary": _summarize_probability(
                applicable_current_player_win_probabilities
            ),
            "eventual_current_player_wins": int(applicable_current_player_wins),
            "eventual_current_player_win_rate": _safe_rate(
                applicable_current_player_wins,
                applicable,
            ),
        },
        "top1_filtered_context": {
            "positions": int(counts["top_move_filtered"]),
            "current_player_win_probability_summary": _summarize_probability(
                top1_filtered_current_player_win_probabilities
            ),
            "eventual_current_player_wins": int(top1_filtered_eventual_current_player_wins),
            "eventual_current_player_win_rate": _safe_rate(
                top1_filtered_eventual_current_player_wins,
                counts["top_move_filtered"],
            ),
            "eventual_result_by_value_band": {
                bucket: {
                    "positions": int(bucket_counts["positions"]),
                    "eventual_current_player_wins": int(
                        bucket_counts["eventual_current_player_wins"]
                    ),
                    "eventual_current_player_win_rate": _safe_rate(
                        bucket_counts["eventual_current_player_wins"],
                        bucket_counts["positions"],
                    ),
                }
                for bucket, bucket_counts in sorted(top1_filtered_results_by_bucket.items())
            },
        },
        "rank_bins": {
            key: {
                label: int(counter.get(label, 0))
                for label in ("none",) + RANK_BIN_LABELS
            }
            for key, counter in rank_bins.items()
        },
        "per_k": {},
        "samples": {
            "top1_filtered": sample_top1_filtered,
            f"top{largest_k}_filtered": sample_largest_k_filtered,
        },
    }

    for k in top_ks:
        aggregate = per_k_counters[k]
        first_values = sorted(aggregate["first_filtered_move_index_by_game"].values())
        summary["per_k"][str(k)] = {
            "positions_with_any_classified_weak": int(
                aggregate["positions_with_any_classified_weak"]
            ),
            "rate_any_classified_weak": _safe_rate(
                aggregate["positions_with_any_classified_weak"],
                applicable,
            ),
            "positions_with_any_filtered": int(aggregate["positions_with_any_filtered"]),
            "rate_any_filtered": _safe_rate(
                aggregate["positions_with_any_filtered"],
                applicable,
            ),
            "filtered_move_index_summary": _summarize_numeric(
                aggregate["all_filtered_move_indices"]
            ),
            "first_filtered_move_index_summary": _summarize_numeric(first_values),
        }

    _print_summary(summary)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print()
        print(f"Wrote JSON summary to {args.output_json}")


if __name__ == "__main__":
    main()
