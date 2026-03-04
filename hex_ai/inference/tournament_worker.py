"""
Worker entrypoints for memory-bounded tournament execution.

Each worker process executes exactly one heavy tournament unit and writes a
compact JSON summary for the parent coordinator to consume.
"""

import argparse
import json
import os
from typing import Any, Dict, List

from hex_ai.config import BOARD_SIZE
from hex_ai.inference.game_execution import (
    OpeningPosition,
    play_deterministic_game,
    run_round_robin_tournament,
)
from hex_ai.inference.knockout_tournament import TournamentParticipant
from hex_ai.inference.model_cache import create_temporary_model_cache
from hex_ai.inference.strategy_config import StrategyConfig
from hex_ai.inference.tournament import TournamentPlayConfig
from hex_ai.move_provenance import make_move_provenance_record, sidecar_path_for_trmph
from hex_ai.utils.random_utils import set_deterministic_seeds
from hex_ai.utils.deterministic_tournament_utils import setup_strategy_pair_files
from hex_ai.utils.tournament_logging import (
    append_trmph_winner_line,
    write_tournament_trmph_header,
)


def _deserialize_openings(openings_payload: List[Dict[str, Any]]) -> List[OpeningPosition]:
    """Deserialize openings from JSON payload."""
    openings = []
    for opening_data in openings_payload:
        moves = [(int(move[0]), int(move[1])) for move in opening_data["moves"]]
        opening = OpeningPosition(
            moves=moves,
            source_game=opening_data.get("source_game", ""),
            opening_length=int(opening_data.get("opening_length", len(moves))),
        )
        openings.append(opening)
    return openings


def _deserialize_strategy_config(strategy_payload: Dict[str, Any]) -> StrategyConfig:
    """Deserialize StrategyConfig from JSON payload."""
    return StrategyConfig(
        name=strategy_payload["name"],
        strategy_type=strategy_payload["strategy_type"],
        config=dict(strategy_payload["config"]),
        model_path=strategy_payload["model_path"],
        original_name=strategy_payload.get("original_name"),
        temperature=strategy_payload.get("temperature"),
    )


def run_round_robin_pair_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one round-robin strategy pair in a fresh worker process."""
    worker_seed = payload.get("seed")
    if worker_seed is not None:
        set_deterministic_seeds(int(worker_seed))

    strategy_a = _deserialize_strategy_config(payload["strategy_a"])
    strategy_b = _deserialize_strategy_config(payload["strategy_b"])
    openings = _deserialize_openings(payload["openings"])

    tournament_result = run_round_robin_tournament(
        strategy_configs=[strategy_a, strategy_b],
        openings=openings,
        temperature=float(payload["temperature"]),
        verbose=1,
        seed=worker_seed,
        output_dir=payload["output_dir"],
        command_line=payload.get("command_line"),
        run_desc=payload.get("run_desc"),
        mps_empty_cache_per_pair=bool(payload.get("mps_empty_cache_per_pair", False)),
    )

    return {
        "strategy_a": strategy_a.name,
        "strategy_b": strategy_b.name,
        "a_vs_b": tournament_result.results[strategy_a.name][strategy_b.name],
        "b_vs_a": tournament_result.results[strategy_b.name][strategy_a.name],
        "total_games": tournament_result.total_games,
        "strategy_timings": {
            strategy_a.name: tournament_result.strategy_timings.get(strategy_a.name, 0.0),
            strategy_b.name: tournament_result.strategy_timings.get(strategy_b.name, 0.0),
        },
        "strategy_move_counts": {
            strategy_a.name: tournament_result.strategy_move_counts.get(strategy_a.name, 0),
            strategy_b.name: tournament_result.strategy_move_counts.get(strategy_b.name, 0),
        },
    }


def run_knockout_match_worker(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one knockout match in a fresh worker process."""
    worker_seed = payload.get("seed")
    if worker_seed is not None:
        set_deterministic_seeds(int(worker_seed))

    participant1 = TournamentParticipant(
        name=payload["participant1"]["name"],
        strategy_config=payload["participant1"]["strategy_config"],
        metadata=payload["participant1"].get("metadata"),
    )
    participant2 = TournamentParticipant(
        name=payload["participant2"]["name"],
        strategy_config=payload["participant2"]["strategy_config"],
        metadata=payload["participant2"].get("metadata"),
    )

    games = int(payload["games"])
    openings = _deserialize_openings(payload["openings"])
    knockout_config = dict(payload.get("knockout_config", {}))

    strategy_a = participant1.to_strategy_config()
    strategy_b = participant2.to_strategy_config()

    model_cache = create_temporary_model_cache([strategy_a.model_path, strategy_b.model_path], verbose=0)

    trmph_file, _ = setup_strategy_pair_files(payload["output_dir"], strategy_a, strategy_b)
    play_config = TournamentPlayConfig(
        temperature=knockout_config.get("temperature", 1.0),
        random_seed=42,
        command_line=payload.get("command_line"),
        run_desc=payload.get("run_desc"),
    )
    actual_trmph_file = write_tournament_trmph_header(
        trmph_file,
        [strategy_a.model_path, strategy_b.model_path],
        games * 2,
        play_config,
        BOARD_SIZE,
        strategy_configs=[strategy_a, strategy_b],
    )
    provenance_file = str(sidecar_path_for_trmph(actual_trmph_file))
    if os.path.exists(provenance_file):
        raise RuntimeError(
            "Expected fresh provenance sidecar path for knockout worker but file already exists: "
            f"{provenance_file}"
        )

    participant1_wins = 0
    participant2_wins = 0
    openings_used = []
    provenance_records_written = 0

    with open(provenance_file, "w", encoding="utf-8") as provenance_handle:
        for opening_idx, opening in enumerate(openings):
            result_1 = play_deterministic_game(
                model_cache=model_cache,
                strategy_a=strategy_a,
                strategy_b=strategy_b,
                opening=opening,
                temperature=knockout_config.get("temperature", 1.0),
                verbose=0,
                strategy_a_is_blue=True,
            )
            append_trmph_winner_line(result_1["trmph_str"], result_1["winner_char"], actual_trmph_file)
            provenance_handle.write(
                make_move_provenance_record(
                    game_index=provenance_records_written,
                    move_codes=result_1["move_provenance_codes"],
                    policy_targets=result_1["policy_targets_matrix"],
                    policy_target_source_codes=result_1.get("policy_target_source_codes"),
                    policy_target_version=result_1.get("policy_target_version"),
                ).to_json_line()
            )
            provenance_handle.write("\n")
            provenance_records_written += 1

            result_2 = play_deterministic_game(
                model_cache=model_cache,
                strategy_a=strategy_b,
                strategy_b=strategy_a,
                opening=opening,
                temperature=knockout_config.get("temperature", 1.0),
                verbose=0,
                strategy_a_is_blue=True,
            )
            append_trmph_winner_line(result_2["trmph_str"], result_2["winner_char"], actual_trmph_file)
            provenance_handle.write(
                make_move_provenance_record(
                    game_index=provenance_records_written,
                    move_codes=result_2["move_provenance_codes"],
                    policy_targets=result_2["policy_targets_matrix"],
                    policy_target_source_codes=result_2.get("policy_target_source_codes"),
                    policy_target_version=result_2.get("policy_target_version"),
                ).to_json_line()
            )
            provenance_handle.write("\n")
            provenance_records_written += 1

            openings_used.append(opening.get_trmph_string())

            if result_1["winner_strategy"] == participant1.name:
                participant1_wins += 1
            else:
                participant2_wins += 1

            if result_2["winner_strategy"] == participant1.name:
                participant1_wins += 1
            else:
                participant2_wins += 1

            if opening_idx % 10 == 0:
                print(".", end="", flush=True)

    total_games = games * 2
    if participant1_wins > participant2_wins:
        winner_name = participant1.name
    elif participant2_wins > participant1_wins:
        winner_name = participant2.name
    else:
        winner_name = participant1.name

    participant1_pct = (participant1_wins / total_games) * 100 if total_games > 0 else 0.0
    participant2_pct = (participant2_wins / total_games) * 100 if total_games > 0 else 0.0
    print(
        f" {participant1.name}:{participant1_wins}/{total_games} ({participant1_pct:.1f}%) "
        f"{participant2.name}:{participant2_wins}/{total_games} ({participant2_pct:.1f}%) -> {winner_name} wins"
    )

    return {
        "participant1_wins": participant1_wins,
        "participant2_wins": participant2_wins,
        "total_games": total_games,
        "openings_used": openings_used,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Execute one tournament worker unit.")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["round_robin_pair", "knockout_match"],
        help="Worker mode to execute.",
    )
    parser.add_argument("--input-json", type=str, required=True, help="Path to worker input JSON payload.")
    parser.add_argument("--output-json", type=str, required=True, help="Path to worker output JSON payload.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.input_json, "r") as input_file:
        payload = json.load(input_file)

    if args.mode == "round_robin_pair":
        result = run_round_robin_pair_worker(payload)
    elif args.mode == "knockout_match":
        result = run_knockout_match_worker(payload)
    else:
        raise ValueError(f"Unsupported worker mode: {args.mode}")

    with open(args.output_json, "w") as output_file:
        json.dump(result, output_file, indent=2)


if __name__ == "__main__":
    main()
