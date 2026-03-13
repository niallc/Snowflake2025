#!/usr/bin/env python3
"""
Run a tournament between Snowflake 2025 models and a local KataHex engine.
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from hex_ai.config import (
    BOARD_SIZE,
    DEFAULT_BATCH_CAP,
    DEFAULT_C_PUCT,
    DEFAULT_GUMBEL_C_SCALE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET,
    DEFAULT_GUMBEL_CANDIDATE_POWER_RATE,
    DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE,
    DEFAULT_GUMBEL_SIM_THRESHOLD,
    DEFAULT_MCTS_SIMS,
    POLICY_TARGET_CONSTRUCTION_VERSION,
    TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD,
    TRMPH_PREFIX,
)
from hex_ai.enums import Player, Winner
from hex_ai.inference.game_engine import apply_move_to_state
from hex_ai.inference.katahex_client import (
    KataHexPlayer,
    alternating_player_for_move_index,
    build_katahex_override_config,
)
from hex_ai.inference.move_selection import MoveSelectionConfig, get_strategy
from hex_ai.inference.strategy_config import StrategyConfig
from hex_ai.move_provenance import (
    MOVE_CODE_VISIT_COUNT,
    make_move_provenance_record,
    sidecar_path_for_trmph,
)
from hex_ai.utils.deterministic_tournament_utils import (
    create_play_config_for_pair,
    save_opening_positions,
    setup_strategy_pair_files,
    setup_tournament_output,
)
from hex_ai.utils.format_conversion import rowcol_to_trmph, trmph_move_to_rowcol
from hex_ai.utils.random_utils import set_deterministic_seeds
from hex_ai.utils.tournament_logging import (
    append_trmph_winner_line,
    find_available_csv_filename,
    get_command_line,
    write_tournament_trmph_header,
)
from hex_ai.utils.tournament_utils import (
    create_strategy_configs_for_tournament,
    format_strategy_configuration_details,
    parse_model_specifications,
)
from hex_ai.inference.model_cache import create_temporary_model_cache
from hex_ai.inference.game_execution import (
    OpeningPosition,
    TOURNAMENT_NON_TRAINABLE_PROVENANCE_CODE,
    _build_policy_target_vector_from_gumbel_final_pair_scores,
    _build_policy_target_vector_from_mcts_result,
    _one_hot_policy_target_vector,
    _resolve_provenance_code_from_selected_move_source,
    _zero_policy_target_vector,
    find_trmph_files,
    generate_diverse_openings,
    load_openings_from_file,
    select_random_openings,
)


logger = logging.getLogger(__name__)

DEFAULT_OPENING_LENGTH = 5
DEFAULT_NUM_OPENINGS = 100
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = None
DEFAULT_VERBOSE = 1
TRMPH_SOURCE_DIR = "data/sf25/sep28"
OUTPUT_DIR_PREFIX = "data/tournament_play/katahex_vs_sf25/katahex_tournament_"

DEFAULT_KATAHEX_NAME = "KataHex"
DEFAULT_KATAHEX_ENGINE = "katahex/build/katahex"
DEFAULT_KATAHEX_CONFIG = "katahex/config.cfg"
DEFAULT_KATAHEX_MODEL = "katahex/hex3_27x_b28.bin.gz"
DEFAULT_KATAHEX_MAX_VISITS = 512
DEFAULT_KATAHEX_NUM_SEARCH_THREADS = 4
DEFAULT_KATAHEX_CACHE_POWER = 19
DEFAULT_KATAHEX_COMMAND_TIMEOUT = 300.0


class KataHexTournamentResult:
    """Tournament summary for Snowflake vs KataHex matches."""

    def __init__(self, sf25_models: List[str], katahex_name: str):
        self.sf25_models = sf25_models
        self.katahex_name = katahex_name
        self.results: Dict[str, Dict[str, Any]] = {}
        self.opening_results: Dict[str, List[Dict[str, Any]]] = {
            model_name: [] for model_name in sf25_models
        }

        for model_name in sf25_models:
            self.results[model_name] = self._new_result_bucket()

    @staticmethod
    def _new_result_bucket() -> Dict[str, Any]:
        return {
            "sf25_wins": 0,
            "katahex_wins": 0,
            "total_games": 0,
            "sf25_wins_as_blue": 0,
            "sf25_wins_as_red": 0,
            "katahex_wins_as_blue": 0,
            "katahex_wins_as_red": 0,
            "openings_sf25_won_both": 0,
            "openings_katahex_won_both": 0,
            "openings_split": 0,
        }

    def record_game(self, sf25_model: str, winner: str, game_data: Dict[str, Any]) -> None:
        bucket = self.results.setdefault(sf25_model, self._new_result_bucket())
        bucket["total_games"] += 1

        winner_color = game_data.get("winner")
        if winner == "sf25":
            bucket["sf25_wins"] += 1
            if winner_color == "blue":
                bucket["sf25_wins_as_blue"] += 1
            elif winner_color == "red":
                bucket["sf25_wins_as_red"] += 1
        elif winner == "katahex":
            bucket["katahex_wins"] += 1
            if winner_color == "blue":
                bucket["katahex_wins_as_blue"] += 1
            elif winner_color == "red":
                bucket["katahex_wins_as_red"] += 1
        else:
            raise ValueError(f"Unknown winner label: {winner!r}")

    def record_opening_results(
        self,
        sf25_model: str,
        opening_idx: int,
        opening: OpeningPosition,
        sf25_won_first: bool,
        sf25_won_second: bool,
    ) -> None:
        bucket = self.results.setdefault(sf25_model, self._new_result_bucket())
        if sf25_won_first and sf25_won_second:
            bucket["openings_sf25_won_both"] += 1
            outcome = "sf25_sweep"
        elif (not sf25_won_first) and (not sf25_won_second):
            bucket["openings_katahex_won_both"] += 1
            outcome = "katahex_sweep"
        else:
            bucket["openings_split"] += 1
            outcome = "split"

        opening_bare = opening.get_trmph_string(BOARD_SIZE)
        if opening_bare.startswith(TRMPH_PREFIX):
            opening_bare = opening_bare[len(TRMPH_PREFIX):]

        first_move = (
            rowcol_to_trmph(*opening.moves[0], BOARD_SIZE) if len(opening.moves) == 1 else None
        )
        self.opening_results.setdefault(sf25_model, []).append(
            {
                "opening_index": opening_idx,
                "opening_length": len(opening.moves),
                "opening_bare": opening_bare,
                "opening_trmph": opening.get_trmph_string(BOARD_SIZE),
                "first_move": first_move,
                "sf25_won_as_blue": sf25_won_first,
                "sf25_won_as_red": sf25_won_second,
                "sf25_won_both": sf25_won_first and sf25_won_second,
                "katahex_won_both": (not sf25_won_first) and (not sf25_won_second),
                "outcome": outcome,
            }
        )

    def get_summary(self) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "sf25_models": self.sf25_models,
            "katahex_name": self.katahex_name,
            "model_results": {},
        }
        for model_name, bucket in self.results.items():
            if bucket["total_games"] == 0:
                continue
            win_rate = bucket["sf25_wins"] / bucket["total_games"]
            summary["model_results"][model_name] = {
                **bucket,
                "sf25_win_rate": win_rate,
                "sf25_win_percentage": win_rate * 100.0,
                "opening_results": self.opening_results.get(model_name, []),
            }
        return summary

    def print_results(self) -> None:
        print("\n" + "=" * 60)
        print("KATAHEX vs SF25 TOURNAMENT RESULTS")
        print("=" * 60)
        print(f"Opponent: {self.katahex_name}")
        print(f"SF25 Models: {len(self.sf25_models)}")
        print()

        for model_name in self.sf25_models:
            bucket = self.results.get(model_name)
            if not bucket or bucket["total_games"] <= 0:
                continue

            win_rate = bucket["sf25_wins"] / bucket["total_games"]
            print(f"Model: {model_name}")
            print(f"  SF25 Wins: {bucket['sf25_wins']}")
            print(f"  {self.katahex_name} Wins: {bucket['katahex_wins']}")
            print(f"  Total Games: {bucket['total_games']}")
            print(f"  SF25 Win Rate: {win_rate * 100:.1f}%")
            print()
            print("  Color-specific wins:")
            print(f"    SF25 as Blue: {bucket['sf25_wins_as_blue']}")
            print(f"    SF25 as Red: {bucket['sf25_wins_as_red']}")
            print(f"    {self.katahex_name} as Blue: {bucket['katahex_wins_as_blue']}")
            print(f"    {self.katahex_name} as Red: {bucket['katahex_wins_as_red']}")
            print()

            total_openings = (
                bucket["openings_sf25_won_both"]
                + bucket["openings_katahex_won_both"]
                + bucket["openings_split"]
            )
            print(f"  Opening-level results (out of {total_openings} openings):")
            print(f"    Openings SF25 won both ways: {bucket['openings_sf25_won_both']}")
            print(
                f"    Openings {self.katahex_name} won both ways: "
                f"{bucket['openings_katahex_won_both']}"
            )
            print(f"    Openings split (1-1): {bucket['openings_split']}")

            winning_openings = [
                entry
                for entry in self.opening_results.get(model_name, [])
                if entry["sf25_won_both"]
            ]
            if winning_openings:
                print("  SF25 sweep openings:")
                for entry in winning_openings[:20]:
                    label = entry["first_move"] or entry["opening_bare"]
                    print(f"    {label}")
                if len(winning_openings) > 20:
                    print(f"    ... and {len(winning_openings) - 20} more")
            print()

        print("=" * 60)


def build_first_move_openings(move_texts: List[str]) -> List[OpeningPosition]:
    """Create openings from explicit first-move TRMPH coordinates."""
    openings: List[OpeningPosition] = []
    for idx, move_text in enumerate(move_texts, start=1):
        move = move_text.strip().lower()
        if not move:
            continue
        row, col = trmph_move_to_rowcol(move, BOARD_SIZE)
        openings.append(
            OpeningPosition(
                moves=[(row, col)],
                source_game=f"explicit_first_move_{idx}:{move}",
                opening_length=1,
            )
        )
    return openings


def build_all_first_move_openings() -> List[OpeningPosition]:
    """Create one opening for every legal first move on the 13x13 board."""
    return build_first_move_openings(
        [
            rowcol_to_trmph(row, col, BOARD_SIZE)
            for row in range(BOARD_SIZE)
            for col in range(BOARD_SIZE)
        ]
    )


def play_katahex_vs_sf25_game(
    model_cache: Any,
    sf25_strategy: StrategyConfig,
    katahex_player: KataHexPlayer,
    katahex_engine: Any,
    opening: OpeningPosition,
    temperature: float = DEFAULT_TEMPERATURE,
    board_size: int = BOARD_SIZE,
    verbose: int = 0,
    sf25_is_blue: bool = True,
) -> Dict[str, Any]:
    """Play one game between Snowflake and KataHex from a fixed opening."""
    state = opening.get_state(board_size)
    katahex_engine.load_moves(
        [
            (alternating_player_for_move_index(index), row, col)
            for index, (row, col) in enumerate(opening.moves)
        ],
        board_size=board_size,
    )

    move_config = MoveSelectionConfig(temperature=temperature, **sf25_strategy.config)
    sf25_strategy_obj = get_strategy(sf25_strategy.strategy_type)

    move_sequence = list(opening.moves)
    move_provenance_codes: List[str] = []
    policy_target_rows: List[np.ndarray] = []
    for opening_move in opening.moves:
        move_provenance_codes.append(MOVE_CODE_VISIT_COUNT)
        policy_target_rows.append(_one_hot_policy_target_vector(opening_move, board_size))

    move_count = len(opening.moves)
    max_moves = board_size * board_size

    while move_count < max_moves:
        if state.game_over:
            break

        if verbose >= 3 and move_count % 10 == 0 and move_count > 0:
            print(f"    Move {move_count}...", end="", flush=True)

        current_player = state.current_player_enum
        try:
            if (current_player == Player.BLUE and sf25_is_blue) or (
                current_player == Player.RED and not sf25_is_blue
            ):
                model = model_cache.get_simple_model(sf25_strategy.model_path)
                mcts_verbose = max(0, verbose - 2) if verbose >= 4 else 0
                row, col = sf25_strategy_obj.select_move(
                    state,
                    model,
                    move_config,
                    verbose=mcts_verbose,
                )
                katahex_engine.play_move(current_player, row, col, board_size=board_size)
                move_actor = "SF25"
                move_metadata = sf25_strategy_obj.pop_last_move_metadata()
                if move_metadata is None:
                    provenance_code = TOURNAMENT_NON_TRAINABLE_PROVENANCE_CODE
                    policy_target = _zero_policy_target_vector(board_size)
                else:
                    provenance_code = _resolve_provenance_code_from_selected_move_source(
                        move_metadata.get("selected_move_source")
                    )
                    if provenance_code == "G":
                        mcts_result = move_metadata.get("mcts_result")
                        if mcts_result is None:
                            raise RuntimeError(
                                "Gumbel-root SF25 tournament move missing MCTS result payload."
                            )
                        policy_target = _build_policy_target_vector_from_gumbel_final_pair_scores(
                            mcts_result,
                            board_size=board_size,
                        )
                    elif provenance_code in {"V", "T"}:
                        mcts_result = move_metadata.get("mcts_result")
                        if mcts_result is None:
                            raise RuntimeError(
                                "Trainable SF25 tournament move missing MCTS result payload."
                            )
                        policy_target = _build_policy_target_vector_from_mcts_result(
                            mcts_result,
                            board_size=board_size,
                        )
                    else:
                        policy_target = _zero_policy_target_vector(board_size)
            else:
                if katahex_player.policy_only:
                    row, col = katahex_engine.raw_policy_move(
                        current_player,
                        board_size=board_size,
                    )
                    move_actor = f"{katahex_player.name} policy"
                else:
                    row, col = katahex_engine.genmove(
                        current_player,
                        board_size=board_size,
                    )
                    move_actor = katahex_player.name
                provenance_code = TOURNAMENT_NON_TRAINABLE_PROVENANCE_CODE
                policy_target = _zero_policy_target_vector(board_size)

            state = apply_move_to_state(state, row, col)
            move_sequence.append((row, col))
            move_provenance_codes.append(provenance_code)
            policy_target_rows.append(policy_target)
            move_count += 1

            if verbose >= 4:
                print(
                    f"  Move {move_count}: {rowcol_to_trmph(row, col, board_size)} "
                    f"by {move_actor}"
                )

        except Exception as exc:
            logger.error("Error during KataHex game play: %s", exc)
            break

    if verbose >= 3 and move_count > 10:
        print()

    trmph_moves = "".join(rowcol_to_trmph(r, c, board_size) for r, c in move_sequence)
    trmph_str = f"{TRMPH_PREFIX}{trmph_moves}"

    if not state.game_over:
        raise RuntimeError(
            "Game did not finish while playing KataHex.\n"
            f"  Final TRMPH: {trmph_str}\n"
            f"  Opening: {opening}"
        )
    if state.winner is None:
        raise RuntimeError(
            "Game is over but winner is None while playing KataHex.\n"
            f"  Final TRMPH: {trmph_str}"
        )

    winner = state.winner
    if winner == Winner.BLUE:
        winner_str = "blue"
        winner_strategy = "SF25" if sf25_is_blue else katahex_player.name
    elif winner == Winner.RED:
        winner_str = "red"
        winner_strategy = katahex_player.name if sf25_is_blue else "SF25"
    else:
        raise RuntimeError(f"Unknown winner enum: {winner!r}")

    if verbose >= 1:
        print(f"    Game complete: {winner_strategy} wins in {move_count} moves")

    expected_move_count = len(move_sequence)
    if len(move_provenance_codes) != expected_move_count:
        raise RuntimeError(
            "KataHex tournament move provenance length mismatch: "
            f"expected {expected_move_count}, got {len(move_provenance_codes)}"
        )
    if len(policy_target_rows) != expected_move_count:
        raise RuntimeError(
            "KataHex tournament policy-target row count mismatch: "
            f"expected {expected_move_count}, got {len(policy_target_rows)}"
        )

    if policy_target_rows:
        policy_targets_matrix = np.stack(policy_target_rows, axis=0).astype(
            np.float32,
            copy=False,
        )
    else:
        policy_targets_matrix = np.zeros((0, board_size * board_size), dtype=np.float32)

    return {
        "winner": winner_str,
        "winner_char": winner_str[0],
        "winner_strategy": winner_strategy,
        "trmph_str": trmph_str,
        "total_moves": move_count,
        "opening": opening,
        "sf25_strategy": sf25_strategy.name,
        "katahex_name": katahex_player.name,
        "move_provenance_codes": "".join(move_provenance_codes),
        "policy_targets_matrix": policy_targets_matrix,
        "policy_target_source_codes": "".join(move_provenance_codes),
        "policy_target_version": POLICY_TARGET_CONSTRUCTION_VERSION,
    }


def run_katahex_tournament(
    strategy_configs: List[StrategyConfig],
    katahex_player: KataHexPlayer,
    openings: List[OpeningPosition],
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: int = DEFAULT_VERBOSE,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    command_line: Optional[str] = None,
    run_desc: Optional[str] = None,
) -> tuple[KataHexTournamentResult, str]:
    """Run Snowflake-vs-KataHex matches across all selected openings."""
    result = KataHexTournamentResult(
        sf25_models=[config.name for config in strategy_configs],
        katahex_name=katahex_player.name,
    )

    if output_dir is None:
        output_dir, openings_file = setup_tournament_output(OUTPUT_DIR_PREFIX)
    else:
        os.makedirs(output_dir, exist_ok=True)
        openings_file = os.path.join(output_dir, "openings.txt")
    save_opening_positions(openings, openings_file)

    for strategy_config in strategy_configs:
        logger.info(
            "Playing %d openings: %s vs %s",
            len(openings),
            strategy_config.name,
            katahex_player.name,
        )

        model_cache = create_temporary_model_cache([strategy_config.model_path], verbose=0)
        trmph_file, csv_file = setup_strategy_pair_files(output_dir, strategy_config, katahex_player)
        play_config = create_play_config_for_pair(
            strategy_config,
            katahex_player,
            temperature,
            seed,
            command_line,
            run_desc=run_desc,
        )
        pair_model_paths = [strategy_config.model_path, katahex_player.model_path]
        pair_strategy_configs = [strategy_config, katahex_player]
        actual_trmph_file = write_tournament_trmph_header(
            trmph_file,
            pair_model_paths,
            len(openings),
            play_config,
            BOARD_SIZE,
            strategy_configs=pair_strategy_configs,
        )
        provenance_file = str(sidecar_path_for_trmph(actual_trmph_file))
        if os.path.exists(provenance_file):
            raise RuntimeError(
                "Expected fresh provenance sidecar path for KataHex tournament but file already exists: "
                f"{provenance_file}"
            )
        provenance_records_written = 0
        _ = find_available_csv_filename(csv_file)

        with katahex_player.create_engine() as katahex_engine, open(
            provenance_file,
            "w",
            encoding="utf-8",
        ) as provenance_handle:
            for opening_idx, opening in enumerate(openings):
                if verbose >= 1:
                    print(f"  Game {opening_idx + 1}/{len(openings)}: {opening}")

                result_1 = play_katahex_vs_sf25_game(
                    model_cache,
                    strategy_config,
                    katahex_player,
                    katahex_engine,
                    opening,
                    temperature,
                    verbose=verbose,
                    sf25_is_blue=True,
                )
                result_2 = play_katahex_vs_sf25_game(
                    model_cache,
                    strategy_config,
                    katahex_player,
                    katahex_engine,
                    opening,
                    temperature,
                    verbose=verbose,
                    sf25_is_blue=False,
                )

                winner_1 = "sf25" if result_1["winner_strategy"] == "SF25" else "katahex"
                winner_2 = "sf25" if result_2["winner_strategy"] == "SF25" else "katahex"

                result.record_game(strategy_config.name, winner_1, result_1)
                result.record_game(strategy_config.name, winner_2, result_2)
                result.record_opening_results(
                    strategy_config.name,
                    opening_idx,
                    opening,
                    winner_1 == "sf25",
                    winner_2 == "sf25",
                )

                append_trmph_winner_line(
                    result_1["trmph_str"],
                    result_1["winner_char"],
                    actual_trmph_file,
                )
                append_trmph_winner_line(
                    result_2["trmph_str"],
                    result_2["winner_char"],
                    actual_trmph_file,
                )

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

        print()
        print(f"Results for {strategy_config.name}:")
        bucket = result.results[strategy_config.name]
        if bucket["total_games"] > 0:
            win_rate = bucket["sf25_wins"] / bucket["total_games"]
            print(f"  SF25 Wins: {bucket['sf25_wins']}")
            print(f"  {katahex_player.name} Wins: {bucket['katahex_wins']}")
            print(f"  Total Games: {bucket['total_games']}")
            print(f"  SF25 Win Rate: {win_rate:.3f}")
        print()

    return result, output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tournament between Snowflake 2025 models and a local KataHex engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --models=best --strategies=mcts --mcts-sims=122 --num-openings=20
  %(prog)s --models=best --strategies=mcts --mcts-sims=122 --first-move-sweep
  %(prog)s --models=best --strategies=mcts --mcts-sims=122 --first-move-list=a1,g7,m13
        """,
    )

    parser.add_argument("--models", type=str, help='Comma-separated model registry names (e.g. "best,model2")')
    parser.add_argument("--model-files", type=str, help='Comma-separated model file names')
    parser.add_argument("--model-dirs", type=str, help="Comma-separated model directories")
    parser.add_argument("--strategies", type=str, required=True, help='Comma-separated strategies (e.g. "mcts,policy")')

    parser.add_argument("--num-openings", type=int, default=DEFAULT_NUM_OPENINGS, help=f"Number of opening positions to use (default: {DEFAULT_NUM_OPENINGS})")
    parser.add_argument("--opening-length", type=int, default=DEFAULT_OPENING_LENGTH, help=f"Number of moves per opening (default: {DEFAULT_OPENING_LENGTH})")
    parser.add_argument("--opening-file", type=str, help="File containing pre-generated openings")
    parser.add_argument("--cache-file", type=str, help="File to cache generated openings")
    parser.add_argument("--trmph-source", type=str, default=TRMPH_SOURCE_DIR, help=f"Directory containing TRMPH files for opening generation (default: {TRMPH_SOURCE_DIR})")
    parser.add_argument("--first-move-sweep", action="store_true", help="Use every legal first move as a separate opening")
    parser.add_argument("--first-move-list", type=str, help='Comma-separated TRMPH first moves to test, e.g. "a1,g7,m13"')

    parser.add_argument("--mcts-sims", type=str, help=f"Comma-separated MCTS simulation counts (default: {DEFAULT_MCTS_SIMS})")
    parser.add_argument("--base-fraction-mcts-moves", type=str, help="Comma-separated fraction of moves using full MCTS")
    parser.add_argument("--batch-sizes", type=str, help=f"Comma-separated batch sizes (default: {DEFAULT_BATCH_CAP})")
    parser.add_argument("--c-puct", type=str, help=f"Comma-separated PUCT constants (default: {DEFAULT_C_PUCT})")
    parser.add_argument("--enable-gumbel", type=str, help="Comma-separated booleans to enable Gumbel root selection")
    parser.add_argument("--gumbel-sim-threshold", type=str, help=f"Comma-separated Gumbel sim thresholds (default: {DEFAULT_GUMBEL_SIM_THRESHOLD})")
    parser.add_argument("--gumbel-candidate-power-scale", type=str, help=f"Comma-separated Gumbel power scales (default: {DEFAULT_GUMBEL_CANDIDATE_POWER_SCALE})")
    parser.add_argument("--gumbel-candidate-power-rate", type=str, help=f"Comma-separated Gumbel power rates (default: {DEFAULT_GUMBEL_CANDIDATE_POWER_RATE})")
    parser.add_argument("--gumbel-candidate-power-offset", type=str, help=f"Comma-separated Gumbel power offsets (default: {DEFAULT_GUMBEL_CANDIDATE_POWER_OFFSET})")
    parser.add_argument("--gumbel-c-scale", type=str, help=f"Comma-separated Gumbel c_scale values (default: {DEFAULT_GUMBEL_C_SCALE})")
    parser.add_argument("--enable-dead-cell-pruning", type=str, help="Comma-separated booleans for dead-cell pruning")
    parser.add_argument("--dead-cell-enable-four-run", type=str, help="Comma-separated booleans for dead-cell D1 motif")
    parser.add_argument("--dead-cell-enable-two-two-split", type=str, help="Comma-separated booleans for dead-cell D2 motif")
    parser.add_argument("--dead-cell-enable-three-plus-one", type=str, help="Comma-separated booleans for dead-cell D3 motif")
    parser.add_argument("--dead-cell-enable-a1b2a3-discouraged", type=str, help="Comma-separated booleans for A1B2A3 discouraged motif")
    parser.add_argument("--dead-cell-enable-double-dead-pairs", type=str, help="Comma-separated booleans for dead-pair motif")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE, help=f"Global temperature (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--temperatures", type=str, help="Comma-separated per-strategy temperatures")

    parser.add_argument("--katahex-name", type=str, default=DEFAULT_KATAHEX_NAME, help=f"Display name for the opponent (default: {DEFAULT_KATAHEX_NAME})")
    parser.add_argument("--katahex-engine", type=str, default=DEFAULT_KATAHEX_ENGINE, help=f"Path to the KataHex binary (default: {DEFAULT_KATAHEX_ENGINE})")
    parser.add_argument("--katahex-config", type=str, default=DEFAULT_KATAHEX_CONFIG, help=f"Path to the KataHex config file (default: {DEFAULT_KATAHEX_CONFIG})")
    parser.add_argument("--katahex-model", type=str, default=DEFAULT_KATAHEX_MODEL, help=f"Path to the KataHex model file (default: {DEFAULT_KATAHEX_MODEL})")
    parser.add_argument("--katahex-max-visits", type=int, default=DEFAULT_KATAHEX_MAX_VISITS, help=f"KataHex maxVisits/maxPlayouts override (default: {DEFAULT_KATAHEX_MAX_VISITS})")
    parser.add_argument("--katahex-num-threads", type=int, help=f"Convenience CPU-thread budget for KataHex. Sets numSearchThreads, and on Eigen/CPU builds also sets numEigenThreadsPerModel unless overridden separately (default search threads: {DEFAULT_KATAHEX_NUM_SEARCH_THREADS})")
    parser.add_argument("--katahex-num-search-threads", type=int, help=f"KataHex numSearchThreads override (default: {DEFAULT_KATAHEX_NUM_SEARCH_THREADS})")
    parser.add_argument("--katahex-num-eigen-threads", type=int, help="KataHex numEigenThreadsPerModel override for Eigen/CPU builds (default: follow numSearchThreads)")
    parser.add_argument("--katahex-nn-cache-power", type=int, default=DEFAULT_KATAHEX_CACHE_POWER, help=f"KataHex nnCacheSizePowerOfTwo override (default: {DEFAULT_KATAHEX_CACHE_POWER})")
    parser.add_argument("--katahex-command-timeout", type=float, default=DEFAULT_KATAHEX_COMMAND_TIMEOUT, help=f"Per-command timeout in seconds for KataHex GTP commands (default: {DEFAULT_KATAHEX_COMMAND_TIMEOUT})")
    parser.add_argument("--katahex-override-config", type=str, help="Additional raw KataHex override-config entries to append")
    parser.add_argument("--katahex-policy-only", action="store_true", help="Use KataHex raw neural-network policy directly instead of genmove/MCTS")

    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for opening selection")
    parser.add_argument("--verbose", type=int, default=DEFAULT_VERBOSE, help=f"Verbosity level (default: {DEFAULT_VERBOSE})")
    parser.add_argument("--run-desc", type=str, help="Optional description for the tournament run")

    return parser.parse_args()


def resolve_openings(args: argparse.Namespace) -> List[OpeningPosition]:
    """Load openings from either explicit first-move settings or standard sources."""
    if args.first_move_sweep and args.first_move_list:
        raise ValueError("Use only one of --first-move-sweep or --first-move-list")

    if args.first_move_sweep:
        return build_all_first_move_openings()

    if args.first_move_list:
        openings = build_first_move_openings(
            [move.strip() for move in args.first_move_list.split(",")]
        )
        if not openings:
            raise ValueError("No valid moves were provided via --first-move-list")
        return openings

    if args.opening_file and os.path.exists(args.opening_file):
        print(f"Loading openings from: {args.opening_file}")
        return load_openings_from_file(args.opening_file, args.opening_length)

    print("Generating diverse openings...")
    trmph_files = find_trmph_files(args.trmph_source)
    if not trmph_files:
        raise ValueError(f"No TRMPH files found in {args.trmph_source}")

    all_openings = generate_diverse_openings(
        trmph_files,
        opening_length=args.opening_length,
        target_count=args.num_openings,
        cache_file=args.cache_file,
    )
    if not all_openings:
        raise ValueError("No opening positions generated")

    print(
        f"Randomly selecting {args.num_openings} openings from pool of {len(all_openings)}..."
    )
    return select_random_openings(all_openings, args.num_openings, seed=args.seed)


def validate_katahex_paths(args: argparse.Namespace) -> None:
    """Fail fast if required KataHex files are missing."""
    for label, path in (
        ("KataHex engine", args.katahex_engine),
        ("KataHex config", args.katahex_config),
        ("KataHex model", args.katahex_model),
    ):
        if not os.path.exists(path):
            raise ValueError(f"{label} path does not exist: {path}")

    if not os.path.isfile(args.katahex_engine):
        raise ValueError(f"KataHex engine path is not a file: {args.katahex_engine}")
    if not os.access(args.katahex_engine, os.X_OK):
        raise ValueError(f"KataHex engine is not executable: {args.katahex_engine}")


def main() -> None:
    args = parse_args()

    try:
        command_line = get_command_line()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    if args.seed is None:
        args.seed = int(time.time())
        print(f"Auto-generated seed: {args.seed}")
    set_deterministic_seeds(args.seed)

    if args.models and (args.model_files or args.model_dirs):
        print("ERROR: Cannot specify both --models and --model-files/--model-dirs.")
        sys.exit(1)
    if not args.models and not (args.model_files and args.model_dirs):
        print("ERROR: Must specify either --models or both --model-files and --model-dirs.")
        sys.exit(1)

    try:
        validate_katahex_paths(args)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    strategy_names = [name.strip() for name in args.strategies.split(",")]
    model_paths = parse_model_specifications(args, strategy_names)

    try:
        strategy_configs = create_strategy_configs_for_tournament(
            args,
            strategy_names,
            model_paths,
            num_games=args.num_openings,
            board_size=BOARD_SIZE,
            pie_rule=False,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    try:
        openings = resolve_openings(args)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    effective_katahex_num_search_threads = (
        args.katahex_num_search_threads
        if args.katahex_num_search_threads is not None
        else (
            args.katahex_num_threads
            if args.katahex_num_threads is not None
            else DEFAULT_KATAHEX_NUM_SEARCH_THREADS
        )
    )
    effective_katahex_num_eigen_threads = (
        args.katahex_num_eigen_threads
        if args.katahex_num_eigen_threads is not None
        else args.katahex_num_threads
    )
    if effective_katahex_num_search_threads <= 0:
        print("ERROR: --katahex-num-search-threads/--katahex-num-threads must be positive.")
        sys.exit(1)
    if (
        effective_katahex_num_eigen_threads is not None
        and effective_katahex_num_eigen_threads <= 0
    ):
        print("ERROR: --katahex-num-eigen-threads/--katahex-num-threads must be positive when provided.")
        sys.exit(1)

    katahex_override_config = build_katahex_override_config(
        max_visits=args.katahex_max_visits,
        num_search_threads=effective_katahex_num_search_threads,
        num_eigen_threads=effective_katahex_num_eigen_threads,
        nn_cache_size_power_of_two=args.katahex_nn_cache_power,
        extra_override_config=args.katahex_override_config,
    )
    katahex_player = KataHexPlayer(
        name=args.katahex_name,
        engine_path=args.katahex_engine,
        config_path=args.katahex_config,
        model_path=args.katahex_model,
        override_config=katahex_override_config,
        policy_only=args.katahex_policy_only,
        board_size=BOARD_SIZE,
        command_timeout=args.katahex_command_timeout,
        startup_timeout=args.katahex_command_timeout,
    )

    print("\n" + "=" * 60)
    print("KATAHEX vs SF25 TOURNAMENT CONFIGURATION")
    print("=" * 60)
    print(f"SF25 Models: {len(strategy_configs)}")
    for config in strategy_configs:
        print(f"  - {config.name}")
    print("SF25 Strategy configurations:")
    for config in strategy_configs:
        print(f"  - {config.name}: {format_strategy_configuration_details(config)}")
    print(f"KataHex Name: {args.katahex_name}")
    print(f"KataHex Engine: {args.katahex_engine}")
    print(f"KataHex Config: {args.katahex_config}")
    print(f"KataHex Model: {args.katahex_model}")
    print(f"KataHex Policy Only: {args.katahex_policy_only}")
    print(f"KataHex Search Threads: {effective_katahex_num_search_threads}")
    if effective_katahex_num_eigen_threads is None:
        print("KataHex Eigen Threads: default (follow KataHex/Eigen default, normally numSearchThreads)")
    else:
        print(f"KataHex Eigen Threads: {effective_katahex_num_eigen_threads}")
    print(f"KataHex Override Config: {katahex_override_config}")
    print(f"Number of openings: {len(openings)}")
    if args.first_move_sweep:
        print("Opening source: all legal first moves")
    elif args.first_move_list:
        print("Opening source: explicit first-move list")
    else:
        print(f"Opening length: {args.opening_length}")
    print(f"Temperature: {args.temperature}")
    print(
        "Early termination threshold (MCTS): "
        f"{TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD}"
    )
    print(f"Random seed: {args.seed}")
    print("=" * 60)
    print()

    result, output_dir = run_katahex_tournament(
        strategy_configs=strategy_configs,
        katahex_player=katahex_player,
        openings=openings,
        temperature=args.temperature,
        verbose=args.verbose,
        seed=args.seed,
        command_line=command_line,
        run_desc=args.run_desc,
    )

    result.print_results()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        output_dir,
        f"katahex_tournament_results_{timestamp}.json",
    )
    with open(results_file, "w", encoding="utf-8") as handle:
        json.dump(result.get_summary(), handle, indent=2)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
