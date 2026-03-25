"""Simple benchmark CLI for ladder-template certificate matching."""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import random
from pathlib import Path
from time import perf_counter

import numpy as np

from hex_ai.enums import Piece

from .data_targets import find_ladder_matches_for_training_example
from .library_loader import (
    LoadedLadderTemplate,
    load_hexwiki_generated_ladder_templates,
    load_ladder_template_library,
)
from .matcher import find_ladder_template_matches


def _build_synthetic_board(
    template: LoadedLadderTemplate,
    *,
    board_size: int,
    origin_row: int,
    origin_col: int,
    fill_empty_with_attacker: bool,
) -> np.ndarray:
    board = np.full((board_size, board_size), Piece.EMPTY.value, dtype="U1")
    for local_row, local_col in template.attacker_required_local:
        board[origin_row + local_row, origin_col + local_col] = Piece.RED.value
    for local_row, local_col in template.defender_required_local:
        board[origin_row + local_row, origin_col + local_col] = Piece.BLUE.value
    if fill_empty_with_attacker:
        for local_row, local_col in template.empty_carrier_local:
            board[origin_row + local_row, origin_col + local_col] = Piece.RED.value
    return board


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark ladder-template matching on synthetic positions.",
    )
    parser.add_argument(
        "--library-dir",
        type=str,
        default=None,
        help="Optional ladder-template directory. Defaults to the generated HexWiki corpus.",
    )
    parser.add_argument(
        "--board-size",
        type=int,
        default=13,
        help="Board size for synthetic benchmark positions.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of full benchmark passes over the synthetic board set.",
    )
    parser.add_argument(
        "--fill-empty-with-attacker",
        action="store_true",
        help="Populate template empty carrier cells with attacker stones.",
    )
    parser.add_argument(
        "--must-include-carrier-tail",
        action="store_true",
        help="Filter each match run to embeddings containing the last carrier cell.",
    )
    parser.add_argument(
        "--orientations",
        nargs="*",
        default=["red_bottom", "blue_right"],
        help="Orientations to search. Default: red_bottom blue_right",
    )
    parser.add_argument(
        "--processed-shard",
        type=str,
        default=None,
        help="Optional processed training shard (.pkl.gz). When set, benchmark real positions instead of synthetic boards.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=128,
        help="Maximum number of processed examples to benchmark in real-data mode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for processed-example sampling.",
    )
    parser.add_argument(
        "--compare-last-move-filter",
        action="store_true",
        help="In processed-shard mode, compare full scan versus last-move-filter matching.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.board_size <= 0:
        raise ValueError(f"--board-size must be positive, got {args.board_size}")
    if args.iterations <= 0:
        raise ValueError(f"--iterations must be positive, got {args.iterations}")

    if args.library_dir:
        templates = load_ladder_template_library(args.library_dir, source_set="custom")
    else:
        templates = load_hexwiki_generated_ladder_templates()

    if args.processed_shard is not None:
        summary = _benchmark_processed_shard(
            Path(args.processed_shard),
            templates=templates,
            sample_size=args.sample_size,
            seed=args.seed,
            orientations=tuple(args.orientations),
            compare_last_move_filter=bool(args.compare_last_move_filter),
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    boards: list[tuple[np.ndarray, tuple[int, int] | None]] = []
    for template in templates:
        origin_row = max(0, (args.board_size - template.local_rows) // 2)
        origin_col = max(0, (args.board_size - template.local_cols) // 2)
        board = _build_synthetic_board(
            template,
            board_size=args.board_size,
            origin_row=origin_row,
            origin_col=origin_col,
            fill_empty_with_attacker=args.fill_empty_with_attacker,
        )
        must_include_cell = None
        if args.must_include_carrier_tail:
            local_row, local_col = template.carrier_local[-1]
            must_include_cell = (origin_row + local_row, origin_col + local_col)
        boards.append((board, must_include_cell))

    started_at = perf_counter()
    total_embeddings = 0
    total_matches = 0
    total_elapsed_ms = 0.0
    for _ in range(args.iterations):
        for board, must_include_cell in boards:
            result = find_ladder_template_matches(
                board,
                templates,
                orientations=tuple(args.orientations),
                must_include_cell=must_include_cell,
                allow_attacker_superset_on_empty=True,
            )
            total_embeddings += result.stats.embeddings_considered
            total_matches += result.stats.matches_found
            total_elapsed_ms += result.stats.elapsed_ms

    wall_elapsed_ms = (perf_counter() - started_at) * 1000.0
    position_count = len(boards) * args.iterations
    summary = {
        "positions_benchmarked": position_count,
        "template_count": len(templates),
        "orientations": list(args.orientations),
        "fill_empty_with_attacker": bool(args.fill_empty_with_attacker),
        "must_include_carrier_tail": bool(args.must_include_carrier_tail),
        "matcher_elapsed_ms_total": total_elapsed_ms,
        "matcher_elapsed_ms_avg_per_position": total_elapsed_ms / position_count,
        "wall_elapsed_ms_total": wall_elapsed_ms,
        "embeddings_considered_total": total_embeddings,
        "embeddings_considered_avg_per_position": total_embeddings / position_count,
        "matches_found_total": total_matches,
        "matches_found_avg_per_position": total_matches / position_count,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


def _benchmark_processed_shard(
    processed_shard: Path,
    *,
    templates: tuple[LoadedLadderTemplate, ...],
    sample_size: int,
    seed: int,
    orientations: tuple[str, ...],
    compare_last_move_filter: bool,
) -> dict:
    if sample_size <= 0:
        raise ValueError(f"--sample-size must be positive, got {sample_size}")
    if not processed_shard.exists():
        raise FileNotFoundError(f"Processed shard not found: {processed_shard}")

    payload = _load_processed_shard_payload(processed_shard)
    if not isinstance(payload, dict) or "examples" not in payload:
        raise ValueError(
            f"Processed shard {processed_shard} does not contain an 'examples' payload."
        )
    examples = payload["examples"]
    if not isinstance(examples, list) or not examples:
        raise ValueError(f"Processed shard {processed_shard} contains no examples.")

    rng = random.Random(seed)
    if len(examples) <= sample_size:
        sampled_examples = list(examples)
    else:
        sampled_examples = rng.sample(examples, sample_size)

    started_at = perf_counter()
    full_scan_elapsed_ms = 0.0
    full_scan_embeddings = 0
    full_scan_matches = 0

    filtered_scan_elapsed_ms = 0.0
    filtered_scan_embeddings = 0
    filtered_scan_matches = 0
    filtered_positions_used = 0
    filtered_positions_fell_back = 0
    filtered_lookup_errors = 0

    for example in sampled_examples:
        full_match = find_ladder_matches_for_training_example(
            example,
            templates,
            orientations=orientations,
            use_last_move_filter=False,
            allow_last_move_lookup_fallback=False,
        )
        full_scan_elapsed_ms += full_match.match_result.stats.elapsed_ms
        full_scan_embeddings += full_match.match_result.stats.embeddings_considered
        full_scan_matches += full_match.match_result.stats.matches_found

        if compare_last_move_filter:
            filtered_match = find_ladder_matches_for_training_example(
                example,
                templates,
                orientations=orientations,
                use_last_move_filter=True,
                allow_last_move_lookup_fallback=True,
            )
            filtered_scan_elapsed_ms += filtered_match.match_result.stats.elapsed_ms
            filtered_scan_embeddings += filtered_match.match_result.stats.embeddings_considered
            filtered_scan_matches += filtered_match.match_result.stats.matches_found
            if filtered_match.used_last_move_filter:
                filtered_positions_used += 1
            else:
                filtered_positions_fell_back += 1
            if filtered_match.last_move_lookup_error is not None:
                filtered_lookup_errors += 1

    wall_elapsed_ms = (perf_counter() - started_at) * 1000.0
    position_count = len(sampled_examples)
    summary = {
        "mode": "processed_shard",
        "processed_shard": str(processed_shard),
        "positions_benchmarked": position_count,
        "template_count": len(templates),
        "orientations": list(orientations),
        "full_scan": {
            "matcher_elapsed_ms_total": full_scan_elapsed_ms,
            "matcher_elapsed_ms_avg_per_position": full_scan_elapsed_ms / position_count,
            "embeddings_considered_total": full_scan_embeddings,
            "embeddings_considered_avg_per_position": full_scan_embeddings / position_count,
            "matches_found_total": full_scan_matches,
            "matches_found_avg_per_position": full_scan_matches / position_count,
        },
        "wall_elapsed_ms_total": wall_elapsed_ms,
    }
    if compare_last_move_filter:
        summary["last_move_filter"] = {
            "filter_semantics": "only embeddings whose carrier contains the reconstructed last move",
            "matcher_elapsed_ms_total": filtered_scan_elapsed_ms,
            "matcher_elapsed_ms_avg_per_position": filtered_scan_elapsed_ms / position_count,
            "embeddings_considered_total": filtered_scan_embeddings,
            "embeddings_considered_avg_per_position": filtered_scan_embeddings / position_count,
            "matches_found_total": filtered_scan_matches,
            "matches_found_avg_per_position": filtered_scan_matches / position_count,
            "positions_using_filter": filtered_positions_used,
            "positions_fell_back_to_full_scan": filtered_positions_fell_back,
            "last_move_lookup_errors": filtered_lookup_errors,
        }
    return summary


def _load_processed_shard_payload(processed_shard: Path):
    with open(processed_shard, "rb") as handle:
        prefix = handle.read(2)
    if prefix == b"\x1f\x8b":
        with gzip.open(processed_shard, "rb") as handle:
            return pickle.load(handle)
    with open(processed_shard, "rb") as handle:
        return pickle.load(handle)


if __name__ == "__main__":
    main()
