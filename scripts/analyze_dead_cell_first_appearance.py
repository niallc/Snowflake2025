#!/usr/bin/env python3
"""Report the first board state where dead cells appear in TRMPH game logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from hex_ai.inference.game_engine import apply_move_to_state, make_empty_hex_state
from hex_ai.utils.format_conversion import (
    rowcol_to_trmph,
    split_trmph_moves,
    trmph_move_to_rowcol,
)
from hex_ai.utils.weaks_cells import find_dead_cells


TRMPH_PREFIX_RE = re.compile(r"^#(\d+),")


def _iter_game_entries(input_dir: Path) -> Iterator[Tuple[str, int, str]]:
    """Yield (filename, line_no, trmph_string) entries in stable file/line order."""
    trmph_files = sorted(input_dir.glob("*.trmph"))
    for trmph_file in trmph_files:
        with trmph_file.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                if line.startswith("#") and TRMPH_PREFIX_RE.match(line) is None:
                    continue
                token = line.split()[0]
                if TRMPH_PREFIX_RE.match(token) is None:
                    continue
                yield trmph_file.name, line_no, token


def _find_first_dead_state(
    trmph: str,
) -> Tuple[Optional[str], Optional[int], List[str], int, int]:
    """Return first dead-cell state url suffix and metadata for one game."""
    match = TRMPH_PREFIX_RE.match(trmph)
    if match is None:
        raise ValueError(f"Invalid TRMPH string: {trmph!r}")

    board_size = int(match.group(1))
    moves_blob = trmph[match.end() :]
    move_tokens = split_trmph_moves(moves_blob)
    state = make_empty_hex_state(board_size=board_size)

    for move_index, move in enumerate(move_tokens, start=1):
        row, col = trmph_move_to_rowcol(move, board_size=board_size)
        state = apply_move_to_state(state, row, col)
        dead_cells = sorted(find_dead_cells(state.board))
        if dead_cells:
            dead_labels = [rowcol_to_trmph(r, c, board_size) for (r, c) in dead_cells]
            state_moves = "".join(rowcol_to_trmph(r, c, board_size) for (r, c) in state.move_history)
            return (
                f"#{board_size},{state_moves}",
                move_index,
                dead_labels,
                len(dead_labels),
                len(move_tokens),
            )

    return None, None, [], 0, len(move_tokens)


def _build_report_lines(
    input_dir: Path,
    max_games: int,
) -> List[str]:
    lines: List[str] = []
    game_counter = 0
    for file_name, line_no, trmph in _iter_game_entries(input_dir):
        game_counter += 1
        if game_counter > max_games:
            break

        state_suffix, first_move_idx, dead_labels, dead_count, total_moves = _find_first_dead_state(trmph)
        if state_suffix is None:
            lines.append(
                f"{game_counter:03d} | {file_name}:{line_no} | NO_DEAD_CELL_BY_END "
                f"(moves={total_moves}) | https://trmph.com/hex/board{trmph}"
            )
            continue

        preview = ",".join(dead_labels[:12])
        if len(dead_labels) > 12:
            preview = f"{preview},..."
        lines.append(
            f"{game_counter:03d} | {file_name}:{line_no} | first_dead_move={first_move_idx}/{total_moves} "
            f"| dead_cells={dead_count} [{preview}] | https://trmph.com/hex/board{state_suffix}"
        )

    return lines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find the first board state where dead cells appear in TRMPH logs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/sf25/mar1"),
        help="Directory containing .trmph files (default: data/sf25/mar1)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=50,
        help="How many games to analyze (default: 50)",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/sf25/mar1/dead_cell_first_appearance_first50.txt"),
        help="Path for report output file",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_games <= 0:
        raise ValueError(f"--max-games must be positive, got {args.max_games}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    report_lines = _build_report_lines(args.input_dir, args.max_games)
    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    header = [
        f"# input_dir={args.input_dir}",
        f"# max_games={args.max_games}",
        "# format:",
        "# <game_idx> | <source_file:line> | first_dead_move=M/T | dead_cells=N [coord,...] | <trmph_url>",
        "# OR",
        "# <game_idx> | <source_file:line> | NO_DEAD_CELL_BY_END (moves=T) | <trmph_url>",
        "",
    ]
    args.output_file.write_text("\n".join(header + report_lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(report_lines)} entries to {args.output_file}")


if __name__ == "__main__":
    main()
