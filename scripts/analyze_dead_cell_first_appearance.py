#!/usr/bin/env python3
"""Report the first board state where dead cells appear in TRMPH game logs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from hex_ai.inference.game_engine import apply_move_to_state, make_empty_hex_state
from hex_ai.utils.format_conversion import (
    rowcol_to_trmph,
    split_trmph_moves,
    trmph_move_to_rowcol,
)
from hex_ai.utils.weaks_cells import find_dead_cells


TRMPH_PREFIX_RE = re.compile(r"^#(\d+),")


def _rule_cfg(**overrides: bool) -> Dict[str, bool]:
    cfg: Dict[str, bool] = {
        "enable_four_run": False,
        "enable_two_two_split": False,
        "enable_three_plus_one": False,
        "enable_a1b2a3_discouraged": False,
        "enable_double_dead_pairs": False,
    }
    cfg.update(overrides)
    return cfg


RULE_PRESETS: Dict[str, Dict[str, bool]] = {
    "all_masked": _rule_cfg(
        enable_four_run=True,
        enable_two_two_split=True,
        enable_three_plus_one=True,
        enable_a1b2a3_discouraged=True,
        enable_double_dead_pairs=False,
    ),
    "d1": _rule_cfg(enable_four_run=True),
    "d2": _rule_cfg(enable_two_two_split=True),
    "d3": _rule_cfg(enable_three_plus_one=True),
    "a1b2a3": _rule_cfg(enable_a1b2a3_discouraged=True),
    "pair_triple": _rule_cfg(enable_double_dead_pairs=True),
}
MASKED_RULE_SEQUENCE: Tuple[str, ...] = (
    "d1",
    "d2",
    "d3",
    "a1b2a3",
    "pair_triple",
)


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
    *,
    dead_cell_kwargs: Dict[str, bool],
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
        dead_cells = sorted(find_dead_cells(state.board, **dead_cell_kwargs))
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
    *,
    dead_cell_kwargs: Dict[str, bool],
) -> List[str]:
    lines: List[str] = []
    game_counter = 0
    for file_name, line_no, trmph in _iter_game_entries(input_dir):
        game_counter += 1
        if game_counter > max_games:
            break

        state_suffix, first_move_idx, dead_labels, dead_count, total_moves = _find_first_dead_state(
            trmph,
            dead_cell_kwargs=dead_cell_kwargs,
        )
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


def _count_report_outcomes(lines: List[str]) -> Tuple[int, int]:
    with_dead = sum(1 for line in lines if "first_dead_move=" in line)
    no_dead = sum(1 for line in lines if "NO_DEAD_CELL_BY_END" in line)
    return with_dead, no_dead


def _build_header(
    *,
    input_dir: Path,
    max_games: int,
    rule_name: str,
    dead_cell_kwargs: Dict[str, bool],
) -> List[str]:
    cfg_text = ", ".join(f"{k}={v}" for k, v in sorted(dead_cell_kwargs.items()))
    return [
        f"# input_dir={input_dir}",
        f"# max_games={max_games}",
        f"# rule={rule_name}",
        f"# rule_kwargs={cfg_text}",
        "# format:",
        "# <game_idx> | <source_file:line> | first_dead_move=M/T | dead_cells=N [coord,...] | <trmph_url>",
        "# OR",
        "# <game_idx> | <source_file:line> | NO_DEAD_CELL_BY_END (moves=T) | <trmph_url>",
        "",
    ]


def _write_report(
    *,
    output_file: Path,
    header: List[str],
    lines: List[str],
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(header + lines) + "\n", encoding="utf-8")


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
        help="Path for report output file (single-rule mode)",
    )
    parser.add_argument(
        "--rule",
        type=str,
        default="all_masked",
        choices=sorted(RULE_PRESETS.keys()),
        help="Dead-cell rule preset to evaluate (default: all_masked)",
    )
    parser.add_argument(
        "--run-each-masked-rule",
        action="store_true",
        help=(
            "Generate separate files for each masked motif in sequence: "
            "d1, d2, d3, a1b2a3, pair_triple"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/sf25/mar1"),
        help="Directory for per-rule output files when --run-each-masked-rule is set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_games <= 0:
        raise ValueError(f"--max-games must be positive, got {args.max_games}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
    if args.run_each_masked_rule:
        summary_lines = [
            f"# input_dir={args.input_dir}",
            f"# max_games={args.max_games}",
            "# format: <rule> | with_dead=N | no_dead=M | <output_file>",
            "",
        ]
        for rule_name in MASKED_RULE_SEQUENCE:
            dead_cell_kwargs = RULE_PRESETS[rule_name]
            report_lines = _build_report_lines(
                args.input_dir,
                args.max_games,
                dead_cell_kwargs=dead_cell_kwargs,
            )
            output_file = args.output_dir / (
                f"dead_cell_first_appearance_first{args.max_games}_{rule_name}.txt"
            )
            header = _build_header(
                input_dir=args.input_dir,
                max_games=args.max_games,
                rule_name=rule_name,
                dead_cell_kwargs=dead_cell_kwargs,
            )
            _write_report(output_file=output_file, header=header, lines=report_lines)
            with_dead, no_dead = _count_report_outcomes(report_lines)
            summary_lines.append(
                f"{rule_name} | with_dead={with_dead} | no_dead={no_dead} | {output_file}"
            )
            print(f"Wrote {len(report_lines)} entries to {output_file}")

        summary_file = args.output_dir / (
            f"dead_cell_first_appearance_first{args.max_games}_by_rule_summary.txt"
        )
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
        print(f"Wrote per-rule summary to {summary_file}")
        return

    dead_cell_kwargs = RULE_PRESETS[args.rule]
    report_lines = _build_report_lines(
        args.input_dir,
        args.max_games,
        dead_cell_kwargs=dead_cell_kwargs,
    )
    header = _build_header(
        input_dir=args.input_dir,
        max_games=args.max_games,
        rule_name=args.rule,
        dead_cell_kwargs=dead_cell_kwargs,
    )
    _write_report(output_file=args.output_file, header=header, lines=report_lines)
    print(f"Wrote {len(report_lines)} entries to {args.output_file}")


if __name__ == "__main__":
    main()
