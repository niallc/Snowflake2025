#!/usr/bin/env python3
"""Preview a ladder-template annotation as a concrete board embedding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hex_ai.config import BOARD_SIZE
from hex_ai.inference.board_display import display_hex_board
from hex_ai.ladder_templates import (
    load_ladder_template_annotation,
    materialize_ladder_template,
)
from hex_ai.utils.format_conversion import rowcol_to_trmph


def _format_coords(coords: tuple[tuple[int, int], ...], board_size: int) -> list[str]:
    labels: list[str] = []
    for row, col in coords:
        labels.append(rowcol_to_trmph(row, col, board_size=board_size))
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load a ladder-template annotation and preview it on a concrete board."
    )
    parser.add_argument("template_json", type=Path, help="Path to annotation JSON exported by the annotator.")
    parser.add_argument(
        "--board-size",
        type=int,
        default=BOARD_SIZE,
        help=f"Concrete board size for the embedding (default: {BOARD_SIZE}).",
    )
    parser.add_argument("--anchor-row", type=int, default=0, help="Top-left embedding row (default: 0).")
    parser.add_argument("--anchor-col", type=int, default=0, help="Top-left embedding col (default: 0).")
    parser.add_argument(
        "--summary-json",
        action="store_true",
        help="Print a JSON summary after the ASCII board preview.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotation = load_ladder_template_annotation(args.template_json)
    materialized = materialize_ladder_template(
        annotation,
        board_size=args.board_size,
        anchor_row=args.anchor_row,
        anchor_col=args.anchor_col,
    )

    print(f"Template: {annotation.metadata.name or args.template_json.stem}")
    print(f"Family: {annotation.metadata.family or '-'}")
    print(f"Attacker: {annotation.metadata.attacker}")
    print(f"Target edge: {annotation.metadata.target_edge}")
    print(f"Open left/right: {annotation.metadata.open_left}/{annotation.metadata.open_right}")
    print(f"Board size: {materialized.board_size}")
    print(f"Anchor: ({materialized.anchor_row}, {materialized.anchor_col})")
    print("")
    display_hex_board(materialized.board)
    print("")
    print(f"Red stones: {_format_coords(materialized.red_stones, materialized.board_size)}")
    print(f"Blue stones: {_format_coords(materialized.blue_stones, materialized.board_size)}")
    print(f"Left boundary: {_format_coords(materialized.left_boundary, materialized.board_size)}")
    print(f"Right boundary: {_format_coords(materialized.right_boundary, materialized.board_size)}")
    print(f"Shaded cells: {_format_coords(materialized.shaded_cells, materialized.board_size)}")

    if args.summary_json:
        print("")
        print(json.dumps(materialized.summary_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
