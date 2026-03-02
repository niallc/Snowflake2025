#!/usr/bin/env python3
"""Inspect dead-cell motif matches for a TRMPH position."""

from __future__ import annotations

import argparse
from typing import Dict, Iterable, List, Sequence, Tuple

from hex_ai.inference.game_engine import HexGameState
from hex_ai.utils.format_conversion import rowcol_to_trmph, trmph_move_to_rowcol
import hex_ai.utils.weaks_cells as ws


def _letters(n: int) -> List[str]:
    return [chr(ord("a") + i) for i in range(n)]


def _display_piece(token: str) -> str:
    if token == ws._BLUE:
        return "B"
    if token == ws._RED:
        return "R"
    if token == ws._EMPTY:
        return "."
    return token


def _print_board_matrix(board, board_size: int) -> None:
    letters = _letters(board_size)
    header = "    " + " ".join(f"{c:>2s}" for c in letters)
    print(header)
    for r in range(board_size):
        row_label = f"{r + 1:>2d}"
        cells = " ".join(f"{_display_piece(str(board[r, c])):>2s}" for c in range(board_size))
        print(f"{row_label}  {cells}")


def _parse_pair(pair_arg: str, board_size: int) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    parts = [p.strip() for p in pair_arg.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--pair must be two TRMPH coords like 'k6,k7'")
    a = trmph_move_to_rowcol(parts[0], board_size=board_size)
    b = trmph_move_to_rowcol(parts[1], board_size=board_size)
    return a, b


def _find_dir_idx(a: Tuple[int, int], b: Tuple[int, int]) -> int | None:
    r, c = a
    for i, (dr, dc) in enumerate(ws._RING_OFFSETS):
        if (r + dr, c + dc) == b:
            return i
    return None


def _isolated_rule_kwargs() -> Dict[str, Dict[str, bool]]:
    base = dict(
        enable_four_run=False,
        enable_two_two_split=False,
        enable_three_plus_one=False,
        three_plus_one_requires_adjacent_opposite=False,
        enable_a1b2a3_discouraged=False,
        enable_double_dead_pairs=False,
    )

    def with_overrides(**overrides: bool) -> Dict[str, bool]:
        cfg = dict(base)
        cfg.update(overrides)
        return cfg

    return {
        "all_masked": with_overrides(
            enable_four_run=True,
            enable_two_two_split=True,
            enable_three_plus_one=True,
            enable_a1b2a3_discouraged=True,
            enable_double_dead_pairs=True,
        ),
        "d1": with_overrides(enable_four_run=True),
        "d2": with_overrides(enable_two_two_split=True),
        "d3": with_overrides(enable_three_plus_one=True),
        "a1b2a3": with_overrides(enable_a1b2a3_discouraged=True),
        "pair_triple": with_overrides(enable_double_dead_pairs=True),
    }


def _coord_in_dead(board, coord: Tuple[int, int], kwargs: Dict[str, bool]) -> bool:
    dead = ws.find_dead_cells(board, **kwargs)
    return coord in dead


def _print_cell_details(board, board_size: int, cell: Tuple[int, int]) -> None:
    r, c = cell
    print("")
    print(f"Cell: {rowcol_to_trmph(r, c, board_size)} at ({r},{c}) token={board[r, c]}")
    ring_tokens = ws._ring_tokens(board, r, c)
    ring_labels: List[str] = []
    for dr, dc in ws._RING_OFFSETS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < board_size and 0 <= nc < board_size:
            ring_labels.append(rowcol_to_trmph(nr, nc, board_size))
        else:
            ring_labels.append("OFF")
    print("Ring (E,SE,SW,W,NW,NE):")
    for label, token in zip(ring_labels, ring_tokens):
        print(f"  {label:>4s}: {_display_piece(str(token))} ({token})")

    rules = _isolated_rule_kwargs()
    print("Rule matches:")
    for name, kwargs in rules.items():
        print(f"  {name:>12s}: {_coord_in_dead(board, cell, kwargs)}")


def _print_pair_details(board, board_size: int, pair: Tuple[Tuple[int, int], Tuple[int, int]]) -> None:
    a, b = pair
    dir_idx = _find_dir_idx(a, b)
    if dir_idx is None:
        dir_idx = _find_dir_idx(b, a)
        if dir_idx is None:
            raise ValueError("Pair cells are not adjacent on hex grid")
        a, b = b, a
    ar, ac = a

    print("")
    print(
        f"Pair: {rowcol_to_trmph(a[0], a[1], board_size)},"
        f"{rowcol_to_trmph(b[0], b[1], board_size)} dir_idx={dir_idx}"
    )
    print(f"  tokens: {board[a]} {board[b]}")
    triple = ws._is_double_dead_pair_triple_flank_template(board, ar, ac, dir_idx)
    print(f"  triple_flank_template: {triple}")

    boundary = ws._pair_boundary_tokens(board, ar, ac, dir_idx)
    if boundary is None:
        print("  boundary: OFFBOARD")
        return

    d0 = (dir_idx - 1) % 6
    d1 = dir_idx
    d2 = (dir_idx + 1) % 6
    d3 = (dir_idx + 2) % 6
    d4 = (dir_idx + 3) % 6
    d5 = (dir_idx + 4) % 6

    def off(rr: int, cc: int, d: int) -> Tuple[int, int]:
        odr, odc = ws._RING_OFFSETS[d]
        return rr + odr, cc + odc

    coords = [
        off(a[0], a[1], d0),
        off(b[0], b[1], d0),
        off(b[0], b[1], d1),
        off(b[0], b[1], d2),
        off(a[0], a[1], d2),
        off(a[0], a[1], d3),
        off(a[0], a[1], d4),
        off(a[0], a[1], d5),
    ]
    labels = [rowcol_to_trmph(r, c, board_size) for r, c in coords]
    print("  boundary (ordered 8 around pair):")
    for label, tok in zip(labels, boundary):
        print(f"    {label:>4s}: {_display_piece(str(tok))} ({tok})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect dead-cell motif matches for one position.")
    parser.add_argument(
        "--trmph",
        type=str,
        required=True,
        help="TRMPH state string, e.g. '#13,c3e3c4e4'",
    )
    parser.add_argument(
        "--cell",
        type=str,
        help="Optional single cell to inspect, e.g. 'c4'",
    )
    parser.add_argument(
        "--pair",
        type=str,
        help="Optional adjacent pair to inspect, e.g. 'k6,k7'",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    state = HexGameState.from_trmph(args.trmph)
    board = ws._normalize_board(state.board)
    board_size = int(board.shape[0])

    print("Board:")
    _print_board_matrix(board, board_size)

    if args.cell:
        cell = trmph_move_to_rowcol(args.cell, board_size=board_size)
        _print_cell_details(board, board_size, cell)

    if args.pair:
        pair = _parse_pair(args.pair, board_size)
        _print_pair_details(board, board_size, pair)


if __name__ == "__main__":
    main()
