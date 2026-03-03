"""Dead-cell motif detection utilities for Hex.

The functions in this module intentionally implement a small, fast subset of
dead-cell motifs suitable for hard masking in MCTS.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from hex_ai.enums import Piece

BoardLike = Sequence[Sequence[str]] | np.ndarray

# Clockwise ring around (r,c) in row/col coordinates.
# Order: E, SE, SW, W, NW, NE
_RING_OFFSETS: List[Tuple[int, int]] = [
    (0, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
]
_CANONICAL_PAIR_DIRS: Tuple[int, int, int] = (0, 1, 2)  # E, SE, SW

_EMPTY = Piece.EMPTY.value
_RED = Piece.RED.value
_BLUE = Piece.BLUE.value
_OFFBOARD = "#"


_RULE_D1 = "D1"
_RULE_D2 = "D2"
_RULE_D3 = "D3"
_RULE_A1B2A3 = "A1B2A3"
_RULE_PAIR_TRIPLE = "pair_triple"


def _normalize_board(board: BoardLike) -> np.ndarray:
    """Convert board-like input into a validated square ndarray."""
    board_np = np.asarray(board)
    if board_np.ndim != 2:
        raise ValueError(f"Expected board with shape (N, N), got {board_np.shape}")
    rows, cols = int(board_np.shape[0]), int(board_np.shape[1])
    if rows != cols:
        raise ValueError(f"Expected square board, got shape {board_np.shape}")
    if rows <= 0:
        raise ValueError(f"Board size must be positive, got {rows}")
    return board_np


def _in_bounds(n: int, r: int, c: int) -> bool:
    return 0 <= r < n and 0 <= c < n


def _opp(color: str) -> str:
    return _BLUE if color == _RED else _RED


def _ring_tokens(board: np.ndarray, r: int, c: int) -> List[str]:
    """Return 6 ring tokens around (r,c) with an off-board sentinel."""
    n = int(board.shape[0])
    ring: List[str] = []
    for dr, dc in _RING_OFFSETS:
        nr, nc = r + dr, c + dc
        if _in_bounds(n, nr, nc):
            ring.append(str(board[nr, nc]))
        else:
            ring.append(_OFFBOARD)
    return ring


def _max_samecolor_run_cyclic(ring: List[str], color: str) -> int:
    """Longest cyclic run (0-6) of the given color token."""
    doubled = ring + ring
    best = cur = 0
    for token in doubled:
        if token == color:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return min(best, 6)


def _has_two_two_split_with_single_gaps(ring: List[str], color: str) -> bool:
    """Detect canonical cyclic *AA*BB around the focal cell."""
    opp = _opp(color)
    for i in range(6):
        if (
            ring[(i + 1) % 6] == color
            and ring[(i + 2) % 6] == color
            and ring[(i + 4) % 6] == opp
            and ring[(i + 5) % 6] == opp
        ):
            return True
    return False


def _has_three_plus_one_opposite(
    ring: List[str],
    color: str,
) -> bool:
    """Detect the canonical D3 motif `AAA*B*` in cyclic ring order."""
    opp = _opp(color)
    for i in range(6):
        if (
            ring[i] != color
            or ring[(i + 1) % 6] != color
            or ring[(i + 2) % 6] != color
        ):
            continue

        if ring[(i + 4) % 6] == opp:
            return True

    return False


def _has_a1b2a3_gap_pattern(
    ring: List[str],
    color: str,
) -> bool:
    """Detect cyclic A1B2A3 with explicit empty gaps.

    Ring positions must match:
    A, empty, B, empty, A, empty
    where A is `color` and B is the opposite color.
    """
    opp = _opp(color)
    for i in range(6):
        if (
            ring[i] == color
            and ring[(i + 1) % 6] == _EMPTY
            and ring[(i + 2) % 6] == opp
            and ring[(i + 3) % 6] == _EMPTY
            and ring[(i + 4) % 6] == color
            and ring[(i + 5) % 6] == _EMPTY
        ):
            return True
    return False


def _is_dead_cell_single_motifs(
    board: np.ndarray,
    r: int,
    c: int,
    *,
    enable_four_run: bool,
    enable_two_two_split: bool,
    enable_three_plus_one: bool,
    enable_a1b2a3_discouraged: bool,
) -> bool:
    if str(board[r, c]) != _EMPTY:
        return False

    ring = _ring_tokens(board, r, c)

    # D1: >=4 consecutive same-color ring neighbors.
    if enable_four_run:
        if _max_samecolor_run_cyclic(ring, _RED) >= 4:
            return True
        if _max_samecolor_run_cyclic(ring, _BLUE) >= 4:
            return True

    # D2: cyclic *AA*BB (single-cell gap between the color-pairs on both sides).
    if enable_two_two_split and (
        _has_two_two_split_with_single_gaps(ring, _RED)
        or _has_two_two_split_with_single_gaps(ring, _BLUE)
    ):
        return True

    # D3: 3+1 opposite motif.
    if enable_three_plus_one:
        if _has_three_plus_one_opposite(
            ring,
            _RED,
        ):
            return True
        if _has_three_plus_one_opposite(
            ring,
            _BLUE,
        ):
            return True

    # A1B2A3 central taboo motif.
    if enable_a1b2a3_discouraged:
        if _has_a1b2a3_gap_pattern(ring, _RED):
            return True
        if _has_a1b2a3_gap_pattern(ring, _BLUE):
            return True

    return False


def _dead_cell_single_motif_reasons(
    board: np.ndarray,
    r: int,
    c: int,
    *,
    enable_four_run: bool,
    enable_two_two_split: bool,
    enable_three_plus_one: bool,
    enable_a1b2a3_discouraged: bool,
) -> Set[str]:
    """Return rule names matched by enabled single-cell motifs."""
    if str(board[r, c]) != _EMPTY:
        return set()

    ring = _ring_tokens(board, r, c)
    reasons: Set[str] = set()

    if enable_four_run:
        if _max_samecolor_run_cyclic(ring, _RED) >= 4 or _max_samecolor_run_cyclic(ring, _BLUE) >= 4:
            reasons.add(_RULE_D1)

    if enable_two_two_split and (
        _has_two_two_split_with_single_gaps(ring, _RED)
        or _has_two_two_split_with_single_gaps(ring, _BLUE)
    ):
        reasons.add(_RULE_D2)

    if enable_three_plus_one:
        red_d3 = _has_three_plus_one_opposite(
            ring,
            _RED,
        )
        blue_d3 = _has_three_plus_one_opposite(
            ring,
            _BLUE,
        )
        if red_d3 or blue_d3:
            reasons.add(_RULE_D3)

    if enable_a1b2a3_discouraged:
        red_a1b2a3 = _has_a1b2a3_gap_pattern(ring, _RED)
        blue_a1b2a3 = _has_a1b2a3_gap_pattern(ring, _BLUE)
        if red_a1b2a3 or blue_a1b2a3:
            reasons.add(_RULE_A1B2A3)

    return reasons


def _pair_boundary_tokens(
    board: np.ndarray,
    r: int,
    c: int,
    dir_idx: int,
) -> List[str] | None:
    """Return ordered 8-cell boundary tokens around an adjacent pair."""
    n = int(board.shape[0])
    dr, dc = _RING_OFFSETS[dir_idx]
    r2, c2 = r + dr, c + dc
    if not _in_bounds(n, r, c) or not _in_bounds(n, r2, c2):
        return None

    d0 = (dir_idx - 1) % 6
    d1 = dir_idx
    d2 = (dir_idx + 1) % 6
    d3 = (dir_idx + 2) % 6
    d4 = (dir_idx + 3) % 6
    d5 = (dir_idx + 4) % 6

    def _offset_pos(rr: int, cc: int, d: int) -> Tuple[int, int]:
        odr, odc = _RING_OFFSETS[d]
        return rr + odr, cc + odc

    boundary_coords = [
        _offset_pos(r, c, d0),
        _offset_pos(r2, c2, d0),
        _offset_pos(r2, c2, d1),
        _offset_pos(r2, c2, d2),
        _offset_pos(r, c, d2),
        _offset_pos(r, c, d3),
        _offset_pos(r, c, d4),
        _offset_pos(r, c, d5),
    ]
    for rr, cc in boundary_coords:
        if not _in_bounds(n, rr, cc):
            return None
    return [str(board[rr, cc]) for rr, cc in boundary_coords]


def _matches_aaa_star_bbb_star(tokens: List[str], color: str) -> bool:
    """Detect cyclic AAA*BBB* on an 8-token boundary."""
    opp = _opp(color)
    for i in range(8):
        if (
            tokens[i] == color
            and tokens[(i + 1) % 8] == color
            and tokens[(i + 2) % 8] == color
            and tokens[(i + 4) % 8] == opp
            and tokens[(i + 5) % 8] == opp
            and tokens[(i + 6) % 8] == opp
        ):
            return True
    return False


def _is_double_dead_pair_triple_flank_template(
    board: np.ndarray,
    r: int,
    c: int,
    dir_idx: int,
) -> bool:
    """Check AAA*BBB* boundary template around an empty adjacent pair."""
    n = int(board.shape[0])
    dr, dc = _RING_OFFSETS[dir_idx]
    r2, c2 = r + dr, c + dc
    if not _in_bounds(n, r2, c2):
        return False
    if str(board[r, c]) != _EMPTY or str(board[r2, c2]) != _EMPTY:
        return False

    boundary = _pair_boundary_tokens(board, r, c, dir_idx)
    if boundary is None:
        return False
    return _matches_aaa_star_bbb_star(boundary, _RED) or _matches_aaa_star_bbb_star(
        boundary, _BLUE
    )


def _is_double_dead_pair(
    board: np.ndarray,
    r: int,
    c: int,
    dir_idx: int,
) -> bool:
    """Check the triple-flank two-cell dead-pair template around an empty pair."""
    return _is_double_dead_pair_triple_flank_template(board, r, c, dir_idx)


def is_dead_cell(
    board: BoardLike,
    r: int,
    c: int,
    *,
    # Compatibility parameter retained intentionally; no border-fill logic is used.
    red_connects_rows: bool = True,
    enable_four_run: bool = True,
    enable_two_two_split: bool = True,
    enable_three_plus_one: bool = True,
    enable_a1b2a3_discouraged: bool = True,
) -> bool:
    """Return True when an empty cell matches single-cell dead motifs."""
    _ = red_connects_rows
    board_np = _normalize_board(board)
    n = int(board_np.shape[0])
    if not _in_bounds(n, r, c):
        raise ValueError(f"Cell ({r}, {c}) is out of bounds for board size {n}")
    return _is_dead_cell_single_motifs(
        board_np,
        r,
        c,
        enable_four_run=enable_four_run,
        enable_two_two_split=enable_two_two_split,
        enable_three_plus_one=enable_three_plus_one,
        enable_a1b2a3_discouraged=enable_a1b2a3_discouraged,
    )


def _find_double_dead_pairs_on_normalized_board(
    board_np: np.ndarray,
) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    n = int(board_np.shape[0])
    dead_pairs: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
    for r in range(n):
        for c in range(n):
            if str(board_np[r, c]) != _EMPTY:
                continue
            for dir_idx in _CANONICAL_PAIR_DIRS:
                if not _is_double_dead_pair(
                    board_np,
                    r,
                    c,
                    dir_idx,
                ):
                    continue
                dr, dc = _RING_OFFSETS[dir_idx]
                a = (r, c)
                b = (r + dr, c + dc)
                pair = (a, b) if a <= b else (b, a)
                dead_pairs.add(pair)
    return dead_pairs


def find_double_dead_pairs(
    board: BoardLike,
    *,
    red_connects_rows: bool = True,
) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Return adjacent empty pairs matching the double-dead template."""
    _ = red_connects_rows
    board_np = _normalize_board(board)
    return _find_double_dead_pairs_on_normalized_board(board_np)


def find_dead_cells(
    board: BoardLike,
    *,
    # Compatibility parameter retained intentionally; no border-fill logic is used.
    red_connects_rows: bool = True,
    enable_four_run: bool = True,
    enable_two_two_split: bool = True,
    enable_three_plus_one: bool = True,
    enable_a1b2a3_discouraged: bool = True,
    enable_double_dead_pairs: bool = False,
) -> Set[Tuple[int, int]]:
    """Return all empty coordinates currently matched by configured motifs."""
    _ = red_connects_rows
    board_np = _normalize_board(board)
    n = int(board_np.shape[0])
    dead: Set[Tuple[int, int]] = set()

    for r in range(n):
        for c in range(n):
            if _is_dead_cell_single_motifs(
                board_np,
                r,
                c,
                enable_four_run=enable_four_run,
                enable_two_two_split=enable_two_two_split,
                enable_three_plus_one=enable_three_plus_one,
                enable_a1b2a3_discouraged=enable_a1b2a3_discouraged,
            ):
                dead.add((r, c))

    if enable_double_dead_pairs:
        for a, b in _find_double_dead_pairs_on_normalized_board(board_np):
            dead.add(a)
            dead.add(b)

    return dead


def find_dead_cells_with_reasons(
    board: BoardLike,
    *,
    red_connects_rows: bool = True,
    enable_four_run: bool = True,
    enable_two_two_split: bool = True,
    enable_three_plus_one: bool = True,
    enable_a1b2a3_discouraged: bool = True,
    enable_double_dead_pairs: bool = False,
) -> Dict[Tuple[int, int], Set[str]]:
    """Return dead-cell matches annotated with the rule names that triggered each cell."""
    _ = red_connects_rows
    board_np = _normalize_board(board)
    n = int(board_np.shape[0])
    dead_with_reasons: Dict[Tuple[int, int], Set[str]] = {}

    for r in range(n):
        for c in range(n):
            reasons = _dead_cell_single_motif_reasons(
                board_np,
                r,
                c,
                enable_four_run=enable_four_run,
                enable_two_two_split=enable_two_two_split,
                enable_three_plus_one=enable_three_plus_one,
                enable_a1b2a3_discouraged=enable_a1b2a3_discouraged,
            )
            if reasons:
                dead_with_reasons[(r, c)] = reasons

    if enable_double_dead_pairs:
        for a, b in _find_double_dead_pairs_on_normalized_board(board_np):
            dead_with_reasons.setdefault(a, set()).add(_RULE_PAIR_TRIPLE)
            dead_with_reasons.setdefault(b, set()).add(_RULE_PAIR_TRIPLE)

    return dead_with_reasons
