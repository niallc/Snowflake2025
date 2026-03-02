"""Dead-cell motif detection utilities for Hex.

The functions in this module intentionally implement a small, fast subset of
dead-cell motifs suitable for hard masking in MCTS.
"""

from __future__ import annotations

from typing import List, Sequence, Set, Tuple

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
_DOUBLE_PAIR_SUPPORT_OFFSETS = {
    # Pair axis E/W -> supports on SW/NE
    0: ((1, -1), (-1, 1)),
    # Pair axis SE/NW -> supports on E/W
    1: ((0, 1), (0, -1)),
    # Pair axis SW/NE -> supports on SE/NW
    2: ((1, 0), (-1, 0)),
}

_EMPTY = Piece.EMPTY.value
_RED = Piece.RED.value
_BLUE = Piece.BLUE.value
_OFFBOARD = "#"


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


def _has_adjacent_pair(ring: List[str], color: str) -> bool:
    for i in range(6):
        if ring[i] == color and ring[(i + 1) % 6] == color:
            return True
    return False


def _has_three_plus_one_opposite(
    ring: List[str],
    color: str,
    *,
    require_adjacent_opposite: bool,
) -> bool:
    """Detect a 3+1 motif around one color.

    With `require_adjacent_opposite=False`, detect the D3 pattern:
    `AAAXBX` (cyclic), where:
    - `A` is `color`
    - `B` is the opposite color
    - `X` is empty

    With `require_adjacent_opposite=True`, use a stricter adjacent-opposite
    variant where the opposite color is directly next to the `AAA` run and the
    remaining two ring cells are empty:
    - `AAABXX` or `AAAXXB` (cyclic)
    """
    opp = _opp(color)
    for i in range(6):
        if (
            ring[i] != color
            or ring[(i + 1) % 6] != color
            or ring[(i + 2) % 6] != color
        ):
            continue

        if require_adjacent_opposite:
            if (
                ring[(i + 3) % 6] == opp
                and ring[(i + 4) % 6] == _EMPTY
                and ring[(i + 5) % 6] == _EMPTY
            ):
                return True
            if (
                ring[(i + 3) % 6] == _EMPTY
                and ring[(i + 4) % 6] == _EMPTY
                and ring[(i + 5) % 6] == opp
            ):
                return True
            continue

        if (
            ring[(i + 3) % 6] == _EMPTY
            and ring[(i + 4) % 6] == opp
            and ring[(i + 5) % 6] == _EMPTY
        ):
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
    three_plus_one_requires_adjacent_opposite: bool,
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

    # D2: adjacent pair of red and adjacent pair of blue.
    if enable_two_two_split and _has_adjacent_pair(ring, _RED) and _has_adjacent_pair(ring, _BLUE):
        return True

    # D3: 3+1 opposite motif.
    if enable_three_plus_one:
        if _has_three_plus_one_opposite(
            ring,
            _RED,
            require_adjacent_opposite=three_plus_one_requires_adjacent_opposite,
        ):
            return True
        if _has_three_plus_one_opposite(
            ring,
            _BLUE,
            require_adjacent_opposite=three_plus_one_requires_adjacent_opposite,
        ):
            return True

    # A1B2A3 central taboo motif.
    if enable_a1b2a3_discouraged:
        if _has_a1b2a3_gap_pattern(ring, _RED):
            return True
        if _has_a1b2a3_gap_pattern(ring, _BLUE):
            return True

    return False


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


def _is_double_dead_pair_support_template(
    board: np.ndarray,
    r: int,
    c: int,
    dir_idx: int,
) -> bool:
    """Check 2+2 support template around an empty adjacent pair."""
    n = int(board.shape[0])
    dr, dc = _RING_OFFSETS[dir_idx]
    r2, c2 = r + dr, c + dc
    if not _in_bounds(n, r2, c2):
        return False
    if str(board[r, c]) != _EMPTY or str(board[r2, c2]) != _EMPTY:
        return False

    (s1r, s1c), (s2r, s2c) = _DOUBLE_PAIR_SUPPORT_OFFSETS[dir_idx]
    p1 = (r + s1r, c + s1c)
    q1 = (r2 + s1r, c2 + s1c)
    p2 = (r + s2r, c + s2c)
    q2 = (r2 + s2r, c2 + s2c)
    for rr, cc in (p1, q1, p2, q2):
        if not _in_bounds(n, rr, cc):
            return False

    c11 = str(board[p1])
    c12 = str(board[q1])
    c21 = str(board[p2])
    c22 = str(board[q2])
    if c11 == _EMPTY or c21 == _EMPTY:
        return False
    if c11 != c12 or c21 != c22:
        return False
    if c11 == c21:
        return False
    if c11 not in (_RED, _BLUE) or c21 not in (_RED, _BLUE):
        return False
    return True


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
    *,
    enable_support_template: bool,
    enable_triple_flank_template: bool,
) -> bool:
    """Check supported two-cell dead-pair templates around an empty pair."""
    if enable_support_template and _is_double_dead_pair_support_template(
        board, r, c, dir_idx
    ):
        return True
    if enable_triple_flank_template and _is_double_dead_pair_triple_flank_template(
        board, r, c, dir_idx
    ):
        return True
    return False


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
    three_plus_one_requires_adjacent_opposite: bool = False,
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
        three_plus_one_requires_adjacent_opposite=three_plus_one_requires_adjacent_opposite,
        enable_a1b2a3_discouraged=enable_a1b2a3_discouraged,
    )


def _find_double_dead_pairs_on_normalized_board(
    board_np: np.ndarray,
    *,
    enable_support_template: bool = True,
    enable_triple_flank_template: bool = True,
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
                    enable_support_template=enable_support_template,
                    enable_triple_flank_template=enable_triple_flank_template,
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
    enable_support_template: bool = True,
    enable_triple_flank_template: bool = True,
) -> Set[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Return adjacent empty pairs matching the double-dead template."""
    _ = red_connects_rows
    board_np = _normalize_board(board)
    return _find_double_dead_pairs_on_normalized_board(
        board_np,
        enable_support_template=enable_support_template,
        enable_triple_flank_template=enable_triple_flank_template,
    )


def find_dead_cells(
    board: BoardLike,
    *,
    # Compatibility parameter retained intentionally; no border-fill logic is used.
    red_connects_rows: bool = True,
    enable_four_run: bool = True,
    enable_two_two_split: bool = True,
    enable_three_plus_one: bool = True,
    three_plus_one_requires_adjacent_opposite: bool = False,
    enable_a1b2a3_discouraged: bool = True,
    enable_double_dead_pairs: bool = True,
    enable_double_dead_pair_support_template: bool = True,
    enable_double_dead_pair_triple_flank_template: bool = True,
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
                three_plus_one_requires_adjacent_opposite=three_plus_one_requires_adjacent_opposite,
                enable_a1b2a3_discouraged=enable_a1b2a3_discouraged,
            ):
                dead.add((r, c))

    if enable_double_dead_pairs:
        for a, b in _find_double_dead_pairs_on_normalized_board(
            board_np,
            enable_support_template=enable_double_dead_pair_support_template,
            enable_triple_flank_template=enable_double_dead_pair_triple_flank_template,
        ):
            dead.add(a)
            dead.add(b)

    return dead
