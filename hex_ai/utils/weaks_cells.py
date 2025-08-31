# dead_cells.py
from typing import List, Tuple, Set
from hex_ai.enums import Piece

# How to use this code:
# 
# dead = find_dead_cells(board, red_connects_rows=True)
# # mask at the root:
# legal = [(r,c) for (r,c) in legal if (r,c) not in dead]
# # and zero/renormalize policy targets during training when you hard-mask.

# Note, currently the board is List[List[Piece]] with "e","b","r"
# Migrate to the data structures used everywhere else in the codebase.
# Use hex_ai/utils/format_conversion.py to see what we use now.

Piece = str  # "e", "b", "r"

# Clockwise ring around (r,c) in axial-like offsets for your (row,col) grid.
# Order: E, SE, SW, W, NW, NE
_RING_OFFSETS: List[Tuple[int, int]] = [
    (0, +1), (+1, 0), (+1, -1), (0, -1), (-1, 0), (-1, +1)
]

def _opp(color: str) -> str:
    return "b" if color == "r" else "r"

def _in_bounds(n: int, r: int, c: int) -> bool:
    return 0 <= r < n and 0 <= c < n

def _border_fill_color(
    n: int, r: int, c: int, red_connects_rows: bool
) -> str:
    """
    For an off-board neighbor at (r,c), return the virtual border color:
    - If it's off by row (top/bottom), that's the color connecting rows.
    - If it's off by col (left/right), that's the other color.
    Immediate neighbors are never off in both row and col simultaneously.
    """
    if r < 0 or r >= n:
        return "r" if red_connects_rows else "b"
    # otherwise it's a column overflow
    return "b" if red_connects_rows else "r"

def _ring_tokens(
    board: List[List[Piece]], r: int, c: int, red_connects_rows: bool
) -> List[str]:
    """Return 6 tokens for the ring around (r,c): 'r','b','e' (with border fill)."""
    n = len(board)
    ring: List[str] = []
    for dr, dc in _RING_OFFSETS:
        nr, nc = r + dr, c + dc
        if _in_bounds(n, nr, nc):
            ring.append(board[nr][nc])
        else:
            ring.append(_border_fill_color(n, nr, nc, red_connects_rows))
    return ring

def _max_samecolor_run_cyclic(ring: List[str], color: str) -> int:
    """Longest cyclic run (length 0–6) of the given non-empty color."""
    assert color in ("r", "b")
    # duplicate to allow wraparound
    doubled = ring + ring
    best = cur = 0
    for x in doubled:
        if x == color:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    # cap at 6 because the ring is length 6
    return min(best, 6)

def _has_adjacent_pair(ring: List[str], color: str) -> bool:
    for i in range(6):
        if ring[i] == color and ring[(i + 1) % 6] == color:
            return True
    return False

def _has_three_plus_one_opposite(ring: List[str], color: str) -> bool:
    """Exists ... color,color,color ... with an adjacent opposite on either end."""
    opp = _opp(color)
    for i in range(6):
        if (
            ring[i] == color
            and ring[(i + 1) % 6] == color
            and ring[(i + 2) % 6] == color
            and (
                ring[(i - 1) % 6] == opp or ring[(i + 3) % 6] == opp
            )
        ):
            return True
    return False

def is_dead_cell(
    board: List[List[Piece]],
    r: int,
    c: int,
    *,
    red_connects_rows: bool = True,
    enable_two_two_split: bool = True,
    enable_three_plus_one: bool = True,
) -> bool:
    """
    Return True if empty (r,c) is dead under the minimal single-cell rules:
      (D1) ≥4 consecutive same-color in the 6-ring (with border fill)
      (D2) Adjacent 'rr' somewhere AND adjacent 'bb' somewhere (2+2 split)
      (D3) A 'ccc' run with adjacent '¬c' (3+1 opposite)
    These are safe to hard-mask. Edges/corners are handled via border fill.
    """
    if board[r][c] != "e":
        return False

    ring = _ring_tokens(board, r, c, red_connects_rows)

    # D1: four-in-a-row (covers interior and edge/corner after border fill)
    if _max_samecolor_run_cyclic(ring, "r") >= 4:
        return True
    if _max_samecolor_run_cyclic(ring, "b") >= 4:
        return True

    # D2: 2+2 split
    if enable_two_two_split:
        if _has_adjacent_pair(ring, "r") and _has_adjacent_pair(ring, "b"):
            return True

    # D3: 3+1 opposite
    if enable_three_plus_one:
        if _has_three_plus_one_opposite(ring, "r") or _has_three_plus_one_opposite(ring, "b"):
            return True

    return False

def find_dead_cells(
    board: List[List[Piece]],
    *,
    red_connects_rows: bool = True,
    enable_two_two_split: bool = True,
    enable_three_plus_one: bool = True,
) -> Set[Tuple[int, int]]:
    """Return a set of (row, col) for all empty cells that are dead."""
    n = len(board)
    dead: Set[Tuple[int, int]] = set()
    for r in range(n):
        for c in range(n):
            if board[r][c] != "e":
                continue
            if is_dead_cell(
                board, r, c,
                red_connects_rows=red_connects_rows,
                enable_two_two_split=enable_two_two_split,
                enable_three_plus_one=enable_three_plus_one,
            ):
                dead.add((r, c))
    return dead

# --- Optional hook for the double-cell pattern (two adjacent stars) ---
# def find_double_dead_pairs(board: List[List[Piece]], red_connects_rows: bool = True) -> Set[Tuple[Tuple[int,int], Tuple[int,int]]]:
#     """
#     TODO: detect the 'two adjacent empties with 3-of-red vs 3-of-blue on opposite ends'
#     and mark both cells as dead. This needs a careful fan decomposition around the pair.
#     """
#     ...
