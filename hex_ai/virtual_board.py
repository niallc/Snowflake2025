"""Shared virtual-board prefill definitions and helpers."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

from hex_ai.config import BOARD_SIZE
from hex_ai.utils import format_conversion as fc


# Derived from legacy_code/FileConversion.py and existing rules.html guidance.
# Values are bare TRMPH move strings (no "#13," prefix).
VIRTUAL_BOARD_PREFILL_MOVES = {
    13: "",
    12: "a13m1b13m2c13m3d13m4e13m5f13m6g13m7h13m8i13m9j13m10k13m11l13m12",
    11: "a12l1b12l2c12l3d12l4e12l5f12l6g12l7h12l8i12l9j12l10k12l11k13m11",
    10: "a11k1b11k2c11k3d11k4e11k5f11k6g11k7h11k8i11k9j11k10j12l10j13m10",
    9: "a10j1b10j2c10j3d10j4e10j5f10j6g10j7h10j8i10j9i11k9i12l9i13m9",
    8: "a9i1b9i2c9i3d9i4e9i5f9i6g9i7h9i8h10j8h11k8h12l8h13m8",
    7: "a8h1b8h2c8h3d8h4e8h5f8h6g8h7g9i7g10j7g11k7g12l7g13m7",
    6: "a7g1b7g2c7g3d7g4e7g5f7g6f8h6f9i6f10j6f11k6f12l6f13m6",
    5: "a6f1b6f2c6f3d6f4e6f5e7g5e8h5e9i5e10j5e11k5e12l5e13m5",
    4: "a5e1b5e2c5e3d5e4d6f4d7g4d8h4d9i4d10j4d11k4d12l4d13m4",
    3: "a4d1b4d2c4d3c5e3c6f3c7g3c8h3c9i3c10j3c11k3c12l3c13m3",
    2: "a3c1b3c2b4d2b5e2b6f2b7g2b8h2b9i2b10j2b11k2b12l2b13m2",
}

MIN_VIRTUAL_DISPLAY_BOARD_SIZE = min(VIRTUAL_BOARD_PREFILL_MOVES.keys())
MAX_VIRTUAL_DISPLAY_BOARD_SIZE = max(VIRTUAL_BOARD_PREFILL_MOVES.keys())


def get_virtual_prefill_moves(display_board_size: int) -> str:
    """Get the prefill move sequence for a virtual KxK display board."""
    try:
        return VIRTUAL_BOARD_PREFILL_MOVES[int(display_board_size)]
    except (TypeError, ValueError) as exc:
        raise ValueError("display_board_size must be an integer") from exc
    except KeyError as exc:
        raise ValueError(f"Unsupported display_board_size: {display_board_size}") from exc


@lru_cache(maxsize=None)
def get_virtual_prefill_move_coords(
    display_board_size: int,
    *,
    network_board_size: int = BOARD_SIZE,
) -> Tuple[Tuple[int, int], ...]:
    """Return prefill coordinates for a display board size on the network board."""
    prefill = get_virtual_prefill_moves(display_board_size)
    if not prefill:
        return tuple()
    coords = tuple(
        fc.trmph_move_to_rowcol(move, board_size=network_board_size)
        for move in fc.split_trmph_moves(prefill)
    )
    return coords


@lru_cache(maxsize=None)
def get_virtual_prefill_move_count(display_board_size: int) -> int:
    """Return number of prefill moves for a virtual display board size."""
    return len(get_virtual_prefill_move_coords(display_board_size))


def detect_virtual_prefill_display_board_size(
    bare_moves: str,
    *,
    min_display_board_size: int = MIN_VIRTUAL_DISPLAY_BOARD_SIZE,
    max_display_board_size: int = BOARD_SIZE - 1,
) -> Optional[int]:
    """
    Detect whether a game starts with one of the configured virtual prefills.

    Returns:
        Matching display board size when a known prefill prefix is present, else ``None``.
    """
    if not isinstance(bare_moves, str):
        raise TypeError(f"bare_moves must be str, got {type(bare_moves)}")

    min_size = int(min_display_board_size)
    max_size = int(max_display_board_size)
    if min_size > max_size:
        raise ValueError(
            "min_display_board_size must be <= max_display_board_size, "
            f"got {min_size} > {max_size}"
        )

    for size in range(max_size, min_size - 1, -1):
        prefill = VIRTUAL_BOARD_PREFILL_MOVES.get(size)
        if prefill and bare_moves.startswith(prefill):
            return size
    return None


def get_virtual_prefill_prefix_move_count_for_trmph(
    trmph_text: str,
    *,
    min_display_board_size: int = MIN_VIRTUAL_DISPLAY_BOARD_SIZE,
    max_display_board_size: int = BOARD_SIZE - 1,
) -> int:
    """Return how many leading moves in a TRMPH game are known virtual-prefill moves."""
    bare_moves = fc.strip_trmph_preamble((trmph_text or "").strip())
    display_size = detect_virtual_prefill_display_board_size(
        bare_moves,
        min_display_board_size=min_display_board_size,
        max_display_board_size=max_display_board_size,
    )
    if display_size is None:
        return 0
    return get_virtual_prefill_move_count(display_size)
