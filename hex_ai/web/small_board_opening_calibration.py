"""Score remapping helpers for virtual small-board opening positions."""

from __future__ import annotations

from typing import Mapping

import hex_ai.utils.format_conversion as fc


SMALL_BOARD_OPENING_TARGETS = {
    "a1": 0.10,
    "a2": 0.50,
    "center": 0.90,
}
_STRICT_X_EPSILON = 1e-6


def should_apply_small_board_opening_remap(
    *,
    display_board_size: int,
    network_board_size: int,
) -> bool:
    """Return True when calibration should run for this board size."""
    return int(display_board_size) < int(network_board_size)


def get_small_board_opening_anchor_moves(display_board_size: int) -> dict[str, str]:
    """
    Return opening calibration anchor moves for a display board.

    Anchors:
    - `a1` corner (target 10%)
    - `a2` near-edge (target 50%)
    - `center` main-center approximation (target 90%)
    """
    size = int(display_board_size)
    if size < 2:
        raise ValueError("display_board_size must be >= 2 for opening calibration")

    # For even sizes (e.g. 2x2), pick upper-right of the central 2x2 block so
    # "center" is always distinct from "a1".
    center_idx = size // 2
    center_move = fc.rowcol_to_trmph(center_idx, center_idx, board_size=size)
    anchors = {
        "a1": "a1",
        "a2": "a2",
        "center": center_move,
    }
    if len(set(anchors.values())) != len(anchors):
        raise ValueError(f"Non-distinct opening anchors for display_board_size={size}: {anchors}")
    return anchors


def _clamp_0_1(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _piecewise_linear_map(
    value: float,
    *,
    xs: list[float],
    ys: list[float],
) -> float:
    """Evaluate piecewise-linear interpolation/extrapolation for three anchor points."""
    x = float(value)

    if x <= xs[0]:
        left = 0
    elif x >= xs[-1]:
        left = len(xs) - 2
    else:
        left = 0
        for idx in range(len(xs) - 1):
            if xs[idx] <= x <= xs[idx + 1]:
                left = idx
                break

    x0 = xs[left]
    x1 = xs[left + 1]
    y0 = ys[left]
    y1 = ys[left + 1]

    if x1 <= x0:
        return _clamp_0_1(y0)
    t = (x - x0) / (x1 - x0)
    return _clamp_0_1(y0 + t * (y1 - y0))


def remap_small_board_opening_scores(
    scores: Mapping[str, float],
    *,
    display_board_size: int,
    network_board_size: int,
) -> tuple[dict[str, float], dict[str, object]]:
    """
    Remap opening scores so anchor moves hit stable target win rates.

    Uses a piecewise-linear map that satisfies:
    - a1 -> 0.10
    - a2 -> 0.50
    - center -> 0.90
    """
    normalized_scores = {str(move): float(score) for move, score in scores.items()}

    if not should_apply_small_board_opening_remap(
        display_board_size=display_board_size,
        network_board_size=network_board_size,
    ):
        return normalized_scores, {"applied": False, "reason": "not_small_board"}

    try:
        anchor_moves = get_small_board_opening_anchor_moves(display_board_size)
    except ValueError as exc:
        return normalized_scores, {
            "applied": False,
            "reason": "invalid_anchor_configuration",
            "error": str(exc),
        }
    missing = [name for name, move in anchor_moves.items() if move not in normalized_scores]
    if missing:
        return normalized_scores, {
            "applied": False,
            "reason": "missing_anchor_moves",
            "missing_anchor_names": missing,
            "anchor_moves": anchor_moves,
        }

    anchor_points = [
        (normalized_scores[anchor_moves["a1"]], SMALL_BOARD_OPENING_TARGETS["a1"]),
        (normalized_scores[anchor_moves["a2"]], SMALL_BOARD_OPENING_TARGETS["a2"]),
        (normalized_scores[anchor_moves["center"]], SMALL_BOARD_OPENING_TARGETS["center"]),
    ]
    anchor_points.sort(key=lambda item: item[0])

    xs = [float(point[0]) for point in anchor_points]
    ys = [float(point[1]) for point in anchor_points]

    for idx in range(1, len(xs)):
        if xs[idx] <= xs[idx - 1]:
            xs[idx] = xs[idx - 1] + _STRICT_X_EPSILON

    remapped = {
        move: _piecewise_linear_map(score, xs=xs, ys=ys)
        for move, score in normalized_scores.items()
    }

    return remapped, {
        "applied": True,
        "reason": "ok",
        "anchor_moves": anchor_moves,
        "anchor_targets": dict(SMALL_BOARD_OPENING_TARGETS),
        "anchor_raw_scores": {
            name: normalized_scores[move] for name, move in anchor_moves.items()
        },
    }
