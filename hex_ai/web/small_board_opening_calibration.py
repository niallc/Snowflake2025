"""Score remapping helpers for virtual small-board opening positions."""

from __future__ import annotations

from typing import Mapping

import hex_ai.utils.format_conversion as fc
import numpy as np


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


def _score_distribution_stats(scores: Mapping[str, float]) -> dict[str, float | int | None]:
    values = [float(score) for score in scores.values()]
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "median": None,
        }
    return {
        "count": len(values),
        "min": float(min(values)),
        "max": float(max(values)),
        "median": float(np.median(values)),
    }


def _round_sig(value: float | None, sig_figs: int = 3) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric) or numeric == 0.0:
        return numeric
    return float(f"{numeric:.{sig_figs}g}")


def _compact_rounded_stats(stats: dict[str, float | int | None]) -> dict[str, float | int | None]:
    return {
        "n": int(stats["count"]),
        "min": _round_sig(stats["min"]),
        "med": _round_sig(stats["median"]),
        "max": _round_sig(stats["max"]),
    }


def build_small_board_opening_diagnostics(
    raw_scores: Mapping[str, float],
    remapped_scores: Mapping[str, float],
    *,
    display_board_size: int,
    network_board_size: int,
) -> dict[str, object]:
    """
    Build concise diagnostics payload for opening-score remap debugging.

    Includes:
    - min/max/median before and after remap
    - key watch moves (a1, a2, center, (k,k), (k,k-1))
    """
    anchor_moves = get_small_board_opening_anchor_moves(display_board_size)
    k = int(display_board_size)
    kk = fc.rowcol_to_trmph(k - 1, k - 1, board_size=k)
    kk_minus_1 = fc.rowcol_to_trmph(k - 1, max(0, k - 2), board_size=k)

    watch_moves = {
        "a1": "a1",
        "a2": "a2",
        "center": anchor_moves["center"],
        "k_k": kk,
        "k_k_minus_1": kk_minus_1,
    }

    watch_scores = {
        label: {
            "move": move,
            "raw": (
                float(raw_scores[move])
                if move in raw_scores
                else None
            ),
            "remapped": (
                float(remapped_scores[move])
                if move in remapped_scores
                else None
            ),
        }
        for label, move in watch_moves.items()
    }

    compact_watch_scores = {}
    for label, score_payload in watch_scores.items():
        move = score_payload["move"]
        compact_label = label if label in {"a1", "a2"} else f"{label}:{move}"
        compact_watch_scores[compact_label] = {
            "raw": _round_sig(score_payload["raw"]),
            "norm": _round_sig(score_payload["remapped"]),
        }

    return {
        "board": (
            f"{int(display_board_size)}x{int(display_board_size)}"
            f"->{int(network_board_size)}x{int(network_board_size)}"
        ),
        "before": _compact_rounded_stats(_score_distribution_stats(raw_scores)),
        "after": _compact_rounded_stats(_score_distribution_stats(remapped_scores)),
        "watch": compact_watch_scores,
    }


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
