"""Incremental ladder-certificate labeling over ordered move sequences."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import numpy as np

from hex_ai.data_utils import remove_repeated_moves
from hex_ai.enums import Piece
from hex_ai.utils.format_conversion import (
    split_trmph_moves,
    strip_trmph_preamble,
    trmph_move_to_rowcol,
)

from .labeler import (
    build_ladder_certificate_labels,
    ladder_certificate_plane_names,
)
from .library_loader import LoadedLadderTemplate
from .matcher import BoardCoord, LadderTemplateMatch, find_ladder_template_matches

_TRMPH_PREAMBLE_RE = re.compile(r"#(\d+),")

MatchKey = tuple[str, str, str, BoardCoord]


@dataclass(frozen=True)
class LadderSequencePositionStats:
    """Summary stats for one labeled position in a move sequence."""

    position_in_game: int
    last_move: BoardCoord | None
    used_must_include_filter: bool
    embeddings_considered: int
    scan_matches_found: int
    active_matches_after_update: int
    elapsed_ms: float


@dataclass(frozen=True)
class LadderGameLabelSequence:
    """Dense ladder labels for every position in one ordered game."""

    plane_names: tuple[str, ...]
    template_origin_maps: np.ndarray
    carrier_maps: np.ndarray
    position_stats: tuple[LadderSequencePositionStats, ...]


def build_ladder_certificate_label_sequence_for_trmph_game(
    trmph_text: str,
    templates: tuple[LoadedLadderTemplate, ...] | list[LoadedLadderTemplate],
    *,
    board_size: int | None = None,
    orientations: tuple[str, ...] = ("red_bottom", "blue_right"),
    allow_attacker_superset_on_empty: bool = True,
) -> LadderGameLabelSequence:
    """
    Build exact full-position ladder labels for one TRMPH game.

    This uses an incremental exact update after the opening full scan:
    only matches whose carrier contains the latest move are reconsidered.
    Under the current phase-1 matcher semantics, that is exact because only
    carrier cells contribute board-occupancy constraints.
    """
    clean_moves, resolved_board_size = _resolve_clean_game_moves(
        trmph_text,
        board_size=board_size,
    )

    piece_board = np.full(
        (resolved_board_size, resolved_board_size),
        Piece.EMPTY.value,
        dtype="U1",
    )
    position_count = len(clean_moves) + 1
    plane_names = ladder_certificate_plane_names(suffix="template_origin")
    plane_count = len(plane_names)

    template_origin_maps = np.zeros(
        (position_count, plane_count, resolved_board_size, resolved_board_size),
        dtype=np.uint8,
    )
    carrier_maps = np.zeros_like(template_origin_maps)
    position_stats: list[LadderSequencePositionStats] = []

    active_matches_by_key: dict[MatchKey, LadderTemplateMatch] = {}
    active_match_keys_by_cell: dict[BoardCoord, set[MatchKey]] = {}

    initial_result = find_ladder_template_matches(
        piece_board,
        templates,
        orientations=orientations,
        allow_attacker_superset_on_empty=allow_attacker_superset_on_empty,
        must_include_cell=None,
    )
    for match in initial_result.matches:
        _add_active_match(
            active_matches_by_key=active_matches_by_key,
            active_match_keys_by_cell=active_match_keys_by_cell,
            match=match,
        )
    _store_position_labels(
        template_origin_maps=template_origin_maps,
        carrier_maps=carrier_maps,
        position_index=0,
        board_size=resolved_board_size,
        active_matches=tuple(active_matches_by_key.values()),
    )
    position_stats.append(
        LadderSequencePositionStats(
            position_in_game=0,
            last_move=None,
            used_must_include_filter=False,
            embeddings_considered=initial_result.stats.embeddings_considered,
            scan_matches_found=initial_result.stats.matches_found,
            active_matches_after_update=len(active_matches_by_key),
            elapsed_ms=initial_result.stats.elapsed_ms,
        )
    )

    for move_index, move in enumerate(clean_moves):
        row, col = trmph_move_to_rowcol(move, board_size=resolved_board_size)
        moved_cell = (row, col)
        piece_board[moved_cell] = (
            Piece.BLUE.value if move_index % 2 == 0 else Piece.RED.value
        )

        filtered_result = find_ladder_template_matches(
            piece_board,
            templates,
            orientations=orientations,
            allow_attacker_superset_on_empty=allow_attacker_superset_on_empty,
            must_include_cell=moved_cell,
        )

        stale_keys = tuple(active_match_keys_by_cell.get(moved_cell, ()))
        for match_key in stale_keys:
            _remove_active_match(
                active_matches_by_key=active_matches_by_key,
                active_match_keys_by_cell=active_match_keys_by_cell,
                match_key=match_key,
            )
        for match in filtered_result.matches:
            _add_active_match(
                active_matches_by_key=active_matches_by_key,
                active_match_keys_by_cell=active_match_keys_by_cell,
                match=match,
            )

        position_index = move_index + 1
        _store_position_labels(
            template_origin_maps=template_origin_maps,
            carrier_maps=carrier_maps,
            position_index=position_index,
            board_size=resolved_board_size,
            active_matches=tuple(active_matches_by_key.values()),
        )
        position_stats.append(
            LadderSequencePositionStats(
                position_in_game=position_index,
                last_move=moved_cell,
                used_must_include_filter=True,
                embeddings_considered=filtered_result.stats.embeddings_considered,
                scan_matches_found=filtered_result.stats.matches_found,
                active_matches_after_update=len(active_matches_by_key),
                elapsed_ms=filtered_result.stats.elapsed_ms,
            )
        )

    return LadderGameLabelSequence(
        plane_names=plane_names,
        template_origin_maps=template_origin_maps,
        carrier_maps=carrier_maps,
        position_stats=tuple(position_stats),
    )


def _resolve_clean_game_moves(
    trmph_text: str,
    *,
    board_size: int | None,
) -> tuple[list[str], int]:
    bare_moves = strip_trmph_preamble((trmph_text or "").strip())
    clean_moves = remove_repeated_moves(split_trmph_moves(bare_moves))
    if clean_moves is None:
        raise ValueError("Duplicate moves encountered while building ladder labels.")
    resolved_board_size = _resolve_board_size_from_trmph(
        trmph_text,
        fallback_board_size=board_size,
    )
    return clean_moves, resolved_board_size


def _resolve_board_size_from_trmph(
    trmph_text: str,
    *,
    fallback_board_size: int | None,
) -> int:
    match = _TRMPH_PREAMBLE_RE.search((trmph_text or "").strip())
    if match is not None:
        size = int(match.group(1))
    elif fallback_board_size is not None:
        size = int(fallback_board_size)
    else:
        raise ValueError(
            "Could not infer board size from TRMPH preamble and no fallback was provided."
        )
    if size <= 0:
        raise ValueError(f"Board size must be positive, got {size}")
    return size


def _store_position_labels(
    *,
    template_origin_maps: np.ndarray,
    carrier_maps: np.ndarray,
    position_index: int,
    board_size: int,
    active_matches: Iterable[LadderTemplateMatch],
) -> None:
    labels = build_ladder_certificate_labels(
        tuple(active_matches),
        board_size=board_size,
    )
    template_origin_maps[position_index] = labels.template_origin_maps
    carrier_maps[position_index] = labels.carrier_maps


def _match_key(match: LadderTemplateMatch) -> MatchKey:
    return (
        match.source_path,
        match.orientation,
        match.transform_name,
        match.template_origin,
    )


def _add_active_match(
    *,
    active_matches_by_key: dict[MatchKey, LadderTemplateMatch],
    active_match_keys_by_cell: dict[BoardCoord, set[MatchKey]],
    match: LadderTemplateMatch,
) -> None:
    match_key = _match_key(match)
    previous = active_matches_by_key.get(match_key)
    if previous is not None:
        if previous == match:
            return
        _remove_active_match(
            active_matches_by_key=active_matches_by_key,
            active_match_keys_by_cell=active_match_keys_by_cell,
            match_key=match_key,
        )
    active_matches_by_key[match_key] = match
    for cell in match.carrier_cells:
        active_match_keys_by_cell.setdefault(cell, set()).add(match_key)


def _remove_active_match(
    *,
    active_matches_by_key: dict[MatchKey, LadderTemplateMatch],
    active_match_keys_by_cell: dict[BoardCoord, set[MatchKey]],
    match_key: MatchKey,
) -> None:
    match = active_matches_by_key.pop(match_key, None)
    if match is None:
        return
    for cell in match.carrier_cells:
        keys = active_match_keys_by_cell.get(cell)
        if keys is None:
            continue
        keys.discard(match_key)
        if not keys:
            del active_match_keys_by_cell[cell]
