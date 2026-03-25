"""Helpers for ladder-certificate viewer payloads."""

from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Any

import numpy as np

from hex_ai.data_processing import parse_trmph_line_flexible
from hex_ai.data_utils import load_trmph_file, remove_repeated_moves
from hex_ai.enums import Piece
from hex_ai.utils.format_conversion import (
    split_trmph_moves,
    strip_trmph_preamble,
    trmph_move_to_rowcol,
)

from .labeler import build_ladder_certificate_labels
from .library_loader import LoadedLadderTemplate
from .matcher import LadderTemplateMatch, find_ladder_template_matches
from .sidecar import (
    LadderCertificateSidecarRecord,
    ladder_certificate_sidecar_path_for_trmph,
    load_ladder_certificate_sidecar,
)

_TRMPH_PREAMBLE_RE = re.compile(r"#(\d+),")


@dataclass(frozen=True)
class ViewerGameEntry:
    """One parseable TRMPH game eligible for viewer navigation."""

    game_index: int
    line_index: int
    trmph_text: str
    winner: str | None
    move_count: int
    preview_moves: tuple[str, ...]


def list_trmph_files_under_root(root: str | Path) -> tuple[Path, ...]:
    """List `.trmph` files recursively under a root directory."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"Root directory not found: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_path}")
    return tuple(sorted(root_path.glob("**/*.trmph")))


def extract_viewer_game_entries(
    file_path: str | Path,
    *,
    include_winnerless: bool = False,
) -> tuple[ViewerGameEntry, ...]:
    """Parse a TRMPH file into viewer-eligible games."""
    entries: list[ViewerGameEntry] = []
    eligible_index = 0
    for line_index, line in enumerate(load_trmph_file(Path(file_path))):
        try:
            trmph_text, winner = parse_trmph_line_flexible(line)
        except ValueError:
            continue
        if winner is None and not include_winnerless:
            continue
        moves = _resolve_clean_moves(trmph_text)
        entries.append(
            ViewerGameEntry(
                game_index=eligible_index,
                line_index=line_index,
                trmph_text=trmph_text,
                winner=winner,
                move_count=len(moves),
                preview_moves=tuple(moves[:8]),
            )
        )
        eligible_index += 1
    return tuple(entries)


def build_ladder_certificate_file_summary(
    file_path: str | Path,
    *,
    include_winnerless: bool = False,
) -> dict[str, Any]:
    """Build cheap per-file summary payload for the ladder viewer."""
    path = Path(file_path)
    entries = extract_viewer_game_entries(
        path,
        include_winnerless=include_winnerless,
    )
    sidecar_path = ladder_certificate_sidecar_path_for_trmph(path)
    sidecar_records: tuple[LadderCertificateSidecarRecord, ...] = ()
    if sidecar_path.exists():
        sidecar_records = load_ladder_certificate_sidecar(sidecar_path)

    games: list[dict[str, Any]] = []
    for entry in entries:
        record = sidecar_records[entry.game_index] if entry.game_index < len(sidecar_records) else None
        positions_with_any_match = None
        max_active_matches = None
        if record is not None:
            positions_with_any_match = int(
                sum(count > 0 for count in record.match_counts_by_position)
            )
            max_active_matches = int(max(record.match_counts_by_position))
        games.append(
            {
                "game_index": entry.game_index,
                "line_index": entry.line_index,
                "winner": entry.winner,
                "move_count": entry.move_count,
                "preview_moves": list(entry.preview_moves),
                "has_sidecar_record": record is not None,
                "positions_with_any_match": positions_with_any_match,
                "max_active_matches": max_active_matches,
            }
        )

    return {
        "file_path": str(path),
        "sidecar_path": str(sidecar_path),
        "sidecar_present": bool(sidecar_records),
        "sidecar_record_count": len(sidecar_records),
        "eligible_game_count": len(entries),
        "sidecar_aligned": len(sidecar_records) == len(entries),
        "games": games,
    }


def build_ladder_certificate_viewer_game_payload(
    file_path: str | Path,
    *,
    game_index: int,
    templates: tuple[LoadedLadderTemplate, ...] | list[LoadedLadderTemplate],
    include_winnerless: bool = False,
    orientations: tuple[str, ...] = ("red_bottom", "blue_right"),
    allow_attacker_superset_on_empty: bool = True,
) -> dict[str, Any]:
    """Build full viewer payload for one selected game."""
    path = Path(file_path)
    entries = extract_viewer_game_entries(
        path,
        include_winnerless=include_winnerless,
    )
    if not (0 <= int(game_index) < len(entries)):
        raise IndexError(
            f"game_index {game_index} out of range for {path} (games={len(entries)})"
        )
    entry = entries[int(game_index)]
    moves = _resolve_clean_moves(entry.trmph_text)
    board_size = _resolve_board_size(entry.trmph_text)
    boards = _build_piece_board_sequence(moves, board_size=board_size)

    sidecar_record = _load_sidecar_record_if_available(
        path,
        game_index=entry.game_index,
    )
    sidecar_origin_maps = None
    sidecar_carrier_maps = None
    if sidecar_record is not None:
        sidecar_origin_maps = sidecar_record.decode_template_origin_maps()
        sidecar_carrier_maps = sidecar_record.decode_carrier_maps()
        plane_names = tuple(sidecar_record.plane_names)
    else:
        plane_names = build_ladder_certificate_labels(
            (),
            board_size=board_size,
        ).plane_names

    positions: list[dict[str, Any]] = []
    for position_index, board in enumerate(boards):
        live_result = find_ladder_template_matches(
            board,
            templates,
            orientations=orientations,
            allow_attacker_superset_on_empty=allow_attacker_superset_on_empty,
        )
        live_labels = build_ladder_certificate_labels(
            live_result.matches,
            board_size=board_size,
        )

        if sidecar_origin_maps is not None and sidecar_carrier_maps is not None:
            origin_maps = sidecar_origin_maps[position_index]
            carrier_maps = sidecar_carrier_maps[position_index]
            origin_equal = bool(
                np.array_equal(origin_maps, live_labels.template_origin_maps)
            )
            carrier_equal = bool(
                np.array_equal(carrier_maps, live_labels.carrier_maps)
            )
        else:
            origin_maps = live_labels.template_origin_maps
            carrier_maps = live_labels.carrier_maps
            origin_equal = True
            carrier_equal = True

        last_move = None
        if position_index > 0:
            last_move = trmph_move_to_rowcol(
                moves[position_index - 1],
                board_size=board_size,
            )

        positions.append(
            {
                "position_in_game": position_index,
                "last_move": _coord_to_json(last_move),
                "player_to_move": "blue" if position_index % 2 == 0 else "red",
                "board_rows": _board_rows(board),
                "stone_count": int(np.count_nonzero(board != Piece.EMPTY.value)),
                "plane_labels": _plane_labels_payload(
                    live_labels.plane_names,
                    origin_maps,
                    carrier_maps,
                ),
                "matches": [_match_to_payload(match) for match in live_result.matches],
                "stats": {
                    "elapsed_ms": float(live_result.stats.elapsed_ms),
                    "embeddings_considered": int(live_result.stats.embeddings_considered),
                    "matches_found": int(live_result.stats.matches_found),
                },
                "sidecar_consistency": {
                    "using_sidecar_dense_labels": sidecar_record is not None,
                    "template_origin_equal_to_live": origin_equal,
                    "carrier_equal_to_live": carrier_equal,
                },
            }
        )

    return {
        "file_path": str(path),
        "sidecar_path": str(ladder_certificate_sidecar_path_for_trmph(path)),
        "sidecar_present": sidecar_record is not None,
        "game_index": entry.game_index,
        "line_index": entry.line_index,
        "winner": entry.winner,
        "trmph_text": entry.trmph_text,
        "move_count": entry.move_count,
        "board_size": board_size,
        "plane_names": list(plane_names),
        "positions": positions,
    }


def _load_sidecar_record_if_available(
    file_path: Path,
    *,
    game_index: int,
) -> LadderCertificateSidecarRecord | None:
    sidecar_path = ladder_certificate_sidecar_path_for_trmph(file_path)
    if not sidecar_path.exists():
        return None
    records = load_ladder_certificate_sidecar(sidecar_path)
    if not (0 <= game_index < len(records)):
        return None
    return records[game_index]


def _resolve_clean_moves(trmph_text: str) -> list[str]:
    bare_moves = strip_trmph_preamble((trmph_text or "").strip())
    clean_moves = remove_repeated_moves(split_trmph_moves(bare_moves))
    if clean_moves is None:
        raise ValueError("Duplicate moves encountered in TRMPH game.")
    return clean_moves


def _resolve_board_size(trmph_text: str) -> int:
    match = _TRMPH_PREAMBLE_RE.search((trmph_text or "").strip())
    if match is None:
        raise ValueError(f"TRMPH text is missing board-size preamble: {trmph_text[:32]!r}")
    board_size = int(match.group(1))
    if board_size <= 0:
        raise ValueError(f"Board size must be positive, got {board_size}")
    return board_size


def _build_piece_board_sequence(
    moves: list[str],
    *,
    board_size: int,
) -> tuple[np.ndarray, ...]:
    board = np.full((board_size, board_size), Piece.EMPTY.value, dtype="U1")
    boards = [board.copy()]
    for move_index, move in enumerate(moves):
        row, col = trmph_move_to_rowcol(move, board_size=board_size)
        board[row, col] = Piece.BLUE.value if move_index % 2 == 0 else Piece.RED.value
        boards.append(board.copy())
    return tuple(boards)


def _board_rows(board: np.ndarray) -> list[str]:
    return ["".join(str(cell) for cell in row) for row in board]


def _coords_from_mask(mask: np.ndarray) -> list[list[int]]:
    rows, cols = np.nonzero(mask)
    return [[int(row), int(col)] for row, col in zip(rows, cols, strict=True)]


def _coord_to_json(coord: tuple[int, int] | None) -> list[int] | None:
    if coord is None:
        return None
    return [int(coord[0]), int(coord[1])]


def _match_to_payload(match: LadderTemplateMatch) -> dict[str, Any]:
    return {
        "template_id": match.template_id,
        "row_class": int(match.row_class),
        "orientation": match.orientation,
        "transform_name": match.transform_name,
        "template_origin": _coord_to_json(match.template_origin),
        "carrier_cells": [_coord_to_json(coord) for coord in match.carrier_cells],
        "attacker_required_cells": [
            _coord_to_json(coord) for coord in match.attacker_required_cells
        ],
        "defender_required_cells": [
            _coord_to_json(coord) for coord in match.defender_required_cells
        ],
        "boundary_cells": [_coord_to_json(coord) for coord in match.boundary_cells],
        "attacker_superset_cells": [
            _coord_to_json(coord) for coord in match.attacker_superset_cells
        ],
        "source_path": match.source_path,
        "source_set": match.source_set,
    }


def _plane_labels_payload(
    plane_names: tuple[str, ...],
    template_origin_maps: np.ndarray,
    carrier_maps: np.ndarray,
) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for plane_index, plane_name in enumerate(plane_names):
        origin_cells = _coords_from_mask(template_origin_maps[plane_index])
        carrier_cells = _coords_from_mask(carrier_maps[plane_index])
        if not origin_cells and not carrier_cells:
            continue
        payload.append(
            {
                "plane_name": plane_name,
                "origin_cells": origin_cells,
                "carrier_cells": carrier_cells,
            }
        )
    return payload
