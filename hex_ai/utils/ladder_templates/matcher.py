"""Deterministic ladder-template matching for certificate supervision."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Iterable

import numpy as np
import torch

from hex_ai.enums import Piece

from .library_loader import LoadedLadderTemplate

TemplateCoord = tuple[int, int]
BoardCoord = tuple[int, int]

_MATCHABLE_TEMPLATE_TARGET_EDGE = "red_bottom"
_MATCHABLE_TEMPLATE_ATTACKER = "red"


@dataclass(frozen=True)
class LadderTemplateMatch:
    """One successful ladder-template certificate match on a concrete board."""

    template_id: str
    row_class: int
    orientation: str
    transform_name: str
    template_origin: BoardCoord
    carrier_cells: tuple[BoardCoord, ...]
    attacker_required_cells: tuple[BoardCoord, ...]
    defender_required_cells: tuple[BoardCoord, ...]
    boundary_cells: tuple[BoardCoord, ...]
    attacker_superset_cells: tuple[BoardCoord, ...]
    source_path: str
    source_set: str


@dataclass(frozen=True)
class LadderMatchStats:
    """Simple timing and search-volume summary for a match run."""

    elapsed_ms: float
    templates_considered: int
    embeddings_considered: int
    matches_found: int
    must_include_cell: BoardCoord | None
    allow_attacker_superset_on_empty: bool


@dataclass(frozen=True)
class LadderMatchResult:
    """Match records plus timing/search summary."""

    matches: tuple[LadderTemplateMatch, ...]
    stats: LadderMatchStats


def normalize_board_for_ladder_matching(
    board: np.ndarray | torch.Tensor | object,
) -> np.ndarray:
    """Normalize board input into an `N x N` character board."""
    if hasattr(board, "board"):
        board = getattr(board, "board")
    if isinstance(board, torch.Tensor):
        board = board.detach().cpu().numpy()
    if not isinstance(board, np.ndarray):
        raise TypeError(
            "board must be a numpy array, torch tensor, or object with a .board "
            f"numpy array, got {type(board)!r}"
        )

    if board.ndim == 2:
        normalized = board.astype("U1", copy=False)
        _validate_piece_board(normalized)
        return normalized

    if board.ndim != 3 or board.shape[0] not in {2, 3}:
        raise ValueError(
            "Expected board with shape (N, N), (2, N, N), or (3, N, N); "
            f"got {tuple(board.shape)}"
        )

    blue_plane = board[0]
    red_plane = board[1]
    if blue_plane.shape != red_plane.shape:
        raise ValueError(
            "Blue/red planes must have matching shapes, got "
            f"{blue_plane.shape} and {red_plane.shape}"
        )
    result = np.full(blue_plane.shape, Piece.EMPTY.value, dtype="U1")
    blue_mask = blue_plane > 0.5
    red_mask = red_plane > 0.5
    overlap = blue_mask & red_mask
    if np.any(overlap):
        overlap_count = int(np.count_nonzero(overlap))
        raise ValueError(
            f"Board planes overlap on {overlap_count} cells; cannot ladder-match "
            "ambiguous occupancy."
        )
    result[blue_mask] = Piece.BLUE.value
    result[red_mask] = Piece.RED.value
    return result


def transform_board_long_diagonal_color_swap(board: np.ndarray) -> np.ndarray:
    """Transpose the board and swap piece colors."""
    transposed = np.transpose(board)
    return np.where(
        transposed == Piece.RED.value,
        Piece.BLUE.value,
        np.where(
            transposed == Piece.BLUE.value,
            Piece.RED.value,
            transposed,
        ),
    )


def find_ladder_template_matches(
    board: np.ndarray | torch.Tensor | object,
    templates: Iterable[LoadedLadderTemplate],
    *,
    orientations: tuple[str, ...] = ("red_bottom", "blue_right"),
    allow_attacker_superset_on_empty: bool = True,
    must_include_cell: BoardCoord | None = None,
) -> LadderMatchResult:
    """
    Find ladder-template matches on a concrete board.

    Phase-1 semantics are intentionally conservative:
    - current direct matching supports templates authored in the current generated
      HexWiki orientation (`attacker=red`, `target_edge=red_bottom`)
    - `+` / `-` cells are treated as boundary metadata rather than board
      occupancy constraints
    - empty carrier cells may optionally be occupied by the attacker, capturing
      the monotone “extra attacker stones keep a certificate valid” closure
    """
    piece_board = normalize_board_for_ladder_matching(board)
    board_size = int(piece_board.shape[0])
    if piece_board.shape[1] != board_size:
        raise ValueError(f"Expected square board for ladder matching, got {piece_board.shape}")

    started_at = perf_counter()
    matches: list[LadderTemplateMatch] = []
    templates_considered = 0
    embeddings_considered = 0

    for orientation in orientations:
        view_board, transform_name, to_view_coord, from_view_coord = _board_view_for_orientation(
            piece_board,
            orientation=orientation,
        )
        view_must_include = (
            to_view_coord(must_include_cell) if must_include_cell is not None else None
        )
        for template in templates:
            _validate_template_for_phase1_matching(template)
            templates_considered += 1
            candidate_origins = _candidate_origins_for_template(
                template,
                board_size=board_size,
                must_include_cell=view_must_include,
            )
            for origin_row, origin_col in candidate_origins:
                embeddings_considered += 1
                match = _match_one_embedding(
                    view_board,
                    template,
                    origin_row=origin_row,
                    origin_col=origin_col,
                    orientation=orientation,
                    transform_name=transform_name,
                    allow_attacker_superset_on_empty=allow_attacker_superset_on_empty,
                    from_view_coord=from_view_coord,
                )
                if match is not None:
                    matches.append(match)

    elapsed_ms = (perf_counter() - started_at) * 1000.0
    return LadderMatchResult(
        matches=tuple(matches),
        stats=LadderMatchStats(
            elapsed_ms=elapsed_ms,
            templates_considered=templates_considered,
            embeddings_considered=embeddings_considered,
            matches_found=len(matches),
            must_include_cell=must_include_cell,
            allow_attacker_superset_on_empty=allow_attacker_superset_on_empty,
        ),
    )


def _validate_piece_board(board: np.ndarray) -> None:
    valid_values = {Piece.EMPTY.value, Piece.BLUE.value, Piece.RED.value}
    bad_values = {str(value) for value in np.unique(board) if str(value) not in valid_values}
    if bad_values:
        raise ValueError(
            f"Board contains unsupported values for ladder matching: {sorted(bad_values)}"
        )


def _board_view_for_orientation(
    board: np.ndarray,
    *,
    orientation: str,
) -> tuple[np.ndarray, str, Callable[[BoardCoord], BoardCoord], Callable[[BoardCoord], BoardCoord]]:
    if orientation == "red_bottom":
        identity = lambda coord: coord
        return board, "identity", identity, identity
    if orientation == "blue_right":
        transpose = lambda coord: (coord[1], coord[0])
        return (
            transform_board_long_diagonal_color_swap(board),
            "long_diagonal_color_swap",
            transpose,
            transpose,
        )
    raise ValueError(
        f"Unsupported ladder-certificate orientation {orientation!r}. "
        "Expected 'red_bottom' or 'blue_right'."
    )


def _validate_template_for_phase1_matching(template: LoadedLadderTemplate) -> None:
    if template.attacker != _MATCHABLE_TEMPLATE_ATTACKER:
        raise NotImplementedError(
            "Phase-1 ladder matcher currently supports only red-attacker templates, "
            f"got {template.attacker!r} in {template.source_path}"
        )
    if template.target_edge != _MATCHABLE_TEMPLATE_TARGET_EDGE:
        raise NotImplementedError(
            "Phase-1 ladder matcher currently supports only red_bottom templates, "
            f"got {template.target_edge!r} in {template.source_path}"
        )


def _candidate_origins_for_template(
    template: LoadedLadderTemplate,
    *,
    board_size: int,
    must_include_cell: BoardCoord | None,
) -> tuple[BoardCoord, ...]:
    max_origin_row = board_size - template.local_rows
    max_origin_col = board_size - template.local_cols
    if max_origin_row < 0 or max_origin_col < 0:
        return ()

    if must_include_cell is None:
        return tuple(
            (origin_row, origin_col)
            for origin_row in range(max_origin_row + 1)
            for origin_col in range(max_origin_col + 1)
        )

    include_row, include_col = must_include_cell
    origins: set[BoardCoord] = set()
    for local_row, local_col in template.carrier_local:
        origin_row = include_row - local_row
        origin_col = include_col - local_col
        if 0 <= origin_row <= max_origin_row and 0 <= origin_col <= max_origin_col:
            origins.add((origin_row, origin_col))
    return tuple(sorted(origins))


def _match_one_embedding(
    board: np.ndarray,
    template: LoadedLadderTemplate,
    *,
    origin_row: int,
    origin_col: int,
    orientation: str,
    transform_name: str,
    allow_attacker_superset_on_empty: bool,
    from_view_coord: Callable[[BoardCoord], BoardCoord],
) -> LadderTemplateMatch | None:
    attacker_required_cells: list[BoardCoord] = []
    defender_required_cells: list[BoardCoord] = []
    attacker_superset_cells: list[BoardCoord] = []
    carrier_cells: list[BoardCoord] = []

    for local_row, local_col in template.attacker_required_local:
        board_coord = (origin_row + local_row, origin_col + local_col)
        if board[board_coord] != Piece.RED.value:
            return None
        mapped_coord = from_view_coord(board_coord)
        attacker_required_cells.append(mapped_coord)
        carrier_cells.append(mapped_coord)

    for local_row, local_col in template.defender_required_local:
        board_coord = (origin_row + local_row, origin_col + local_col)
        if board[board_coord] != Piece.BLUE.value:
            return None
        mapped_coord = from_view_coord(board_coord)
        defender_required_cells.append(mapped_coord)
        carrier_cells.append(mapped_coord)

    for local_row, local_col in template.empty_carrier_local:
        board_coord = (origin_row + local_row, origin_col + local_col)
        piece = board[board_coord]
        if piece == Piece.EMPTY.value:
            carrier_cells.append(from_view_coord(board_coord))
            continue
        if allow_attacker_superset_on_empty and piece == Piece.RED.value:
            mapped_coord = from_view_coord(board_coord)
            attacker_superset_cells.append(mapped_coord)
            carrier_cells.append(mapped_coord)
            continue
        return None

    boundary_cells = tuple(
        from_view_coord((origin_row + local_row, origin_col + local_col))
        for local_row, local_col in (
            template.left_boundary_local + template.right_boundary_local
        )
    )

    return LadderTemplateMatch(
        template_id=template.template_id,
        row_class=template.row_class,
        orientation=orientation,
        transform_name=transform_name,
        template_origin=from_view_coord((origin_row, origin_col)),
        carrier_cells=tuple(sorted(carrier_cells)),
        attacker_required_cells=tuple(sorted(attacker_required_cells)),
        defender_required_cells=tuple(sorted(defender_required_cells)),
        boundary_cells=tuple(sorted(boundary_cells)),
        attacker_superset_cells=tuple(sorted(attacker_superset_cells)),
        source_path=str(template.source_path),
        source_set=template.source_set,
    )
