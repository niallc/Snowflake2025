"""Load and materialize ladder-template annotations.

This module is the bridge between reviewed annotation JSON and concrete board
positions that Snowflake can inspect.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from hex_ai.config import BOARD_SIZE
from hex_ai.enums import Piece, Player
from hex_ai.inference.game_engine import HexGameState

TemplateCoord = tuple[int, int]

_VALID_ATTACKERS = {"red", "blue"}
_VALID_TARGET_EDGES = {"red_bottom", "red_top", "blue_left", "blue_right", "none"}
_VALID_CELL_STATES = {"empty", "red", "blue", "plus", "minus", "shaded"}


@dataclass(frozen=True)
class LadderTemplateCell:
    row: int
    col: int
    state: str


@dataclass(frozen=True)
class LadderTemplateMetadata:
    name: str
    family: str
    attacker: str
    target_edge: str
    open_left: bool
    open_right: bool
    notes: str


@dataclass(frozen=True)
class LadderTemplateGrid:
    rows: int
    cols: int
    radius: float
    origin_x: float
    origin_y: float
    grid_opacity: float
    show_coords: bool


@dataclass(frozen=True)
class LadderTemplateImageOverlay:
    image_name: str
    image_width: float
    image_height: float
    image_opacity: float
    image_scale: float
    image_offset_x: float
    image_offset_y: float


@dataclass(frozen=True)
class LadderTemplateAnnotation:
    schema_version: int
    kind: str
    coord_system: str
    metadata: LadderTemplateMetadata
    grid: LadderTemplateGrid
    image_overlay: LadderTemplateImageOverlay
    cells: tuple[LadderTemplateCell, ...]


@dataclass(frozen=True)
class MaterializedLadderTemplate:
    annotation: LadderTemplateAnnotation
    board: np.ndarray
    board_size: int
    anchor_row: int
    anchor_col: int
    red_stones: tuple[TemplateCoord, ...]
    blue_stones: tuple[TemplateCoord, ...]
    empty_cells: tuple[TemplateCoord, ...]
    left_boundary: tuple[TemplateCoord, ...]
    right_boundary: tuple[TemplateCoord, ...]
    shaded_cells: tuple[TemplateCoord, ...]

    def to_hex_game_state(self, *, current_player: Player) -> HexGameState:
        """Create a HexGameState from the materialized template."""
        return HexGameState(_current_player=current_player, board=self.board.copy())

    def summary_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serializable summary for CLI/debug output."""
        return {
            "name": self.annotation.metadata.name,
            "family": self.annotation.metadata.family,
            "attacker": self.annotation.metadata.attacker,
            "target_edge": self.annotation.metadata.target_edge,
            "board_size": self.board_size,
            "anchor_row": self.anchor_row,
            "anchor_col": self.anchor_col,
            "red_stones": list(self.red_stones),
            "blue_stones": list(self.blue_stones),
            "empty_cells": list(self.empty_cells),
            "left_boundary": list(self.left_boundary),
            "right_boundary": list(self.right_boundary),
            "shaded_cells": list(self.shaded_cells),
        }


def _expect_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object, got {type(value).__name__}")
    return value


def _read_string(value: Any, *, field_name: str, default: str | None = None) -> str:
    if value is None:
        if default is None:
            raise TypeError(f"{field_name} must be a string, got None")
        return default
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value).__name__}")
    return value


def _read_bool(value: Any, *, field_name: str, default: bool = False) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TypeError(f"{field_name} must be a boolean, got {type(value).__name__}")
    return value


def _read_int(value: Any, *, field_name: str, default: int | None = None, minimum: int | None = None) -> int:
    if value is None:
        if default is None:
            raise TypeError(f"{field_name} must be an integer, got None")
        result = default
    else:
        if isinstance(value, bool):
            raise TypeError(f"{field_name} must be an integer, got bool")
        try:
            result = int(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field_name} must be an integer, got {value!r}") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {result}")
    return result


def _read_float(
    value: Any,
    *,
    field_name: str,
    default: float | None = None,
    minimum: float | None = None,
) -> float:
    if value is None:
        if default is None:
            raise TypeError(f"{field_name} must be a number, got None")
        result = default
    else:
        if isinstance(value, bool):
            raise TypeError(f"{field_name} must be a number, got bool")
        try:
            result = float(value)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{field_name} must be a number, got {value!r}") from exc
    if minimum is not None and result < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {result}")
    return result


def _normalize_target_edge(target_edge: str) -> str:
    if target_edge not in _VALID_TARGET_EDGES:
        raise ValueError(
            f"target_edge must be one of {sorted(_VALID_TARGET_EDGES)}, got {target_edge!r}"
        )
    return target_edge


def _normalize_attacker(attacker: str) -> str:
    if attacker not in _VALID_ATTACKERS:
        raise ValueError(f"attacker must be one of {sorted(_VALID_ATTACKERS)}, got {attacker!r}")
    return attacker


def _normalize_cell_state(cell_state: str) -> str:
    if cell_state not in _VALID_CELL_STATES:
        raise ValueError(
            f"cell state must be one of {sorted(_VALID_CELL_STATES)}, got {cell_state!r}"
        )
    return cell_state


def _validate_unique_cells(cells: Iterable[LadderTemplateCell], *, rows: int, cols: int) -> tuple[LadderTemplateCell, ...]:
    seen: set[TemplateCoord] = set()
    normalized: list[LadderTemplateCell] = []
    for cell in cells:
        coord = (cell.row, cell.col)
        if not (0 <= cell.row < rows and 0 <= cell.col < cols):
            raise ValueError(
                f"Cell {coord} is outside the declared grid bounds {(rows, cols)}"
            )
        if coord in seen:
            raise ValueError(f"Duplicate cell annotation for {coord}")
        seen.add(coord)
        normalized.append(cell)
    normalized.sort(key=lambda item: (item.row, item.col, item.state))
    return tuple(normalized)


def ladder_template_annotation_from_dict(payload: Mapping[str, Any]) -> LadderTemplateAnnotation:
    """Validate and normalize annotation JSON payload into domain dataclasses."""
    root = _expect_mapping(payload, field_name="annotation")
    metadata_raw = _expect_mapping(root.get("metadata", {}), field_name="metadata")
    grid_raw = _expect_mapping(root.get("grid", {}), field_name="grid")
    image_raw = _expect_mapping(root.get("image_overlay", {}), field_name="image_overlay")

    metadata = LadderTemplateMetadata(
        name=_read_string(metadata_raw.get("name"), field_name="metadata.name", default=""),
        family=_read_string(metadata_raw.get("family"), field_name="metadata.family", default=""),
        attacker=_normalize_attacker(
            _read_string(metadata_raw.get("attacker"), field_name="metadata.attacker", default="red")
        ),
        target_edge=_normalize_target_edge(
            _read_string(
                metadata_raw.get("target_edge"),
                field_name="metadata.target_edge",
                default="red_bottom",
            )
        ),
        open_left=_read_bool(metadata_raw.get("open_left"), field_name="metadata.open_left", default=False),
        open_right=_read_bool(metadata_raw.get("open_right"), field_name="metadata.open_right", default=False),
        notes=_read_string(metadata_raw.get("notes"), field_name="metadata.notes", default=""),
    )

    grid = LadderTemplateGrid(
        rows=_read_int(grid_raw.get("rows"), field_name="grid.rows", default=7, minimum=1),
        cols=_read_int(grid_raw.get("cols"), field_name="grid.cols", default=7, minimum=1),
        radius=_read_float(grid_raw.get("radius"), field_name="grid.radius", default=34.0, minimum=1.0),
        origin_x=_read_float(grid_raw.get("origin_x"), field_name="grid.origin_x", default=80.0),
        origin_y=_read_float(grid_raw.get("origin_y"), field_name="grid.origin_y", default=80.0),
        grid_opacity=_read_float(grid_raw.get("grid_opacity"), field_name="grid.grid_opacity", default=1.0, minimum=0.0),
        show_coords=_read_bool(grid_raw.get("show_coords"), field_name="grid.show_coords", default=False),
    )

    image_overlay = LadderTemplateImageOverlay(
        image_name=_read_string(image_raw.get("image_name"), field_name="image_overlay.image_name", default=""),
        image_width=_read_float(image_raw.get("image_width"), field_name="image_overlay.image_width", default=0.0, minimum=0.0),
        image_height=_read_float(image_raw.get("image_height"), field_name="image_overlay.image_height", default=0.0, minimum=0.0),
        image_opacity=_read_float(image_raw.get("image_opacity"), field_name="image_overlay.image_opacity", default=0.8, minimum=0.0),
        image_scale=_read_float(image_raw.get("image_scale"), field_name="image_overlay.image_scale", default=1.0, minimum=0.05),
        image_offset_x=_read_float(image_raw.get("image_offset_x"), field_name="image_overlay.image_offset_x", default=0.0, minimum=0.0),
        image_offset_y=_read_float(image_raw.get("image_offset_y"), field_name="image_overlay.image_offset_y", default=0.0, minimum=0.0),
    )

    cells_raw = root.get("cells", [])
    if not isinstance(cells_raw, list):
        raise TypeError(f"cells must be an array, got {type(cells_raw).__name__}")

    cells = _validate_unique_cells(
        (
            LadderTemplateCell(
                row=_read_int(_expect_mapping(cell, field_name=f"cells[{index}]").get("row"), field_name=f"cells[{index}].row"),
                col=_read_int(_expect_mapping(cell, field_name=f"cells[{index}]").get("col"), field_name=f"cells[{index}].col"),
                state=_normalize_cell_state(
                    _read_string(
                        _expect_mapping(cell, field_name=f"cells[{index}]").get("state"),
                        field_name=f"cells[{index}].state",
                    )
                ),
            )
            for index, cell in enumerate(cells_raw)
        ),
        rows=grid.rows,
        cols=grid.cols,
    )

    return LadderTemplateAnnotation(
        schema_version=_read_int(root.get("schema_version"), field_name="schema_version", default=1, minimum=1),
        kind=_read_string(root.get("kind"), field_name="kind", default="ladder_template_annotation"),
        coord_system=_read_string(root.get("coord_system"), field_name="coord_system", default="row_col_offset"),
        metadata=metadata,
        grid=grid,
        image_overlay=image_overlay,
        cells=cells,
    )


def load_ladder_template_annotation(path: str | Path) -> LadderTemplateAnnotation:
    """Load annotation JSON from disk."""
    template_path = Path(path)
    with template_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ladder_template_annotation_from_dict(payload)


def materialize_ladder_template(
    annotation: LadderTemplateAnnotation,
    *,
    board_size: int = BOARD_SIZE,
    anchor_row: int = 0,
    anchor_col: int = 0,
) -> MaterializedLadderTemplate:
    """Embed an annotated template into a concrete square board."""
    normalized_board_size = _read_int(board_size, field_name="board_size", minimum=1)
    normalized_anchor_row = _read_int(anchor_row, field_name="anchor_row", minimum=0)
    normalized_anchor_col = _read_int(anchor_col, field_name="anchor_col", minimum=0)

    board = np.full((normalized_board_size, normalized_board_size), Piece.EMPTY.value, dtype="U1")
    red_stones: list[TemplateCoord] = []
    blue_stones: list[TemplateCoord] = []
    empty_cells: list[TemplateCoord] = []
    left_boundary: list[TemplateCoord] = []
    right_boundary: list[TemplateCoord] = []
    shaded_cells: list[TemplateCoord] = []

    for cell in annotation.cells:
        board_row = normalized_anchor_row + cell.row
        board_col = normalized_anchor_col + cell.col
        if not (0 <= board_row < normalized_board_size and 0 <= board_col < normalized_board_size):
            raise ValueError(
                f"Template cell {(cell.row, cell.col)} anchored at "
                f"{(normalized_anchor_row, normalized_anchor_col)} exceeds board size {normalized_board_size}"
            )

        board_coord = (board_row, board_col)
        if cell.state == "red":
            board[board_row, board_col] = Piece.RED.value
            red_stones.append(board_coord)
        elif cell.state == "blue":
            board[board_row, board_col] = Piece.BLUE.value
            blue_stones.append(board_coord)
        elif cell.state == "empty":
            empty_cells.append(board_coord)
        elif cell.state == "plus":
            left_boundary.append(board_coord)
        elif cell.state == "minus":
            right_boundary.append(board_coord)
        elif cell.state == "shaded":
            shaded_cells.append(board_coord)
        else:
            raise ValueError(f"Unhandled cell state {cell.state!r}")

    return MaterializedLadderTemplate(
        annotation=annotation,
        board=board,
        board_size=normalized_board_size,
        anchor_row=normalized_anchor_row,
        anchor_col=normalized_anchor_col,
        red_stones=tuple(red_stones),
        blue_stones=tuple(blue_stones),
        empty_cells=tuple(empty_cells),
        left_boundary=tuple(left_boundary),
        right_boundary=tuple(right_boundary),
        shaded_cells=tuple(shaded_cells),
    )

