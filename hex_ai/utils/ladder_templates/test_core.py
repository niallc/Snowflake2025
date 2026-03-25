import numpy as np

from hex_ai.enums import Piece, Player
from hex_ai.utils.ladder_templates.core import (
    ladder_template_annotation_from_dict,
    materialize_ladder_template,
)


def _sample_payload() -> dict:
    return {
        "schema_version": 1,
        "kind": "ladder_template_annotation",
        "coord_system": "row_col_offset",
        "metadata": {
            "name": "sample_escape",
            "family": "unit_test",
            "attacker": "red",
            "target_edge": "red_bottom",
            "open_left": True,
            "open_right": False,
            "notes": "sample",
        },
        "grid": {
            "rows": 4,
            "cols": 5,
            "radius": 32,
            "origin_x": 80,
            "origin_y": 70,
            "grid_opacity": 1,
            "show_coords": False,
        },
        "image_overlay": {
            "image_name": "sample.png",
            "image_width": 512,
            "image_height": 512,
            "image_opacity": 0.8,
            "image_scale": 1.0,
            "image_offset_x": 24,
            "image_offset_y": 48,
        },
        "cells": [
            {"row": 0, "col": 1, "state": "red"},
            {"row": 0, "col": 2, "state": "red"},
            {"row": 1, "col": 0, "state": "plus"},
            {"row": 1, "col": 1, "state": "empty"},
            {"row": 1, "col": 2, "state": "blue"},
            {"row": 2, "col": 3, "state": "minus"},
            {"row": 3, "col": 4, "state": "shaded"},
        ],
    }


def test_annotation_loader_normalizes_and_validates_fields():
    annotation = ladder_template_annotation_from_dict(_sample_payload())

    assert annotation.metadata.name == "sample_escape"
    assert annotation.metadata.target_edge == "red_bottom"
    assert annotation.metadata.open_left is True
    assert annotation.grid.rows == 4
    assert annotation.image_overlay.image_offset_x == 24.0
    assert annotation.image_overlay.image_offset_y == 48.0
    assert tuple((cell.row, cell.col, cell.state) for cell in annotation.cells) == (
        (0, 1, "red"),
        (0, 2, "red"),
        (1, 0, "plus"),
        (1, 1, "empty"),
        (1, 2, "blue"),
        (2, 3, "minus"),
        (3, 4, "shaded"),
    )


def test_materialize_places_stones_and_preserves_boundaries_as_metadata():
    annotation = ladder_template_annotation_from_dict(_sample_payload())
    materialized = materialize_ladder_template(annotation, board_size=9, anchor_row=2, anchor_col=3)

    assert materialized.board.shape == (9, 9)
    assert materialized.board[2, 4] == Piece.RED.value
    assert materialized.board[2, 5] == Piece.RED.value
    assert materialized.board[3, 5] == Piece.BLUE.value
    assert materialized.board[3, 3] == Piece.EMPTY.value
    assert materialized.board[4, 6] == Piece.EMPTY.value
    assert materialized.board[5, 7] == Piece.EMPTY.value
    assert materialized.red_stones == ((2, 4), (2, 5))
    assert materialized.blue_stones == ((3, 5),)
    assert materialized.left_boundary == ((3, 3),)
    assert materialized.right_boundary == ((4, 6),)
    assert materialized.shaded_cells == ((5, 7),)


def test_materialized_template_can_create_game_state():
    annotation = ladder_template_annotation_from_dict(_sample_payload())
    materialized = materialize_ladder_template(annotation, board_size=9)
    state = materialized.to_hex_game_state(current_player=Player.BLUE)

    assert state.board.shape == (9, 9)
    assert state.current_player_enum == Player.BLUE
    np.testing.assert_array_equal(state.board, materialized.board)
