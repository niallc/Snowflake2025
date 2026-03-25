import json
import gzip
import pickle

import numpy as np

from hex_ai.data_utils import create_board_from_moves
from hex_ai.enums import Piece
from hex_ai.utils.format_conversion import rowcol_to_trmph
from hex_ai.utils.ladder_templates import (
    build_ladder_certificate_labels,
    build_ladder_certificate_target_for_training_example,
    ladder_certificate_plane_index,
    load_hexwiki_generated_ladder_templates,
    load_ladder_template_library,
    resolve_last_move_for_training_example,
)
from hex_ai.utils.ladder_templates.benchmark import _benchmark_processed_shard
from hex_ai.utils.ladder_templates.matcher import (
    find_ladder_template_matches,
    transform_board_long_diagonal_color_swap,
)


def _write_template(tmp_path, *, template_name: str = "second_row_escape_unit") -> None:
    payload = {
        "schema_version": 1,
        "kind": "ladder_template_annotation",
        "coord_system": "row_col_offset",
        "metadata": {
            "name": template_name,
            "family": "unit_test",
            "attacker": "red",
            "target_edge": "red_bottom",
            "open_left": True,
            "open_right": False,
            "notes": "unit",
        },
        "grid": {
            "rows": 2,
            "cols": 2,
            "radius": 32,
            "origin_x": 80,
            "origin_y": 70,
            "grid_opacity": 1,
            "show_coords": False,
        },
        "image_overlay": {
            "image_name": "sample.svg",
            "image_width": 128,
            "image_height": 128,
            "image_opacity": 0.8,
            "image_scale": 1.0,
            "image_offset_x": 0.0,
            "image_offset_y": 0.0,
        },
        "cells": [
            {"row": 0, "col": 0, "state": "plus"},
            {"row": 0, "col": 1, "state": "empty"},
            {"row": 1, "col": 0, "state": "plus"},
            {"row": 1, "col": 1, "state": "red"},
        ],
    }
    (tmp_path / f"{template_name}.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def _build_exact_board() -> np.ndarray:
    board = np.full((7, 7), Piece.EMPTY.value, dtype="U1")
    board[3, 4] = Piece.RED.value
    return board


def test_generated_hexwiki_loader_is_available():
    templates = load_hexwiki_generated_ladder_templates()

    assert templates
    assert {template.row_class for template in templates} == {2, 3, 4, 5}
    assert all(template.source_set == "generated" for template in templates)


def test_matcher_finds_exact_match_and_builds_labels(tmp_path):
    _write_template(tmp_path)
    templates = load_ladder_template_library(tmp_path, source_set="generated")
    board = _build_exact_board()

    result = find_ladder_template_matches(
        board,
        templates,
        orientations=("red_bottom",),
    )

    assert result.stats.matches_found == 1
    match = result.matches[0]
    assert match.orientation == "red_bottom"
    assert match.template_origin == (2, 3)
    assert match.carrier_cells == ((2, 4), (3, 4))
    assert match.attacker_required_cells == ((3, 4),)
    assert match.attacker_superset_cells == ()

    labels = build_ladder_certificate_labels(result.matches, board_size=7)
    plane_index = ladder_certificate_plane_index(orientation="red_bottom", row_class=2)
    assert labels.template_origin_maps[plane_index, 2, 3] == 1
    assert labels.carrier_maps[plane_index, 2, 4] == 1
    assert labels.carrier_maps[plane_index, 3, 4] == 1


def test_matcher_allows_attacker_superset_on_empty_cells(tmp_path):
    _write_template(tmp_path)
    templates = load_ladder_template_library(tmp_path, source_set="generated")
    board = _build_exact_board()
    board[2, 4] = Piece.RED.value

    allowed = find_ladder_template_matches(
        board,
        templates,
        orientations=("red_bottom",),
        allow_attacker_superset_on_empty=True,
    )
    disallowed = find_ladder_template_matches(
        board,
        templates,
        orientations=("red_bottom",),
        allow_attacker_superset_on_empty=False,
    )

    assert allowed.stats.matches_found >= 1
    assert any(match.attacker_superset_cells == ((2, 4),) for match in allowed.matches)
    assert all(not match.attacker_superset_cells for match in disallowed.matches)


def test_must_include_cell_reduces_embedding_checks(tmp_path):
    _write_template(tmp_path)
    templates = load_ladder_template_library(tmp_path, source_set="generated")
    board = _build_exact_board()

    unrestricted = find_ladder_template_matches(
        board,
        templates,
        orientations=("red_bottom",),
    )
    filtered = find_ladder_template_matches(
        board,
        templates,
        orientations=("red_bottom",),
        must_include_cell=(3, 4),
    )

    assert unrestricted.stats.matches_found == 1
    assert filtered.stats.matches_found == 1
    assert unrestricted.stats.embeddings_considered == 36
    assert filtered.stats.embeddings_considered == 2


def test_matcher_supports_blue_right_via_long_diagonal_color_swap(tmp_path):
    _write_template(tmp_path)
    templates = load_ladder_template_library(tmp_path, source_set="generated")
    red_view_board = _build_exact_board()
    blue_right_board = transform_board_long_diagonal_color_swap(red_view_board)

    result = find_ladder_template_matches(
        blue_right_board,
        templates,
        orientations=("blue_right",),
    )

    assert result.stats.matches_found == 1
    match = result.matches[0]
    assert match.orientation == "blue_right"
    assert match.template_origin == (3, 2)
    assert match.carrier_cells == ((4, 2), (4, 3))
    assert match.attacker_required_cells == ((4, 3),)


def test_matcher_accepts_three_channel_training_board(tmp_path):
    _write_template(tmp_path)
    templates = load_ladder_template_library(tmp_path, source_set="generated")
    board_3ch = np.zeros((3, 7, 7), dtype=np.float32)
    board_3ch[1, 3, 4] = 1.0

    result = find_ladder_template_matches(
        board_3ch,
        templates,
        orientations=("red_bottom",),
    )

    assert result.stats.matches_found == 1


def test_training_example_target_uses_last_move_filter(tmp_path):
    _write_template(tmp_path)
    templates = load_ladder_template_library(tmp_path, source_set="generated")

    moves = [
        rowcol_to_trmph(0, 0, board_size=13),
        rowcol_to_trmph(3, 4, board_size=13),
    ]
    source_file = tmp_path / "games.trmph"
    source_file.write_text(f"#13,{''.join(moves)}\n", encoding="utf-8")

    example = {
        "board": create_board_from_moves(moves[:2]),
        "metadata": {
            "source_file": str(source_file),
            "game_id": (0, 0),
            "position_in_game": 2,
        },
    }

    last_move = resolve_last_move_for_training_example(example)
    target = build_ladder_certificate_target_for_training_example(
        example,
        templates,
        orientations=("red_bottom",),
        use_last_move_filter=True,
    )

    assert last_move == (3, 4)
    assert target.match.used_last_move_filter is True
    assert target.match.last_move == (3, 4)
    assert target.match.match_result.stats.embeddings_considered == 2
    plane_index = ladder_certificate_plane_index(orientation="red_bottom", row_class=2)
    assert target.labels.carrier_maps[plane_index, 2, 4] == 1
    assert target.labels.carrier_maps[plane_index, 3, 4] == 1


def test_processed_shard_benchmark_reports_last_move_filter_usage(tmp_path):
    _write_template(tmp_path)
    templates = load_ladder_template_library(tmp_path, source_set="generated")

    moves = [
        rowcol_to_trmph(0, 0, board_size=13),
        rowcol_to_trmph(3, 4, board_size=13),
    ]
    source_file = tmp_path / "games.trmph"
    source_file.write_text(f"#13,{''.join(moves)}\n", encoding="utf-8")

    processed_shard = tmp_path / "processed.pkl.gz"
    payload = {
        "examples": [
            {
                "board": create_board_from_moves(moves[:2]),
                "metadata": {
                    "source_file": str(source_file),
                    "game_id": (0, 0),
                    "position_in_game": 2,
                },
            }
        ]
    }
    with gzip.open(processed_shard, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    summary = _benchmark_processed_shard(
        processed_shard,
        templates=templates,
        sample_size=1,
        seed=0,
        orientations=("red_bottom",),
        compare_last_move_filter=True,
    )

    assert summary["mode"] == "processed_shard"
    assert summary["positions_benchmarked"] == 1
    assert summary["last_move_filter"]["positions_using_filter"] == 1
    assert summary["last_move_filter"]["last_move_lookup_errors"] == 0
