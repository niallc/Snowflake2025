import json

import numpy as np

from hex_ai.data_utils import create_board_from_moves
from hex_ai.utils.format_conversion import rowcol_to_trmph
from hex_ai.utils.ladder_templates import (
    build_ladder_certificate_label_sequence_for_trmph_game,
    build_ladder_certificate_labels,
    build_ladder_certificate_sidecar_record_for_trmph_game,
    build_ladder_certificate_sidecar_records_for_trmph_file,
    ladder_certificate_plane_index,
    load_ladder_template_library,
)
from hex_ai.utils.ladder_templates.matcher import find_ladder_template_matches


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


def _full_rescan_labels(moves, templates):
    board = create_board_from_moves(moves)
    result = find_ladder_template_matches(
        board,
        templates,
        orientations=("red_bottom",),
    )
    return build_ladder_certificate_labels(result.matches, board_size=13)


def test_incremental_sequence_matches_full_rescan_and_preserves_persistent_match(
    tmp_path,
):
    _write_template(tmp_path)
    templates = load_ladder_template_library(tmp_path, source_set="generated")
    moves = [
        rowcol_to_trmph(0, 0, board_size=13),
        rowcol_to_trmph(3, 4, board_size=13),
        rowcol_to_trmph(0, 1, board_size=13),
    ]
    trmph_text = f"#13,{''.join(moves)}"

    sequence = build_ladder_certificate_label_sequence_for_trmph_game(
        trmph_text,
        templates,
        orientations=("red_bottom",),
    )

    for position in range(len(moves) + 1):
        expected = _full_rescan_labels(moves[:position], templates)
        np.testing.assert_array_equal(
            sequence.template_origin_maps[position],
            expected.template_origin_maps,
        )
        np.testing.assert_array_equal(
            sequence.carrier_maps[position],
            expected.carrier_maps,
        )

    plane_index = ladder_certificate_plane_index(orientation="red_bottom", row_class=2)
    assert sequence.template_origin_maps[2, plane_index, 2, 3] == 1
    assert sequence.template_origin_maps[3, plane_index, 2, 3] == 1
    assert sequence.position_stats[3].used_must_include_filter is True
    assert sequence.position_stats[3].scan_matches_found == 0
    assert sequence.position_stats[3].active_matches_after_update == 1


def test_incremental_sequence_drops_match_when_carrier_cell_changes(tmp_path):
    _write_template(tmp_path)
    templates = load_ladder_template_library(tmp_path, source_set="generated")
    moves = [
        rowcol_to_trmph(0, 0, board_size=13),
        rowcol_to_trmph(3, 4, board_size=13),
        rowcol_to_trmph(2, 4, board_size=13),
    ]
    trmph_text = f"#13,{''.join(moves)}"

    sequence = build_ladder_certificate_label_sequence_for_trmph_game(
        trmph_text,
        templates,
        orientations=("red_bottom",),
    )
    plane_index = ladder_certificate_plane_index(orientation="red_bottom", row_class=2)

    assert sequence.template_origin_maps[2, plane_index, 2, 3] == 1
    assert sequence.template_origin_maps[3, plane_index].sum() == 0
    assert sequence.position_stats[3].active_matches_after_update == 0


def test_ladder_sidecar_record_round_trip_preserves_dense_maps(tmp_path):
    _write_template(tmp_path)
    templates = load_ladder_template_library(tmp_path, source_set="generated")
    moves = [
        rowcol_to_trmph(0, 0, board_size=13),
        rowcol_to_trmph(3, 4, board_size=13),
        rowcol_to_trmph(0, 1, board_size=13),
    ]
    trmph_text = f"#13,{''.join(moves)}"

    sequence = build_ladder_certificate_label_sequence_for_trmph_game(
        trmph_text,
        templates,
        orientations=("red_bottom",),
    )
    record = build_ladder_certificate_sidecar_record_for_trmph_game(
        game_index=0,
        trmph_text=trmph_text,
        templates=templates,
        orientations=("red_bottom",),
    )

    np.testing.assert_array_equal(
        record.decode_template_origin_maps(),
        sequence.template_origin_maps,
    )
    np.testing.assert_array_equal(
        record.decode_carrier_maps(),
        sequence.carrier_maps,
    )
    assert record.used_must_include_filter_by_position == "0111"


def test_file_level_sidecar_builder_skips_winnerless_games_by_default(tmp_path):
    _write_template(tmp_path)
    templates = load_ladder_template_library(tmp_path, source_set="generated")
    winnerless_game = "#13,e4\n"
    winner_game = "#13,a1e4 r\n"
    trmph_file = tmp_path / "games.trmph"
    trmph_file.write_text(
        "noise header\n" + winnerless_game + winner_game,
        encoding="utf-8",
    )

    training_aligned = build_ladder_certificate_sidecar_records_for_trmph_file(
        trmph_file,
        templates,
        include_winnerless=False,
        orientations=("red_bottom",),
    )
    inclusive = build_ladder_certificate_sidecar_records_for_trmph_file(
        trmph_file,
        templates,
        include_winnerless=True,
        orientations=("red_bottom",),
    )

    assert len(training_aligned) == 1
    assert len(inclusive) == 2
    assert training_aligned[0].move_count == 2
    assert inclusive[0].move_count == 1
