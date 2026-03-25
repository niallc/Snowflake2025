import json

from hex_ai.utils.format_conversion import rowcol_to_trmph
from hex_ai.utils.ladder_templates import (
    build_ladder_certificate_file_summary,
    build_ladder_certificate_sidecar_records_for_trmph_file,
    build_ladder_certificate_viewer_game_payload,
    load_ladder_template_library,
    write_ladder_certificate_sidecar,
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


def test_viewer_summary_uses_sidecar_match_counts(tmp_path):
    _write_template(tmp_path)
    templates = load_ladder_template_library(tmp_path, source_set="generated")

    moves = [
        rowcol_to_trmph(0, 0, board_size=13),
        rowcol_to_trmph(3, 4, board_size=13),
        rowcol_to_trmph(0, 1, board_size=13),
    ]
    trmph_file = tmp_path / "games.trmph"
    trmph_file.write_text(
        "#13,e4\n" + f"#13,{''.join(moves)} r\n",
        encoding="utf-8",
    )

    records = build_ladder_certificate_sidecar_records_for_trmph_file(
        trmph_file,
        templates,
        include_winnerless=False,
        orientations=("red_bottom",),
    )
    write_ladder_certificate_sidecar(
        trmph_file.with_suffix(".ladder_certificates.jsonl"),
        records,
    )

    summary = build_ladder_certificate_file_summary(
        trmph_file,
        include_winnerless=False,
    )

    assert summary["eligible_game_count"] == 1
    assert summary["sidecar_present"] is True
    assert summary["sidecar_aligned"] is True
    assert summary["games"][0]["positions_with_any_match"] == 2
    assert summary["games"][0]["max_active_matches"] == 1


def test_viewer_payload_marks_attacker_superset_cells_and_sidecar_consistency(
    tmp_path,
):
    _write_template(tmp_path)
    templates = load_ladder_template_library(tmp_path, source_set="generated")

    moves = [
        rowcol_to_trmph(0, 0, board_size=13),
        rowcol_to_trmph(3, 4, board_size=13),
        rowcol_to_trmph(0, 1, board_size=13),
        rowcol_to_trmph(2, 4, board_size=13),
    ]
    trmph_file = tmp_path / "games.trmph"
    trmph_file.write_text(f"#13,{''.join(moves)} r\n", encoding="utf-8")

    records = build_ladder_certificate_sidecar_records_for_trmph_file(
        trmph_file,
        templates,
        include_winnerless=False,
        orientations=("red_bottom",),
    )
    write_ladder_certificate_sidecar(
        trmph_file.with_suffix(".ladder_certificates.jsonl"),
        records,
    )

    payload = build_ladder_certificate_viewer_game_payload(
        trmph_file,
        game_index=0,
        templates=templates,
        include_winnerless=False,
        orientations=("red_bottom",),
    )

    final_position = payload["positions"][4]
    assert final_position["sidecar_consistency"]["using_sidecar_dense_labels"] is True
    assert final_position["sidecar_consistency"]["template_origin_equal_to_live"] is True
    assert final_position["sidecar_consistency"]["carrier_equal_to_live"] is True
    assert len(final_position["matches"]) >= 1
    assert any(
        match["attacker_superset_cells"] == [[2, 4]]
        for match in final_position["matches"]
    )

    plane = next(
        label
        for label in final_position["plane_labels"]
        if label["plane_name"] == "red_bottom_row2_template_origin"
    )
    assert [2, 3] in plane["origin_cells"]
    assert [2, 4] in plane["carrier_cells"]
