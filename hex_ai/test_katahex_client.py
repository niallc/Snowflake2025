from hex_ai.inference.katahex_client import (
    katahex_vertex_to_snowflake_rowcol,
    snowflake_rowcol_to_katahex_vertex,
)


def test_katahex_vertex_mapping_uses_top_left_origin():
    assert snowflake_rowcol_to_katahex_vertex(0, 0, board_size=13) == "A1"
    assert snowflake_rowcol_to_katahex_vertex(12, 12, board_size=13) == "M13"


def test_katahex_vertex_mapping_round_trips_without_vertical_flip():
    for row, col in ((0, 0), (0, 12), (3, 9), (12, 0), (12, 12)):
        vertex = snowflake_rowcol_to_katahex_vertex(row, col, board_size=13)
        assert katahex_vertex_to_snowflake_rowcol(vertex, board_size=13) == (row, col)
