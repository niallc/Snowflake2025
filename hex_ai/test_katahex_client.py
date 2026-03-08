from hex_ai.inference.katahex_client import (
    build_katahex_override_config,
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


def test_katahex_override_config_omits_num_eigen_threads_by_default():
    override = build_katahex_override_config(
        max_visits=4,
        num_search_threads=3,
        nn_cache_size_power_of_two=15,
    )
    assert "numSearchThreads=3" in override
    assert "numEigenThreadsPerModel" not in override


def test_katahex_override_config_includes_num_eigen_threads_when_requested():
    override = build_katahex_override_config(
        max_visits=4,
        num_search_threads=3,
        num_eigen_threads=2,
        nn_cache_size_power_of_two=15,
    )
    assert "numSearchThreads=3" in override
    assert "numEigenThreadsPerModel=2" in override
