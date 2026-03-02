import math

from hex_ai.selfplay.selfplay_engine import SelfPlayEngine
from hex_ai.virtual_board import (
    VIRTUAL_BOARD_PREFILL_MOVES,
    detect_virtual_prefill_display_board_size,
    get_virtual_prefill_move_coords,
    get_virtual_prefill_move_count,
    get_virtual_prefill_prefix_move_count_for_trmph,
)


def test_virtual_prefill_move_counts_match_coords_for_training_sizes():
    for size in range(6, 13):
        coords = get_virtual_prefill_move_coords(size)
        assert len(coords) == get_virtual_prefill_move_count(size)


def test_detect_virtual_prefill_display_board_size_from_prefix():
    bare_moves = VIRTUAL_BOARD_PREFILL_MOVES[10] + "a1b2c3"
    detected = detect_virtual_prefill_display_board_size(
        bare_moves,
        min_display_board_size=6,
        max_display_board_size=12,
    )
    assert detected == 10


def test_detect_virtual_prefill_display_board_size_returns_none_without_prefix():
    assert (
        detect_virtual_prefill_display_board_size(
            "a1b2c3",
            min_display_board_size=6,
            max_display_board_size=12,
        )
        is None
    )


def test_get_virtual_prefill_prefix_move_count_for_trmph_uses_known_prefix():
    trmph_text = f"#13,{VIRTUAL_BOARD_PREFILL_MOVES[12]}a1b1"
    assert get_virtual_prefill_prefix_move_count_for_trmph(
        trmph_text,
        min_display_board_size=6,
        max_display_board_size=12,
    ) == get_virtual_prefill_move_count(12)


def test_small_board_sampling_plan_is_linear_and_prefers_larger_sizes():
    sizes, weights = SelfPlayEngine._build_small_board_sampling_plan(6, 12)
    assert sizes == (6, 7, 8, 9, 10, 11, 12)
    assert weights == (1, 2, 3, 4, 5, 6, 7)
    assert math.isclose(weights[-1] / sum(weights), 0.25, rel_tol=0.0, abs_tol=1e-12)
