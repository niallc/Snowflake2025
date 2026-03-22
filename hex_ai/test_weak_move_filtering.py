from types import SimpleNamespace

import numpy as np

from hex_ai.enums import Piece
from hex_ai.inference.game_engine import HexGameState
from hex_ai.inference.mcts import BaselineMCTS
from hex_ai.utils.weaks_cells import (
    PolicyOrderedMoveFilterResult,
    _RING_OFFSETS,
    _RULE_D1,
    _RULE_V1,
    _RULE_V2,
    _RULE_V3,
    _WEAK_MOVE_STATUS_DEAD,
    _WEAK_MOVE_STATUS_SAFE,
    _WEAK_MOVE_STATUS_VULNERABLE,
    WeakMoveClassification,
    classify_weak_move,
    select_policy_ordered_weak_moves,
)


def _empty_board(size: int) -> np.ndarray:
    return np.full((size, size), Piece.EMPTY.value, dtype="<U1")


def _set_ring_tokens(
    board: np.ndarray,
    center: tuple[int, int],
    tokens: list[str],
) -> None:
    row, col = center
    for token, (dr, dc) in zip(tokens, _RING_OFFSETS):
        nr, nc = row + dr, col + dc
        if 0 <= nr < board.shape[0] and 0 <= nc < board.shape[1]:
            board[nr, nc] = token


def _classify_trmph_f4(trmph: str, player_color: str) -> WeakMoveClassification:
    state = HexGameState.from_trmph(trmph)
    return classify_weak_move(state.board, 3, 5, player_color=player_color)


def test_classify_weak_move_detects_dead_d1_on_interior_cell():
    board = _empty_board(7)
    center = (3, 3)
    _set_ring_tokens(
        board,
        center,
        [
            Piece.RED.value,
            Piece.RED.value,
            Piece.RED.value,
            Piece.RED.value,
            Piece.EMPTY.value,
            Piece.EMPTY.value,
        ],
    )

    classification = classify_weak_move(
        board,
        center[0],
        center[1],
        player_color=Piece.BLUE,
    )

    assert classification.status == _WEAK_MOVE_STATUS_DEAD
    assert classification.reasons == (_RULE_D1,)
    assert classification.vulnerable_reply_moves == ()


def test_classify_weak_move_detects_vulnerable_v1_completion():
    board = _empty_board(7)
    center = (3, 3)
    _set_ring_tokens(
        board,
        center,
        [
            Piece.BLUE.value,
            Piece.BLUE.value,
            Piece.EMPTY.value,
            Piece.BLUE.value,
            Piece.EMPTY.value,
            Piece.EMPTY.value,
        ],
    )

    classification = classify_weak_move(
        board,
        center[0],
        center[1],
        player_color=Piece.RED,
    )

    assert classification.status == _WEAK_MOVE_STATUS_VULNERABLE
    assert classification.reasons == (_RULE_V1,)
    assert classification.vulnerable_reply_moves == ((4, 2),)


def test_classify_weak_move_treats_edges_as_colored_neighbors():
    board = _empty_board(5)
    center = (2, 0)
    _set_ring_tokens(
        board,
        center,
        [
            Piece.EMPTY.value,
            Piece.EMPTY.value,
            Piece.EMPTY.value,
            Piece.EMPTY.value,
            Piece.RED.value,
            Piece.RED.value,
        ],
    )

    classification = classify_weak_move(
        board,
        center[0],
        center[1],
        player_color=Piece.BLUE,
    )

    assert classification.status == _WEAK_MOVE_STATUS_DEAD
    assert classification.reasons == (_RULE_D1,)


def test_example1_f4_is_vulnerable_for_red_and_blue():
    red = _classify_trmph_f4("#13,f3e5g3", "r")
    blue = _classify_trmph_f4("#13,f3e5g3", "b")

    assert red.status == _WEAK_MOVE_STATUS_VULNERABLE
    assert red.reasons == (_RULE_V3,)
    assert red.vulnerable_reply_moves == ((3, 6),)

    assert blue.status == _WEAK_MOVE_STATUS_VULNERABLE
    assert blue.reasons == (_RULE_V2,)
    assert blue.vulnerable_reply_moves == ((4, 5),)


def test_example2_f4_is_vulnerable_for_red_and_safe_for_blue():
    red = _classify_trmph_f4("#13,f3e5g4", "r")
    blue = _classify_trmph_f4("#13,f3e5g4", "b")

    assert red.status == _WEAK_MOVE_STATUS_VULNERABLE
    assert red.reasons == (_RULE_V3,)
    assert red.vulnerable_reply_moves == ((2, 6),)

    assert blue.status == _WEAK_MOVE_STATUS_SAFE
    assert blue.reasons == ()
    assert blue.vulnerable_reply_moves == ()


def test_example3_f4_is_v1_vulnerable_for_red_and_safe_for_blue():
    red = _classify_trmph_f4("#13,f3e11g4f11f5", "r")
    blue = _classify_trmph_f4("#13,f3e11g4f11f5", "b")

    assert red.status == _WEAK_MOVE_STATUS_VULNERABLE
    assert red.reasons == (_RULE_V1,)
    assert red.vulnerable_reply_moves == ((2, 6),)

    assert blue.status == _WEAK_MOVE_STATUS_SAFE
    assert blue.reasons == ()
    assert blue.vulnerable_reply_moves == ()


def test_select_policy_ordered_weak_moves_keeps_all_safe_and_filters_vulnerable_when_safe_exists():
    board = _empty_board(9)
    dead_move = (1, 1)
    vulnerable_move = (4, 4)
    safe_move = (7, 7)

    _set_ring_tokens(
        board,
        dead_move,
        [
            Piece.RED.value,
            Piece.RED.value,
            Piece.RED.value,
            Piece.RED.value,
            Piece.EMPTY.value,
            Piece.EMPTY.value,
        ],
    )
    _set_ring_tokens(
        board,
        vulnerable_move,
        [
            Piece.BLUE.value,
            Piece.BLUE.value,
            Piece.EMPTY.value,
            Piece.BLUE.value,
            Piece.EMPTY.value,
            Piece.EMPTY.value,
        ],
    )

    result = select_policy_ordered_weak_moves(
        board,
        [dead_move, vulnerable_move, safe_move],
        [0.9, 0.8, 0.7],
        player_color=Piece.RED,
    )
    assert result.keep_indices == (2,)
    assert result.dead_indices == (0,)
    assert result.vulnerable_indices == (1,)
    assert result.safe_indices == (2,)
    assert result.weak_filtered_indices == (0, 1)
    assert result.only_vulnerable_remaining is False
    assert result.classifications_by_index[2].status == _WEAK_MOVE_STATUS_SAFE


def test_select_policy_ordered_weak_moves_marks_only_vulnerable_remaining():
    board = _empty_board(7)
    vulnerable_move = (3, 3)
    _set_ring_tokens(
        board,
        vulnerable_move,
        [
            Piece.BLUE.value,
            Piece.BLUE.value,
            Piece.EMPTY.value,
            Piece.BLUE.value,
            Piece.EMPTY.value,
            Piece.EMPTY.value,
        ],
    )

    result = select_policy_ordered_weak_moves(
        board,
        [vulnerable_move],
        [0.5],
        player_color=Piece.RED,
    )

    assert result.keep_indices == (0,)
    assert result.safe_indices == ()
    assert result.vulnerable_indices == (0,)
    assert result.dead_indices == ()
    assert result.only_vulnerable_remaining is True


def test_root_weak_move_debug_logging_uses_pre_filter_legal_move_snapshot():
    mcts = object.__new__(BaselineMCTS)
    mcts.cfg = SimpleNamespace(
        dead_cell_debug_log_path="temp/test_root_weak_move_debug.jsonl",
        dead_cell_debug_max_records_per_move=10,
        dead_cell_enable_four_run=True,
        dead_cell_enable_two_two_split=True,
        dead_cell_enable_three_plus_one=True,
        dead_cell_enable_a1b2a3_discouraged=True,
        dead_cell_enable_double_dead_pairs=False,
        dead_cell_debug_strategy_label="unit_test_strategy",
    )
    mcts._dead_cell_debug_records_written = 0
    captured_events: list[dict] = []
    mcts._append_dead_cell_debug_event = lambda _path, event: captured_events.append(event)

    node = SimpleNamespace(
        legal_moves=[(7, 7)],
        board_size=9,
        state=SimpleNamespace(
            move_history=[(0, 0), (0, 1)],
            current_player_enum=SimpleNamespace(name="RED"),
            to_trmph=lambda: "#9,a1b1",
        ),
    )
    legal_moves_before = [(1, 1), (4, 4), (7, 7)]
    filter_result = PolicyOrderedMoveFilterResult(
        keep_indices=(2,),
        safe_indices=(2,),
        vulnerable_indices=(1,),
        dead_indices=(0,),
        weak_filtered_indices=(0, 1),
        only_vulnerable_remaining=False,
        classifications_by_index={
            0: WeakMoveClassification(
                status=_WEAK_MOVE_STATUS_DEAD,
                reasons=("D1",),
                vulnerable_reply_moves=(),
            ),
            1: WeakMoveClassification(
                status=_WEAK_MOVE_STATUS_VULNERABLE,
                reasons=("V1",),
                vulnerable_reply_moves=((5, 4),),
            ),
            2: WeakMoveClassification(
                status=_WEAK_MOVE_STATUS_SAFE,
                reasons=(),
                vulnerable_reply_moves=(),
            ),
        },
    )

    BaselineMCTS._log_root_weak_move_filter_event(
        mcts,
        node=node,
        legal_moves_before=legal_moves_before,
        filter_result=filter_result,
        legal_moves_before_count=len(legal_moves_before),
    )

    assert len(captured_events) == 2
    assert captured_events[0]["filtered_move"]["row"] == 1
    assert captured_events[0]["filtered_move"]["col"] == 1
    assert captured_events[0]["status"] == _WEAK_MOVE_STATUS_DEAD
    assert captured_events[1]["filtered_move"]["row"] == 4
    assert captured_events[1]["filtered_move"]["col"] == 4
    assert captured_events[1]["status"] == _WEAK_MOVE_STATUS_VULNERABLE
    assert captured_events[1]["vulnerable_reply_moves"][0]["row"] == 5
    assert captured_events[1]["vulnerable_reply_moves"][0]["col"] == 4
