"""Helpers for generating ladder-certificate targets from processed examples."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from hex_ai.data_utils import extract_games_from_file, remove_repeated_moves
from hex_ai.utils.format_conversion import (
    split_trmph_moves,
    strip_trmph_preamble,
    trmph_move_to_rowcol,
)

from .labeler import LadderCertificateLabels, build_ladder_certificate_labels
from .library_loader import LoadedLadderTemplate
from .matcher import BoardCoord, LadderMatchResult, find_ladder_template_matches


@dataclass(frozen=True)
class LadderExampleMatch:
    """Matcher output annotated with example-level last-move metadata."""

    match_result: LadderMatchResult
    last_move: BoardCoord | None
    used_last_move_filter: bool
    last_move_lookup_error: str | None


@dataclass(frozen=True)
class LadderExampleTarget:
    """Dense ladder-certificate labels plus example-level match metadata."""

    labels: LadderCertificateLabels
    match: LadderExampleMatch


def resolve_last_move_for_training_example(
    example: dict,
    *,
    strict: bool = True,
) -> BoardCoord | None:
    """Resolve the last played move for one processed training example."""
    metadata = example.get("metadata")
    if not isinstance(metadata, dict):
        return _handle_last_move_resolution_error(
            "Example is missing metadata dictionary.",
            strict=strict,
        )

    position_in_game = metadata.get("position_in_game")
    if position_in_game is None:
        return _handle_last_move_resolution_error(
            "Example metadata is missing position_in_game.",
            strict=strict,
        )
    position_in_game = int(position_in_game)
    if position_in_game < 0:
        return _handle_last_move_resolution_error(
            f"position_in_game must be >= 0, got {position_in_game}",
            strict=strict,
        )
    if position_in_game == 0:
        return None

    source_file = metadata.get("source_file")
    if not source_file:
        return _handle_last_move_resolution_error(
            "Example metadata is missing source_file.",
            strict=strict,
        )
    source_path = Path(str(source_file))
    if not source_path.exists():
        return _handle_last_move_resolution_error(
            f"Source file does not exist: {source_path}",
            strict=strict,
        )

    game_id = metadata.get("game_id")
    if (
        not isinstance(game_id, tuple)
        or len(game_id) != 2
        or not isinstance(game_id[1], int)
    ):
        return _handle_last_move_resolution_error(
            f"Example metadata has invalid game_id: {game_id!r}",
            strict=strict,
        )
    line_idx = int(game_id[1])

    games = _load_games_from_source_file(str(source_path))
    if not (0 <= line_idx < len(games)):
        return _handle_last_move_resolution_error(
            f"game_id line index {line_idx} out of range for {source_path} "
            f"(games={len(games)})",
            strict=strict,
        )

    game_line = str(games[line_idx]).split()[0]
    moves = remove_repeated_moves(
        split_trmph_moves(strip_trmph_preamble(game_line))
    )
    if moves is None:
        return _handle_last_move_resolution_error(
            f"Duplicate moves encountered in source game {source_path}:{line_idx}",
            strict=strict,
        )
    if position_in_game > len(moves):
        return _handle_last_move_resolution_error(
            f"position_in_game {position_in_game} exceeds move count {len(moves)} "
            f"for {source_path}:{line_idx}",
            strict=strict,
        )

    board = example.get("board")
    if board is None or not hasattr(board, "shape") or len(board.shape) < 2:
        return _handle_last_move_resolution_error(
            "Example is missing a usable board array for board-size inference.",
            strict=strict,
        )
    board_size = int(board.shape[-1])
    move = moves[position_in_game - 1]
    return trmph_move_to_rowcol(move, board_size=board_size)


def find_ladder_matches_for_training_example(
    example: dict,
    templates: tuple[LoadedLadderTemplate, ...] | list[LoadedLadderTemplate],
    *,
    orientations: tuple[str, ...] = ("red_bottom", "blue_right"),
    use_last_move_filter: bool = True,
    allow_last_move_lookup_fallback: bool = False,
    allow_attacker_superset_on_empty: bool = True,
) -> LadderExampleMatch:
    """Match ladder templates against one processed training example."""
    last_move: BoardCoord | None = None
    used_last_move_filter = False
    last_move_lookup_error: str | None = None

    if use_last_move_filter:
        try:
            last_move = resolve_last_move_for_training_example(example, strict=True)
            used_last_move_filter = last_move is not None
        except Exception as exc:
            if not allow_last_move_lookup_fallback:
                raise
            last_move_lookup_error = str(exc)
            last_move = None
            used_last_move_filter = False

    match_result = find_ladder_template_matches(
        example["board"],
        templates,
        orientations=orientations,
        allow_attacker_superset_on_empty=allow_attacker_superset_on_empty,
        must_include_cell=last_move if used_last_move_filter else None,
    )
    return LadderExampleMatch(
        match_result=match_result,
        last_move=last_move,
        used_last_move_filter=used_last_move_filter,
        last_move_lookup_error=last_move_lookup_error,
    )


def build_ladder_certificate_target_for_training_example(
    example: dict,
    templates: tuple[LoadedLadderTemplate, ...] | list[LoadedLadderTemplate],
    *,
    orientations: tuple[str, ...] = ("red_bottom", "blue_right"),
    use_last_move_filter: bool = True,
    allow_last_move_lookup_fallback: bool = False,
    allow_attacker_superset_on_empty: bool = True,
) -> LadderExampleTarget:
    """Build dense ladder-certificate labels for one processed training example."""
    example_match = find_ladder_matches_for_training_example(
        example,
        templates,
        orientations=orientations,
        use_last_move_filter=use_last_move_filter,
        allow_last_move_lookup_fallback=allow_last_move_lookup_fallback,
        allow_attacker_superset_on_empty=allow_attacker_superset_on_empty,
    )
    board_size = int(example["board"].shape[-1])
    labels = build_ladder_certificate_labels(
        example_match.match_result.matches,
        board_size=board_size,
    )
    return LadderExampleTarget(labels=labels, match=example_match)


@lru_cache(maxsize=256)
def _load_games_from_source_file(source_file: str) -> tuple[str, ...]:
    return tuple(extract_games_from_file(Path(source_file)))


def _handle_last_move_resolution_error(
    message: str,
    *,
    strict: bool,
) -> BoardCoord | None:
    if strict:
        raise ValueError(message)
    return None
