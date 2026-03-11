"""Shared validation and state-construction helpers for interactive web endpoints."""

from __future__ import annotations

import re
import string
from typing import Callable, Iterable

import hex_ai.utils.format_conversion as fc
from hex_ai.inference.game_engine import HexGameState


def build_whitelisted_trmph_url_prefixes(
    *,
    board_size: int | None = None,
    board_sizes: Iterable[int] | None = None,
) -> tuple[str, ...]:
    """Return the exact full-URL prefixes accepted for TRMPH board links."""
    sizes = []
    if board_sizes is not None:
        sizes.extend(int(size) for size in board_sizes)
    if board_size is not None:
        sizes.append(int(board_size))
    if not sizes:
        raise ValueError("At least one board size is required")

    unique_sizes = tuple(dict.fromkeys(sizes))
    prefixes = []
    for size in unique_sizes:
        prefixes.extend(
            (
                f"https://trmph.com/hex/board#{size},",
                f"http://trmph.com/hex/board#{size},",
                f"trmph.com/hex/board#{size},",
            )
        )
    return tuple(prefixes)


def normalize_game_input_with_exact_trmph_url_whitelist(
    text: str,
    *,
    board_size: int,
    allowed_url_prefixes: Iterable[str] | None = None,
) -> str:
    """
    Normalize move input while only allowing full TRMPH links from exact prefixes.

    This preserves the existing bare-move and `#<size>,` flows, but prevents generic
    URL-like strings from being cleaned into something that accidentally parses.
    """
    prefixes = tuple(allowed_url_prefixes or build_whitelisted_trmph_url_prefixes(board_size=board_size))

    if text is None:
        return ""
    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    stripped = text.strip()
    for prefix in prefixes:
        if stripped.startswith(prefix):
            stripped = f"#{board_size},{stripped[len(prefix):]}"
            break
    else:
        if "://" in stripped or stripped.startswith("www.") or stripped.startswith("trmph.com/"):
            raise ValueError(
                "Unsupported URL format. Only exact TRMPH URL prefixes are allowed: "
                + ", ".join(prefixes)
            )

    return fc.normalize_game_input(stripped, board_size=board_size)


def validate_trmph_input(trmph_string, *, board_size: int):
    """
    Validate TRMPH input format.

    Returns:
        tuple[bool, str | None]: (is_valid, error_message)
    """
    if trmph_string is None:
        trmph_string = ""

    if not isinstance(trmph_string, str):
        return False, "TRMPH input must be a string"

    trmph_string = trmph_string.strip()
    if not trmph_string:
        return True, None

    max_col = string.ascii_lowercase[board_size - 1]
    if board_size <= 9:
        number_pattern = f"[1-{board_size}]"
    else:
        number_pattern = f"(1[0-{board_size % 10}]|[1-9])"

    trmph_pattern_with_prefix = re.compile(f"^#{board_size},([a-{max_col}]{number_pattern})+$")
    trmph_pattern_without_prefix = re.compile(f"^([a-{max_col}]{number_pattern})+$")

    if not (trmph_pattern_with_prefix.match(trmph_string) or trmph_pattern_without_prefix.match(trmph_string)):
        return (
            False,
            f"Invalid TRMPH format. Only letters a-{max_col} followed by numbers 1-{board_size} "
            f"are allowed (e.g., a1b2c3 or #{board_size},a1b2c3)",
        )

    try:
        prefix = f"#{board_size},"
        bare_moves = trmph_string[len(prefix):] if trmph_string.startswith(prefix) else trmph_string
        moves = fc.split_trmph_moves(bare_moves)
        max_moves = board_size * board_size
        if len(moves) > max_moves:
            return False, f"Too many moves (maximum {max_moves} moves for a complete game)"
    except ValueError as exc:
        return False, f"Invalid TRMPH format: {exc}"

    return True, None


def validate_elo_rating(value, *, min_elo: int, max_elo: int) -> int:
    """Validate and convert ELO rating to an integer in range."""
    if value is None:
        raise ValueError("ELO rating is required and cannot be null")

    try:
        elo_float = float(value)
        elo_int = int(round(elo_float))
    except (ValueError, TypeError) as exc:
        raise ValueError("ELO rating must be a number") from exc

    if not (min_elo <= elo_int <= max_elo):
        raise ValueError(f"ELO rating must be between {min_elo} and {max_elo}")
    return elo_int


def validate_boolean_flag(value, field_name: str):
    """Validate and normalize a boolean feature flag from JSON input."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean (or one of true/false, 1/0)")


def validate_api_input(
    data,
    required_fields=None,
    optional_fields=None,
    *,
    logger=None,
    reject_unexpected: bool = True,
    normalize_game_input_fn: Callable[[str], str] = fc.normalize_game_input,
    trmph_validator: Callable[[str], tuple[bool, str | None]] | None = None,
    display_board_size_field: str = "display_board_size",
    default_display_board_size=None,
    display_board_size_validator: Callable[[object], int] | None = None,
    normalize_trmph_for_display_fn: Callable[[str, int, str], str] | None = None,
    move_in_display_validator: Callable[[str, int], None] | None = None,
    elo_field: str = "elo_rating",
    elo_validator: Callable[[object], int] | None = None,
    boolean_fields: Iterable[str] | None = None,
):
    """
    Validate API input with configurable strictness and normalization strategy.

    Returns:
        tuple[bool, str | None, dict | None]: (is_valid, error_message, validated_data)
    """
    if not isinstance(data, dict):
        return False, "Request data must be a JSON object", None

    all_allowed = set(required_fields or []) | set(optional_fields or [])
    if reject_unexpected:
        unexpected = set(data.keys()) - all_allowed
        if unexpected:
            return False, f"Unexpected fields: {list(unexpected)}", None

    validated_data = {}
    display_board_size = default_display_board_size
    if (
        display_board_size_validator is not None
        and display_board_size_field in all_allowed
    ):
        try:
            display_board_size = display_board_size_validator(
                data.get(display_board_size_field, default_display_board_size)
            )
            validated_data[display_board_size_field] = display_board_size
        except ValueError as exc:
            if logger is not None:
                logger.warning("Validation failed for %s: %s", display_board_size_field, exc)
            return False, f"Invalid {display_board_size_field}: {exc}", None

    boolean_field_set = set(boolean_fields or [])
    processed_fields = all_allowed if reject_unexpected else set(data.keys()) | all_allowed
    for field in processed_fields:
        if field not in data or field == display_board_size_field:
            continue

        if field in {"trmph", "move", "trmph_sequence"}:
            try:
                normalized_input = normalize_game_input_fn(data[field])
            except ValueError as exc:
                if logger is not None:
                    logger.warning("Normalization failed for %s: %s", field, exc)
                return False, f"Invalid format for {field}: {exc}", None
            except Exception as exc:  # pragma: no cover - defense-in-depth
                if logger is not None:
                    logger.error("Input normalization failed for %s: %s", field, exc)
                return False, f"Normalization error for {field}: {exc}", None

            validator = trmph_validator
            if validator is None:
                validator = lambda value: validate_trmph_input(value, board_size=13)
            is_valid, error_msg = validator(normalized_input)
            if not is_valid:
                return False, f"Invalid {field} [DEBUG-CHECK]: {error_msg}", None

            try:
                if (
                    field in {"trmph", "trmph_sequence"}
                    and normalize_trmph_for_display_fn is not None
                    and display_board_size is not None
                ):
                    validated_data[field] = normalize_trmph_for_display_fn(
                        normalized_input,
                        display_board_size,
                        field,
                    )
                elif (
                    field == "move"
                    and move_in_display_validator is not None
                    and display_board_size is not None
                ):
                    move_in_display_validator(normalized_input, display_board_size)
                    validated_data[field] = normalized_input
                else:
                    validated_data[field] = normalized_input
            except ValueError as exc:
                if logger is not None:
                    logger.warning("Display-board validation failed for %s: %s", field, exc)
                return False, f"Invalid {field}: {exc}", None
        elif field == elo_field and elo_validator is not None:
            try:
                validated_data[field] = elo_validator(data[field])
            except ValueError as exc:
                if logger is not None:
                    logger.warning("Validation failed for %s: %s", field, exc)
                return False, f"Invalid {field}: {exc}", None
        elif field in boolean_field_set:
            try:
                validated_data[field] = validate_boolean_flag(data[field], field)
            except ValueError as exc:
                if logger is not None:
                    logger.warning("Validation failed for %s: %s", field, exc)
                return False, f"Invalid {field}: {exc}", None
        else:
            validated_data[field] = data[field]

    for field in required_fields or []:
        if field not in validated_data:
            return False, f"Missing required field: {field}", None

    return True, None, validated_data


def create_game_state_from_trmph_input(
    trmph,
    *,
    context: str = "",
    display_board_size=None,
    display_board_size_validator: Callable[[object], int] | None = None,
    normalize_user_trmph_fn: Callable[[str, int, str], str] | None = None,
    compose_full_trmph_fn: Callable[[str, int], str] | None = None,
    apply_display_mask_fn: Callable[[HexGameState, int], HexGameState] | None = None,
):
    """
    Create a HexGameState from endpoint TRMPH input.

    This supports both plain full-board TRMPH parsing and virtual-board
    display-size translation workflows.
    """
    context_msg = f" {context}" if context else ""

    if normalize_user_trmph_fn is None or compose_full_trmph_fn is None:
        try:
            return HexGameState.from_trmph((trmph or "").strip())
        except Exception as exc:
            raise ValueError(f"Invalid TRMPH{context_msg}: {exc}") from exc

    if display_board_size_validator is not None:
        display_board_size = display_board_size_validator(display_board_size)
    if display_board_size is None:
        raise ValueError("display_board_size is required for virtual-board TRMPH parsing")

    user_bare_moves = normalize_user_trmph_fn(trmph, display_board_size, field_name="trmph")
    full_trmph = compose_full_trmph_fn(user_bare_moves, display_board_size)

    try:
        state = HexGameState.from_trmph(full_trmph)
    except Exception as exc:
        raise ValueError(
            f"Invalid TRMPH{context_msg} for display_board_size={display_board_size}: {exc}"
        ) from exc

    if apply_display_mask_fn is not None:
        state = apply_display_mask_fn(state, display_board_size)
    return state
