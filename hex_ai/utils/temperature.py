"""
Temperature utility functions for the hex_ai library.
"""

from typing import Optional


def _require_positive_board_size(board_size: Optional[int], *, required: bool) -> Optional[int]:
    """Validate board_size when provided (or required)."""
    if board_size is None:
        if required:
            raise ValueError("board_size must be provided for game_progress temperature decay")
        return None
    if isinstance(board_size, bool):
        raise TypeError("board_size must be an integer, got bool")

    try:
        size = int(board_size)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"board_size must be an integer, got {type(board_size)}") from exc
    if size <= 0:
        raise ValueError(f"board_size must be positive, got {board_size}")
    return size


def calculate_temperature_decay(
    temperature_start: float,
    temperature_end: float,
    temperature_decay_type: str,
    temperature_decay_moves: int,
    temperature_step_thresholds: list[int],
    temperature_step_values: list[float],
    move_count: int,
    start_temp_override: Optional[float] = None,
    board_size: Optional[int] = None,
) -> float:
    """
    Calculate temperature based on decay configuration and current move count.
    
    Args:
        temperature_start: Starting temperature
        temperature_end: Final temperature (minimum)
        temperature_decay_type: Type of decay ("linear", "exponential", "step", "game_progress")
        temperature_decay_moves: Number of moves for decay (for linear/exponential)
        temperature_step_thresholds: Move thresholds for step decay
        temperature_step_values: Temperature values for step decay
        move_count: Number of moves played so far (0-based)
        start_temp_override: Optional override for starting temperature
        board_size: Explicit board size. Required for "game_progress" decay.
    
    Returns:
        Current temperature value
    """
    # Determine the starting temperature
    # Use override if provided, otherwise use temperature_start
    if start_temp_override is not None:
        start_temp = start_temp_override
    else:
        start_temp = temperature_start
    
    if temperature_decay_type == "linear":
        # Linear decay from temperature_start to temperature_end over temperature_decay_moves
        progress = min(move_count / max(1, temperature_decay_moves), 1.0)
        return start_temp + (temperature_end - start_temp) * progress
    
    elif temperature_decay_type == "exponential":
        # Exponential decay: T = T_start * (T_end/T_start)^(move_count/decay_moves)
        if start_temp <= 0 or temperature_end <= 0:
            return temperature_end  # Safety fallback
        progress = min(move_count / max(1, temperature_decay_moves), 1.0)
        decay_factor = (temperature_end / start_temp) ** progress
        return start_temp * decay_factor
    
    elif temperature_decay_type == "step":
        # Step decay: temperature drops at specific move thresholds
        if not temperature_step_thresholds or not temperature_step_values:
            return start_temp
        
        # Find the appropriate temperature for current move count
        for i, threshold in enumerate(temperature_step_thresholds):
            if move_count < threshold:
                return temperature_step_values[i] if i < len(temperature_step_values) else temperature_end
        
        # If we've passed all thresholds, use the final temperature
        return temperature_end
    
    elif temperature_decay_type == "game_progress":
        # Temperature based on percentage of game completed
        # Estimate total game length as board_size^2 (full board)
        resolved_board_size = _require_positive_board_size(board_size, required=True)
        estimated_total_moves = resolved_board_size * resolved_board_size
        progress = min(move_count / max(1, estimated_total_moves), 1.0)
        return start_temp + (temperature_end - start_temp) * progress
    
    else:
        # Unknown decay type, return starting temperature
        return start_temp
