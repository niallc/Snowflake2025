"""
Temperature utility functions for the hex_ai library.
"""

from typing import Optional
from hex_ai.config import BOARD_SIZE as CFG_BOARD_SIZE


def calculate_temperature_decay(
    temperature_start: float,
    temperature_end: float,
    temperature_decay_type: str,
    temperature_decay_moves: int,
    temperature_step_thresholds: list[int],
    temperature_step_values: list[float],
    move_count: int,
    start_temp_override: Optional[float] = None
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
        board_size = CFG_BOARD_SIZE
        estimated_total_moves = board_size * board_size
        progress = min(move_count / max(1, estimated_total_moves), 1.0)
        return start_temp + (temperature_end - start_temp) * progress
    
    else:
        # Unknown decay type, return starting temperature
        return start_temp
