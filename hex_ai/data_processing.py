"""
Data processing utilities for Hex AI.

This module provides utilities for processing TRMPH game data into training-ready formats.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import torch
import gzip
import pickle
import numpy as np
from tqdm import tqdm

from .config import BOARD_SIZE, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE, TRMPH_BLUE_WIN, TRMPH_RED_WIN
from hex_ai.utils.format_conversion import trmph_to_moves
from hex_ai.value_utils import trmph_winner_to_training_value
from hex_ai.enums import Player

logger = logging.getLogger(__name__)


def parse_trmph_line_flexible(line: str) -> Tuple[str, Optional[str]]:
    """
    Parse a TRMPH line that can be in any of these formats:
    1. Old format: "http://www.trmph.com/hex/board#13,moves b"
    2. New format: "#13,moves" (no winner indicator)
    3. Tournament format: "#13,moves winner" (with winner indicator)
    
    Args:
        line: TRMPH line to parse
        
    Returns:
        Tuple of (trmph_string, winner_indicator) where winner_indicator is None for new format
        
    Raises:
        ValueError: If the line format is invalid
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty line")
    
    # Handle new format: "#13,moves" (no winner indicator)
    if line.startswith('#13,'):
        # Remove any inline comments (everything after # that's not part of the game)
        if ' # ' in line:
            line = line.split(' # ')[0].strip()
        
        # Check if there's a winner indicator at the end
        parts = line.split()
        if len(parts) == 2:
            # Tournament format: "#13,moves winner"
            trmph_string, winner_indicator = parts
            if winner_indicator in ['b', 'r']:
                return trmph_string, winner_indicator
            else:
                raise ValueError(f"Invalid winner indicator '{winner_indicator}' in tournament format")
        elif len(parts) == 1:
            # New format: "#13,moves" (no winner indicator)
            return line, None
        else:
            raise ValueError(f"Invalid #13 format: expected 1 or 2 parts, got {len(parts)}")
    
    # Handle old format: "http://www.trmph.com/hex/board#13,moves b"
    elif line.startswith('http://www.trmph.com/hex/board#'):
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Invalid old format TRMPH line: expected 2 parts, got {len(parts)}")
        trmph_url, winner_indicator = parts
        
        # Check for legacy formats and raise exceptions
        if winner_indicator == "1":
            raise ValueError(f"Legacy TRMPH_BLUE_WIN value ('1') detected in line: {repr(line)}. Use new format ('b') instead.")
        elif winner_indicator == "2":
            raise ValueError(f"Legacy TRMPH_RED_WIN value ('2') detected in line: {repr(line)}. Use new format ('r') instead.")
        
        if winner_indicator not in {TRMPH_BLUE_WIN, TRMPH_RED_WIN}:
            raise ValueError(f"Invalid winner indicator: {winner_indicator} in line: {repr(line)}")
        
        return trmph_url, winner_indicator
    
    else:
        raise ValueError(f"Unrecognized TRMPH format: {line}")


def extract_games_from_file_flexible(file_path: Path) -> List[Tuple[str, Optional[str]]]:
    """
    Extract game lines from a TRMPH file, handling both old and new formats.
    
    Args:
        file_path: Path to the TRMPH file
        
    Returns:
        List of tuples (trmph_string, winner_indicator) where winner_indicator is None for new format
    """
    games = []
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                trmph_string, winner_indicator = parse_trmph_line_flexible(line)
                games.append((trmph_string, winner_indicator))
            except ValueError as e:
                logger.warning(f"Skipping invalid line {line_num}: {e}")
                continue
    
    return games


def parse_trmph_to_gamerecord(trmph_string: str, winner_indicator: Optional[str] = None):
    """
    Parse a TRMPH string into a GameRecord object.
    
    Args:
        trmph_string: TRMPH format game string (with or without preamble)
        winner_indicator: Optional winner indicator ('b' or 'r')
        
    Returns:
        GameRecord object
    """
    # Import here to avoid circular imports
    from hex_ai.eval.strength_evaluator import GameRecord
    
    # Parse moves from TRMPH string
    moves = trmph_to_moves(trmph_string, BOARD_SIZE)
    
    # Convert to GameRecord format
    game_moves = []
    for i, (row, col) in enumerate(moves):
        player = Player.BLUE if i % 2 == 0 else Player.RED
        game_moves.append((row, col, player))
    
    return GameRecord(
        board_size=BOARD_SIZE,
        moves=game_moves,
        starting_player=Player.BLUE,
        metadata={"source": "trmph", "trmph_string": trmph_string, "winner_indicator": winner_indicator}
    )

