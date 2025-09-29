"""
Game execution utilities for tournaments.

This module provides functions for executing games between strategies,
including opening position generation and deterministic game play.
"""

import glob
import json
import logging
import os
import random
import time
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from hex_ai.config import (
    BOARD_SIZE, EMPTY_PIECE, TRMPH_BLUE_WIN, TRMPH_RED_WIN, TRMPH_PREFIX
)
from hex_ai.data_processing import parse_trmph_line_flexible
from hex_ai.enums import Player, Piece
from hex_ai.inference.game_engine import HexGameState, apply_move_to_state
from hex_ai.inference.move_selection import get_strategy, MoveSelectionConfig
from hex_ai.inference.strategy_config import StrategyConfig
from hex_ai.utils.format_conversion import (
    rowcol_to_trmph, trmph_to_moves
)

logger = logging.getLogger(__name__)

# Constants
DEFAULT_OPENING_LENGTH = 5
DEFAULT_TEMPERATURE = 0.0
TRMPH_SOURCE_DIR = "data/sf25/sep28"
TRMPH_FILE_PATTERN = "*.trmph"


class OpeningPosition:
    """Represents an opening position with moves and metadata."""
    
    def __init__(self, moves: List[Tuple[int, int]], source_game: str = "", 
                 opening_length: int = DEFAULT_OPENING_LENGTH):
        self.moves = moves
        self.source_game = source_game
        self.opening_length = opening_length
    
    def get_state(self, board_size: int = BOARD_SIZE) -> HexGameState:
        """Create a game state from this opening position."""
        # Initialize empty board
        board = np.full((board_size, board_size), EMPTY_PIECE, dtype='U1')
        state = HexGameState(board=board, _current_player=Player.BLUE)
        
        # Apply the opening moves
        for row, col in self.moves:
            state = apply_move_to_state(state, row, col)
        
        return state
    
    def get_trmph_string(self, board_size: int = BOARD_SIZE) -> str:
        """Get TRMPH representation of the opening moves."""
        trmph_moves = ''.join([rowcol_to_trmph(r, c, board_size) for r, c in self.moves])
        return f"{TRMPH_PREFIX}{trmph_moves}"
    
    def __str__(self) -> str:
        return f"Opening({len(self.moves)} moves from {self.source_game})"


def get_move_config_for_strategy(strategy_config: StrategyConfig, global_temperature: float = DEFAULT_TEMPERATURE) -> MoveSelectionConfig:
    """Create a MoveSelectionConfig for a strategy with specified temperature."""
    config_dict = strategy_config.config.copy()
    # Use strategy-specific temperature if available, otherwise use global temperature
    temperature = strategy_config.temperature if strategy_config.temperature is not None else global_temperature
    config_dict['temperature'] = temperature
    return MoveSelectionConfig(**config_dict)


def extract_openings_from_trmph_file(file_path: str, opening_length: int = DEFAULT_OPENING_LENGTH, 
                                   max_openings: int = 500) -> List[OpeningPosition]:
    """
    Extract diverse opening positions from a TRMPH file.
    
    Args:
        file_path: Path to TRMPH file
        opening_length: Number of moves to extract for each opening
        max_openings: Maximum number of openings to extract
    
    Returns:
        List of OpeningPosition objects
    
    Raises:
        ValueError: If file format is invalid or moves are malformed
    """
    openings = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            if len(openings) >= max_openings:
                break
            
            line = line.strip()
            if not line or not line.startswith(TRMPH_PREFIX):
                continue
            
            # Parse TRMPH line using centralized utility
            try:
                trmph_string, winner_indicator = parse_trmph_line_flexible(line)
                
                # Skip lines without winner indicator (we need completed games for openings)
                if winner_indicator is None:
                    continue
                
                # Convert TRMPH moves to row,col coordinates using centralized utility
                try:
                    moves = trmph_to_moves(trmph_string, BOARD_SIZE)
                except ValueError as e:
                    logger.warning(f"Could not parse moves in line {line_num}: {e}")
                    continue
                
                # Only use openings with enough moves
                if len(moves) >= opening_length:
                    opening_moves = moves[:opening_length]
                    
                    # Check for duplicate moves within the opening
                    unique_moves = set(opening_moves)
                    if len(unique_moves) == len(opening_moves):
                        # No duplicates within this opening
                        source_game = f"{os.path.basename(file_path)}:line{line_num}"
                        openings.append(OpeningPosition(opening_moves, source_game, opening_length))
                    else:
                        logger.warning(f"Skipping opening with duplicate moves in line {line_num}: {opening_moves}")
                
            except Exception as e:
                logger.warning(f"Could not parse line {line_num} in {file_path}: {e}")
                continue
    
    return openings


def find_trmph_files(source_dir: str) -> List[str]:
    """Find all TRMPH files in the source directory."""
    pattern = os.path.join(source_dir, TRMPH_FILE_PATTERN)
    files = glob.glob(pattern)
    logger.info(f"Found {len(files)} TRMPH files in {source_dir}")
    return sorted(files)


def generate_diverse_openings(trmph_files: List[str], opening_length: int = DEFAULT_OPENING_LENGTH,
                            target_count: int = 500, cache_file: str = None) -> List[OpeningPosition]:
    """
    Generate diverse opening positions from multiple TRMPH files.
    
    This function ensures uniqueness by checking each opening against previously
    collected ones before adding it to the list.
    
    Args:
        trmph_files: List of TRMPH file paths
        opening_length: Number of moves per opening
        target_count: Target number of openings to generate
        cache_file: Optional file to save/load openings for faster subsequent runs
    
    Returns:
        List of diverse OpeningPosition objects
    """
    # Try to load from cache first
    if cache_file and os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                if (cached_data.get('opening_length') == opening_length and 
                    len(cached_data.get('openings', [])) >= target_count):
                    logger.info(f"Loading {target_count} openings from cache: {cache_file}")
                    openings = []
                    for i, opening_data in enumerate(cached_data['openings'][:target_count]):
                        opening = OpeningPosition(
                            moves=opening_data['moves'],
                            source_game=opening_data['source'],
                            opening_length=opening_length
                        )
                        openings.append(opening)
                    return openings
        except Exception as e:
            logger.warning(f"Could not load cache file {cache_file}: {e}")
    
    logger.info(f"Generating {target_count} unique openings...")
    
    # Set to track unique opening move sequences
    unique_openings = set()
    diverse_openings = []
    
    # Process files until we have enough unique openings
    for file_path in trmph_files:
        if len(diverse_openings) >= target_count:
            break
            
        if not os.path.exists(file_path):
            continue
            
        logger.info(f"Processing {os.path.basename(file_path)}...")
        
        # Extract all openings from this file
        file_openings = extract_openings_from_trmph_file(
            file_path, opening_length, max_openings=1000  # Extract many to find unique ones
        )
        
        # Check each opening for uniqueness
        for opening in file_openings:
            if len(diverse_openings) >= target_count:
                break
                
            # Create a tuple of moves for comparison (tuples are hashable)
            moves_tuple = tuple(opening.moves)
            
            if moves_tuple not in unique_openings:
                unique_openings.add(moves_tuple)
                diverse_openings.append(opening)
                
                if len(diverse_openings) % 50 == 0:
                    logger.info(f"  Found {len(diverse_openings)} unique openings so far...")
    
    logger.info(f"Generated {len(diverse_openings)} unique openings from {len(trmph_files)} files")
    
    # Save to cache if requested
    if cache_file and diverse_openings:
        try:
            cache_data = {
                'opening_length': opening_length,
                'openings': [
                    {
                        'moves': opening.moves,
                        'source': opening.source_game
                    }
                    for opening in diverse_openings
                ]
            }
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.info(f"Saved {len(diverse_openings)} openings to cache: {cache_file}")
        except Exception as e:
            logger.warning(f"Could not save cache file {cache_file}: {e}")
    
    return diverse_openings


def play_deterministic_game(
    model_cache,
    strategy_a: StrategyConfig,
    strategy_b: StrategyConfig,
    opening: OpeningPosition,
    temperature: float = DEFAULT_TEMPERATURE,
    board_size: int = BOARD_SIZE,
    verbose: int = 0,
    strategy_a_is_blue: bool = True
) -> Dict[str, Any]:
    """
    Play a deterministic game from an opening position.
    
    Args:
        model_cache: Model cache to get models for each strategy
        strategy_a: Strategy configuration for player A
        strategy_b: Strategy configuration for player B
        opening: Opening position to start from
        temperature: Temperature for move selection (0.0 = deterministic)
        board_size: Board size for the game
        verbose: Verbosity level
        strategy_a_is_blue: Whether strategy_a plays as Blue (True) or Red (False)
    
    Returns:
        Dictionary with game results including timing information
    """
    # Initialize timing tracking
    strategy_timings = {
        strategy_a.name: 0.0,
        strategy_b.name: 0.0
    }
    move_count = 0
    
    # Start from the opening position
    state = opening.get_state(board_size)
    
    # Create strategy configurations with specified temperature
    config_a = get_move_config_for_strategy(strategy_a, temperature)
    config_b = get_move_config_for_strategy(strategy_b, temperature)
    
    # Get strategy objects
    strategy_a_obj = get_strategy(strategy_a.strategy_type)
    strategy_b_obj = get_strategy(strategy_b.strategy_type)
    
    # Play the game from the opening position
    move_sequence = list(opening.moves)  # Start with opening moves
    
    logger.debug(f"Starting game: {strategy_a.name} vs {strategy_b.name}")
    logger.debug(f"Opening moves: {opening.moves}")
    logger.debug(f"Initial state current player: {state.current_player_enum}")
    logger.debug(f"Strategy A is Blue: {strategy_a_is_blue}")
    
    while not state.game_over:
        # Determine which strategy to use based on current player and color assignment
        current_player = state.current_player_enum
        if current_player == Player.BLUE:
            if strategy_a_is_blue:
                strategy_obj = strategy_a_obj
                strategy_config = config_a
                strategy_name = strategy_a.name
                model = model_cache.get_simple_model(strategy_a.model_path)
            else:
                strategy_obj = strategy_b_obj
                strategy_config = config_b
                strategy_name = strategy_b.name
                model = model_cache.get_simple_model(strategy_b.model_path)
        else:  # Player.RED
            if strategy_a_is_blue:
                strategy_obj = strategy_b_obj
                strategy_config = config_b
                strategy_name = strategy_b.name
                model = model_cache.get_simple_model(strategy_b.model_path)
            else:
                strategy_obj = strategy_a_obj
                strategy_config = config_a
                strategy_name = strategy_a.name
                model = model_cache.get_simple_model(strategy_a.model_path)
    
        # Time the move selection
        start_time = time.perf_counter()
        move = strategy_obj.select_move(state, model, strategy_config, verbose=verbose)
        end_time = time.perf_counter()
        
        if move is None:
            raise ValueError(
                f"Move selection returned None for {strategy_obj.get_name()}. "
                f"This indicates a bug in the strategy implementation. "
                f"Please check the strategy code and ensure it always returns a valid move."
            )
        
        # Record timing for this strategy
        move_time = end_time - start_time
        strategy_timings[strategy_name] += move_time
        move_count += 1
        
        logger.debug(f"Player {current_player.name} ({strategy_name}) plays move {move} in {move_time:.3f}s")
        
        # Apply move
        move_sequence.append(move)
        state = apply_move_to_state(state, *move)
        
        if verbose >= 2:
            print("-", end="", flush=True)
    
    # Convert to TRMPH format
    trmph_moves = ''.join([rowcol_to_trmph(r, c, board_size) for r, c in move_sequence])
    trmph_str = f"{TRMPH_PREFIX}{trmph_moves}"
    
    # Determine winner
    winner_enum = state.winner_enum
    if winner_enum is None:
        raise ValueError(
            "Game is not over or winner missing. "
            "This indicates a bug in the game engine. "
            "Please check the game state and ensure the game has properly ended."
        )
    
    if winner_enum.name == 'BLUE':
        winner_strategy = strategy_a.name if strategy_a_is_blue else strategy_b.name
        winner_char = TRMPH_BLUE_WIN
    elif winner_enum.name == 'RED':
        winner_strategy = strategy_b.name if strategy_a_is_blue else strategy_a.name
        winner_char = TRMPH_RED_WIN
    else:
        raise ValueError(f"Unknown winner enum: {winner_enum}")
    
    logger.debug(f"Game complete: {winner_strategy} wins with {len(move_sequence)} moves")
    logger.debug(f"Final TRMPH: {trmph_str}")
    logger.debug(f"Timing summary: {strategy_a.name}={strategy_timings[strategy_a.name]:.3f}s, {strategy_b.name}={strategy_timings[strategy_b.name]:.3f}s")
    
    return {
        'winner_strategy': winner_strategy,
        'winner_char': winner_char,
        'trmph_str': trmph_str,
        'move_sequence': move_sequence,
        'num_moves': len(move_sequence),
        'opening': opening,
        'strategy_timings': strategy_timings,
        'total_moves': move_count
    }
