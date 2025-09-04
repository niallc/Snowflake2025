#!/usr/bin/env python3
"""
Run a deterministic strategy tournament using pre-generated opening positions.

This script compares different strategies using the same model and the same
set of opening positions, eliminating randomness as a confounding factor.

The key insight is that by starting from the same opening positions,
we can directly compare how different strategies perform from identical
starting points, making the comparison much more robust.

Different runs can use different opening sets by changing the --seed parameter,
allowing you to gather more data across multiple tournament runs while
maintaining deterministic gameplay within each run.

Examples:

1. Compare strategies using 100 diverse openings:
   PYTHONPATH=. python scripts/run_deterministic_tournament.py \
     --model=current_best \
     --strategies=policy,mcts_122,fixed_tree_13_8 \
     --num-openings=100

2. Use specific opening file:
   PYTHONPATH=. python scripts/run_deterministic_tournament.py \
     --model=current_best \
     --strategies=mcts_100,mcts_200 \
     --opening-file=data/deterministic_openings.txt

3. Use custom temperature:
   PYTHONPATH=. python scripts/run_deterministic_tournament.py \
     --model=current_best \
     --strategies=policy,mcts_122 \
     --num-openings=150 \
     --temperature=0.1

4. Get different opening sets for multiple runs:
   # Each run automatically gets a different seed (from time)
   PYTHONPATH=. python scripts/run_deterministic_tournament.py \
     --model=current_best \
     --strategies=policy,mcts_122 \
     --num-openings=100
   
   # Or manually specify seeds for reproducible results
   PYTHONPATH=. python scripts/run_deterministic_tournament.py \
     --model=current_best \
     --strategies=policy,mcts_122 \
     --num-openings=100 \
     --seed=123
"""

import argparse
import csv
import glob
import itertools
import json
import logging
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from hex_ai.config import (
    BOARD_SIZE, EMPTY_PIECE, TRMPH_BLUE_WIN, TRMPH_RED_WIN, TRMPH_PREFIX
)
from hex_ai.enums import Player, Piece
from hex_ai.inference.game_engine import HexGameState, apply_move_to_state
from hex_ai.inference.model_config import get_model_path, validate_model_path
from hex_ai.inference.move_selection import get_strategy, MoveSelectionConfig
from hex_ai.inference.strategy_config import StrategyConfig, parse_strategy_configs
from hex_ai.inference.tournament import TournamentResult as BaseTournamentResult
from hex_ai.config import DEFAULT_BATCH_CAP, DEFAULT_C_PUCT
from hex_ai.utils.tournament_stats import print_comprehensive_tournament_analysis, calculate_head_to_head_stats, print_head_to_head_stats
from hex_ai.utils.format_conversion import (
    rowcol_to_trmph, trmph_move_to_rowcol, strip_trmph_preamble, split_trmph_moves
)
from hex_ai.utils.tournament_logging import append_trmph_winner_line
from hex_ai.utils.perf import PERF
from hex_ai.utils.random_utils import set_deterministic_seeds

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OPENING_LENGTH = 5
DEFAULT_NUM_OPENINGS = 100
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = None  # Will be set to int(time.time()) if None
DEFAULT_VERBOSE = 1
TRMPH_SOURCE_DIR = "data/twoNetGames"
TRMPH_FILE_PATTERN = "*.trmph"
OUTPUT_DIR_PREFIX = "data/tournament_play/deterministic_tournament_"

# TODO: Consider adding configuration for:
# Low priority: Timeout handling for long-running strategies
# Low priority: Progress saving/resume functionality for interrupted tournaments


class DeterministicTournamentResult(BaseTournamentResult):
    """Extended tournament result with timing tracking."""
    
    def __init__(self, participants: List[str]):
        super().__init__(participants)
        # Track timing data for each strategy
        self.strategy_timings = {name: 0.0 for name in participants}
        self.strategy_move_counts = {name: 0 for name in participants}
        self.game_timings = []  # List of individual game timing data
    
    def record_game_with_timing(self, winner: str, loser: str, game_timing_data: Dict[str, Any]):
        """Record a game result with timing information."""
        # Record the basic game result
        self.record_game(winner, loser)
        
        # Record timing data
        strategy_timings = game_timing_data.get('strategy_timings', {})
        for strategy_name, time_taken in strategy_timings.items():
            if strategy_name in self.strategy_timings:
                self.strategy_timings[strategy_name] += time_taken
        
        # Record move counts
        total_moves = game_timing_data.get('total_moves', 0)
        for strategy_name in strategy_timings:
            if strategy_name in self.strategy_move_counts:
                self.strategy_move_counts[strategy_name] += total_moves
        
        # Store individual game timing data
        self.game_timings.append(game_timing_data)
    
    def get_timing_summary(self) -> Dict[str, Any]:
        """Get a summary of timing statistics."""
        summary = {}
        
        for strategy_name in self.participants:
            total_time = self.strategy_timings.get(strategy_name, 0.0)
            total_moves = self.strategy_move_counts.get(strategy_name, 0)
            
            summary[strategy_name] = {
                'total_time': total_time,
                'total_moves': total_moves,
                'avg_time_per_move': total_time / max(1, total_moves),
                'total_games': sum(1 for game in self.game_timings 
                                 if strategy_name in game.get('strategy_timings', {}))
            }
        
        return summary
    
    def print_timing_summary(self):
        """Print a formatted timing summary."""
        summary = self.get_timing_summary()
        
        print("\n" + "="*60)
        print("TIMING SUMMARY")
        print("="*60)
        
        # Sort strategies by total time
        sorted_strategies = sorted(summary.items(), key=lambda x: x[1]['total_time'], reverse=True)
        
        for strategy_name, stats in sorted_strategies:
            print(f"{strategy_name}:")
            print(f"  Total time: {stats['total_time']:.3f}s")
            print(f"  Total moves: {stats['total_moves']}")
            print(f"  Average time per move: {stats['avg_time_per_move']:.3f}s")
            print(f"  Games played: {stats['total_games']}")
            print()
        
        # Print overall tournament timing
        total_tournament_time = sum(stats['total_time'] for stats in summary.values())
        print(f"Total tournament time: {total_tournament_time:.3f}s")
        print("="*60)


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


def get_move_config_for_strategy(strategy_config: StrategyConfig, temperature: float = DEFAULT_TEMPERATURE) -> MoveSelectionConfig:
    """Create a MoveSelectionConfig for a strategy with specified temperature."""
    config_dict = strategy_config.config.copy()
    config_dict['temperature'] = temperature
    return MoveSelectionConfig(**config_dict)


def validate_coordinates(row: int, col: int, board_size: int = BOARD_SIZE) -> None:
    """Validate that coordinates are within board bounds."""
    if not (0 <= row < board_size and 0 <= col < board_size):
        raise ValueError(f"Invalid coordinates ({row}, {col}) for board size {board_size}")


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
            if not line or not line.startswith('http://www.trmph.com/hex/board#13,'):
                continue
            
            # Parse TRMPH line
            try:
                # Extract moves and winner
                parts = line.split('#13,')
                if len(parts) != 2:
                    continue
                
                moves_winner = parts[1]
                # Find the winner indicator (b or r) at the end
                if moves_winner.endswith(f' {TRMPH_BLUE_WIN}'):
                    moves_str = moves_winner[:-2]
                    winner = TRMPH_BLUE_WIN
                elif moves_winner.endswith(f' {TRMPH_RED_WIN}'):
                    moves_str = moves_winner[:-2]
                    winner = TRMPH_RED_WIN
                else:
                    continue
                
                # Convert TRMPH moves to row,col coordinates with strict validation
                moves = []
                try:
                    # Use the proper TRMPH parsing functions
                    trmph_moves = split_trmph_moves(moves_str)
                    for trmph_move in trmph_moves:
                        try:
                            row, col = trmph_move_to_rowcol(trmph_move, BOARD_SIZE)
                            validate_coordinates(row, col, BOARD_SIZE)
                            moves.append((row, col))
                        except ValueError as e:
                            # Log and skip invalid moves, but continue processing
                            logger.warning(f"Invalid move '{trmph_move}' in line {line_num}: {e}")
                            continue
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


def load_openings_from_file(file_path: str, opening_length: int = DEFAULT_OPENING_LENGTH) -> List[OpeningPosition]:
    """
    Load opening positions from a file.
    
    Expected format: One TRMPH string per line, optionally with winner indicator.
    Examples:
        #13,a1b2c3d4e5f6g7
        #13,a1b2c3d4e5f6g7 b
        #13,a1b2c3d4e5f6g7 r
    
    Args:
        file_path: Path to file containing opening positions
        opening_length: Number of moves per opening (will truncate if longer)
    
    Returns:
        List of OpeningPosition objects
    
    Raises:
        ValueError: If file format is invalid or moves are malformed
    """
    openings = []
    
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            # Parse TRMPH string
            try:
                # Extract moves (remove winner indicator if present)
                if line.endswith(f' {TRMPH_BLUE_WIN}') or line.endswith(f' {TRMPH_RED_WIN}'):
                    moves_str = line[:-2]
                else:
                    moves_str = line
                
                # Validate TRMPH format
                if not moves_str.startswith(TRMPH_PREFIX):
                    raise ValueError(f"Line {line_num}: Expected TRMPH format starting with '{TRMPH_PREFIX}'")
                
                # Parse moves
                trmph_moves = split_trmph_moves(moves_str[len(TRMPH_PREFIX):])  # Remove prefix
                moves = []
                
                for trmph_move in trmph_moves:
                    row, col = trmph_move_to_rowcol(trmph_move, BOARD_SIZE)
                    validate_coordinates(row, col, BOARD_SIZE)
                    moves.append((row, col))
                
                # Truncate to opening length if necessary
                if len(moves) >= opening_length:
                    opening_moves = moves[:opening_length]
                    source_game = f"{os.path.basename(file_path)}:line{line_num}"
                    openings.append(OpeningPosition(opening_moves, source_game, opening_length))
                else:
                    logger.warning(f"Line {line_num} has only {len(moves)} moves, need {opening_length}")
                
            except Exception as e:
                raise ValueError(f"Error parsing line {line_num} in {file_path}: {e}")
    
    if not openings:
        raise ValueError(f"No valid openings found in {file_path}")
    
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


def select_random_openings(openings: List[OpeningPosition], num_openings: int, seed: Optional[int] = None) -> List[OpeningPosition]:
    """
    Randomly select a subset of unique openings from the available pool.
    
    This function ensures that different tournament runs can use different
    opening sets while maintaining deterministic gameplay within each run.
    It also maintains the uniqueness property of the original code.
    
    Args:
        openings: List of all available opening positions (assumed to be unique)
        num_openings: Number of openings to select
        seed: Optional random seed for reproducible selection
    
    Returns:
        List of randomly selected unique OpeningPosition objects
    """
    if seed is not None:
        random.seed(seed)
    
    if num_openings >= len(openings):
        # If we need all or more openings than available, return all
        logger.info(f"Requested {num_openings} openings, returning all {len(openings)} available")
        return openings.copy()
    
    # Randomly sample without replacement - this maintains uniqueness
    # since the input openings are already unique and we're sampling without replacement
    selected_indices = random.sample(range(len(openings)), num_openings)
    selected_openings = [openings[i] for i in selected_indices]
    
    logger.info(f"Randomly selected {len(selected_openings)} unique openings from pool of {len(openings)}")
    return selected_openings


def play_deterministic_game(
    model,
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
        model: The model to use for both strategies
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
    
    # The filename that a logger writes to (if using a FileHandler) can be accessed via:
    # logger.handlers[0].baseFilename  # if the first handler is a FileHandler
    #
    # If logger.handlers has length 0, then the logger has no handlers attached.
    # In this case, logging calls will propagate up to the parent logger (unless propagate=False).
    # If no ancestor logger has a handler, the logging output is lost (not shown anywhere).
    # By default, the root logger has a StreamHandler to stderr, so output is usually visible unless all handlers are removed.
    #
    # To inspect a logger for an upstream (parent) logger and its handlers in the interactive debugger, you can use:
    #   logger.parent         # This gives you the parent logger object (or None for the root logger)
    #   logger.parent.handlers
    #   logger.parent.name
    #   logger.parent.parent  # And so on, up the chain
    #
    # To walk up the logger hierarchy and print all handlers, you can use:
    #   l = logger
    #   while l:
    #       print(f"Logger: {l.name}, Handlers: {l.handlers}")
    #       l = l.parent
    #
    # If you see <StreamHandler <stderr> (NOTSET)> in logger.parent.handlers[0], this means that
    # the parent logger (often the root logger) is configured to output log messages to standard error (stderr).
    # So, unless you have added a FileHandler or other handler, your log output will go to the terminal's stderr.
    #
    # If you are NOT seeing log output in your terminal, possible reasons include:
    #   - The logger's level is set higher than the messages you are emitting (e.g., logger.level is WARNING, but you are logging INFO).
    #   - The handler's level is set higher than your messages.
    #   - The terminal in Cursor may not be showing stderr output, or stderr is not connected to the visible terminal.
    #   - Some environments (e.g., certain IDEs, Jupyter, or subprocesses) may redirect or suppress stderr.
    #   - There may be code elsewhere that removes or reconfigures handlers.
    #
    # To debug, try adding this at the top of your script (after imports) to force logging to stdout:
    # import logging, sys
    # root = logging.getLogger()
    # root.setLevel(logging.DEBUG)
    # for h in root.handlers:
    #     root.removeHandler(h)
    # handler = logging.StreamHandler(sys.stdout)
    # handler.setLevel(logging.DEBUG)
    # root.addHandler(handler)
    #
    # This will ensure all log output goes to stdout, which is almost always visible in terminal windows.
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
            else:
                strategy_obj = strategy_b_obj
                strategy_config = config_b
                strategy_name = strategy_b.name
        else:  # Player.RED
            if strategy_a_is_blue:
                strategy_obj = strategy_b_obj
                strategy_config = config_b
                strategy_name = strategy_b.name
            else:
                strategy_obj = strategy_a_obj
                strategy_config = config_a
                strategy_name = strategy_a.name
    
        # Time the move selection
        start_time = time.perf_counter()
        move = strategy_obj.select_move(state, model, strategy_config)
        end_time = time.perf_counter()
        
        if move is None:
            raise ValueError(f"Move selection returned None for {strategy_obj.get_name()}")
        
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
        raise ValueError("Game is not over or winner missing")
    
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


def write_csv_results(rows: List[Dict[str, Any]], csv_file: str) -> None:
    """Write CSV results to file."""
    csv_path = Path(csv_file)
    write_header = not csv_path.exists()
    headers = list(rows[0].keys())
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)



def run_deterministic_tournament(
    model_path: str,
    strategy_configs: List[StrategyConfig],
    openings: List[OpeningPosition],
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: int = DEFAULT_VERBOSE
) -> DeterministicTournamentResult:
    """
    Run a deterministic tournament using pre-generated opening positions.
    
    Args:
        model_path: Path to the model checkpoint
        strategy_configs: List of strategy configurations
        openings: List of opening positions to use
        temperature: Temperature for move selection (0.0 = deterministic)
        verbose: Verbosity level
    
    Returns:
        TournamentResult with results
    """
    # TODO: Add progress tracking and resume functionality
    # TODO: Add parallel processing for multiple strategy pairs
    # TODO: Add memory usage monitoring for large tournaments
    # TODO: Consider adding early termination if one strategy dominates
    
    # Create tournament result tracking strategy names
    strategy_names = [config.name for config in strategy_configs]
    result = DeterministicTournamentResult(strategy_names)
    
    # Preload the model for efficiency
    from hex_ai.inference.model_cache import preload_tournament_models, get_model_cache
    preload_tournament_models([model_path])
    model_cache = get_model_cache()
    model = model_cache.get_simple_model(model_path)
    
    # Create output directory and files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"{OUTPUT_DIR_PREFIX}{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save opening positions
    openings_file = os.path.join(output_dir, "openings.txt")
    with open(openings_file, 'w') as f:
        for i, opening in enumerate(openings):
            f.write(f"Opening {i+1}: {opening.get_trmph_string()}\n")
    
    # Track games for duplicate detection with more detailed tracking
    seen_games = set()  # All games across all strategy pairs and openings
    current_pair_games = {}  # Games from current strategy pair: {opening_idx: [game1_key, game2_key]}
    
    # Run round-robin between all strategy pairs
    for strategy_a, strategy_b in itertools.combinations(strategy_configs, 2):
        logger.info(f"\nPlaying {len(openings)} games: {strategy_a.name} vs {strategy_b.name}")
        
        # Create output files for this strategy pair
        pair_name = f"{strategy_a.name}_vs_{strategy_b.name}"
        trmph_file = os.path.join(output_dir, f"{pair_name}.trmph")
        csv_file = os.path.join(output_dir, f"{pair_name}.csv")
        
        # Reset tracking for this strategy pair
        current_pair_games = {}
        
        # Play games from each opening position
        game_results = []
        for opening_idx, opening in enumerate(openings):
            if verbose >= 1:
                if opening_idx == 0:
                    print(f"  Opening {opening_idx + 1}/{len(openings)}", end="", flush=True)
                else:
                    print(",", opening_idx + 1, end="", flush=True)
            
            # Game 1: Strategy A (Blue) vs Strategy B (Red)
            logger.debug(f"Playing game 1: {strategy_a.name} (Blue) vs {strategy_b.name} (Red) from opening {opening_idx + 1}")
            result_1 = play_deterministic_game(
                model, strategy_a, strategy_b, opening, temperature, verbose=verbose, strategy_a_is_blue=True
            )
            game_results.append(result_1)
            
            # Game 2: Strategy B (Blue) vs Strategy A (Red)
            logger.debug(f"Playing game 2: {strategy_b.name} (Blue) vs {strategy_a.name} (Red) from opening {opening_idx + 1}")
            result_2 = play_deterministic_game(
                model, strategy_a, strategy_b, opening, temperature, verbose=verbose, strategy_a_is_blue=False
            )
            game_results.append(result_2)
            
            # Store games for this opening to check for immediate duplicates (Case 1)
            game_1_key = f"{result_1['trmph_str']}_{result_1['winner_char']}"
            game_2_key = f"{result_2['trmph_str']}_{result_2['winner_char']}"
            current_pair_games[opening_idx] = [game_1_key, game_2_key]
            
            # Case 1: Check if both strategies produced the same game from this opening
            if game_1_key == game_2_key:
                opening_trmph = opening.get_trmph_string()
                print(f"Warning: {strategy_a.name} and {strategy_b.name} both produced same game {opening_trmph}")
            
            # Case 2: Check if either game duplicates a game from a different opening (SHOULD BE IMPOSSIBLE)
            for other_opening_idx, other_games in current_pair_games.items():
                if other_opening_idx != opening_idx:  # Different opening
                    duplicate_game = None
                    if game_1_key in other_games:
                        duplicate_game = game_1_key
                    elif game_2_key in other_games:
                        duplicate_game = game_2_key
                    
                    if duplicate_game:
                        print(f"ERROR: Game from opening {opening_idx + 1} duplicates game from opening {other_opening_idx + 1}")
                        print(f"  This should be impossible! Opening positions should guarantee unique games.")
                        print(f"  Opening {opening_idx + 1}: {opening.moves}")
                        print(f"  Opening {other_opening_idx + 1}: {openings[other_opening_idx].moves}")
                        sys.exit(1)
            
            # Log TRMPH results
            append_trmph_winner_line(result_1['trmph_str'], result_1['winner_char'], trmph_file)
            append_trmph_winner_line(result_2['trmph_str'], result_2['winner_char'], trmph_file)
            
            # Log CSV results
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
            rows = [
                {
                    "timestamp": timestamp,
                    "strategy_a": strategy_a.name,
                    "strategy_b": strategy_b.name,
                    "opening_idx": opening_idx,
                    "opening_source": opening.source_game,
                    "game": "A_first",
                    "trmph": result_1['trmph_str'],
                    "winner": result_1['winner_char'],
                    "winner_strategy": result_1['winner_strategy'],
                    "num_moves": result_1['num_moves'],
                    "opening_length": opening.opening_length,
                    "temperature": temperature,
                    "strategy_a_time": result_1['strategy_timings'].get(strategy_a.name, 0.0),
                    "strategy_b_time": result_1['strategy_timings'].get(strategy_b.name, 0.0),
                    "total_game_time": sum(result_1['strategy_timings'].values())
                },
                {
                    "timestamp": timestamp,
                    "strategy_a": strategy_b.name,
                    "strategy_b": strategy_a.name,
                    "opening_idx": opening_idx,
                    "opening_source": opening.source_game,
                    "game": "B_first",
                    "trmph": result_2['trmph_str'],
                    "winner": result_2['winner_char'],
                    "winner_strategy": result_2['winner_strategy'],
                    "num_moves": result_2['num_moves'],
                    "opening_length": opening.opening_length,
                    "temperature": temperature,
                    "strategy_a_time": result_2['strategy_timings'].get(strategy_b.name, 0.0),
                    "strategy_b_time": result_2['strategy_timings'].get(strategy_a.name, 0.0),
                    "total_game_time": sum(result_2['strategy_timings'].values())
                }
            ]
            
            write_csv_results(rows, csv_file)
            
            # Record results for tournament tracking with timing data
            # Game 1: Strategy A vs Strategy B
            winner_1 = result_1['winner_strategy']
            loser_1 = strategy_b.name if winner_1 == strategy_a.name else strategy_a.name
            result.record_game_with_timing(winner_1, loser_1, result_1)
            
            # Game 2: Strategy B vs Strategy A
            winner_2 = result_2['winner_strategy']
            loser_2 = strategy_a.name if winner_2 == strategy_b.name else strategy_b.name
            result.record_game_with_timing(winner_2, loser_2, result_2)
            
            if verbose >= 1:
                print(f":{result_1['winner_char']}/{result_2['winner_char']}", end="", flush=True)
        
        # Case 3: Check for duplicate games across different strategy pairs
        for result_data in game_results:
            game_key = f"{result_data['trmph_str']}_{result_data['winner_char']}"
            
            if game_key in seen_games:
                opening_trmph = result_data['opening'].get_trmph_string()
                print(f"Warning: Duplicate game across strategy pairs from {opening_trmph}")
            else:
                seen_games.add(game_key)
        
        if verbose >= 1:
            print()  # New line after games
        
        # Print statistics for the current strategy pair using shared utility
        # Extract head-to-head results for this specific pair
        strategy_a_wins = sum(1 for result in game_results if result['winner_strategy'] == strategy_a.name)
        strategy_b_wins = sum(1 for result in game_results if result['winner_strategy'] == strategy_b.name)
        total_games = len(game_results)
        
        if total_games > 0:
            stats = calculate_head_to_head_stats(strategy_a.name, strategy_b.name, strategy_a_wins, strategy_b_wins, total_games)
            print_head_to_head_stats(stats)
            
            # Print timing summary for this match
            total_time_a = sum(game['strategy_timings'].get(strategy_a.name, 0.0) for game in game_results)
            total_time_b = sum(game['strategy_timings'].get(strategy_b.name, 0.0) for game in game_results)
            
            print(f"  Timing: {strategy_a.name}={total_time_a:.3f}s, {strategy_b.name}={total_time_b:.3f}s")
    
    logger.info(f"Tournament complete. Total unique games played: {len(seen_games)}")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a deterministic strategy tournament using pre-generated opening positions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare strategies using 100 diverse openings
  %(prog)s --model=current_best --strategies=policy,mcts_122,fixed_tree_13_8 --num-openings=100
  
  # Use specific opening file
  %(prog)s --model=current_best --strategies=mcts_100,mcts_200 --opening-file=data/deterministic_openings.txt
  
  # Compare with custom opening length and temperature
  %(prog)s --model=current_best --strategies=policy,mcts_122 --num-openings=200 --opening-length=5 --temperature=0.1
  
  # Compare different batch sizes for MCTS
  %(prog)s --model=current_best --strategies=mcts_500,mcts_500,mcts_500 --batch-sizes=64,128,256 --num-openings=50
  
  # Compare different PUCT exploration constants
  %(prog)s --model=current_best --strategies=mcts_100,mcts_100,mcts_100 --c-puct=2.2,2.8,3.6 --num-openings=50
        """
    )
    
    parser.add_argument('--model', type=str, default='current_best',
                       help='Model to use for all strategies (default: current_best)')
    parser.add_argument('--strategies', type=str, required=True,
                       help='Comma-separated list of strategies to compare')
    parser.add_argument('--num-openings', type=int, default=DEFAULT_NUM_OPENINGS,
                       help=f'Number of opening positions to generate (default: {DEFAULT_NUM_OPENINGS})')
    parser.add_argument('--opening-length', type=int, default=DEFAULT_OPENING_LENGTH,
                       help=f'Number of moves per opening (default: {DEFAULT_OPENING_LENGTH})')
    parser.add_argument('--opening-file', type=str,
                       help='File containing pre-generated openings (overrides num-openings)')
    parser.add_argument('--cache-file', type=str,
                       help='File to cache generated openings for faster subsequent runs')
    parser.add_argument('--trmph-source', type=str, default=TRMPH_SOURCE_DIR,
                       help=f'Directory containing TRMPH files for opening generation (default: {TRMPH_SOURCE_DIR})')
    parser.add_argument('--mcts-sims', type=str,
                       help='Comma-separated MCTS simulation counts (overrides strategy names)')
    parser.add_argument('--search-widths', type=str,
                       help='Semicolon-separated search width sets (e.g., "13,8;20,10")')
    parser.add_argument('--batch-sizes', type=str,
                       help=f'Comma-separated batch sizes for MCTS strategies (e.g., "64,128,256", default: {DEFAULT_BATCH_CAP})')
    parser.add_argument('--c-puct', type=str,
                       help=f'Comma-separated PUCT exploration constants for MCTS strategies (e.g., "2.4,2.8,3.6", default: {DEFAULT_C_PUCT})')
    parser.add_argument('--enable-gumbel', type=str,
                       help='Comma-separated boolean values to enable Gumbel AlphaZero root selection for MCTS strategies (e.g., "true,false,true")')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'Temperature for move selection (0.0 = deterministic, default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help=f'Random seed for opening selection (different seeds produce different opening sets) (default: auto-generated from time)')
    parser.add_argument('--verbose', type=int, default=DEFAULT_VERBOSE,
                       help=f'Verbosity level (default: {DEFAULT_VERBOSE})')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Generate seed if none provided, or use provided seed
    if args.seed is None:
        args.seed = int(time.time())
        print(f"Auto-generated seed: {args.seed}")
    
    # Set random seed for reproducible opening selection
    set_deterministic_seeds(args.seed)
    
    # Validate model path
    try:
        model_path = get_model_path(args.model)
        if not validate_model_path(model_path):
            print(f"ERROR: Model file does not exist: {model_path}")
            sys.exit(1)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Parse strategies
    strategy_names = [name.strip() for name in args.strategies.split(',')]
    
    # Parse optional parameters
    mcts_sims = None
    if args.mcts_sims:
        mcts_sims = [int(s.strip()) for s in args.mcts_sims.split(',')]
    
    search_widths = None
    if args.search_widths:
        search_widths = [s.strip() for s in args.search_widths.split(';')]
    
    batch_sizes = None
    if args.batch_sizes:
        batch_sizes = [int(s.strip()) for s in args.batch_sizes.split(',')]
    
    c_pucts = None
    if args.c_puct:
        c_pucts = [float(s.strip()) for s in args.c_puct.split(',')]
    
    enable_gumbel = None
    if args.enable_gumbel:
        enable_gumbel = [s.strip().lower() == 'true' for s in args.enable_gumbel.split(',')]
    
    # Parse strategy configurations
    try:
        strategy_configs = parse_strategy_configs(strategy_names, mcts_sims, search_widths, batch_sizes, c_pucts, enable_gumbel)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Generate or load opening positions
    if args.opening_file and os.path.exists(args.opening_file):
        print(f"Loading openings from: {args.opening_file}")
        try:
            all_openings = load_openings_from_file(args.opening_file, args.opening_length)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    else:
        print(f"Generating diverse openings...")
        
        # Find TRMPH files
        trmph_files = find_trmph_files(args.trmph_source)
        if not trmph_files:
            print(f"ERROR: No TRMPH files found in {args.trmph_source}")
            sys.exit(1)
        
        # Generate diverse openings (generate more than needed to allow for random selection)
        target_generation = max(args.num_openings * 2, 500)  # Generate at least 2x what we need
        all_openings = generate_diverse_openings(
            trmph_files, 
            opening_length=args.opening_length,
            target_count=target_generation,
            cache_file=args.cache_file
        )
    
    if not all_openings:
        print("ERROR: No opening positions generated")
        sys.exit(1)
    
    # Randomly select the desired number of openings from the available pool
    print(f"Randomly selecting {args.num_openings} openings from pool of {len(all_openings)}...")
    openings = select_random_openings(all_openings, args.num_openings, seed=args.seed)
    
    # Print configuration
    print("\nDeterministic Tournament Configuration:")
    print(f"  Model: {args.model} ({os.path.basename(model_path)})")
    print(f"  Strategies: {[str(c) for c in strategy_configs]}")
    print(f"  Number of openings: {len(openings)} (randomly selected from pool of {len(all_openings)})")
    print(f"  Opening length: {args.opening_length} moves")
    print(f"  Temperature: {args.temperature}")
    if args.batch_sizes:
        print(f"  Batch sizes: {args.batch_sizes}")
    if args.c_puct:
        print(f"  C_PUCT values: {args.c_puct}")
    print(f"  Dirichlet noise: alpha=0.3, eps=0.25 (MCTS default)")
    print(f"  Root noise: disabled (add_root_noise=False)")
    print(f"  Random seed: {args.seed} (for opening selection)")
    
    # Print timestamp and git state
    timestamp = datetime.now()
    print(f"  Run time: {timestamp.strftime('%Y-%m-%d %H:%M')}")
    
    try:
        # Get git commit hash
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD'], 
                                         text=True, stderr=subprocess.DEVNULL).strip()[:8]
        
        # Check if there are uncommitted changes
        git_status = subprocess.check_output(['git', 'status', '--porcelain'], 
                                           text=True, stderr=subprocess.DEVNULL).strip()
        has_changes = bool(git_status)
        
        print(f"  Git: {git_hash}{' +changes' if has_changes else ''}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  Git: not available")
    
    print()
    
    # Run tournament
    result = run_deterministic_tournament(
        model_path=model_path,
        strategy_configs=strategy_configs,
        openings=openings,
        temperature=args.temperature,
        verbose=args.verbose
    )
    
    # Print results
    print("\nDeterministic Tournament Complete!")
    print_comprehensive_tournament_analysis(result)
    
    # Print timing summary
    result.print_timing_summary()
    
    # Print output location
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"{OUTPUT_DIR_PREFIX}{timestamp}"
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
