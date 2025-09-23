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
from hex_ai.inference.strategy_config import StrategyConfig, create_unified_config_from_args, create_strategy_configs_from_unified_config
from hex_ai.inference.tournament import TournamentResult as BaseTournamentResult
from hex_ai.config import DEFAULT_BATCH_CAP, DEFAULT_C_PUCT
from hex_ai.utils.format_conversion import (
    rowcol_to_trmph, trmph_to_moves
)
from hex_ai.data_processing import parse_trmph_line_flexible
from hex_ai.utils.tournament_logging import append_trmph_winner_line, write_tournament_trmph_header, find_available_csv_filename
from hex_ai.utils.deterministic_tournament_utils import (
    setup_tournament_output,
    save_opening_positions,
    setup_strategy_pair_files,
    create_play_config_for_pair,
    GameDuplicateTracker,
    play_strategy_pair_games,
    report_strategy_pair_results
)
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
TRMPH_SOURCE_DIR = "data/sf25/sep21"
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
                
                # Parse moves using centralized utility
                moves = trmph_to_moves(moves_str, BOARD_SIZE)
                
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


def run_deterministic_tournament(
    strategy_configs: List[StrategyConfig],
    openings: List[OpeningPosition],
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: int = DEFAULT_VERBOSE,
    seed: Optional[int] = None
) -> DeterministicTournamentResult:
    """
    Run a deterministic tournament using pre-generated opening positions.
    
    Args:
        strategy_configs: List of strategy configurations (each with its own model)
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
    # Use unique strategy names for tournament tracking (after parameter modifications)
    unique_strategy_names = [config.name for config in strategy_configs]
    result = DeterministicTournamentResult(unique_strategy_names)
    
    # Preload all models for efficiency
    from hex_ai.inference.model_cache import preload_tournament_models, get_model_cache
    model_paths = [config.model_path for config in strategy_configs]
    preload_tournament_models(model_paths)
    model_cache = get_model_cache()
    
    # Set up tournament output using utilities
    output_dir, openings_file = setup_tournament_output(OUTPUT_DIR_PREFIX)
    save_opening_positions(openings, openings_file)
    
    # Initialize game duplicate tracker
    duplicate_tracker = GameDuplicateTracker()
    
    # Run round-robin between all strategy pairs
    for strategy_a, strategy_b in itertools.combinations(strategy_configs, 2):
        logger.info(f"\nPlaying {len(openings)} games: {strategy_a.name} vs {strategy_b.name}")
        
        # Set up output files for this strategy pair
        trmph_file, csv_file = setup_strategy_pair_files(output_dir, strategy_a, strategy_b)
        
        # Create play configuration
        play_config = create_play_config_for_pair(strategy_a, strategy_b, temperature, seed)
        
        # Write TRMPH header
        pair_model_paths = [strategy_a.model_path, strategy_b.model_path]
        pair_strategy_configs = [strategy_a, strategy_b]
        actual_trmph_file = write_tournament_trmph_header(
            trmph_file, pair_model_paths, len(openings), play_config, BOARD_SIZE, 
            strategy_configs=pair_strategy_configs
        )
        
        # Find available CSV filename
        actual_csv_file = find_available_csv_filename(csv_file)
        
        # Play all games for this strategy pair using utility function
        game_results = play_strategy_pair_games(
            model_cache, strategy_a, strategy_b, openings, temperature, verbose,
            duplicate_tracker, actual_trmph_file, actual_csv_file, play_deterministic_game, result
        )
        
        # Report results for this pair
        report_strategy_pair_results(verbose, strategy_a, strategy_b, result)
    
    logger.info(f"Tournament complete. Total unique games played: {len(duplicate_tracker.seen_games)}")
    return result


# Import the utility function
from hex_ai.utils.gumbel_utils import generate_gumbel_summary_from_configs as generate_gumbel_summary
from hex_ai.utils.script_logging import ScriptConfig, print_script_configuration, print_script_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a deterministic strategy tournament using pre-generated opening positions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare strategies using model registry names
  %(prog)s --models=current_best,model1,model2 --strategies=policy,mcts,mcts --mcts-sims=100,200 --num-openings=100
  
  # Compare same strategy with different models from registry
  %(prog)s --models=current_best,previous_best --strategies=mcts,mcts --mcts-sims=100,100 --num-openings=50
  
  # Compare strategies using direct model file specification
  %(prog)s --model-files=epoch13_mini31.pt.gz,epoch13_mini27.pt.gz --model-dirs=checkpoints/dir1,checkpoints/dir2 --strategies=mcts,mcts --mcts-sims=30,30 --num-openings=50
  
  # Use specific opening file with different models
  %(prog)s --models=current_best,model1 --strategies=mcts,mcts --mcts-sims=100,200 --opening-file=data/deterministic_openings.txt
  
  # Compare with custom opening length and temperature
  %(prog)s --models=current_best,model1 --strategies=policy,mcts --mcts-sims=122 --num-openings=200 --opening-length=5 --temperature=0.1
  
  # Compare same strategy with different temperatures
  %(prog)s --models=current_best,current_best --strategies=policy,policy --temperatures=0.1,1.0 --num-openings=100
        """
    )
    
    parser.add_argument('--models', type=str,
                       help='Comma-separated list of model registry names (e.g., "current_best,model1,model2")')
    parser.add_argument('--model-files', type=str,
                       help='Comma-separated list of model file names (e.g., "epoch13_mini31.pt.gz,epoch13_mini27.pt.gz")')
    parser.add_argument('--model-dirs', type=str,
                       help='Comma-separated list of model directories (used with --model-files)')
    parser.add_argument('--strategies', type=str, required=True,
                       help='Comma-separated list of strategies to compare (e.g., "mcts,policy")')
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
    parser.add_argument('--batch-sizes', type=str,
                       help=f'Comma-separated batch sizes for MCTS strategies (e.g., "64,128,256", default: {DEFAULT_BATCH_CAP})')
    parser.add_argument('--c-puct', type=str,
                       help=f'Comma-separated PUCT exploration constants for MCTS strategies (e.g., "2.4,2.8,3.6", default: {DEFAULT_C_PUCT})')
    parser.add_argument('--enable-gumbel', type=str,
                       help='Comma-separated boolean values to enable Gumbel AlphaZero root selection for MCTS strategies (e.g., "true,false,true")')
    parser.add_argument('--gumbel-sim-threshold', type=str,
                       help='Comma-separated simulation thresholds for Gumbel AlphaZero root selection (e.g., "200,500,1000")')
    parser.add_argument('--gumbel-candidate-log-base', type=str,
                       help='Comma-separated log bases for Gumbel candidate scaling (e.g., "1.5,1.7,2.0")')
    parser.add_argument('--gumbel-candidate-log-offset', type=str,
                       help='Comma-separated log offsets for Gumbel candidate scaling (e.g., "-1.5,-2.0,-2.5")')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'Global temperature for move selection (0.0 = deterministic, default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--temperatures', type=str,
                       help='Comma-separated temperatures for each strategy (e.g., "0.1,1.0,0.5"). Overrides --temperature.')
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
    
    # Parse models and strategies using clean separation of concerns
    if args.models and (args.model_files or args.model_dirs):
        print("ERROR: Cannot specify both --models and --model-files/--model-dirs. Use one or the other.")
        sys.exit(1)
    
    if not args.models and not (args.model_files and args.model_dirs):
        print("ERROR: Must specify either --models (registry) or both --model-files and --model-dirs (direct)")
        sys.exit(1)
    
    # Parse strategy names (e.g., "mcts", "policy")
    strategy_names = [name.strip() for name in args.strategies.split(',')]
    
    # Parse model specifications
    if args.models:
        # Use model registry names
        model_names = [name.strip() for name in args.models.split(',')]
        
        # Validate that we have the same number of models and strategies
        if len(model_names) != len(strategy_names):
            print(f"ERROR: Number of models ({len(model_names)}) must match number of strategies ({len(strategy_names)})")
            sys.exit(1)
        
        # Validate model paths using registry
        model_paths = []
        for model_name in model_names:
            try:
                model_path = get_model_path(model_name)
                if not validate_model_path(model_path):
                    print(f"ERROR: Model file does not exist: {model_path}")
                    sys.exit(1)
                model_paths.append(model_path)
            except ValueError as e:
                print(f"ERROR: {e}")
                sys.exit(1)
    
    else:
        # Use direct model file specification
        model_files = [file.strip() for file in args.model_files.split(',')]
        model_dirs = [dir.strip() for dir in args.model_dirs.split(',')]
        
        # Validate that we have the same number of files, directories, and strategies
        if len(model_files) != len(model_dirs) or len(model_files) != len(strategy_names):
            print(f"ERROR: Number of model files ({len(model_files)}), directories ({len(model_dirs)}), and strategies ({len(strategy_names)}) must all match")
            sys.exit(1)
        
        # Build model paths
        model_paths = []
        for model_file, model_dir in zip(model_files, model_dirs):
            model_path = os.path.join(model_dir, model_file)
            
            # Validate model path exists
            if not os.path.exists(model_path):
                print(f"ERROR: Model file does not exist: {model_path}")
                sys.exit(1)
            
            model_paths.append(model_path)
    
    # Parse optional parameters using shared utility
    from hex_ai.utils.tournament_utils import parse_tournament_parameters
    parsed_params = parse_tournament_parameters(args)
    mcts_sims = parsed_params['mcts_sims']
    batch_sizes = parsed_params['batch_sizes']
    c_pucts = parsed_params['c_pucts']
    enable_gumbel = parsed_params['enable_gumbel']
    gumbel_sim_thresholds = parsed_params['gumbel_sim_thresholds']
    gumbel_candidate_log_bases = parsed_params['gumbel_candidate_log_bases']
    gumbel_candidate_log_offsets = parsed_params['gumbel_candidate_log_offsets']
    temperatures = parsed_params['temperatures']
    
    # Create strategy configurations using new unified system
    try:
        # Create unified config
        unified_config = create_unified_config_from_args(
            strategies=strategy_names,
            model_paths=model_paths,
            mcts_sims=mcts_sims,
            temperatures=temperatures,
            batch_sizes=batch_sizes,
            c_pucts=c_pucts,
            enable_gumbel=enable_gumbel,
            gumbel_sim_thresholds=gumbel_sim_thresholds,
            gumbel_candidate_log_bases=gumbel_candidate_log_bases,
            gumbel_candidate_log_offsets=gumbel_candidate_log_offsets,
            num_games=args.num_openings,  # Use num_openings as num_games for deterministic tournaments
            board_size=13,
            random_seed=args.seed,
            pie_rule=False  # Deterministic tournaments don't use pie rule
        )
        
        # Create strategy configs from unified config
        strategy_configs = create_strategy_configs_from_unified_config(unified_config)
        
        # Create unique strategy names by combining model file names with strategy names
        # This preserves the old behavior where different model files create different strategy names
        for i, config in enumerate(strategy_configs):
            model_path = config.model_path
            model_file = os.path.basename(model_path)
            model_name = os.path.splitext(model_file)[0]  # Remove .pt.gz extension
            unique_name = f"{model_name}_{config.original_name}"
            config.name = unique_name
        
        # Validate that all strategy configurations are unique
        # Check for duplicates by considering both strategy name and model path
        strategy_signatures = []
        for config in strategy_configs:
            signature = f"{config.original_name}:{config.model_path}"
            strategy_signatures.append(signature)
        
        if len(strategy_signatures) != len(set(strategy_signatures)):
            print("ERROR: Duplicate strategy configurations detected.")
            print("Each strategy must be unique in both name and model path.")
            sys.exit(1)
                
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
    
    # Print configuration using unified logging
    # Extract strategy names and Gumbel parameters
    strategy_names = [str(c) for c in strategy_configs]
    
    # Extract Gumbel parameters from strategy configs
    enable_gumbel = any(c.config.get('enable_gumbel_root_selection', False) for c in strategy_configs)
    gumbel_sim_threshold = None
    gumbel_c_visit = None
    gumbel_c_scale = None
    gumbel_candidate_log_base = None
    gumbel_candidate_log_offset = None
    gumbel_m_candidates = None
    
    for config in strategy_configs:
        if config.config.get('enable_gumbel_root_selection', False):
            gumbel_sim_threshold = config.config.get('gumbel_sim_threshold', gumbel_sim_threshold)
            gumbel_c_visit = config.config.get('gumbel_c_visit', gumbel_c_visit)
            gumbel_c_scale = config.config.get('gumbel_c_scale', gumbel_c_scale)
            gumbel_candidate_log_base = config.config.get('gumbel_candidate_log_base', gumbel_candidate_log_base)
            gumbel_candidate_log_offset = config.config.get('gumbel_candidate_log_offset', gumbel_candidate_log_offset)
            gumbel_m_candidates = config.config.get('gumbel_m_candidates', gumbel_m_candidates)
    
    # Create unified script config
    script_config = ScriptConfig(
        script_type="deterministic_tournament",
        models=model_paths,
        strategies=strategy_names,
        num_games=len(openings),
        strategy_config={},  # Strategy configs are handled individually
        temperatures=args.temperatures if args.temperatures else args.temperature,
        random_seed=args.seed,
        pie_rule=False,  # Deterministic tournaments don't use pie rule
        opening_length=args.opening_length,
        batch_sizes=args.batch_sizes,
        c_puct=args.c_puct,
        enable_gumbel=enable_gumbel,
        gumbel_sim_threshold=gumbel_sim_threshold,
        gumbel_c_visit=gumbel_c_visit,
        gumbel_c_scale=gumbel_c_scale,
        gumbel_candidate_log_base=gumbel_candidate_log_base,
        gumbel_candidate_log_offset=gumbel_candidate_log_offset,
        gumbel_m_candidates=gumbel_m_candidates
    )
    
    print_script_configuration(script_config)
    
    # Print additional deterministic tournament specific info
    print(f"  Number of openings: {len(openings)} (randomly selected from pool of {len(all_openings)})")
    print()
    
    # Run tournament
    result = run_deterministic_tournament(
        strategy_configs=strategy_configs,
        openings=openings,
        temperature=args.temperature,
        verbose=args.verbose,
        seed=args.seed
    )
    
    # Print results using unified analyzer
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    output_dir = f"{OUTPUT_DIR_PREFIX}{timestamp}"
    
    output_files = {
        "directory": output_dir
    }
    
    print_script_results("deterministic_tournament", result, script_config, output_files)


if __name__ == "__main__":
    main()
