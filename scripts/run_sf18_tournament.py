#!/usr/bin/env python3
"""
Run a tournament between Snowflake 2025 models and Snowflake 2018 (SF18) AI.

This script allows you to compare your current SF25 models against the older
SF18 AI by running tournaments with pre-generated opening positions.

The SF18 AI runs as a separate webserver that must be started independently.
This script communicates with it via HTTP API calls.

Examples:

1. Compare SF25 models against SF18 with default settings:
   PYTHONPATH=. python scripts/run_sf18_tournament.py \
     --model-files=epoch11_mini15.pt.gz,epoch16_mini23.pt.gz \
     --model-dirs=checkpoints/dir1,checkpoints/dir2 \
     --strategies=mcts,mcts \
     --mcts-sims=30,30 \
     --num-openings=100

2. Use specific SF18 difficulty and server URL:
   PYTHONPATH=. python scripts/run_sf18_tournament.py \
     --models=current_best,model1 \
     --strategies=mcts,mcts \
     --mcts-sims=100,100 \
     --sf18-difficulty=8 \
     --sf18-server-url=http://localhost:8088 \
     --num-openings=50

3. Use custom opening file:
   PYTHONPATH=. python scripts/run_sf18_tournament.py \
     --models=current_best \
     --strategies=mcts \
     --mcts-sims=30 \
     --opening-file=data/deterministic_openings.txt \
     --sf18-difficulty=9
"""

import argparse
import itertools
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from hex_ai.config import (
    BOARD_SIZE, EMPTY_PIECE, TRMPH_BLUE_WIN, TRMPH_RED_WIN, TRMPH_PREFIX
)
from hex_ai.enums import Player, Winner
from hex_ai.inference.game_engine import HexGameState, apply_move_to_state
from hex_ai.inference.model_config import get_model_path, validate_model_path
from hex_ai.inference.sf18_client import SF18Client, SF18Player
from hex_ai.inference.move_selection import get_strategy, MoveSelectionConfig
from hex_ai.inference.strategy_config import StrategyConfig, create_unified_config_from_args, create_strategy_configs_from_unified_config, to_list_if_needed
from hex_ai.inference.tournament import TournamentResult as BaseTournamentResult
from hex_ai.config import DEFAULT_BATCH_CAP, DEFAULT_C_PUCT
from hex_ai.utils.format_conversion import (
    rowcol_to_trmph, trmph_to_moves
)
from hex_ai.data_processing import parse_trmph_line_flexible
from hex_ai.utils.tournament_logging import append_trmph_winner_line, write_tournament_trmph_header, find_available_csv_filename, get_command_line
from hex_ai.utils.tournament_utils import parse_tournament_parameters
from hex_ai.utils.deterministic_tournament_utils import (
    setup_tournament_output,
    save_opening_positions,
    setup_strategy_pair_files,
    create_play_config_for_pair,
    GameDuplicateTracker,
    play_strategy_pair_games,
    report_strategy_pair_results
)
from hex_ai.utils.random_utils import set_deterministic_seeds
from hex_ai.utils.script_logging import ScriptConfig, print_script_configuration, print_script_results
from hex_ai.inference.model_cache import create_temporary_model_cache
from hex_ai.inference.game_execution import (
    play_deterministic_game,
    extract_openings_from_trmph_file,
    find_trmph_files,
    generate_diverse_openings,
    run_round_robin_tournament,
    DeterministicTournamentResult
)

# Configure logging
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OPENING_LENGTH = 5
DEFAULT_NUM_OPENINGS = 100
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = None  # Will be set to int(time.time()) if None
DEFAULT_VERBOSE = 1
DEFAULT_SF18_DIFFICULTY = 9
DEFAULT_SF18_SERVER_URL = "http://localhost:8088"
TRMPH_SOURCE_DIR = "data/sf25/sep28"
TRMPH_FILE_PATTERN = "*.trmph"
OUTPUT_DIR_PREFIX = "data/tournament_play/sf18_vs_sf25/sf18_tournament_"


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


class SF18TournamentResult:
    """Tournament result for SF18 vs SF25 matches."""
    
    def __init__(self, sf25_models: List[str], sf18_difficulty: int):
        self.sf25_models = sf25_models
        self.sf18_difficulty = sf18_difficulty
        self.results = {}  # Dict mapping (sf25_model, sf18_difficulty) -> win/loss counts
        self.game_results = []  # List of individual game results
        
        # Initialize results tracking
        for model in sf25_models:
            self.results[(model, sf18_difficulty)] = {
                'sf25_wins': 0,
                'sf18_wins': 0,
                'total_games': 0
            }
    
    def record_game(self, sf25_model: str, winner: str, game_data: Dict[str, Any]):
        """Record the result of a single game."""
        key = (sf25_model, self.sf18_difficulty)
        
        if key not in self.results:
            self.results[key] = {
                'sf25_wins': 0,
                'sf18_wins': 0,
                'total_games': 0
            }
        
        self.results[key]['total_games'] += 1
        
        if winner == 'sf25':
            self.results[key]['sf25_wins'] += 1
        elif winner == 'sf18':
            self.results[key]['sf18_wins'] += 1
        
        self.game_results.append({
            'sf25_model': sf25_model,
            'winner': winner,
            'game_data': game_data
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get tournament summary."""
        summary = {
            'sf25_models': self.sf25_models,
            'sf18_difficulty': self.sf18_difficulty,
            'model_results': {}
        }
        
        for (model, difficulty), stats in self.results.items():
            if stats['total_games'] > 0:
                win_rate = stats['sf25_wins'] / stats['total_games']
                summary['model_results'][model] = {
                    'sf25_wins': stats['sf25_wins'],
                    'sf18_wins': stats['sf18_wins'],
                    'total_games': stats['total_games'],
                    'sf25_win_rate': win_rate
                }
        
        return summary
    
    def print_results(self):
        """Print tournament results."""
        print("\n" + "="*60)
        print("SF18 vs SF25 TOURNAMENT RESULTS")
        print("="*60)
        print(f"SF18 Difficulty: {self.sf18_difficulty}")
        print(f"SF25 Models: {len(self.sf25_models)}")
        print()
        
        for (model, difficulty), stats in self.results.items():
            if stats['total_games'] > 0:
                win_rate = stats['sf25_wins'] / stats['total_games']
                print(f"Model: {model}")
                print(f"  SF25 Wins: {stats['sf25_wins']}")
                print(f"  SF18 Wins: {stats['sf18_wins']}")
                print(f"  Total Games: {stats['total_games']}")
                print(f"  SF25 Win Rate: {win_rate:.3f}")
                print()
        
        print("="*60)


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


def select_random_openings(openings: List[OpeningPosition], num_openings: int, seed: Optional[int] = None) -> List[OpeningPosition]:
    """
    Randomly select a subset of unique openings from the available pool.
    
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
        logger.info(f"Requested {num_openings} openings, returning all {len(openings)} available")
        return openings.copy()
    
    # Randomly sample without replacement
    selected_indices = random.sample(range(len(openings)), num_openings)
    selected_openings = [openings[i] for i in selected_indices]
    
    logger.info(f"Randomly selected {len(selected_openings)} unique openings from pool of {len(openings)}")
    return selected_openings


def play_sf18_vs_sf25_game(
    model_cache,
    sf25_strategy: StrategyConfig,
    sf18_player: SF18Player,
    opening: OpeningPosition,
    temperature: float = DEFAULT_TEMPERATURE,
    board_size: int = BOARD_SIZE,
    verbose: int = 0,
    sf25_is_blue: bool = True
) -> Dict[str, Any]:
    """
    Play a single game between an SF25 model and SF18 AI.
    
    Args:
        model_cache: Model cache for SF25 model
        sf25_strategy: Strategy configuration for SF25 model
        sf18_player: SF18 player instance
        opening: Opening position to start from
        temperature: Temperature for SF25 move selection
        board_size: Board size
        verbose: Verbosity level
        sf25_is_blue: Whether SF25 plays as blue (first player)
        
    Returns:
        Dictionary containing game result information
    """
    # Start from opening position
    state = opening.get_state(board_size)
    
    # Get move selection config for SF25
    move_config = MoveSelectionConfig(
        temperature=temperature,
        **sf25_strategy.config
    )
    
    # Get strategy object for SF25
    sf25_strategy_obj = get_strategy(sf25_strategy.strategy_type)
    
    # Track move sequence for TRMPH generation
    move_sequence = list(opening.moves)  # Start with opening moves
    move_count = len(opening.moves)
    max_moves = board_size * board_size
    
    # Play the game
    while move_count < max_moves:
        if state.game_over:
            break
        
        # Show progress for verbose >= 3
        if verbose >= 3 and move_count % 10 == 0 and move_count > 0:
            print(f"    Move {move_count}...", end="", flush=True)
        
        try:
            if (state.current_player_enum == Player.BLUE and sf25_is_blue) or \
               (state.current_player_enum == Player.RED and not sf25_is_blue):
                # SF25's turn
                model = model_cache.get_simple_model(sf25_strategy.model_path)
                # Only show detailed MCTS output at very high verbosity
                mcts_verbose = max(0, verbose - 2) if verbose >= 4 else 0
                row, col = sf25_strategy_obj.select_move(state, model, move_config, verbose=mcts_verbose)
            else:
                # SF18's turn
                row, col = sf18_player.get_move(state)
            
            # Apply the move
            state = apply_move_to_state(state, row, col)
            move_sequence.append((row, col))  # Track the move
            move_count += 1
            
            if verbose >= 4:
                print(f"  Move {move_count}: {rowcol_to_trmph(row, col, board_size)} by {'SF25' if ((state.current_player_enum == Player.RED and sf25_is_blue) or (state.current_player_enum == Player.BLUE and not sf25_is_blue)) else 'SF18'}")
        
        except Exception as e:
            logger.error(f"Error during game play: {e}")
            break
    
    # Add newline after progress indicator if we were showing progress
    if verbose >= 3 and move_count > 10:
        print()  # Newline after progress dots
    
    # Convert move sequence to TRMPH (preserving move order) - do this before error checking
    trmph_moves = ''.join([rowcol_to_trmph(r, c, board_size) for r, c in move_sequence])
    trmph_str = f"{TRMPH_PREFIX}{trmph_moves}"
    
    # Determine winner - this should always be valid in Hex
    if not state.game_over:
        # Game didn't finish - this indicates a serious bug
        print(f"\nERROR: Game did not finish properly!")
        print(f"Game state debug info:")
        print(f"  - Game over: {state.game_over}")
        print(f"  - Winner: {state.winner}")
        print(f"  - Current player: {state.current_player_enum}")
        print(f"  - Move count: {move_count}")
        print(f"  - Max moves: {max_moves}")
        print(f"  - Final TRMPH: {trmph_str}")
        print(f"  - Opening: {opening}")
        raise RuntimeError("Game did not finish - this should never happen in Hex")
    
    if state.winner is None:
        # Winner is None but game is over - this is a serious bug
        print(f"\nERROR: Game is over but winner is None!")
        print(f"Game state debug info:")
        print(f"  - Game over: {state.game_over}")
        print(f"  - Winner: {state.winner}")
        print(f"  - Winner type: {type(state.winner)}")
        print(f"  - Current player: {state.current_player_enum}")
        print(f"  - Move count: {move_count}")
        print(f"  - Final TRMPH: {trmph_str}")
        raise RuntimeError("Game is over but winner is None - this is a bug in the game engine")
    
    winner = state.winner
    if winner == Winner.BLUE:
        winner_str = "blue"
        winner_strategy = "SF25" if sf25_is_blue else "SF18"
    elif winner == Winner.RED:
        winner_str = "red"
        winner_strategy = "SF18" if sf25_is_blue else "SF25"
    else:
        # Unknown winner enum - this is a serious bug
        print(f"\nERROR: Unknown winner enum: {winner} (type: {type(winner)})")
        print(f"Game state debug info:")
        print(f"  - Game over: {state.game_over}")
        print(f"  - Winner: {state.winner}")
        print(f"  - Current player: {state.current_player_enum}")
        print(f"  - Move count: {move_count}")
        print(f"  - Final TRMPH: {trmph_str}")
        raise RuntimeError(f"Unknown winner enum: {winner} - this is a bug in the game engine")
        
    # Show game summary at verbose >= 1
    if verbose >= 1:
        print(f"    Game complete: {winner_strategy} wins in {move_count} moves")
    
    return {
        'winner': winner_str,
        'winner_strategy': winner_strategy,
        'trmph_str': trmph_str,
        'total_moves': move_count,
        'opening': opening,
        'sf25_strategy': sf25_strategy.name,
        'sf18_difficulty': sf18_player.difficulty
    }


def run_sf18_tournament(
    strategy_configs: List[StrategyConfig],
    sf18_client: SF18Client,
    sf18_difficulty: int,
    openings: List[OpeningPosition],
    temperature: float = DEFAULT_TEMPERATURE,
    verbose: int = DEFAULT_VERBOSE,
    seed: Optional[int] = None,
    output_dir: Optional[str] = None,
    command_line: str = None
) -> SF18TournamentResult:
    """
    Run a tournament between SF25 models and SF18 AI.
    
    Args:
        strategy_configs: List of SF25 strategy configurations
        sf18_client: SF18 client for communication
        sf18_difficulty: Difficulty level for SF18
        openings: List of opening positions to use
        temperature: Temperature for SF25 move selection
        verbose: Verbosity level
        seed: Random seed for reproducibility
        output_dir: Output directory for tournament files
        command_line: Command line that was used to run the tournament
        
    Returns:
        SF18TournamentResult with tournament results
    """
    # Create tournament result tracker
    sf25_models = [config.name for config in strategy_configs]
    result = SF18TournamentResult(sf25_models, sf18_difficulty)
    
    # Set up tournament output
    if output_dir is None:
        output_dir, openings_file = setup_tournament_output(OUTPUT_DIR_PREFIX)
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    save_opening_positions(openings, openings_file)
    
    # Create SF18 player
    sf18_player = SF18Player("SF18", sf18_client, sf18_difficulty)
    
    # Play games for each SF25 model
    for strategy_config in strategy_configs:
        logger.info(f"\nPlaying {len(openings)} games: {strategy_config.name} vs SF18 (difficulty {sf18_difficulty})")
        
        # Load model temporarily for this match
        model_cache = create_temporary_model_cache([strategy_config.model_path], verbose=0)
        
        # Set up output files for this match
        trmph_file, csv_file = setup_strategy_pair_files(output_dir, strategy_config, sf18_player)
        
        # Write TRMPH header
        play_config = create_play_config_for_pair(strategy_config, sf18_player, temperature, seed, command_line)
        pair_model_paths = [strategy_config.model_path, "SF18"]
        pair_strategy_configs = [strategy_config, sf18_player]
        actual_trmph_file = write_tournament_trmph_header(
            trmph_file, pair_model_paths, len(openings), play_config, BOARD_SIZE, 
            strategy_configs=pair_strategy_configs
        )
        
        # Find available CSV filename
        actual_csv_file = find_available_csv_filename(csv_file)
        
        # Play games
        for opening_idx, opening in enumerate(openings):
            if verbose >= 1:
                print(f"  Game {opening_idx + 1}/{len(openings)}: {opening}")
            
            # Game 1: SF25 (Blue) vs SF18 (Red)
            result_1 = play_sf18_vs_sf25_game(
                model_cache, strategy_config, sf18_player, opening, temperature, 
                verbose=verbose, sf25_is_blue=True
            )
            
            # Game 2: SF18 (Blue) vs SF25 (Red)
            result_2 = play_sf18_vs_sf25_game(
                model_cache, strategy_config, sf18_player, opening, temperature, 
                verbose=verbose, sf25_is_blue=False
            )
            
            # Record results
            winner_1 = "sf25" if result_1['winner_strategy'] == "SF25" else "sf18"
            winner_2 = "sf25" if result_2['winner_strategy'] == "SF25" else "sf18"
            
            result.record_game(strategy_config.name, winner_1, result_1)
            result.record_game(strategy_config.name, winner_2, result_2)
            
            # Log TRMPH results
            append_trmph_winner_line(result_1['trmph_str'], result_1['winner'][0], actual_trmph_file)
            append_trmph_winner_line(result_2['trmph_str'], result_2['winner'][0], actual_trmph_file)
            
        # Report results for this model
        print()
        print(f"Results for {strategy_config.name}:")
        model_stats = result.results[(strategy_config.name, sf18_difficulty)]
        if model_stats['total_games'] > 0:
            win_rate = model_stats['sf25_wins'] / model_stats['total_games']
            print(f"  SF25 Wins: {model_stats['sf25_wins']}")
            print(f"  SF18 Wins: {model_stats['sf18_wins']}")
            print(f"  Total Games: {model_stats['total_games']}")
            print(f"  SF25 Win Rate: {win_rate:.3f}")
        print()
    
    return result, output_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a tournament between Snowflake 2025 models and Snowflake 2018 AI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare SF25 models against SF18 with default settings
  %(prog)s --model-files=epoch11_mini15.pt.gz,epoch16_mini23.pt.gz --model-dirs=checkpoints/dir1,checkpoints/dir2 --strategies=mcts,mcts --mcts-sims=30,30 --num-openings=100
  
  # Use specific SF18 difficulty and server URL
  %(prog)s --models=current_best,model1 --strategies=mcts,mcts --mcts-sims=100,100 --sf18-difficulty=8 --sf18-server-url=http://localhost:8088 --num-openings=50
  
  # Use custom opening file
  %(prog)s --models=current_best --strategies=mcts --mcts-sims=30 --opening-file=data/deterministic_openings.txt --sf18-difficulty=9
        """
    )
    
    # SF25 model arguments (reused from run_tournament.py)
    parser.add_argument('--models', type=str,
                       help='Comma-separated list of model registry names (e.g., "current_best,model1,model2")')
    parser.add_argument('--model-files', type=str,
                       help='Comma-separated list of model file names (e.g., "epoch13_mini31.pt.gz,epoch13_mini27.pt.gz")')
    parser.add_argument('--model-dirs', type=str,
                       help='Comma-separated list of model directories (used with --model-files)')
    parser.add_argument('--strategies', type=str, required=True,
                       help='Comma-separated list of strategies to compare (e.g., "mcts,policy")')
    
    # Opening arguments
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
    
    # SF25 strategy arguments
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
    parser.add_argument('--gumbel-progressive-widening', type=str,
                       help='Comma-separated boolean values to enable progressive widening batching for Gumbel strategies (e.g., "true,false,true")')
    parser.add_argument('--gumbel-batch-scaling-factors', type=str,
                       help='Comma-separated scaling factors for progressive widening batching (e.g., "0.5,1.0,2.0")')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'Global temperature for move selection (0.0 = deterministic, default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--temperatures', type=str,
                       help='Comma-separated temperatures for each strategy (e.g., "0.1,1.0,0.5"). Overrides --temperature.')
    
    # SF18 arguments
    parser.add_argument('--sf18-difficulty', type=int, default=DEFAULT_SF18_DIFFICULTY,
                       help=f'Difficulty level for SF18 AI (1-10, default: {DEFAULT_SF18_DIFFICULTY})')
    parser.add_argument('--sf18-server-url', type=str, default=DEFAULT_SF18_SERVER_URL,
                       help=f'URL of SF18 server (default: {DEFAULT_SF18_SERVER_URL})')
    parser.add_argument('--sf18-timeout', type=int, default=30,
                       help='Timeout for SF18 server requests in seconds (default: 30)')
    
    # General arguments
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help=f'Random seed for opening selection (different seeds produce different opening sets) (default: auto-generated from time)')
    parser.add_argument('--verbose', type=int, default=DEFAULT_VERBOSE,
                       help=f'Verbosity level (default: {DEFAULT_VERBOSE})')
    
    return parser.parse_args()


def parse_model_specifications(args, strategy_names):
    """Parse and validate model specifications from command line arguments."""
    if args.models:
        # Use model registry names
        model_names = [name.strip() for name in args.models.split(',')]
        
        # Use existing utility for consistency with other parameters
        # This handles single value replication and validation automatically
        original_model_names = model_names.copy()
        model_names = to_list_if_needed(
            model_names, 
            len(strategy_names),
            parameter_name="models",
            original_values=original_model_names,
            strategy_names=strategy_names
        )
        
        # Show info message if single model was replicated
        if len(original_model_names) == 1 and len(strategy_names) > 1:
            print(f"INFO: Using single model '{model_names[0]}' for all {len(strategy_names)} strategies")
        
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
        return model_paths
    
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
        return model_paths


def create_strategy_configurations(args, strategy_names, model_paths):
    """Create strategy configurations for the tournament."""
    # Parse optional parameters using shared utility
    parsed_params = parse_tournament_parameters(args)
    mcts_sims = parsed_params['mcts_sims']
    batch_sizes = parsed_params['batch_sizes']
    c_pucts = parsed_params['c_pucts']
    enable_gumbel = parsed_params['enable_gumbel']
    gumbel_sim_thresholds = parsed_params['gumbel_sim_thresholds']
    gumbel_candidate_log_bases = parsed_params['gumbel_candidate_log_bases']
    gumbel_candidate_log_offsets = parsed_params['gumbel_candidate_log_offsets']
    temperatures = parsed_params['temperatures']
    
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
            num_games=args.num_openings,
            board_size=13,
            random_seed=args.seed,
            pie_rule=False
        )
        
        # Create strategy configs from unified config
        strategy_configs = create_strategy_configs_from_unified_config(unified_config)
        
        # Create unique strategy names
        for i, config in enumerate(strategy_configs):
            model_path = config.model_path
            model_file = os.path.basename(model_path)
            model_name = os.path.splitext(model_file)[0]
            
            # Create a parameter suffix to distinguish strategies
            param_parts = []
            if config.temperature is not None:
                param_parts.append(f"t{config.temperature}")
            if config.config.get('enable_gumbel_root_selection'):
                param_parts.append("gumbel")
            if config.config.get('mcts_c_puct') is not None:
                param_parts.append(f"cpuct{config.config['mcts_c_puct']}")
            if config.config.get('mcts_sims') is not None:
                param_parts.append(f"sims{config.config['mcts_sims']}")
            if config.config.get('gumbel_c_scale') is not None:
                param_parts.append(f"cscale{config.config['gumbel_c_scale']}")
            
            param_suffix = f"_{'_'.join(param_parts)}" if param_parts else ""
            unique_name = f"{model_name}_{config.original_name}{param_suffix}"
            config.name = unique_name
        
        # Validate that all strategy configurations are unique
        strategy_signatures = []
        for config in strategy_configs:
            signature_parts = [
                config.original_name,
                config.model_path,
                str(config.temperature),
                str(config.config.get('mcts_sims', '')),
                str(config.config.get('mcts_c_puct', '')),
                str(config.config.get('batch_size', '')),
                str(config.config.get('enable_gumbel_root_selection', '')),
                str(config.config.get('gumbel_sim_threshold', '')),
                str(config.config.get('gumbel_candidate_log_base', '')),
                str(config.config.get('gumbel_candidate_log_offset', '')),
                str(config.config.get('gumbel_c_scale', ''))
            ]
            signature = ':'.join(signature_parts)
            strategy_signatures.append(signature)
        
        if len(strategy_signatures) != len(set(strategy_signatures)):
            print("ERROR: Duplicate strategy configurations detected.")
            print("Each strategy must be unique in name, model path, and all configuration parameters.")
            sys.exit(1)
                
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    return strategy_configs


def main():
    args = parse_args()
    
    # Get command line early - crash if not available
    try:
        command_line = get_command_line()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Generate seed if none provided
    if args.seed is None:
        args.seed = int(time.time())
        print(f"Auto-generated seed: {args.seed}")
    
    # Set random seed for reproducible opening selection
    set_deterministic_seeds(args.seed)
    
    # Validate arguments
    if args.models and (args.model_files or args.model_dirs):
        print("ERROR: Cannot specify both --models and --model-files/--model-dirs. Use one or the other.")
        sys.exit(1)
    
    if not args.models and not (args.model_files and args.model_dirs):
        print("ERROR: Must specify either --models (registry) or both --model-files and --model-dirs (direct)")
        sys.exit(1)
    
    # Parse strategy names
    strategy_names = [name.strip() for name in args.strategies.split(',')]
    
    # Parse model specifications
    model_paths = parse_model_specifications(args, strategy_names)
    
    # Create strategy configurations
    strategy_configs = create_strategy_configurations(args, strategy_names, model_paths)
    
    # Check SF18 server connectivity
    sf18_client = SF18Client(args.sf18_server_url, args.sf18_timeout)
    if not sf18_client.is_server_running():
        print(f"ERROR: Cannot connect to SF18 server at {args.sf18_server_url}")
        print("Please make sure the SF18 server is running.")
        print("You can start it with:")
        print("python HttpGameServer.py --valueBuilderPath13=... --policyBuilderPath13=... --twoHeadBuilderPath=... --policyNetworkType=twoHeaded --portNum=8088")
        sys.exit(1)
    
    print(f"✓ SF18 server is running at {args.sf18_server_url}")
    
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
        
        # Generate diverse openings
        target_generation = args.num_openings
        all_openings = generate_diverse_openings(
            trmph_files, 
            opening_length=args.opening_length,
            target_count=target_generation,
            cache_file=args.cache_file
        )
    
    if not all_openings:
        print("ERROR: No opening positions generated")
        sys.exit(1)
    
    # Randomly select the desired number of openings
    print(f"Randomly selecting {args.num_openings} openings from pool of {len(all_openings)}...")
    openings = select_random_openings(all_openings, args.num_openings, seed=args.seed)
    
    # Print configuration
    print("\n" + "="*60)
    print("SF18 vs SF25 TOURNAMENT CONFIGURATION")
    print("="*60)
    print(f"SF25 Models: {len(strategy_configs)}")
    for config in strategy_configs:
        print(f"  - {config.name}")
    print(f"SF18 Difficulty: {args.sf18_difficulty}")
    print(f"SF18 Server: {args.sf18_server_url}")
    print(f"Number of openings: {len(openings)}")
    print(f"Opening length: {args.opening_length}")
    print(f"Temperature: {args.temperature}")
    print(f"Random seed: {args.seed}")
    print("="*60)
    print()
    
    # Run the tournament
    result, output_dir = run_sf18_tournament(
        strategy_configs=strategy_configs,
        sf18_client=sf18_client,
        sf18_difficulty=args.sf18_difficulty,
        openings=openings,
        temperature=args.temperature,
        verbose=args.verbose,
        seed=args.seed,
        command_line=command_line
    )
    
    # Print results
    result.print_results()
    
    # Save results to file in the tournament output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"sf18_tournament_results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(result.get_summary(), f, indent=2)
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
