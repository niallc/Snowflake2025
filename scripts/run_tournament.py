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
   PYTHONPATH=. python scripts/run_tournament.py \
     --model=current_best \
     --strategies=policy,mcts_122,fixed_tree_13_8 \
     --num-openings=100

2. Use specific opening file:
   PYTHONPATH=. python scripts/run_tournament.py \
     --model=current_best \
     --strategies=mcts_100,mcts_200 \
     --opening-file=data/deterministic_openings.txt

3. Use custom temperature:
   PYTHONPATH=. python scripts/run_tournament.py \
     --model=current_best \
     --strategies=policy,mcts_122 \
     --num-openings=150 \
     --temperature=0.1

4. Get different opening sets for multiple runs:
   # Each run automatically gets a different seed (from time)
   PYTHONPATH=. python scripts/run_tournament.py \
     --model=current_best \
     --strategies=policy,mcts_122 \
     --num-openings=100
   
   # Or manually specify seeds for reproducible results
   PYTHONPATH=. python scripts/run_tournament.py \
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
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import numpy as np

from hex_ai.config import (
    BOARD_SIZE, EMPTY_PIECE, TRMPH_BLUE_WIN, TRMPH_RED_WIN, TRMPH_PREFIX
)
from hex_ai.enums import Player
from hex_ai.inference.game_engine import HexGameState, apply_move_to_state
from hex_ai.inference.model_config import get_model_path, validate_model_path
from hex_ai.utils.gumbel_validation import validate_gumbel_configurations, print_gumbel_warnings, check_gumbel_configurations
from hex_ai.inference.move_selection import get_strategy, MoveSelectionConfig
from hex_ai.inference.strategy_config import StrategyConfig, create_unified_config_from_args, create_strategy_configs_from_unified_config, to_list_if_needed
from hex_ai.inference.tournament import TournamentResult as BaseTournamentResult
from hex_ai.config import DEFAULT_BATCH_CAP, DEFAULT_C_PUCT, DEFAULT_GUMBEL_C_SCALE
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
from hex_ai.inference.two_stage_tournament import TwoStageTournament
from hex_ai.inference.knockout_tournament import TournamentParticipant

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/tournament.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_OPENING_LENGTH = 5
DEFAULT_NUM_OPENINGS = 100
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = None  # Will be set to int(time.time()) if None
DEFAULT_VERBOSE = 1
TRMPH_SOURCE_DIR = "data/sf25/sep28"
TRMPH_FILE_PATTERN = "*.trmph"
OUTPUT_DIR_PREFIX = "data/tournament_play/tournament_"

# TODO: Consider adding configuration for:
# Low priority: Timeout handling for long-running strategies
# Low priority: Progress saving/resume functionality for interrupted tournaments


# DeterministicTournamentResult moved to hex_ai.inference.game_execution


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


# extract_openings_from_trmph_file function moved to hex_ai.inference.game_execution


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


# find_trmph_files function moved to hex_ai.inference.game_execution


# generate_diverse_openings function moved to hex_ai.inference.game_execution


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


# play_deterministic_game function moved to hex_ai.inference.game_execution


# run_round_robin_tournament moved to hex_ai.inference.game_execution


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
  
  # Use single model for multiple strategies (convenience feature)
  %(prog)s --models=current_best --strategies=policy,mcts,mcts --mcts-sims=100,200 --num-openings=100
        """
    )
    
    parser.add_argument('--models', type=str,
                       help='Comma-separated list of model registry names (e.g., "current_best,model1,model2"). If only one model is provided, it will be used for all strategies.')
    parser.add_argument('--model-files', type=str,
                       help='Comma-separated list of model file names (e.g., "epoch13_mini31.pt.gz,epoch13_mini27.pt.gz")')
    parser.add_argument('--model-dirs', type=str,
                       help='Comma-separated list of model directories (used with --model-files)')
    parser.add_argument('--strategies', type=str,
                       help='Comma-separated list of strategies to compare (e.g., "mcts,policy"). Required for traditional tournaments, optional for knockout-only tournaments.')
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
    parser.add_argument('--gumbel-candidate-power-scale', type=str,
                       help='Comma-separated power scales for Gumbel candidate scaling (e.g., "60.0,80.0,100.0")')
    parser.add_argument('--gumbel-candidate-power-rate', type=str,
                       help='Comma-separated power rates for Gumbel candidate scaling (e.g., "0.39,0.45,0.50")')
    parser.add_argument('--gumbel-candidate-power-offset', type=str,
                       help='Comma-separated power offsets for Gumbel candidate scaling (e.g., "-4.0,-3.0,-2.0")')
    parser.add_argument('--gumbel-progressive-widening', type=str,
                       help='Comma-separated boolean values to enable progressive widening batching for Gumbel strategies (e.g., "true,false,true")')
    parser.add_argument('--gumbel-batch-scaling-factors', type=str,
                       help='Comma-separated scaling factors for progressive widening batching (e.g., "0.5,1.0,2.0")')
    parser.add_argument('--gumbel-c-scale', type=str,
                       help=f'Comma-separated c_scale parameters for Gumbel AlphaZero root selection (e.g., "1000,5000,10000", default: {DEFAULT_GUMBEL_C_SCALE})')
    parser.add_argument('--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'Global temperature for move selection (0.0 = deterministic, default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('--temperatures', type=str,
                       help='Comma-separated temperatures for each strategy (e.g., "0.1,1.0,0.5"). Overrides --temperature.')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                       help=f'Random seed for opening selection (different seeds produce different opening sets) (default: auto-generated from time)')
    parser.add_argument('--verbose', type=int, default=DEFAULT_VERBOSE,
                       help=f'Verbosity level (default: {DEFAULT_VERBOSE})')
    
    # 2-stage tournament arguments
    parser.add_argument('--knockout-dir', type=str,
                       help='Directory containing checkpoints for knockout stage')
    parser.add_argument('--knockout-config', type=str,
                       help='JSON configuration for knockout stage MCTS strategy (e.g., \'{"mcts_sims": 100, "enable_gumbel_root_selection": true}\')')
    parser.add_argument('--epoch-range', type=str,
                       help='Epoch range for knockout stage. Format: "N" for single epoch, "N,M" for range N to M-1 (e.g., "16" for epoch 16 only, "16,19" for epochs 16,17,18)')
    parser.add_argument('--mini-epoch-range', type=str,
                       help='Mini epoch range for knockout stage. Format: "N" for single mini epoch, "N,M" for range N to M-1 (e.g., "14" for mini epoch 14 only, "14,20" for mini epochs 14-19)')
    parser.add_argument('--games-per-match', type=int, default=50,
                       help='Number of games per knockout match (default: 50)')
    parser.add_argument('--top-k', type=int, default=2,
                       help='Number of winners from knockout stage to advance (default: 2)')
    parser.add_argument('--round-robin-games', type=int, default=100,
                       help='Number of games per round-robin match (default: 100)')
    parser.add_argument('--run-desc', type=str,
                       help='Description of this tournament run (e.g., "Testing c_scale = 1.5") - will be included in output headers')
    
    return parser.parse_args()


def parse_epoch_range(epoch_range_str: str) -> Tuple[int, int]:
    """
    Parse epoch range string into start and end epoch numbers.
    
    Args:
        epoch_range_str: String like "16" for single epoch or "16,19" for range 16-19 (inclusive)
        
    Returns:
        Tuple of (start_epoch, end_epoch) where end_epoch is exclusive
        Examples:
            "16" -> (16, 17)  # Just epoch 16
            "16,19" -> (16, 20)  # Epochs 16, 17, 18, 19
        
    Raises:
        ValueError: If format is invalid
    """
    if not epoch_range_str:
        raise ValueError("Epoch range string cannot be empty")
    
    parts = epoch_range_str.split(',')
    if len(parts) == 1:
        # Single epoch: "16" -> start=16, end=17
        start_epoch = int(parts[0].strip())
        end_epoch = start_epoch + 1
    elif len(parts) == 2:
        # Range: "16,19" -> start=16, end=20 (includes 16,17,18,19)
        start_epoch = int(parts[0].strip())
        end_epoch = int(parts[1].strip()) + 1
    else:
        raise ValueError(f"Invalid epoch range format: '{epoch_range_str}'. Expected format: 'N' for single epoch (e.g., '16') or 'N,M' for range N to M inclusive (e.g., '16,19' for epochs 16,17,18,19)")
    
    if start_epoch < 1:
        raise ValueError(f"Start epoch must be >= 1, got {start_epoch}")
    if end_epoch <= start_epoch:
        raise ValueError(f"End epoch must be > start epoch, got start={start_epoch}, end={end_epoch}")
    
    return start_epoch, end_epoch


def parse_mini_epoch_range(mini_epoch_range_str: str) -> Tuple[int, int]:
    """
    Parse mini epoch range string into start and end mini epoch numbers.
    
    Args:
        mini_epoch_range_str: String like "14" for single mini epoch or "14,20" for range 14-20 (inclusive)
        
    Returns:
        Tuple of (start_mini_epoch, end_mini_epoch) where end_mini_epoch is exclusive
        Examples:
            "14" -> (14, 15)  # Just mini epoch 14
            "14,20" -> (14, 21)  # Mini epochs 14, 15, 16, 17, 18, 19, 20
        
    Raises:
        ValueError: If format is invalid
    """
    if not mini_epoch_range_str:
        raise ValueError("Mini epoch range string cannot be empty")
    
    parts = mini_epoch_range_str.split(',')
    if len(parts) == 1:
        # Single mini epoch: "14" -> start=14, end=15
        start_mini_epoch = int(parts[0].strip())
        end_mini_epoch = start_mini_epoch + 1
    elif len(parts) == 2:
        # Range: "14,20" -> start=14, end=21 (includes 14,15,16,17,18,19,20)
        start_mini_epoch = int(parts[0].strip())
        end_mini_epoch = int(parts[1].strip()) + 1
    else:
        raise ValueError(f"Invalid mini epoch range format: '{mini_epoch_range_str}'. Expected format: 'N' for single mini epoch (e.g., '14') or 'N,M' for range N to M inclusive (e.g., '14,20' for mini epochs 14-20)")
    
    if start_mini_epoch < 1:
        raise ValueError(f"Start mini epoch must be >= 1, got {start_mini_epoch}")
    if end_mini_epoch <= start_mini_epoch:
        raise ValueError(f"End mini epoch must be > start mini epoch, got start={start_mini_epoch}, end={end_mini_epoch}")
    
    return start_mini_epoch, end_mini_epoch


def is_knockout_only_tournament(args) -> bool:
    """Check if this is a knockout-only tournament (no round-robin participants)."""
    return args.knockout_dir and not args.models and not args.model_files


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
        if len([name.strip() for name in args.models.split(',')]) == 1 and len(strategy_names) > 1:
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
        
        # Validate that we have the same number of files, directories, and strategies (only if strategies are specified)
        if strategy_names and (len(model_files) != len(model_dirs) or len(model_files) != len(strategy_names)):
            print(f"ERROR: Number of model files ({len(model_files)}), directories ({len(model_dirs)}), and strategies ({len(strategy_names)}) must all match")
            sys.exit(1)
        elif len(model_files) != len(model_dirs):
            print(f"ERROR: Number of model files ({len(model_files)}) must match number of model directories ({len(model_dirs)})")
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
    if is_knockout_only_tournament(args):
        return []
    
    # Parse optional parameters using shared utility
    parsed_params = parse_tournament_parameters(args)
    mcts_sims = parsed_params['mcts_sims']
    batch_sizes = parsed_params['batch_sizes']
    c_pucts = parsed_params['c_pucts']
    enable_gumbel = parsed_params['enable_gumbel']
    gumbel_sim_thresholds = parsed_params['gumbel_sim_thresholds']
    gumbel_candidate_power_scales = parsed_params['gumbel_candidate_power_scales']
    gumbel_candidate_power_rates = parsed_params['gumbel_candidate_power_rates']
    gumbel_candidate_power_offsets = parsed_params['gumbel_candidate_power_offsets']
    gumbel_c_scales = parsed_params['gumbel_c_scales']
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
            gumbel_candidate_power_scales=gumbel_candidate_power_scales,
            gumbel_candidate_power_rates=gumbel_candidate_power_rates,
            gumbel_candidate_power_offsets=gumbel_candidate_power_offsets,
            gumbel_c_scales=gumbel_c_scales,
            num_games=args.num_openings,  # Use num_openings as num_games for deterministic tournaments
            board_size=13,
            random_seed=args.seed,
            pie_rule=False  # Deterministic tournaments don't use pie rule
        )
        
        # Create strategy configs from unified config
        strategy_configs = create_strategy_configs_from_unified_config(unified_config)
        
        # Create unique strategy names by combining model file names with strategy names and key parameters
        # This ensures strategies with different parameters get different names even with the same model
        for i, config in enumerate(strategy_configs):
            model_path = config.model_path
            model_file = os.path.basename(model_path)
            model_name = os.path.splitext(model_file)[0]  # Remove .pt.gz extension
            
            # Create a parameter suffix to distinguish strategies with different parameters
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
        # Check for duplicates by considering strategy name, model path, and all configuration parameters
        strategy_signatures = []
        for config in strategy_configs:
            # Create a comprehensive signature that includes all relevant parameters
            signature_parts = [
                config.original_name,
                config.model_path,
                str(config.temperature),
                str(config.config.get('mcts_sims', '')),
                str(config.config.get('mcts_c_puct', '')),
                str(config.config.get('batch_size', '')),
                str(config.config.get('enable_gumbel_root_selection', '')),
                str(config.config.get('gumbel_sim_threshold', '')),
                str(config.config.get('gumbel_candidate_power_scale', '')),
                str(config.config.get('gumbel_candidate_power_rate', '')),
                str(config.config.get('gumbel_candidate_power_offset', '')),
                str(config.config.get('gumbel_c_scale', ''))
            ]
            signature = ':'.join(signature_parts)
            strategy_signatures.append(signature)
        
        if len(strategy_signatures) != len(set(strategy_signatures)):
            print("ERROR: Duplicate strategy configurations detected.")
            print("Each strategy must be unique in name, model path, and all configuration parameters.")
            print("Strategies that differ in any parameter (temperature, enable_gumbel, c_puct, etc.) are considered distinct.")
            sys.exit(1)
                
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    return strategy_configs


def run_two_stage_tournament(args, strategy_configs, model_paths, openings, command_line):
    """
    Run a 2-stage tournament: knockout elimination followed by round-robin.
    
    Args:
        args: Parsed command line arguments
        strategy_configs: Strategy configurations for round-robin stage
        model_paths: Model paths for round-robin stage
        openings: Opening positions for the tournament
        command_line: Command line that was used to run the tournament
        
    Returns:
        Tournament result object
    """
    
    # Parse knockout configuration
    knockout_config = {}
    if args.knockout_config:
        print(f"DEBUG: Received knockout-config string: '{args.knockout_config}'")
        print(f"DEBUG: String length: {len(args.knockout_config)}")
        print(f"DEBUG: First 10 chars: '{args.knockout_config[:10]}'")
        print(f"DEBUG: Last 10 chars: '{args.knockout_config[-10:]}'")
        try:
            knockout_config = json.loads(args.knockout_config)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in --knockout-config: {e}")
            sys.exit(1)
    
    # Parse epoch range if specified
    epoch_range = None
    if args.epoch_range:
        try:
            epoch_range = parse_epoch_range(args.epoch_range)
            print(f"Using epoch range: {epoch_range[0]}-{epoch_range[1]-1}")
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    
    # Parse mini epoch range if specified
    mini_epoch_range = None
    if args.mini_epoch_range:
        try:
            mini_epoch_range = parse_mini_epoch_range(args.mini_epoch_range)
            print(f"Using mini epoch range: {mini_epoch_range[0]}-{mini_epoch_range[1]-1}")
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
    
    # Create round-robin participants from existing strategy configs
    round_robin_participants = []
    for i, (strategy_config, model_path) in enumerate(zip(strategy_configs, model_paths)):
        # Create strategy_config with model_path, strategy type, and temperature included
        participant_strategy_config = strategy_config.config.copy()
        participant_strategy_config["model_path"] = model_path
        participant_strategy_config["strategy"] = strategy_config.strategy_type  # FIX: Add strategy type
        participant_strategy_config["temperature"] = strategy_config.temperature  # FIX: Add temperature
        
        participant = TournamentParticipant(
            name=f"round_robin_{i}",
            strategy_config=participant_strategy_config,
            metadata={
                "strategy_name": str(strategy_config)
            }
        )
        round_robin_participants.append(participant)
    
    # Create and run two-stage tournament
    tournament = TwoStageTournament(
        knockout_dir=args.knockout_dir,
        knockout_config=knockout_config,
        round_robin_participants=round_robin_participants,
        games_per_match=args.games_per_match,
        top_k=args.top_k,
        round_robin_games=args.round_robin_games,
        epoch_range=epoch_range,
        mini_epoch_range=mini_epoch_range,
        command_line=command_line,
        run_desc=args.run_desc
    )
    
    print("Running 2-stage tournament...")
    print(f"  Knockout directory: {args.knockout_dir}")
    print(f"  Knockout config: {knockout_config}")
    print(f"  Games per match: {args.games_per_match}")
    print(f"  Top K: {args.top_k}")
    print(f"  Round-robin games: {args.round_robin_games}")
    print(f"  Round-robin participants: {len(round_robin_participants)}")
    print()
    
    # Run the tournament
    results = tournament.run_tournament()
    
    # Convert results to a format compatible with existing tournament result system
    # For now, create a simple result object
    class TwoStageTournamentResult:
        def __init__(self, results):
            self.results = results
            self.participants = []
            
            # Extract participants from results
            if results.get("knockout_results"):
                self.participants.extend([p.name for p in results["knockout_results"]["winners"]])
            if results.get("round_robin_results"):
                self.participants.extend([p["name"] for p in results["round_robin_results"]["participants"]])
        
        def get_summary(self):
            return self.results
    
        def print_results(self):
            print("\n" + "="*60)
            print("2-STAGE TOURNAMENT RESULTS")
            print("="*60)
            
            if self.results.get("knockout_results"):
                ko_results = self.results["knockout_results"]
                print(f"Knockout Stage:")
                print(f"  Total participants: {ko_results['total_participants']}")
                print(f"  Winners: {[p.name for p in ko_results['winners']]}")
                print()
            
            if self.results.get("round_robin_results"):
                rr_results = self.results["round_robin_results"]
                print(f"Round-Robin Stage:")
                print(f"  Total participants: {rr_results['total_participants']}")
                print(f"  Final ranking: {rr_results['ranking']}")
                print()
            
            if self.results.get("final_ranking"):
                print(f"Final Tournament Ranking: {self.results['final_ranking']}")
            
            print("="*60)
    
    return TwoStageTournamentResult(results), tournament.output_dir


def main():
    args = parse_args()
    
    # Get command line early - crash if not available
    try:
        command_line = get_command_line()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
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
    
    # For 2-stage tournaments, models/strategies are optional (only for round-robin stage)
    if not args.knockout_dir:
        if not args.models and not (args.model_files and args.model_dirs):
            print("ERROR: Must specify either --models (registry) or both --model-files and --model-dirs (direct)")
            sys.exit(1)
    
    # Parse strategy names (e.g., "mcts", "policy")
    # For knockout-only tournaments, strategies are optional
    if args.strategies:
        strategy_names = [name.strip() for name in args.strategies.split(',')]
    else:
        strategy_names = []
    
    # Parse model specifications
    model_paths = parse_model_specifications(args, strategy_names)
    
    # Create strategy configurations
    strategy_configs = create_strategy_configurations(args, strategy_names, model_paths)
    
    # Check for Gumbel algorithm issues and print warnings
    check_gumbel_configurations(args, strategy_configs)
    
    # Determine how many games to play
    # Always use --round-robin-games for the unified tournament system
    # If knockout_dir is provided, --round-robin-games is used for the round-robin stage
    # If knockout_dir is None, --round-robin-games is used for the entire tournament
    games_to_play = args.round_robin_games
    
    # Generate or load opening positions (skip for knockout-only tournaments)
    if is_knockout_only_tournament(args):
        # Knockout-only tournament - no openings needed
        all_openings = []
        openings = []
    elif args.opening_file and os.path.exists(args.opening_file):
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
        # target_generation = max(games_to_play * 2, 500)  # Generate at least 2x what we need
        # TODO: Figure out whether we need to generate more that we're planning to use for anything.
        target_generation = games_to_play
        all_openings = generate_diverse_openings(
            trmph_files, 
            opening_length=args.opening_length,
            target_count=target_generation,
            cache_file=args.cache_file
        )
    
    if not is_knockout_only_tournament(args):
        if not all_openings:
            print("ERROR: No opening positions generated")
            sys.exit(1)
        
        # Randomly select the desired number of openings from the available pool
        print(f"Randomly selecting {games_to_play} openings from pool of {len(all_openings)}...")
        openings = select_random_openings(all_openings, games_to_play, seed=args.seed)
    
    # Print configuration using unified logging
    # Extract strategy names and Gumbel parameters
    if strategy_configs:
        strategy_names = [str(c) for c in strategy_configs]
    else:
        strategy_names = []
    
    # Extract Gumbel parameters from strategy configs
    if strategy_configs:
        enable_gumbel = any(c.config.get('enable_gumbel_root_selection', False) for c in strategy_configs)
        gumbel_sim_threshold = None
        gumbel_c_visit = None
        gumbel_c_scale = None
        gumbel_candidate_power_scale = None
        gumbel_candidate_power_rate = None
        gumbel_candidate_power_offset = None
        gumbel_m_candidates = None
        
        for config in strategy_configs:
            if config.config.get('enable_gumbel_root_selection', False):
                gumbel_sim_threshold = config.config.get('gumbel_sim_threshold', gumbel_sim_threshold)
                gumbel_c_visit = config.config.get('gumbel_c_visit', gumbel_c_visit)
                gumbel_c_scale = config.config.get('gumbel_c_scale', gumbel_c_scale)
                gumbel_candidate_power_scale = config.config.get('gumbel_candidate_power_scale', gumbel_candidate_power_scale)
                gumbel_candidate_power_rate = config.config.get('gumbel_candidate_power_rate', gumbel_candidate_power_rate)
                gumbel_candidate_power_offset = config.config.get('gumbel_candidate_power_offset', gumbel_candidate_power_offset)
                gumbel_m_candidates = config.config.get('gumbel_m_candidates', gumbel_m_candidates)
    else:
        # No strategy configs for knockout-only tournaments
        enable_gumbel = False
        gumbel_sim_threshold = None
        gumbel_c_visit = None
        gumbel_c_scale = None
        gumbel_candidate_power_scale = None
        gumbel_candidate_power_rate = None
        gumbel_candidate_power_offset = None
        gumbel_m_candidates = None
    
    # Create unified script config (skip for knockout-only tournaments)
    if not is_knockout_only_tournament(args):
        script_config = ScriptConfig(
            script_type="tournament",
            models=model_paths,
            strategies=strategy_names,
            num_games=games_to_play,  # Use the determined number of games
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
            gumbel_candidate_power_scale=gumbel_candidate_power_scale,
            gumbel_candidate_power_rate=gumbel_candidate_power_rate,
            gumbel_candidate_power_offset=gumbel_candidate_power_offset,
            gumbel_m_candidates=gumbel_m_candidates
        )
    else:
        # No script config needed for knockout-only tournaments
        script_config = None
    
    if script_config:
        print_script_configuration(script_config)
    
    # Print additional deterministic tournament specific info
    if not is_knockout_only_tournament(args):
        print(f"  Number of openings: {len(openings)} (randomly selected from pool of {len(all_openings)})")
        print()
    
    # Always use the unified 2-stage tournament system
    # If knockout_dir is None, it will skip the knockout stage and go straight to round-robin
    result, actual_output_dir = run_two_stage_tournament(args, strategy_configs, model_paths, openings, command_line)
    
    # Print results using unified analyzer
    # Use the actual output directory from the tournament, not a new timestamp
    output_files = {
        "directory": actual_output_dir
    }
    
    if script_config:
        print_script_results("tournament", result, script_config, output_files)
    else:
        # For knockout-only tournaments, just print the results directly
        result.print_results()


if __name__ == "__main__":
    main()
