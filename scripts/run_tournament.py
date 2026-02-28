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
   python scripts/run_tournament.py \
     --model=best \
     --strategies=policy,mcts_122,fixed_tree_13_8 \
     --num-openings=100

2. Use specific opening file:
   python scripts/run_tournament.py \
     --model=best \
     --strategies=mcts_100,mcts_200 \
     --opening-file=data/deterministic_openings.txt

3. Use custom temperature:
   python scripts/run_tournament.py \
     --model=best \
     --strategies=policy,mcts_122 \
     --num-openings=150 \
     --temperature=0.1

4. Get different opening sets for multiple runs:
   # Each run automatically gets a different seed (from time)
   python scripts/run_tournament.py \
     --model=best \
     --strategies=policy,mcts_122 \
     --num-openings=100
   
   # Or manually specify seeds for reproducible results
   python scripts/run_tournament.py \
     --model=best \
     --strategies=policy,mcts_122 \
     --num-openings=100 \
     --seed=123
"""

import argparse
import json
import logging
import os
import sys
import time
from typing import List, Dict, Any, Optional, Tuple

from hex_ai.memory_profiler import start_profiling, stop_profiling
from hex_ai.config import (
    DEFAULT_BATCH_CAP,
    DEFAULT_C_PUCT,
    DEFAULT_GUMBEL_C_SCALE,
    TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD,
)
from hex_ai.inference.model_config import get_model_path, validate_model_path, get_all_model_participants_from_generations
from hex_ai.utils.gumbel_validation import check_gumbel_configurations
from hex_ai.inference.strategy_config import StrategyConfig, create_unified_config_from_args, create_strategy_configs_from_unified_config, to_list_if_needed
from hex_ai.utils.tournament_logging import get_command_line
from hex_ai.utils.tournament_utils import parse_tournament_parameters
from hex_ai.utils.random_utils import set_deterministic_seeds
from hex_ai.utils.script_logging import ScriptConfig, print_script_configuration, print_script_results
from hex_ai.inference.game_execution import (
    find_trmph_files,
    generate_diverse_openings,
    load_openings_from_file,
    select_random_openings,
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

# Configure logging verbosity based on command line argument
def configure_logging_verbosity(verbose_level: int):
    """Configure logging verbosity based on the --verbose argument."""
    if verbose_level <= 0:
        # Very quiet - only show essential tournament progress
        logging.getLogger('hex_ai.inference.model_wrapper').setLevel(logging.ERROR)
        logging.getLogger('hex_ai.inference.checkpoint_discovery').setLevel(logging.ERROR)
        logging.getLogger('hex_ai.inference.game_execution').setLevel(logging.ERROR)
        logging.getLogger('hex_ai.inference.knockout_tournament').setLevel(logging.ERROR)
    elif verbose_level == 1:
        # Default - show tournament progress but reduce repetitive logs
        logging.getLogger('hex_ai.inference.model_wrapper').setLevel(logging.WARNING)
        logging.getLogger('hex_ai.inference.checkpoint_discovery').setLevel(logging.WARNING)
        logging.getLogger('hex_ai.inference.game_execution').setLevel(logging.WARNING)
        logging.getLogger('hex_ai.inference.knockout_tournament').setLevel(logging.WARNING)
    elif verbose_level >= 2:
        # Verbose - show all logs
        # Keep default INFO level for all loggers
        pass

# Constants
DEFAULT_OPENING_LENGTH = 5
DEFAULT_NUM_OPENINGS = 100
DEFAULT_TEMPERATURE = 0.0
DEFAULT_SEED = None  # Will be set to int(time.time()) if None
DEFAULT_VERBOSE = 1
TRMPH_SOURCE_DIR = "data/sf25/sep28"

# TODO: Consider adding configuration for:
# Low priority: Timeout handling for long-running strategies
# Low priority: Progress saving/resume functionality for interrupted tournaments


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a deterministic strategy tournament using pre-generated opening positions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare strategies using model registry names
  %(prog)s --models=best,model2 --strategies=policy,mcts,mcts --mcts-sims=100,200 --num-openings=100
  
  # Compare same strategy with different models from registry
  %(prog)s --models=best,previous_best --strategies=mcts,mcts --mcts-sims=100,100 --num-openings=50
  
  # Compare strategies using direct model file specification
  %(prog)s --model-files=epoch13_mini31.pt.gz,epoch13_mini27.pt.gz --model-dirs=checkpoints/dir1,checkpoints/dir2 --strategies=mcts,mcts --mcts-sims=30,30 --num-openings=50
  
  # Use specific opening file with different models
  %(prog)s --models=best --strategies=mcts,mcts --mcts-sims=100,200 --opening-file=data/deterministic_openings.txt
  
  # Compare with custom opening length and temperature
  %(prog)s --models=best --strategies=policy,mcts --mcts-sims=122 --num-openings=200 --opening-length=5 --temperature=0.1
  
  # Compare same strategy with different temperatures
  %(prog)s --models=best,best --strategies=policy,policy --temperatures=0.1,1.0 --num-openings=100
  
  # Use single model for multiple strategies (convenience feature)
  %(prog)s --models=best --strategies=policy,mcts,mcts --mcts-sims=100,200 --num-openings=100
        """
    )
    
    parser.add_argument('--models', type=str,
                       help='Comma-separated list of model registry names (e.g., "best,model2"). If only one model is provided, it will be used for all strategies.')
    parser.add_argument('--model-files', type=str,
                       help='Comma-separated list of model file names (e.g., "epoch13_mini31.pt.gz,epoch13_mini27.pt.gz")')
    parser.add_argument('--model-dirs', type=str,
                       help='Comma-separated list of model directories (used with --model-files)')
    parser.add_argument('--strategies', type=str,
                       help='Comma-separated list of strategies to compare (e.g., "mcts,policy"). Required for traditional tournaments, optional for knockout-only tournaments.')
    parser.add_argument('--num-openings', type=int, default=DEFAULT_NUM_OPENINGS,
                       help=f'Deprecated alias for --round-robin-games (default: {DEFAULT_NUM_OPENINGS})')
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
                       help='Directory containing checkpoints for knockout stage (mutually exclusive with --knockout-from-generations)')
    parser.add_argument('--knockout-from-generations', action='store_true',
                       help='Use all models from MODEL_GENERATIONS for knockout stage (mutually exclusive with --knockout-dir)')
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
    parser.add_argument('--round-robin-games', type=int, default=DEFAULT_NUM_OPENINGS,
                       help='Number of openings per round-robin pair (actual games are doubled via color swap, default: 100)')
    parser.add_argument('--run-desc', type=str,
                       help='Description of this tournament run (e.g., "Testing c_scale = 1.5") - will be included in output headers')

    # Memory profiling / leak triage
    parser.add_argument('--memory-profile', action='store_true',
                       help='Enable RSS/heap memory profiling (writes to temp/memoryProfile/).')
    parser.add_argument('--memory-profile-interval', type=int, default=60,
                       help='Seconds between automatic memory timeline samples (default: 60).')
    parser.add_argument('--memory-profile-dir', type=str, default="temp/memoryProfile",
                       help='Output directory for memory profiling files (default: temp/memoryProfile).')
    parser.add_argument('--mps-empty-cache-per-pair', action='store_true',
                       help='If running on MPS, call torch.mps.empty_cache() after each match/pair (diagnostic only).')

    # Lightweight MCTS timing profiler (GPU vs CPU breakdown)
    parser.add_argument('--mcts-profile', action='store_true',
                       help='Print lightweight MCTS timing breakdown every N calls (GPU vs CPU time).')
    parser.add_argument('--mcts-profile-every', type=int, default=10,
                       help='Print MCTS profile once every N MCTS move selections (default: 10).')
    parser.add_argument('--mcts-profile-max-calls', type=int, default=50,
                       help='Maximum number of MCTS move selections to profile (default: 50).')
    
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
    has_knockout = args.knockout_dir or args.knockout_from_generations
    return has_knockout and not args.models and not args.model_files


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
            # Keep strategy config metadata aligned with actual round-robin execution.
            num_games=args.round_robin_games,
            board_size=13,
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


def print_round_robin_strategy_summary(strategy_configs: List[StrategyConfig]) -> None:
    """Print key per-strategy settings for round-robin tournament participants."""
    if not strategy_configs:
        return

    print("  Round-robin strategy configurations:")
    for i, strategy in enumerate(strategy_configs, start=1):
        if strategy.strategy_type != "mcts":
            print(
                f"    {i}. {strategy.name}: "
                f"type={strategy.strategy_type}, "
                f"temperature={strategy.temperature}"
            )
            continue

        cfg = strategy.config
        summary_parts = [
            "type=mcts",
            f"temperature={strategy.temperature}",
            f"sims={cfg.get('mcts_sims')}",
            f"c_puct={cfg.get('mcts_c_puct')}",
            f"batch_size={cfg.get('batch_size')}",
            f"gumbel={cfg.get('enable_gumbel_root_selection', False)}",
        ]

        if cfg.get("enable_gumbel_root_selection", False):
            if cfg.get("gumbel_sim_threshold") is not None:
                summary_parts.append(f"gumbel_sim_threshold={cfg.get('gumbel_sim_threshold')}")
            if cfg.get("gumbel_c_scale") is not None:
                summary_parts.append(f"gumbel_c_scale={cfg.get('gumbel_c_scale')}")
            if cfg.get("gumbel_candidate_power_scale") is not None:
                summary_parts.append(
                    f"gumbel_power_scale={cfg.get('gumbel_candidate_power_scale')}"
                )
            if cfg.get("gumbel_candidate_power_rate") is not None:
                summary_parts.append(
                    f"gumbel_power_rate={cfg.get('gumbel_candidate_power_rate')}"
                )
            if cfg.get("gumbel_candidate_power_offset") is not None:
                summary_parts.append(
                    f"gumbel_power_offset={cfg.get('gumbel_candidate_power_offset')}"
                )

        print(f"    {i}. {strategy.name}: {', '.join(summary_parts)}")




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
    # Start with JSON config if provided
    knockout_config = {}
    if args.knockout_config:
        try:
            knockout_config = json.loads(args.knockout_config)
        except json.JSONDecodeError as e:
            print(f"ERROR: Invalid JSON in --knockout-config: {e}")
            sys.exit(1)
    
    # Parse command-line MCTS parameters and merge into knockout_config
    # This allows users to specify --mcts-sims, --enable-gumbel, etc. directly
    parsed_params = parse_tournament_parameters(args)
    
    # For knockout-only tournaments, validate that only single values are provided
    # (Multiple values are only meaningful for round-robin stage where different configs are compared)
    is_knockout_only = is_knockout_only_tournament(args)
    if is_knockout_only:
        params_with_multiple_values = []
        if parsed_params.get('mcts_sims') and len(parsed_params['mcts_sims']) > 1:
            params_with_multiple_values.append(f"mcts_sims (got {len(parsed_params['mcts_sims'])} values: {parsed_params['mcts_sims']})")
        if parsed_params.get('enable_gumbel') and len(parsed_params['enable_gumbel']) > 1:
            params_with_multiple_values.append(f"enable_gumbel (got {len(parsed_params['enable_gumbel'])} values: {parsed_params['enable_gumbel']})")
        if parsed_params.get('temperatures') and isinstance(parsed_params['temperatures'], list) and len(parsed_params['temperatures']) > 1:
            params_with_multiple_values.append(f"temperatures (got {len(parsed_params['temperatures'])} values: {parsed_params['temperatures']})")
        if parsed_params.get('c_pucts') and len(parsed_params['c_pucts']) > 1:
            params_with_multiple_values.append(f"c_puct (got {len(parsed_params['c_pucts'])} values: {parsed_params['c_pucts']})")
        if parsed_params.get('batch_sizes') and len(parsed_params['batch_sizes']) > 1:
            params_with_multiple_values.append(f"batch_sizes (got {len(parsed_params['batch_sizes'])} values: {parsed_params['batch_sizes']})")
        if parsed_params.get('gumbel_sim_thresholds') and len(parsed_params['gumbel_sim_thresholds']) > 1:
            params_with_multiple_values.append(f"gumbel_sim_threshold (got {len(parsed_params['gumbel_sim_thresholds'])} values: {parsed_params['gumbel_sim_thresholds']})")
        if parsed_params.get('gumbel_c_scales') and len(parsed_params['gumbel_c_scales']) > 1:
            params_with_multiple_values.append(f"gumbel_c_scale (got {len(parsed_params['gumbel_c_scales'])} values: {parsed_params['gumbel_c_scales']})")
        
        if params_with_multiple_values:
            print("ERROR: Knockout-only tournaments use a single config for all participants.")
            print("Multiple values provided for the following parameters:")
            for param in params_with_multiple_values:
                print(f"  - {param}")
            print("\nFor knockout-only tournaments, provide only a single value per parameter.")
            print("Multiple values are only meaningful when you also have round-robin participants.")
            sys.exit(1)
    
    # Extract first value from each parameter list (for knockout, we use single values)
    # Only override if the parameter was actually provided (non-empty list)
    if parsed_params.get('mcts_sims') and len(parsed_params['mcts_sims']) > 0:
        knockout_config['mcts_sims'] = parsed_params['mcts_sims'][0]
    if parsed_params.get('enable_gumbel') and len(parsed_params['enable_gumbel']) > 0:
        # enable_gumbel is already a list of booleans from parse_tournament_parameters
        knockout_config['enable_gumbel_root_selection'] = parsed_params['enable_gumbel'][0]
    if parsed_params.get('temperatures') and isinstance(parsed_params['temperatures'], list) and len(parsed_params['temperatures']) > 0:
        knockout_config['temperature'] = parsed_params['temperatures'][0]
    elif args.temperature is not None:
        # Also check for singular --temperature argument
        knockout_config['temperature'] = args.temperature
    if parsed_params.get('c_pucts') and len(parsed_params['c_pucts']) > 0:
        knockout_config['c_puct'] = parsed_params['c_pucts'][0]
    if parsed_params.get('batch_sizes') and len(parsed_params['batch_sizes']) > 0:
        knockout_config['batch_size'] = parsed_params['batch_sizes'][0]
    if parsed_params.get('gumbel_sim_thresholds') and len(parsed_params['gumbel_sim_thresholds']) > 0:
        knockout_config['gumbel_sim_threshold'] = parsed_params['gumbel_sim_thresholds'][0]
    if parsed_params.get('gumbel_c_scales') and len(parsed_params['gumbel_c_scales']) > 0:
        knockout_config['gumbel_c_scale'] = parsed_params['gumbel_c_scales'][0]
    
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
        
        # Use the strategy config's name (which includes model, strategy, and parameters)
        # This ensures the participant name matches what will be used in tournament results
        participant = TournamentParticipant(
            name=strategy_config.name,
            strategy_config=participant_strategy_config,
            metadata={
                "strategy_name": str(strategy_config)
            }
        )
        round_robin_participants.append(participant)
    
    # Handle knockout participants: either from directory or from MODEL_GENERATIONS
    knockout_participants = None
    if args.knockout_from_generations:
        # Get all participants from MODEL_GENERATIONS
        knockout_participants = get_all_model_participants_from_generations(knockout_config)
        print(f"Loaded {len(knockout_participants)} models from MODEL_GENERATIONS")
    
    # Validate epoch/mini epoch ranges are not used with knockout-from-generations
    if args.knockout_from_generations and (epoch_range or mini_epoch_range):
        print("WARNING: --epoch-range and --mini-epoch-range are ignored when using --knockout-from-generations")
        epoch_range = None
        mini_epoch_range = None
    
    # Create and run two-stage tournament
    tournament = TwoStageTournament(
        knockout_dir=args.knockout_dir if not args.knockout_from_generations else None,
        knockout_participants=knockout_participants,
        knockout_config=knockout_config,
        round_robin_participants=round_robin_participants,
        games_per_match=args.games_per_match,
        top_k=args.top_k,
        round_robin_games=args.round_robin_games,
        epoch_range=epoch_range,
        mini_epoch_range=mini_epoch_range,
        command_line=command_line,
        run_desc=args.run_desc,
        seed=args.seed,
        trmph_source=args.trmph_source,
        mps_empty_cache_per_pair=args.mps_empty_cache_per_pair
    )
    
    print("Running 2-stage tournament...")
    if args.knockout_from_generations:
        print(f"  Knockout participants: {len(knockout_participants)} models from MODEL_GENERATIONS")
    else:
        print(f"  Knockout directory: {args.knockout_dir}")
    print(f"  Knockout config: {knockout_config}")
    print(f"  Games per match: {args.games_per_match}")
    print(f"  Top K: {args.top_k}")
    print(f"  Round-robin openings per pair: {args.round_robin_games}")
    print(f"  Round-robin games per pair (with color swap): {args.round_robin_games * 2}")
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
    
    # Configure logging verbosity based on --verbose argument
    configure_logging_verbosity(args.verbose)

    # Optional: lightweight MCTS timing breakdown (GPU vs CPU).
    if args.mcts_profile:
        from hex_ai.inference.move_selection import configure_mcts_profiling
        configure_mcts_profiling(
            enabled=True,
            every_n_calls=args.mcts_profile_every,
            max_calls=args.mcts_profile_max_calls,
        )
    
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

    # Backward compatibility: --num-openings is an alias for --round-robin-games.
    if args.num_openings != DEFAULT_NUM_OPENINGS:
        if args.round_robin_games == DEFAULT_NUM_OPENINGS:
            args.round_robin_games = args.num_openings
            print(
                f"INFO: Interpreting --num-openings={args.num_openings} as "
                f"--round-robin-games={args.round_robin_games}."
            )
        elif args.round_robin_games != args.num_openings:
            print(
                "ERROR: --num-openings and --round-robin-games disagree. "
                f"Got --num-openings={args.num_openings}, "
                f"--round-robin-games={args.round_robin_games}."
            )
            sys.exit(1)
    
    # Set random seed for reproducible opening selection
    set_deterministic_seeds(args.seed)
    
    # Parse models and strategies using clean separation of concerns
    if args.models and (args.model_files or args.model_dirs):
        print("ERROR: Cannot specify both --models and --model-files/--model-dirs. Use one or the other.")
        sys.exit(1)
    
    # Validate knockout tournament arguments
    if args.knockout_dir and args.knockout_from_generations:
        print("ERROR: Cannot specify both --knockout-dir and --knockout-from-generations. Use one or the other.")
        sys.exit(1)
    
    # For 2-stage tournaments, models/strategies are optional (only for round-robin stage)
    # Skip this check if using knockout-from-generations (which provides its own participants)
    if not args.knockout_dir and not args.knockout_from_generations:
        if not args.models and not (args.model_files and args.model_dirs):
            print("ERROR: Must specify either --models (registry) or both --model-files and --model-dirs (direct)")
            sys.exit(1)
    
    # Parse strategy names (e.g., "mcts", "policy")
    # For knockout-only tournaments, strategies are optional
    if args.strategies:
        strategy_names = [name.strip() for name in args.strategies.split(',')]
    else:
        strategy_names = []
    
    # Parse model specifications (only needed if not knockout-only tournament)
    if is_knockout_only_tournament(args):
        # Knockout-only tournament - no model paths or strategy configs needed
        model_paths = []
        strategy_configs = []
    else:
        # Parse model specifications
        model_paths = parse_model_specifications(args, strategy_names)
        
        # Create strategy configurations
        strategy_configs = create_strategy_configurations(args, strategy_names, model_paths)
        
        # Check for Gumbel algorithm issues and print warnings
        check_gumbel_configurations(args, strategy_configs)
    
    # Determine how many openings to play per pair.
    # Always use --round-robin-games for the unified tournament system
    # (legacy name retained for CLI compatibility).
    openings_to_play = args.round_robin_games
    
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
        # target_generation = max(openings_to_play * 2, 500)  # Generate at least 2x what we need
        # TODO: Figure out whether we need to generate more that we're planning to use for anything.
        target_generation = openings_to_play
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
        print(f"Randomly selecting {openings_to_play} openings from pool of {len(all_openings)}...")
        openings = select_random_openings(all_openings, openings_to_play, seed=args.seed)
    
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
            num_games=openings_to_play * 2,  # Each opening is played twice with swapped colors.
            strategy_config={},  # Strategy configs are handled individually
            temperatures=args.temperatures if args.temperatures else args.temperature,
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
            gumbel_m_candidates=gumbel_m_candidates,
            confidence_termination_threshold=TOURNAMENT_CONFIDENCE_TERMINATION_THRESHOLD,
        )
    else:
        # No script config needed for knockout-only tournaments
        script_config = None
    
    if script_config:
        print_script_configuration(script_config)
    
    # Print additional deterministic tournament specific info
    if not is_knockout_only_tournament(args):
        print(f"  Number of openings: {len(openings)} (randomly selected from pool of {len(all_openings)})")
        print(f"  Games per strategy pair (from openings): {len(openings) * 2}")
        print_round_robin_strategy_summary(strategy_configs)
        print()
    
    # Optional: memory profiling (RSS + tracemalloc heap).
    # This is intended for long-run leak triage. It should not be enabled by default.
    if args.memory_profile:
        start_profiling(output_dir=args.memory_profile_dir, interval_seconds=args.memory_profile_interval)

    try:
        # Always use the unified 2-stage tournament system
        # If knockout_dir is None, it will skip the knockout stage and go straight to round-robin
        result, actual_output_dir = run_two_stage_tournament(args, strategy_configs, model_paths, openings, command_line)
    finally:
        if args.memory_profile:
            stop_profiling()
    
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
