"""
Run a round-robin tournament between selected model checkpoints.
Each pair plays N games (N/2 as first, N/2 as second). Results are logged to disk.
Win rates and Elo scores are printed at the end.

This script uses the unified parameter system for clean, consistent configuration.

Examples:

1. Compare models using policy-based selection:
   PYTHONPATH=. python scripts/run_tournament.py \
     --model-files="epoch1_mini50.pt.gz,epoch1_mini75.pt.gz" \
     --model-dirs="checkpoints/dir1,checkpoints/dir2" \
     --strategies="policy,policy" \
     --num-games=50

2. Compare models using MCTS with different simulation counts:
   PYTHONPATH=. python scripts/run_tournament.py \
     --model-files="epoch1_mini201.pt.gz,epoch1_mini75.pt.gz" \
     --model-dirs="checkpoints/dir1,checkpoints/dir2" \
     --strategies="mcts,mcts" \
     --mcts-sims="150,200" \
     --c-puct="1.5,2.0" \
     --num-games=50

3. Compare models using MCTS with Gumbel-AlphaZero enabled:
   PYTHONPATH=. python scripts/run_tournament.py \
     --model-files="epoch1_mini201.pt.gz,epoch1_mini75.pt.gz" \
     --model-dirs="checkpoints/dir1,checkpoints/dir2" \
     --strategies="mcts,mcts" \
     --mcts-sims="200,200" \
     --enable-gumbel="true,false" \
     --num-games=50

4. Compare different strategies:
   PYTHONPATH=. python scripts/run_tournament.py \
     --model-files="epoch1_mini50.pt.gz,epoch1_mini75.pt.gz" \
     --model-dirs="checkpoints/dir1,checkpoints/dir2" \
     --strategies="policy,mcts" \
     --mcts-sims="200" \
     --temperatures="1.2,0.0" \
     --num-games=100
"""

import argparse
import os
import sys
import time
from datetime import datetime
from typing import List, Optional

from hex_ai.inference.model_config import get_model_path, validate_model_path
from hex_ai.inference.strategy_config import create_unified_config_from_args, create_strategy_configs_from_unified_config
from hex_ai.inference.tournament import run_round_robin_tournament, TournamentConfig, TournamentPlayConfig
from hex_ai.training_logger import ScriptConfig, print_script_configuration, print_script_results


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a round-robin tournament between model checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare models using policy-based selection
  %(prog)s --model-files="epoch1_mini50.pt.gz,epoch1_mini75.pt.gz" --model-dirs="checkpoints/dir1,checkpoints/dir2" --strategies="policy,policy" --num-games=50
  
  # Compare models using MCTS with different simulation counts
  %(prog)s --model-files="epoch1_mini201.pt.gz,epoch1_mini75.pt.gz" --model-dirs="checkpoints/dir1,checkpoints/dir2" --strategies="mcts,mcts" --mcts-sims="150,200" --num-games=50
  
  # Compare models using MCTS with Gumbel-AlphaZero enabled
  %(prog)s --model-files="epoch1_mini201.pt.gz,epoch1_mini75.pt.gz" --model-dirs="checkpoints/dir1,checkpoints/dir2" --strategies="mcts,mcts" --mcts-sims="200,200" --enable-gumbel="true,false" --num-games=50
  
  # Compare different strategies
  %(prog)s --model-files="epoch1_mini50.pt.gz,epoch1_mini75.pt.gz" --model-dirs="checkpoints/dir1,checkpoints/dir2" --strategies="policy,mcts" --mcts-sims="200" --temperatures="1.2,0.0" --num-games=100
        """
    )
    
    # Tournament settings
    parser.add_argument('--num-games', type=int, default=50,
                       help='Number of games per pair (default: 50)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (default: auto-generated from time)')
    parser.add_argument('--no-pie-rule', action='store_true',
                       help='Disable pie rule (pie rule is enabled by default)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    
    # Model specification - clean separation
    parser.add_argument('--models', type=str,
                       help='Comma-separated list of model registry names (e.g., "current_best,current_best")')
    parser.add_argument('--model-files', type=str,
                       help='Comma-separated list of model checkpoint filenames (e.g., "epoch1_mini50.pt.gz,epoch1_mini75.pt.gz")')
    parser.add_argument('--model-dirs', type=str,
                       help='Comma-separated list of model checkpoint directories (e.g., "checkpoints/dir1,checkpoints/dir2")')
    
    # Strategy specification
    parser.add_argument('--strategies', type=str, required=True,
                       help='Comma-separated list of strategy types (e.g., "mcts,policy,mcts")')
    
    # MCTS parameters
    parser.add_argument('--mcts-sims', type=str,
                       help='Comma-separated list of MCTS simulation counts (e.g., "100,200,300")')
    parser.add_argument('--c-puct', type=str,
                       help='Comma-separated list of MCTS c_puct values (e.g., "1.5,2.0,1.0")')
    parser.add_argument('--batch-sizes', type=str,
                       help='Comma-separated list of batch sizes (e.g., "64,128,64")')
    
    # Gumbel parameters
    parser.add_argument('--enable-gumbel', type=str,
                       help='Comma-separated list of Gumbel enable flags (e.g., "true,false,true")')
    parser.add_argument('--gumbel-sim-threshold', type=str,
                       help='Comma-separated list of Gumbel simulation thresholds (e.g., "200,100,200")')
    parser.add_argument('--gumbel-candidate-log-base', type=str,
                       help='Comma-separated list of Gumbel candidate log bases (e.g., "2.0,1.5,2.0")')
    parser.add_argument('--gumbel-candidate-log-offset', type=str,
                       help='Comma-separated list of Gumbel candidate log offsets (e.g., "0.0,0.5,0.0")')
    
    # Temperature parameters
    parser.add_argument('--temperatures', type=str,
                       help='Comma-separated list of temperatures (e.g., "1.2,0.0,1.0")')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Generate seed if none provided, or use provided seed
    if args.seed is None:
        args.seed = int(time.time())
        print(f"Auto-generated seed: {args.seed}")
    
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
            num_games=args.num_games,
            board_size=13,
            random_seed=args.seed,
            pie_rule=not args.no_pie_rule
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
        strategy_signatures = [f"{config.original_name}:{config.model_path}" for config in strategy_configs]
        if len(strategy_signatures) != len(set(strategy_signatures)):
            print("ERROR: Tournament requires unique strategy configurations.")
            print("Each strategy must differ in at least one of:")
            print("  - Strategy type (policy, mcts)")
            print("  - Model checkpoint") 
            print("  - MCTS simulation count")
            print("  - Search parameters (batch size, c_puct, etc.)")
            print("  - Gumbel settings")
            sys.exit(1)
    
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(1)
    
    # Create tournament configuration for the round-robin tournament
    player_labels = [config.name for config in strategy_configs]
    label_to_checkpoint = {config.name: config.model_path for config in strategy_configs}
    
    config = TournamentConfig(
        checkpoint_paths=model_paths,
        num_games=args.num_games,
        player_labels=player_labels,
        label_to_checkpoint=label_to_checkpoint
    )
    
    # Create play configuration
    # For round-robin tournaments, we use a single strategy type and let the individual
    # strategy configs handle the differences
    primary_strategy_type = strategy_configs[0].strategy_type
    
    # Create strategy config for the play config (used by the tournament system)
    play_strategy_config = {}
    if primary_strategy_type == 'mcts':
        # Use the first strategy's MCTS config as the base
        first_config = strategy_configs[0].config
        play_strategy_config.update({
            'mcts_sims': first_config.get('mcts_sims', 200),
            'mcts_c_puct': first_config.get('c_puct', 1.5),
            'enable_gumbel_root_selection': first_config.get('enable_gumbel', False),
            'gumbel_sim_threshold': first_config.get('gumbel_sim_threshold', 200),
            'gumbel_c_visit': first_config.get('gumbel_c_visit', 50.0),
            'gumbel_c_scale': first_config.get('gumbel_c_scale', 1.0),
        })
        if 'gumbel_candidate_log_base' in first_config:
            play_strategy_config['gumbel_candidate_log_base'] = first_config['gumbel_candidate_log_base']
        if 'gumbel_candidate_log_offset' in first_config:
            play_strategy_config['gumbel_candidate_log_offset'] = first_config['gumbel_candidate_log_offset']
    
    # Create per-participant temperature mapping
    participant_temperatures = {}
    for config in strategy_configs:
        if config.temperature is not None:
            participant_temperatures[config.name] = config.temperature
    
    play_config = TournamentPlayConfig(
        temperature=1.2,  # Default temperature (individual strategies use their own)
        random_seed=args.seed,
        pie_rule=not args.no_pie_rule,
        strategy=primary_strategy_type,
        strategy_config=play_strategy_config,
        participant_temperatures=participant_temperatures
    )
    
    # Create log files with descriptive names
    timestamp = (
        f"tournament_{args.num_games}games_{len(model_paths)}models_"
        f"{datetime.now().strftime('%y%m%d_%H')}"
    )
    LOG_DIR = "data/tournament_play"
    GAMES_FILE = os.path.join(LOG_DIR, f"{timestamp}/tournament.trmph")
    CSV_FILE = os.path.join(LOG_DIR, f"{timestamp}/tournament.csv")
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(GAMES_FILE), exist_ok=True)
    
    # Script configuration for logging
    script_config = ScriptConfig(
        script_type="tournament",
        models=model_paths,
        strategies=player_labels,
        num_games=args.num_games,
        strategy_config={
            "mcts_sims": mcts_sims,
            "c_puct": c_pucts,
            "enable_gumbel": enable_gumbel,
            "gumbel_sim_threshold": gumbel_sim_thresholds,
            "gumbel_candidate_log_base": gumbel_candidate_log_bases,
            "gumbel_candidate_log_offset": gumbel_candidate_log_offsets,
            "batch_sizes": batch_sizes
        },
        temperatures=temperatures,
        random_seed=args.seed,
        pie_rule=not args.no_pie_rule,
    )
    
    print_script_configuration(script_config)
    print(f"  Results: {GAMES_FILE}, {CSV_FILE}")
    print()
    
    # Run the tournament
    result, actual_games_file, actual_csv_file = run_round_robin_tournament(
        config,
        verbose=args.verbose,
        log_file=GAMES_FILE,
        csv_file=CSV_FILE,
        play_config=play_config
    )
    
    # Print results using unified analyzer
    output_files = {
        "games": actual_games_file,
        "csv": actual_csv_file
    }
    
    # Print actual file paths used (in case collision avoidance changed them)
    if actual_games_file != GAMES_FILE:
        print(f"Note: Tournament results written to {actual_games_file} (original filename was in use)")
    if actual_csv_file != CSV_FILE:
        print(f"Note: CSV results written to {actual_csv_file} (original filename was in use)")
    
    print_script_results("tournament", result, script_config, output_files)


if __name__ == "__main__":
    main()