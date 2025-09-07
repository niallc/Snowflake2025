"""
Run a round-robin tournament between selected model checkpoints.
Each pair plays N games (N/2 as first, N/2 as second). Results are logged to disk.
Win rates and Elo scores are printed at the end.

This script supports comparing models from different directories by specifying
both checkpoint filenames and their corresponding directories.

Examples:

1. Compare models using policy-based selection:
   PYTHONPATH=. python scripts/run_tournament.py \
     --num-games=50 \
     --checkpoints="epoch1_mini50.pt.gz,epoch1_mini75.pt.gz,epoch1_mini100.pt.gz" \
     --strategy=policy

2. Compare models using MCTS with different simulation counts:
   PYTHONPATH=. python scripts/run_tournament.py \
     --num-games=50 \
     --checkpoints="epoch1_mini201.pt.gz,epoch1_mini75.pt.gz" \
     --strategy=mcts \
     --mcts-sims=150 \
     --mcts-c-puct=1.5

3. Compare models using MCTS with Gumbel-AlphaZero enabled:
   PYTHONPATH=. python scripts/run_tournament.py \
     --num-games=50 \
     --checkpoints="epoch1_mini201.pt.gz,epoch1_mini75.pt.gz" \
     --strategy=mcts \
     --mcts-sims=200 \
     --enable-gumbel \
     --gumbel-c-visit=50.0

4. Compare models using fixed tree search:
   PYTHONPATH=. python scripts/run_tournament.py \
     --num-games=50 \
     --checkpoints="epoch1_mini50.pt.gz,epoch1_mini75.pt.gz" \
     --strategy=fixed_tree \
     --search-widths="20,10,5"

4. Compare different inference strategies:
   # Policy vs MCTS vs Fixed Tree
   PYTHONPATH=. python scripts/run_tournament.py \
     --num-games=100 \
     --checkpoints="epoch1_mini50.pt.gz,epoch1_mini75.pt.gz" \
     --strategy=mcts \
     --mcts-sims=200 \
     --temperature=1.2

Note: When using --checkpoint_dirs, the number of directories must match the number of checkpoints,
or you can specify a single directory for all checkpoints.

"""
import argparse
import os
import sys
from datetime import datetime
from typing import List

from hex_ai.inference.model_config import get_model_dir, get_model_path
from hex_ai.inference.tournament import (
    TournamentConfig, TournamentPlayConfig, run_round_robin_tournament
)
from hex_ai.utils.tournament_stats import print_comprehensive_tournament_analysis

# Get the current best model directory from model config
DEFAULT_CHKPT_DIR = get_model_dir("current_best")

# Default list of checkpoints to compare (epoch1 mini epochs 100, 120, 155)
DEFAULT_CHECKPOINTS = [
    "epoch1_mini100.pt.gz", 
    "epoch1_mini120.pt.gz",
    "epoch1_mini155.pt.gz",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a round-robin tournament between model checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare models using policy-based selection
  %(prog)s --num-games=50 --checkpoints="epoch1_mini50.pt.gz,epoch1_mini75.pt.gz" --strategy=policy
  
  # Compare models using MCTS
  %(prog)s --num-games=50 --checkpoints="epoch1_mini50.pt.gz,epoch1_mini75.pt.gz" --strategy=mcts --mcts-sims=200
  
  # Compare models using MCTS with Gumbel-AlphaZero
  %(prog)s --num-games=50 --checkpoints="epoch2_mini201.pt.gz,epoch1_mini75.pt.gz" --strategy=mcts --mcts-sims=200 --enable-gumbel
  
  # Compare models using fixed tree search
  %(prog)s --num-games=50 --checkpoints="epoch1_mini50.pt.gz,epoch1_mini75.pt.gz" --strategy=fixed_tree --search-widths="20,10,5"
        """
    )
    parser.add_argument('--num-games', type=int, default=50, 
                       help='Number of games per pair (default: 50)')
    parser.add_argument('--checkpoints', type=str, 
                       help='Comma-separated list of checkpoint filenames (e.g., "epoch1_mini50.pt.gz,epoch1_mini75.pt.gz,epoch1_mini100.pt.gz")')
    parser.add_argument(
        '--checkpoint_dirs', type=str, 
        help=(
            'Comma-separated list of checkpoint directories. Must match number of checkpoints, '
            'or specify one directory for all checkpoints. (e.g., '
            '"loss_weight_sweep_exp0_bs256_98f719_20250724_233408,round2_training")'
        )
    )
    parser.add_argument('--temperature', type=str, default='1.2',
                       help='Temperature for move selection. Can be a single number (e.g., "1.2") or comma-separated list (e.g., "0.3,0.6,1.0"). If a list, must match number of participants (default: 1.2)')
    parser.add_argument('--strategy', type=str, default='policy',
                       choices=['policy', 'fixed_tree', 'mcts'],
                       help='Move selection strategy (default: policy)')
    parser.add_argument('--mcts-sims', type=int, default=200,
                       help='Number of MCTS simulations (default: 200)')
    parser.add_argument('--mcts-c-puct', type=float, default=1.5,
                       help='MCTS c_puct parameter (default: 1.5)')
    parser.add_argument('--enable-gumbel', action='store_true',
                       help='Enable Gumbel-AlphaZero root selection for MCTS (default: False)')
    parser.add_argument('--gumbel-sim-threshold', type=int, default=200,
                       help='Use Gumbel selection when sims <= this threshold (default: 200)')
    parser.add_argument('--gumbel-c-visit', type=float, default=50.0,
                       help='Gumbel-AlphaZero c_visit parameter (default: 50.0)')
    parser.add_argument('--gumbel-c-scale', type=float, default=1.0,
                       help='Gumbel-AlphaZero c_scale parameter (default: 1.0)')
    parser.add_argument('--gumbel-m-candidates', type=int,
                       help='Number of candidates to consider for Gumbel (None for auto)')
    parser.add_argument('--search-widths', type=str,
                       help='Comma-separated search widths for fixed tree search (e.g., "20,10,5")')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed (default: auto-generated from time)')
    parser.add_argument('--no-pie-rule', action='store_true',
                       help='Disable pie rule (pie rule is enabled by default)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    
    # Generate seed if none provided, or use provided seed
    if args.seed is None:
        import time
        args.seed = int(time.time())
        print(f"Auto-generated seed: {args.seed}")
    
    # Determine which checkpoints to use
    if args.checkpoints:
        # User provided specific checkpoint names
        checkpoint_names = [name.strip() for name in args.checkpoints.split(',')]
        
        # Build full paths for checkpoint directories
        if args.checkpoint_dirs:
            checkpoint_dirs = [dir_name.strip() for dir_name in args.checkpoint_dirs.split(',')]
        else:
            # Use the current best model directory from model config
            checkpoint_dirs = [DEFAULT_CHKPT_DIR]

        # Build checkpoint paths
        if len(checkpoint_dirs) == 1:
            # Single directory for all checkpoints
            base_dir = checkpoint_dirs[0]
            checkpoint_paths = [os.path.join(base_dir, fname) for fname in checkpoint_names]
        elif len(checkpoint_dirs) == len(checkpoint_names):
            # One directory per checkpoint
            checkpoint_paths = []
            for dir_name, fname in zip(checkpoint_dirs, checkpoint_names):
                checkpoint_paths.append(os.path.join(dir_name, fname))
        else:
            print(f"ERROR: Number of checkpoint directories ({len(checkpoint_dirs)}) must be 1 or match the number of checkpoints ({len(checkpoint_names)})")
            print(f"  Provided checkpoint names: {checkpoint_names}")
            print(f"  Provided checkpoint directories: {checkpoint_dirs}")
            sys.exit(1)
    else:
        # Use defaults from the current best model directory
        checkpoint_paths = [os.path.join(DEFAULT_CHKPT_DIR, fname) for fname in DEFAULT_CHECKPOINTS]

    # Generate unique player labels for duplicate checkpoints
    player_labels = []
    label_to_checkpoint = {}
    
    # Track checkpoint usage to detect duplicates
    checkpoint_usage = {}
    for i, checkpoint_path in enumerate(checkpoint_paths):
        if checkpoint_path in checkpoint_usage:
            # This is a duplicate - create a unique player label
            checkpoint_usage[checkpoint_path] += 1
            player_label = f"Player{checkpoint_usage[checkpoint_path]}_{os.path.basename(checkpoint_path)}"
        else:
            # First occurrence - use just the filename for consistency
            checkpoint_usage[checkpoint_path] = 1
            player_label = os.path.basename(checkpoint_path)
        
        player_labels.append(player_label)
        label_to_checkpoint[player_label] = checkpoint_path
    
    # Check if we have duplicates and inform the user
    unique_paths = list(dict.fromkeys(checkpoint_paths))
    if len(unique_paths) != len(checkpoint_paths):
        print("INFO: Duplicate checkpoints detected. Using unique player labels:")
        for i, (label, path) in enumerate(zip(player_labels, checkpoint_paths)):
            if label != path:
                print(f"  {label} -> {os.path.basename(path)}")
            else:
                print(f"  {os.path.basename(path)}")
        print()

    # Validate that we have at least 2 checkpoints for a meaningful tournament
    if len(checkpoint_paths) < 2:
        print("ERROR: Need at least 2 unique checkpoints for a tournament.")
        print(f"  Provided checkpoints: {[os.path.basename(p) for p in checkpoint_paths]}")
        sys.exit(1)

    # Check that all checkpoint paths exist before proceeding
    missing_paths = [p for p in checkpoint_paths if not os.path.isfile(p)]
    if missing_paths:
        print("\nERROR: The following checkpoint files do not exist:")
        for p in missing_paths:
            print(f"  {p}")
        print("\nDebug info:")
        if args.checkpoints:
            checkpoint_names = [name.strip() for name in args.checkpoints.split(',')]
            print(f"  Provided checkpoint names: {checkpoint_names}")
            if args.checkpoint_dirs:
                checkpoint_dirs = [dir_name.strip() for dir_name in args.checkpoint_dirs.split(',')]
                print(f"  Provided checkpoint directories: {checkpoint_dirs}")
        print(f"  Constructed checkpoint paths: {checkpoint_paths}")
        print("\nPlease check that the checkpoint files exist and the paths are correct.")
        sys.exit(1)

    # Parse search widths if provided
    search_widths = None
    if args.search_widths:
        search_widths = [int(w.strip()) for w in args.search_widths.split(',')]
    
    # Parse temperature argument
    temperature_values = [float(t.strip()) for t in args.temperature.split(',')]
    
    # Validate temperature configuration
    if len(temperature_values) == 1:
        # Single temperature for all participants
        temperature_config = temperature_values[0]
        participant_temperatures = None
    elif len(temperature_values) == len(player_labels):
        # Per-participant temperatures - use player labels as keys
        temperature_config = temperature_values  # Keep as list for backward compatibility
        participant_temperatures = {label: temp for label, temp in zip(player_labels, temperature_values)}
    else:
        print(f"ERROR: Temperature list length ({len(temperature_values)}) must be 1 or match the number of participants ({len(player_labels)})")
        print(f"  Provided temperatures: {temperature_values}")
        print(f"  Number of participants: {len(player_labels)}")
        print(f"  Participants: {[os.path.basename(p) for p in checkpoint_paths]}")
        sys.exit(1)
    
    # Create strategy configuration
    strategy_config = {}
    if args.strategy == 'mcts':
        strategy_config.update({
            'mcts_sims': args.mcts_sims,
            'mcts_c_puct': args.mcts_c_puct,
            'enable_gumbel_root_selection': args.enable_gumbel,
            'gumbel_sim_threshold': args.gumbel_sim_threshold,
            'gumbel_c_visit': args.gumbel_c_visit,
            'gumbel_c_scale': args.gumbel_c_scale,
        })
        if args.gumbel_m_candidates is not None:
            strategy_config['gumbel_m_candidates'] = args.gumbel_m_candidates
    elif args.strategy == 'fixed_tree':
        if not search_widths:
            print("ERROR: --search-widths is required for fixed_tree strategy")
            sys.exit(1)
        strategy_config['search_widths'] = search_widths
    
    # Create configs
    config = TournamentConfig(
        checkpoint_paths=checkpoint_paths, 
        num_games=args.num_games,
        player_labels=player_labels,
        label_to_checkpoint=label_to_checkpoint
    )
    play_config = TournamentPlayConfig(
        temperature=temperature_config, 
        random_seed=args.seed, 
        pie_rule=not args.no_pie_rule,
        strategy=args.strategy,
        strategy_config=strategy_config,
        search_widths=search_widths,  # Legacy support
        participant_temperatures=participant_temperatures
    )

    # Create log files with descriptive names
    timestamp = (
        f"tournament_{args.num_games}games_{len(checkpoint_paths)}models_"
        f"{datetime.now().strftime('%y%m%d_%H')}"
    )
    LOG_DIR = "data/tournament_play"
    GAMES_FILE = os.path.join(LOG_DIR, f"{timestamp}/tournament.trmph")
    CSV_FILE = os.path.join(LOG_DIR, f"{timestamp}/tournament.csv")
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(GAMES_FILE), exist_ok=True)

    print(f"Tournament Configuration:")
    print(f"  Participants: {len(player_labels)}")
    for i, (label, path) in enumerate(zip(player_labels, checkpoint_paths)):
        if label != path:
            print(f"    {label} ({os.path.basename(path)})")
        else:
            print(f"    {os.path.basename(path)}")
    if args.checkpoints and args.checkpoint_dirs:
        checkpoint_dirs = [dir_name.strip() for dir_name in args.checkpoint_dirs.split(',')]
        print(f"  Checkpoint directories: {checkpoint_dirs}")
    else:
        print(f"  Checkpoint directory: {DEFAULT_CHKPT_DIR}")
    print(f"  Number of games per pair: {args.num_games}")
    print(f"  Strategy: {play_config.strategy}")
    print(f"  Strategy config: {play_config.strategy_config}")
    if participant_temperatures:
        print(f"  Temperature (per participant):")
        for label, temp in participant_temperatures.items():
            model_name = os.path.basename(label_to_checkpoint[label])
            print(f"    {label} ({model_name}): {temp}")
    else:
        print(f"  Temperature: {play_config.temperature}")
    print(f"  Pie rule: {play_config.pie_rule}")
    print(f"  Random seed: {play_config.random_seed}")
    print(f"  Results: {GAMES_FILE}, {CSV_FILE}")
    print()
    
    result, actual_games_file, actual_csv_file = run_round_robin_tournament(
        config,
        verbose=args.verbose,
        log_file=GAMES_FILE,
        csv_file=CSV_FILE,
        play_config=play_config
    )
    print("\nTournament complete!")
    
    # Print actual file paths used (in case collision avoidance changed them)
    if actual_games_file != GAMES_FILE:
        print(f"Note: Tournament results written to {actual_games_file} (original filename was in use)")
    if actual_csv_file != CSV_FILE:
        print(f"Note: CSV results written to {actual_csv_file} (original filename was in use)")
    
    print_comprehensive_tournament_analysis(result) 