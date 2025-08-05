"""
Run a round-robin tournament between selected model checkpoints.
Each pair plays N games (N/2 as first, N/2 as second). Results are logged to disk.
Win rates and Elo scores are printed at the end.

This script supports comparing models from different directories by specifying
both checkpoint filenames and their corresponding directories.

Examples:

1. Compare models from the same directory:
   PYTHONPATH=. python scripts/run_tournament.py \
     --num-games=50 \
     --checkpoints="epoch1_mini1.pt.gz,epoch2_mini16.pt.gz,epoch2_mini26.pt.gz"

2. Compare models from different directories:
   PYTHONPATH=. python scripts/run_tournament.py \
     --num-games=50 \
     --checkpoints="epoch2_mini4.pt.gz,epoch2_mini26.pt.gz,epoch3_mini13.pt.gz,epoch3_mini69.pt.gz" \
     --checkpoint_dirs="loss_weight_sweep_exp0_bs256_98f719_20250724_233408,loss_weight_sweep_exp0_bs256_98f719_20250724_233408,checkpoints/round2_training,checkpoints/round2_training"

3. Compare old vs new training rounds (convenience example):
   PYTHONPATH=. python scripts/run_tournament.py \
     --num-games=200 \
     --checkpoints="epoch2_mini4.pt.gz,epoch2_mini26.pt.gz,epoch2_mini36.pt.gz,epoch3_mini13.pt.gz,epoch3_mini69.pt.gz,epoch3_mini174.pt.gz" \
     --checkpoint_dirs="loss_weight_sweep_exp0_bs256_98f719_20250724_233408,loss_weight_sweep_exp0_bs256_98f719_20250724_233408,loss_weight_sweep_exp0_bs256_98f719_20250724_233408,checkpoints/round2_training,checkpoints/round2_training,checkpoints/round2_training" \
     --temperature=1.2 \
     --no-pie-rule \
     --seed=43

Note: When using --checkpoint_dirs, the number of directories must match the number of checkpoints,
or you can specify a single directory for all checkpoints.

"""
import argparse
import os
import sys
from hex_ai.inference.tournament import (
    TournamentConfig, run_round_robin_tournament, TournamentPlayConfig
)

# Directory containing checkpoints
CHKPT_BASE_DIR = "checkpoints/hyperparameter_tuning"
DEFAULT_CHKPT_DIR = os.path.join(CHKPT_BASE_DIR, "loss_weight_sweep_exp0_bs256_98f719_20250724_233408")

# Default list of checkpoints to compare
DEFAULT_CHECKPOINTS = [
    "epoch1_mini1.pt.gz",
    "epoch1_mini27.pt.gz",
    "epoch1_mini30.pt.gz",
    "epoch1_mini36.pt.gz",
    "epoch2_mini4.pt.gz",
    "epoch2_mini16.pt.gz",
    "epoch2_mini18.pt.gz",
    "epoch2_mini20.pt.gz",
    "epoch2_mini26.pt.gz",
]

def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a round-robin tournament between model checkpoints',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare models from same directory
  %(prog)s --num-games=50 --checkpoints="epoch1_mini1.pt.gz,epoch2_mini16.pt.gz"
  
  # Compare models from different directories  
  %(prog)s --num-games=50 \\
    --checkpoints="epoch2_mini4.pt.gz,epoch3_mini13.pt.gz" \\
    --checkpoint_dirs="loss_weight_sweep_exp0_bs256_98f719_20250724_233408,round2_training"
        """
    )
    parser.add_argument('--num-games', type=int, default=50, 
                       help='Number of games per pair (default: 50)')
    parser.add_argument('--checkpoints', type=str, 
                       help='Comma-separated list of checkpoint filenames (e.g., "epoch1_mini1.pt.gz,epoch2_mini16.pt.gz")')
    parser.add_argument(
        '--checkpoint_dirs', type=str, 
        help=(
            'Comma-separated list of checkpoint directories. Must match number of checkpoints, '
            'or specify one directory for all checkpoints. (e.g., '
            '"loss_weight_sweep_exp0_bs256_98f719_20250724_233408,round2_training")'
        )
    )
    parser.add_argument('--temperature', type=float, default=1.2,
                       help='Temperature for move selection (default: 0.3)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--no-pie-rule', action='store_true',
                       help='Disable pie rule (pie rule is enabled by default)')
    parser.add_argument('--verbose', type=int, default=1,
                       help='Verbosity level (default: 1)')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    
    # Determine which checkpoints to use
    if args.checkpoints:
        checkpoint_names = [name.strip() for name in args.checkpoints.split(',')]
    else:
        checkpoint_names = DEFAULT_CHECKPOINTS

    # Build full paths for checkpoint directories
    if args.checkpoint_dirs:
        checkpoint_dirs = [dir_name.strip() for dir_name in args.checkpoint_dirs.split(',')]
    else:
        checkpoint_dirs = [DEFAULT_CHKPT_DIR]

    # Build checkpoint paths
    if len(checkpoint_dirs) == 1:
        # Single directory for all checkpoints
        base_dir = checkpoint_dirs[0]
        if not base_dir.startswith('checkpoints/'):
            base_dir = os.path.join(CHKPT_BASE_DIR, base_dir)
        checkpoint_paths = [os.path.join(base_dir, fname) for fname in checkpoint_names]
    elif len(checkpoint_dirs) == len(checkpoint_names):
        # One directory per checkpoint
        checkpoint_paths = []
        for dir_name, fname in zip(checkpoint_dirs, checkpoint_names):
            if not dir_name.startswith('checkpoints/'):
                dir_name = os.path.join(CHKPT_BASE_DIR, dir_name)
            checkpoint_paths.append(os.path.join(dir_name, fname))
    else:
        print(f"ERROR: Number of checkpoint directories ({len(checkpoint_dirs)}) must be 1 or match the number of checkpoints ({len(checkpoint_names)})")
        print(f"  Provided checkpoint names: {checkpoint_names}")
        print(f"  Provided checkpoint directories: {checkpoint_dirs}")
        sys.exit(1)

    # Check that all checkpoint paths exist before proceeding
    missing_paths = [p for p in checkpoint_paths if not os.path.isfile(p)]
    if missing_paths:
        print("\nERROR: The following checkpoint files do not exist:")
        for p in missing_paths:
            print(f"  {p}")
        print("\nDebug info:")
        print(f"  Provided checkpoint names: {checkpoint_names}")
        print(f"  Provided checkpoint directories: {checkpoint_dirs}")
        print(f"  Constructed checkpoint paths: {checkpoint_paths}")
        print("\nPlease check that the checkpoint files exist and the paths are correct.")
        sys.exit(1)

    # Create configs
    config = TournamentConfig(checkpoint_paths=checkpoint_paths, num_games=args.num_games)
    play_config = TournamentPlayConfig(
        temperature=args.temperature, 
        random_seed=args.seed, 
        pie_rule=not args.no_pie_rule
    )

    # Create log files with descriptive names
    timestamp = f"tournament_{args.num_games}games_{len(checkpoint_names)}models"
    LOG_DIR = "data/tournament_play"
    GAMES_FILE = os.path.join(LOG_DIR, f"{timestamp}/tournament.trmph")
    CSV_FILE = os.path.join(LOG_DIR, f"{timestamp}/tournament.csv")
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(GAMES_FILE), exist_ok=True)

    print(f"Tournament Configuration:")
    print(f"  Checkpoints: {checkpoint_names}")
    print(f"  Checkpoint directories: {checkpoint_dirs}")
    print(f"  Number of games per pair: {args.num_games}")
    print(f"  Temperature: {play_config.temperature}")
    print(f"  Pie rule: {play_config.pie_rule}")
    print(f"  Random seed: {play_config.random_seed}")
    print(f"  Results: {GAMES_FILE}, {CSV_FILE}")
    print()
    
    result = run_round_robin_tournament(
        config,
        verbose=args.verbose,
        log_file=GAMES_FILE,
        csv_file=CSV_FILE,
        play_config=play_config
    )
    print("\nTournament complete!")
    result.print_detailed_analysis() 