"""
Run a round-robin tournament between selected model checkpoints.
Each pair plays 10 games (5 as first, 5 as second). Results are logged to disk.
Win rates and Elo scores are printed at the end, with epoch1_mini1.pt as the Elo baseline (1000).

Expected: epoch1_mini1.pt and epoch1_mini5.pt should lose to the others, with 1 being the weakest.

This run uses the pie rule, temperature=0.5, and a fixed random seed for reproducibility.

Example full usage:
PYTHONPATH=. python scripts/run_tournament.py \
  --num-games=100 \
  --checkpoints="epoch1_mini4.pt,epoch2_mini16.pt,epoch1_mini1.pt,epoch1_mini9.pt,epoch2_mini5.pt,epoch2_mini9.pt" \
  --checkpoint_dirs="loss_weight_sweep_exp0_bs256_98f719_20250724_233408,loss_weight_sweep_exp0_bs256_98f719_20250724_233408,loss_weight_sweep_exp1_bs1024_98234d_20250724_233408,loss_weight_sweep_exp1_bs1024_98234d_20250724_233408,loss_weight_sweep_exp1_bs1024_98234d_20250724_233408,loss_weight_sweep_exp1_bs1024_98234d_20250724_233408" \
  --temperature=0.25

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
    "epoch1_mini1.pt",
    "epoch1_mini27.pt",
    "epoch1_mini30.pt",
    "epoch1_mini36.pt",
    "epoch2_mini4.pt",
    "epoch2_mini16.pt",
    "epoch2_mini18.pt",
    "epoch2_mini20.pt",
    "epoch2_mini26.pt",
]

def parse_args():
    parser = argparse.ArgumentParser(description='Run a round-robin tournament between model checkpoints')
    parser.add_argument('--num-games', type=int, default=50, 
                       help='Number of games per pair (default: 50)')
    parser.add_argument('--checkpoints', type=str, 
                       help='Comma-separated list of checkpoint filenames (e.g., "epoch1_mini1.pt,epoch2_mini16.pt")')
    parser.add_argument(
        '--checkpoint_dirs', type=str, 
        help=(
            'Comma-separated list of checkpoint directories (e.g., '
            '"loss_weight_sweep_exp0_bs256_98f719_20250724_233408,'
            'loss_weight_sweep_exp1_bs1024_98234d_20250724_233408")'
        )
    )
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Temperature for move selection (default: 0.3)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--no-pie-rule', action='store_true',
                       help='Disable pie rule (pie rule is enabled by default; use this flag to disable it)')
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
        CHKPT_DIRS = [os.path.join(CHKPT_BASE_DIR, dir_name.strip()) for dir_name in args.checkpoint_dirs.split(',')]
    else:
        CHKPT_DIRS = [DEFAULT_CHKPT_DIR]

    # Build checkpoint paths by pairing each checkpoint name with its corresponding directory
    if len(CHKPT_DIRS) == 1:
        checkpoint_paths = [os.path.join(CHKPT_DIRS[0], fname) for fname in checkpoint_names]
    elif len(CHKPT_DIRS) == len(checkpoint_names):
        checkpoint_paths = [os.path.join(dir_name, fname) for dir_name, fname in zip(CHKPT_DIRS, checkpoint_names)]
    else:
        # If the number of checkpoint directories does not match the number of checkpoint names,
        # and is not 1, this is ambiguous and should be considered an error.
        raise ValueError("Number of checkpoint directories must be 1 or match the number of checkpoint names")

    # Check that all checkpoint paths exist before proceeding
    missing_paths = [p for p in checkpoint_paths if not os.path.isfile(p)]
    if missing_paths:
        print("\nERROR: The following checkpoint files do not exist:")
        for p in missing_paths:
            print(f"  {p}")
        print("\nDebug info:")
        print(f"  Provided checkpoint names: {checkpoint_names}")
        print(f"  Checkpoint directories: {CHKPT_DIRS}")
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
    timestamp = f"test_{args.num_games}games_{len(checkpoint_names)}models"
    LOG_DIR = "data/tournament_play"
    LOG_FILE = os.path.join(LOG_DIR, f"{timestamp}/tournament.log")
    CSV_FILE = os.path.join(LOG_DIR, f"{timestamp}/tournament.csv")
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    print(f"Tournament Configuration:")
    print(f"  Checkpoints: {checkpoint_names}")
    print(f"  Number of games per pair: {args.num_games}")
    print(f"  Temperature: {play_config.temperature}")
    print(f"  Pie rule: {play_config.pie_rule}")
    print(f"  Random seed: {play_config.random_seed}")
    print(f"  Results: {LOG_FILE}, {CSV_FILE}")
    print()
    
    result = run_round_robin_tournament(
        config,
        verbose=args.verbose,
        log_file=LOG_FILE,
        csv_file=CSV_FILE,
        play_config=play_config
    )
    print("\nTournament complete!")
    result.print_detailed_analysis() 