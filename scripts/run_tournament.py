"""
Run a round-robin tournament between selected model checkpoints.
Each pair plays 10 games (5 as first, 5 as second). Results are logged to disk.
Win rates and Elo scores are printed at the end, with epoch1_mini1.pt as the Elo baseline (1000).

Expected: epoch1_mini1.pt and epoch1_mini5.pt should lose to the others, with 1 being the weakest.

This run uses the pie rule, temperature=0.5, and a fixed random seed for reproducibility.
"""
import os
import argparse
from hex_ai.inference.tournament import TournamentConfig, run_round_robin_tournament, TournamentPlayConfig

# Directory containing checkpoints
CHKPT_DIR="checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408"

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
    parser.add_argument('--num-games', type=int, default=500, 
                       help='Number of games per pair (default: 500)')
    parser.add_argument('--checkpoints', type=str, 
                       help='Comma-separated list of checkpoint filenames (e.g., "epoch1_mini1.pt,epoch2_mini16.pt")')
    parser.add_argument('--temperature', type=float, default=0.3,
                       help='Temperature for move selection (default: 0.3)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--no-pie-rule', action='store_true',
                       help='Disable pie rule (default: enabled)')
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
    
    # Build full paths
    checkpoint_paths = [os.path.join(CHKPT_DIR, fname) for fname in checkpoint_names]
    
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