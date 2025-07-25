"""
Run a round-robin tournament between selected model checkpoints.
Each pair plays 10 games (5 as first, 5 as second). Results are logged to disk.
Win rates and Elo scores are printed at the end, with epoch1_mini1.pt as the Elo baseline (1000).

Expected: epoch1_mini1.pt and epoch1_mini5.pt should lose to the others, with 1 being the weakest.

This run uses the pie rule, temperature=0.5, and a fixed random seed for reproducibility.
"""
import os
from hex_ai.inference.tournament import TournamentConfig, run_round_robin_tournament, TournamentPlayConfig

# Directory containing checkpoints
CHKPT_DIR = "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408"

# List of checkpoints to compare
CHECKPOINTS = [
    "epoch1_mini1.pt",
    "epoch1_mini5.pt",
    "epoch1_mini10.pt",
    "epoch1_mini36.pt",
    "epoch2_mini10.pt",
    "epoch2_mini25.pt",
]
CHKPT_PATHS = [os.path.join(CHKPT_DIR, fname) for fname in CHECKPOINTS]

# Tournament config: 10 games per pair (5 as first, 5 as second)
config = TournamentConfig(checkpoint_paths=CHKPT_PATHS, num_games=5)

# Play config: pie rule enabled, temperature=0.5, fixed random seed
play_config = TournamentPlayConfig(temperature=0.5, random_seed=42, pie_rule=True)

# Output files
LOG_DIR = "data/tournament_play"
LOG_FILE = os.path.join(LOG_DIR, "tournament.log")
CSV_FILE = os.path.join(LOG_DIR, "tournament.csv")

if __name__ == "__main__":
    print("Running tournament with checkpoints:")
    for path in CHKPT_PATHS:
        print("  ", path)
    print(f"\nResults will be logged to {LOG_FILE} and {CSV_FILE}\n")
    print(f"Pie rule: {play_config.pie_rule}, temperature: {play_config.temperature}, random_seed: {play_config.random_seed}")
    # Set verbosity=1 to suppress per-move details
    result = run_round_robin_tournament(config, verbose=1, log_file=LOG_FILE, csv_file=CSV_FILE, play_config=play_config)
    print("\nTournament complete. Win rates:")
    result.print_summary()
    print("Elo ratings (baseline: epoch1_mini1.pt = 1000):")
    elos = result.elo_ratings()
    baseline = elos.get(os.path.join(CHKPT_DIR, "epoch1_mini1.pt"), 1500)
    for path in CHECKPOINTS:
        full_path = os.path.join(CHKPT_DIR, path)
        rel_elo = 1000 + (elos[full_path] - baseline) if full_path in elos else 'N/A'
        print(f"  {path:20s} Elo: {rel_elo}") 