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
    "epoch2_mini10.pt",
]
CHKPT_PATHS = [os.path.join(CHKPT_DIR, fname) for fname in CHECKPOINTS]


config = TournamentConfig(checkpoint_paths=CHKPT_PATHS, num_games=3)
play_config = TournamentPlayConfig(temperature=0.01, random_seed=42, pie_rule=False)

LOG_DIR = "data/tournament_play"
LOG_FILE = os.path.join(LOG_DIR, "test2/tournament.log")
CSV_FILE = os.path.join(LOG_DIR, "test2/tournament.csv")

if __name__ == "__main__":
    print("Quick test tournament: epoch1_mini1.pt vs. epoch2_mini10.pt")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"Temperature: {play_config.temperature}, Pie rule: {play_config.pie_rule}, Seed: {play_config.random_seed}")
    print(f"Results: {LOG_FILE}, {CSV_FILE}\n")
    result = run_round_robin_tournament(
        config,
        verbose=2,
        log_file=LOG_FILE,
        csv_file=CSV_FILE,
        play_config=play_config
    )
    print("\nTournament complete. Win rates:")
    result.print_summary()
    print("Elo ratings:")
    elos = result.elo_ratings()
    for path in CHECKPOINTS:
        full_path = os.path.join(CHKPT_DIR, path)
        # rel_elo = 1000 + (elos[full_path] - baseline) if full_path in elos else 'N/A'
        print(f"  {path:20s} Elo: {elos.get(full_path, 'N/A')}") 