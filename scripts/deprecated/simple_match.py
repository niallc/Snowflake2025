"""
Simple match runner between two model checkpoints.
Runs 3 games with each model as blue and 3 as red.
Uses model1/model2 naming to avoid confusion with colors.
"""
import os
import random
import numpy as np
from hex_ai.inference.tournament import TournamentPlayConfig, play_single_game
from hex_ai.inference.simple_model_inference import SimpleModelInference
from hex_ai.config import BLUE_PLAYER, RED_PLAYER

def run_simple_match(checkpoint1: str, checkpoint2: str, 
                    games_per_color: int = 3,
                    temperature: float = 0.03,
                    random_seed: int = 42,
                    pie_rule: bool = False,
                    verbose: int = 1):
    """
    Run a simple match between two checkpoints.
    Each model plays games_per_color games as blue and games_per_color as red.
    """
    print(f"Loading models...")
    print(f"Model 1: {checkpoint1}")
    print(f"Model 2: {checkpoint2}")
    
    # Load models
    model1 = SimpleModelInference(checkpoint1)
    model2 = SimpleModelInference(checkpoint2)
    
    # Create play config
    play_config = TournamentPlayConfig(
        temperature=temperature, 
        random_seed=random_seed, 
        pie_rule=pie_rule
    )
    
    print(f"\nMatch configuration:")
    print(f"Games per color: {games_per_color}")
    print(f"Temperature: {temperature}")
    print(f"Random seed: {random_seed}")
    print(f"Pie rule: {pie_rule}")
    print(f"Total games: {games_per_color * 2}")
    print("-" * 50)
    
    # Track results
    model1_wins = 0
    model2_wins = 0
    model1_as_blue_wins = 0
    model1_as_red_wins = 0
    
    # Games with model1 as blue
    print(f"\nGames with {os.path.basename(checkpoint1)} as blue:")
    for i in range(games_per_color):
        print(f"\nGame {i+1}/{games_per_color}:")
        result, trmph_str, winner_char, swap_decision = play_single_game(
            model1, model2, board_size=13, 
            verbose=verbose, play_config=play_config
        )
        if result == "1":  # model1 wins
            model1_wins += 1
            model1_as_blue_wins += 1
        else:  # result == "2"
            model2_wins += 1
    
    # Games with model2 as blue
    print(f"\nGames with {os.path.basename(checkpoint2)} as blue:")
    for i in range(games_per_color):
        print(f"\nGame {i+1}/{games_per_color}:")
        result, trmph_str, winner_char, swap_decision = play_single_game(
            model2, model1, board_size=13, 
            verbose=verbose, play_config=play_config
        )
        if result == "1":  # model_1 wins (which is model2 in this call)
            model2_wins += 1
        else:  # result == "2" - model_2 wins (which is model1 in this call)
            model1_wins += 1
            model1_as_red_wins += 1
    
    # Print results
    total_games = games_per_color * 2
    print("\n" + "=" * 50)
    print("MATCH RESULTS")
    print("=" * 50)
    print(f"{os.path.basename(checkpoint1)}: {model1_wins} wins ({model1_wins/total_games*100:.1f}%)")
    print(f"  - As blue: {model1_as_blue_wins}/{games_per_color} ({model1_as_blue_wins/games_per_color*100:.1f}%)")
    print(f"  - As red: {model1_as_red_wins}/{games_per_color} ({model1_as_red_wins/games_per_color*100:.1f}%)")
    print(f"{os.path.basename(checkpoint2)}: {model2_wins} wins ({model2_wins/total_games*100:.1f}%)")
    print(f"  - As blue: {games_per_color - model1_as_blue_wins}/{games_per_color} ({(games_per_color - model1_as_blue_wins)/games_per_color*100:.1f}%)")
    print(f"  - As red: {games_per_color - model1_as_red_wins}/{games_per_color} ({(games_per_color - model1_as_red_wins)/games_per_color*100:.1f}%)")
    
    if model1_wins > model2_wins:
        print(f"\n🏆 WINNER: {os.path.basename(checkpoint1)}")
    elif model2_wins > model1_wins:
        print(f"\n🏆 WINNER: {os.path.basename(checkpoint2)}")
    else:
        print(f"\n🤝 TIE")

if __name__ == "__main__":
    # Configuration
    CHKPT_DIR = "checkpoints/hyperparameter_tuning/loss_weight_sweep_exp0_bs256_98f719_20250724_233408"
    
    # List of checkpoints to compare
    CHECKPOINTS = [
        "epoch1_mini1.pt",
        "epoch2_mini10.pt",
    ]
    
    checkpoint1 = os.path.join(CHKPT_DIR, CHECKPOINTS[0])
    checkpoint2 = os.path.join(CHKPT_DIR, CHECKPOINTS[1])
    
    # Run the match
    run_simple_match(
        checkpoint1=checkpoint1,
        checkpoint2=checkpoint2,
        games_per_color=3,
        temperature=0.03,
        random_seed=42,
        pie_rule=False,
        verbose=1
    ) 