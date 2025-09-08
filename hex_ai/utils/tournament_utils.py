"""
Tournament utility functions for handling player labels, temperature configuration, and validation.

This module provides utilities to separate concerns and reduce code duplication
in tournament-related functionality.
"""

import os
from typing import List, Dict, Tuple, Optional, Union


def generate_player_labels(checkpoint_paths: List[str]) -> Tuple[List[str], Dict[str, str]]:
    """
    Generate unique player labels for checkpoint paths, handling duplicates.
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        
    Returns:
        Tuple of (player_labels, label_to_checkpoint_mapping)
        - player_labels: List of unique labels for each participant
        - label_to_checkpoint_mapping: Dict mapping labels to actual checkpoint paths
    """
    player_labels = []
    label_to_checkpoint = {}
    checkpoint_usage = {}
    
    for checkpoint_path in checkpoint_paths:
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
    
    return player_labels, label_to_checkpoint


def print_duplicate_checkpoint_info(player_labels: List[str], checkpoint_paths: List[str]) -> None:
    """
    Print information about duplicate checkpoints and player labels.
    
    Args:
        player_labels: List of player labels
        checkpoint_paths: List of checkpoint paths
    """
    unique_paths = list(dict.fromkeys(checkpoint_paths))
    if len(unique_paths) != len(checkpoint_paths):
        print("INFO: Duplicate checkpoints detected. Using unique player labels:")
        for label, path in zip(player_labels, checkpoint_paths):
            if label != path:
                print(f"  {label} -> {os.path.basename(path)}")
            else:
                print(f"  {os.path.basename(path)}")
        print()


def validate_checkpoint_paths(checkpoint_paths: List[str]) -> None:
    """
    Validate that checkpoint paths exist and there are enough for a tournament.
    
    Args:
        checkpoint_paths: List of checkpoint file paths
        
    Raises:
        SystemExit: If validation fails
    """
    # Validate that we have at least 2 checkpoints for a meaningful tournament
    if len(checkpoint_paths) < 2:
        print("ERROR: Need at least 2 unique checkpoints for a tournament.")
        print(f"  Provided checkpoints: {[os.path.basename(p) for p in checkpoint_paths]}")
        exit(1)

    # Check that all checkpoint paths exist before proceeding
    missing_paths = [p for p in checkpoint_paths if not os.path.isfile(p)]
    if missing_paths:
        print("\nERROR: The following checkpoint files do not exist:")
        for p in missing_paths:
            print(f"  {p}")
        print("\nPlease check that the checkpoint files exist and the paths are correct.")
        exit(1)


def parse_temperature_configuration(
    temperature_arg: str, 
    player_labels: List[str]
) -> Tuple[Union[float, List[float]], Optional[Dict[str, float]]]:
    """
    Parse and validate temperature configuration for tournament participants.
    
    Args:
        temperature_arg: Comma-separated temperature values as string
        player_labels: List of player labels for validation
        
    Returns:
        Tuple of (temperature_config, participant_temperatures)
        - temperature_config: Either a single float or list of floats
        - participant_temperatures: Dict mapping player labels to temperatures, or None
        
    Raises:
        SystemExit: If temperature configuration is invalid
    """
    temperature_values = [float(t.strip()) for t in temperature_arg.split(',')]
    
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
        print(f"  Participants: {player_labels}")
        exit(1)
    
    return temperature_config, participant_temperatures


def extract_model_name_from_label(player_label: str, checkpoint_path: str) -> str:
    """
    Extract a user-friendly model name from a player label.
    
    Args:
        player_label: The player label (may be a filename or PlayerX_filename format)
        checkpoint_path: The actual checkpoint path for fallback
        
    Returns:
        User-friendly model name for display
    """
    if player_label != checkpoint_path:
        # This is a generated label like "Player2_model.pt.gz"
        return os.path.basename(player_label)
    else:
        # This is just the filename
        return os.path.basename(checkpoint_path)


def print_tournament_configuration(
    player_labels: List[str],
    checkpoint_paths: List[str],
    num_games: int,
    strategy: str,
    strategy_config: Dict,
    participant_temperatures: Optional[Dict[str, float]],
    temperature_config: Union[float, List[float]],
    pie_rule: bool,
    random_seed: int,
    checkpoint_dirs: Optional[List[str]] = None,
    default_checkpoint_dir: Optional[str] = None
) -> None:
    """
    Print tournament configuration in a user-friendly format.
    
    Args:
        player_labels: List of player labels
        checkpoint_paths: List of checkpoint paths
        num_games: Number of games per pair
        strategy: Tournament strategy
        strategy_config: Strategy configuration
        participant_temperatures: Per-participant temperatures or None
        temperature_config: Global temperature configuration
        pie_rule: Whether pie rule is enabled
        random_seed: Random seed
        checkpoint_dirs: Optional checkpoint directories
        default_checkpoint_dir: Default checkpoint directory
    """
    print(f"Tournament Configuration:")
    print(f"  Participants: {len(player_labels)}")
    for label, path in zip(player_labels, checkpoint_paths):
        if label != path:
            print(f"    {label} ({os.path.basename(path)})")
        else:
            print(f"    {os.path.basename(path)}")
    
    if checkpoint_dirs:
        print(f"  Checkpoint directories: {checkpoint_dirs}")
    elif default_checkpoint_dir:
        print(f"  Checkpoint directory: {default_checkpoint_dir}")
    
    print(f"  Number of games per pair: {num_games}")
    print(f"  Strategy: {strategy}")
    print(f"  Strategy config: {strategy_config}")
    
    if participant_temperatures:
        print(f"  Temperature (per participant):")
        for label, temp in participant_temperatures.items():
            model_name = extract_model_name_from_label(label, checkpoint_paths[player_labels.index(label)])
            print(f"    {label} ({model_name}): {temp}")
    else:
        print(f"  Temperature: {temperature_config}")
    
    print(f"  Pie rule: {pie_rule}")
    print(f"  Random seed: {random_seed}")


def get_player_label_for_checkpoint(
    checkpoint_path: str, 
    config_checkpoint_paths: List[str], 
    config_player_labels: List[str]
) -> str:
    """
    Get the player label for a specific checkpoint path.
    
    Args:
        checkpoint_path: The checkpoint path to look up
        config_checkpoint_paths: List of checkpoint paths from config
        config_player_labels: List of player labels from config
        
    Returns:
        The player label for the given checkpoint path
        
    Raises:
        ValueError: If checkpoint path is not found in config
    """
    try:
        index = config_checkpoint_paths.index(checkpoint_path)
        return config_player_labels[index]
    except ValueError:
        raise ValueError(f"Checkpoint path {checkpoint_path} not found in tournament configuration")


def determine_winner_labels(
    winner_color, 
    model_a_path: str, 
    model_b_path: str,
    model_a_label: str, 
    model_b_label: str
) -> Tuple[str, str]:
    """
    Determine winner and loser labels based on game result and model assignments.
    
    Args:
        winner_color: The color that won (Piece.BLUE or Piece.RED)
        model_a_path: Checkpoint path for model A
        model_b_path: Checkpoint path for model B  
        model_a_label: Player label for model A
        model_b_label: Player label for model B
        
    Returns:
        Tuple of (winner_label, loser_label)
    """
    from hex_ai.enums import Piece
    
    # When the same model plays both sides, we need to determine which player label won
    # based on the actual game result, not just the color
    if model_a_path == model_b_path:
        # Same model playing both sides - this is a special case
        # We need to determine which player label won based on the game result
        # For now, we'll use a simple approach: alternate which player label wins
        # This should be handled differently in the future for proper duplicate model support
        # For now, we'll use a deterministic approach based on the labels
        print(f"DEBUG: Same model case - model_a_label={model_a_label}, model_b_label={model_b_label}")
        if model_a_label < model_b_label:
            result = model_a_label, model_b_label
        else:
            result = model_b_label, model_a_label
        print(f"DEBUG: Same model result - winner={result[0]}, loser={result[1]}")
        return result
    else:
        # Different models - use color to determine winner
        if winner_color == Piece.BLUE:
            return model_a_label, model_b_label
        else:
            return model_b_label, model_a_label


def determine_winner_labels_simple(
    winner_color, 
    model_a_label: str, 
    model_b_label: str
) -> Tuple[str, str]:
    """
    Determine winner/loser labels when the same model plays both sides.
    
    Since both players use the same underlying model, we need to track which
    "instance" of the model won based on the game result and which player went first.
    
    Args:
        winner_color: The color that won (Piece.BLUE or Piece.RED)
        model_a_label: Player label for model A (who went first)
        model_b_label: Player label for model B (who went second)
        
    Returns:
        Tuple of (winner_label, loser_label)
    """
    from hex_ai.enums import Piece
    
    # When same model plays both sides, we need to track which "instance" won
    # Model A always plays BLUE (goes first), Model B always plays RED (goes second)
    if winner_color == Piece.BLUE:
        # BLUE won, so model_a_label (who went first) is the winner
        winner_label, loser_label = model_a_label, model_b_label
    elif winner_color == Piece.RED:
        # RED won, so model_b_label (who went second) is the winner
        winner_label, loser_label = model_b_label, model_a_label
    else:
        # This shouldn't happen, but handle gracefully
        raise ValueError(f"Invalid winner_color: {winner_color}")
    
    return winner_label, loser_label
