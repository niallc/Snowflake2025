import os
from pathlib import Path
import csv
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from hex_ai.file_utils import validate_output_directory
from hex_ai.system_utils import get_git_commit_info

def find_available_filename(base_path: str) -> str:
    """
    Find an available filename by appending an index if the base path exists.
    
    Args:
        base_path: The base file path to check
        
    Returns:
        An available file path (either the original or with an index appended)
        
    Example:
        If "tournament.trmph" exists, returns "tournament_2.trmph"
        If "tournament_2.trmph" also exists, returns "tournament_3.trmph"
        And so on...
    """
    if not os.path.exists(base_path):
        return base_path
    
    # Split the path into directory, filename, and extension
    path_obj = Path(base_path)
    directory = path_obj.parent
    stem = path_obj.stem  # filename without extension
    suffix = path_obj.suffix  # extension including the dot
    
    # Find the first available index
    index = 2
    while True:
        new_filename = f"{stem}_{index}{suffix}"
        new_path = directory / new_filename
        if not new_path.exists():
            return str(new_path)
        index += 1

def write_trmph_header(trmph_file: str, header_type: str, metadata: Dict[str, Any], 
                      random_seed: Optional[int] = None):
    """
    Write a generic header to a .trmph file.
    
    Args:
        trmph_file: Path to the .trmph file
        header_type: Type of header (e.g., "Self-play games", "Tournament games")
        metadata: Dictionary of metadata to include in header
        random_seed: Random seed to include (if provided)
    """
    output_path = Path(trmph_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get git info
    git_info = get_git_commit_info()
    
    with open(output_path, 'w') as f:
        f.write(f"# {header_type} - {datetime.now().isoformat()}\n")
        
        # Write metadata
        for key, value in metadata.items():
            f.write(f"# {key}: {value}\n")
        
        # Write random seed if provided
        if random_seed is not None:
            f.write(f"# Random seed: {random_seed}\n")
        
        f.write(f"# Git commit: {git_info['status']}\n")
        f.write("# Format: trmph_string winner\n")

def append_trmph_winner_line(trmph_sequence: str, winner: str, output_file: str):
    """
    Append a line to the tournament log file in the format:
    <trmph_move_sequence> <w>\n
    where w = 'b' if blue (first mover) wins, 'r' if red (second mover) wins.
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'a') as f:
        f.write(f"{trmph_sequence} {winner}\n")

def write_tournament_trmph_header(trmph_file: str, checkpoint_paths: list, 
                                 num_games: int, play_config, board_size: int = 13,
                                 player_labels: list = None, participant_temperatures: dict = None,
                                 strategy_configs: list = None):
    """
    Write header information to a tournament .trmph file.
    
    Args:
        trmph_file: Path to the .trmph file
        checkpoint_paths: List of checkpoint file paths
        num_games: Number of games per pair
        play_config: TournamentPlayConfig object
        board_size: Board size (default 13)
        player_labels: List of player labels (for duplicate model support)
        participant_temperatures: Dict mapping player labels to their temperatures
        strategy_configs: List of StrategyConfig objects for detailed strategy information
        
    Returns:
        The actual file path used (may be different from input if collision avoidance was needed)
    """
    # Find an available filename to avoid collisions
    actual_file_path = find_available_filename(trmph_file)
    
    # Build metadata dictionary
    metadata = {
        "Board size": board_size,
        "Number of games per pair": num_games,
        "Models": [os.path.basename(p) for p in checkpoint_paths],
        "Strategy": play_config.strategy,
        "Temperature": play_config.temperature,
        "Pie rule": play_config.pie_rule,
    }
    
    # Add detailed strategy information if available
    if strategy_configs:
        strategy_details = []
        for i, config in enumerate(strategy_configs):
            model_name = os.path.basename(checkpoint_paths[i]) if i < len(checkpoint_paths) else "unknown"
            strategy_info = f"{model_name}: {config.strategy_type}"
            
            # Add strategy-specific parameters
            if config.strategy_type.startswith('mcts_'):
                # Extract simulation count from strategy type (e.g., "mcts_140" -> 140)
                try:
                    sims = int(config.strategy_type.split('_')[1])
                    strategy_info += f" (sims={sims})"
                except (IndexError, ValueError):
                    pass
                
                # Add MCTS-specific config
                if hasattr(config, 'config') and config.config:
                    mcts_config = config.config
                    if 'c_puct' in mcts_config:
                        strategy_info += f" (c_puct={mcts_config['c_puct']})"
                    if 'enable_gumbel' in mcts_config and mcts_config['enable_gumbel']:
                        strategy_info += " (gumbel=True)"
                        if 'gumbel_sim_threshold' in mcts_config:
                            strategy_info += f" (gumbel_threshold={mcts_config['gumbel_sim_threshold']})"
                    else:
                        strategy_info += " (gumbel=False)"
            
            # Add temperature if different from global
            if hasattr(config, 'temperature') and config.temperature is not None:
                strategy_info += f" (temp={config.temperature})"
            
            strategy_details.append(strategy_info)
        
        if strategy_details:
            metadata["Strategy details"] = strategy_details
    
    # Add per-participant temperature information if available
    if participant_temperatures and player_labels:
        temp_info = []
        for label in player_labels:
            if label in participant_temperatures:
                temp_info.append(f"{label}: {participant_temperatures[label]}")
        if temp_info:
            metadata["Temperature (per participant)"] = temp_info
    
    # Add strategy-specific config
    if play_config.strategy_config:
        for key, value in play_config.strategy_config.items():
            metadata[key] = value
    
    # Use generic header writer
    write_trmph_header(actual_file_path, "Tournament games", metadata, play_config.random_seed)
    
    return actual_file_path

def log_game_csv(row: dict, csv_file: str, headers: list = None):
    """
    Log a game result as a row in a CSV file. If the file does not exist, write headers first.
    Args:
        row: dict of game info (e.g., model_a, model_b, color_a, trmph, winner, etc.)
        csv_file: path to CSV file
        headers: optional list of headers (if not provided, use row.keys())
    """
    csv_path = Path(csv_file)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    if headers is None:
        headers = list(row.keys())
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

def find_available_csv_filename(base_path: str) -> str:
    """
    Find an available CSV filename by appending an index if the base path exists.
    Similar to find_available_filename but specifically for CSV files.
    """
    return find_available_filename(base_path) 