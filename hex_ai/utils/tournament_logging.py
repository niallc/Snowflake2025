import os
from pathlib import Path
import csv
from datetime import datetime
from typing import Dict, Any, Optional
from hex_ai.file_utils import validate_output_directory
from hex_ai.system_utils import get_git_commit_info

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
                                 num_games: int, play_config, board_size: int = 13):
    """
    Write header information to a tournament .trmph file.
    
    Args:
        trmph_file: Path to the .trmph file
        checkpoint_paths: List of checkpoint file paths
        num_games: Number of games per pair
        play_config: TournamentPlayConfig object
        board_size: Board size (default 13)
    """
    # Build metadata dictionary
    metadata = {
        "Board size": board_size,
        "Number of games per pair": num_games,
        "Models": [os.path.basename(p) for p in checkpoint_paths],
        "Strategy": play_config.strategy,
        "Temperature": play_config.temperature,
        "Pie rule": play_config.pie_rule,
    }
    
    # Add strategy-specific config
    if play_config.strategy_config:
        for key, value in play_config.strategy_config.items():
            metadata[key] = value
    
    # Use generic header writer
    write_trmph_header(trmph_file, "Tournament games", metadata, play_config.random_seed)

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