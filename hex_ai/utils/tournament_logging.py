import os
from pathlib import Path
import csv
from hex_ai.file_utils import validate_output_directory

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