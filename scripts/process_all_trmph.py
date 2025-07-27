"""
Process all .trmph files into sharded .pkl.gz files with network-ready data.

This is a simple CLI wrapper that imports the main processing logic from
hex_ai.trmph_processing.cli.

DATA FLOW & OUTPUT FORMAT:
- This script finds all .trmph files in the data directory and processes them into training examples.
- Each example includes:
    - board: (2, N, N) numpy array
    - policy: (N*N,) numpy array or None
    - value: float
    - player_to_move: int (0=Blue, 1=Red) [NEW, required for all downstream code]
    - metadata: dict with game_id, position_in_game, winner, etc.
- Output files include a 'source_file' field (the original .trmph file) and are tracked in processing_state.json.
- The game_id in each example's metadata can be mapped back to the original .trmph file using the state file and file lookup utilities in hex_ai/data_utils.py.
- The player_to_move field is critical for correct model training and inference; its absence will cause downstream failures.

"""

# Import the main function from the new location
from hex_ai.trmph_processing.cli import main

if __name__ == "__main__":
    main() 