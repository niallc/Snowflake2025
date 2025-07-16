#!/usr/bin/env python3
"""
Find games that start with invalid color sequences indicating color swapping.
"""

import sys
from pathlib import Path
import hex_ai.data_utils as data_utils

def analyze_game_starts(file_path: str, max_games: int = 100):
    """Analyze game starts to find invalid color sequences."""
    file_path = Path(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split into individual games (each line is a game)
    games = content.strip().split('\n')
    
    print(f"Analyzing {min(len(games), max_games)} games from {file_path.name}")
    print("=" * 60)
    
    problematic_games = []
    
    for i, game_line in enumerate(games[:max_games]):
        # Extract just the moves part (after the URL)
        if '#' in game_line:
            moves_part = game_line.split('#')[1].split()[0]  # Get moves before the result
        else:
            continue
            
        # Get the first few moves
        first_moves = moves_part[:10]  # Look at first 10 characters
        
        # In Hex, first move should be blue, second should be red, etc.
        # If colors are swapped, we might see patterns that indicate this
        
        # For now, let's just print the first few moves of each game
        # and look for patterns manually
        print(f"Game {i+1}: First 10 chars: '{first_moves}'")
        
        # Check if this game has any obvious issues
        # (We'll need to implement proper move parsing to do this properly)
        
    print(f"\nAnalyzed {min(len(games), max_games)} games")
    print("Look for patterns in the first moves that might indicate color swapping")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python find_invalid_game_starts.py <trmph_file>")
        sys.exit(1)
    
    analyze_game_starts(sys.argv[1]) 