#!/usr/bin/env python3
"""
Verify that colors are swapped in the problematic TRMPH file by tracing through the data processing pipeline.
"""

import sys
import numpy as np
from pathlib import Path
import hex_ai.data_utils as data_utils
import hex_ai.data_pipeline as data_pipeline

def analyze_game_start(game_moves):
    """Analyze the start of a game to check for invalid color sequences."""
    moves = game_moves.strip().split()
    if len(moves) < 2:
        return None
    
    # Check first few moves for invalid patterns
    # In Hex, first move should be blue, second should be red, etc.
    # If colors are swapped, we might see red moves before blue moves
    
    # Convert first few moves to board positions to check
    board_state = np.zeros((2, 13, 13), dtype=np.int8)
    
    for i, move in enumerate(moves[:10]):  # Check first 10 moves
        try:
            row, col = data_utils.trmph_move_to_rowcol(move)
            if i % 2 == 0:  # Should be blue (first player)
                board_state[0, row, col] = 1
            else:  # Should be red (second player)
                board_state[1, row, col] = 1
        except:
            continue
    
    # Count pieces
    blue_count = np.sum(board_state[0])
    red_count = np.sum(board_state[1])
    
    # Check for invalid patterns
    if red_count > blue_count:
        return {
            'moves': moves[:10],
            'blue_count': blue_count,
            'red_count': red_count,
            'issue': 'red_count > blue_count'
        }
    
    return None

def trace_game_through_pipeline(game_moves, game_index):
    """Trace a specific game through the data processing pipeline."""
    print(f"\n=== Tracing Game {game_index} ===")
    print(f"Original moves: {' '.join(game_moves.strip().split()[:10])}")
    
    # Step 1: Convert to board state using data_utils
    board_state = np.zeros((2, 13, 13), dtype=np.int8)
    moves = game_moves.strip().split()
    
    for i, move in enumerate(moves[:10]):
        try:
            row, col = data_utils.trmph_move_to_rowcol(move)
            if i % 2 == 0:  # Blue
                board_state[0, row, col] = 1
            else:  # Red
                board_state[1, row, col] = 1
        except:
            continue
    
    blue_count = np.sum(board_state[0])
    red_count = np.sum(board_state[1])
    
    print(f"After conversion: blue={blue_count}, red={red_count}")
    
    # Step 2: Check if this matches any problematic samples from our error analysis
    # We know problematic samples have red_count = blue_count + 1
    if red_count == blue_count + 1:
        print("*** MATCHES PROBLEMATIC PATTERN ***")
        return True
    
    return False

def analyze_trmph_file(file_path: str, max_games: int = 50):
    """Analyze a TRMPH file for problematic games and trace them through the pipeline."""
    file_path = Path(file_path)
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    moves_text = data_utils.strip_trmph_preamble(content)
    games = data_utils.split_trmph_moves(moves_text)
    
    print(f"Analyzing {min(len(games), max_games)} games from {file_path.name}")
    print("=" * 60)
    
    problematic_games = []
    
    for i, game in enumerate(games[:max_games]):
        # Check for problematic game starts
        issue = analyze_game_start(game)
        if issue:
            print(f"Game {i+1}: {issue}")
            problematic_games.append((i+1, game, issue))
        
        # Trace through pipeline
        if trace_game_through_pipeline(game, i+1):
            problematic_games.append((i+1, game, {'issue': 'matches_error_pattern'}))
    
    print(f"\nSummary:")
    print(f"Problematic games found: {len(problematic_games)}")
    
    if problematic_games:
        print("\nProblematic games:")
        for game_num, game, issue in problematic_games:
            print(f"  Game {game_num}: {issue}")
            print(f"    First 10 moves: {' '.join(game.strip().split()[:10])}")
    
    return problematic_games

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python verify_color_swap.py <trmph_file>")
        sys.exit(1)
    
    analyze_trmph_file(sys.argv[1]) 