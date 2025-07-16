#!/usr/bin/env python3
"""
Search for a board state's move sequence in TRMPH data using existing utilities.
"""

import argparse
import numpy as np
import sys
from pathlib import Path
import hex_ai.data_utils as data_utils

def board_to_moves(board_state):
    """Convert a 2-channel board state back to a sequence of moves using existing utilities."""
    if board_state.shape[0] != 2:
        raise ValueError("Expected 2-channel board state")
    
    blue_positions = np.where(board_state[0] == 1)
    red_positions = np.where(board_state[1] == 1)
    
    moves = []
    
    # Convert positions to move strings using existing utility
    for i in range(len(blue_positions[0])):
        row, col = blue_positions[0][i], blue_positions[1][i]
        move = data_utils.rowcol_to_trmph(row, col)
        moves.append(move)
    
    for i in range(len(red_positions[0])):
        row, col = red_positions[0][i], red_positions[1][i]
        move = data_utils.rowcol_to_trmph(row, col)
        moves.append(move)
    
    return moves

def search_moves_in_trmph(trmph_file, target_moves):
    """Search for a sequence of moves in a TRMPH file."""
    with open(trmph_file, 'r') as f:
        content = f.read()
    
    moves_text = data_utils.strip_trmph_preamble(content)
    games = data_utils.split_trmph_moves(moves_text)
    
    print(f"Searching for moves: {target_moves}")
    print(f"Total games in file: {len(games)}")
    
    found_games = []
    
    for game_idx, game_moves in enumerate(games):
        # Parse the moves for this game
        game_move_list = []
        for move_str in game_moves.split():
            if move_str.strip():
                game_move_list.append(move_str.strip())
        
        # Check if this game contains our target moves
        if len(game_move_list) >= len(target_moves):
            # Check if the first N moves match
            if game_move_list[:len(target_moves)] == target_moves:
                found_games.append((game_idx, game_move_list))
                print(f"Found match in game {game_idx}:")
                print(f"  Game moves: {game_move_list}")
                print(f"  Target moves: {target_moves}")
                print()
    
    return found_games

def main():
    parser = argparse.ArgumentParser(description="Search for board state's moves in TRMPH data")
    parser.add_argument("--trmph-file", required=True, help="TRMPH file to search in")
    parser.add_argument("--moves", nargs="+", help="Direct move sequence to search for")
    parser.add_argument("--blue-positions", nargs="+", type=int, help="Blue piece positions as row,col pairs")
    parser.add_argument("--red-positions", nargs="+", type=int, help="Red piece positions as row,col pairs")
    
    args = parser.parse_args()
    
    if args.moves:
        target_moves = args.moves
    elif args.blue_positions or args.red_positions:
        # Create a board state from positions
        board_state = np.zeros((2, 13, 13))
        
        if args.blue_positions:
            for i in range(0, len(args.blue_positions), 2):
                if i + 1 < len(args.blue_positions):
                    row, col = args.blue_positions[i], args.blue_positions[i+1]
                    board_state[0, row, col] = 1
        
        if args.red_positions:
            for i in range(0, len(args.red_positions), 2):
                if i + 1 < len(args.red_positions):
                    row, col = args.red_positions[i], args.red_positions[i+1]
                    board_state[1, row, col] = 1
        
        target_moves = board_to_moves(board_state)
    else:
        print("Please provide either --moves or --blue-positions/--red-positions")
        return
    
    found = search_moves_in_trmph(args.trmph_file, target_moves)
    
    if not found:
        print("No matching games found!")
    else:
        print(f"Found {len(found)} matching games")

if __name__ == "__main__":
    main() 