#!/usr/bin/env python3
"""
Analyze TRMPH data for anomalies that might cause color swapping issues.
"""

import argparse
import sys
from pathlib import Path
import hex_ai.data_utils as data_utils
from collections import Counter
from hex_ai.utils.format_conversion import parse_trmph_game_record

def analyze_trmph_file(file_path: str, max_games: int = 100):
    """Analyze a TRMPH file for anomalies and print detailed info about problematic games."""
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"File {file_path} not found")
        return
    
    print(f"Analyzing {file_path.name}...")
    
    try:
        # Load the TRMPH file
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        games = []
        for line_num, line in enumerate(lines, 1):
            try:
                trmph_url, winner = parse_trmph_game_record(line)
                # Strip preamble and get moves
                moves_text = data_utils.strip_trmph_preamble(trmph_url)
                moves = data_utils.split_trmph_moves(moves_text)
                games.append(moves)
            except Exception as e:
                print(f"Error parsing line {line_num}: {e}")
                continue
        
        print(f"Found {len(games)} games. Checking for anomalies in the first {min(max_games, len(games))} games...")
        
        problematic_games = []
        for idx, moves in enumerate(games[:max_games]):
            issues = []
            # Check for duplicate moves
            move_counts = Counter(moves)
            dups = [move for move, count in move_counts.items() if count > 1]
            if dups:
                issues.append(f"Duplicate moves: {dups}")
            # Check for alternating colors (should alternate B/R)
            # In TRMPH, moves alternate by index, so just check length
            if len(moves) > 0:
                if len(moves) % 2 == 0:
                    expected = ['B', 'R'] * (len(moves) // 2)
                else:
                    expected = ['B', 'R'] * (len(moves) // 2) + ['B']
                # We can't check color directly unless moves are annotated, but can check for odd move counts
                if len(moves) < 5:
                    issues.append("Unusually short game")
            # Check for invalid move strings
            for m in moves:
                if not isinstance(m, str) or len(m) < 2:
                    issues.append(f"Invalid move string: {m}")
            if issues:
                problematic_games.append((idx, moves, issues))
        if problematic_games:
            print(f"\nFound {len(problematic_games)} problematic games:")
            for idx, moves, issues in problematic_games:
                print(f"\nGame #{idx}:")
                print(f"  Moves: {moves}")
                print(f"  Issues: {issues}")
        else:
            print(f"No problematic games found in the first {max_games}.")
    except Exception as e:
        print(f"Error analyzing file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Analyze TRMPH data for anomalies.")
    parser.add_argument("file", type=str, help="Path to TRMPH file")
    parser.add_argument("--max-games", type=int, default=100, help="Number of games to check")
    args = parser.parse_args()
    analyze_trmph_file(args.file, max_games=args.max_games)

if __name__ == "__main__":
    main() 