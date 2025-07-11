#!/usr/bin/env python3
"""
Debug script to test win detection step by step.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'legacy_code'))

from hex_ai.data_utils import detect_winner, parse_trmph_to_board, display_board
import BoardUtils as bu
import FileConversion as fc
import numpy as np

def test_win_detection_debug():
    """Debug win detection step by step."""
    
    # Test cases with corrected patterns
    test_cases = [
        ("Blue vertical win", "http://www.trmph.com/hex/board#13,a1a2a3a4a5a6a7a8a9a10a11a12a13"),
        ("Red horizontal win", "http://www.trmph.com/hex/board#13,a1b1c1d1e1f1g1h1i1j1k1l1m1"),
    ]
    
    for test_name, trmph_text in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {test_name}")
        print(f"Trmph: {trmph_text}")
        
        # First, let's see what our parser produces
        print("\n=== Our parser ===")
        board = parse_trmph_to_board(trmph_text)
        print("Board shape:", board.shape)
        print("Blue pieces:", np.sum(board == 1))
        print("Red pieces:", np.sum(board == 2))
        print("Board:")
        print(display_board(board, "matrix"))
        
        # Let's also check what the legacy parser produces
        print("\n=== Legacy parser ===")
        try:
            # Get the bare moves
            bare_moves = fc.StripTrmphPreamble(trmph_text, fc.GetEmptyBoardsByModel(13, 13))
            print("Bare moves:", bare_moves)
            
            # Convert to dot-hex format
            dot_hex_boards = fc.trmphToDotHex([bare_moves], boardWidth=13, positions="last")
            print("Dot-hex board shape:", dot_hex_boards["boardList"][0].shape)
            print("Dot-hex board:")
            print(dot_hex_boards["boardList"][0])
            
        except Exception as e:
            print(f"Legacy parser error: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            # Try to initialize connections
            print("\n=== Win detection ===")
            print("Initializing connections...")
            connections = bu.InitBoardConns(trmph_text, playBoardSize=13, modelBoardSize=13)
            print("Connections initialized successfully")
            
            # Try to find winner
            print("Finding winner...")
            winner = bu.FindWinner(connections)
            print(f"Winner: {winner}")
            
            # Test our wrapper function
            print("Testing our wrapper function...")
            our_winner = detect_winner(trmph_text)
            print(f"Our winner: {our_winner}")
            
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_win_detection_debug() 