#!/usr/bin/env python3
"""
Display test cases and trmph strings for verification.
"""

from hex_ai.data_utils import parse_trmph_to_board, display_board, detect_winner
from hex_ai.config import BOARD_SIZE
import numpy as np

def display_test_cases():
    """Display test cases and their board states."""
    
    # Test cases from our tests
    test_cases = [
        {
            "name": "Simple game parsing",
            "trmph": "http://www.trmph.com/hex/board#13,a1b2c3",
            "description": "Three moves: a1(blue), b2(red), c3(blue)"
        },
        {
            "name": "Complex game parsing", 
            "trmph": "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7h8i9j10k11l12m13",
            "description": "13 moves alternating blue/red"
        },
        {
            "name": "Blue winner (7x7)",
            "trmph": "http://www.trmph.com/hex/board#11,a8h1b8h2c8h3d8h4e8h5f8h6g8h7a7a6b6b5c5c4d4d3e3e2f2f1g1",
            "description": "Blue wins by connecting top to bottom"
        },
        {
            "name": "Red winner (7x7)",
            "trmph": "http://www.trmph.com/hex/board#11,a8h1b8h2c8h3d8h4e8h5f8h6g8h7b1a1a3a2c1b2c2b3a4c3b4d3e3d4c5e4f4e5d6f5g5f6g6f7e7g7",
            "description": "Red wins by connecting left to right"
        },
        {
            "name": "Incomplete game",
            "trmph": "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7",
            "description": "7 moves, no winner yet"
        },
        {
            "name": "Blue winner 2 (7x7)",
            "trmph": "http://www.trmph.com/hex/board#11,a8h1b8h2c8h3d8h4e8h5f8h6g8h7d7c7d6c6d5c5d4c4d3c3d2c2d1c1",
            "description": "Another blue winning pattern"
        },
        {
            "name": "Red winner 2 (7x7)",
            "trmph": "http://www.trmph.com/hex/board#11,a8h1b8h2c8h3d8h4e8h5f8h6g8h7g3g4f3f4e3e4d3d4c3c4b3b4a3a4",
            "description": "Another red winning pattern"
        }
    ]
    
    print("=" * 80)
    print("TEST CASES AND BOARD STATES")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   Description: {case['description']}")
        print(f"   Trmph: {case['trmph']}")
        
        try:
            # Parse the board
            board = parse_trmph_to_board(case['trmph'])
            
            # Display board
            print("   Board state:")
            visual = display_board(board, "visual")
            print("   " + visual.replace("\n", "\n   "))
            
            # Show winner
            winner = detect_winner(case['trmph'])
            print(f"   Winner: {winner}")
            
            # Show piece counts
            blue_pieces = np.sum(board == 1)
            red_pieces = np.sum(board == 2)
            print(f"   Blue pieces: {blue_pieces}, Red pieces: {red_pieces}")
            
        except Exception as e:
            print(f"   Error: {e}")
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("TRMPH STRINGS FOR EXTERNAL VERIFICATION")
    print("=" * 80)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{i}. {case['name']}")
        print(f"   {case['trmph']}")
        print(f"   Expected winner: {detect_winner(case['trmph'])}")

if __name__ == "__main__":
    display_test_cases() 