#!/usr/bin/env python3
"""Simple test for winner detection implementation."""

import numpy as np
from hex_ai.inference.game_engine import HexGameState, HexGameEngine, RED, BLUE, EMPTY
from hex_ai.inference.board_display import display_hex_board

def test_simple_red_win():
    """Test a simple red horizontal win."""
    board = np.zeros((13, 13), dtype=np.int8)
    for col in range(13):
        board[6, col] = RED
    print("\nSimple Red Win Board:")
    display_hex_board(board)
    state = HexGameState(board=board)
    winner = state._find_winner()
    print(f"Simple red win test: {winner}")
    assert winner == "red", f"Expected red win, got {winner}"

def test_simple_blue_win():
    """Test a simple blue vertical win."""
    board = np.zeros((13, 13), dtype=np.int8)
    for row in range(13):
        board[row, 6] = BLUE
    print("\nSimple Blue Win Board:")
    display_hex_board(board)
    state = HexGameState(board=board)
    winner = state._find_winner()
    print(f"Simple blue win test: {winner}")
    assert winner == "blue", f"Expected blue win, got {winner}"

def test_no_winner():
    """Test that empty board has no winner."""
    board = np.zeros((13, 13), dtype=np.int8)
    print("\nEmpty Board:")
    display_hex_board(board)
    state = HexGameState(board=board)
    winner = state._find_winner()
    print(f"No winner test: {winner}")
    assert winner is None, f"Expected no winner, got {winner}"

def test_partial_red_line():
    """Test that partial red line doesn't win."""
    board = np.zeros((13, 13), dtype=np.int8)
    for col in range(6):
        board[6, col] = RED
    print("\nPartial Red Line Board:")
    display_hex_board(board)
    state = HexGameState(board=board)
    winner = state._find_winner()
    print(f"Partial red line test: {winner}")
    assert winner is None, f"Expected no winner for partial line, got {winner}"

def test_red_win_with_gaps():
    """Test red win with some gaps in the line."""
    board = np.zeros((13, 13), dtype=np.int8)
    red_positions = [(6, 0), (6, 1), (6, 2), (6, 4), (6, 5), (6, 6), 
                     (6, 8), (6, 9), (6, 10), (6, 11), (6, 12)]
    for row, col in red_positions:
        board[row, col] = RED
    print("\nRed Win With Gaps Board:")
    display_hex_board(board)
    state = HexGameState(board=board)
    winner = state._find_winner()
    print(f"Red win with gaps test: {winner}")
    assert winner == "red", f"Expected red win, got {winner}"

if __name__ == "__main__":
    print("Testing winner detection...")
    test_no_winner()
    test_partial_red_line()
    test_simple_red_win()
    test_simple_blue_win()
    test_red_win_with_gaps()
    print("All tests passed!") 