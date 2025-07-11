"""
Tests for real data integration and board utilities.

This module tests the integration with real .trmph data, including
board construction, win detection, and display functions.
"""

import unittest
import numpy as np
from hex_ai.data_utils import (
    parse_trmph_to_board, detect_winner, display_board, 
    board_to_trmph, rowcol_to_trmph, trmph_move_to_rowcol
)
from hex_ai.config import BOARD_SIZE


class TestRealDataIntegration(unittest.TestCase):
    """Test real data integration and board utilities."""
    
    def test_simple_game_parsing(self):
        """Test parsing a simple game with known moves."""
        trmph = "http://www.trmph.com/hex/board#13,a1b2c3"
        board = parse_trmph_to_board(trmph)
        
        # Check that pieces are in correct positions
        self.assertEqual(board[0, 0], 1)  # a1 = blue at (0,0)
        self.assertEqual(board[1, 1], 2)  # b2 = red at (1,1)
        self.assertEqual(board[2, 2], 1)  # c3 = blue at (2,2)
        
        # Check that other positions are empty
        self.assertEqual(board[0, 1], 0)  # Should be empty
        self.assertEqual(board[1, 0], 0)  # Should be empty
    
    def test_complex_game_parsing(self):
        """Test parsing a more complex game."""
        trmph = "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7h8i9j10k11l12m13"
        board = parse_trmph_to_board(trmph)
        
        # Check that we have the right number of pieces
        blue_pieces = np.sum(board == 1)
        red_pieces = np.sum(board == 2)
        self.assertEqual(blue_pieces, 7)  # Blue starts, so more pieces
        self.assertEqual(red_pieces, 6)
        
        # Check specific positions
        self.assertEqual(board[0, 0], 1)   # a1 = blue
        self.assertEqual(board[1, 1], 2)   # b2 = red
        self.assertEqual(board[12, 12], 1) # m13 = blue
    
    def test_coordinate_conversion_roundtrip(self):
        """Test that coordinate conversions work in both directions."""
        test_cases = [
            ("a1", (0, 0)),
            ("m13", (12, 12)),
            ("g7", (6, 6)),
            ("d4", (3, 3)),
        ]
        
        for trmph_move, (row, col) in test_cases:
            # Test trmph to rowcol
            converted_row, converted_col = trmph_move_to_rowcol(trmph_move)
            self.assertEqual((converted_row, converted_col), (row, col))
            
            # Test rowcol to trmph
            converted_move = rowcol_to_trmph(row, col)
            self.assertEqual(converted_move, trmph_move)
    
    def test_win_detection(self):
        """Test win detection with known winning games."""
        # Test a simple winning game (blue wins by connecting top to bottom)
        # Using a known working pattern from legacy tests
        blue_winner = "http://www.trmph.com/hex/board#11,a8h1b8h2c8h3d8h4e8h5f8h6g8h7a7a6b6b5c5c4d4d3e3e2f2f1g1"
        winner = detect_winner(blue_winner)
        self.assertEqual(winner, "blue")
        
        # Test a red winning game (red wins by connecting left to right)
        # Using a known working pattern from legacy tests
        red_winner = "http://www.trmph.com/hex/board#11,a8h1b8h2c8h3d8h4e8h5f8h6g8h7b1a1a3a2c1b2c2b3a4c3b4d3e3d4c5e4f4e5d6f5g5f6g6f7e7g7"
        winner = detect_winner(red_winner)
        self.assertEqual(winner, "red")
        
        # Test an incomplete game
        incomplete = "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7"
        winner = detect_winner(incomplete)
        self.assertEqual(winner, "no winner")
    
    def test_board_display(self):
        """Test board display functions."""
        # Create a simple board
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board[0, 0] = 1  # Blue at a1
        board[1, 1] = 2  # Red at b2
        
        # Test matrix display
        matrix_display = display_board(board, "matrix")
        self.assertIn("1", matrix_display)
        self.assertIn("2", matrix_display)
        
        # Test visual display
        visual_display = display_board(board, "visual")
        self.assertIn("B", visual_display)  # Blue piece
        self.assertIn("R", visual_display)  # Red piece
        self.assertIn(".", visual_display)  # Empty spaces
    
    def test_board_to_trmph_conversion(self):
        """Test converting board back to trmph format."""
        # Create a board with known pieces
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board[0, 0] = 1  # Blue at a1
        board[1, 1] = 2  # Red at b2
        board[2, 2] = 1  # Blue at c3
        
        # Convert back to trmph
        trmph = board_to_trmph(board)
        
        # Should contain the moves in some order
        self.assertIn("a1", trmph)
        self.assertIn("b2", trmph)
        self.assertIn("c3", trmph)
        self.assertTrue(trmph.startswith("#13,"))
    
    def test_2channel_board_display(self):
        """Test display functions with 2-channel board format."""
        # Create 2-channel board
        board_2ch = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board_2ch[0, 0, 0] = 1  # Blue at (0,0)
        board_2ch[1, 1, 1] = 1  # Red at (1,1)
        
        # Test display
        visual_display = display_board(board_2ch, "visual")
        self.assertIn("B", visual_display)
        self.assertIn("R", visual_display)
    
    def test_error_handling(self):
        """Test error handling for invalid inputs."""
        # Test invalid trmph move
        with self.assertRaises(ValueError):
            trmph_move_to_rowcol("z1")  # Invalid letter
        
        with self.assertRaises(ValueError):
            trmph_move_to_rowcol("a14")  # Invalid number
        
        # Test invalid coordinates
        with self.assertRaises(ValueError):
            rowcol_to_trmph(13, 0)  # Row out of bounds
        
        with self.assertRaises(ValueError):
            rowcol_to_trmph(0, 13)  # Column out of bounds
        
        # Test invalid display format
        board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        with self.assertRaises(ValueError):
            display_board(board, "invalid_format")
    
    def test_duplicate_move_detection(self):
        """Test that duplicate moves are detected."""
        trmph_with_duplicate = "http://www.trmph.com/hex/board#13,a1a1b2c3"
        
        with self.assertRaises(ValueError):
            parse_trmph_to_board(trmph_with_duplicate)
    
    def test_board_construction_accuracy(self):
        """Test that board construction matches expected patterns."""
        # Test a game with alternating moves
        trmph = "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7h8i9j10k11l12m13"
        board = parse_trmph_to_board(trmph)
        
        # Check that moves alternate properly
        moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board[row, col] != 0:
                    moves.append((row, col, board[row, col]))
        
        # First move should be blue (1), second red (2), etc.
        for i, (row, col, color) in enumerate(moves):
            expected_color = 1 if i % 2 == 0 else 2
            self.assertEqual(color, expected_color, 
                           f"Move {i} at ({row},{col}) should be {expected_color}, got {color}")


class TestWinDetection(unittest.TestCase):
    """Test win detection with various game scenarios."""
    
    def test_blue_winning_patterns(self):
        """Test various blue winning patterns."""
        # Using known working patterns from legacy tests
        blue_winner1 = "http://www.trmph.com/hex/board#11,a8h1b8h2c8h3d8h4e8h5f8h6g8h7a7a6b6b5c5c4d4d3e3e2f2f1g1"
        winner = detect_winner(blue_winner1)
        self.assertEqual(winner, "blue")
        
        # Another blue winning pattern
        blue_winner2 = "http://www.trmph.com/hex/board#11,a8h1b8h2c8h3d8h4e8h5f8h6g8h7d7c7d6c6d5c5d4c4d3c3d2c2d1c1"
        winner = detect_winner(blue_winner2)
        self.assertEqual(winner, "blue")

    def test_red_winning_patterns(self):
        """Test various red winning patterns."""
        # Using known working patterns from legacy tests
        red_winner1 = "http://www.trmph.com/hex/board#11,a8h1b8h2c8h3d8h4e8h5f8h6g8h7b1a1a3a2c1b2c2b3a4c3b4d3e3d4c5e4f4e5d6f5g5f6g6f7e7g7"
        winner = detect_winner(red_winner1)
        self.assertEqual(winner, "red")
        
        # Another red winning pattern
        red_winner2 = "http://www.trmph.com/hex/board#11,a8h1b8h2c8h3d8h4e8h5f8h6g8h7g3g4f3f4e3e4d3d4c3c4b3b4a3a4"
        winner = detect_winner(red_winner2)
        self.assertEqual(winner, "red")
    
    def test_incomplete_games(self):
        """Test games that don't have a winner yet."""
        # Very short game
        short_game = "http://www.trmph.com/hex/board#13,a1b2c3"
        winner = detect_winner(short_game)
        self.assertEqual(winner, "no winner")
        
        # Medium length game
        medium_game = "http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7h8i9j10k11l12"
        winner = detect_winner(medium_game)
        self.assertEqual(winner, "no winner")


if __name__ == '__main__':
    unittest.main() 