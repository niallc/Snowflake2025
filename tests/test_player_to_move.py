# NOTE: This test is for the player-to-move utility, which expects 
# 2-channel (blue/red) input. The 3-channel format is only required for 
# model inference/training.
import unittest
import numpy as np
from hex_ai.data_utils import get_player_to_move_from_board
from hex_ai.value_utils import Player, player_to_int, int_to_player
from hex_ai.config import BOARD_SIZE

class TestPlayerToMoveUtility(unittest.TestCase):
    def test_empty_board(self):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.assertEqual(get_player_to_move_from_board(board), Player.BLUE)

    def test_blues_move(self):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = 1  # One blue stone
        self.assertEqual(get_player_to_move_from_board(board), Player.RED)

    def test_reds_move(self):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = 1  # Blue
        board[1, 0, 1] = 1  # Red
        self.assertEqual(get_player_to_move_from_board(board), Player.BLUE)

    def test_illegal_more_red(self):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[1, 0, 0] = 1  # Red
        with self.assertRaises(ValueError):
            get_player_to_move_from_board(board)

    def test_illegal_blue_ahead_by_two(self):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = 1
        board[0, 0, 1] = 1
        self.assertRaises(ValueError, get_player_to_move_from_board, board)

    def test_enum_benefits(self):
        """Test the benefits of using enums over integer constants."""
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        player = get_player_to_move_from_board(board)
        
        # Type safety - can't accidentally compare with invalid values
        self.assertTrue(player == Player.BLUE)
        self.assertFalse(player == Player.RED)
        
        # Clear intent - much more readable than magic numbers
        self.assertEqual(player.value, 0)  # But we prefer Player.BLUE
        
        # Can convert to integer for backward compatibility
        player_int = player_to_int(player)
        self.assertEqual(player_int, 0)
        
        # Can convert back from integer
        player_from_int = int_to_player(player_int)
        self.assertEqual(player_from_int, Player.BLUE)

    def test_enum_comparison(self):
        """Test that enum comparisons work correctly."""
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = 1  # One blue stone
        player = get_player_to_move_from_board(board)
        
        # Direct enum comparison
        self.assertEqual(player, Player.RED)
        self.assertNotEqual(player, Player.BLUE)
        
        # Value comparison
        self.assertEqual(player.value, 1)
        self.assertNotEqual(player.value, 0)

if __name__ == "__main__":
    unittest.main() 