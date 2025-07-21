# NOTE: This test is for the player-to-move utility, which expects 
# 2-channel (blue/red) input. The 3-channel format is only required for 
# model inference/training.
import unittest
import numpy as np
from hex_ai.data_utils import get_player_to_move_from_board
from hex_ai.config import BLUE_PLAYER, RED_PLAYER
from hex_ai.config import BOARD_SIZE

class TestPlayerToMoveUtility(unittest.TestCase):
    def test_empty_board(self):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        self.assertEqual(get_player_to_move_from_board(board), BLUE_PLAYER)

    def test_blues_move(self):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = 1  # One blue stone
        self.assertEqual(get_player_to_move_from_board(board), RED_PLAYER)

    def test_reds_move(self):
        board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board[0, 0, 0] = 1  # Blue
        board[1, 0, 1] = 1  # Red
        self.assertEqual(get_player_to_move_from_board(board), BLUE_PLAYER)

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

if __name__ == "__main__":
    unittest.main() 