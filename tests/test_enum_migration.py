"""
Test that the enum migration works correctly with the data processing pipeline.
"""

import pytest
import numpy as np
import torch
from hex_ai.data_utils import preprocess_example_for_model, get_player_to_move_from_board
from hex_ai.value_utils import Player
from hex_ai.config import BOARD_SIZE


class TestEnumMigration:
    """Test that the enum migration works correctly."""
    
    def test_preprocess_example_for_model_with_enums(self):
        """Test that preprocess_example_for_model works with the new enum system."""
        # Create a simple example
        board_2ch = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        board_2ch[0, 0, 0] = 1  # One blue piece
        board_2ch[1, 0, 1] = 1  # One red piece
        
        # Create a simple policy (all zeros for this test)
        policy = np.zeros(BOARD_SIZE * BOARD_SIZE, dtype=np.float32)
        
        # Create a simple value
        value = 0.5
        
        example = {
            'board': board_2ch,
            'policy': policy,
            'value': value
        }
        
        # Process the example
        board_3ch, processed_policy, processed_value = preprocess_example_for_model(example)
        
        # Check that the board has the correct shape
        assert board_3ch.shape == (3, BOARD_SIZE, BOARD_SIZE)
        
        # Check that the first two channels match the input
        np.testing.assert_array_equal(board_3ch[0].numpy(), board_2ch[0])
        np.testing.assert_array_equal(board_3ch[1].numpy(), board_2ch[1])
        
        # Check that the third channel (player-to-move) is correct
        # With 1 blue and 1 red piece, it should be blue's turn
        expected_player = get_player_to_move_from_board(board_2ch)
        expected_player_int = expected_player.value
        np.testing.assert_array_equal(
            board_3ch[2].numpy(), 
            np.full((BOARD_SIZE, BOARD_SIZE), expected_player_int, dtype=np.float32)
        )
        
        # Check that policy and value are correct
        assert processed_policy.shape == (BOARD_SIZE * BOARD_SIZE,)
        assert processed_value.shape == (1,) or processed_value.shape == ()
        assert processed_value.item() == value
    
    def test_player_to_move_enum_consistency(self):
        """Test that the enum system is consistent across different board states."""
        # Test empty board
        empty_board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        player_empty = get_player_to_move_from_board(empty_board)
        assert player_empty == Player.BLUE
        assert player_empty.value == 0
        
        # Test board with one blue piece
        one_blue_board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        one_blue_board[0, 0, 0] = 1
        player_one_blue = get_player_to_move_from_board(one_blue_board)
        assert player_one_blue == Player.RED
        assert player_one_blue.value == 1
        
        # Test board with equal pieces
        equal_board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        equal_board[0, 0, 0] = 1  # Blue
        equal_board[1, 0, 1] = 1  # Red
        player_equal = get_player_to_move_from_board(equal_board)
        assert player_equal == Player.BLUE
        assert player_equal.value == 0
    
    def test_enum_backward_compatibility(self):
        """Test that the enum system maintains backward compatibility."""
        from hex_ai.value_utils import player_to_int, int_to_player
        
        # Test conversion functions
        assert player_to_int(Player.BLUE) == 0
        assert player_to_int(Player.RED) == 1
        assert int_to_player(0) == Player.BLUE
        assert int_to_player(1) == Player.RED
        
        # Test that we can still use integer values where needed
        board_2ch = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        player = get_player_to_move_from_board(board_2ch)
        player_int = player.value
        
        # This should work the same as before
        assert player_int == 0
        assert player == Player.BLUE 