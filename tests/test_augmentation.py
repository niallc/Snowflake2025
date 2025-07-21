import unittest
import numpy as np
from hex_ai.data_utils import (
    rotate_board_180, reflect_board_long_diagonal, reflect_board_short_diagonal,
    create_augmented_boards, create_augmented_policies
)
from hex_ai.config import BOARD_SIZE, BLUE_PIECE, RED_PIECE, EMPTY_PIECE, PIECE_ONEHOT, EMPTY_ONEHOT

class TestAugmentation(unittest.TestCase):
    def setUp(self):
        """Create test boards and policies."""
        # Create a simple test board with 2 channels
        self.board_2ch = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.board_2ch[0, 0, 0] = PIECE_ONEHOT  # Blue piece at (0,0)
        self.board_2ch[1, 1, 1] = PIECE_ONEHOT  # Red piece at (1,1)
        
        # Create a test board with single channel
        self.board_1ch = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.board_1ch[0, 0] = BLUE_PIECE  # Blue piece at (0,0)
        self.board_1ch[1, 1] = RED_PIECE   # Red piece at (1,1)
        
        # Create a more complex test policy with asymmetric values
        self.policy = np.zeros(169, dtype=np.float32)
        self.policy[0] = 1.0   # High probability at position 0 (0,0)
        self.policy[14] = 0.5  # Medium probability at position 14 (1,1)
        self.policy[168] = 0.3 # Low probability at position 168 (12,12)
        self.policy[13] = 0.2  # Very low probability at position 13 (0,12)

    def test_rotate_board_180_2ch(self):
        """Test 180-degree rotation with 2-channel board."""
        rotated = rotate_board_180(self.board_2ch)
        
        # Check shape
        self.assertEqual(rotated.shape, (2, BOARD_SIZE, BOARD_SIZE))
        
        # Check that pieces moved to opposite corners (no color swap)
        self.assertEqual(rotated[0, BOARD_SIZE-1, BOARD_SIZE-1], PIECE_ONEHOT)  # Blue piece moved to opposite corner
        self.assertEqual(rotated[1, BOARD_SIZE-2, BOARD_SIZE-2], PIECE_ONEHOT)  # Red piece moved to opposite corner
        
        # Check that original positions are empty
        self.assertEqual(rotated[0, 0, 0], EMPTY_ONEHOT)  # Blue channel should be empty at original position
        self.assertEqual(rotated[1, 1, 1], EMPTY_ONEHOT)  # Red channel should be empty at original position

    def test_rotate_board_180_1ch(self):
        """Test 180-degree rotation with single-channel board."""
        rotated = rotate_board_180(self.board_1ch)
        
        # Check shape
        self.assertEqual(rotated.shape, (BOARD_SIZE, BOARD_SIZE))
        
        # Check that pieces moved to opposite corners and colors swapped
        self.assertEqual(rotated[BOARD_SIZE-1, BOARD_SIZE-1], BLUE_PIECE) 
        self.assertEqual(rotated[BOARD_SIZE-2, BOARD_SIZE-2], RED_PIECE) 

    def test_reflect_board_long_diagonal_2ch(self):
        """Test long diagonal reflection with 2-channel board."""
        reflected = reflect_board_long_diagonal(self.board_2ch)
        
        # Check shape
        self.assertEqual(reflected.shape, (2, BOARD_SIZE, BOARD_SIZE))
        
        # Check that pieces moved to transposed positions
        self.assertEqual(reflected[1, 0, 0], PIECE_ONEHOT)  # Blue piece transposed and channel swapped
        self.assertEqual(reflected[0, 1, 1], PIECE_ONEHOT)  # Red piece transposed and channel swapped

    def test_reflect_board_long_diagonal_1ch(self):
        """Test long diagonal reflection with single-channel board."""
        reflected = reflect_board_long_diagonal(self.board_1ch)
        
        # Check shape
        self.assertEqual(reflected.shape, (BOARD_SIZE, BOARD_SIZE))
        
        # Check that pieces moved to transposed positions and colors swapped
        self.assertEqual(reflected[0, 0], RED_PIECE)  # Blue became red
        self.assertEqual(reflected[1, 1], BLUE_PIECE)  # Red became blue

    def test_reflect_board_short_diagonal_2ch(self):
        """Test short diagonal reflection with 2-channel board."""
        reflected = reflect_board_short_diagonal(self.board_2ch)
        
        # Check shape
        self.assertEqual(reflected.shape, (2, BOARD_SIZE, BOARD_SIZE))
        
        # Check that pieces moved to correct positions
        # Short diagonal reflection = 180° rotation + long diagonal reflection
        self.assertEqual(reflected[1, BOARD_SIZE-1, BOARD_SIZE-1], PIECE_ONEHOT)  # Blue piece
        self.assertEqual(reflected[0, BOARD_SIZE-2, BOARD_SIZE-2], PIECE_ONEHOT)  # Red piece

    def test_reflect_board_short_diagonal_1ch(self):
        """Test short diagonal reflection with single-channel board."""
        reflected = reflect_board_short_diagonal(self.board_1ch)
        
        # Check shape
        self.assertEqual(reflected.shape, (BOARD_SIZE, BOARD_SIZE))
        
        # Check that pieces moved to correct positions and colors swapped
        self.assertEqual(reflected[BOARD_SIZE-1, BOARD_SIZE-1], RED_PIECE)  # Blue became red
        self.assertEqual(reflected[BOARD_SIZE-2, BOARD_SIZE-2], BLUE_PIECE)  # Red became blue

    def test_create_augmented_boards(self):
        """Test creating all 4 augmented boards."""
        augmented = create_augmented_boards(self.board_2ch)
        
        # Check that we get 4 boards
        self.assertEqual(len(augmented), 4)
        
        # Check that all boards have correct shape
        for board in augmented:
            self.assertEqual(board.shape, (2, BOARD_SIZE, BOARD_SIZE))
        
        # Check that boards are different (not all identical)
        self.assertFalse(np.allclose(augmented[0], augmented[1]))
        self.assertFalse(np.allclose(augmented[0], augmented[2]))
        self.assertFalse(np.allclose(augmented[0], augmented[3]))

    def test_create_augmented_policies(self):
        """Test creating all 4 augmented policies."""
        augmented = create_augmented_policies(self.policy)
        
        # Check that we get 4 policies
        self.assertEqual(len(augmented), 4)
        
        # Check that all policies have correct shape
        for policy in augmented:
            self.assertEqual(policy.shape, (169,))
        
        # Check that at least some policies are different (not all identical)
        # The original and rotated should be different
        self.assertFalse(np.allclose(augmented[0], augmented[1]))
        # The original and long diagonal reflection should be different
        self.assertFalse(np.allclose(augmented[0], augmented[2]))
        # The original and short diagonal reflection should be different
        self.assertFalse(np.allclose(augmented[0], augmented[3]))

    def test_augmentation_preserves_sum(self):
        """Test that augmentation preserves the sum of policy probabilities."""
        augmented = create_augmented_policies(self.policy)
        
        original_sum = np.sum(self.policy)
        for policy in augmented:
            self.assertAlmostEqual(np.sum(policy), original_sum, places=6)

    def test_augmentation_idempotent(self):
        """Test that applying the same augmentation twice returns to original."""
        # Test 180° rotation
        rotated_once = rotate_board_180(self.board_2ch)
        rotated_twice = rotate_board_180(rotated_once)
        np.testing.assert_array_equal(rotated_twice, self.board_2ch)
        
        # Test long diagonal reflection
        reflected_once = reflect_board_long_diagonal(self.board_2ch)
        reflected_twice = reflect_board_long_diagonal(reflected_once)
        np.testing.assert_array_equal(reflected_twice, self.board_2ch)

    def test_augmentation_edge_cases(self):
        """Test augmentation with edge cases."""
        # Empty board
        empty_board = np.zeros((2, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        rotated_empty = rotate_board_180(empty_board)
        np.testing.assert_array_equal(rotated_empty, empty_board)
        
        # Full board
        full_board = np.ones((2, BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        rotated_full = rotate_board_180(full_board)
        np.testing.assert_array_equal(rotated_full, full_board[::-1])  # Channels swapped

if __name__ == '__main__':
    unittest.main() 