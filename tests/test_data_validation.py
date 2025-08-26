"""
Test suite for data validation functions.

This module provides pytest-compatible tests for validating the data processing pipeline.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import gzip
import pickle
import random

from hex_ai.data_utils import (
    tensor_to_rowcol, rowcol_to_trmph, trmph_move_to_rowcol,
    strip_trmph_preamble, split_trmph_moves
)
from hex_ai.config import BOARD_SIZE, POLICY_OUTPUT_SIZE
from hex_ai.enums import Piece, piece_to_char


class TestDataValidation:
    """Test class for data validation functions."""
    
    @pytest.fixture
    def sample_shard_data(self):
        """Load sample shard data for testing."""
        processed_dir = Path("data/processed_fixed")
        shard_files = list(processed_dir.glob("*.pkl.gz"))
        
        if not shard_files:
            pytest.skip("No processed shard files found")
        
        # Load a random shard
        shard_file = random.choice(shard_files)
        with gzip.open(shard_file, 'rb') as f:
            shard_data = pickle.load(f)
        
        return shard_data
    
    def test_roundtrip_conversion(self, sample_shard_data):
        """Test that we can convert processed data back to trmph format."""
        boards = sample_shard_data['boards']
        policies = sample_shard_data['policies']
        
        # Test first example
        board = boards[0]
        policy = policies[0]
        
        # Convert board tensor to matrix format
        board_np = board.numpy()
        board_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        board_matrix[board_np[0] > 0.5] = piece_to_char(Piece.BLUE)  # Blue pieces
        board_matrix[board_np[1] > 0.5] = piece_to_char(Piece.RED)   # Red pieces
        
        # Get the predicted move from policy
        move_pos = policy.argmax().item()
        row, col = tensor_to_rowcol(move_pos)
        predicted_move = rowcol_to_trmph(row, col)
        
        # Convert back to trmph format
        reconstructed_trmph = self._board_matrix_to_trmph(board_matrix)
        
        # Validate the reconstructed trmph
        assert self._validate_trmph_format(reconstructed_trmph), "Invalid trmph format"
        
        # The predicted move should NOT be in the reconstructed trmph since it's the next move to be played
        # Check that the predicted move position is empty in the board
        assert board_matrix[row, col] == piece_to_char(Piece.EMPTY), f"Predicted move position ({row}, {col}) is occupied"
        
        # Check that the reconstructed trmph contains valid moves
        bare_moves = strip_trmph_preamble(reconstructed_trmph)
        moves = split_trmph_moves(bare_moves)
        assert len(moves) > 0, "Reconstructed trmph has no moves"
    
    def test_policy_validity(self, sample_shard_data):
        """Test that policies are valid probability distributions."""
        policies = sample_shard_data['policies']
        
        # Test first few policies
        for i in range(min(3, policies.shape[0])):
            policy = policies[i]
            policy_np = policy.numpy()
            
            # Check if policy sums to 1 (or close to it)
            assert abs(policy_np.sum() - 1.0) < 1e-6, f"Policy {i} sum is {policy_np.sum()}, expected 1.0"
            
            # Check if all values are non-negative
            assert policy_np.min() >= 0, f"Policy {i} has negative values"
            
            # Check if max value is 1.0 (one-hot)
            assert policy_np.max() == 1.0, f"Policy {i} max is {policy_np.max()}, expected 1.0"
    
    def test_value_accuracy(self, sample_shard_data):
        """Test that value targets are valid."""
        values = sample_shard_data['values']
        
        # Test first few values
        for i in range(min(3, values.shape[0])):
            value = values[i]
            value_np = value.numpy()[0]  # Extract scalar value
            
            # Check if value is in valid range (0.0, 0.5, or 1.0)
            assert value_np in [0.0, 0.5, 1.0], f"Value {i} is {value_np}, expected 0.0, 0.5, or 1.0"
    
    def test_move_consistency(self, sample_shard_data):
        """Test that predicted moves are valid for the board state."""
        boards = sample_shard_data['boards']
        policies = sample_shard_data['policies']
        
        # Test first few examples
        for i in range(min(3, boards.shape[0])):
            board = boards[i]
            policy = policies[i]
            
            # Get predicted move
            move_pos = policy.argmax().item()
            row, col = tensor_to_rowcol(move_pos)
            
            # Convert board to matrix format
            board_np = board.numpy()
            board_matrix = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
            board_matrix[board_np[0] > 0.5] = piece_to_char(Piece.BLUE)  # Blue pieces
            board_matrix[board_np[1] > 0.5] = piece_to_char(Piece.RED)   # Red pieces
            
            # Check if predicted move position is empty
            assert board_matrix[row, col] == piece_to_char(Piece.EMPTY), f"Predicted move at ({row}, {col}) is occupied"
    
    def test_dataset_statistics(self, sample_shard_data):
        """Test that dataset has expected structure and statistics."""
        boards = sample_shard_data['boards']
        policies = sample_shard_data['policies']
        values = sample_shard_data['values']
        
        # Check shapes
        assert boards.shape[0] == policies.shape[0], "Board and policy counts don't match"
        assert boards.shape[0] == values.shape[0], "Board and value counts don't match"
        
        # Check board shape
        assert boards.shape[1:] == (2, BOARD_SIZE, BOARD_SIZE), f"Board shape is {boards.shape}, expected (2, {BOARD_SIZE}, {BOARD_SIZE})"
        
        # Check policy shape
        assert policies.shape[1] == POLICY_OUTPUT_SIZE, f"Policy shape is {policies.shape}, expected (..., {POLICY_OUTPUT_SIZE})"
        
        # Check value shape
        assert values.shape[1] == 1, f"Value shape is {values.shape}, expected (..., 1)"
    
    def test_trmph_conversion_functions(self):
        """Test trmph conversion utility functions."""
        # Test rowcol_to_trmph
        assert rowcol_to_trmph(0, 0) == "a1"
        assert rowcol_to_trmph(12, 12) == "m13"
        
        # Test tensor_to_rowcol
        row, col = tensor_to_rowcol(0)
        assert row == 0 and col == 0
        row, col = tensor_to_rowcol(168)  # Last position in 13x13 board
        assert row == 12 and col == 12
        
        # Test trmph_move_to_rowcol
        row, col = trmph_move_to_rowcol("a1")
        assert row == 0 and col == 0
        row, col = trmph_move_to_rowcol("m13")
        assert row == 12 and col == 12
    
    def _board_matrix_to_trmph(self, board_matrix: np.ndarray) -> str:
        """Convert board matrix back to trmph format."""
        # Start with empty board preamble
        if BOARD_SIZE == 13:
            preamble = "http://www.trmph.com/hex/board#13,"
        else:
            preamble = "http://www.trmph.com/hex/board#11,"
        
        # Add moves based on board state
        moves = []
        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if board_matrix[row, col] in [piece_to_char(Piece.BLUE), piece_to_char(Piece.RED)]:  # Blue or Red piece
                    move = rowcol_to_trmph(row, col)
                    moves.append(move)
        
        return preamble + "".join(moves)
    
    def _validate_trmph_format(self, trmph_text: str) -> bool:
        """Validate trmph format."""
        try:
            # Basic format validation
            if not trmph_text.startswith("http://www.trmph.com/hex/board#"):
                return False
            
            # Check for board size indicator
            if "#13," not in trmph_text and "#11," not in trmph_text:
                return False
            
            # Check for valid moves
            bare_moves = strip_trmph_preamble(trmph_text)
            moves = split_trmph_moves(bare_moves)
            
            # Validate each move
            for move in moves:
                if not self._is_valid_trmph_move(move):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def _is_valid_trmph_move(self, move: str) -> bool:
        """Check if a trmph move is valid."""
        try:
            if len(move) < 2 or len(move) > 3:
                return False
            
            # Check if it's letter + number format
            if not move[0].isalpha() or not move[1:].isdigit():
                return False
            
            # Check if letter is in valid range
            letter = move[0].lower()
            if letter not in 'abcdefghijklmnopqrstuvwxyz':
                return False
            
            # Check if number is in valid range
            number = int(move[1:])
            if number < 1 or number > BOARD_SIZE:
                return False
            
            return True
            
        except Exception:
            return False


def test_data_processing_pipeline():
    """Integration test for the entire data processing pipeline."""
    # This test validates that the data processing pipeline produces valid output
    processed_dir = Path("data/processed_fixed")
    shard_files = list(processed_dir.glob("*.pkl.gz"))
    
    if not shard_files:
        pytest.skip("No processed shard files found")
    
    # Test a few shards
    for shard_file in random.sample(shard_files, min(3, len(shard_files))):
        with gzip.open(shard_file, 'rb') as f:
            shard_data = pickle.load(f)
        
        # Basic structure validation
        assert 'boards' in shard_data
        assert 'policies' in shard_data
        assert 'values' in shard_data
        
        # Check that all tensors have the same first dimension
        num_examples = shard_data['boards'].shape[0]
        assert shard_data['policies'].shape[0] == num_examples
        assert shard_data['values'].shape[0] == num_examples
        
        # Check that we have at least some examples
        assert num_examples > 0, f"Shard {shard_file.name} has no examples"


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__]) 