"""
Tests for the HexDataset and related functionality.

This module contains unit tests to ensure the dataset works correctly
and handles various data formats properly.
"""

import unittest
import torch
import numpy as np
from pathlib import Path
import tempfile
import os

# Add the parent directory to the path so we can import hex_ai
import sys
sys.path.append(str(Path(__file__).parent.parent))

from hex_ai.dataset import HexDataset, create_dataloader
from hex_ai.utils import create_sample_data, validate_board_shape
from hex_ai.config import BOARD_SIZE, NUM_PLAYERS, POLICY_OUTPUT_SIZE


class TestHexDataset(unittest.TestCase):
    """Test cases for the HexDataset class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_data_path = Path(self.temp_dir)
        
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_initialization(self):
        """Test that HexDataset can be initialized."""
        # TODO: Implement test once dataset is implemented
        dataset = HexDataset(data_path=self.test_data_path)
        self.assertIsInstance(dataset, HexDataset)
    
    def test_dataset_length(self):
        """Test that dataset returns correct length."""
        # TODO: Implement test once dataset is implemented
        dataset = HexDataset(data_path=self.test_data_path)
        # This should be 0 for now since we haven't implemented data loading
        self.assertEqual(len(dataset), 0)
    
    def test_dataset_getitem(self):
        """Test that dataset returns correct tensor shapes."""
        # TODO: Implement test once dataset is implemented
        dataset = HexDataset(data_path=self.test_data_path)
        
        # For now, this should return placeholder tensors
        board, policy, value = dataset[0]
        self.assertEqual(board.shape, (2, 13, 13))
        self.assertEqual(policy.shape, (169,))
        self.assertEqual(value.shape, (1,))
    
    def test_validate_board_shape(self):
        """Test board shape validation."""
        # Valid shape
        valid_tensor = torch.randn(2, 13, 13)
        self.assertTrue(validate_board_shape(valid_tensor))
        
        # Invalid shapes
        invalid_shapes = [
            torch.randn(1, 13, 13),  # Wrong number of channels
            torch.randn(2, 12, 13),  # Wrong height
            torch.randn(2, 13, 12),  # Wrong width
            torch.randn(2, 13),      # Missing dimension
        ]
        
        for tensor in invalid_shapes:
            self.assertFalse(validate_board_shape(tensor))
    
    def test_create_sample_data(self):
        """Test sample data creation."""
        num_samples = 10
        boards, policies, values = create_sample_data(num_samples)
        
        # Check shapes
        self.assertEqual(boards.shape, (num_samples, NUM_PLAYERS, BOARD_SIZE, BOARD_SIZE))
        self.assertEqual(policies.shape, (num_samples, POLICY_OUTPUT_SIZE))
        self.assertEqual(values.shape, (num_samples, 1))
        
        # Check data types
        self.assertEqual(boards.dtype, torch.float32)
        self.assertEqual(policies.dtype, torch.float32)
        self.assertEqual(values.dtype, torch.float32)


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader functionality."""
    
    def test_create_dataloader(self):
        """Test that DataLoader can be created."""
        # Create a dummy dataset
        dataset = HexDataset(data_path=tempfile.mkdtemp())
        
        # Create DataLoader
        dataloader = create_dataloader(
            dataset, 
            batch_size=4, 
            shuffle=False, 
            num_workers=0
        )
        
        self.assertIsInstance(dataloader, torch.utils.data.DataLoader)
        self.assertEqual(dataloader.batch_size, 4)


if __name__ == '__main__':
    unittest.main() 