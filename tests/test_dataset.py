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
from torch.utils.data import DataLoader


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
        # Create a temporary directory with a dummy .trmph file
        temp_dir = tempfile.mkdtemp()
        dummy_file = os.path.join(temp_dir, "dummy.trmph")
        with open(dummy_file, 'w') as f:
            f.write("http://www.trmph.com/hex/board#13,a1b2c3 1\n")
            f.write("http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7 0\n")
        
        try:
            dataset = HexDataset(data_dir=temp_dir)
            self.assertIsInstance(dataset, HexDataset)
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_dataset_length(self):
        """Test that dataset returns correct length."""
        # Create a temporary directory with a dummy .trmph file
        temp_dir = tempfile.mkdtemp()
        dummy_file = os.path.join(temp_dir, "dummy.trmph")
        with open(dummy_file, 'w') as f:
            f.write("http://www.trmph.com/hex/board#13,a1b2c3 1\n")
            f.write("http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7 0\n")
        
        try:
            dataset = HexDataset(data_dir=temp_dir)
            self.assertEqual(len(dataset), 2)
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_dataset_getitem(self):
        """Test that dataset returns correct tensor shapes."""
        # Create a temporary directory with a dummy .trmph file
        temp_dir = tempfile.mkdtemp()
        dummy_file = os.path.join(temp_dir, "dummy.trmph")
        with open(dummy_file, 'w') as f:
            f.write("http://www.trmph.com/hex/board#13,a1b2c3 1\n")
        
        try:
            dataset = HexDataset(data_dir=temp_dir, augment=False)  # Disable augmentation for testing
            board, policy, value = dataset[0]
            self.assertEqual(board.shape, (2, 13, 13))
            self.assertEqual(policy.shape, (169,))
            self.assertEqual(value.shape, (1,))
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
    def test_dataset_with_augmentation(self):
        """Test that dataset works with augmentation enabled."""
        # Create a temporary directory with a dummy .trmph file
        temp_dir = tempfile.mkdtemp()
        dummy_file = os.path.join(temp_dir, "dummy.trmph")
        with open(dummy_file, 'w') as f:
            f.write("http://www.trmph.com/hex/board#13,a1b2c3 1\n")
        
        try:
            dataset = HexDataset(data_dir=temp_dir, augment=True)
            board, policy, value = dataset[0]
            self.assertEqual(board.shape, (2, 13, 13))
            self.assertEqual(policy.shape, (169,))
            self.assertEqual(value.shape, (1,))
        finally:
            import shutil
            shutil.rmtree(temp_dir)
    
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
        # Create a temporary directory with a dummy .trmph file
        temp_dir = tempfile.mkdtemp()
        dummy_file = os.path.join(temp_dir, "dummy.trmph")
        with open(dummy_file, 'w') as f:
            f.write("http://www.trmph.com/hex/board#13,a1b2c3 1\n")
            f.write("http://www.trmph.com/hex/board#13,a1b2c3d4e5f6g7 0\n")
        
        try:
            dataloader = create_dataloader(data_dir=temp_dir)
            self.assertIsInstance(dataloader, DataLoader)
        finally:
            import shutil
            shutil.rmtree(temp_dir)


if __name__ == '__main__':
    unittest.main() 