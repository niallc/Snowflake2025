"""
Tests for the Hex AI model architectures.

This module contains unit tests to ensure the ResNet models work correctly
and produce the expected output shapes.
"""

import unittest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add the parent directory to the path so we can import hex_ai
sys.path.append(str(Path(__file__).parent.parent))

from hex_ai.models import (
    ResNetBlock, TwoHeadedResNet, create_model, 
    count_parameters, get_model_summary
)
from hex_ai.config import BOARD_SIZE, NUM_PLAYERS, POLICY_OUTPUT_SIZE, VALUE_OUTPUT_SIZE


class TestResNetBlock(unittest.TestCase):
    """Test cases for the ResNetBlock class."""
    
    def test_resnet_block_creation(self):
        """Test that ResNetBlock can be created."""
        block = ResNetBlock(in_channels=64, out_channels=64, stride=1)
        self.assertIsInstance(block, ResNetBlock)
    
    def test_resnet_block_forward(self):
        """Test that ResNetBlock forward pass works."""
        block = ResNetBlock(in_channels=64, out_channels=64, stride=1)
        x = torch.randn(2, 64, 13, 13)
        output = block(x)
        
        # Check output shape
        self.assertEqual(output.shape, (2, 64, 13, 13))
    
    def test_resnet_block_stride_2(self):
        """Test ResNetBlock with stride=2 (downsampling)."""
        block = ResNetBlock(in_channels=64, out_channels=128, stride=2)
        x = torch.randn(2, 64, 13, 13)
        output = block(x)
        
        # Check output shape (should be downsampled)
        self.assertEqual(output.shape, (2, 128, 7, 7))
    
    def test_resnet_block_shortcut(self):
        """Test that shortcut connection works correctly."""
        block = ResNetBlock(in_channels=64, out_channels=128, stride=2)
        x = torch.randn(2, 64, 13, 13)
        output = block(x)
        
        # Output should be the sum of main path and shortcut
        self.assertEqual(output.shape, (2, 128, 7, 7))


class TestTwoHeadedResNet(unittest.TestCase):
    """Test cases for the TwoHeadedResNet class."""
    
    def test_model_creation(self):
        """Test that TwoHeadedResNet can be created."""
        model = TwoHeadedResNet()
        self.assertIsInstance(model, TwoHeadedResNet)
    
    def test_model_forward(self):
        """Test that model forward pass works."""
        model = TwoHeadedResNet()
        x = torch.randn(4, 3, BOARD_SIZE, BOARD_SIZE)
        policy_logits, value_logit = model(x)
        
        # Check output shapes
        self.assertEqual(policy_logits.shape, (4, POLICY_OUTPUT_SIZE))
        self.assertEqual(value_logit.shape, (4, VALUE_OUTPUT_SIZE))
    
    def test_model_parameters(self):
        """Test that model has reasonable number of parameters."""
        model = TwoHeadedResNet()
        num_params = count_parameters(model)
        
        # ResNet-18 should have ~11M parameters
        self.assertGreater(num_params, 10_000_000)  # At least 10M
        self.assertLess(num_params, 15_000_000)     # Less than 15M
    
    def test_model_summary(self):
        """Test that model summary works."""
        model = TwoHeadedResNet()
        summary = get_model_summary(model)
        
        # Check that summary contains expected information
        self.assertIn("Total Parameters", summary)
        self.assertIn("TwoHeadedResNet", summary)
        self.assertIn("Policy Head", summary)
        self.assertIn("Value Head", summary)
    
    def test_model_device_transfer(self):
        """Test that model can be moved to different devices."""
        model = TwoHeadedResNet()
        
        # Test CPU
        model_cpu = model.cpu()
        x = torch.randn(2, 3, BOARD_SIZE, BOARD_SIZE)
        policy, value = model_cpu(x)
        self.assertEqual(policy.device, torch.device('cpu'))
        self.assertEqual(value.device, torch.device('cpu'))
        
        # Test CUDA if available
        if torch.cuda.is_available():
            model_cuda = model.cuda()
            x_cuda = x.cuda()
            policy, value = model_cuda(x_cuda)
            self.assertEqual(policy.device, torch.device('cuda'))
            self.assertEqual(value.device, torch.device('cuda'))
    
    def test_model_gradients(self):
        """Test that model can compute gradients."""
        model = TwoHeadedResNet()
        x = torch.randn(2, 3, BOARD_SIZE, BOARD_SIZE, requires_grad=True)
        policy_logits, value_logit = model(x)
        
        # Compute loss and backward pass
        loss = policy_logits.sum() + value_logit.sum()
        loss.backward()
        
        # Check that gradients were computed
        self.assertIsNotNone(x.grad)
    
    def test_model_batch_sizes(self):
        """Test that model works with different batch sizes."""
        model = TwoHeadedResNet()
        
        batch_sizes = [1, 4, 8, 16]
        for batch_size in batch_sizes:
            x = torch.randn(batch_size, 3, BOARD_SIZE, BOARD_SIZE)
            policy_logits, value_logit = model(x)
            
            self.assertEqual(policy_logits.shape, (batch_size, POLICY_OUTPUT_SIZE))
            self.assertEqual(value_logit.shape, (batch_size, VALUE_OUTPUT_SIZE))


class TestModelFactory(unittest.TestCase):
    """Test cases for the model factory function."""
    
    def test_create_model_resnet18(self):
        """Test creating ResNet-18 model."""
        model = create_model("resnet18")
        self.assertIsInstance(model, TwoHeadedResNet)
    
    def test_create_model_invalid_type(self):
        """Test that invalid model type raises error."""
        with self.assertRaises(ValueError):
            create_model("invalid_model")

if __name__ == '__main__':
    unittest.main() 